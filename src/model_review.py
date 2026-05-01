from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.advanced_metrics import build_advanced_team_metrics
from src.analytics import prepare_matches_dataframe
from src.context_engine import build_context_adjusted_edge
from src.db import get_connection, init_db
from src.matchup_analysis import build_predictor_context, build_style_advantage, identify_key_mismatches
from src.predictor import predict_match
from src.schedule_context import build_match_schedule_context, build_schedule_data_audit
from src.team_profiles import build_team_profile_context, build_team_profile_with_ratings


EDGE_THRESHOLD = 2.0
HIGH_DRAW_RISK_THRESHOLD = 55.0
HIGH_UPSET_RISK_THRESHOLD = 55.0
LOW_CONFIDENCE_THRESHOLD = 45.0
HIGH_CONFIDENCE_THRESHOLD = 65.0


def _prepare_optional_schedule_df(schedule_df: pd.DataFrame | None, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty:
        candidate_df = schedule_df.copy()
        if "match_date" not in candidate_df.columns:
            return fallback_df
        if "id" not in candidate_df.columns:
            candidate_df["id"] = range(1, len(candidate_df) + 1)
        try:
            return prepare_matches_dataframe(candidate_df)
        except Exception:
            return fallback_df
    return fallback_df


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _load_ratings_history() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        if not _table_exists(conn, "team_ratings"):
            return pd.DataFrame(columns=["team_name", "rating_date", "rating_type", "rating_value", "source_name", "source_url"])
        try:
            ratings_df = pd.read_sql_query(
                """
                SELECT team_name, rating_date, rating_type, rating_value, source_name, source_url
                FROM team_ratings
                WHERE rating_type = 'elo'
                """,
                conn,
            )
        except sqlite3.Error:
            return pd.DataFrame(columns=["team_name", "rating_date", "rating_type", "rating_value", "source_name", "source_url"])

    if ratings_df.empty:
        return ratings_df

    ratings_df["team_name"] = ratings_df["team_name"].astype(str)
    ratings_df["rating_value"] = pd.to_numeric(ratings_df["rating_value"], errors="coerce")
    ratings_df["rating_date"] = pd.to_datetime(ratings_df["rating_date"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["team_name", "rating_value", "rating_date"]).reset_index(drop=True)
    return ratings_df


def _build_ratings_snapshot(
    ratings_history_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    teams: list[str],
) -> pd.DataFrame:
    if ratings_history_df.empty:
        return pd.DataFrame(columns=["team_name", "rating_date", "rating_type", "rating_value", "source_name", "source_url"])

    snapshot_df = ratings_history_df.loc[
        (ratings_history_df["rating_date"] <= as_of_date) & (ratings_history_df["team_name"].isin(teams))
    ].copy()
    if snapshot_df.empty:
        return snapshot_df

    snapshot_df = snapshot_df.sort_values(["team_name", "rating_date"], ascending=[True, False])
    snapshot_df = snapshot_df.drop_duplicates(subset=["team_name"], keep="first").reset_index(drop=True)
    return snapshot_df


def build_ratings_audit(
    season_df: pd.DataFrame,
    ratings_history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(season_df)
    audit = {
        "available": False,
        "historical_ready": False,
        "unique_rating_dates": 0,
        "earliest_rating_date": None,
        "latest_rating_date": None,
        "matches_with_both_ratings": 0,
        "matches_with_partial_ratings": 0,
        "pre_match_coverage_pct": 0.0,
        "status": "assente",
        "note": "Nessun rating Elo disponibile per il backtest storico.",
    }
    if prepared_df.empty:
        audit["status"] = "stagione vuota"
        audit["note"] = "La stagione selezionata non contiene partite utilizzabili per audit Elo."
        return audit

    ratings_history_df = ratings_history_df.copy() if ratings_history_df is not None else _load_ratings_history()
    if ratings_history_df.empty:
        return audit

    teams = sorted(set(prepared_df["home_team"].astype(str).tolist()) | set(prepared_df["away_team"].astype(str).tolist()))
    relevant_df = ratings_history_df.loc[ratings_history_df["team_name"].astype(str).isin(teams)].copy()
    if relevant_df.empty:
        audit["status"] = "fuori copertura"
        audit["note"] = "Il layer Elo non copre le squadre presenti nella stagione analizzata."
        return audit

    relevant_df["rating_date"] = pd.to_datetime(relevant_df["rating_date"], errors="coerce")
    relevant_df = relevant_df.dropna(subset=["rating_date"]).sort_values(["rating_date", "team_name"]).reset_index(drop=True)
    if relevant_df.empty:
        audit["status"] = "date non valide"
        audit["note"] = "I rating Elo presenti non hanno date storiche utilizzabili."
        return audit

    audit["available"] = True
    audit["unique_rating_dates"] = int(relevant_df["rating_date"].dt.normalize().nunique())
    audit["earliest_rating_date"] = relevant_df["rating_date"].min().strftime("%Y-%m-%d")
    audit["latest_rating_date"] = relevant_df["rating_date"].max().strftime("%Y-%m-%d")

    rating_records = relevant_df.to_dict(orient="records")
    match_records = prepared_df.sort_values(["match_date", "home_team", "away_team"]).to_dict(orient="records")
    available_teams: set[str] = set()
    rating_idx = 0
    both_count = 0
    partial_count = 0

    for match in match_records:
        match_date = pd.to_datetime(match.get("match_date"), errors="coerce")
        while rating_idx < len(rating_records) and pd.to_datetime(rating_records[rating_idx]["rating_date"], errors="coerce") <= match_date:
            team_name = str(rating_records[rating_idx].get("team_name") or "")
            if team_name:
                available_teams.add(team_name)
            rating_idx += 1

        home_team = str(match.get("home_team") or "")
        away_team = str(match.get("away_team") or "")
        available_count = int(home_team in available_teams) + int(away_team in available_teams)
        if available_count == 2:
            both_count += 1
        elif available_count == 1:
            partial_count += 1

    total_matches = max(len(match_records), 1)
    coverage_pct = round(both_count / total_matches * 100.0, 1)
    audit["matches_with_both_ratings"] = both_count
    audit["matches_with_partial_ratings"] = partial_count
    audit["pre_match_coverage_pct"] = coverage_pct

    historical_ready = audit["unique_rating_dates"] >= 3 and coverage_pct >= 25.0
    audit["historical_ready"] = historical_ready

    if both_count == 0:
        audit["status"] = "informativo / non usato per calibrazione storica"
        audit["note"] = (
            "Il seed Elo attuale e uno snapshot successivo alle partite backtestate, "
            "quindi nel review non entra come fattore pre-match."
        )
    elif not historical_ready:
        audit["status"] = "copertura storica parziale"
        audit["note"] = (
            "Elo appare solo in una parte limitata del backtest: per ora lo trattiamo "
            "con prudenza e non lo usiamo come base forte di calibrazione."
        )
    else:
        audit["status"] = "storico utilizzabile"
        audit["note"] = "La copertura Elo storica e sufficiente per entrare anche nella calibrazione del review."

    return audit


def _actual_outcome(row: pd.Series) -> tuple[str, int]:
    home_goals = int(row.get("home_goals", 0) or 0)
    away_goals = int(row.get("away_goals", 0) or 0)
    if home_goals > away_goals:
        return "1", 1
    if home_goals < away_goals:
        return "2", -1
    return "X", 0


def _favorite_from_edge(edge: float | None, threshold: float = EDGE_THRESHOLD) -> str:
    if edge is None:
        return "none"
    if edge >= threshold:
        return "home"
    if edge <= -threshold:
        return "away"
    return "none"


def _favorite_not_lose(favorite: str, actual_outcome: str) -> bool | None:
    if favorite == "home":
        return actual_outcome != "2"
    if favorite == "away":
        return actual_outcome != "1"
    return None


def _favorite_win(favorite: str, actual_outcome: str) -> bool | None:
    if favorite == "home":
        return actual_outcome == "1"
    if favorite == "away":
        return actual_outcome == "2"
    return None


def _edge_quality_score(favorite: str, actual_outcome: str) -> float:
    if favorite == "home":
        return 1.0 if actual_outcome == "1" else 0.5 if actual_outcome == "X" else 0.0
    if favorite == "away":
        return 1.0 if actual_outcome == "2" else 0.5 if actual_outcome == "X" else 0.0
    return 1.0 if actual_outcome == "X" else 0.25


def _bucket_label(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "n/d"
    if value < low:
        return "basso"
    if value < high:
        return "medio"
    return "alto"


def _confidence_bucket(confidence: float | None) -> str:
    return _bucket_label(confidence, LOW_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD)


def _risk_bucket(value: float | None) -> str:
    return _bucket_label(value, 35.0, 55.0)


def _edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "n/d"
    edge_abs = abs(edge)
    if edge_abs < 4.0:
        return "basso"
    if edge_abs < 8.0:
        return "medio"
    return "alto"


def _factor_help_label(weighted_impact: float, actual_sign: int) -> str:
    if actual_sign == 0:
        if abs(weighted_impact) <= 1.0:
            return "helped"
        return "hurt"

    if weighted_impact == 0:
        return "neutral"
    if actual_sign > 0 and weighted_impact > 0:
        return "helped"
    if actual_sign < 0 and weighted_impact < 0:
        return "helped"
    return "hurt"


def build_backtest_rows(
    season_df: pd.DataFrame,
    minimum_team_history: int = 1,
    schedule_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_df = prepare_matches_dataframe(season_df)
    if prepared_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    schedule_prepared_df = _prepare_optional_schedule_df(schedule_df, prepared_df)
    ratings_history_df = _load_ratings_history()
    rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []

    for idx, row in prepared_df.iterrows():
        historical_df = prepared_df.iloc[:idx].copy()
        if historical_df.empty:
            continue

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_history_count = int(((historical_df["home_team"] == home_team) | (historical_df["away_team"] == home_team)).sum())
        away_history_count = int(((historical_df["home_team"] == away_team) | (historical_df["away_team"] == away_team)).sum())
        if home_history_count < minimum_team_history or away_history_count < minimum_team_history:
            continue

        historical_teams = sorted(set(historical_df["home_team"].astype(str).tolist()) | set(historical_df["away_team"].astype(str).tolist()))
        ratings_snapshot_df = _build_ratings_snapshot(
            ratings_history_df,
            pd.to_datetime(row["match_date"], errors="coerce"),
            historical_teams,
        )

        advanced_df = build_advanced_team_metrics(historical_df, ratings_df=ratings_snapshot_df)
        profile_context = build_team_profile_context(
            historical_df,
            ratings_df=ratings_snapshot_df,
            advanced_metrics_df=advanced_df,
        )
        home_profile = build_team_profile_with_ratings(
            historical_df,
            home_team,
            ratings_df=ratings_snapshot_df,
            advanced_metrics_df=advanced_df,
            context=profile_context,
        )
        away_profile = build_team_profile_with_ratings(
            historical_df,
            away_team,
            ratings_df=ratings_snapshot_df,
            advanced_metrics_df=advanced_df,
            context=profile_context,
        )
        if not home_profile.get("ok") or not away_profile.get("ok"):
            continue

        predictor = predict_match(historical_df, home_team, away_team, max_goals=6)
        predictor_context = build_predictor_context(predictor, home_profile, away_profile)
        mismatches = identify_key_mismatches(
            home_profile,
            away_profile,
            predictor_context=predictor_context,
        )
        style_advantage = build_style_advantage(home_profile, away_profile, predictor)
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        if isinstance(schedule_prepared_df, pd.DataFrame) and not schedule_prepared_df.empty and not pd.isna(match_date):
            schedule_history_df = schedule_prepared_df.loc[schedule_prepared_df["match_date"] < match_date].copy()
        else:
            schedule_history_df = historical_df
        if schedule_history_df.empty:
            schedule_history_df = historical_df

        schedule_context = build_match_schedule_context(
            schedule_history_df,
            home_team,
            away_team,
            match_date=row.get("match_date"),
        )
        context_engine = build_context_adjusted_edge(
            home_profile,
            away_profile,
            predictor_context=predictor_context,
            mismatches=mismatches,
            style_advantage=style_advantage,
            schedule_context=schedule_context,
        )
        actual_outcome, actual_sign = _actual_outcome(row)
        base_edge = _safe_float(context_engine.get("base_edge")) or 0.0
        adjusted_edge = _safe_float(context_engine.get("adjusted_edge")) or 0.0
        draw_risk = _safe_float(context_engine.get("draw_risk")) or 0.0
        upset_risk = _safe_float(context_engine.get("upset_risk")) or 0.0
        confidence = _safe_float(context_engine.get("confidence")) or 0.0
        uncertainty = _safe_float(context_engine.get("uncertainty_index")) or 0.0
        base_favorite = _favorite_from_edge(base_edge)
        adjusted_favorite = _favorite_from_edge(adjusted_edge)

        base_score = _edge_quality_score(base_favorite, actual_outcome)
        adjusted_score = _edge_quality_score(adjusted_favorite, actual_outcome)
        context_delta = round(adjusted_score - base_score, 2)

        base_not_lose = _favorite_not_lose(base_favorite, actual_outcome)
        adjusted_not_lose = _favorite_not_lose(adjusted_favorite, actual_outcome)
        adjusted_win = _favorite_win(adjusted_favorite, actual_outcome)

        rows.append(
            {
                "match_date": pd.to_datetime(row["match_date"], errors="coerce"),
                "season": row.get("season"),
                "home_team": home_team,
                "away_team": away_team,
                "score": f"{int(row.get('home_goals', 0) or 0)}-{int(row.get('away_goals', 0) or 0)}",
                "actual_outcome": actual_outcome,
                "actual_sign": actual_sign,
                "base_edge": round(base_edge, 2),
                "adjusted_edge": round(adjusted_edge, 2),
                "draw_risk": round(draw_risk, 1),
                "upset_risk": round(upset_risk, 1),
                "confidence": round(confidence, 1),
                "uncertainty_index": round(uncertainty, 1),
                "base_favorite": base_favorite,
                "adjusted_favorite": adjusted_favorite,
                "base_favorite_not_lose": base_not_lose,
                "adjusted_favorite_not_lose": adjusted_not_lose,
                "adjusted_favorite_win": adjusted_win,
                "base_score": base_score,
                "adjusted_score": adjusted_score,
                "context_delta": context_delta,
                "context_helped": context_delta > 0,
                "context_hurt": context_delta < 0,
                "high_draw_risk": draw_risk >= HIGH_DRAW_RISK_THRESHOLD,
                "high_upset_risk": upset_risk >= HIGH_UPSET_RISK_THRESHOLD,
                "predictor_available": bool(predictor_context.get("available")),
                "predictor_home_probability": round(float(predictor_context.get("home_probability", 0.0) or 0.0), 4)
                if predictor_context.get("available")
                else None,
                "predictor_draw_probability": round(float(predictor_context.get("draw_probability", 0.0) or 0.0), 4)
                if predictor_context.get("available")
                else None,
                "predictor_away_probability": round(float(predictor_context.get("away_probability", 0.0) or 0.0), 4)
                if predictor_context.get("available")
                else None,
                "predictor_score": predictor_context.get("most_likely_score"),
                "confidence_bucket": _confidence_bucket(confidence),
                "draw_risk_bucket": _risk_bucket(draw_risk),
                "upset_risk_bucket": _risk_bucket(upset_risk),
                "adjusted_edge_bucket": _edge_bucket(adjusted_edge),
                "top_context_factors": ", ".join(
                    factor.get("label", "") for factor in context_engine.get("weighted_factors", [])[:3] if factor.get("label")
                ),
                "summary": context_engine.get("textual_explanation"),
            }
        )

        for factor in context_engine.get("weighted_factors", []):
            weighted_impact = float(factor.get("weighted_impact", 0.0) or 0.0)
            factor_rows.append(
                {
                    "match_date": pd.to_datetime(row["match_date"], errors="coerce"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "factor": factor.get("factor"),
                    "label": factor.get("label"),
                    "signal": float(factor.get("signal", 0.0) or 0.0),
                    "reliability": float(factor.get("reliability", 0.0) or 0.0),
                    "context_relevance": float(factor.get("context_relevance", 0.0) or 0.0),
                    "matchup_multiplier": float(factor.get("matchup_multiplier", 1.0) or 1.0),
                    "weighted_impact": weighted_impact,
                    "impact_abs": abs(weighted_impact),
                    "available": bool(factor.get("available", True)),
                    "actual_outcome": actual_outcome,
                    "actual_sign": actual_sign,
                    "help_label": _factor_help_label(weighted_impact, actual_sign),
                    "used_actively": abs(weighted_impact) >= 1.0,
                }
            )

    backtest_df = pd.DataFrame(rows)
    factors_df = pd.DataFrame(factor_rows)
    return backtest_df, factors_df


def build_general_review(backtest_df: pd.DataFrame) -> dict[str, Any]:
    if backtest_df.empty:
        return {
            "matches_analyzed": 0,
            "base_favorite_non_loss_hits": 0,
            "base_favorite_non_loss_total": 0,
            "adjusted_favorite_non_loss_hits": 0,
            "adjusted_favorite_non_loss_total": 0,
            "high_draw_hits": 0,
            "high_draw_total": 0,
            "high_upset_hits": 0,
            "high_upset_total": 0,
            "context_helped": 0,
            "context_hurt": 0,
        }

    base_mask = backtest_df["base_favorite"] != "none"
    adjusted_mask = backtest_df["adjusted_favorite"] != "none"
    high_draw_mask = backtest_df["high_draw_risk"]
    high_upset_mask = backtest_df["high_upset_risk"] & adjusted_mask

    return {
        "matches_analyzed": int(len(backtest_df)),
        "base_favorite_non_loss_hits": int(backtest_df.loc[base_mask, "base_favorite_not_lose"].fillna(False).sum()),
        "base_favorite_non_loss_total": int(base_mask.sum()),
        "adjusted_favorite_non_loss_hits": int(backtest_df.loc[adjusted_mask, "adjusted_favorite_not_lose"].fillna(False).sum()),
        "adjusted_favorite_non_loss_total": int(adjusted_mask.sum()),
        "high_draw_hits": int((backtest_df.loc[high_draw_mask, "actual_outcome"] == "X").sum()),
        "high_draw_total": int(high_draw_mask.sum()),
        "high_upset_hits": int(backtest_df.loc[high_upset_mask, "adjusted_favorite_win"].fillna(False).eq(False).sum()),
        "high_upset_total": int(high_upset_mask.sum()),
        "context_helped": int(backtest_df["context_helped"].sum()),
        "context_hurt": int(backtest_df["context_hurt"].sum()),
        "mean_base_edge_abs": round(float(backtest_df["base_edge"].abs().mean()), 2),
        "mean_adjusted_edge_abs": round(float(backtest_df["adjusted_edge"].abs().mean()), 2),
    }


def build_diagnostic_tables(backtest_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if backtest_df.empty:
        empty = pd.DataFrame()
        return {
            "context_helped": empty,
            "context_hurt": empty,
            "high_uncertainty": empty,
            "high_draw_risk": empty,
            "high_upset_risk": empty,
        }

    display_columns = [
        "match_date",
        "home_team",
        "away_team",
        "score",
        "actual_outcome",
        "base_edge",
        "adjusted_edge",
        "draw_risk",
        "upset_risk",
        "confidence",
        "top_context_factors",
    ]
    diagnostics = {
        "context_helped": backtest_df.loc[backtest_df["context_helped"]].sort_values(
            ["context_delta", "confidence"], ascending=[False, False]
        )[display_columns].head(20),
        "context_hurt": backtest_df.loc[backtest_df["context_hurt"]].sort_values(
            ["context_delta", "confidence"], ascending=[True, True]
        )[display_columns].head(20),
        "high_uncertainty": backtest_df.loc[backtest_df["confidence"] < LOW_CONFIDENCE_THRESHOLD].sort_values(
            ["confidence", "draw_risk"], ascending=[True, False]
        )[display_columns].head(20),
        "high_draw_risk": backtest_df.loc[backtest_df["high_draw_risk"]].sort_values(
            ["draw_risk", "confidence"], ascending=[False, True]
        )[display_columns].head(20),
        "high_upset_risk": backtest_df.loc[backtest_df["high_upset_risk"]].sort_values(
            ["upset_risk", "confidence"], ascending=[False, True]
        )[display_columns].head(20),
    }
    for key, table in diagnostics.items():
        if not table.empty:
            diagnostics[key] = table.assign(match_date=table["match_date"].dt.strftime("%Y-%m-%d"))
    return diagnostics


def build_bucket_review(backtest_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if backtest_df.empty:
        empty = pd.DataFrame()
        return {
            "confidence": empty,
            "draw_risk": empty,
            "upset_risk": empty,
            "adjusted_edge": empty,
        }

    def _aggregate_by_bucket(column: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for bucket, bucket_df in backtest_df.groupby(column):
            adjusted_mask = bucket_df["adjusted_favorite"] != "none"
            rows.append(
                {
                    "Bucket": bucket,
                    "Partite": int(len(bucket_df)),
                    "Favorito contestuale non perde %": round(
                        float(bucket_df.loc[adjusted_mask, "adjusted_favorite_not_lose"].fillna(False).mean() * 100.0)
                        if adjusted_mask.any()
                        else 0.0,
                        1,
                    ),
                    "Pareggi %": round(float((bucket_df["actual_outcome"] == "X").mean() * 100.0), 1),
                    "Favorito non vince %": round(
                        float(bucket_df.loc[adjusted_mask, "adjusted_favorite_win"].fillna(False).eq(False).mean() * 100.0)
                        if adjusted_mask.any()
                        else 0.0,
                        1,
                    ),
                    "Confidenza media": round(float(bucket_df["confidence"].mean()), 1),
                }
            )
        return pd.DataFrame(rows)

    return {
        "confidence": _aggregate_by_bucket("confidence_bucket"),
        "draw_risk": _aggregate_by_bucket("draw_risk_bucket"),
        "upset_risk": _aggregate_by_bucket("upset_risk_bucket"),
        "adjusted_edge": _aggregate_by_bucket("adjusted_edge_bucket"),
    }


def build_factor_review(
    factors_df: pd.DataFrame,
    ratings_audit: dict[str, Any] | None = None,
    schedule_audit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if factors_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total_matches = max(int(factors_df[["match_date", "home_team", "away_team"]].drop_duplicates().shape[0]), 1)
    for factor_name, factor_df in factors_df.groupby("factor"):
        available_df = factor_df.loc[factor_df["available"].fillna(True)] if "available" in factor_df.columns else factor_df.copy()
        available_matches = int(available_df[["match_date", "home_team", "away_team"]].drop_duplicates().shape[0]) if not available_df.empty else 0
        use_for_calibration = True
        calibration_status = "calibrabile"
        calibration_note = ""

        if factor_name == "elo" and ratings_audit and not ratings_audit.get("historical_ready", False):
            use_for_calibration = False
            calibration_status = str(ratings_audit.get("status") or "informativo")
            calibration_note = str(ratings_audit.get("note") or "")
        elif factor_name == "schedule" and schedule_audit and schedule_audit.get("only_league_data", True):
            calibration_status = "parziale: solo partite disponibili"
            calibration_note = "La valutazione calendario e prudente perche mancano dati coppe/europee."
        elif factor_name == "stakes":
            calibration_status = "proxy prudente"
            calibration_note = "Proxy sintetico di pressione partita: utile come contesto, ma da pesare poco."
        elif available_matches == 0:
            calibration_status = "non disponibile nel backtest"
            calibration_note = "Il fattore non ha avuto dati pre-match sufficienti nelle partite analizzate."

        rows.append(
            {
                "factor_key": factor_name,
                "use_for_calibration": use_for_calibration,
                "sort_priority": 1 if use_for_calibration else 0,
                "Fattore": factor_df["label"].iloc[0],
                "Disponibile in partite": available_matches,
                "Disponibilita storica %": round(float(available_matches / total_matches * 100.0), 1),
                "Frequenza utilizzo %": round(float(available_df["used_actively"].mean() * 100.0), 1) if not available_df.empty else 0.0,
                "Direzione media": round(float(available_df["weighted_impact"].mean()), 2) if not available_df.empty else 0.0,
                "Impatto assoluto medio": round(float(available_df["impact_abs"].mean()), 2) if not available_df.empty else 0.0,
                "Aiuta": int((available_df["help_label"] == "helped").sum()) if not available_df.empty else 0,
                "Non aiuta": int((available_df["help_label"] == "hurt").sum()) if not available_df.empty else 0,
                "Help rate %": round(float((available_df["help_label"] == "helped").mean() * 100.0), 1) if not available_df.empty else 0.0,
                "Stato calibrazione": calibration_status,
                "Nota calibrazione": calibration_note,
            }
        )

    factor_review_df = pd.DataFrame(rows).sort_values(
        ["sort_priority", "Impatto assoluto medio", "Frequenza utilizzo %"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return factor_review_df


def build_calibration_guidance(
    backtest_df: pd.DataFrame,
    factor_review_df: pd.DataFrame,
    general_review: dict[str, Any],
    bucket_review: dict[str, pd.DataFrame],
    ratings_audit: dict[str, Any] | None = None,
    schedule_audit: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    guidance = {
        "factors_to_value": [],
        "factors_to_reduce": [],
        "promising_metrics": [],
        "metrics_to_review": [],
        "sample_warnings": [],
    }
    if backtest_df.empty:
        guidance["metrics_to_review"].append("Dati insufficienti per generare indicazioni di calibrazione affidabili.")
        return guidance

    factor_rows = {}
    if not factor_review_df.empty and "factor_key" in factor_review_df.columns:
        factor_rows = {str(row["factor_key"]): row for row in factor_review_df.to_dict(orient="records")}

    preferred_up = ["table", "home_away", "matchup", "bucket_performance"]
    for factor_key in preferred_up:
        row = factor_rows.get(factor_key)
        if not row or not row.get("use_for_calibration", False):
            continue
        guidance["factors_to_value"].append(
            f"{row['Fattore']}: impatto medio {row['Impatto assoluto medio']:.2f} e help rate {row['Help rate %']:.1f}%."
        )

    recent_row = factor_rows.get("recent_form")
    if recent_row and recent_row.get("use_for_calibration", False):
        guidance["promising_metrics"].append(
            f"{recent_row['Fattore']}: utile nel backtest, ma da mantenere come segnale secondario."
        )

    predictor_row = factor_rows.get("predictor")
    if predictor_row:
        guidance["factors_to_reduce"].append(
            "Predictor esistente come fattore contestuale: meglio usarlo come supporto leggero, non come driver dominante."
        )

    stakes_row = factor_rows.get("stakes")
    if stakes_row:
        guidance["factors_to_reduce"].append(
            "Pressione / stakes: proxy ancora debole, da trattare con prudenza nel peso finale."
        )

    ratings_audit = ratings_audit or {}
    if ratings_audit and not ratings_audit.get("historical_ready", False):
        guidance["metrics_to_review"].append(
            f"Rating Elo da verificare: {ratings_audit.get('note', 'copertura storica non sufficiente per calibrazione.')}"
        )

    schedule_audit = schedule_audit or {}
    if schedule_audit.get("only_league_data", False):
        guidance["sample_warnings"].append(
            "La valutazione calendario e parziale perche nel database corrente mancano dati coppe/europee."
        )

    overall_draw_rate = float((backtest_df["actual_outcome"] == "X").mean())
    adjusted_mask = backtest_df["adjusted_favorite"] != "none"
    overall_adjusted_non_win_rate = (
        float(backtest_df.loc[adjusted_mask, "adjusted_favorite_win"].fillna(False).eq(False).mean())
        if adjusted_mask.any()
        else 0.0
    )

    base_total = int(general_review.get("base_favorite_non_loss_total", 0) or 0)
    adjusted_total = int(general_review.get("adjusted_favorite_non_loss_total", 0) or 0)
    base_rate = float(general_review.get("base_favorite_non_loss_hits", 0) or 0) / base_total if base_total else 0.0
    adjusted_rate = float(general_review.get("adjusted_favorite_non_loss_hits", 0) or 0) / adjusted_total if adjusted_total else 0.0
    if adjusted_rate > base_rate:
        guidance["promising_metrics"].append(
            f"Adjusted edge migliora il favorito base: {adjusted_rate * 100:.1f}% contro {base_rate * 100:.1f}% di non-sconfitta."
        )

    high_draw_total = int(general_review.get("high_draw_total", 0) or 0)
    high_draw_hits = int(general_review.get("high_draw_hits", 0) or 0)
    if high_draw_total:
        high_draw_rate = high_draw_hits / high_draw_total
        if high_draw_rate > overall_draw_rate:
            guidance["promising_metrics"].append(
                f"Draw risk promettente: il bucket alto chiude in pari nel {high_draw_rate * 100:.1f}% dei casi."
            )
        else:
            guidance["metrics_to_review"].append("Draw risk alto non sta ancora separando bene i pareggi rispetto alla media.")
        if high_draw_total < 20:
            guidance["sample_warnings"].append(
                f"Draw risk alto con campione piccolo: {high_draw_total} partite nel bucket alto."
            )

    high_upset_total = int(general_review.get("high_upset_total", 0) or 0)
    high_upset_hits = int(general_review.get("high_upset_hits", 0) or 0)
    if high_upset_total:
        high_upset_rate = high_upset_hits / high_upset_total
        if high_upset_rate <= overall_adjusted_non_win_rate + 0.05:
            guidance["metrics_to_review"].append("Upset risk ancora poco separante: va reso piu specifico sul profilo fragile del favorito.")
        else:
            guidance["promising_metrics"].append("Upset risk mostra segnali utili, ma richiede ancora conferma su un campione piu ampio.")
        if high_upset_total < 20:
            guidance["sample_warnings"].append(
                f"Upset risk alto con campione piccolo: {high_upset_total} partite nel bucket alto."
            )
    else:
        guidance["metrics_to_review"].append("Upset risk ancora poco separante: il bucket alto non ha ancora un campione utile per valutarlo.")

    high_conf_df = backtest_df.loc[(backtest_df["confidence_bucket"] == "alto") & adjusted_mask]
    low_conf_df = backtest_df.loc[(backtest_df["confidence_bucket"] == "basso") & adjusted_mask]
    if not high_conf_df.empty and not low_conf_df.empty:
        high_conf_rate = float(high_conf_df["adjusted_favorite_not_lose"].fillna(False).mean())
        low_conf_rate = float(low_conf_df["adjusted_favorite_not_lose"].fillna(False).mean())
        if high_conf_rate > low_conf_rate:
            guidance["promising_metrics"].append(
                f"Confidence alta utile: {high_conf_rate * 100:.1f}% contro {low_conf_rate * 100:.1f}% della confidence bassa."
            )
        else:
            guidance["metrics_to_review"].append("Confidence non separa ancora bene i match piu affidabili da quelli fragili.")

    if int(general_review.get("matches_analyzed", 0) or 0) < 80:
        guidance["sample_warnings"].append("Campione review ancora ridotto: le indicazioni vanno lette con prudenza.")

    for key in guidance:
        deduped: list[str] = []
        for item in guidance[key]:
            if item and item not in deduped:
                deduped.append(item)
        guidance[key] = deduped

    return guidance


def build_review_conclusions(
    backtest_df: pd.DataFrame,
    factor_review_df: pd.DataFrame,
    general_review: dict[str, Any],
) -> list[str]:
    if backtest_df.empty:
        return ["Dati insufficienti per formulare conclusioni automatiche sul motore contestuale."]

    conclusions: list[str] = []
    overall_draw_rate = float((backtest_df["actual_outcome"] == "X").mean())
    adjusted_mask = backtest_df["adjusted_favorite"] != "none"
    overall_adjusted_non_win_rate = float(
        backtest_df.loc[adjusted_mask, "adjusted_favorite_win"].fillna(False).eq(False).mean()
    ) if adjusted_mask.any() else 0.0

    base_total = int(general_review.get("base_favorite_non_loss_total", 0) or 0)
    adjusted_total = int(general_review.get("adjusted_favorite_non_loss_total", 0) or 0)
    base_rate = (
        float(general_review["base_favorite_non_loss_hits"]) / base_total
        if base_total
        else 0.0
    )
    adjusted_rate = (
        float(general_review["adjusted_favorite_non_loss_hits"]) / adjusted_total
        if adjusted_total
        else 0.0
    )
    if adjusted_rate >= base_rate + 0.05:
        conclusions.append("L'edge corretto sembra aggiungere valore rispetto al vantaggio base nelle letture complessive.")
    elif adjusted_rate <= base_rate - 0.05:
        conclusions.append("L'edge corretto non sta ancora migliorando in modo chiaro il giudizio base e va calibrato meglio.")
    else:
        conclusions.append("La differenza tra vantaggio base e vantaggio corretto e per ora contenuta.")

    equilibrate_df = backtest_df.loc[backtest_df["adjusted_edge"].abs() < 6.0]
    if not equilibrate_df.empty and float(equilibrate_df["context_helped"].mean()) >= 0.40:
        conclusions.append("Il context_engine sembra aiutare soprattutto nelle partite piu equilibrate.")

    high_draw_df = backtest_df.loc[backtest_df["high_draw_risk"]]
    if len(high_draw_df) >= 5:
        high_draw_rate = float((high_draw_df["actual_outcome"] == "X").mean())
        if high_draw_rate >= overall_draw_rate + 0.08:
            conclusions.append("Il draw_risk alto sembra intercettare abbastanza bene una parte delle partite che finiscono in pari.")
        else:
            conclusions.append("Il draw_risk alto non mostra ancora un vantaggio netto rispetto al tasso medio di pareggi.")

    high_upset_df = backtest_df.loc[backtest_df["high_upset_risk"] & adjusted_mask]
    if len(high_upset_df) >= 5:
        upset_hit_rate = float(high_upset_df["adjusted_favorite_win"].fillna(False).eq(False).mean())
        if upset_hit_rate >= overall_adjusted_non_win_rate + 0.08:
            conclusions.append("L'upset_risk alto sembra utile per segnalare partite in cui il favorito contestuale rischia davvero di non vincere.")
        else:
            conclusions.append("L'upset_risk alto non sembra ancora abbastanza separante rispetto alla media delle partite aperte.")

    high_conf_df = backtest_df.loc[(backtest_df["confidence_bucket"] == "alto") & adjusted_mask]
    low_conf_df = backtest_df.loc[(backtest_df["confidence_bucket"] == "basso") & adjusted_mask]
    if not high_conf_df.empty and not low_conf_df.empty:
        high_conf_rate = float(high_conf_df["adjusted_favorite_not_lose"].fillna(False).mean())
        low_conf_rate = float(low_conf_df["adjusted_favorite_not_lose"].fillna(False).mean())
        if high_conf_rate >= low_conf_rate + 0.10:
            conclusions.append("La confidence alta sembra davvero piu affidabile della confidence bassa.")
        else:
            conclusions.append("La confidence non separa ancora in modo forte i matchup piu affidabili da quelli piu fragili.")

    if not factor_review_df.empty:
        top_factors = factor_review_df.head(3)["Fattore"].tolist()
        conclusions.append(f"I fattori piu influenti in media sembrano essere: {', '.join(top_factors)}.")

    return conclusions[:6]


def build_model_review(
    season_df: pd.DataFrame,
    minimum_team_history: int = 1,
    schedule_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    schedule_source_df = schedule_df if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty else season_df
    ratings_audit = build_ratings_audit(season_df)
    schedule_audit = build_schedule_data_audit(schedule_source_df)
    backtest_df, factors_df = build_backtest_rows(
        season_df,
        minimum_team_history=minimum_team_history,
        schedule_df=schedule_source_df,
    )
    general_review = build_general_review(backtest_df)
    diagnostic_tables = build_diagnostic_tables(backtest_df)
    bucket_review = build_bucket_review(backtest_df)
    factor_review_df = build_factor_review(factors_df, ratings_audit=ratings_audit, schedule_audit=schedule_audit)
    conclusions = build_review_conclusions(backtest_df, factor_review_df, general_review)
    calibration_guidance = build_calibration_guidance(
        backtest_df,
        factor_review_df,
        general_review,
        bucket_review,
        ratings_audit=ratings_audit,
        schedule_audit=schedule_audit,
    )

    return {
        "ok": not backtest_df.empty,
        "message": None if not backtest_df.empty else "Dati insufficienti per costruire il backtest stagionale.",
        "backtest_df": backtest_df,
        "factors_df": factors_df,
        "ratings_audit": ratings_audit,
        "schedule_audit": schedule_audit,
        "general_review": general_review,
        "diagnostic_tables": diagnostic_tables,
        "bucket_review": bucket_review,
        "factor_review": factor_review_df,
        "calibration_guidance": calibration_guidance,
        "conclusions": conclusions,
    }
