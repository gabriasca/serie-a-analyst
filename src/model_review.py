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
from src.team_profiles import build_team_profile_context, build_team_profile_with_ratings


EDGE_THRESHOLD = 2.0
HIGH_DRAW_RISK_THRESHOLD = 55.0
HIGH_UPSET_RISK_THRESHOLD = 55.0
LOW_CONFIDENCE_THRESHOLD = 45.0
HIGH_CONFIDENCE_THRESHOLD = 65.0


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_df = prepare_matches_dataframe(season_df)
    if prepared_df.empty:
        return pd.DataFrame(), pd.DataFrame()

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
        context_engine = build_context_adjusted_edge(
            home_profile,
            away_profile,
            predictor_context=predictor_context,
            mismatches=mismatches,
            style_advantage=style_advantage,
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


def build_factor_review(factors_df: pd.DataFrame) -> pd.DataFrame:
    if factors_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total_matches = max(int(factors_df[["match_date", "home_team", "away_team"]].drop_duplicates().shape[0]), 1)
    for factor_name, factor_df in factors_df.groupby("factor"):
        rows.append(
            {
                "Fattore": factor_df["label"].iloc[0],
                "Presente in partite": int(len(factor_df)),
                "Frequenza utilizzo %": round(float(factor_df["used_actively"].mean() * 100.0), 1),
                "Direzione media": round(float(factor_df["weighted_impact"].mean()), 2),
                "Impatto assoluto medio": round(float(factor_df["impact_abs"].mean()), 2),
                "Aiuta": int((factor_df["help_label"] == "helped").sum()),
                "Non aiuta": int((factor_df["help_label"] == "hurt").sum()),
                "Help rate %": round(float((factor_df["help_label"] == "helped").mean() * 100.0), 1),
                "Copertura match %": round(float(factor_df["match_date"].count() / total_matches * 100.0), 1),
            }
        )

    factor_review_df = pd.DataFrame(rows).sort_values(
        ["Impatto assoluto medio", "Frequenza utilizzo %"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return factor_review_df


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
) -> dict[str, Any]:
    backtest_df, factors_df = build_backtest_rows(season_df, minimum_team_history=minimum_team_history)
    general_review = build_general_review(backtest_df)
    diagnostic_tables = build_diagnostic_tables(backtest_df)
    bucket_review = build_bucket_review(backtest_df)
    factor_review_df = build_factor_review(factors_df)
    conclusions = build_review_conclusions(backtest_df, factor_review_df, general_review)

    return {
        "ok": not backtest_df.empty,
        "message": None if not backtest_df.empty else "Dati insufficienti per costruire il backtest stagionale.",
        "backtest_df": backtest_df,
        "factors_df": factors_df,
        "general_review": general_review,
        "diagnostic_tables": diagnostic_tables,
        "bucket_review": bucket_review,
        "factor_review": factor_review_df,
        "conclusions": conclusions,
    }
