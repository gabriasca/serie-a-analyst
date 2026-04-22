from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics import RESULT_LABELS, build_standings, get_teams, prepare_matches_dataframe
from src.ratings import fetch_latest_team_ratings


DEFAULT_RECENT_MATCHES = 5
METRIC_COLUMNS = [
    "offensive_threat_index",
    "defensive_solidity_index",
    "offensive_volume_index",
    "defensive_risk_index",
    "finishing_efficiency_index",
    "home_dependency_index",
    "recent_momentum_index",
    "schedule_strength_index",
]
DISPLAY_METRIC_LABELS = {
    "offensive_threat_index": "Pericolosita offensiva",
    "defensive_solidity_index": "Solidita difensiva",
    "offensive_volume_index": "Volume offensivo",
    "defensive_risk_index": "Rischio difensivo",
    "finishing_efficiency_index": "Efficienza realizzativa",
    "home_dependency_index": "Dipendenza casa",
    "recent_momentum_index": "Momento recente",
    "schedule_strength_index": "Forza calendario",
}


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _safe_ratio(numerator: float | None, denominator: float | None, multiplier: float = 1.0) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator) * multiplier


def _empty_metrics_dataframe() -> pd.DataFrame:
    columns = [
        "team",
        "position",
        "points",
        "matches",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points_per_match",
        "goals_for_avg",
        "goals_against_avg",
        "shots_avg",
        "shots_conceded_avg",
        "shots_on_target_avg",
        "shots_on_target_conceded_avg",
        "corners_avg",
        "cards_avg",
        "goals_per_shot",
        "goals_per_shot_on_target",
        "home_points",
        "away_points",
        "home_ppm",
        "away_ppm",
        "home_goal_balance_avg",
        "away_goal_balance_avg",
        "recent_form",
        "recent_matches",
        "recent_points",
        "recent_goals_for",
        "recent_goals_against",
        "recent_points_per_match",
        "recent_goal_difference_per_match",
        "recent_goals_for_avg",
        "recent_goals_against_avg",
        "elo_rating",
        "elo_rank",
        "strength_band",
        "rating_date",
        "rating_source",
        "schedule_strength_raw",
        "schedule_strength_source",
        "schedule_strength_note",
        *METRIC_COLUMNS,
    ]
    metrics_df = pd.DataFrame(columns=columns)
    metrics_df.attrs["schedule_strength_source"] = "non disponibile"
    return metrics_df


def _build_team_match_logs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return {}

    records: list[dict[str, Any]] = []
    for row in prepared_df.to_dict(orient="records"):
        home_goals = int(row.get("home_goals", 0) or 0)
        away_goals = int(row.get("away_goals", 0) or 0)

        if row.get("home_team"):
            home_result = "W" if home_goals > away_goals else "L" if home_goals < away_goals else "D"
            home_points = 3 if home_result == "W" else 1 if home_result == "D" else 0
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "team": str(row.get("home_team")),
                    "opponent": str(row.get("away_team")),
                    "venue": "Casa",
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                    "goal_difference": home_goals - away_goals,
                    "points": home_points,
                    "result": home_result,
                    "display_result": RESULT_LABELS[home_result],
                    "shots_for": _safe_float(row.get("home_shots")),
                    "shots_against": _safe_float(row.get("away_shots")),
                    "shots_on_target_for": _safe_float(row.get("home_shots_on_target")),
                    "shots_on_target_against": _safe_float(row.get("away_shots_on_target")),
                    "corners_for": _safe_float(row.get("home_corners")),
                    "corners_against": _safe_float(row.get("away_corners")),
                    "cards_for": _safe_float(row.get("home_cards")),
                    "cards_against": _safe_float(row.get("away_cards")),
                }
            )

        if row.get("away_team"):
            away_result = "W" if away_goals > home_goals else "L" if away_goals < home_goals else "D"
            away_points = 3 if away_result == "W" else 1 if away_result == "D" else 0
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "team": str(row.get("away_team")),
                    "opponent": str(row.get("home_team")),
                    "venue": "Trasferta",
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                    "goal_difference": away_goals - home_goals,
                    "points": away_points,
                    "result": away_result,
                    "display_result": RESULT_LABELS[away_result],
                    "shots_for": _safe_float(row.get("away_shots")),
                    "shots_against": _safe_float(row.get("home_shots")),
                    "shots_on_target_for": _safe_float(row.get("away_shots_on_target")),
                    "shots_on_target_against": _safe_float(row.get("home_shots_on_target")),
                    "corners_for": _safe_float(row.get("away_corners")),
                    "corners_against": _safe_float(row.get("home_corners")),
                    "cards_for": _safe_float(row.get("away_cards")),
                    "cards_against": _safe_float(row.get("home_cards")),
                }
            )

    if not records:
        return {}

    team_logs: dict[str, pd.DataFrame] = {}
    matches_df = pd.DataFrame(records)
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"], errors="coerce")
    matches_df = matches_df.sort_values("match_date").reset_index(drop=True)

    for team_name, team_df in matches_df.groupby("team", sort=True):
        team_logs[str(team_name)] = team_df.reset_index(drop=True)
    return team_logs


def normalize_metric_0_100(values: pd.Series | list[float] | tuple[float, ...], higher_is_better: bool = True) -> pd.Series:
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="float64")

    std = float(valid.std(ddof=0))
    if std <= 0:
        normalized = pd.Series(50.0, index=series.index, dtype="float64")
    else:
        z_score = (numeric - float(valid.mean())) / std
        if not higher_is_better:
            z_score = -z_score
        normalized = (50.0 + z_score * 15.0).clip(lower=0.0, upper=100.0)

    normalized[numeric.isna()] = pd.NA
    return normalized.round(1)


def _combine_component_scores(base_df: pd.DataFrame, components: list[tuple[str, pd.Series, float]]) -> pd.Series:
    if base_df.empty:
        return pd.Series(dtype="float64")

    component_df = pd.DataFrame(index=base_df.index)
    weight_map: dict[str, float] = {}
    for name, series, weight in components:
        component_df[name] = pd.to_numeric(series, errors="coerce")
        weight_map[name] = weight

    weighted_sum = pd.Series(0.0, index=base_df.index, dtype="float64")
    total_weight = pd.Series(0.0, index=base_df.index, dtype="float64")
    for name, weight in weight_map.items():
        series = component_df[name]
        available = series.notna().astype(float)
        weighted_sum = weighted_sum + series.fillna(0.0) * weight
        total_weight = total_weight + available * weight

    scores = pd.Series(50.0, index=base_df.index, dtype="float64")
    valid_rows = total_weight > 0
    scores.loc[valid_rows] = (weighted_sum.loc[valid_rows] / total_weight.loc[valid_rows]).round(1)
    return scores.clip(lower=0.0, upper=100.0)


def compute_team_base_stats(df: pd.DataFrame, ratings_df: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return _empty_metrics_dataframe()

    standings = build_standings(prepared_df)
    teams = get_teams(prepared_df)
    if not teams:
        return _empty_metrics_dataframe()

    if ratings_df is None:
        ratings_df = fetch_latest_team_ratings(teams=teams)
    ratings_df = ratings_df.copy() if ratings_df is not None else pd.DataFrame()
    if not ratings_df.empty:
        ratings_df["team_name"] = ratings_df["team_name"].astype(str)

    rating_map = {
        str(row["team_name"]): {
            "rating_value": _safe_float(row.get("rating_value")),
            "rating_date": row.get("rating_date"),
            "source_name": row.get("source_name"),
        }
        for row in ratings_df.to_dict(orient="records")
        if row.get("team_name")
    }
    ratings_available_for_all = bool(teams) and all(team in rating_map for team in teams)

    position_map = {str(row["Team"]): int(position) for position, row in standings.iterrows()}
    team_count = len(position_map)
    position_strength_map = {team_name: float(team_count - position + 1) for team_name, position in position_map.items()}

    rating_rank_map: dict[str, int] = {}
    if ratings_available_for_all:
        ordered_by_rating = (
            ratings_df.sort_values(["rating_value", "team_name"], ascending=[False, True])["team_name"].astype(str).tolist()
        )
        rating_rank_map = {team_name: index for index, team_name in enumerate(ordered_by_rating, start=1)}

    if ratings_available_for_all:
        schedule_source = "elo"
    elif position_map:
        schedule_source = "classifica"
    else:
        schedule_source = "non disponibile"

    team_logs = _build_team_match_logs(prepared_df)
    rows: list[dict[str, Any]] = []
    for team in teams:
        team_df = team_logs.get(team, pd.DataFrame())
        if team_df.empty:
            continue

        row = standings.loc[standings["Team"] == team]
        if row.empty:
            continue
        standing_row = row.iloc[0]

        home_df = team_df.loc[team_df["venue"] == "Casa"]
        away_df = team_df.loc[team_df["venue"] == "Trasferta"]
        recent_df = team_df.tail(DEFAULT_RECENT_MATCHES)
        rating_info = rating_map.get(team, {})

        schedule_values: list[float] = []
        if schedule_source == "elo":
            schedule_values = [
                float(rating_map[opponent]["rating_value"])
                for opponent in team_df["opponent"].astype(str).tolist()
                if opponent in rating_map and rating_map[opponent]["rating_value"] is not None
            ]
        elif schedule_source == "classifica":
            schedule_values = [
                float(position_strength_map[opponent])
                for opponent in team_df["opponent"].astype(str).tolist()
                if opponent in position_strength_map
            ]

        rating_date = rating_info.get("rating_date")
        if isinstance(rating_date, pd.Timestamp):
            rating_date = rating_date.strftime("%Y-%m-%d")

        rows.append(
            {
                "team": team,
                "position": position_map.get(team),
                "points": int(standing_row["Pts"]),
                "matches": int(standing_row["GP"]),
                "goals_for": int(standing_row["GF"]),
                "goals_against": int(standing_row["GA"]),
                "goal_difference": int(standing_row["DR"]),
                "points_per_match": round(int(standing_row["Pts"]) / max(int(standing_row["GP"]), 1), 2),
                "goals_for_avg": round(float(team_df["goals_for"].mean()), 2),
                "goals_against_avg": round(float(team_df["goals_against"].mean()), 2),
                "shots_avg": _safe_mean(team_df["shots_for"]),
                "shots_conceded_avg": _safe_mean(team_df["shots_against"]),
                "shots_on_target_avg": _safe_mean(team_df["shots_on_target_for"]),
                "shots_on_target_conceded_avg": _safe_mean(team_df["shots_on_target_against"]),
                "corners_avg": _safe_mean(team_df["corners_for"]),
                "cards_avg": _safe_mean(team_df["cards_for"]),
                "goals_per_shot": _safe_ratio(
                    float(team_df["goals_for"].sum()),
                    pd.to_numeric(team_df["shots_for"], errors="coerce").sum(),
                    multiplier=100.0,
                ),
                "goals_per_shot_on_target": _safe_ratio(
                    float(team_df["goals_for"].sum()),
                    pd.to_numeric(team_df["shots_on_target_for"], errors="coerce").sum(),
                    multiplier=100.0,
                ),
                "home_points": int(home_df["points"].sum()) if not home_df.empty else 0,
                "away_points": int(away_df["points"].sum()) if not away_df.empty else 0,
                "home_ppm": round(float(home_df["points"].sum()) / len(home_df), 2) if not home_df.empty else 0.0,
                "away_ppm": round(float(away_df["points"].sum()) / len(away_df), 2) if not away_df.empty else 0.0,
                "home_goal_balance_avg": round(float(home_df["goal_difference"].mean()), 2) if not home_df.empty else 0.0,
                "away_goal_balance_avg": round(float(away_df["goal_difference"].mean()), 2) if not away_df.empty else 0.0,
                "recent_form": " ".join(recent_df["display_result"].tolist()) or "-",
                "recent_matches": int(len(recent_df)),
                "recent_points": int(recent_df["points"].sum()) if not recent_df.empty else 0,
                "recent_goals_for": int(recent_df["goals_for"].sum()) if not recent_df.empty else 0,
                "recent_goals_against": int(recent_df["goals_against"].sum()) if not recent_df.empty else 0,
                "recent_points_per_match": round(float(recent_df["points"].sum()) / len(recent_df), 2)
                if not recent_df.empty
                else 0.0,
                "recent_goal_difference_per_match": round(float(recent_df["goal_difference"].mean()), 2)
                if not recent_df.empty
                else 0.0,
                "recent_goals_for_avg": round(float(recent_df["goals_for"].mean()), 2) if not recent_df.empty else 0.0,
                "recent_goals_against_avg": round(float(recent_df["goals_against"].mean()), 2)
                if not recent_df.empty
                else 0.0,
                "elo_rating": rating_info.get("rating_value"),
                "elo_rank": rating_rank_map.get(team),
                "strength_band": None,
                "rating_date": rating_date,
                "rating_source": rating_info.get("source_name"),
                "schedule_strength_raw": round(sum(schedule_values) / len(schedule_values), 2) if schedule_values else None,
                "schedule_strength_source": schedule_source,
                "schedule_strength_note": "",
            }
        )

    base_stats = pd.DataFrame(rows)
    if base_stats.empty:
        return _empty_metrics_dataframe()

    if ratings_available_for_all and not base_stats["elo_rank"].isna().all():
        total = len(base_stats)
        bands: list[str] = []
        for rank in base_stats["elo_rank"].fillna(total).astype(int).tolist():
            percentile = (rank - 1) / max(total - 1, 1)
            if percentile <= 0.2:
                bands.append("molto alta")
            elif percentile <= 0.45:
                bands.append("alta")
            elif percentile <= 0.75:
                bands.append("media")
            else:
                bands.append("bassa")
        base_stats["strength_band"] = bands

    source_notes = {
        "elo": "Indice costruito con la media Elo degli avversari gia affrontati.",
        "classifica": "Indice costruito con la classifica corrente perche il layer Elo non copre tutta la stagione.",
        "non disponibile": "Indice non disponibile con i dati correnti.",
    }
    base_stats["schedule_strength_note"] = base_stats["schedule_strength_source"].map(source_notes).fillna(
        source_notes["non disponibile"]
    )
    base_stats.attrs["schedule_strength_source"] = schedule_source
    return base_stats


def compute_offensive_threat_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("goals_for_avg", normalize_metric_0_100(base_stats["goals_for_avg"]), 0.32),
            ("shots_avg", normalize_metric_0_100(base_stats["shots_avg"]), 0.22),
            ("shots_on_target_avg", normalize_metric_0_100(base_stats["shots_on_target_avg"]), 0.24),
            ("corners_avg", normalize_metric_0_100(base_stats["corners_avg"]), 0.10),
            ("recent_goals_for_avg", normalize_metric_0_100(base_stats["recent_goals_for_avg"]), 0.12),
        ],
    )


def compute_defensive_solidity_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("goals_against_avg", normalize_metric_0_100(base_stats["goals_against_avg"], higher_is_better=False), 0.35),
            (
                "shots_conceded_avg",
                normalize_metric_0_100(base_stats["shots_conceded_avg"], higher_is_better=False),
                0.20,
            ),
            (
                "shots_on_target_conceded_avg",
                normalize_metric_0_100(base_stats["shots_on_target_conceded_avg"], higher_is_better=False),
                0.25,
            ),
            (
                "recent_goals_against_avg",
                normalize_metric_0_100(base_stats["recent_goals_against_avg"], higher_is_better=False),
                0.20,
            ),
        ],
    )


def compute_offensive_volume_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("shots_avg", normalize_metric_0_100(base_stats["shots_avg"]), 0.45),
            ("shots_on_target_avg", normalize_metric_0_100(base_stats["shots_on_target_avg"]), 0.35),
            ("corners_avg", normalize_metric_0_100(base_stats["corners_avg"]), 0.20),
        ],
    )


def compute_defensive_risk_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("goals_against_avg", normalize_metric_0_100(base_stats["goals_against_avg"]), 0.40),
            ("shots_conceded_avg", normalize_metric_0_100(base_stats["shots_conceded_avg"]), 0.25),
            ("shots_on_target_conceded_avg", normalize_metric_0_100(base_stats["shots_on_target_conceded_avg"]), 0.35),
        ],
    )


def compute_finishing_efficiency_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("goals_per_shot_on_target", normalize_metric_0_100(base_stats["goals_per_shot_on_target"]), 0.60),
            ("goals_per_shot", normalize_metric_0_100(base_stats["goals_per_shot"]), 0.40),
        ],
    )


def compute_home_dependency_index(base_stats: pd.DataFrame) -> pd.Series:
    dependency_df = pd.DataFrame(index=base_stats.index)
    dependency_df["ppm_gap_abs"] = (base_stats["home_ppm"] - base_stats["away_ppm"]).abs()
    dependency_df["goal_balance_gap_abs"] = (
        pd.to_numeric(base_stats["home_goal_balance_avg"], errors="coerce")
        - pd.to_numeric(base_stats["away_goal_balance_avg"], errors="coerce")
    ).abs()
    return _combine_component_scores(
        dependency_df,
        [
            ("ppm_gap_abs", normalize_metric_0_100(dependency_df["ppm_gap_abs"]), 0.60),
            ("goal_balance_gap_abs", normalize_metric_0_100(dependency_df["goal_balance_gap_abs"]), 0.40),
        ],
    )


def compute_recent_momentum_index(base_stats: pd.DataFrame) -> pd.Series:
    return _combine_component_scores(
        base_stats,
        [
            ("recent_points_per_match", normalize_metric_0_100(base_stats["recent_points_per_match"]), 0.55),
            (
                "recent_goal_difference_per_match",
                normalize_metric_0_100(base_stats["recent_goal_difference_per_match"]),
                0.25,
            ),
            ("recent_goals_for_avg", normalize_metric_0_100(base_stats["recent_goals_for_avg"]), 0.20),
        ],
    )


def compute_schedule_strength_index(base_stats: pd.DataFrame) -> pd.Series:
    if base_stats.empty:
        return pd.Series(dtype="float64")
    if base_stats.attrs.get("schedule_strength_source", "non disponibile") == "non disponibile":
        return pd.Series(pd.NA, index=base_stats.index, dtype="float64")
    return normalize_metric_0_100(base_stats["schedule_strength_raw"])


def build_metric_explanations(team_metrics: dict[str, Any]) -> dict[str, str]:
    schedule_source = str(team_metrics.get("schedule_strength_source") or "non disponibile")
    schedule_suffix = {
        "elo": "Usa la forza Elo media degli avversari gia affrontati.",
        "classifica": "Usa la classifica corrente degli avversari perche il layer Elo non e completo.",
        "non disponibile": "Non e disponibile con i dati correnti.",
    }.get(schedule_source, "Non e disponibile con i dati correnti.")

    return {
        "Pericolosita offensiva": (
            "Combina gol medi, tiri, tiri in porta, corner e spinta offensiva recente. "
            f"Valore attuale: {team_metrics.get('offensive_threat_index', 'n/d')}/100."
        ),
        "Solidita difensiva": (
            "Legge quanti gol e conclusioni pulite la squadra concede in media, con peso anche sulla fase recente. "
            f"Valore attuale: {team_metrics.get('defensive_solidity_index', 'n/d')}/100."
        ),
        "Volume offensivo": (
            "Misura la quantita di produzione offensiva attraverso tiri, tiri in porta e corner. "
            f"Valore attuale: {team_metrics.get('offensive_volume_index', 'n/d')}/100."
        ),
        "Rischio difensivo": (
            "Misura quanta pressione concede la squadra agli avversari. Un valore alto indica rischio maggiore. "
            f"Valore attuale: {team_metrics.get('defensive_risk_index', 'n/d')}/100."
        ),
        "Efficienza realizzativa": (
            "Legge quanto bene la squadra converte tiri e tiri in porta in gol, senza usare metriche esterne. "
            f"Valore attuale: {team_metrics.get('finishing_efficiency_index', 'n/d')}/100."
        ),
        "Dipendenza casa": (
            "Pesa il divario tra rendimento interno ed esterno in punti e differenza reti media. "
            f"Valore attuale: {team_metrics.get('home_dependency_index', 'n/d')}/100."
        ),
        "Momento recente": (
            "Considera punti, differenza reti e spinta offensiva nelle ultime cinque partite disponibili. "
            f"Valore attuale: {team_metrics.get('recent_momentum_index', 'n/d')}/100."
        ),
        "Forza calendario": (
            "Misura il livello medio degli avversari gia incontrati. "
            f"{schedule_suffix} Valore attuale: {team_metrics.get('schedule_strength_index', 'n/d')}/100."
        ),
    }


def build_metric_strengths_and_weaknesses(team_metrics: dict[str, Any]) -> dict[str, list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    offensive_threat = _safe_float(team_metrics.get("offensive_threat_index"))
    defensive_solidity = _safe_float(team_metrics.get("defensive_solidity_index"))
    offensive_volume = _safe_float(team_metrics.get("offensive_volume_index"))
    defensive_risk = _safe_float(team_metrics.get("defensive_risk_index"))
    finishing_efficiency = _safe_float(team_metrics.get("finishing_efficiency_index"))
    home_dependency = _safe_float(team_metrics.get("home_dependency_index"))
    recent_momentum = _safe_float(team_metrics.get("recent_momentum_index"))
    schedule_strength = _safe_float(team_metrics.get("schedule_strength_index"))

    if offensive_threat is not None and offensive_threat >= 62:
        strengths.append("Crea pericoli offensivi con continuita sopra la media del campionato.")
    if defensive_solidity is not None and defensive_solidity >= 62:
        strengths.append("Tiene una struttura difensiva solida e concede poco di pulito.")
    if offensive_volume is not None and offensive_volume >= 60:
        strengths.append("Sostiene bene il volume offensivo tra tiri, tiri in porta e corner.")
    if finishing_efficiency is not None and finishing_efficiency >= 60:
        strengths.append("Converte bene il volume offensivo disponibile in gol.")
    if recent_momentum is not None and recent_momentum >= 60:
        strengths.append("Il momento recente e positivo e accompagna il rendimento generale.")
    if schedule_strength is not None and schedule_strength >= 60:
        strengths.append("Ha costruito i propri numeri contro un calendario gia piuttosto impegnativo.")

    if offensive_threat is not None and offensive_threat <= 40:
        weaknesses.append("La pericolosita offensiva resta sotto media e produce pochi segnali forti.")
    if defensive_solidity is not None and defensive_solidity <= 40:
        weaknesses.append("La tenuta difensiva e fragile e il profilo senza palla va protetto meglio.")
    if defensive_risk is not None and defensive_risk >= 60:
        weaknesses.append("Concede troppo volume o troppe conclusioni pulite agli avversari.")
    if finishing_efficiency is not None and finishing_efficiency <= 40:
        weaknesses.append("La finalizzazione pesa poco rispetto al volume creato.")
    if home_dependency is not None and home_dependency >= 65:
        weaknesses.append("Il profilo resta molto legato al rendimento interno.")
    if recent_momentum is not None and recent_momentum <= 40:
        weaknesses.append("Il momento recente e in calo e frena la lettura complessiva del team.")

    if not strengths:
        strengths.append("Il profilo non mostra un vantaggio netto, ma resta abbastanza ordinato nelle varie aree.")
    if not weaknesses:
        weaknesses.append("Non emergono criticita nette, anche se il margine di crescita resta distribuito.")

    return {"strengths": strengths[:4], "weaknesses": weaknesses[:4]}


def build_metric_summary(team_metrics: dict[str, Any]) -> str:
    team_name = str(team_metrics.get("team") or "La squadra")
    schedule_source = str(team_metrics.get("schedule_strength_source") or "non disponibile")
    schedule_label = {
        "elo": "con forza avversaria letta via Elo",
        "classifica": "con forza avversaria letta via classifica",
        "non disponibile": "senza un indice affidabile di forza calendario",
    }.get(schedule_source, "senza un indice affidabile di forza calendario")

    lines = [
        (
            f"{team_name} ha una pericolosita offensiva di {team_metrics.get('offensive_threat_index', 'n/d')}/100 "
            f"e una solidita difensiva di {team_metrics.get('defensive_solidity_index', 'n/d')}/100."
        ),
        (
            f"Il volume offensivo si assesta a {team_metrics.get('offensive_volume_index', 'n/d')}/100, "
            f"mentre il rischio difensivo e {team_metrics.get('defensive_risk_index', 'n/d')}/100."
        ),
        (
            f"L'efficienza realizzativa vale {team_metrics.get('finishing_efficiency_index', 'n/d')}/100 "
            f"e la dipendenza casa {team_metrics.get('home_dependency_index', 'n/d')}/100."
        ),
        (
            f"Il momento recente vale {team_metrics.get('recent_momentum_index', 'n/d')}/100, "
            f"con sequenza {team_metrics.get('recent_form') or '-'} nelle ultime {team_metrics.get('recent_matches') or 0} gare."
        ),
        (
            f"La lettura del calendario affrontato e {schedule_label}: "
            f"indice attuale {team_metrics.get('schedule_strength_index', 'n/d')}/100."
        ),
        "Interpretazione prudente: questi indicatori sono interni e descrivono tendenze aggregate, non certezze.",
    ]
    return "\n".join(lines[:6])


def get_team_advanced_metrics(metrics_df: pd.DataFrame, team_name: str) -> dict[str, Any] | None:
    if metrics_df.empty or "team" not in metrics_df.columns:
        return None
    team_row = metrics_df.loc[metrics_df["team"] == team_name]
    if team_row.empty:
        return None
    data = team_row.iloc[0].to_dict()
    for key, value in list(data.items()):
        if isinstance(value, pd.Timestamp):
            data[key] = value.strftime("%Y-%m-%d")
        elif pd.isna(value):
            data[key] = None
    return data


def build_advanced_team_metrics(df: pd.DataFrame, ratings_df: pd.DataFrame | None = None) -> pd.DataFrame:
    base_stats = compute_team_base_stats(df, ratings_df=ratings_df)
    if base_stats.empty:
        return base_stats

    metrics_df = base_stats.copy()
    metrics_df["offensive_threat_index"] = compute_offensive_threat_index(base_stats)
    metrics_df["defensive_solidity_index"] = compute_defensive_solidity_index(base_stats)
    metrics_df["offensive_volume_index"] = compute_offensive_volume_index(base_stats)
    metrics_df["defensive_risk_index"] = compute_defensive_risk_index(base_stats)
    metrics_df["finishing_efficiency_index"] = compute_finishing_efficiency_index(base_stats)
    metrics_df["home_dependency_index"] = compute_home_dependency_index(base_stats)
    metrics_df["recent_momentum_index"] = compute_recent_momentum_index(base_stats)
    metrics_df["schedule_strength_index"] = compute_schedule_strength_index(base_stats)

    for column in METRIC_COLUMNS:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce").round(1)

    metrics_df = metrics_df.sort_values(["position", "points", "team"], ascending=[True, False, True]).reset_index(
        drop=True
    )
    metrics_df.attrs["schedule_strength_source"] = base_stats.attrs.get("schedule_strength_source", "non disponibile")
    return metrics_df
