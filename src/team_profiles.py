from __future__ import annotations

from typing import Any

import pandas as pd

from src.advanced_metrics import (
    build_advanced_team_metrics,
    build_metric_strengths_and_weaknesses,
    get_team_advanced_metrics,
)
from src.analytics import RESULT_LABELS, build_standings, get_teams, prepare_matches_dataframe
from src.ratings import build_strength_bucket_map, enrich_standings_with_ratings


TOP_BUCKET_KEY = "top"
MIDDLE_BUCKET_KEY = "middle"
BOTTOM_BUCKET_KEY = "bottom"
TOP_BUCKET_LABEL = "vs top 6"
MIDDLE_BUCKET_LABEL = "vs medio gruppo"
BOTTOM_BUCKET_LABEL = "vs ultime 6"


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
    return round(float(numeric.mean()), 2)


def _safe_sum(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.sum())


def _safe_ratio(numerator: float, denominator: float, multiplier: float = 1.0) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * multiplier, 2)


def _clamp_index(value: float, lower: float = 40.0, upper: float = 160.0) -> float:
    return round(max(lower, min(upper, value)), 1)


def _ratio_index(value: float | None, reference: float | None, inverse: bool = False) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    if inverse:
        adjusted = max(value, 0.05)
        return _clamp_index((reference / adjusted) * 100.0)
    return _clamp_index((value / reference) * 100.0)


def _weighted_index(score_weights: list[tuple[float | None, float]]) -> float:
    usable = [(score, weight) for score, weight in score_weights if score is not None and weight > 0]
    if not usable:
        return 100.0
    weighted_sum = sum(score * weight for score, weight in usable)
    total_weight = sum(weight for _, weight in usable)
    return round(weighted_sum / total_weight, 1)


def _result_and_points(goals_for: int, goals_against: int) -> tuple[str, int]:
    if goals_for > goals_against:
        return "W", 3
    if goals_for < goals_against:
        return "L", 0
    return "D", 1


def _build_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in prepared_df.to_dict(orient="records"):
        if row.get("home_team") == team:
            goals_for = int(row.get("home_goals", 0) or 0)
            goals_against = int(row.get("away_goals", 0) or 0)
            result, points = _result_and_points(goals_for, goals_against)
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "team": team,
                    "opponent": row.get("away_team"),
                    "venue": "Casa",
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goals_for - goals_against,
                    "points": points,
                    "result": result,
                    "display_result": RESULT_LABELS[result],
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
        elif row.get("away_team") == team:
            goals_for = int(row.get("away_goals", 0) or 0)
            goals_against = int(row.get("home_goals", 0) or 0)
            result, points = _result_and_points(goals_for, goals_against)
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "team": team,
                    "opponent": row.get("home_team"),
                    "venue": "Trasferta",
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goals_for - goals_against,
                    "points": points,
                    "result": result,
                    "display_result": RESULT_LABELS[result],
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
        return pd.DataFrame()

    team_matches = pd.DataFrame(records)
    team_matches["match_date"] = pd.to_datetime(team_matches["match_date"], errors="coerce")
    team_matches = team_matches.sort_values("match_date").reset_index(drop=True)
    return team_matches


def _compute_league_baselines(team_logs: dict[str, pd.DataFrame]) -> dict[str, float | None]:
    rows: list[dict[str, float | None]] = []
    for team, team_df in team_logs.items():
        if team_df.empty:
            continue
        matches = len(team_df)
        rows.append(
            {
                "team": team,
                "points_per_match": round(float(team_df["points"].sum()) / matches, 2),
                "goals_for_avg": _safe_mean(team_df["goals_for"]),
                "goals_against_avg": _safe_mean(team_df["goals_against"]),
                "shots_for_avg": _safe_mean(team_df["shots_for"]),
                "shots_against_avg": _safe_mean(team_df["shots_against"]),
                "shots_on_target_for_avg": _safe_mean(team_df["shots_on_target_for"]),
                "shots_on_target_against_avg": _safe_mean(team_df["shots_on_target_against"]),
                "corners_for_avg": _safe_mean(team_df["corners_for"]),
                "cards_for_avg": _safe_mean(team_df["cards_for"]),
            }
        )

    if not rows:
        return {}

    baseline_df = pd.DataFrame(rows)
    baselines: dict[str, float | None] = {}
    for column in baseline_df.columns:
        if column == "team":
            continue
        baselines[column] = _safe_mean(baseline_df[column])
    return baselines


def _bucket_display_label(bucket_key: str, bucket_source: str) -> str:
    if bucket_source == "elo":
        if bucket_key == TOP_BUCKET_KEY:
            return "vs fascia alta (Elo)"
        if bucket_key == MIDDLE_BUCKET_KEY:
            return "vs fascia media (Elo)"
        return "vs fascia bassa (Elo)"

    if bucket_key == TOP_BUCKET_KEY:
        return TOP_BUCKET_LABEL
    if bucket_key == MIDDLE_BUCKET_KEY:
        return MIDDLE_BUCKET_LABEL
    return BOTTOM_BUCKET_LABEL


def _bucket_sentence_label(bucket_row: dict[str, Any]) -> str:
    label = str(bucket_row.get("bucket_label", "fascia avversaria"))
    return label[3:] if label.startswith("vs ") else label


def _build_profile_context(df: pd.DataFrame) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    standings = build_standings(prepared_df)
    enriched_standings = enrich_standings_with_ratings(standings)
    teams = get_teams(prepared_df)
    team_logs = {team: _build_team_matches(prepared_df, team) for team in teams}
    baselines = _compute_league_baselines(team_logs)
    bucket_map, bucket_teams, bucket_source = build_strength_bucket_map(enriched_standings)
    return {
        "prepared_df": prepared_df,
        "teams": teams,
        "standings": standings,
        "enriched_standings": enriched_standings,
        "team_logs": team_logs,
        "baselines": baselines,
        "bucket_map": bucket_map,
        "bucket_teams": bucket_teams,
        "bucket_source": bucket_source,
    }


def _build_notes(team_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    missing_columns = []
    for column in [
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "corners_for",
        "cards_for",
    ]:
        if column not in team_df.columns or team_df[column].dropna().empty:
            missing_columns.append(column)

    if missing_columns:
        notes.append(
            "Alcuni indicatori interni usano un set dati parziale: dove tiri, corner o cartellini "
            "mancano, il profilo si appoggia soprattutto a gol e punti."
        )
    return notes


def _merge_unique_items(primary: list[str], secondary: list[str], limit: int = 4) -> list[str]:
    merged: list[str] = []
    for item in primary + secondary:
        if item and item not in merged:
            merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def compute_offensive_profile(
    df: pd.DataFrame,
    team: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())
    baselines = context.get("baselines", {})

    if team_df.empty:
        return {
            "matches": 0,
            "goals_avg": 0.0,
            "shots_avg": None,
            "shots_on_target_avg": None,
            "efficienza_realizzativa": None,
            "indice_pericolosita_offensiva": 100.0,
        }

    shots_total = _safe_sum(team_df["shots_for"])
    goals_total = float(team_df["goals_for"].sum())
    goals_avg = _safe_mean(team_df["goals_for"]) or 0.0
    shots_avg = _safe_mean(team_df["shots_for"])
    shots_on_target_avg = _safe_mean(team_df["shots_on_target_for"])
    efficienza_realizzativa = _safe_ratio(goals_total, shots_total, multiplier=100.0)

    index = _weighted_index(
        [
            (_ratio_index(goals_avg, baselines.get("goals_for_avg")), 0.45),
            (_ratio_index(shots_on_target_avg, baselines.get("shots_on_target_for_avg")), 0.35),
            (_ratio_index(shots_avg, baselines.get("shots_for_avg")), 0.20),
        ]
    )

    return {
        "matches": int(len(team_df)),
        "goals_avg": round(goals_avg, 2),
        "shots_avg": shots_avg,
        "shots_on_target_avg": shots_on_target_avg,
        "efficienza_realizzativa": efficienza_realizzativa,
        "indice_pericolosita_offensiva": index,
    }


def compute_defensive_profile(
    df: pd.DataFrame,
    team: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())
    baselines = context.get("baselines", {})

    if team_df.empty:
        return {
            "matches": 0,
            "goals_against_avg": 0.0,
            "shots_conceded_avg": None,
            "shots_on_target_conceded_avg": None,
            "indice_solidita_difensiva": 100.0,
        }

    goals_against_avg = _safe_mean(team_df["goals_against"]) or 0.0
    shots_conceded_avg = _safe_mean(team_df["shots_against"])
    shots_on_target_conceded_avg = _safe_mean(team_df["shots_on_target_against"])

    index = _weighted_index(
        [
            (_ratio_index(goals_against_avg, baselines.get("goals_against_avg"), inverse=True), 0.45),
            (_ratio_index(shots_on_target_conceded_avg, baselines.get("shots_on_target_against_avg"), inverse=True), 0.35),
            (_ratio_index(shots_conceded_avg, baselines.get("shots_against_avg"), inverse=True), 0.20),
        ]
    )

    return {
        "matches": int(len(team_df)),
        "goals_against_avg": round(goals_against_avg, 2),
        "shots_conceded_avg": shots_conceded_avg,
        "shots_on_target_conceded_avg": shots_on_target_conceded_avg,
        "indice_solidita_difensiva": index,
    }


def compute_home_away_identity(
    df: pd.DataFrame,
    team: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())

    if team_df.empty:
        return {
            "points_home": 0,
            "points_away": 0,
            "ppm_home": 0.0,
            "ppm_away": 0.0,
            "indice_dipendenza_casa": 100.0,
            "note": "Dati insufficienti per valutare il peso del fattore casa.",
        }

    home_df = team_df[team_df["venue"] == "Casa"]
    away_df = team_df[team_df["venue"] == "Trasferta"]

    points_home = int(home_df["points"].sum()) if not home_df.empty else 0
    points_away = int(away_df["points"].sum()) if not away_df.empty else 0
    ppm_home = round(points_home / len(home_df), 2) if not home_df.empty else 0.0
    ppm_away = round(points_away / len(away_df), 2) if not away_df.empty else 0.0
    ppm_gap = round(ppm_home - ppm_away, 2)
    indice_dipendenza_casa = _clamp_index(100.0 + ppm_gap * 35.0, 50.0, 160.0)

    if ppm_gap >= 0.6:
        note = "Il rendimento interno pesa molto di piu di quello esterno."
    elif ppm_gap >= 0.25:
        note = "La squadra rende meglio in casa, pur restando competitiva fuori."
    elif ppm_gap <= -0.25:
        note = "Il rendimento esterno e pari o superiore a quello interno."
    else:
        note = "Profilo casa e trasferta abbastanza equilibrato."

    return {
        "points_home": points_home,
        "points_away": points_away,
        "ppm_home": ppm_home,
        "ppm_away": ppm_away,
        "ppm_gap": ppm_gap,
        "indice_dipendenza_casa": indice_dipendenza_casa,
        "note": note,
    }


def compute_recent_identity(
    df: pd.DataFrame,
    team: str,
    last_n: int = 5,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())

    if team_df.empty:
        return {
            "matches": 0,
            "last_n": last_n,
            "form_string": "-",
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "ppm": 0.0,
        }

    recent_df = team_df.tail(last_n)
    matches = len(recent_df)
    points = int(recent_df["points"].sum())
    goals_for = int(recent_df["goals_for"].sum())
    goals_against = int(recent_df["goals_against"].sum())
    goal_difference = goals_for - goals_against

    return {
        "matches": int(matches),
        "last_n": int(last_n),
        "form_string": " ".join(recent_df["display_result"].tolist()) or "-",
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goal_difference,
        "ppm": round(points / matches, 2) if matches else 0.0,
    }


def compute_vs_strength_buckets(
    df: pd.DataFrame,
    team: str,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    context = context or _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())
    bucket_map = context.get("bucket_map", {})

    if team_df.empty:
        return []

    profiled_df = team_df.copy()
    bucket_source = str(context.get("bucket_source", "classifica"))
    profiled_df["bucket_key"] = profiled_df["opponent"].map(bucket_map).fillna(MIDDLE_BUCKET_KEY)
    rows = []
    for bucket_key in [TOP_BUCKET_KEY, MIDDLE_BUCKET_KEY, BOTTOM_BUCKET_KEY]:
        bucket_df = profiled_df[profiled_df["bucket_key"] == bucket_key]
        matches = len(bucket_df)
        points = int(bucket_df["points"].sum()) if matches else 0
        goals_for = int(bucket_df["goals_for"].sum()) if matches else 0
        goals_against = int(bucket_df["goals_against"].sum()) if matches else 0
        rows.append(
            {
                "bucket_key": bucket_key,
                "bucket_label": _bucket_display_label(bucket_key, bucket_source),
                "matches": int(matches),
                "points": points,
                "ppm": round(points / matches, 2) if matches else 0.0,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goals_for - goals_against,
            }
        )
    return rows


def _compute_intensity_index(
    team_df: pd.DataFrame,
    baselines: dict[str, float | None],
) -> float:
    return _weighted_index(
        [
            (_ratio_index(_safe_mean(team_df["shots_for"]), baselines.get("shots_for_avg")), 0.45),
            (_ratio_index(_safe_mean(team_df["corners_for"]), baselines.get("corners_for_avg")), 0.30),
            (_ratio_index(_safe_mean(team_df["cards_for"]), baselines.get("cards_for_avg")), 0.25),
        ]
    )


def classify_team_archetypes(profile: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    offensive = profile.get("offensive", {})
    defensive = profile.get("defensive", {})
    home_away = profile.get("home_away", {})
    recent = profile.get("recent", {})
    versus = {row.get("bucket_key", row["bucket_label"]): row for row in profile.get("vs_strength_buckets", [])}
    general = profile.get("general", {})

    off_index = float(offensive.get("indice_pericolosita_offensiva", 100.0))
    def_index = float(defensive.get("indice_solidita_difensiva", 100.0))
    home_gap = float(home_away.get("ppm_gap", 0.0))
    overall_ppm = float(general.get("ppm", 0.0))

    if off_index >= 112:
        labels.append("offensiva")
    if def_index >= 112:
        labels.append("difensivamente solida")
    if def_index <= 90:
        labels.append("fragile difensivamente")
    if off_index <= 90:
        labels.append("bassa produzione offensiva")
    if home_gap >= 0.55:
        labels.append("dipendente dal fattore casa")
    if 95 <= off_index <= 108 and 95 <= def_index <= 108 and abs(home_gap) < 0.3:
        labels.append("squadra equilibrata")

    if recent.get("matches", 0) >= 3:
        recent_ppm = float(recent.get("ppm", 0.0))
        recent_gd = int(recent.get("goal_difference", 0))
        if recent_ppm >= 2.0 and recent_gd >= 2:
            labels.append("forma recente positiva")
        elif recent_ppm <= 0.8 and recent_gd <= -2:
            labels.append("forma recente negativa")

    top_bucket = versus.get(TOP_BUCKET_KEY)
    if top_bucket and top_bucket["matches"] >= 3 and top_bucket["ppm"] >= overall_ppm + 0.2:
        labels.append("efficace contro squadre forti")

    bottom_bucket = versus.get(BOTTOM_BUCKET_KEY)
    if bottom_bucket and bottom_bucket["matches"] >= 3 and bottom_bucket["ppm"] >= max(2.0, overall_ppm):
        labels.append("efficace contro squadre deboli")

    if not labels:
        labels.append("profilo in evoluzione")

    return labels


def build_strengths_and_weaknesses(profile: dict[str, Any]) -> dict[str, list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []

    offensive = profile.get("offensive", {})
    defensive = profile.get("defensive", {})
    home_away = profile.get("home_away", {})
    recent = profile.get("recent", {})
    general = profile.get("general", {})
    indicators = profile.get("indicators", {})
    advanced = profile.get("advanced_metrics", {})
    versus = {row.get("bucket_key", row["bucket_label"]): row for row in profile.get("vs_strength_buckets", [])}

    off_index = float(offensive.get("indice_pericolosita_offensiva", 100.0))
    def_index = float(defensive.get("indice_solidita_difensiva", 100.0))
    intensity_index = float(indicators.get("indice_intensita", 100.0))
    home_gap = float(home_away.get("ppm_gap", 0.0))
    overall_ppm = float(general.get("ppm", 0.0))

    if off_index >= 112:
        strengths.append("Costruisce un volume offensivo superiore alla media del campionato.")
    if defensive.get("goals_against_avg", 0.0) <= 1.0 or def_index >= 112:
        strengths.append("Mantiene una buona tenuta difensiva e concede poco con continuita.")
    if recent.get("matches", 0) >= 3 and recent.get("ppm", 0.0) >= 2.0:
        strengths.append("Arriva con forma recente positiva nelle ultime uscite.")
    if home_away.get("ppm_home", 0.0) >= 2.0:
        strengths.append("In casa riesce ad alzare rendimento e raccolta punti.")
    if intensity_index >= 110:
        strengths.append("Mostra intensita alta tra volume di gioco, corner e partecipazione alla gara.")
    top_bucket = versus.get(TOP_BUCKET_KEY)
    if top_bucket and top_bucket["matches"] >= 3 and top_bucket["ppm"] >= overall_ppm + 0.2:
        strengths.append("Sa restare competitiva anche contro avversarie di alta classifica.")

    if off_index <= 90:
        weaknesses.append("Produce meno pericoli offensivi della media del campionato.")
    if def_index <= 90:
        weaknesses.append("Concede troppo in area o nelle conclusioni pulite degli avversari.")
    if recent.get("matches", 0) >= 3 and recent.get("ppm", 0.0) <= 0.8:
        weaknesses.append("La forma recente e sotto tono e rallenta la crescita del profilo.")
    if home_gap >= 0.55 and home_away.get("ppm_away", 0.0) < 1.0:
        weaknesses.append("Fuori casa il rendimento cala sensibilmente rispetto alle gare interne.")
    bottom_bucket = versus.get(BOTTOM_BUCKET_KEY)
    if bottom_bucket and bottom_bucket["matches"] >= 3 and bottom_bucket["ppm"] < 1.5:
        weaknesses.append("Contro squadre della parte bassa non sta convertendo abbastanza il potenziale in punti.")
    if intensity_index <= 92:
        weaknesses.append("Il livello di intensita media resta contenuto e limita la pressione complessiva sulla gara.")

    if not strengths:
        strengths.append("Il profilo e abbastanza ordinato e senza squilibri estremi.")
    if not weaknesses:
        weaknesses.append("Non emergono criticita nette, ma il margine di miglioramento resta distribuito su piu aree.")

    if advanced:
        advanced_feedback = build_metric_strengths_and_weaknesses(advanced)
        strengths = _merge_unique_items(strengths, advanced_feedback["strengths"])
        weaknesses = _merge_unique_items(weaknesses, advanced_feedback["weaknesses"])

    return {
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:4],
    }


def build_team_profile_summary(profile: dict[str, Any]) -> str:
    if not profile.get("ok"):
        return "Dati insufficienti per costruire una sintesi affidabile del profilo squadra."

    general = profile["general"]
    offensive = profile["offensive"]
    defensive = profile["defensive"]
    home_away = profile["home_away"]
    recent = profile["recent"]
    rating = profile.get("rating", {})
    advanced = profile.get("advanced_metrics", {})
    versus = {row.get("bucket_key", row["bucket_label"]): row for row in profile["vs_strength_buckets"]}
    archetypes = ", ".join(profile["archetypes"][:4])

    lines = [
        (
            f"{profile['team']} occupa la posizione {general['position']} con {general['points']} punti "
            f"in {general['matches']} partite, con differenza reti {general['goal_difference']}."
        ),
        (
            f"In fase offensiva viaggia a {offensive['goals_avg']:.2f} gol di media "
            f"e l'indice di pericolosita offensiva si colloca a {offensive['indice_pericolosita_offensiva']:.1f}."
        ),
        (
            f"Senza palla concede {defensive['goals_against_avg']:.2f} gol a partita "
            f"e l'indice di solidita difensiva e pari a {defensive['indice_solidita_difensiva']:.1f}."
        ),
        (
            f"In casa raccoglie {home_away['ppm_home']:.2f} punti per gara, fuori {home_away['ppm_away']:.2f}: "
            f"{home_away['note'].lower()}"
        ),
        (
            f"Nelle ultime {recent['matches']} partite ha prodotto la sequenza {recent['form_string']} "
            f"e ha raccolto {recent['points']} punti con {recent['goals_for']} gol fatti e {recent['goals_against']} subiti."
        ),
    ]
    if rating.get("available"):
        lines.append(
            f"Il rating Elo attuale e {rating['rating_value']:.0f}, con fascia forza {rating['strength_band']} "
            f"tra le squadre coperte dal seed."
        )
    if advanced:
        lines.append(
            f"Nel layer Metriche Avanzate v1 emergono pericolosita offensiva a "
            f"{advanced.get('offensive_threat_index', 'n/d')}/100, solidita difensiva a "
            f"{advanced.get('defensive_solidity_index', 'n/d')}/100 e momento recente a "
            f"{advanced.get('recent_momentum_index', 'n/d')}/100."
        )

    top_bucket = versus.get(TOP_BUCKET_KEY)
    middle_bucket = versus.get(MIDDLE_BUCKET_KEY)
    bottom_bucket = versus.get(BOTTOM_BUCKET_KEY)
    if top_bucket:
        lines.append(
            f"Contro {_bucket_sentence_label(top_bucket)} sta viaggiando a {top_bucket['ppm']:.2f} punti per partita."
        )
    if middle_bucket:
        lines.append(
            f"Contro {_bucket_sentence_label(middle_bucket)} il rendimento e di {middle_bucket['ppm']:.2f} punti per partita."
        )
    if bottom_bucket:
        lines.append(
            f"Contro {_bucket_sentence_label(bottom_bucket)} il rendimento e di {bottom_bucket['ppm']:.2f} punti per gara."
        )

    lines.append(f"Le etichette che descrivono meglio il profilo oggi sono: {archetypes}.")
    lines.append(
        "Lettura prudente: il profilo fotografa tendenze aggregate della stagione e puo cambiare "
        "con campioni ridotti o con una sequenza di partite molto sbilanciata."
    )

    return "\n".join(lines[:11])


def build_team_profile(df: pd.DataFrame, team: str) -> dict[str, Any]:
    context = _build_profile_context(df)
    team_df = context["team_logs"].get(team, pd.DataFrame())

    if team_df.empty:
        return {
            "ok": False,
            "team": team,
            "message": "Nessun dato disponibile per la squadra selezionata.",
            "notes": [],
        }

    standings = context["enriched_standings"]
    standing_row = standings[standings["Team"] == team]
    if standing_row.empty:
        return {
            "ok": False,
            "team": team,
            "message": "La squadra selezionata non compare nella classifica corrente.",
            "notes": [],
        }

    row = standing_row.iloc[0]
    position = int(standing_row.index[0])
    general = {
        "position": position,
        "points": int(row["Pts"]),
        "matches": int(row["GP"]),
        "goals_for": int(row["GF"]),
        "goals_against": int(row["GA"]),
        "goal_difference": int(row["DR"]),
        "ppm": round(int(row["Pts"]) / max(int(row["GP"]), 1), 2),
    }

    offensive = compute_offensive_profile(df, team, context=context)
    defensive = compute_defensive_profile(df, team, context=context)
    home_away = compute_home_away_identity(df, team, context=context)
    recent = compute_recent_identity(df, team, last_n=5, context=context)
    vs_strength_buckets = compute_vs_strength_buckets(df, team, context=context)
    indicators = {
        "indice_pericolosita_offensiva": offensive["indice_pericolosita_offensiva"],
        "indice_solidita_difensiva": defensive["indice_solidita_difensiva"],
        "indice_intensita": _compute_intensity_index(team_df, context.get("baselines", {})),
        "indice_dipendenza_casa": home_away["indice_dipendenza_casa"],
    }
    rating_value = _safe_float(row.get("Elo"))
    rating_date = row.get("Rating Date")
    if isinstance(rating_date, pd.Timestamp):
        rating_date = rating_date.strftime("%Y-%m-%d")
    rating = {
        "available": rating_value is not None,
        "rating_type": "elo",
        "rating_value": rating_value,
        "rating_date": rating_date,
        "source_name": row.get("Rating Source"),
        "strength_band": row.get("Fascia forza") if pd.notna(row.get("Fascia forza")) else None,
        "rating_rank": int(row.get("Elo Rank")) if pd.notna(row.get("Elo Rank")) else None,
    }
    advanced_metrics = get_team_advanced_metrics(build_advanced_team_metrics(df), team) or {}

    profile = {
        "ok": True,
        "team": team,
        "general": general,
        "rating": rating,
        "advanced_metrics": advanced_metrics,
        "offensive": offensive,
        "defensive": defensive,
        "home_away": home_away,
        "recent": recent,
        "vs_strength_buckets": vs_strength_buckets,
        "indicators": indicators,
        "notes": _build_notes(team_df),
        "league_bucket_teams": context.get("bucket_teams", {}),
        "strength_bucket_source": context.get("bucket_source", "classifica"),
    }
    profile["archetypes"] = classify_team_archetypes(profile)
    strength_weakness = build_strengths_and_weaknesses(profile)
    profile.update(strength_weakness)
    profile["summary"] = build_team_profile_summary(profile)
    return profile
