from __future__ import annotations

from typing import Any

import pandas as pd


FACTOR_BASE_WEIGHTS = {
    "table": 1.18,
    "elo": 0.9,
    "predictor": 0.72,
    "recent_form": 0.98,
    "home_away": 1.12,
    "schedule": 0.64,
    "matchup": 1.16,
    "bucket_performance": 1.1,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _scaled_signal(value: float | None, divisor: float, cap: float) -> float:
    if value is None:
        return 0.0
    return round(_clamp(value / divisor, -cap, cap), 2)


def _team_count_from_profile(profile: dict[str, Any]) -> int:
    bucket_teams = profile.get("league_bucket_teams", {})
    all_teams = set()
    for teams in bucket_teams.values():
        all_teams.update(str(team_name) for team_name in teams)
    return len(all_teams) if all_teams else 20


def _infer_team_bucket(profile: dict[str, Any], opponent_team: str) -> str:
    bucket_teams = profile.get("league_bucket_teams", {})
    for bucket_key, teams in bucket_teams.items():
        if opponent_team in teams:
            return str(bucket_key)
    return "middle"


def _get_bucket_row(profile: dict[str, Any], bucket_key: str) -> dict[str, Any] | None:
    for row in profile.get("vs_strength_buckets", []):
        if row.get("bucket_key") == bucket_key:
            return row
    return None


def _factor_note(feature_name: str, factor_data: dict[str, Any], weighted_impact: float, home_team: str, away_team: str) -> str:
    direction = _sign(weighted_impact)
    leader = home_team if direction > 0 else away_team if direction < 0 else "nessuna squadra"

    if feature_name == "predictor":
        return f"Il predictor probabilistico spinge di piu verso {leader}." if direction else "Il predictor vede una gara molto aperta."
    if feature_name == "elo":
        return f"Il rating Elo premia maggiormente {leader}." if direction else "Il rating Elo non crea un margine netto."
    if feature_name == "recent_form":
        return f"Il momento recente pesa a favore di {leader}." if direction else "La forma recente non crea un vero margine."
    if feature_name == "home_away":
        return f"Il contesto casa/fuori favorisce {leader}." if direction else "Il fattore casa/fuori non produce un vantaggio forte."
    if feature_name == "schedule":
        if factor_data.get("only_league_data"):
            return "Il calendario entra come segnale prudente perche oggi copre solo le partite disponibili."
        return f"Riposo e carico partite favoriscono {leader}." if direction else "Riposo e carico partite non creano un margine netto."
    if feature_name == "matchup":
        return f"Il mismatch attacco vs difesa pende verso {leader}." if direction else "I mismatch principali restano distribuiti."
    if feature_name == "bucket_performance":
        return f"Il rendimento contro la fascia dell'avversaria aiuta di piu {leader}." if direction else "Il rendimento contro fasce simili non sposta molto."
    if feature_name == "table":
        return f"Classifica e ritmo punti premiano {leader}." if direction else "Classifica e punti non definiscono un vantaggio netto."
    if feature_name == "stakes":
        return "La pressione di classifica riduce leggermente il margine percepito."
    return factor_data.get("note", "Fattore contestuale considerato nel calcolo.")


def _factor_base_weight(feature_name: str) -> float:
    return float(FACTOR_BASE_WEIGHTS.get(feature_name, 1.0))


def _active_context_factors(weighted_factors: list[dict[str, Any]], threshold: float = 1.0) -> list[dict[str, Any]]:
    return [
        factor
        for factor in weighted_factors
        if factor.get("factor") != "stakes"
        and factor.get("available", True)
        and abs(float(factor.get("weighted_impact", 0.0) or 0.0)) >= threshold
    ]


def _compute_factor_coherence(weighted_factors: list[dict[str, Any]], dominant_sign: int) -> float:
    active = _active_context_factors(weighted_factors, threshold=1.0)
    if not active:
        return 0.35
    if dominant_sign == 0:
        return 0.45

    aligned = sum(1 for factor in active if _sign(float(factor.get("weighted_impact", 0.0) or 0.0)) == dominant_sign)
    conflicting = sum(1 for factor in active if _sign(float(factor.get("weighted_impact", 0.0) or 0.0)) == -dominant_sign)
    raw_score = (aligned - conflicting * 0.35) / max(len(active), 1)
    return round(_clamp(0.45 + raw_score * 0.55, 0.0, 1.0), 2)


def compute_feature_reliability(feature_name: str, factor_data: dict[str, Any]) -> float:
    if not factor_data.get("available", True):
        return 0.0

    min_matches = int(factor_data.get("min_matches", 0) or 0)
    recent_matches = int(factor_data.get("recent_matches", 0) or 0)
    bucket_matches = int(factor_data.get("bucket_matches", 0) or 0)

    if feature_name == "predictor":
        return 0.78 if factor_data.get("predictor_available") else 0.0
    if feature_name == "elo":
        if factor_data.get("both_available"):
            return 0.82
        return 0.40 if factor_data.get("one_available") else 0.0
    if feature_name == "recent_form":
        return round(_clamp(0.44 + (recent_matches / 5.0) * 0.40, 0.44, 0.86), 2)
    if feature_name == "home_away":
        return round(_clamp(0.60 + (min_matches / 18.0) * 0.28, 0.60, 0.92), 2)
    if feature_name == "schedule":
        if not factor_data.get("rest_available") and not factor_data.get("load_available"):
            return 0.0
        base = 0.42 if factor_data.get("only_league_data") else 0.56
        base += min_matches / 20.0 * 0.18
        if factor_data.get("rest_available"):
            base += 0.08
        if factor_data.get("load_available"):
            base += 0.06
        upper = 0.74 if factor_data.get("only_league_data") else 0.88
        return round(_clamp(base, 0.30, upper), 2)
    if feature_name == "matchup":
        metric_coverage = float(factor_data.get("metric_coverage", 0.0) or 0.0)
        return round(_clamp(0.52 + metric_coverage * 0.28 + (min_matches / 18.0) * 0.16, 0.46, 0.92), 2)
    if feature_name == "bucket_performance":
        return round(_clamp(0.38 + (bucket_matches / 5.0) * 0.42, 0.0, 0.88), 2)
    if feature_name == "table":
        return round(_clamp(0.62 + (min_matches / 18.0) * 0.28, 0.62, 0.93), 2)
    return round(_clamp(0.5 + (min_matches / 20.0) * 0.3, 0.4, 0.9), 2)


def compute_context_relevance(
    feature_name: str,
    factor_data: dict[str, Any],
    stakes_pressure_index: float | None = None,
) -> float:
    stakes_pressure_index = stakes_pressure_index or 50.0
    gap = abs(float(factor_data.get("gap", 0.0) or 0.0))
    recent_gap = abs(float(factor_data.get("recent_gap", 0.0) or 0.0))
    opponent_bucket = str(factor_data.get("opponent_bucket") or "middle")

    relevance = 0.5
    if feature_name == "predictor":
        relevance = 0.44 + min(gap / 24.0, 0.15)
        if float(factor_data.get("draw_probability", 0.0) or 0.0) > 0.30:
            relevance -= 0.05
    elif feature_name == "elo":
        relevance = 0.40
        if gap >= 100:
            relevance += 0.16
        elif gap >= 50:
            relevance += 0.08
        if factor_data.get("conflicts_with_recent"):
            relevance -= 0.15
    elif feature_name == "recent_form":
        relevance = 0.42 + (0.15 if gap >= 15 else 0.08 if gap >= 8 else 0.0)
        if not factor_data.get("predictor_available"):
            relevance += 0.08
    elif feature_name == "home_away":
        relevance = 0.48 + (0.18 if gap >= 0.50 else 0.08 if gap >= 0.25 else 0.0)
        if factor_data.get("home_dependency_high") or factor_data.get("away_dependency_high"):
            relevance += 0.10
    elif feature_name == "schedule":
        rest_gap = abs(float(factor_data.get("rest_gap", 0.0) or 0.0))
        load_gap = abs(float(factor_data.get("load_gap", 0.0) or 0.0))
        relevance = 0.34 + min(rest_gap / 10.0, 0.16) + min(load_gap * 0.08, 0.12)
        if factor_data.get("multi_competition_available"):
            relevance += 0.08
        if factor_data.get("only_league_data"):
            relevance -= 0.05
    elif feature_name == "matchup":
        relevance = 0.62 + (0.18 if gap >= 18 else 0.10 if gap >= 10 else 0.0)
        if int(factor_data.get("mismatch_count", 0) or 0) >= 2:
            relevance += 0.08
    elif feature_name == "bucket_performance":
        relevance = 0.52 + (0.14 if opponent_bucket in {"top", "bottom"} else 0.05)
        if gap >= 0.50:
            relevance += 0.10
    elif feature_name == "table":
        relevance = 0.48
        if int(factor_data.get("position_gap", 99) or 99) <= 3 or float(factor_data.get("points_gap", 99.0) or 99.0) <= 3.0:
            relevance += 0.12
        if stakes_pressure_index >= 70:
            relevance += 0.05

    return round(_clamp(relevance, 0.2, 1.2), 2)


def compute_matchup_multiplier(
    feature_name: str,
    factor_data: dict[str, Any],
    style_advantage: dict[str, Any] | None = None,
    predictor_context: dict[str, Any] | None = None,
) -> float:
    style_advantage = style_advantage or {}
    predictor_context = predictor_context or {}
    style_sign = _sign(float(style_advantage.get("score", 0.0) or 0.0))
    factor_sign = _sign(float(factor_data.get("signal", 0.0) or 0.0))

    multiplier = 1.0
    if feature_name == "predictor":
        if factor_sign and factor_sign == style_sign:
            multiplier = 1.04
        elif factor_sign and style_sign and factor_sign != style_sign:
            multiplier = 0.88
    elif feature_name == "elo":
        if factor_data.get("conflicts_with_recent"):
            multiplier = 0.82
        elif factor_sign and factor_sign == style_sign:
            multiplier = 1.02
    elif feature_name == "recent_form":
        if factor_sign and factor_sign == style_sign:
            multiplier = 1.06
        elif predictor_context.get("available") and factor_sign and factor_sign != _sign(float(predictor_context.get("home_probability", 0.0) - predictor_context.get("away_probability", 0.0))):
            multiplier = 0.95
    elif feature_name == "home_away":
        if float(factor_data.get("gap", 0.0) or 0.0) > 0 and factor_data.get("home_dependency_high"):
            multiplier = 1.14
        elif float(factor_data.get("gap", 0.0) or 0.0) < 0:
            multiplier = 1.06
        else:
            multiplier = 1.0
    elif feature_name == "schedule":
        signal_abs = abs(float(factor_data.get("signal", 0.0) or 0.0))
        if signal_abs >= 1.5 and factor_data.get("multi_competition_available"):
            multiplier = 1.08
        elif factor_data.get("only_league_data"):
            multiplier = 0.90
        else:
            multiplier = 1.0
    elif feature_name == "matchup":
        gap = abs(float(factor_data.get("gap", 0.0) or 0.0))
        multiplier = 1.18 if gap >= 20 else 1.10 if gap >= 10 else 1.0
    elif feature_name == "bucket_performance":
        multiplier = 1.12 if int(factor_data.get("bucket_matches", 0) or 0) >= 4 else 0.96
    elif feature_name == "table":
        if predictor_context.get("available") and factor_sign and factor_sign == _sign(float(predictor_context.get("home_probability", 0.0) - predictor_context.get("away_probability", 0.0))):
            multiplier = 1.08
        else:
            multiplier = 1.0

    return round(_clamp(multiplier, 0.8, 1.2), 2)


def compute_stakes_pressure_index(home_profile: dict[str, Any], away_profile: dict[str, Any]) -> float:
    home_general = home_profile.get("general", {})
    away_general = away_profile.get("general", {})
    team_count = max(_team_count_from_profile(home_profile), _team_count_from_profile(away_profile), 2)

    home_position = int(home_general.get("position", team_count) or team_count)
    away_position = int(away_general.get("position", team_count) or team_count)
    home_points = int(home_general.get("points", 0) or 0)
    away_points = int(away_general.get("points", 0) or 0)
    min_matches = min(int(home_general.get("matches", 0) or 0), int(away_general.get("matches", 0) or 0))

    pressure = 28.0
    position_gap = abs(home_position - away_position)
    points_gap = abs(home_points - away_points)
    bottom_cutoff = max(team_count - 5, 1)

    if home_position <= 6 and away_position <= 6:
        pressure += 14.0
    if home_position >= bottom_cutoff and away_position >= bottom_cutoff:
        pressure += 14.0
    if max(home_position, away_position) <= 8 and position_gap <= 3:
        pressure += 5.0
    if position_gap <= 3:
        pressure += 8.0
    elif position_gap <= 6:
        pressure += 4.0
    if points_gap <= 3:
        pressure += 8.0
    elif points_gap <= 6:
        pressure += 5.0
    if min_matches >= 30:
        pressure += 9.0
    elif min_matches >= 20:
        pressure += 6.0
    elif min_matches >= 10:
        pressure += 3.0

    return round(_clamp(pressure, 12.0, 88.0), 1)


def compute_uncertainty_index(
    weighted_factors: list[dict[str, Any]],
    predictor_context: dict[str, Any] | None = None,
    stakes_pressure_index: float = 50.0,
    adjusted_edge: float = 0.0,
    draw_risk: float = 35.0,
    coherence_score: float = 0.5,
) -> float:
    predictor_context = predictor_context or {}
    usable = [factor for factor in weighted_factors if factor.get("factor") != "stakes" and factor.get("available", True)]
    if usable:
        support_scores = [
            _clamp(
                float(factor.get("reliability", 0.0) or 0.0)
                * float(factor.get("context_relevance", 0.0) or 0.0)
                * float(factor.get("matchup_multiplier", 1.0) or 1.0)
                / 1.2,
                0.0,
                1.0,
            )
            for factor in usable
        ]
        average_support = sum(support_scores) / len(support_scores)
    else:
        average_support = 0.35

    signs = [_sign(float(factor.get("weighted_impact", 0.0) or 0.0)) for factor in usable if abs(float(factor.get("weighted_impact", 0.0) or 0.0)) >= 0.75]
    dominant_sign = _sign(adjusted_edge)
    conflict_count = sum(1 for sign in signs if dominant_sign and sign and sign != dominant_sign)
    conflict_component = min(conflict_count / max(len(signs), 1), 1.0)

    balance_component = _clamp(1.0 - min(abs(adjusted_edge) / 14.0, 1.0), 0.0, 1.0)
    draw_component = _clamp(draw_risk / 100.0, 0.0, 1.0)
    pressure_component = _clamp((stakes_pressure_index - 60.0) / 35.0, 0.0, 1.0)
    coherence_component = _clamp(1.0 - coherence_score, 0.0, 1.0)
    missing_predictor_component = 0.15 if not predictor_context.get("available") else 0.0

    uncertainty = (
        (1.0 - average_support) * 36.0
        + balance_component * 24.0
        + draw_component * 16.0
        + conflict_component * 12.0
        + coherence_component * 10.0
        + pressure_component * 4.0
        + missing_predictor_component * 8.0
    )
    return round(_clamp(uncertainty, 8.0, 90.0), 1)


def explain_context_adjustments(context_result: dict[str, Any]) -> str:
    home_team = str(context_result.get("home_team") or "La squadra di casa")
    away_team = str(context_result.get("away_team") or "La squadra ospite")
    base_edge = float(context_result.get("base_edge", 0.0) or 0.0)
    adjusted_edge = float(context_result.get("adjusted_edge", 0.0) or 0.0)
    draw_risk = float(context_result.get("draw_risk", 0.0) or 0.0)
    upset_risk = float(context_result.get("upset_risk", 0.0) or 0.0)
    confidence = float(context_result.get("confidence", 0.0) or 0.0)

    if adjusted_edge > 1.5:
        leader = home_team
    elif adjusted_edge < -1.5:
        leader = away_team
    else:
        leader = "nessuna delle due"

    if abs(adjusted_edge) < abs(base_edge) - 0.4:
        edge_note = "Il contesto riduce il margine base, quindi il dato grezzo va letto con piu prudenza."
    elif abs(adjusted_edge) > abs(base_edge) + 0.4:
        edge_note = "Il contesto rafforza il margine base, perche alcuni fattori di matchup e forma vanno nella stessa direzione."
    else:
        edge_note = "Il contesto conferma in buona parte il margine base senza stravolgerlo."

    top_factors = context_result.get("weighted_factors", [])[:3]
    if top_factors:
        factor_names = ", ".join(str(factor.get("label", "fattore")) for factor in top_factors)
        factor_note = f"I fattori che pesano di piu sono: {factor_names}."
    else:
        factor_note = "Non emergono fattori dominanti oltre al bilanciamento generale del match."

    if leader == "nessuna delle due":
        leader_note = "Il motore legge una partita ancora molto aperta."
    else:
        leader_note = f"Il vantaggio corretto oggi resta dalla parte di {leader}."

    return " ".join(
        [
            leader_note,
            edge_note,
            f"Rischio pareggio a {draw_risk:.1f}/100, rischio upset a {upset_risk:.1f}/100 e confidenza del quadro a {confidence:.1f}/100.",
            factor_note,
        ]
    )


def build_context_adjusted_edge(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor_context: dict[str, Any] | None = None,
    mismatches: list[str] | None = None,
    style_advantage: dict[str, Any] | None = None,
    schedule_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predictor_context = predictor_context or {}
    mismatches = mismatches or []
    style_advantage = style_advantage or {}
    schedule_context = schedule_context or {}

    home_team = str(home_profile.get("team") or "Casa")
    away_team = str(away_profile.get("team") or "Trasferta")
    home_general = home_profile.get("general", {})
    away_general = away_profile.get("general", {})
    home_recent = home_profile.get("recent", {})
    away_recent = away_profile.get("recent", {})
    home_home_away = home_profile.get("home_away", {})
    away_home_away = away_profile.get("home_away", {})
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    stakes_pressure = compute_stakes_pressure_index(home_profile, away_profile)

    home_matches = int(home_general.get("matches", 0) or 0)
    away_matches = int(away_general.get("matches", 0) or 0)
    min_matches = min(home_matches, away_matches)

    home_position = int(home_general.get("position", 0) or 0)
    away_position = int(away_general.get("position", 0) or 0)
    position_gap = abs(home_position - away_position)
    points_gap = abs(float(home_general.get("points", 0) or 0.0) - float(away_general.get("points", 0) or 0.0))

    home_points_per_match = float(home_general.get("ppm", 0.0) or 0.0)
    away_points_per_match = float(away_general.get("ppm", 0.0) or 0.0)
    home_table_value = (away_position - home_position) * 0.35 + (float(home_general.get("points", 0) or 0.0) - float(away_general.get("points", 0) or 0.0)) * 0.12 + (home_points_per_match - away_points_per_match) * 1.8

    home_elo = _safe_float(home_profile.get("rating", {}).get("rating_value"))
    away_elo = _safe_float(away_profile.get("rating", {}).get("rating_value"))
    elo_gap = None if home_elo is None or away_elo is None else home_elo - away_elo

    home_recent_ppm = float(home_recent.get("ppm", 0.0) or 0.0)
    away_recent_ppm = float(away_recent.get("ppm", 0.0) or 0.0)
    home_recent_momentum = _safe_float(home_advanced.get("recent_momentum_index"))
    away_recent_momentum = _safe_float(away_advanced.get("recent_momentum_index"))
    recent_gap = None if home_recent_momentum is None or away_recent_momentum is None else home_recent_momentum - away_recent_momentum
    recent_signal_raw = (home_recent_ppm - away_recent_ppm) * 2.2 + ((recent_gap or 0.0) / 18.0)

    home_context_gap = float(home_home_away.get("ppm_home", 0.0) or 0.0) - float(away_home_away.get("ppm_away", 0.0) or 0.0)
    home_dependency = _safe_float(home_advanced.get("home_dependency_index")) or 0.0
    away_dependency = _safe_float(away_advanced.get("home_dependency_index")) or 0.0

    schedule_audit = schedule_context.get("competition_audit", {}) if isinstance(schedule_context, dict) else {}
    home_schedule_load = schedule_context.get("home_schedule_load", {}) if isinstance(schedule_context, dict) else {}
    away_schedule_load = schedule_context.get("away_schedule_load", {}) if isinstance(schedule_context, dict) else {}
    home_all_form = schedule_context.get("home_recent_all_comp_form", {}) if isinstance(schedule_context, dict) else {}
    away_all_form = schedule_context.get("away_recent_all_comp_form", {}) if isinstance(schedule_context, dict) else {}
    rest_advantage = _safe_float(schedule_context.get("rest_advantage")) if isinstance(schedule_context, dict) else None
    home_load_score = _safe_float(home_schedule_load.get("load_score")) or 0.0
    away_load_score = _safe_float(away_schedule_load.get("load_score")) or 0.0
    load_gap = away_load_score - home_load_score
    home_all_ppm = _safe_float(home_all_form.get("ppm")) or 0.0
    away_all_ppm = _safe_float(away_all_form.get("ppm")) or 0.0
    all_comp_form_gap = home_all_ppm - away_all_ppm if schedule_audit.get("multi_competition_available") else 0.0
    schedule_signal_raw = 0.0
    if rest_advantage is not None:
        schedule_signal_raw += _clamp(rest_advantage, -7.0, 7.0) / 2.5
    schedule_signal_raw += load_gap * 0.75
    schedule_signal_raw += all_comp_form_gap * 0.35
    schedule_min_matches = min(
        int(home_all_form.get("matches", 0) or 0),
        int(away_all_form.get("matches", 0) or 0),
    )
    schedule_available = bool(schedule_context.get("schedule_factor_available")) and (
        rest_advantage is not None or bool(home_schedule_load) or bool(away_schedule_load)
    )
    if not schedule_available:
        schedule_signal_raw = 0.0

    home_attack = _safe_float(home_advanced.get("offensive_threat_index"))
    away_attack = _safe_float(away_advanced.get("offensive_threat_index"))
    home_defense = _safe_float(home_advanced.get("defensive_solidity_index"))
    away_defense = _safe_float(away_advanced.get("defensive_solidity_index"))
    home_attack_vs_away_defense = None if home_attack is None or away_defense is None else home_attack - away_defense
    away_attack_vs_home_defense = None if away_attack is None or home_defense is None else away_attack - home_defense
    matchup_gap = None
    if home_attack_vs_away_defense is not None and away_attack_vs_home_defense is not None:
        matchup_gap = home_attack_vs_away_defense - away_attack_vs_home_defense

    metric_inputs = [home_attack, away_attack, home_defense, away_defense]
    metric_coverage = len([value for value in metric_inputs if value is not None]) / 4.0

    away_bucket = _infer_team_bucket(home_profile, away_team)
    home_bucket = _infer_team_bucket(away_profile, home_team)
    home_bucket_row = _get_bucket_row(home_profile, away_bucket)
    away_bucket_row = _get_bucket_row(away_profile, home_bucket)
    home_bucket_ppm = None if not home_bucket_row else _safe_float(home_bucket_row.get("ppm"))
    away_bucket_ppm = None if not away_bucket_row else _safe_float(away_bucket_row.get("ppm"))
    bucket_matches = min(
        int(home_bucket_row.get("matches", 0) or 0) if home_bucket_row else 0,
        int(away_bucket_row.get("matches", 0) or 0) if away_bucket_row else 0,
    )
    bucket_gap = None if home_bucket_ppm is None or away_bucket_ppm is None else home_bucket_ppm - away_bucket_ppm

    predictor_available = bool(predictor_context.get("available"))
    predictor_gap = None
    if predictor_available:
        predictor_gap = (float(predictor_context.get("home_probability", 0.0) or 0.0) - float(predictor_context.get("away_probability", 0.0) or 0.0)) * 100.0

    factor_inputs = [
        {
            "factor": "table",
            "label": "Classifica e punti",
            "signal": _scaled_signal(home_table_value, divisor=1.0, cap=5.0),
            "available": home_position > 0 and away_position > 0,
            "min_matches": min_matches,
            "gap": home_table_value,
            "position_gap": position_gap,
            "points_gap": points_gap,
        },
        {
            "factor": "elo",
            "label": "Rating Elo",
            "signal": _scaled_signal(elo_gap, divisor=40.0, cap=5.0),
            "available": elo_gap is not None,
            "both_available": home_elo is not None and away_elo is not None,
            "one_available": (home_elo is not None) != (away_elo is not None),
            "min_matches": min_matches,
            "gap": abs(elo_gap) if elo_gap is not None else 0.0,
            "recent_gap": abs(recent_gap) if recent_gap is not None else 0.0,
            "conflicts_with_recent": elo_gap is not None and recent_gap is not None and _sign(elo_gap) != _sign(recent_gap) and abs(recent_gap) >= 12,
        },
        {
            "factor": "predictor",
            "label": "Predictor esistente",
            "signal": _scaled_signal(predictor_gap, divisor=5.5, cap=5.0),
            "available": predictor_available,
            "predictor_available": predictor_available,
            "gap": abs(predictor_gap) if predictor_gap is not None else 0.0,
            "draw_probability": float(predictor_context.get("draw_probability", 0.0) or 0.0),
        },
        {
            "factor": "recent_form",
            "label": "Momento recente",
            "signal": _scaled_signal(recent_signal_raw, divisor=1.0, cap=5.0),
            "available": home_recent.get("matches", 0) > 0 and away_recent.get("matches", 0) > 0,
            "recent_matches": min(int(home_recent.get("matches", 0) or 0), int(away_recent.get("matches", 0) or 0)),
            "predictor_available": predictor_available,
            "gap": abs(recent_gap) if recent_gap is not None else abs((home_recent_ppm - away_recent_ppm) * 10.0),
        },
        {
            "factor": "home_away",
            "label": "Contesto casa/fuori",
            "signal": _scaled_signal(home_context_gap, divisor=0.40, cap=5.0),
            "available": True,
            "min_matches": min_matches,
            "gap": home_context_gap,
            "home_dependency_high": home_dependency >= 65.0,
            "away_dependency_high": away_dependency >= 65.0,
        },
        {
            "factor": "schedule",
            "label": "Calendario / riposo",
            "signal": _scaled_signal(schedule_signal_raw, divisor=1.0, cap=3.0),
            "available": schedule_available,
            "min_matches": schedule_min_matches,
            "gap": abs(schedule_signal_raw),
            "rest_gap": abs(rest_advantage) if rest_advantage is not None else 0.0,
            "load_gap": abs(load_gap),
            "rest_available": rest_advantage is not None,
            "load_available": bool(home_schedule_load) and bool(away_schedule_load),
            "only_league_data": bool(schedule_audit.get("only_league_data", True)),
            "multi_competition_available": bool(schedule_audit.get("multi_competition_available", False)),
        },
        {
            "factor": "matchup",
            "label": "Mismatch attacco vs difesa",
            "signal": _scaled_signal(matchup_gap, divisor=4.25, cap=6.0),
            "available": matchup_gap is not None,
            "min_matches": min_matches,
            "gap": abs(matchup_gap) if matchup_gap is not None else 0.0,
            "metric_coverage": metric_coverage,
            "mismatch_count": len(mismatches),
        },
        {
            "factor": "bucket_performance",
            "label": "Rendimento vs fascia avversaria",
            "signal": _scaled_signal(bucket_gap, divisor=0.30, cap=4.0),
            "available": bucket_gap is not None and bucket_matches > 0,
            "bucket_matches": bucket_matches,
            "gap": abs(bucket_gap) if bucket_gap is not None else 0.0,
            "opponent_bucket": away_bucket,
        },
    ]

    weighted_factors: list[dict[str, Any]] = []
    base_edge = 0.0
    adjusted_edge_pre_stakes = 0.0

    for factor_data in factor_inputs:
        factor_name = str(factor_data["factor"])
        signal = float(factor_data.get("signal", 0.0) or 0.0)
        base_edge += signal

        factor_weight = _factor_base_weight(factor_name)
        reliability = compute_feature_reliability(factor_name, factor_data)
        relevance = compute_context_relevance(factor_name, factor_data, stakes_pressure_index=stakes_pressure)
        multiplier = compute_matchup_multiplier(
            factor_name,
            factor_data,
            style_advantage=style_advantage,
            predictor_context=predictor_context,
        )
        weighted_impact = round(signal * factor_weight * reliability * relevance * multiplier, 2)
        adjusted_edge_pre_stakes += weighted_impact

        weighted_factors.append(
            {
                "factor": factor_name,
                "label": factor_data["label"],
                "available": bool(factor_data.get("available", True)),
                "base_weight": round(factor_weight, 2),
                "signal": round(signal, 2),
                "reliability": round(reliability, 2),
                "context_relevance": round(relevance, 2),
                "matchup_multiplier": round(multiplier, 2),
                "weighted_impact": weighted_impact,
                "note": _factor_note(str(factor_data["factor"]), factor_data, weighted_impact, home_team, away_team),
            }
        )

    stakes_adjustment = 0.0
    if abs(adjusted_edge_pre_stakes) > 0.25:
        if stakes_pressure >= 80 and abs(adjusted_edge_pre_stakes) < 7.0:
            stakes_adjustment = round(-_sign(adjusted_edge_pre_stakes) * min(1.0, (stakes_pressure - 76.0) / 14.0), 2)
        elif stakes_pressure <= 28 and abs(adjusted_edge_pre_stakes) >= 10.0:
            stakes_adjustment = round(_sign(adjusted_edge_pre_stakes) * 0.3, 2)

    adjusted_edge = round(adjusted_edge_pre_stakes + stakes_adjustment, 2)

    if stakes_adjustment != 0.0 or stakes_pressure >= 70.0:
        weighted_factors.append(
            {
                "factor": "stakes",
                "label": "Pressione / stakes",
                "available": True,
                "base_weight": 0.4,
                "signal": 0.0,
                "reliability": 0.55,
                "context_relevance": round(_clamp(stakes_pressure / 120.0, 0.15, 0.7), 2),
                "matchup_multiplier": 1.0,
                "weighted_impact": stakes_adjustment,
                "note": _factor_note("stakes", {"note": ""}, stakes_adjustment, home_team, away_team),
            }
        )

    dominant_sign = _sign(adjusted_edge if abs(adjusted_edge) >= 0.35 else base_edge)
    coherence_score = _compute_factor_coherence(weighted_factors, dominant_sign)

    draw_risk = 34.0
    if predictor_available:
        draw_risk = float(predictor_context.get("draw_probability", 0.0) or 0.0) * 100.0
    if abs(adjusted_edge) < 3.0:
        draw_risk += 14.0
    elif abs(adjusted_edge) < 6.0:
        draw_risk += 8.0
    elif abs(adjusted_edge) < 10.0:
        draw_risk += 4.0
    if (home_defense or 0.0) >= 58.0 and (away_defense or 0.0) >= 58.0:
        draw_risk += 5.0
    if (_safe_float(home_advanced.get("finishing_efficiency_index")) or 50.0) < 45.0 and (
        (_safe_float(away_advanced.get("finishing_efficiency_index")) or 50.0) < 45.0
    ):
        draw_risk += 5.0
    if stakes_pressure >= 70.0 and abs(adjusted_edge) < 8.0:
        draw_risk += 6.0
    if abs(home_recent_ppm - away_recent_ppm) < 0.25 and abs(recent_gap or 0.0) < 8.0:
        draw_risk += 2.0
    if coherence_score < 0.42 and abs(adjusted_edge) < 6.0:
        draw_risk += 3.0
    elif coherence_score >= 0.72 and abs(adjusted_edge) >= 8.0:
        draw_risk -= 2.0
    if abs(adjusted_edge) > 12.0:
        draw_risk -= 8.0
    draw_risk = round(_clamp(draw_risk, 8.0, 82.0), 1)

    preliminary_uncertainty = compute_uncertainty_index(
        weighted_factors,
        predictor_context=predictor_context,
        stakes_pressure_index=stakes_pressure,
        adjusted_edge=adjusted_edge,
        draw_risk=draw_risk,
        coherence_score=coherence_score,
    )
    provisional_confidence = round(_clamp(100.0 - preliminary_uncertainty, 8.0, 92.0), 1)

    favorite_sign = _sign(adjusted_edge) if abs(adjusted_edge) >= 2.0 else 0
    upset_risk = 18.0
    if favorite_sign == 0:
        upset_risk = 38.0 + max(draw_risk - 45.0, 0.0) * 0.12
    else:
        favorite_advanced = home_advanced if favorite_sign > 0 else away_advanced
        underdog_advanced = away_advanced if favorite_sign > 0 else home_advanced
        favorite_probability = (
            float(predictor_context.get("home_probability", 0.0) or 0.0)
            if favorite_sign > 0
            else float(predictor_context.get("away_probability", 0.0) or 0.0)
        )
        underdog_recent_gap = (
            ((_safe_float(away_advanced.get("recent_momentum_index")) or 0.0) - (_safe_float(home_advanced.get("recent_momentum_index")) or 0.0))
            if favorite_sign > 0
            else ((_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) - (_safe_float(away_advanced.get("recent_momentum_index")) or 0.0))
        )
        underdog_matchup_edge = away_attack_vs_home_defense if favorite_sign > 0 else home_attack_vs_away_defense
        favorite_defensive_risk = _safe_float(favorite_advanced.get("defensive_risk_index")) or 50.0
        favorite_defensive_solidity = _safe_float(favorite_advanced.get("defensive_solidity_index")) or 50.0
        favorite_dependency = home_dependency if favorite_sign > 0 else away_dependency
        edge_abs = abs(adjusted_edge)
        active_factors = _active_context_factors(weighted_factors, threshold=1.0)
        aligned_advantages = sum(
            1 for factor in active_factors if _sign(float(factor.get("weighted_impact", 0.0) or 0.0)) == favorite_sign
        )
        conflicting_advantages = sum(
            1 for factor in active_factors if _sign(float(factor.get("weighted_impact", 0.0) or 0.0)) == -favorite_sign
        )

        if edge_abs < 4.0:
            upset_risk += 16.0
        elif edge_abs < 7.0:
            upset_risk += 11.0
        elif edge_abs < 10.0:
            upset_risk += 6.0
        elif edge_abs < 14.0:
            upset_risk += 2.0
        else:
            upset_risk -= 4.0

        if provisional_confidence < 35.0:
            upset_risk += 14.0
        elif provisional_confidence < 50.0:
            upset_risk += 9.0
        elif provisional_confidence < 65.0:
            upset_risk += 5.0
        elif provisional_confidence >= 78.0:
            upset_risk -= 6.0

        if underdog_recent_gap > 15.0:
            upset_risk += 10.0
        elif underdog_recent_gap > 8.0:
            upset_risk += 6.0
        elif underdog_recent_gap < -10.0:
            upset_risk -= 4.0

        if underdog_matchup_edge is not None:
            if underdog_matchup_edge > 15.0:
                upset_risk += 10.0
            elif underdog_matchup_edge > 8.0:
                upset_risk += 6.0
            elif underdog_matchup_edge < -8.0:
                upset_risk -= 3.0

        if favorite_defensive_risk >= 62.0:
            upset_risk += 8.0
        elif favorite_defensive_risk >= 55.0:
            upset_risk += 4.0

        if favorite_defensive_solidity <= 42.0:
            upset_risk += 8.0
        elif favorite_defensive_solidity <= 48.0:
            upset_risk += 4.0

        if favorite_dependency >= 72.0 and edge_abs < 10.0:
            upset_risk += 4.0

        if conflicting_advantages >= 2:
            upset_risk += 5.0
        if aligned_advantages >= 3 and conflicting_advantages == 0 and edge_abs >= 8.0 and coherence_score >= 0.72:
            upset_risk -= 10.0
        elif aligned_advantages >= 2 and coherence_score >= 0.60:
            upset_risk -= 5.0

        if predictor_available:
            if favorite_probability < 0.48:
                upset_risk += 4.0
            elif favorite_probability > 0.58:
                upset_risk -= 3.0

        if draw_risk >= 60.0 and edge_abs < 8.0:
            upset_risk += 3.0

        upset_risk += preliminary_uncertainty * 0.10

    upset_risk = round(_clamp(upset_risk, 10.0, 85.0), 1)
    uncertainty_index = round(
        _clamp(
            preliminary_uncertainty
            + max(draw_risk - 55.0, 0.0) * 0.06
            + max(upset_risk - 55.0, 0.0) * 0.08
            - coherence_score * 3.0,
            8.0,
            92.0,
        ),
        1,
    )
    confidence = round(_clamp(100.0 - uncertainty_index, 8.0, 92.0), 1)

    sorted_weighted_factors = sorted(
        weighted_factors,
        key=lambda factor: abs(float(factor.get("weighted_impact", 0.0) or 0.0)),
        reverse=True,
    )

    context_result = {
        "home_team": home_team,
        "away_team": away_team,
        "base_edge": round(base_edge, 2),
        "adjusted_edge": adjusted_edge,
        "draw_risk": draw_risk,
        "upset_risk": upset_risk,
        "confidence": confidence,
        "weighted_factors": sorted_weighted_factors,
        "stakes_pressure_index": round(stakes_pressure, 1),
        "uncertainty_index": uncertainty_index,
        "coherence_score": round(coherence_score * 100.0, 1),
    }
    context_result["textual_explanation"] = explain_context_adjustments(context_result)
    return context_result
