from __future__ import annotations

from typing import Any


OUTCOMES = ("1", "X", "2")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    normalized = {key: max(_safe_float(probabilities.get(key)), 0.0) for key in OUTCOMES}
    total = sum(normalized.values())
    if total <= 0:
        return {"1": 1.0 / 3.0, "X": 1.0 / 3.0, "2": 1.0 / 3.0}
    return {key: normalized[key] / total for key in OUTCOMES}


def _favorite_from_probabilities(probabilities: dict[str, float]) -> str:
    return "1" if probabilities.get("1", 0.0) >= probabilities.get("2", 0.0) else "2"


def _favorite_from_edge(adjusted_edge: float, fallback_probabilities: dict[str, float]) -> str:
    if adjusted_edge > 0.8:
        return "1"
    if adjusted_edge < -0.8:
        return "2"
    return _favorite_from_probabilities(fallback_probabilities)


def _increase_outcome(
    probabilities: dict[str, float],
    target_key: str,
    amount: float,
    source_keys: list[str],
) -> float:
    if amount <= 0:
        return 0.0
    available_by_source = {key: max(probabilities.get(key, 0.0) - 0.04, 0.0) for key in source_keys}
    available = sum(available_by_source.values())
    if available <= 0:
        return 0.0

    actual_amount = min(amount, available)
    for key, available_value in available_by_source.items():
        probabilities[key] = max(probabilities[key] - actual_amount * (available_value / available), 0.0)
    probabilities[target_key] = probabilities.get(target_key, 0.0) + actual_amount
    return actual_amount


def _blend_toward_neutral(probabilities: dict[str, float], weight: float) -> dict[str, float]:
    weight = _clamp(weight, 0.0, 0.25)
    neutral = 1.0 / 3.0
    return {key: probabilities[key] * (1.0 - weight) + neutral * weight for key in OUTCOMES}


def classify_forecast_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "non disponibile"
    if confidence >= 65:
        return "alta"
    if confidence >= 45:
        return "media"
    return "bassa"


def adjust_probabilities_with_context(
    base_probabilities: dict[str, Any],
    context_engine: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[str]]:
    probabilities = _normalize_probabilities(base_probabilities)
    context_engine = context_engine or {}
    adjustments: list[str] = []

    if not context_engine:
        return probabilities, ["Contesto non disponibile: probabilita contestuali uguali alla baseline Poisson."]

    adjusted_edge = _safe_float(context_engine.get("adjusted_edge"))
    draw_risk = _clamp(_safe_float(context_engine.get("draw_risk"), 50.0), 0.0, 100.0)
    upset_risk = _clamp(_safe_float(context_engine.get("upset_risk"), 50.0), 0.0, 100.0)
    confidence = _clamp(_safe_float(context_engine.get("confidence"), 50.0), 0.0, 100.0)

    favorite = _favorite_from_edge(adjusted_edge, probabilities)
    underdog = "2" if favorite == "1" else "1"
    base_favorite = _favorite_from_probabilities(probabilities)

    if confidence < 45:
        blend = min(0.16, (45.0 - confidence) / 45.0 * 0.18)
        probabilities = _blend_toward_neutral(probabilities, blend)
        if blend > 0.01:
            adjustments.append("Confidenza bassa: probabilita rese meno estreme.")

    if draw_risk >= 55:
        amount = min(0.06, (draw_risk - 50.0) / 50.0 * 0.07)
        moved = _increase_outcome(probabilities, "X", amount, ["1", "2"])
        if moved > 0.005:
            adjustments.append(f"Draw risk alto ({draw_risk:.1f}/100): aumentato leggermente il peso del pareggio.")

    if upset_risk >= 55:
        amount = min(0.055, (upset_risk - 50.0) / 50.0 * 0.06)
        available = max(probabilities.get(favorite, 0.0) - 0.05, 0.0)
        moved = min(amount, available)
        if moved > 0.005:
            probabilities[favorite] -= moved
            probabilities[underdog] += moved * 0.65
            probabilities["X"] += moved * 0.35
            adjustments.append(f"Upset risk alto ({upset_risk:.1f}/100): ridotto con prudenza il favorito contestuale.")

    edge_strength = abs(adjusted_edge)
    if edge_strength >= 2.0 and confidence >= 45:
        edge_amount = min(0.045, ((edge_strength - 2.0) / 8.0) * 0.035 + max(confidence - 55.0, 0.0) / 100.0 * 0.025)
        moved = _increase_outcome(probabilities, favorite, edge_amount, [underdog, "X"])
        if moved > 0.005:
            if favorite == base_favorite:
                adjustments.append("Adjusted edge coerente con la baseline: favorito mantenuto leggermente piu forte.")
            else:
                adjustments.append("Adjusted edge diverso dalla baseline: probabilita spostate verso il favorito contestuale.")

    if confidence >= 70 and favorite == base_favorite and edge_strength < 2.0:
        adjustments.append("Confidenza alta ma edge non estremo: mantenuto il segnale base senza forzare la correzione.")

    contextual = _normalize_probabilities(probabilities)
    if not adjustments:
        adjustments.append("Contesto vicino alla baseline: nessuna correzione numerica rilevante.")
    return contextual, adjustments


def summarize_forecast_delta(
    base_probabilities: dict[str, Any],
    contextual_probabilities: dict[str, Any],
) -> list[dict[str, Any]]:
    base = _normalize_probabilities(base_probabilities)
    contextual = _normalize_probabilities(contextual_probabilities)
    labels = {"1": "Casa", "X": "Pareggio", "2": "Trasferta"}
    return [
        {
            "outcome": key,
            "label": labels[key],
            "base_probability": base[key],
            "contextual_probability": contextual[key],
            "delta": contextual[key] - base[key],
        }
        for key in OUTCOMES
    ]


def _factor_adjustment_lines(context_engine: dict[str, Any]) -> list[str]:
    weighted_factors = context_engine.get("weighted_factors", [])
    if not isinstance(weighted_factors, list):
        return []

    lines: list[str] = []
    for factor in weighted_factors[:4]:
        if not isinstance(factor, dict):
            continue
        label = str(factor.get("label") or factor.get("factor") or "Fattore")
        impact = _safe_float(factor.get("weighted_impact"))
        if abs(impact) < 0.02:
            direction = "neutro"
        else:
            direction = "casa" if impact > 0 else "trasferta"
        note = str(factor.get("note") or "").strip()
        if note:
            lines.append(f"{label}: segnale {direction} ({impact:+.2f}). {note}")
        else:
            lines.append(f"{label}: segnale {direction} ({impact:+.2f}).")
    return lines


def build_contextual_forecast_explanation(forecast: dict[str, Any]) -> str:
    confidence_label = forecast.get("confidence_label", "non disponibile")
    adjusted_edge = forecast.get("adjusted_edge")
    draw_risk = forecast.get("draw_risk")
    upset_risk = forecast.get("upset_risk")
    delta_rows = forecast.get("probability_deltas", [])
    strongest_delta = None
    if isinstance(delta_rows, list) and delta_rows:
        strongest_delta = max(delta_rows, key=lambda row: abs(float(row.get("delta", 0.0) or 0.0)))

    lines = [
        "La previsione contestuale parte dalle probabilita Poisson base e applica solo correzioni prudenti.",
        f"Il motore contestuale legge confidenza {confidence_label}, adjusted edge {adjusted_edge:.2f}, draw risk {draw_risk:.1f}/100 e upset risk {upset_risk:.1f}/100.",
    ]
    if strongest_delta:
        lines.append(
            f"Lo spostamento piu visibile riguarda {strongest_delta.get('label')}: "
            f"{float(strongest_delta.get('delta', 0.0)) * 100:+.1f} punti percentuali rispetto alla baseline."
        )
    lines.append("La lettura resta sperimentale: aiuta a interpretare matchup, forma, Elo e calendario, ma non sostituisce il modello base.")
    return " ".join(lines)


def build_contextual_warnings(matchup_analysis: dict[str, Any] | None) -> list[str]:
    if not matchup_analysis:
        return ["Matchup Analysis non disponibile: previsione contestuale uguale o molto vicina alla baseline."]
    warnings: list[str] = []
    if not matchup_analysis.get("ok"):
        warnings.append(str(matchup_analysis.get("message") or "Matchup Analysis non disponibile."))
    for warning in matchup_analysis.get("warnings", [])[:4]:
        warnings.append(str(warning))

    schedule_context = matchup_analysis.get("schedule_context", {})
    if isinstance(schedule_context, dict):
        audit = schedule_context.get("competition_audit", {})
        if isinstance(audit, dict) and audit.get("only_league_data", False):
            warnings.append("Contesto calendario parziale: nel database corrente risultano solo le partite disponibili di campionato.")

    unique: list[str] = []
    for item in warnings:
        if item and item not in unique:
            unique.append(item)
    return unique[:5]


def build_contextual_forecast(
    prediction: dict[str, Any],
    matchup_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_probabilities = _normalize_probabilities(prediction.get("probabilities", {}))
    context_engine = matchup_analysis.get("context_engine", {}) if isinstance(matchup_analysis, dict) else {}
    contextual_probabilities, adjustment_notes = adjust_probabilities_with_context(base_probabilities, context_engine)
    delta_rows = summarize_forecast_delta(base_probabilities, contextual_probabilities)

    adjusted_edge = _safe_float(context_engine.get("adjusted_edge"))
    draw_risk = _clamp(_safe_float(context_engine.get("draw_risk"), 50.0), 0.0, 100.0)
    upset_risk = _clamp(_safe_float(context_engine.get("upset_risk"), 50.0), 0.0, 100.0)
    confidence = _clamp(_safe_float(context_engine.get("confidence"), 50.0), 0.0, 100.0)
    confidence_label = classify_forecast_confidence(confidence)

    key_adjustments = adjustment_notes + _factor_adjustment_lines(context_engine)
    forecast = {
        "ok": bool(prediction.get("ok")),
        "base_probabilities": base_probabilities,
        "contextual_probabilities": contextual_probabilities,
        "probability_deltas": delta_rows,
        "base_most_likely_score": prediction.get("most_likely_score"),
        "adjusted_edge": adjusted_edge,
        "draw_risk": draw_risk,
        "upset_risk": upset_risk,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "key_adjustments": key_adjustments[:8],
        "warnings": build_contextual_warnings(matchup_analysis),
    }
    forecast["contextual_interpretation"] = build_contextual_forecast_explanation(forecast)
    return forecast
