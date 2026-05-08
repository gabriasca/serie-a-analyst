from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics import get_teams, prepare_matches_dataframe
from src.forecast_context import build_contextual_forecast
from src.matchup_analysis import build_matchup_analysis
from src.reporting import build_match_report_data
from src.team_identity import build_team_identity_report


OUTCOMES = ("1", "X", "2")
OUTCOME_LABELS = {"1": "casa", "X": "pareggio", "2": "trasferta"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100:.1f}%"


def _normalize_probabilities(probabilities: dict[str, Any] | None) -> dict[str, float]:
    probabilities = probabilities or {}
    normalized = {key: max(_safe_float(probabilities.get(key)), 0.0) for key in OUTCOMES}
    total = sum(normalized.values())
    if total <= 0:
        return {key: 0.0 for key in OUTCOMES}
    return {key: normalized[key] / total for key in OUTCOMES}


def _favorite_key(probabilities: dict[str, Any] | None) -> str | None:
    normalized = _normalize_probabilities(probabilities)
    if sum(normalized.values()) <= 0:
        return None
    return max(OUTCOMES, key=lambda key: normalized.get(key, 0.0))


def _team_for_outcome(outcome: str | None, home_team: str, away_team: str) -> str:
    if outcome == "1":
        return home_team
    if outcome == "2":
        return away_team
    if outcome == "X":
        return "pareggio"
    return "nessun favorito netto"


def _probability_strength(probability: float) -> str:
    if probability >= 0.60:
        return "un vantaggio chiaro"
    if probability >= 0.50:
        return "un leggero vantaggio"
    if probability >= 0.40:
        return "una partita aperta con una tendenza"
    return "una partita equilibrata"


def _risk_label(value: Any) -> str:
    score = _safe_float(value, 50.0)
    if score >= 60:
        return "alto"
    if score >= 45:
        return "medio"
    return "basso"


def _confidence_label(value: Any) -> str:
    score = _safe_float(value, 50.0)
    if score >= 70:
        return "alta"
    if score < 45:
        return "bassa"
    return "media"


def _dedupe(items: list[Any], limit: int | None = None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_type(metrics: dict[str, Any]) -> str:
    offense = _metric_value(metrics, "offensive_threat_index")
    solidity = _metric_value(metrics, "defensive_solidity_index")
    risk = _metric_value(metrics, "defensive_risk_index")
    if offense is not None and offense >= 60 and (solidity is None or solidity < 58):
        return "piu offensiva"
    if solidity is not None and solidity >= 60 and (offense is None or offense < 58):
        return "piu solida"
    if risk is not None and risk >= 60:
        return "piu vulnerabile senza palla"
    return "equilibrata"


def _volatility_label(identity_report: dict[str, Any]) -> str:
    volatility = identity_report.get("volatility", {}) if isinstance(identity_report, dict) else {}
    return str(volatility.get("label") or "non disponibile")


def _safe_layer(name: str, fn: Any, warnings: list[str]) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive path for Streamlit Cloud/runtime oddities
        warnings.append(f"{name} non disponibile in questo ambiente: {exc}")
        return {}


def build_contextual_prediction_reading(
    prediction: dict[str, Any],
    contextual: dict[str, Any],
    home_team: str,
    away_team: str,
) -> str:
    base_probs = _normalize_probabilities(prediction.get("probabilities", {}))
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    base_favorite = _team_for_outcome(_favorite_key(base_probs), home_team, away_team)
    contextual_key = _favorite_key(contextual_probs)
    contextual_favorite = _team_for_outcome(contextual_key, home_team, away_team)
    contextual_probability = contextual_probs.get(contextual_key or "", 0.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)

    if not prediction.get("ok"):
        return (
            "Il predictor base non e disponibile in modo completo: la lettura numerica resta parziale "
            "e il report si appoggia soprattutto a forma, classifica, profili e matchup."
        )

    base_line = (
        f"Il predictor base parte da {base_favorite}: probabilita "
        f"{_format_pct(base_probs.get('1'))} / {_format_pct(base_probs.get('X'))} / {_format_pct(base_probs.get('2'))}."
    )
    contextual_line = (
        f"La lettura contestuale indica {contextual_favorite} con {_probability_strength(contextual_probability)} "
        f"e probabilita {_format_pct(contextual_probs.get('1'))} / {_format_pct(contextual_probs.get('X'))} / {_format_pct(contextual_probs.get('2'))}."
    )
    risk_line = (
        f"Draw risk {_risk_label(draw_risk)} ({draw_risk:.1f}/100), upset risk {_risk_label(upset_risk)} "
        f"({upset_risk:.1f}/100), confidence {_confidence_label(confidence)} ({confidence:.1f}/100)."
    )
    return " ".join([base_line, contextual_line, risk_line])


def build_match_opening_reading(report: dict[str, Any]) -> str:
    home_team = str(report.get("home_team") or "Casa")
    away_team = str(report.get("away_team") or "Trasferta")
    contextual = report.get("contextual_forecast", {}) or {}
    prediction = report.get("prediction", {}) or {}
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    favorite_key = _favorite_key(contextual_probs)
    favorite = _team_for_outcome(favorite_key, home_team, away_team)
    favorite_prob = contextual_probs.get(favorite_key or "", 0.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    table_context = report.get("match_report", {}).get("table_context", {})
    home_position = table_context.get("home_position")
    away_position = table_context.get("away_position")

    lines = [
        f"{home_team} - {away_team} viene letta dai dati come {_probability_strength(favorite_prob)} per {favorite}.",
        f"La confidenza complessiva e {_confidence_label(confidence)}: {confidence:.1f}/100.",
        f"Il rischio pareggio e {_risk_label(draw_risk)}, mentre il rischio upset e {_risk_label(upset_risk)}.",
    ]
    if home_position and away_position:
        lines.append(f"In classifica il confronto parte da {home_team} in posizione {home_position} e {away_team} in posizione {away_position}.")
    if prediction.get("ok"):
        lines.append(
            f"I gol attesi interni del modello Poisson sono {prediction.get('expected_goals_home', 0):.2f} "
            f"per {home_team} e {prediction.get('expected_goals_away', 0):.2f} per {away_team}."
        )
    lines.append("La lettura resta prudente: non include formazioni ufficiali, assenze, squalifiche o dati tattici granulari.")
    return "\n".join(lines[:6])


def build_probable_match_script(report: dict[str, Any]) -> str:
    home_team = str(report.get("home_team") or "Casa")
    away_team = str(report.get("away_team") or "Trasferta")
    matchup = report.get("matchup_analysis", {}) or {}
    contextual = report.get("contextual_forecast", {}) or {}
    prediction = report.get("prediction", {}) or {}
    style = matchup.get("style_advantage", {}) if isinstance(matchup, dict) else {}
    mismatches = matchup.get("mismatches", []) if isinstance(matchup, dict) else []
    schedule = report.get("match_report", {}).get("schedule_context", {}) or {}
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    favorite = _team_for_outcome(_favorite_key(contextual_probs), home_team, away_team)

    lines = [
        f"La trama piu probabile parte da {favorite}, ma non va letta come una certezza.",
    ]
    if style.get("label"):
        lines.append(f"Il vantaggio stilistico stimato e: {style.get('label')}. {style.get('explanation', '')}")
    if mismatches:
        lines.append(f"Il primo punto di attenzione e questo: {mismatches[0]}")
    if len(mismatches) > 1:
        lines.append(f"Un secondo snodo riguarda: {mismatches[1]}")
    if prediction.get("ok"):
        goals_gap = _safe_float(prediction.get("expected_goals_home")) - _safe_float(prediction.get("expected_goals_away"))
        if abs(goals_gap) < 0.25:
            lines.append("Il modello interno non separa molto la produzione attesa: la partita puo restare aperta a lungo.")
        elif goals_gap > 0:
            lines.append(f"La produzione stimata dal modello e piu alta per {home_team}, quindi il fattore campo entra nella lettura.")
        else:
            lines.append(f"La produzione stimata dal modello e piu alta per {away_team}, segnale che riduce il peso del fattore campo.")
    if isinstance(schedule, dict) and schedule.get("available"):
        lines.append(schedule.get("summary", "Il calendario viene letto solo sulle partite disponibili nel database."))
    lines.append(
        "Non abbiamo dati tattici granulari per dire se una squadra pressa alta, costruisce dal basso o cerca gioco diretto: "
        "possiamo leggere solo produzione, solidita, forma e matchup aggregati."
    )
    return "\n".join(_dedupe(lines, limit=9))


def build_alternative_match_script(report: dict[str, Any]) -> str:
    contextual = report.get("contextual_forecast", {}) or {}
    matchup = report.get("matchup_analysis", {}) or {}
    home_team = str(report.get("home_team") or "Casa")
    away_team = str(report.get("away_team") or "Trasferta")
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    favorite_key = _favorite_key(contextual_probs)
    favorite = _team_for_outcome(favorite_key, home_team, away_team)
    underdog = away_team if favorite_key == "1" else home_team if favorite_key == "2" else "una delle due squadre"
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    mismatches = matchup.get("mismatches", []) if isinstance(matchup, dict) else []

    lines: list[str] = []
    if draw_risk >= 45:
        lines.append("Lo scenario alternativo e una partita piu bloccata, in cui il pareggio pesa piu del vantaggio iniziale.")
    if upset_risk >= 45 and favorite_key in {"1", "2"}:
        lines.append(f"Un'altra via e che {favorite} non trasformi il vantaggio nei dati in controllo reale, lasciando spazio a {underdog}.")
    if confidence < 45:
        lines.append("La confidence bassa rende credibile uno scenario diverso: i segnali disponibili sono meno coerenti tra loro.")
    if len(mismatches) > 1:
        lines.append(f"Il cambio di lettura puo passare da questo mismatch secondario: {mismatches[1]}")
    lines.append("Lineup, assenze, squalifiche ed eventuale turnover potrebbero cambiare il peso dei dati, ma oggi non sono nel database.")
    return "\n".join(_dedupe(lines, limit=5))


def build_team_identity_interaction(report: dict[str, Any]) -> str:
    home_team = str(report.get("home_team") or "Casa")
    away_team = str(report.get("away_team") or "Trasferta")
    matchup = report.get("matchup_analysis", {}) or {}
    home_metrics = matchup.get("home_metrics", {}) if isinstance(matchup, dict) else {}
    away_metrics = matchup.get("away_metrics", {}) if isinstance(matchup, dict) else {}
    home_identity = report.get("home_identity", {}) or {}
    away_identity = report.get("away_identity", {}) or {}
    comparison = matchup.get("metric_comparison", []) if isinstance(matchup, dict) else []

    home_type = _profile_type(home_metrics)
    away_type = _profile_type(away_metrics)
    home_volatility = _volatility_label(home_identity)
    away_volatility = _volatility_label(away_identity)

    lines = [
        f"{home_team} oggi appare come squadra {home_type}; {away_team} come squadra {away_type}.",
        f"Sulla stabilita, {home_team} ha profilo '{home_volatility}', mentre {away_team} ha profilo '{away_volatility}'.",
    ]
    notable = [
        row for row in comparison
        if isinstance(row, dict) and row.get("leader") and row.get("edge") not in {None, "simile", "n/d"}
    ]
    if notable:
        first = notable[0]
        lines.append(f"Il segnale piu leggibile tra le metriche avanzate e {first.get('label')}: {first.get('reading')}")
    if len(notable) > 1:
        second = notable[1]
        lines.append(f"Un secondo segnale utile e {second.get('label')}: {second.get('reading')}")
    lines.append(
        "Questa interazione descrive compatibilita statistiche, non scelte tattiche certe: senza dati evento non possiamo affermare moduli, pressing o costruzione."
    )
    return "\n".join(_dedupe(lines, limit=6))


def build_reliability_assessment(report: dict[str, Any]) -> dict[str, list[str]]:
    contextual = report.get("contextual_forecast", {}) or {}
    prediction = report.get("prediction", {}) or {}
    matchup = report.get("matchup_analysis", {}) or {}
    match_report = report.get("match_report", {}) or {}
    ratings = match_report.get("ratings", {}) if isinstance(match_report, dict) else {}
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    base_favorite = _favorite_key(prediction.get("probabilities", {}))
    contextual_favorite = _favorite_key((contextual.get("contextual_probabilities", {}) or {}))

    reliable: list[str] = []
    fragile: list[str] = []
    if prediction.get("ok") and base_favorite == contextual_favorite:
        reliable.append("Predictor base e lettura contestuale v2 sono allineati sul lato principale.")
    if confidence >= 70:
        reliable.append("Confidence alta: i fattori principali sono abbastanza coerenti.")
    elif confidence < 45:
        fragile.append("Confidence bassa: i segnali interni sono contrastanti.")
    if ratings.get("home") and ratings.get("away"):
        reliable.append("Rating Elo disponibile per entrambe le squadre come indicatore storico/recente.")
    if matchup.get("context_engine", {}).get("weighted_factors"):
        reliable.append("Context engine disponibile: edge, draw risk e upset risk sono spiegabili tramite fattori pesati.")
    if draw_risk >= 60:
        fragile.append("Draw risk alto: il pareggio puo disturbare la lettura del favorito.")
    if upset_risk >= 60:
        fragile.append("Upset risk alto: il favorito non e protetto da segnali coerenti.")
    fragile.extend(
        [
            "Lineup, assenze e squalifiche non sono disponibili.",
            "Dati tattici granulari mancanti: pressing, possesso, passaggi, lanci e shot-by-shot non sono nel database.",
            "La lettura contestuale v2 e sperimentale e non sostituisce il predictor base.",
        ]
    )
    schedule = match_report.get("schedule_context", {}) if isinstance(match_report, dict) else {}
    if isinstance(schedule, dict):
        audit = schedule.get("competition_audit", {})
        if isinstance(audit, dict) and audit.get("only_league_data"):
            fragile.append("Calendario parziale: il carico recente non include coppe se non sono state importate.")

    if not reliable:
        reliable.append("Classifica, gol, forma recente e rendimento casa/fuori restano la base osservata piu solida.")
    return {"reliable": _dedupe(reliable, limit=6), "fragile": _dedupe(fragile, limit=6)}


def build_data_gaps_section(report: dict[str, Any]) -> list[str]:
    return [
        "Formazioni ufficiali e modulo previsto.",
        "Assenze, squalifiche, condizioni fisiche e disponibilita dei giocatori chiave.",
        "Eventuale turnover e gestione dei minuti.",
        "Dati su pressing, possesso, costruzione dal basso, passaggi progressivi e lanci lunghi.",
        "Eventi shot-by-shot e contesto delle conclusioni.",
        "Stato motivazionale o pressione della gara: oggi sarebbe solo un proxy, non un dato certo.",
    ]


def build_what_could_change_match(report: dict[str, Any]) -> list[str]:
    contextual = report.get("contextual_forecast", {}) or {}
    matchup = report.get("matchup_analysis", {}) or {}
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    items: list[str] = []
    if confidence < 55:
        items.append("Segnali poco netti: la partita puo cambiare volto piu facilmente rispetto a match con confidence alta.")
    if draw_risk >= 45:
        items.append("Se il ritmo resta basso, il peso del pareggio puo crescere.")
    if upset_risk >= 45:
        items.append("Se il favorito non concretizza presto il proprio vantaggio statistico, aumenta lo spazio per lo scenario alternativo.")
    mismatches = matchup.get("mismatches", []) if isinstance(matchup, dict) else []
    if mismatches:
        items.append(f"Il matchup piu sensibile da monitorare e: {mismatches[0]}")
    items.extend(
        [
            "Lineup e assenze possono cambiare il significato dei dati pre-partita.",
            "Episodi, gestione dei primi minuti e cartellini non sono prevedibili con i dati aggregati.",
        ]
    )
    return _dedupe(items, limit=6)


def build_final_analyst_take(report: dict[str, Any]) -> str:
    home_team = str(report.get("home_team") or "Casa")
    away_team = str(report.get("away_team") or "Trasferta")
    contextual = report.get("contextual_forecast", {}) or {}
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    favorite_key = _favorite_key(contextual_probs)
    favorite = _team_for_outcome(favorite_key, home_team, away_team)
    favorite_probability = contextual_probs.get(favorite_key or "", 0.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)

    lines = [
        f"La conclusione prudente e che {favorite} parta con {_probability_strength(favorite_probability)}.",
        f"Il livello di fiducia e {_confidence_label(confidence)} ({confidence:.1f}/100), quindi la lettura va pesata ma non trasformata in certezza.",
    ]
    if draw_risk >= 45:
        lines.append("Il pareggio resta uno scenario da tenere vivo, soprattutto se la partita non si apre presto.")
    if upset_risk >= 45:
        lines.append("Lo scenario alternativo e credibile se il favorito non riesce a controllare il proprio vantaggio nei dati.")
    lines.append("Il monitoraggio vicino al match dovrebbe partire da formazioni, assenze e possibili rotazioni.")
    lines.append("Senza quei dati, il report descrive tendenze aggregate: produzione, solidita, forma, Elo, calendario e interazione tra profili.")
    return "\n".join(lines[:8])


def build_narrative_match_report(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    season: str,
    schedule_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return {"ok": False, "message": "La stagione selezionata non contiene dati utilizzabili."}
    if home_team == away_team:
        return {"ok": False, "message": "Seleziona due squadre diverse."}
    teams = get_teams(prepared_df)
    if home_team not in teams or away_team not in teams:
        return {"ok": False, "message": "Una o entrambe le squadre non sono presenti nella stagione selezionata."}

    warnings: list[str] = []
    schedule_source_df = schedule_df if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty else prepared_df
    match_report = _safe_layer(
        "Report Partita",
        lambda: build_match_report_data(prepared_df, season, home_team, away_team, schedule_df=schedule_source_df),
        warnings,
    )
    matchup = _safe_layer(
        "Matchup Analysis",
        lambda: build_matchup_analysis(prepared_df, home_team, away_team, schedule_df=schedule_source_df),
        warnings,
    )
    home_identity = _safe_layer(
        f"Studio Squadra {home_team}",
        lambda: build_team_identity_report(prepared_df, home_team, schedule_df=schedule_source_df),
        warnings,
    )
    away_identity = _safe_layer(
        f"Studio Squadra {away_team}",
        lambda: build_team_identity_report(prepared_df, away_team, schedule_df=schedule_source_df),
        warnings,
    )

    prediction = {}
    if isinstance(match_report, dict):
        prediction = match_report.get("prediction", {}) or {}
    if not prediction and isinstance(matchup, dict):
        prediction = matchup.get("predictor", {}) or {}

    contextual = build_contextual_forecast(prediction, matchup_analysis=matchup if isinstance(matchup, dict) else {})
    report: dict[str, Any] = {
        "ok": True,
        "season": season,
        "home_team": home_team,
        "away_team": away_team,
        "match_title": f"{home_team} - {away_team}",
        "match_report": match_report if isinstance(match_report, dict) else {},
        "matchup_analysis": matchup if isinstance(matchup, dict) else {},
        "home_identity": home_identity if isinstance(home_identity, dict) else {},
        "away_identity": away_identity if isinstance(away_identity, dict) else {},
        "prediction": prediction,
        "contextual_forecast": contextual,
        "warnings": warnings,
    }
    reliability = build_reliability_assessment(report)
    report.update(
        {
            "brief_reading": build_match_opening_reading(report),
            "contextual_prediction_reading": build_contextual_prediction_reading(prediction, contextual, home_team, away_team),
            "probable_match_script": build_probable_match_script(report),
            "alternative_match_script": build_alternative_match_script(report),
            "team_identity_interaction": build_team_identity_interaction(report),
            "reliable_data": reliability["reliable"],
            "fragile_data": reliability["fragile"],
            "data_gaps": build_data_gaps_section(report),
            "what_could_change": build_what_could_change_match(report),
            "final_analyst_take": build_final_analyst_take(report),
            "technical": {
                "base_probabilities": contextual.get("base_probabilities", {}),
                "contextual_probabilities": contextual.get("contextual_probabilities", {}),
                "adjusted_edge": contextual.get("adjusted_edge"),
                "draw_risk": contextual.get("draw_risk"),
                "upset_risk": contextual.get("upset_risk"),
                "confidence": contextual.get("confidence"),
                "weighted_factors": (matchup or {}).get("context_engine", {}).get("weighted_factors", [])
                if isinstance(matchup, dict)
                else [],
            },
        }
    )
    return report
