from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.analytics import get_teams, prepare_matches_dataframe
from src.forecast_context import build_contextual_forecast
from src.matchup_analysis import build_matchup_analysis
from src.reporting import build_match_report_data
from src.team_identity import build_team_identity_report


OUTCOMES = ("1", "X", "2")
OUTCOME_LABELS = {"1": "casa", "X": "pareggio", "2": "trasferta"}
FORBIDDEN_NARRATIVE_PHRASES = [
    "Dato osservato:",
    "Indicatore interno:",
    "Ipotesi prudente:",
    "driver principali",
    "I fattori più solidi sono",
    "I fattori piu solidi sono",
    "vantaggio stilistico stimato",
    "mismatch attacco",
    "mismatch secondario",
    "squadra equilibrata; squadra equilibrata",
]


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


def _probability_tone(probability: float) -> str:
    if probability >= 0.60:
        return "quadro abbastanza netto"
    if probability >= 0.50:
        return "margine presente ma non dominante"
    return "margine sottile"


def _direction_word(probability: float) -> str:
    if probability >= 0.60:
        return "lettura chiaramente orientata"
    if probability >= 0.50:
        return "lettura con un margine presente ma non dominante"
    if probability >= 0.40:
        return "lettura aperta, con una tendenza"
    return "lettura equilibrata"


def _favorite_or_match(home_team: str, away_team: str, favorite_key: str | None, probabilities: dict[str, float]) -> str:
    if favorite_key == "X":
        return f"{home_team} - {away_team}"
    favorite = _team_for_outcome(favorite_key, home_team, away_team)
    probability = probabilities.get(favorite_key or "", 0.0)
    if probability >= 0.60:
        return f"{favorite}, con un vantaggio chiaro"
    if probability >= 0.50:
        return f"{favorite}, con un margine reale ma non definitivo"
    return f"{favorite}, dentro una partita ancora aperta"


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
        return "più offensiva"
    if solidity is not None and solidity >= 60 and (offense is None or offense < 58):
        return "più solida"
    if risk is not None and risk >= 60:
        return "più vulnerabile senza palla"
    return "equilibrata"


def _volatility_label(identity_report: dict[str, Any]) -> str:
    volatility = identity_report.get("volatility", {}) if isinstance(identity_report, dict) else {}
    return str(volatility.get("label") or "non disponibile")


def _volatility_phrase(label: str) -> str:
    text = " ".join(str(label or "non disponibile").split()).lower()
    if text in {"non disponibile", ""}:
        return "non disponibile"
    text = text.replace("volatilità", "").replace("volatilita", "").strip()
    return text or str(label)


def _a_team(team: Any) -> str:
    name = str(team or "questa squadra")
    return f"ad {name}" if name[:1].lower() in {"a", "e", "i", "o", "u"} else f"a {name}"


def _variant_index(*values: Any, modulo: int = 3) -> int:
    text = "|".join(str(value or "") for value in values)
    return sum(ord(char) for char in text) % max(modulo, 1)


def _normalize_metric_label(label: Any) -> str:
    text = " ".join(str(label or "").lower().split())
    replacements = {
        "pericolosita": "pericolosità",
        "solidita": "solidità",
        "volatilita": "volatilità",
        "dipendenza casa": "dipendenza casa",
        "rating elo": "rating elo",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _metric_edge_phrase(label: Any) -> str:
    normalized = _normalize_metric_label(label)
    mapping = {
        "pericolosità offensiva": "una produzione offensiva più incisiva",
        "solidità difensiva": "una tenuta difensiva più affidabile",
        "volume offensivo": "un volume offensivo superiore",
        "rischio difensivo": "un profilo difensivo meno esposto",
        "efficienza realizzativa": "una maggiore efficienza nel trasformare le occasioni",
        "dipendenza casa": "un rapporto casa/fuori più favorevole",
        "momento recente": "un momento recente più convincente",
        "forza calendario": "un contesto calendario che il modello legge in modo leggermente più favorevole",
        "rating elo": "un profilo di forza più solido",
    }
    return mapping.get(normalized, normalized)


def _metric_on_phrase(label: Any) -> str:
    normalized = _normalize_metric_label(label)
    mapping = {
        "pericolosità offensiva": "sulla produzione offensiva",
        "solidità difensiva": "sulla solidità difensiva",
        "volume offensivo": "sul volume offensivo",
        "rischio difensivo": "sul rischio difensivo",
        "efficienza realizzativa": "sull'efficienza realizzativa",
        "dipendenza casa": "sul rapporto casa/fuori",
        "momento recente": "sul momento recente",
        "forza calendario": "sul contesto calendario",
        "rating elo": "sul rating Elo",
    }
    return mapping.get(normalized, f"su {normalized}")


def _metric_short_phrase(label: Any) -> str:
    normalized = _normalize_metric_label(label)
    mapping = {
        "pericolosità offensiva": "produzione offensiva",
        "solidità difensiva": "solidità difensiva",
        "volume offensivo": "volume offensivo",
        "rischio difensivo": "rischio difensivo",
        "efficienza realizzativa": "efficienza realizzativa",
        "dipendenza casa": "rapporto casa/fuori",
        "momento recente": "momento recente",
        "forza calendario": "contesto calendario",
        "rating elo": "rating Elo",
    }
    return mapping.get(normalized, normalized)


def _join_phrases(items: list[str]) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} e {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} e {cleaned[-1]}"


def _humanize_signal(text: Any, home_team: str, away_team: str) -> str:
    sentence = " ".join(str(text or "").split())
    replacements = {
        "mismatch attacco vs difesa": "confronto tra produzione offensiva e tenuta difensiva",
        "mismatch attacco-difesa": "confronto tra produzione offensiva e tenuta difensiva",
        "Il mismatch attacco vs difesa": "Il confronto tra produzione offensiva e tenuta difensiva",
        "Il mismatch attacco-difesa": "Il confronto tra produzione offensiva e tenuta difensiva",
        "mismatch migliore": "segnale migliore",
        "driver principali": "segnali principali",
        "segnale casa": f"favorisce {home_team}",
        "segnale trasferta": f"favorisce {away_team}",
        "Segnale migliore per": "Il dato premia",
    }
    for old, new in replacements.items():
        sentence = sentence.replace(old, new)
    return sentence


def _style_takeaway(style: dict[str, Any], home_team: str, away_team: str) -> str:
    label = str(style.get("label") or "").lower()
    drivers = [str(driver) for driver in style.get("drivers", []) if str(driver or "").strip()]
    if "trasferta" in label:
        base = f"Il contesto premia soprattutto {away_team}: il margine non nasce solo dal campo, ma dalla somma tra qualità del profilo e segnali recenti."
    elif "casa" in label:
        base = f"Il contesto premia soprattutto {home_team}: il fattore campo pesa, ma conta anche la coerenza dei segnali tecnici e recenti."
    elif label:
        base = "Il contesto non crea una frattura netta: diversi indicatori si bilanciano e tengono il match più aperto."
    else:
        return ""

    driver_notes: list[str] = []
    if "rating Elo" in drivers:
        driver_notes.append("il rating Elo sostiene il quadro di forza")
    if "mismatch attacco vs difesa" in drivers or "mismatch attacco-difesa" in drivers:
        driver_notes.append("il confronto attacco-difesa rende il vantaggio più concreto")
    if "momento recente" in drivers:
        driver_notes.append("il momento recente va nella stessa direzione")
    if "rendimento casa/fuori" in drivers:
        driver_notes.append("il rendimento casa/fuori non contraddice la lettura")
    if "predictor" in drivers:
        driver_notes.append("il modello base conferma il segnale")
    if driver_notes:
        base = f"{base} In particolare, {', '.join(driver_notes[:3])}."
    return base


def _combined_mismatch_takeaway(
    mismatches: list[Any],
    home_team: str,
    away_team: str,
    favorite: str,
    underdog: str,
) -> str:
    cleaned = [_humanize_signal(item, home_team, away_team) for item in mismatches if str(item or "").strip()]
    if not cleaned:
        return ""
    favorite_mentions = [item for item in cleaned if favorite != "pareggio" and favorite in item]
    if len(favorite_mentions) >= 2 and underdog not in {"una delle due squadre", "chi resta più lucido nei dettagli", "chi resta piu lucido nei dettagli"}:
        return (
            f"Il margine principale nasce da una doppia direzione: {favorite} può togliere ritmo alla produzione di {underdog} "
            f"e, dall'altra parte, ha più strumenti per mettere pressione alla sua tenuta difensiva."
        )
    return cleaned[0]


def _schedule_takeaway(schedule: dict[str, Any], home_team: str, away_team: str) -> str:
    if not isinstance(schedule, dict) or not schedule.get("available"):
        return ""
    home_load = schedule.get("home_schedule_load", {}) or {}
    away_load = schedule.get("away_schedule_load", {}) or {}
    rest_home = schedule.get("rest_days_home")
    rest_away = schedule.get("rest_days_away")
    load_home = str(home_load.get("load_label") or "non disponibile")
    load_away = str(away_load.get("load_label") or "non disponibile")
    rest_advantage = _safe_float(schedule.get("rest_advantage"), 0.0)
    audit = schedule.get("competition_audit", {}) if isinstance(schedule.get("competition_audit"), dict) else {}

    pieces = []
    if rest_home is not None and rest_away is not None:
        if abs(rest_advantage) >= 2:
            advantaged = home_team if rest_advantage > 0 else away_team
            pieces.append(f"Sul calendario c'è un piccolo vantaggio di recupero per {advantaged}.")
        else:
            pieces.append("Il calendario non segnala una differenza di riposo abbastanza ampia da cambiare da solo la lettura.")
    if load_home != "non disponibile" or load_away != "non disponibile":
        pieces.append(f"Il carico recente è {load_home} per {home_team} e {load_away} per {away_team}.")
    return " ".join(pieces[:3])


def _team_goal_to_confirm(favorite: str, underdog: str, favorite_key: str | None) -> str:
    if favorite_key not in {"1", "2"}:
        return "La squadra che riesce a dare continuità al proprio volume può spostare una gara altrimenti molto sottile."
    return (
        f"Per {favorite}, confermare il vantaggio significa trasformare il margine in presenza offensiva: arrivare più spesso di {underdog} "
        "in zone utili e difendere abbastanza bene da non lasciare il match agli episodi."
    )


def _underdog_route(underdog: str, favorite: str, favorite_key: str | None) -> str:
    if favorite_key not in {"1", "2"}:
        return "In una partita così vicina, il primo tratto può pesare molto: chi concede meno situazioni pericolose si prende il lato migliore del match."
    variant = _variant_index(underdog, favorite, modulo=3)
    if variant == 0:
        return (
            f"A {underdog} serve una gara paziente: ridurre il volume concesso, restare nel punteggio "
            f"e obbligare {favorite} a cercare soluzioni meno pulite."
        )
    if variant == 1:
        return (
            f"{underdog} può renderla più scomoda se limita la produzione di {favorite} e porta la partita dentro un margine più stretto, "
            "dove ogni scelta pesa di più."
        )
    return (
        f"Il percorso di {underdog} passa da una partita meno aperta: concedere poco, non perdere contatto e costringere {favorite} "
        "a costruire il vantaggio con pazienza."
    )


def _metric_edges_for_team(matchup: dict[str, Any], team: str, limit: int = 3) -> list[str]:
    comparison = matchup.get("metric_comparison", []) if isinstance(matchup, dict) else []
    labels: list[str] = []
    for row in comparison:
        if not isinstance(row, dict) or row.get("leader") != team:
            continue
        edge = str(row.get("edge") or "").lower()
        label = _metric_edge_phrase(row.get("label"))
        if not label or edge in {"simile", "n/d"}:
            continue
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _balanced_edge_sentence(matchup: dict[str, Any], home_team: str, away_team: str) -> str:
    home_edges = _metric_edges_for_team(matchup, home_team, limit=2)
    away_edges = _metric_edges_for_team(matchup, away_team, limit=2)
    if home_edges and away_edges:
        return (
            f"L'equilibrio nasce da segnali che non vanno tutti dalla stessa parte: {home_team} ha qualcosa in più su "
            f"{', '.join(home_edges)}, mentre {away_team} risponde su {', '.join(away_edges)}."
        )
    if home_edges:
        return f"Il match resta aperto perché il vantaggio di {home_team} su {', '.join(home_edges)} non basta da solo a chiudere la lettura."
    if away_edges:
        return f"Il match resta aperto perché il vantaggio di {away_team} su {', '.join(away_edges)} non basta da solo a chiudere la lettura."
    return "L'equilibrio nasce soprattutto dall'assenza di un segnale dominante: classifica, forma e produzione non creano uno strappo netto."


def _why_favorite_is_ahead(matchup: dict[str, Any], favorite: str, home_team: str, away_team: str) -> str:
    edges = _metric_edges_for_team(matchup, favorite, limit=3)
    if edges:
        return f"Dai dati disponibili, il vantaggio di {favorite} è sostenuto soprattutto da {_join_phrases(edges)}."
    style = matchup.get("style_advantage", {}) if isinstance(matchup, dict) else {}
    style_takeaway = _style_takeaway(style, home_team, away_team)
    if style_takeaway:
        return style_takeaway
    return f"Il sistema legge {favorite} avanti per la combinazione tra produzione, forma recente e rendimento complessivo."


def _draw_risk_reason(draw_risk: float, confidence: float, favorite_prob: float, prediction: dict[str, Any]) -> str:
    if draw_risk < 45:
        return ""
    goals_home = _safe_float(prediction.get("expected_goals_home"))
    goals_away = _safe_float(prediction.get("expected_goals_away"))
    production_gap = abs(goals_home - goals_away)
    if production_gap < 0.25:
        return "Il pareggio pesa soprattutto perché la produzione offensiva stimata è molto vicina."
    if confidence < 55:
        return "Il pareggio pesa perché la fiducia non è piena: i segnali non sono abbastanza allineati da rendere stabile una direzione."
    if favorite_prob < 0.52:
        return "Il pareggio pesa perché il vantaggio del favorito è sottile e non assorbe bene una partita chiusa."
    return "Il pareggio pesa se la sfavorita riesce a rallentare la produzione offensiva e a far passare minuti senza che il favorito trovi continuità."


def _upset_risk_reason(upset_risk: float, confidence: float, favorite: str, underdog: str, matchup: dict[str, Any]) -> str:
    if upset_risk < 45 or underdog in {"una delle due squadre", "chi resta più lucido nei dettagli", "chi resta piu lucido nei dettagli"}:
        return ""
    underdog_edges = _metric_edges_for_team(matchup, underdog, limit=2)
    if underdog_edges:
        return f"Il rischio upset ha spazio perché {underdog} non parte senza appigli: i segnali migliori arrivano da {', '.join(underdog_edges)}."
    if confidence < 55:
        return f"Il rischio upset nasce dalla bassa coerenza del quadro: {favorite} è avanti, ma i segnali non lo proteggono del tutto."
    return f"Il rischio upset cresce se {favorite} non trasforma il vantaggio in occasioni e permette a {underdog} di restare viva fino alla parte finale."


def _alternative_opening(favorite: str, underdog: str, favorite_prob: float) -> str:
    variant = _variant_index(favorite, underdog, modulo=3)
    if favorite_prob >= 0.60:
        if variant == 0:
            return (
                f"Lo scenario alternativo più realistico non è un dominio di {underdog}: è una partita più chiusa e meno fluida del previsto, "
                f"in cui {favorite} produce meno del previsto e permette a {underdog} di arrivare ancora dentro il match nella seconda parte."
            )
        if variant == 1:
            return (
                f"L'alternativa prende forma se {favorite} non riesce a dare ritmo al proprio vantaggio: {underdog} non deve dominare, "
                "ma restare abbastanza vicino da rendere il finale più incerto."
            )
        return (
            f"Il copione cambia se {favorite} fatica a trasformare il margine in occasioni: in quel caso {underdog} può tenere la gara più stretta "
            "e aumentare il peso dei singoli momenti."
        )
    if variant == 0:
        return (
            f"Lo scenario alternativo nasce se {favorite} non riesce ad allungare il proprio piccolo margine: a quel punto {underdog} può trasformare "
            "una gara di attesa in una partita molto più scomoda."
        )
    if variant == 1:
        return (
            f"L'alternativa passa da un vantaggio di {favorite} che resta troppo sottile: se {underdog} rimane agganciata, la lettura iniziale perde stabilità."
        )
    return (
        f"Il match si apre davvero se {favorite} non separa subito i valori: {underdog} può restare in scia e portare la partita su dettagli più piccoli."
    )


def _alternative_low_volume_route(underdog: str, favorite: str) -> str:
    variant = _variant_index(underdog, favorite, "low-volume", modulo=3)
    if variant == 0:
        return f"Per {underdog}, lo scenario utile è una gara asciutta: meno volume concesso, punteggio aperto più a lungo e meno spazio per il margine di {favorite}."
    if variant == 1:
        return f"{underdog} può restare dentro se abbassa il ritmo della produzione avversaria e porta {favorite} a cercare vantaggi meno naturali."
    return f"La partita diventa più favorevole a {underdog} se resta compressa: poche sequenze davvero pulite e {favorite} costretta a forzare di più."


def _clean_narrative_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "Dato osservato:": "",
        "Indicatore interno:": "",
        "Ipotesi prudente:": "",
        "I fattori più solidi sono:": "La lettura si appoggia soprattutto a",
        "I fattori piu solidi sono:": "La lettura si appoggia soprattutto a",
        "driver principali": "segnali principali",
        "vantaggio stilistico stimato": "vantaggio letto dai dati",
        "mismatch secondario": "segnale alternativo",
        "squadra equilibrata; squadra equilibrata": "profili vicini",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("c'e", "c'è")
    accent_replacements = {
        "piu": "più",
        "perche": "perché",
        "puo": "può",
        "gia": "già",
        "pero": "però",
        "qualita": "qualità",
        "capacita": "capacità",
        "continuita": "continuità",
        "possibilita": "possibilità",
        "probabilita": "probabilità",
        "stabilita": "stabilità",
        "solidita": "solidità",
        "volatilita": "volatilità",
        "disponibilita": "disponibilità",
        "pericolosita": "pericolosità",
    }
    for old, new in accent_replacements.items():
        text = re.sub(rf"\b{old}\b", new, text)
    text = text.replace("1 giorni", "1 giorno")
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _clean_report_narrative(report: dict[str, Any]) -> None:
    text_keys = [
        "brief_reading",
        "contextual_prediction_reading",
        "probable_match_script",
        "alternative_match_script",
        "team_identity_interaction",
        "final_analyst_take",
    ]
    list_keys = ["reliable_data", "fragile_data", "data_gaps", "what_could_change"]
    for key in text_keys:
        report[key] = _clean_narrative_text(report.get(key, ""))
    for key in list_keys:
        report[key] = [_clean_narrative_text(item) for item in report.get(key, [])]


def _narrative_has_forbidden_phrases(report: dict[str, Any]) -> bool:
    chunks: list[str] = []
    for key in ["brief_reading", "probable_match_script", "alternative_match_script", "team_identity_interaction", "final_analyst_take"]:
        chunks.append(str(report.get(key, "")))
    chunks.extend(str(item) for key in ["reliable_data", "fragile_data", "what_could_change"] for item in report.get(key, []))
    full_text = "\n".join(chunks).lower()
    return any(phrase.lower() in full_text for phrase in FORBIDDEN_NARRATIVE_PHRASES)


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
            "Il predictor base non è disponibile in modo completo: la lettura numerica resta parziale "
            "e il report si appoggia soprattutto a forma, classifica, profili e matchup."
        )

    base_line = (
        f"Il predictor base parte da {base_favorite}: probabilità "
        f"{_format_pct(base_probs.get('1'))} / {_format_pct(base_probs.get('X'))} / {_format_pct(base_probs.get('2'))}."
    )
    contextual_line = (
        f"La lettura contestuale indica {contextual_favorite} con {_probability_strength(contextual_probability)} "
        f"e probabilità {_format_pct(contextual_probs.get('1'))} / {_format_pct(contextual_probs.get('X'))} / {_format_pct(contextual_probs.get('2'))}."
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

    if favorite_key == "X":
        opening = (
            f"{home_team} - {away_team} nasce come una partita da margini sottili: il pareggio non è una semplice alternativa, "
            "ma uno degli scenari centrali della fotografia pre-partita."
        )
    elif favorite_prob >= 0.60:
        opening = (
            f"La lettura è chiaramente orientata verso {favorite}. "
            f"Il {favorite_prob * 100:.1f}% contestuale non racconta una certezza, ma un vantaggio abbastanza netto rispetto agli altri esiti."
        )
    elif favorite_prob >= 0.50:
        opening = (
            f"{favorite} parte avanti, ma il margine non è dominante: la previsione vede una direzione, "
            "non una partita già scritta."
        )
    else:
        opening = (
            f"La partita resta aperta: {favorite} ha una leggera tendenza favorevole, "
            "ma il distacco tra gli scenari è abbastanza sottile da lasciare spazio a più copioni."
        )

    lines = [opening]
    if home_position and away_position:
        lines.append(
            f"La classifica pesa nella lettura: {home_team} arriva dalla posizione {home_position}, "
            f"{away_team} dalla posizione {away_position}, e questo aiuta a spiegare il punto di partenza del report."
        )
    if prediction.get("ok"):
        goals_home = _safe_float(prediction.get("expected_goals_home"))
        goals_away = _safe_float(prediction.get("expected_goals_away"))
        production_gap = abs(goals_home - goals_away)
        if production_gap >= 0.65:
            production_note = "con una separazione abbastanza chiara nella produzione offensiva stimata"
        elif production_gap >= 0.25:
            production_note = "con una differenza leggibile ma non enorme nella produzione offensiva stimata"
        else:
            production_note = "con valori di produzione offensiva stimata molto vicini"
        lines.append(
            f"Il modello Poisson interno parte da {goals_home:.2f} gol attesi interni per {home_team} e {goals_away:.2f} per {away_team}, "
            f"{production_note}."
        )
    if confidence >= 70:
        lines.append(f"La fiducia è alta ({confidence:.1f}/100): il quadro è coerente e non dipende da un solo indicatore.")
    elif confidence < 45:
        lines.append(f"La fiducia è bassa ({confidence:.1f}/100): i segnali sono poco allineati e la lettura diventa più fragile.")
    else:
        lines.append(f"La fiducia è media ({confidence:.1f}/100): c'è una direzione, ma il report non la considera abbastanza stabile da chiudere il discorso.")
    if draw_risk >= 60:
        lines.append("Il rischio pareggio è alto: se la partita si blocca, ogni minuto senza strappi può togliere forza al favorito.")
    elif draw_risk >= 45:
        lines.append("Il pareggio non domina la previsione, ma resta uno scenario da non trascurare.")
    if upset_risk >= 60:
        lines.append("Il rischio di ribaltamento è alto: il favorito ha margine, ma non abbastanza controllo da assorbire una partita più ruvida del previsto.")
    elif upset_risk >= 45:
        lines.append("Il rischio di ribaltamento è medio: lo scenario alternativo resta credibile.")
    return "\n".join(lines[:7])


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
    favorite_key = _favorite_key(contextual_probs)
    favorite = _team_for_outcome(favorite_key, home_team, away_team)
    underdog = away_team if favorite_key == "1" else home_team if favorite_key == "2" else "chi resta più lucido nei dettagli"
    favorite_prob = contextual_probs.get(favorite_key or "", 0.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)

    if favorite_key == "X":
        lines = [
            "La partita suggerita dai dati assomiglia a una gara in cui nessuna squadra può permettersi di concedere troppo: il primo vantaggio vero può nascere dal ridurre gli errori prima ancora che dall'aumentare il volume.",
        ]
    elif favorite_prob >= 0.60:
        lines = [
            f"Il copione più probabile vede {favorite} provare a rendere visibile il proprio vantaggio: più presenza offensiva, meno spazi concessi a {underdog}, e una partita che non deve dipendere da singoli episodi.",
        ]
    elif favorite_prob >= 0.50:
        lines = [
            f"La lettura più probabile vede {favorite} leggermente avanti, ma il margine va costruito dentro la partita: deve produrre abbastanza da separarsi e, allo stesso tempo, non offrire a {underdog} una gara fatta di pochi momenti decisivi.",
        ]
    else:
        lines = [
            f"Qui il punto non è ribadire il vantaggio di {favorite}, ma capire quanto sia sottile: i segnali sono vicini e il copione può cambiare direzione con poco.",
        ]
    if favorite_key in {"1", "2"}:
        lines.append(_why_favorite_is_ahead(matchup, favorite, home_team, away_team))
    else:
        lines.append(_balanced_edge_sentence(matchup, home_team, away_team))
    mismatch_takeaway = _combined_mismatch_takeaway(mismatches, home_team, away_team, favorite, underdog)
    if mismatch_takeaway:
        lines.append(mismatch_takeaway)
    if prediction.get("ok"):
        goals_gap = _safe_float(prediction.get("expected_goals_home")) - _safe_float(prediction.get("expected_goals_away"))
        if abs(goals_gap) < 0.25:
            lines.append("La produzione offensiva stimata è vicina: questo aumenta la possibilità di una gara paziente, in cui il primo strappo può pesare molto.")
        elif goals_gap > 0:
            lines.append(f"Il margine di {home_team} nasce soprattutto dalla capacità di trasformare il fattore campo in volume e continuità offensiva.")
        else:
            lines.append(f"Il margine di {away_team} passa dalla capacità di tenere alta la propria produzione anche fuori casa, senza lasciare la gara in balia dei dettagli.")
    if favorite_key in {"1", "2"}:
        lines.append(_team_goal_to_confirm(favorite, underdog, favorite_key))
        lines.append(_underdog_route(underdog, favorite, favorite_key))
    risk_reason = _draw_risk_reason(draw_risk, confidence, favorite_prob, prediction)
    if risk_reason:
        lines.append(risk_reason)
    upset_reason = _upset_risk_reason(upset_risk, confidence, favorite, underdog, matchup)
    if upset_reason:
        lines.append(upset_reason)
    schedule_note = _schedule_takeaway(schedule, home_team, away_team)
    if schedule_note:
        lines.append(schedule_note)
    return "\n".join(_dedupe(lines, limit=7))


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
    favorite_prob = contextual_probs.get(favorite_key or "", 0.0)

    lines: list[str] = []
    if favorite_key in {"1", "2"}:
        lines.append(_alternative_opening(favorite, underdog, favorite_prob))
    else:
        lines.append("Lo scenario alternativo nasce da una squadra che rompe l'equilibrio con più continuità offensiva dell'altra, prima che il pareggio diventi il centro emotivo e statistico della gara.")
    if draw_risk >= 60:
        lines.append(_draw_risk_reason(draw_risk, confidence, favorite_prob, report.get("prediction", {}) or {}))
    elif draw_risk >= 45:
        lines.append(_draw_risk_reason(draw_risk, confidence, favorite_prob, report.get("prediction", {}) or {}))
    elif favorite_key in {"1", "2"}:
        lines.append(_alternative_low_volume_route(underdog, favorite))
    if upset_risk >= 45 and favorite_key in {"1", "2"}:
        lines.append(_upset_risk_reason(upset_risk, confidence, favorite, underdog, matchup))
    if confidence < 45:
        lines.append("La lettura è fragile perché i segnali non sono allineati: basta poco per spostare il peso del report.")
    elif confidence >= 70 and favorite_key in {"1", "2"}:
        lines.append("Proprio perché il quadro è coerente, l'alternativa passa più da ciò che succede in campo che da una contraddizione forte nei dati pre-match.")
    if len(mismatches) > 1 and favorite_key in {"1", "2"}:
        lines.append(
            f"Il punto da spegnere per {underdog} è il principale vantaggio di {favorite}: se riesce ad abbassarlo, il copione diventa meno lineare."
        )
    lines.append("Turnover, assenze e scelte iniziali restano il limite più serio della lettura: possono cambiare il peso del vantaggio tecnico prima ancora del fischio d'inizio.")
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
    contextual = report.get("contextual_forecast", {}) or {}
    contextual_probs = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    favorite_key = _favorite_key(contextual_probs)
    favorite = _team_for_outcome(favorite_key, home_team, away_team)

    home_volatility = _volatility_label(home_identity)
    away_volatility = _volatility_label(away_identity)
    home_volatility_reading = _volatility_phrase(home_volatility)
    away_volatility_reading = _volatility_phrase(away_volatility)

    notable = [
        row for row in comparison
        if isinstance(row, dict) and row.get("leader") and row.get("edge") not in {None, "simile", "n/d"}
    ]
    lines: list[str] = []
    if notable:
        first = notable[0]
        leader = first.get("leader")
        first_label = _metric_on_phrase(first.get("label") or "un indicatore principale")
        if favorite_key in {"1", "2"} and leader != favorite:
            lines.append(
                f"L'interazione tra le squadre introduce un contrappeso: {first_label} il confronto premia {leader}, "
                f"quindi il vantaggio di {favorite} non copre ogni dimensione della partita."
            )
        else:
            lines.append(
                f"L'interazione tra le squadre sembra spostarsi soprattutto {first_label}: in questo punto il confronto premia {leader}."
            )
    else:
        lines.append(
            f"I profili di {home_team} e {away_team} sono vicini: il report non trova una frattura netta, quindi contano soprattutto chi concede meno situazioni pericolose e chi riesce a tenere più stabile la propria produzione."
        )

    if home_volatility != "non disponibile" or away_volatility != "non disponibile":
        if home_volatility_reading == away_volatility_reading:
            lines.append(f"La stabilità dei due profili è simile: entrambi entrano con una volatilità letta come {home_volatility_reading}.")
        else:
            lines.append(
                f"La stabilità può incidere: {home_team} mostra una volatilità {home_volatility_reading}, "
                f"{away_team} una volatilità {away_volatility_reading}."
            )
    if notable:
        top_labels = [
            _metric_short_phrase(row.get("label"))
            for row in notable[:3]
            if str(row.get("label") or "").strip()
        ]
        if len(top_labels) >= 2:
            leader = notable[0].get("leader")
            if favorite_key in {"1", "2"} and leader != favorite:
                lines.append(
                    f"Questi segnali danno {_a_team(leader)} un modo concreto per restare nella partita, soprattutto su {', '.join(top_labels[:2])}."
                )
            else:
                lines.append(
                    f"Il margine non sembra isolato: riguarda più dimensioni del rendimento e costruisce un vantaggio complessivo per {leader}."
                )
        elif top_labels:
            lines.append(f"Il punto più leggibile resta {top_labels[0]}, dove il dato aggregato è meno ambiguo.")
    if notable and notable[0].get("leader") == away_team:
        lines.append(
            f"{home_team} deve probabilmente abbassare la qualità delle situazioni concesse: se lascia campo e continuità, {away_team} ha più margine per confermare la lettura."
        )
    elif notable and notable[0].get("leader") == home_team:
        lines.append(f"{away_team} deve evitare che il vantaggio di {home_team} diventi ritmo stabile: la partita può complicarsi se il fattore campo produce volume continuo.")
    lines.append(
        "Questa è comunque una lettura di indicatori aggregati: non basta per parlare con certezza di moduli, pressing o costruzione dal basso."
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
        reliable.append("La coerenza tra predictor base e lettura contestuale: entrambi indicano la stessa direzione.")
    if confidence >= 70:
        reliable.append("La confidence alta segnala che i fattori principali sono abbastanza allineati.")
    elif confidence < 45:
        fragile.append("La confidence bassa renderebbe il report più instabile, perché i segnali sarebbero poco allineati.")
    if ratings.get("home") and ratings.get("away"):
        reliable.append("Il rating Elo è disponibile per entrambe le squadre e aggiunge un riferimento di forza storica/recente.")
    if matchup.get("context_engine", {}).get("weighted_factors"):
        reliable.append("Il vantaggio del report non nasce da un solo dato: classifica, forma, casa/fuori e confronto tra profili vengono letti insieme.")
    if draw_risk >= 60:
        fragile.append("Draw risk alto: il pareggio può disturbare la lettura del favorito.")
    if upset_risk >= 60:
        fragile.append("Upset risk alto: il favorito non è protetto da segnali coerenti.")
    fragile.extend(
        [
            "Le assenze non note possono cambiare molto il peso della previsione.",
            "Non abbiamo dati evento per leggere davvero pressing, costruzione, possesso o sviluppo tattico.",
            "La lettura contestuale v2 resta sperimentale: aiuta a interpretare, ma non sostituisce il predictor base.",
        ]
    )
    schedule = match_report.get("schedule_context", {}) if isinstance(match_report, dict) else {}
    if isinstance(schedule, dict):
        audit = schedule.get("competition_audit", {})
        if isinstance(audit, dict) and audit.get("only_league_data"):
            fragile.append("Il calendario resta parziale se mancano coppe o competizioni europee nel database.")

    if not reliable:
        reliable.append("Classifica, gol, forma recente e rendimento casa/fuori restano la base più solida del report.")
    return {"reliable": _dedupe(reliable, limit=6), "fragile": _dedupe(fragile, limit=6)}


def build_data_gaps_section(report: dict[str, Any]) -> list[str]:
    return [
        "Formazioni ufficiali e modulo previsto.",
        "Assenze, squalifiche, condizioni fisiche e disponibilità dei giocatori chiave.",
        "Eventuale turnover e gestione dei minuti.",
        "Dati su pressing, possesso, costruzione dal basso, passaggi progressivi e lanci lunghi.",
        "Eventi shot-by-shot e contesto delle conclusioni.",
        "Stato motivazionale o pressione della gara: oggi sarebbe solo un proxy, non un dato certo.",
    ]


def build_what_could_change_match(report: dict[str, Any]) -> list[str]:
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
    items: list[str] = []
    if confidence < 55:
        items.append("Se i primi minuti contraddicono la lettura iniziale, la partita può cambiare volto più rapidamente del previsto.")
    if draw_risk >= 45:
        items.append("Se il favorito non trova ritmo nella prima parte, il pareggio diventa più pesante.")
    if upset_risk >= 45:
        items.append("Se la sfavorita tiene basso il numero di occasioni concesse, il vantaggio del modello perde forza.")
    mismatches = matchup.get("mismatches", []) if isinstance(matchup, dict) else []
    if mismatches and favorite_key in {"1", "2"}:
        items.append(
            f"La partita cambia se {underdog} riesce a spegnere il punto forte di {favorite}: in quel caso il vantaggio letto prima del match diventa meno stabile."
        )
    items.extend(
        [
            "Un gol precoce può trasformare una partita teoricamente controllabile in una gara più aperta.",
            "Turnover o assenze possono cambiare il peso del vantaggio tecnico.",
            "Cartellini, episodi e gestione emotiva dei momenti chiave non sono leggibili dai dati aggregati.",
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
    underdog = away_team if favorite_key == "1" else home_team if favorite_key == "2" else "una delle due squadre"
    favorite_probability = contextual_probs.get(favorite_key or "", 0.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)

    if favorite_key == "X":
        opening = (
            "La conclusione è che il pareggio abbia un peso reale nella fotografia pre-partita: "
            "la partita sembra più vicina a un equilibrio da rompere con produzione e precisione che a un vantaggio già definito."
        )
    elif favorite_probability >= 0.60:
        opening = (
            f"La previsione prudente mette {favorite} davanti in modo chiaro. "
            "Lo scenario principale è che questo vantaggio diventi più presenza offensiva, meno situazioni favorevoli concesse e una partita meno esposta ai singoli episodi."
        )
    elif favorite_probability >= 0.50:
        opening = (
            f"La previsione prudente mette {favorite} avanti, ma non abbastanza da cancellare lo scenario alternativo. "
            "Il margine c'è, però deve essere confermato con volume offensivo e tenuta difensiva, altrimenti può assottigliarsi presto."
        )
    else:
        opening = (
            f"La previsione prudente indica {favorite}, ma con un margine sottile: il report legge una tendenza, non un quadro chiuso, e basta poco per spostare l'equilibrio."
        )
    lines = [opening]
    if confidence >= 70:
        lines.append(f"La fiducia del report è alta ({confidence:.1f}/100): il quadro è coerente e non dipende da un solo segnale.")
    elif confidence < 45:
        lines.append(f"La fiducia del report è bassa ({confidence:.1f}/100): i segnali sono poco allineati e la previsione è più fragile.")
    else:
        lines.append(f"La fiducia del report è media ({confidence:.1f}/100): c'è una direzione, ma il margine non è definitivo.")
    if draw_risk >= 45:
        lines.append("Lo scenario alternativo passa da una partita più chiusa, in cui il pareggio guadagna peso col passare dei minuti.")
    if upset_risk >= 45:
        lines.append("Il ribaltamento della lettura diventa credibile se il favorito non riesce a tradurre il margine in controllo del match.")
    if favorite_key in {"1", "2"} and draw_risk < 45 and upset_risk < 45:
        lines.append(
            f"Lo scenario alternativo più credibile non è un ribaltamento immediato, ma una gara in cui {underdog} tiene basso il volume e costringe {favorite} a forzare soluzioni meno pulite."
        )
    lines.append("La cosa da monitorare vicino alla partita sono formazioni, assenze e possibili rotazioni: possono cambiare il peso della previsione.")
    lines.append("Senza quei dati, questo report resta una lettura di tendenze aggregate: produzione, solidità, forma, Elo, calendario e interazione tra profili.")
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
    _clean_report_narrative(report)
    report["narrative_quality_ok"] = not _narrative_has_forbidden_phrases(report)
    return report
