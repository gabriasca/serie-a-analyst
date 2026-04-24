from __future__ import annotations

from typing import Any

import pandas as pd

from src.advanced_metrics import build_advanced_team_metrics, get_team_advanced_metrics
from src.analytics import get_teams, prepare_matches_dataframe
from src.context_engine import build_context_adjusted_edge
from src.predictor import predict_match
from src.team_profiles import build_team_profile_context, build_team_profile_with_ratings


METRIC_SPECS = [
    {"key": "offensive_threat_index", "label": "Pericolosita offensiva", "higher_is_better": True, "kind": "performance"},
    {"key": "defensive_solidity_index", "label": "Solidita difensiva", "higher_is_better": True, "kind": "performance"},
    {"key": "offensive_volume_index", "label": "Volume offensivo", "higher_is_better": True, "kind": "performance"},
    {"key": "defensive_risk_index", "label": "Rischio difensivo", "higher_is_better": False, "kind": "risk"},
    {"key": "finishing_efficiency_index", "label": "Efficienza realizzativa", "higher_is_better": True, "kind": "performance"},
    {"key": "recent_momentum_index", "label": "Momento recente", "higher_is_better": True, "kind": "performance"},
    {"key": "home_dependency_index", "label": "Dipendenza casa", "higher_is_better": False, "kind": "risk"},
    {"key": "schedule_strength_index", "label": "Forza calendario", "higher_is_better": True, "kind": "context"},
    {"key": "elo_rating", "label": "Rating Elo", "higher_is_better": True, "kind": "elo"},
]


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_gap_level(gap: float | None, is_elo: bool = False) -> str:
    if gap is None:
        return "n/d"
    gap = abs(gap)
    if is_elo:
        if gap < 50:
            return "simile"
        if gap < 100:
            return "vantaggio leggero"
        return "vantaggio evidente"
    if gap <= 10:
        return "simile"
    if gap <= 20:
        return "vantaggio leggero"
    return "vantaggio marcato"


def _rating_block_from_metrics(team_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not team_metrics:
        return {
            "available": False,
            "rating_type": "elo",
            "rating_value": None,
            "rating_date": None,
            "source_name": None,
            "strength_band": None,
            "rating_rank": None,
        }

    rating_value = _safe_float(team_metrics.get("elo_rating"))
    return {
        "available": rating_value is not None,
        "rating_type": "elo",
        "rating_value": rating_value,
        "rating_date": team_metrics.get("rating_date"),
        "source_name": team_metrics.get("rating_source"),
        "strength_band": team_metrics.get("strength_band"),
        "rating_rank": team_metrics.get("elo_rank"),
    }


def compare_advanced_metrics(
    home_metrics: dict[str, Any] | None,
    away_metrics: dict[str, Any] | None,
    home_team: str = "Casa",
    away_team: str = "Trasferta",
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []

    for spec in METRIC_SPECS:
        home_value = _safe_float((home_metrics or {}).get(spec["key"]))
        away_value = _safe_float((away_metrics or {}).get(spec["key"]))
        if home_value is None and away_value is None:
            leader = None
            gap = None
        else:
            raw_gap = None if home_value is None or away_value is None else home_value - away_value
            gap = abs(raw_gap) if raw_gap is not None else None
            if raw_gap is None:
                leader = home_team if away_value is None else away_team
            elif spec["higher_is_better"]:
                leader = home_team if raw_gap > 0 else away_team if raw_gap < 0 else None
            else:
                leader = home_team if raw_gap < 0 else away_team if raw_gap > 0 else None

        level = _metric_gap_level(gap, is_elo=spec["kind"] == "elo")
        if leader is None or level == "simile":
            reading = "Valori simili o non abbastanza distanti."
        elif spec["key"] == "schedule_strength_index":
            reading = f"Calendario fin qui piu duro per {leader}."
        elif spec["key"] == "home_dependency_index":
            reading = f"Profilo piu stabile fuori dal solo fattore campo per {leader}."
        elif spec["key"] == "defensive_risk_index":
            reading = f"Rischio difensivo meglio controllato da {leader}."
        elif spec["kind"] == "elo":
            reading = f"Base storica o recente piu forte per {leader}."
        else:
            reading = f"Segnale migliore per {leader}."

        comparison_rows.append(
            {
                "metric_key": spec["key"],
                "label": spec["label"],
                "home_value": round(home_value, 1) if home_value is not None else None,
                "away_value": round(away_value, 1) if away_value is not None else None,
                "gap": round(gap, 1) if gap is not None else None,
                "leader": leader,
                "edge": level,
                "reading": reading,
            }
        )

    return comparison_rows


def identify_key_mismatches(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor_context: dict[str, Any] | None = None,
    comparison_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    home_team = str(home_profile.get("team") or "La squadra di casa")
    away_team = str(away_profile.get("team") or "La squadra ospite")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    mismatches: list[str] = []

    home_attack_vs_away_defense = (_safe_float(home_advanced.get("offensive_threat_index")) or 0.0) - (
        _safe_float(away_advanced.get("defensive_solidity_index")) or 0.0
    )
    away_attack_vs_home_defense = (_safe_float(away_advanced.get("offensive_threat_index")) or 0.0) - (
        _safe_float(home_advanced.get("defensive_solidity_index")) or 0.0
    )
    recent_gap = (_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) - (
        _safe_float(away_advanced.get("recent_momentum_index")) or 0.0
    )
    elo_gap = (_safe_float(home_advanced.get("elo_rating")) or 0.0) - (_safe_float(away_advanced.get("elo_rating")) or 0.0)
    home_field_gap = (home_profile.get("home_away", {}).get("ppm_home", 0.0) or 0.0) - (
        away_profile.get("home_away", {}).get("ppm_away", 0.0) or 0.0
    )
    home_dependency = _safe_float(home_advanced.get("home_dependency_index"))
    away_dependency = _safe_float(away_advanced.get("home_dependency_index"))

    if home_attack_vs_away_defense > 10:
        mismatches.append(
            f"{home_team} puo avere un vantaggio nella produzione offensiva: la sua pericolosita e piu alta della solidita difensiva di {away_team}."
        )
    elif home_attack_vs_away_defense < -10:
        mismatches.append(
            f"La struttura difensiva di {away_team} sembra ben attrezzata per togliere ritmo all'attacco di {home_team}."
        )

    if away_attack_vs_home_defense > 10:
        mismatches.append(
            f"{away_team} arriva con un attacco che puo mettere in difficolta la tenuta difensiva di {home_team}."
        )
    elif away_attack_vs_home_defense < -10:
        mismatches.append(
            f"{home_team} puo limitare bene la produzione offensiva di {away_team} se la partita resta nei suoi binari difensivi."
        )

    if abs(recent_gap) > 10:
        leader = home_team if recent_gap > 0 else away_team
        mismatches.append(f"{leader} arriva con un trend recente migliore, aspetto che puo spostare l'inerzia del matchup.")

    if abs(home_field_gap) > 0.45:
        if home_field_gap > 0:
            mismatches.append(
                f"Il contesto casa/fuori favorisce {home_team}: in casa raccoglie piu di quanto {away_team} riesca a fare mediamente in trasferta."
            )
        else:
            mismatches.append(
                f"{away_team} porta in trasferta un rendimento abbastanza solido da limitare parte del vantaggio campo di {home_team}."
            )

    if home_dependency is not None and home_dependency >= 65:
        mismatches.append(f"Il fattore campo pesa molto nel profilo di {home_team}, quindi capire se regge il suo standard interno e centrale.")
    if away_dependency is not None and away_dependency >= 65:
        mismatches.append(f"{away_team} ha un profilo molto legato al contesto: lontano da casa la tenuta del matchup va verificata.")

    predictor = predictor_context or {}
    if abs(elo_gap) >= 50 and predictor.get("available"):
        probability_gap = abs(float(predictor.get("home_probability", 0.0)) - float(predictor.get("away_probability", 0.0)))
        if probability_gap < 0.08:
            leader = home_team if elo_gap > 0 else away_team
            mismatches.append(
                f"Il rating storico favorisce {leader}, ma il predictor vede comunque una sfida abbastanza aperta nel contesto attuale."
            )

    if comparison_rows:
        marked_edges = [row for row in comparison_rows if row.get("edge") in {"vantaggio marcato", "vantaggio evidente"}]
        if marked_edges and len(mismatches) < 4:
            top_edge = marked_edges[0]
            leader = top_edge.get("leader")
            if leader:
                mismatches.append(f"Sul dato '{top_edge['label']}' il segnale piu netto e a favore di {leader}.")

    if not mismatches:
        mismatches.append("I profili statistici sono abbastanza vicini e il matchup resta piu equilibrato di quanto dica una sola metrica.")

    unique: list[str] = []
    for item in mismatches:
        if item not in unique:
            unique.append(item)
    return unique[:6]


def build_home_team_risks(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor_context: dict[str, Any] | None = None,
) -> list[str]:
    home_team = str(home_profile.get("team") or "La squadra di casa")
    away_team = str(away_profile.get("team") or "La squadra ospite")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    risks: list[str] = []

    if (_safe_float(away_advanced.get("offensive_threat_index")) or 0.0) > (_safe_float(home_advanced.get("defensive_solidity_index")) or 0.0) + 10:
        risks.append(f"L'attacco di {away_team} puo creare piu problemi del previsto alla fase difensiva di {home_team}.")
    if (_safe_float(away_advanced.get("recent_momentum_index")) or 0.0) > (_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) + 10:
        risks.append(f"{away_team} arriva con un momento recente migliore e puo alzare subito intensita e fiducia.")
    if (_safe_float(away_advanced.get("elo_rating")) or 0.0) > (_safe_float(home_advanced.get("elo_rating")) or 0.0) + 50:
        risks.append(f"Il rating Elo premia di piu {away_team}, quindi il margine qualitativo di fondo non va sottovalutato.")
    if (_safe_float(home_advanced.get("home_dependency_index")) or 0.0) >= 65:
        risks.append(f"{home_team} dipende molto dal fattore casa: se non impone subito il proprio contesto, il profilo si abbassa.")
    if (_safe_float(home_advanced.get("offensive_volume_index")) or 0.0) < 45:
        risks.append(f"Il volume offensivo di {home_team} non e altissimo e puo rendere piu difficile sostenere la pressione per 90 minuti.")
    if (_safe_float(home_advanced.get("finishing_efficiency_index")) or 0.0) < 45:
        risks.append(f"{home_team} rischia di non convertire abbastanza il volume creato se la partita resta bloccata.")

    predictor = predictor_context or {}
    if predictor.get("available") and float(predictor.get("away_probability", 0.0)) >= float(predictor.get("home_probability", 0.0)):
        risks.append("Il predictor non vede un vantaggio netto per la squadra di casa, quindi il margine del fattore campo non basta da solo.")

    if not risks:
        risks.append("Non emergono rischi dominanti per la squadra di casa, ma il match resta comunque sensibile agli episodi.")

    return risks[:5]


def build_away_team_risks(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor_context: dict[str, Any] | None = None,
) -> list[str]:
    home_team = str(home_profile.get("team") or "La squadra di casa")
    away_team = str(away_profile.get("team") or "La squadra ospite")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    risks: list[str] = []

    if (_safe_float(home_advanced.get("offensive_threat_index")) or 0.0) > (_safe_float(away_advanced.get("defensive_solidity_index")) or 0.0) + 10:
        risks.append(f"L'attacco di {home_team} puo mettere sotto pressione la tenuta difensiva di {away_team}.")
    if (home_profile.get("home_away", {}).get("ppm_home", 0.0) or 0.0) > (away_profile.get("home_away", {}).get("ppm_away", 0.0) or 0.0) + 0.5:
        risks.append(f"Il rendimento interno di {home_team} pesa piu del rendimento esterno di {away_team}.")
    if (_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) > (_safe_float(away_advanced.get("recent_momentum_index")) or 0.0) + 10:
        risks.append(f"{home_team} arriva con una forma piu brillante e puo imporre piu inerzia alla gara.")
    if (_safe_float(home_advanced.get("elo_rating")) or 0.0) > (_safe_float(away_advanced.get("elo_rating")) or 0.0) + 50:
        risks.append(f"Il rating Elo premia di piu {home_team}, segnale che puo contare soprattutto se il match si allunga.")
    if (_safe_float(away_advanced.get("home_dependency_index")) or 0.0) >= 65:
        risks.append(f"{away_team} mostra una forte dipendenza dal proprio contesto abituale e fuori casa questo puo pesare.")
    if (_safe_float(away_advanced.get("offensive_volume_index")) or 0.0) < 45:
        risks.append(f"Il volume offensivo di {away_team} non e alto e puo rendere piu difficile ribaltare l'inerzia del match.")

    predictor = predictor_context or {}
    if predictor.get("available") and float(predictor.get("home_probability", 0.0)) > float(predictor.get("away_probability", 0.0)) + 0.08:
        risks.append("Il predictor spinge piu verso la squadra di casa, quindi la trasferta parte con meno margine statistico.")

    if not risks:
        risks.append("Non emergono rischi dominanti per la squadra ospite, ma la trasferta resta comunque esposta al fattore campo.")

    return risks[:5]


def build_style_advantage(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    home_team = str(home_profile.get("team") or "Casa")
    away_team = str(away_profile.get("team") or "Trasferta")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    score = 0
    drivers: list[str] = []

    elo_gap = (_safe_float(home_advanced.get("elo_rating")) or 0.0) - (_safe_float(away_advanced.get("elo_rating")) or 0.0)
    if abs(elo_gap) >= 100:
        score += 2 if elo_gap > 0 else -2
        drivers.append("rating Elo")
    elif abs(elo_gap) >= 50:
        score += 1 if elo_gap > 0 else -1
        drivers.append("rating Elo")

    home_mismatch = ((_safe_float(home_advanced.get("offensive_threat_index")) or 0.0) - (_safe_float(away_advanced.get("defensive_solidity_index")) or 0.0))
    away_mismatch = ((_safe_float(away_advanced.get("offensive_threat_index")) or 0.0) - (_safe_float(home_advanced.get("defensive_solidity_index")) or 0.0))
    net_mismatch = home_mismatch - away_mismatch
    if abs(net_mismatch) > 20:
        score += 2 if net_mismatch > 0 else -2
        drivers.append("mismatch attacco vs difesa")
    elif abs(net_mismatch) > 10:
        score += 1 if net_mismatch > 0 else -1
        drivers.append("mismatch attacco vs difesa")

    recent_gap = (_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) - (
        _safe_float(away_advanced.get("recent_momentum_index")) or 0.0
    )
    if abs(recent_gap) > 20:
        score += 1 if recent_gap > 0 else -1
        drivers.append("momento recente")
    elif abs(recent_gap) > 10:
        score += 1 if recent_gap > 0 else -1
        drivers.append("momento recente")

    home_field_gap = (home_profile.get("home_away", {}).get("ppm_home", 0.0) or 0.0) - (
        away_profile.get("home_away", {}).get("ppm_away", 0.0) or 0.0
    )
    if abs(home_field_gap) > 0.75:
        score += 2 if home_field_gap > 0 else -2
        drivers.append("rendimento casa/fuori")
    elif abs(home_field_gap) > 0.35:
        score += 1 if home_field_gap > 0 else -1
        drivers.append("rendimento casa/fuori")

    if predictor and predictor.get("ok"):
        home_probability = float(predictor["probabilities"]["1"])
        away_probability = float(predictor["probabilities"]["2"])
        probability_gap = home_probability - away_probability
        if abs(probability_gap) > 0.12:
            score += 2 if probability_gap > 0 else -2
            drivers.append("predictor")
        elif abs(probability_gap) > 0.05:
            score += 1 if probability_gap > 0 else -1
            drivers.append("predictor")

    if score >= 4:
        label = "vantaggio chiaro casa"
        explanation = f"{home_team} parte avanti in modo netto per combinazione di matchup, contesto casa e segnali complessivi."
    elif score >= 2:
        label = "leggero vantaggio casa"
        explanation = f"{home_team} ha qualche indicatore in piu dalla sua, ma senza costruire un margine totale."
    elif score <= -4:
        label = "vantaggio chiaro trasferta"
        explanation = f"{away_team} ha un vantaggio leggibile anche fuori casa per qualita del profilo e segnali recenti."
    elif score <= -2:
        label = "leggero vantaggio trasferta"
        explanation = f"{away_team} sembra avere un piccolo margine nel matchup, pur restando dentro una partita aperta."
    else:
        label = "matchup equilibrato"
        explanation = "I principali indicatori si compensano abbastanza e non producono un vantaggio stilistico netto."

    if drivers:
        explanation = f"{explanation} I driver principali sono: {', '.join(dict.fromkeys(drivers))}."

    return {"label": label, "score": score, "drivers": list(dict.fromkeys(drivers)), "explanation": explanation}


def build_tactical_questions(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    predictor_context: dict[str, Any] | None = None,
) -> list[str]:
    home_team = str(home_profile.get("team") or "La squadra di casa")
    away_team = str(away_profile.get("team") or "La squadra ospite")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    questions: list[str] = []

    if (_safe_float(home_advanced.get("offensive_threat_index")) or 0.0) > (_safe_float(away_advanced.get("defensive_solidity_index")) or 0.0) + 10:
        questions.append(f"{home_team} riuscira a trasformare il suo vantaggio offensivo in occasioni pulite contro la difesa di {away_team}?")
    if (_safe_float(away_advanced.get("offensive_threat_index")) or 0.0) > (_safe_float(home_advanced.get("defensive_solidity_index")) or 0.0) + 10:
        questions.append(f"{away_team} riuscira a portare la propria produzione offensiva contro la struttura difensiva di {home_team}?")
    if (_safe_float(home_advanced.get("home_dependency_index")) or 0.0) >= 65:
        questions.append(f"Il vantaggio casa di {home_team} incidera davvero oppure il match uscira presto dal suo contesto ideale?")
    if (_safe_float(away_advanced.get("elo_rating")) or 0.0) > (_safe_float(home_advanced.get("elo_rating")) or 0.0) + 50:
        questions.append(f"La differenza Elo a favore di {away_team} si vedra anche nei dati recenti e nel ritmo della partita?")
    if abs((_safe_float(home_advanced.get("recent_momentum_index")) or 0.0) - (_safe_float(away_advanced.get("recent_momentum_index")) or 0.0)) > 10:
        questions.append("Il momento recente conferma o contraddice la classifica attuale?")

    predictor = predictor_context or {}
    if predictor.get("available"):
        home_probability = float(predictor.get("home_probability", 0.0))
        away_probability = float(predictor.get("away_probability", 0.0))
        if abs(home_probability - away_probability) < 0.08:
            questions.append("Perche il predictor vede una sfida aperta anche se alcuni segnali di profilo sembrano sbilanciati?")
        else:
            questions.append("Il vantaggio letto dal predictor verra confermato anche dal volume di gioco nei primi 60 minuti?")
    else:
        questions.append("Con pochi dati disponibili, quale squadra riuscira prima a imporre il proprio ritmo per chiarire il matchup?")

    if not questions:
        questions.append("Quale squadra riuscira a portare la partita piu vicino al proprio contesto ideale?")

    unique: list[str] = []
    for item in questions:
        if item not in unique:
            unique.append(item)
    return unique[:6]


def build_predictor_context(
    predictor: dict[str, Any],
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
) -> dict[str, Any]:
    home_team = str(home_profile.get("team") or "Casa")
    away_team = str(away_profile.get("team") or "Trasferta")

    if not predictor.get("ok"):
        return {
            "available": False,
            "message": predictor.get("message", "Predictor non disponibile per questa sfida."),
            "bullets": [
                "Il confronto resta leggibile con profili, forma, rendimento casa/fuori e metriche avanzate, ma senza un supporto probabilistico completo."
            ],
        }

    home_probability = float(predictor["probabilities"]["1"])
    draw_probability = float(predictor["probabilities"]["X"])
    away_probability = float(predictor["probabilities"]["2"])
    home_xg = float(predictor["expected_goals_home"])
    away_xg = float(predictor["expected_goals_away"])
    xg_gap = home_xg - away_xg
    home_factors = predictor.get("factors", {}).get("home_team", {})
    away_factors = predictor.get("factors", {}).get("away_team", {})
    league = predictor.get("factors", {}).get("league", {})

    bullets: list[str] = []
    if float(league.get("home_advantage", 1.0)) > 1.05:
        bullets.append(f"Il predictor incorpora un vantaggio casa di lega che spinge leggermente {home_team} nei contesti equilibrati.")
    if xg_gap > 0.30:
        bullets.append(f"Il modello vede piu produzione attesa per {home_team}, quindi le probabilita si inclinano verso la squadra di casa.")
    elif xg_gap < -0.30:
        bullets.append(f"Il modello riconosce piu produzione attesa per {away_team}, nonostante il fattore campo.")
    else:
        bullets.append("Gli expected goals del predictor sono vicini, quindi il modello legge una partita piu aperta del normale.")

    home_form_factor = _safe_float(home_factors.get("form_factor")) or 1.0
    away_form_factor = _safe_float(away_factors.get("form_factor")) or 1.0
    if abs(home_form_factor - away_form_factor) > 0.03:
        leader = home_team if home_form_factor > away_form_factor else away_team
        bullets.append(f"La forma recente stimata dal predictor favorisce {leader} e sposta un po il bilanciamento delle probabilita.")

    home_attack = _safe_float(home_factors.get("attack_strength")) or 1.0
    away_attack = _safe_float(away_factors.get("attack_strength")) or 1.0
    home_defense = _safe_float(home_factors.get("defense_strength")) or 1.0
    away_defense = _safe_float(away_factors.get("defense_strength")) or 1.0
    if home_attack > away_attack and away_defense > home_defense:
        bullets.append("Il modello tiene insieme attacco casa e difesa ospite: per questo il vantaggio resta misurato invece che estremo.")

    return {
        "available": True,
        "message": None,
        "home_probability": home_probability,
        "draw_probability": draw_probability,
        "away_probability": away_probability,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "xg_gap": xg_gap,
        "most_likely_score": predictor.get("most_likely_score"),
        "top_scores": predictor.get("top_scorelines", [])[:3],
        "bullets": bullets[:4],
    }


def build_matchup_summary(
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
    style_advantage: dict[str, Any],
    mismatches: list[str],
    predictor_context: dict[str, Any],
) -> str:
    home_team = str(home_profile.get("team") or "La squadra di casa")
    away_team = str(away_profile.get("team") or "La squadra ospite")
    home_advanced = home_profile.get("advanced_metrics", {})
    away_advanced = away_profile.get("advanced_metrics", {})

    lines = [
        f"{home_team} contro {away_team} e un matchup che va letto insieme su profilo, contesto e dati recenti.",
        f"La lettura rule-based indica {style_advantage['label']}: {style_advantage['explanation']}",
        (
            f"Sui dati interni, {home_team} ha pericolosita offensiva a {home_advanced.get('offensive_threat_index', 'n/d')}/100 "
            f"e {away_team} a {away_advanced.get('offensive_threat_index', 'n/d')}/100."
        ),
        (
            f"La solidita difensiva vale {home_advanced.get('defensive_solidity_index', 'n/d')}/100 per {home_team} "
            f"e {away_advanced.get('defensive_solidity_index', 'n/d')}/100 per {away_team}."
        ),
    ]

    if mismatches:
        lines.append(mismatches[0])
    if len(mismatches) > 1:
        lines.append(mismatches[1])

    if predictor_context.get("available"):
        lines.append(
            f"Il predictor assegna {predictor_context['home_probability'] * 100:.1f}% - {predictor_context['draw_probability'] * 100:.1f}% - "
            f"{predictor_context['away_probability'] * 100:.1f}% e vede come score piu probabile {predictor_context['most_likely_score']}."
        )
    else:
        lines.append("Il predictor non e disponibile in modo affidabile, quindi il peso dell'analisi resta ancora piu qualitativo.")

    lines.append(
        "La lettura prudente e questa: i dati indicano chi parte leggermente meglio e dove nascono i rischi, ma il matchup puo cambiare se il ritmo reale del match rompe i profili medi."
    )
    return "\n".join(lines[:8])


def build_matchup_analysis(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    ratings_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return {"ok": False, "message": "La stagione selezionata non contiene dati sufficienti per costruire il matchup."}

    if home_team == away_team:
        return {"ok": False, "message": "Seleziona due squadre diverse per analizzare il matchup."}

    teams = get_teams(prepared_df)
    if home_team not in teams or away_team not in teams:
        return {"ok": False, "message": "Una o entrambe le squadre non sono presenti nella stagione selezionata."}

    advanced_df = build_advanced_team_metrics(prepared_df, ratings_df=ratings_df)
    home_metrics = get_team_advanced_metrics(advanced_df, home_team) or {}
    away_metrics = get_team_advanced_metrics(advanced_df, away_team) or {}

    profile_context = build_team_profile_context(
        prepared_df,
        ratings_df=ratings_df,
        advanced_metrics_df=advanced_df,
    )
    home_profile = build_team_profile_with_ratings(
        prepared_df,
        home_team,
        ratings_df=ratings_df,
        advanced_metrics_df=advanced_df,
        context=profile_context,
    )
    away_profile = build_team_profile_with_ratings(
        prepared_df,
        away_team,
        ratings_df=ratings_df,
        advanced_metrics_df=advanced_df,
        context=profile_context,
    )
    if not home_profile.get("ok") or not away_profile.get("ok"):
        return {"ok": False, "message": "Dati insufficienti per costruire un profilo affidabile di entrambe le squadre."}

    if home_metrics:
        home_profile["advanced_metrics"] = home_metrics
        home_profile["rating"] = _rating_block_from_metrics(home_metrics)
    if away_metrics:
        away_profile["advanced_metrics"] = away_metrics
        away_profile["rating"] = _rating_block_from_metrics(away_metrics)

    predictor = predict_match(prepared_df, home_team, away_team, max_goals=6)
    predictor_context = build_predictor_context(predictor, home_profile, away_profile)
    comparison_rows = compare_advanced_metrics(home_metrics, away_metrics, home_team=home_team, away_team=away_team)
    mismatches = identify_key_mismatches(home_profile, away_profile, predictor_context=predictor_context, comparison_rows=comparison_rows)
    style_advantage = build_style_advantage(home_profile, away_profile, predictor)
    context_engine = build_context_adjusted_edge(
        home_profile,
        away_profile,
        predictor_context=predictor_context,
        mismatches=mismatches,
        style_advantage=style_advantage,
    )
    home_risks = build_home_team_risks(home_profile, away_profile, predictor_context=predictor_context)
    away_risks = build_away_team_risks(home_profile, away_profile, predictor_context=predictor_context)
    tactical_questions = build_tactical_questions(home_profile, away_profile, predictor_context=predictor_context)
    summary = build_matchup_summary(home_profile, away_profile, style_advantage, mismatches, predictor_context)

    warnings: list[str] = []
    if home_profile["general"]["matches"] < 6 or away_profile["general"]["matches"] < 6:
        warnings.append("Campione partite ancora ridotto: alcuni segnali del matchup possono essere piu instabili.")
    if not home_profile.get("rating", {}).get("available") or not away_profile.get("rating", {}).get("available"):
        warnings.append("Rating Elo non completo: il confronto resta valido ma con un layer storico in meno.")
    if not predictor_context.get("available"):
        warnings.append("Predictor non disponibile: la pagina mostra comunque un'analisi matchup parziale.")
    if not home_metrics or not away_metrics:
        warnings.append("Metriche avanzate non complete: alcuni mismatch sono letti soprattutto con profilo squadra e rendimento.")
    if float(context_engine.get("confidence", 0.0) or 0.0) < 45.0:
        warnings.append("Il contesto abbassa la confidenza della lettura: il matchup resta molto aperto o con segnali contrastanti.")

    return {
        "ok": True,
        "home_team": home_team,
        "away_team": away_team,
        "home_profile": home_profile,
        "away_profile": away_profile,
        "home_metrics": home_metrics,
        "away_metrics": away_metrics,
        "metric_comparison": comparison_rows,
        "predictor": predictor,
        "predictor_context": predictor_context,
        "context_engine": context_engine,
        "mismatches": mismatches,
        "style_advantage": style_advantage,
        "home_risks": home_risks,
        "away_risks": away_risks,
        "tactical_questions": tactical_questions,
        "summary": summary,
        "warnings": warnings,
    }
