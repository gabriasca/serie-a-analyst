from __future__ import annotations

from typing import Any

import pandas as pd

from src.advanced_metrics import build_advanced_team_metrics, get_team_advanced_metrics
from src.analytics import build_standings, compute_team_stats, get_team_match_log
from src.predictor import predict_match
from src.ratings import get_team_rating
from src.schedule_context import build_match_schedule_context


def _empty_split() -> dict[str, float | int | str]:
    return {
        "Contesto": "-",
        "GP": 0,
        "V": 0,
        "N": 0,
        "S": 0,
        "GF": 0,
        "GA": 0,
        "Pts": 0,
        "PPM": 0.0,
    }


def _get_split_row(split_df: pd.DataFrame, context: str) -> dict[str, float | int | str]:
    if split_df.empty:
        row = _empty_split()
        row["Contesto"] = context
        return row

    row_df = split_df.loc[split_df["Contesto"] == context]
    if row_df.empty:
        row = _empty_split()
        row["Contesto"] = context
        return row

    row = row_df.iloc[0].to_dict()
    row["GP"] = int(row.get("GP", 0) or 0)
    row["V"] = int(row.get("V", 0) or 0)
    row["N"] = int(row.get("N", 0) or 0)
    row["S"] = int(row.get("S", 0) or 0)
    row["GF"] = int(row.get("GF", 0) or 0)
    row["GA"] = int(row.get("GA", 0) or 0)
    row["Pts"] = int(row.get("Pts", 0) or 0)
    row["PPM"] = float(row.get("PPM", 0.0) or 0.0)
    return row


def _build_recent_form_block(df: pd.DataFrame, team: str, last_n: int = 5) -> dict[str, Any]:
    match_log = get_team_match_log(df, team)
    if match_log.empty:
        return {
            "matches": 0,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "form_string": "-",
            "table": pd.DataFrame(),
        }

    recent = match_log.tail(last_n).copy()
    recent["Data"] = recent["match_date"].dt.strftime("%Y-%m-%d")
    recent["Partita"] = recent["home_team"] + " " + recent["score"] + " " + recent["away_team"]
    recent["Esito"] = recent["display_result"]
    table = recent[["Data", "Partita", "venue", "Esito", "points"]].rename(
        columns={"venue": "Contesto", "points": "Punti"}
    )

    return {
        "matches": int(len(recent)),
        "points": int(recent["points"].sum()),
        "goals_for": int(recent["goals_for"].sum()),
        "goals_against": int(recent["goals_against"].sum()),
        "form_string": " ".join(recent["display_result"].tolist()) or "-",
        "table": table,
    }


def _build_position_map(standings: pd.DataFrame) -> dict[str, int]:
    if standings.empty:
        return {}

    return {str(row["Team"]): int(position) for position, row in standings.iterrows()}


def _classifica_note(
    team_count: int,
    home_team: str,
    away_team: str,
    home_position: int | None,
    away_position: int | None,
    home_points: int,
    away_points: int,
) -> str:
    if home_position is None or away_position is None:
        return "Classifica attuale non disponibile in modo completo per contestualizzare il match."

    points_gap = abs(home_points - away_points)
    relegation_cutoff = max(team_count - 2, 1)

    if home_position <= 4 and away_position <= 6:
        return (
            f"Entrambe sono in zona alta o a ridosso delle prime posizioni: una vittoria puo pesare "
            f"molto nella corsa europea e, in base al distacco, anche nella lotta di vertice."
        )

    if home_position >= relegation_cutoff or away_position >= relegation_cutoff:
        return (
            "La classifica suggerisce un peso importante in chiave salvezza: anche un pareggio puo "
            "muovere qualcosa, ma una vittoria cambierebbe di piu l'inerzia della zona bassa."
        )

    if points_gap <= 3:
        return (
            "Le due squadre sono vicine in classifica: il match puo spostare subito gli equilibri del "
            "confronto diretto e del gruppo che le circonda."
        )

    if home_points > away_points:
        return (
            f"{home_team} parte davanti in classifica, mentre {away_team} ha l'occasione di ridurre un "
            f"distacco di {points_gap} punti e riaprire il discorso."
        )

    return (
        f"{away_team} ha un margine di {points_gap} punti al momento: per {home_team} sarebbe un risultato "
        "pesante per accorciare e rendere piu compatta la fascia di classifica."
    )


def _safe_metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    value = metrics.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_advanced_comparison(df: pd.DataFrame, home_team: str, away_team: str) -> dict[str, Any]:
    metrics_df = build_advanced_team_metrics(df)
    home_metrics = get_team_advanced_metrics(metrics_df, home_team)
    away_metrics = get_team_advanced_metrics(metrics_df, away_team)
    if not home_metrics or not away_metrics:
        return {
            "available": False,
            "home": home_metrics,
            "away": away_metrics,
            "key_factors": [],
            "note": "Metriche avanzate non disponibili in modo completo per questo confronto.",
        }

    factors: list[str] = []

    offense_gap = _safe_metric_value(home_metrics, "offensive_threat_index")
    away_offense = _safe_metric_value(away_metrics, "offensive_threat_index")
    if offense_gap is not None and away_offense is not None and abs(offense_gap - away_offense) >= 6:
        leader = home_team if offense_gap > away_offense else away_team
        leader_value = max(offense_gap, away_offense)
        trailer = away_team if leader == home_team else home_team
        trailer_value = min(offense_gap, away_offense)
        factors.append(
            f"Pericolosita offensiva: {leader} arriva con un indice piu alto ({leader_value:.1f} contro {trailer_value:.1f}), "
            f"quindi parte con piu segnali di pressione offensiva rispetto a {trailer}."
        )

    home_solidity = _safe_metric_value(home_metrics, "defensive_solidity_index")
    away_solidity = _safe_metric_value(away_metrics, "defensive_solidity_index")
    if home_solidity is not None and away_solidity is not None and abs(home_solidity - away_solidity) >= 6:
        leader = home_team if home_solidity > away_solidity else away_team
        leader_value = max(home_solidity, away_solidity)
        trailer = away_team if leader == home_team else home_team
        trailer_value = min(home_solidity, away_solidity)
        factors.append(
            f"Solidita difensiva: {leader} offre una tenuta migliore ({leader_value:.1f} contro {trailer_value:.1f}), "
            f"aspetto che puo pesare nella gestione dei momenti sporchi della partita."
        )

    home_momentum = _safe_metric_value(home_metrics, "recent_momentum_index")
    away_momentum = _safe_metric_value(away_metrics, "recent_momentum_index")
    if home_momentum is not None and away_momentum is not None and abs(home_momentum - away_momentum) >= 6:
        leader = home_team if home_momentum > away_momentum else away_team
        leader_value = max(home_momentum, away_momentum)
        trailer = away_team if leader == home_team else home_team
        trailer_value = min(home_momentum, away_momentum)
        factors.append(
            f"Momento recente: {leader} entra con un indice migliore ({leader_value:.1f} contro {trailer_value:.1f}), "
            "quindi la forma delle ultime uscite tende a spingere il suo profilo."
        )

    if not factors:
        factors.append(
            "Le metriche avanzate leggono un confronto abbastanza equilibrato: non emerge un margine netto nei principali indicatori interni."
        )

    return {
        "available": True,
        "home": home_metrics,
        "away": away_metrics,
        "key_factors": factors[:3],
        "note": "Questi indicatori interni aiutano a leggere il match, ma non sono xG reali e non danno certezze.",
    }


def build_key_factors(report_data: dict[str, Any]) -> list[str]:
    home = report_data["home_team"]
    away = report_data["away_team"]
    home_recent = report_data["recent_form"]["home"]
    away_recent = report_data["recent_form"]["away"]
    home_general = report_data["general_performance"]["home"]
    away_general = report_data["general_performance"]["away"]
    advanced = report_data.get("advanced_metrics", {})
    schedule_context = report_data.get("schedule_context", {})

    factors: list[str] = []
    factors.extend(advanced.get("key_factors", [])[:3])
    if isinstance(schedule_context, dict) and schedule_context.get("available"):
        factors.append(schedule_context.get("summary", "Il calendario viene letto sulle partite disponibili nel database."))

    recent_gap = home_recent["points"] - away_recent["points"]
    if recent_gap > 0:
        factors.append(
            f"Forma recente: {home} arriva meglio, con {home_recent['points']} punti nelle ultime "
            f"{home_recent['matches']} partite contro i {away_recent['points']} di {away}."
        )
    elif recent_gap < 0:
        factors.append(
            f"Forma recente: {away} ha raccolto piu punti di {home} nelle ultime cinque "
            f"({away_recent['points']} contro {home_recent['points']})."
        )
    else:
        factors.append(
            f"Forma recente in equilibrio: entrambe hanno raccolto {home_recent['points']} punti "
            "nelle ultime uscite considerate."
        )

    attack_gap = home_general["avg_goals_for"] - away_general["avg_goals_for"]
    if attack_gap > 0.1:
        factors.append(
            f"Attacco: {home} produce di piu in media ({home_general['avg_goals_for']:.2f} gol a partita) "
            f"rispetto a {away} ({away_general['avg_goals_for']:.2f})."
        )
    elif attack_gap < -0.1:
        factors.append(
            f"Attacco: {away} ha numeri offensivi migliori ({away_general['avg_goals_for']:.2f} gol medi) "
            f"rispetto a {home} ({home_general['avg_goals_for']:.2f})."
        )
    else:
        factors.append(
            f"Attacco: i numeri offensivi sono vicini, con {home_general['avg_goals_for']:.2f} gol medi "
            f"per {home} e {away_general['avg_goals_for']:.2f} per {away}."
        )

    defense_gap = away_general["avg_goals_against"] - home_general["avg_goals_against"]
    if defense_gap > 0.1:
        factors.append(
            f"Difesa: {home} concede meno in media ({home_general['avg_goals_against']:.2f}) e arriva "
            f"con una struttura difensiva piu solida rispetto a {away} ({away_general['avg_goals_against']:.2f})."
        )
    elif defense_gap < -0.1:
        factors.append(
            f"Difesa: {away} ha tenuto numeri migliori senza palla ({away_general['avg_goals_against']:.2f} gol "
            f"subiti medi) rispetto a {home} ({home_general['avg_goals_against']:.2f})."
        )
    else:
        factors.append(
            f"Difesa: equilibrio anche nel dato dei gol concessi, con valori medi molto simili "
            f"({home_general['avg_goals_against']:.2f} contro {away_general['avg_goals_against']:.2f})."
        )

    home_relevant = home_general["relevant_split"]
    away_relevant = away_general["relevant_split"]
    if home_relevant["PPM"] > away_relevant["PPM"] + 0.2:
        factors.append(
            f"Contesto del match: il rendimento interno di {home} pesa, con {home_relevant['PPM']:.2f} punti "
            f"per gara in casa contro i {away_relevant['PPM']:.2f} di {away} in trasferta."
        )
    elif away_relevant["PPM"] > home_relevant["PPM"] + 0.2:
        factors.append(
            f"Contesto del match: {away} ha mostrato una tenuta esterna interessante "
            f"({away_relevant['PPM']:.2f} punti per gara fuori), abbastanza da limitare parte del fattore campo."
        )
    else:
        factors.append(
            f"Contesto del match: il rendimento casa/fuori non crea un divario netto "
            f"({home_relevant['PPM']:.2f} punti medi in casa per {home}, {away_relevant['PPM']:.2f} fuori per {away})."
        )

    prediction = report_data["prediction"]
    if prediction.get("ok"):
        probs = prediction["probabilities"]
        league = prediction["factors"]["league"]
        xg_gap = prediction["expected_goals_home"] - prediction["expected_goals_away"]
        if xg_gap > 0.35:
            factors.append(
                f"Il predictor accentua il vantaggio di {home}: stima {prediction['expected_goals_home']:.2f} xG "
                f"contro {prediction['expected_goals_away']:.2f} e assegna il {probs['1'] * 100:.1f}% alla vittoria interna."
            )
        elif xg_gap < -0.35:
            factors.append(
                f"Il predictor vede margini per {away}: {prediction['expected_goals_away']:.2f} xG stimati "
                f"contro {prediction['expected_goals_home']:.2f} per {home}, nonostante il fattore campo."
            )
        else:
            factors.append(
                f"Il predictor legge una sfida abbastanza aperta: vantaggio casa stimato a {league['home_advantage']:.2f}, "
                f"ma xG piuttosto vicini ({prediction['expected_goals_home']:.2f} contro {prediction['expected_goals_away']:.2f})."
            )
    else:
        factors.append(
            "Il predictor non ha dati sufficienti per una stima completa: il report va quindi letto soprattutto "
            "attraverso forma, classifica e rendimento medio."
        )

    points_gap = abs(home_general["points"] - away_general["points"])
    if points_gap >= 8:
        leader = home if home_general["points"] > away_general["points"] else away
        factors.append(
            f"Squilibrio di classifica: {leader} ha gia costruito un margine significativo "
            f"di {points_gap} punti sull'altra squadra."
        )

    return factors[:6]


def build_match_summary(report_data: dict[str, Any]) -> str:
    home = report_data["home_team"]
    away = report_data["away_team"]
    home_recent = report_data["recent_form"]["home"]
    away_recent = report_data["recent_form"]["away"]
    home_general = report_data["general_performance"]["home"]
    away_general = report_data["general_performance"]["away"]
    impact = report_data["table_context"]["note"]
    prediction = report_data["prediction"]
    advanced = report_data.get("advanced_metrics", {})
    schedule_context = report_data.get("schedule_context", {})

    form_leader = home if home_recent["points"] >= away_recent["points"] else away
    attack_leader = home if home_general["avg_goals_for"] >= away_general["avg_goals_for"] else away
    defense_leader = (
        home if home_general["avg_goals_against"] <= away_general["avg_goals_against"] else away
    )

    summary_lines = [
        f"{home} - {away} mette di fronte due squadre con profili statistici diversi ma leggibili nei dati della stagione {report_data['season']}.",
        f"La forma recente favorisce {form_leader}, che nelle ultime cinque ha fatto meglio in termini di punti raccolti.",
        f"Sul piano del volume offensivo il riferimento e {attack_leader}, mentre la tenuta difensiva migliore appartiene a {defense_leader}.",
        impact,
    ]
    if advanced.get("available"):
        summary_lines.append(
            f"Le metriche avanzate interne leggono pericolosita offensiva a "
            f"{advanced['home'].get('offensive_threat_index', 'n/d')}/100 per {home} e "
            f"{advanced['away'].get('offensive_threat_index', 'n/d')}/100 per {away}, con un confronto "
            f"anche su solidita difensiva e momento recente."
        )
    if isinstance(schedule_context, dict) and schedule_context.get("available"):
        summary_lines.append(schedule_context.get("summary", "Il contesto calendario viene letto sulle partite disponibili."))

    if prediction.get("ok"):
        probs = prediction["probabilities"]
        summary_lines.append(
            f"Il modello assegna {prediction['expected_goals_home']:.2f} xG a {home} e {prediction['expected_goals_away']:.2f} a {away}, "
            f"con un 1X2 di {probs['1'] * 100:.1f}% - {probs['X'] * 100:.1f}% - {probs['2'] * 100:.1f}%."
        )
        summary_lines.append(
            f"Il risultato singolo piu probabile e {prediction['most_likely_score']}, ma la distribuzione resta aperta e non suggerisce certezze."
        )
    else:
        summary_lines.append(
            "Il predictor non produce una stima affidabile per questa sfida, quindi l'interpretazione va tenuta piu prudente del solito."
        )

    summary_lines.append(
        "La lettura piu utile e quindi quella di un report di contesto: dati, tendenze e fattori di campo aiutano a orientarsi, non a chiudere il discorso."
    )

    return "\n".join(summary_lines[:6])


def build_match_report_data(
    df: pd.DataFrame,
    season: str,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    if df.empty:
        return {"ok": False, "message": "La stagione selezionata non contiene dati sufficienti per creare un report."}

    if home_team == away_team:
        return {"ok": False, "message": "Seleziona due squadre diverse per generare il report."}

    home_stats = compute_team_stats(df, home_team)
    away_stats = compute_team_stats(df, away_team)
    if not home_stats or not away_stats:
        return {"ok": False, "message": "Una o entrambe le squadre non hanno dati sufficienti nella stagione selezionata."}

    standings = build_standings(df)
    positions = _build_position_map(standings)
    team_count = int(len(standings))

    home_recent = _build_recent_form_block(df, home_team, last_n=5)
    away_recent = _build_recent_form_block(df, away_team, last_n=5)

    home_split = _get_split_row(home_stats["home_away_split"], "Casa")
    away_split = _get_split_row(away_stats["home_away_split"], "Trasferta")

    report_data: dict[str, Any] = {
        "ok": True,
        "season": season,
        "match_title": f"{home_team} - {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "match_count": int(len(df)),
        "team_count": team_count,
        "standings": standings,
        "recent_form": {
            "home": home_recent,
            "away": away_recent,
        },
        "general_performance": {
            "home": {
                **home_stats,
                "position": positions.get(home_team),
                "relevant_split": home_split,
            },
            "away": {
                **away_stats,
                "position": positions.get(away_team),
                "relevant_split": away_split,
            },
        },
        "prediction": predict_match(df, home_team, away_team, max_goals=6),
        "ratings": {
            "home": get_team_rating(home_team),
            "away": get_team_rating(away_team),
            "note": "Il rating e usato come indicatore di forza storica/recente, non come certezza.",
        },
        "advanced_metrics": _build_advanced_comparison(df, home_team, away_team),
        "schedule_context": build_match_schedule_context(df, home_team, away_team),
        "table_context": {
            "home_position": positions.get(home_team),
            "away_position": positions.get(away_team),
            "home_points": int(home_stats["points"]),
            "away_points": int(away_stats["points"]),
            "note": _classifica_note(
                team_count=team_count,
                home_team=home_team,
                away_team=away_team,
                home_position=positions.get(home_team),
                away_position=positions.get(away_team),
                home_points=int(home_stats["points"]),
                away_points=int(away_stats["points"]),
            ),
        },
    }

    report_data["key_factors"] = build_key_factors(report_data)
    report_data["summary"] = build_match_summary(report_data)
    return report_data
