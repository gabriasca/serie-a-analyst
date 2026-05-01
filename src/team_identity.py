from __future__ import annotations

from typing import Any

import pandas as pd

from src.advanced_metrics import build_advanced_team_metrics, get_team_advanced_metrics
from src.analytics import build_standings, get_teams, prepare_matches_dataframe
from src.ratings import build_strength_bucket_map
from src.team_profiles import build_team_profile_context, build_team_profile_with_ratings


RESULT_LABELS = {"W": "Vittorie", "D": "Pareggi", "L": "Sconfitte"}
BAND_LABELS = {
    "top": "vs top / fascia alta",
    "middle": "vs medio gruppo",
    "bottom": "vs ultime / fascia bassa",
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
    return round(float(numeric.mean()), 2)


def _safe_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return round(float(numeric.std(ddof=0)), 2)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _prepare_identity_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    prepared_df = df.copy()
    if "id" not in prepared_df.columns:
        prepared_df["id"] = range(1, len(prepared_df) + 1)
    for column in ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_corners", "away_corners"]:
        if column not in prepared_df.columns:
            prepared_df[column] = pd.NA
    return prepare_matches_dataframe(prepared_df)


def _result_and_points(goals_for: int, goals_against: int) -> tuple[str, int]:
    if goals_for > goals_against:
        return "W", 3
    if goals_for < goals_against:
        return "L", 0
    return "D", 1


def _build_team_log(df: pd.DataFrame, team: str, bucket_map: dict[str, str] | None = None) -> pd.DataFrame:
    prepared_df = _prepare_identity_df(df)
    bucket_map = bucket_map or {}
    records: list[dict[str, Any]] = []

    for row in prepared_df.to_dict(orient="records"):
        if row.get("home_team") == team:
            goals_for = int(row.get("home_goals", 0) or 0)
            goals_against = int(row.get("away_goals", 0) or 0)
            result, points = _result_and_points(goals_for, goals_against)
            opponent = str(row.get("away_team") or "")
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "opponent": opponent,
                    "opponent_band": bucket_map.get(opponent, "middle"),
                    "venue": "Casa",
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goals_for - goals_against,
                    "result": result,
                    "points": points,
                    "shots_for": _safe_float(row.get("home_shots")),
                    "shots_against": _safe_float(row.get("away_shots")),
                    "shots_on_target_for": _safe_float(row.get("home_shots_on_target")),
                    "shots_on_target_against": _safe_float(row.get("away_shots_on_target")),
                    "corners_for": _safe_float(row.get("home_corners")),
                    "score": f"{goals_for}-{goals_against}",
                }
            )
        elif row.get("away_team") == team:
            goals_for = int(row.get("away_goals", 0) or 0)
            goals_against = int(row.get("home_goals", 0) or 0)
            result, points = _result_and_points(goals_for, goals_against)
            opponent = str(row.get("home_team") or "")
            records.append(
                {
                    "match_date": row.get("match_date"),
                    "opponent": opponent,
                    "opponent_band": bucket_map.get(opponent, "middle"),
                    "venue": "Trasferta",
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goals_for - goals_against,
                    "result": result,
                    "points": points,
                    "shots_for": _safe_float(row.get("away_shots")),
                    "shots_against": _safe_float(row.get("home_shots")),
                    "shots_on_target_for": _safe_float(row.get("away_shots_on_target")),
                    "shots_on_target_against": _safe_float(row.get("home_shots_on_target")),
                    "corners_for": _safe_float(row.get("away_corners")),
                    "score": f"{int(row.get('home_goals', 0) or 0)}-{int(row.get('away_goals', 0) or 0)}",
                }
            )

    if not records:
        return pd.DataFrame()
    log_df = pd.DataFrame(records)
    log_df["match_date"] = pd.to_datetime(log_df["match_date"], errors="coerce")
    return log_df.sort_values("match_date").reset_index(drop=True)


def _build_bucket_context(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]], str]:
    prepared_df = _prepare_identity_df(df)
    if prepared_df.empty:
        return {}, {}, "classifica"
    advanced_df = build_advanced_team_metrics(prepared_df)
    enriched_standings = advanced_df.attrs.get("enriched_standings")
    if not isinstance(enriched_standings, pd.DataFrame) or enriched_standings.empty:
        enriched_standings = build_standings(prepared_df)
    return build_strength_bucket_map(enriched_standings)


def _band_counts_text(match_df: pd.DataFrame) -> list[str]:
    if match_df.empty:
        return []
    counts = match_df["opponent_band"].value_counts().to_dict()
    return [
        f"{BAND_LABELS.get(str(band), str(band))}: {int(count)}"
        for band, count in counts.items()
        if int(count) > 0
    ]


def _patterns_for_result(result_key: str, result_df: pd.DataFrame, total_matches: int) -> list[str]:
    if result_df.empty:
        return [f"Nessuna partita in categoria {RESULT_LABELS.get(result_key, result_key).lower()} nel campione disponibile."]

    matches = len(result_df)
    avg_for = _safe_mean(result_df["goals_for"]) or 0.0
    avg_against = _safe_mean(result_df["goals_against"]) or 0.0
    home_count = int((result_df["venue"] == "Casa").sum())
    away_count = int((result_df["venue"] == "Trasferta").sum())
    lines = [
        f"{matches} partite su {total_matches} ({matches / max(total_matches, 1) * 100:.1f}%).",
        f"Media gol: {avg_for:.2f} fatti, {avg_against:.2f} subiti.",
        f"Distribuzione campo: {home_count} in casa, {away_count} in trasferta.",
    ]
    band_lines = _band_counts_text(result_df)
    if band_lines:
        lines.append("Fasce avversarie: " + "; ".join(band_lines) + ".")

    if result_key == "W":
        if avg_for >= 2.0:
            lines.append("Pattern osservato: nelle vittorie la produzione gol e alta.")
        if avg_against <= 0.8:
            lines.append("Pattern osservato: molte vittorie arrivano con controllo del gol subito.")
    elif result_key == "L":
        if avg_against >= 1.8:
            lines.append("Pattern osservato: nelle sconfitte cresce molto il peso dei gol concessi.")
        if avg_for <= 1.0:
            lines.append("Pattern osservato: quando perde fatica spesso a segnare piu di una rete.")
    elif result_key == "D":
        total_goals_avg = avg_for + avg_against
        if total_goals_avg <= 2.0:
            lines.append("Pattern osservato: i pareggi tendono a essere partite a punteggio contenuto.")
        else:
            lines.append("Pattern osservato: i pareggi non sono solo partite chiuse, perche la media gol totale resta discreta.")

    return lines


def _result_table(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame(columns=["Data", "Avversario", "Casa/Fuori", "Risultato", "Gol Fatti", "Gol Subiti", "Fascia"])
    table = result_df.copy()
    table["Data"] = table["match_date"].dt.strftime("%Y-%m-%d")
    return table.rename(
        columns={
            "opponent": "Avversario",
            "venue": "Casa/Fuori",
            "score": "Risultato",
            "goals_for": "Gol Fatti",
            "goals_against": "Gol Subiti",
            "opponent_band": "Fascia",
        }
    )[["Data", "Avversario", "Casa/Fuori", "Risultato", "Gol Fatti", "Gol Subiti", "Fascia"]]


def _analyze_result_patterns(df: pd.DataFrame, team: str, result_key: str) -> dict[str, Any]:
    bucket_map, _, bucket_source = _build_bucket_context(df)
    team_log = _build_team_log(df, team, bucket_map=bucket_map)
    if team_log.empty:
        return {"matches": 0, "observed_patterns": [], "table": pd.DataFrame(), "bucket_source": bucket_source}

    result_df = team_log.loc[team_log["result"] == result_key].copy()
    return {
        "matches": int(len(result_df)),
        "share_pct": round(len(result_df) / max(len(team_log), 1) * 100.0, 1),
        "goals_for_avg": _safe_mean(result_df["goals_for"]),
        "goals_against_avg": _safe_mean(result_df["goals_against"]),
        "shots_for_avg": _safe_mean(result_df["shots_for"]),
        "shots_against_avg": _safe_mean(result_df["shots_against"]),
        "home_count": int((result_df["venue"] == "Casa").sum()) if not result_df.empty else 0,
        "away_count": int((result_df["venue"] == "Trasferta").sum()) if not result_df.empty else 0,
        "observed_patterns": _patterns_for_result(result_key, result_df, len(team_log)),
        "table": _result_table(result_df.tail(10)),
        "bucket_source": bucket_source,
    }


def analyze_win_patterns(df: pd.DataFrame, team: str) -> dict[str, Any]:
    return _analyze_result_patterns(df, team, "W")


def analyze_loss_patterns(df: pd.DataFrame, team: str) -> dict[str, Any]:
    return _analyze_result_patterns(df, team, "L")


def analyze_draw_patterns(df: pd.DataFrame, team: str) -> dict[str, Any]:
    return _analyze_result_patterns(df, team, "D")


def analyze_performance_by_opponent_band(df: pd.DataFrame, team: str) -> dict[str, Any]:
    bucket_map, bucket_teams, bucket_source = _build_bucket_context(df)
    team_log = _build_team_log(df, team, bucket_map=bucket_map)
    rows: list[dict[str, Any]] = []
    if team_log.empty:
        return {"rows": rows, "bucket_source": bucket_source, "bucket_teams": bucket_teams, "best_band": None, "weakest_band": None}

    for band in ["top", "middle", "bottom"]:
        band_df = team_log.loc[team_log["opponent_band"] == band]
        matches = int(len(band_df))
        points = int(band_df["points"].sum()) if matches else 0
        rows.append(
            {
                "band": band,
                "Fascia": BAND_LABELS[band],
                "Partite": matches,
                "Punti": points,
                "PPM": round(points / matches, 2) if matches else 0.0,
                "GF": int(band_df["goals_for"].sum()) if matches else 0,
                "GA": int(band_df["goals_against"].sum()) if matches else 0,
                "DR": int(band_df["goal_difference"].sum()) if matches else 0,
            }
        )

    playable_rows = [row for row in rows if row["Partite"] > 0]
    best_band = max(playable_rows, key=lambda row: row["PPM"]) if playable_rows else None
    weakest_band = min(playable_rows, key=lambda row: row["PPM"]) if playable_rows else None
    return {
        "rows": rows,
        "bucket_source": bucket_source,
        "bucket_teams": bucket_teams,
        "best_band": best_band,
        "weakest_band": weakest_band,
    }


def analyze_home_away_identity_shift(df: pd.DataFrame, team: str) -> dict[str, Any]:
    team_log = _build_team_log(df, team)
    if team_log.empty:
        return {"available": False, "note": "Dati insufficienti per leggere casa/fuori."}

    rows: list[dict[str, Any]] = []
    for venue in ["Casa", "Trasferta"]:
        venue_df = team_log.loc[team_log["venue"] == venue]
        matches = int(len(venue_df))
        points = int(venue_df["points"].sum()) if matches else 0
        rows.append(
            {
                "Contesto": venue,
                "Partite": matches,
                "Punti": points,
                "PPM": round(points / matches, 2) if matches else 0.0,
                "GF medi": _safe_mean(venue_df["goals_for"]),
                "GA medi": _safe_mean(venue_df["goals_against"]),
                "DR medio": _safe_mean(venue_df["goal_difference"]),
            }
        )

    home_ppm = rows[0]["PPM"]
    away_ppm = rows[1]["PPM"]
    gap = round(home_ppm - away_ppm, 2)
    if gap >= 0.6:
        note = "Ipotesi prudente: il rendimento sembra molto piu forte nel contesto casa."
    elif gap <= -0.3:
        note = "Ipotesi prudente: il rendimento esterno non e inferiore a quello interno nel campione attuale."
    else:
        note = "Ipotesi prudente: il rendimento casa/fuori appare abbastanza bilanciato."

    return {"available": True, "rows": rows, "ppm_gap": gap, "note": note}


def analyze_recent_trend_shift(df: pd.DataFrame, team: str) -> dict[str, Any]:
    team_log = _build_team_log(df, team)
    if team_log.empty:
        return {"available": False, "note": "Dati recenti non disponibili."}

    recent_df = team_log.tail(5)
    previous_df = team_log.iloc[-10:-5] if len(team_log) > 5 else pd.DataFrame()
    season_ppm = round(float(team_log["points"].sum()) / len(team_log), 2)
    recent_ppm = round(float(recent_df["points"].sum()) / max(len(recent_df), 1), 2)
    previous_ppm = round(float(previous_df["points"].sum()) / len(previous_df), 2) if not previous_df.empty else None
    recent_gd_avg = _safe_mean(recent_df["goal_difference"]) or 0.0

    if recent_ppm >= season_ppm + 0.35 and recent_gd_avg >= 0:
        trend = "in crescita"
    elif recent_ppm <= season_ppm - 0.35 and recent_gd_avg <= 0:
        trend = "in calo"
    else:
        trend = "stabile"

    return {
        "available": True,
        "trend": trend,
        "season_ppm": season_ppm,
        "recent_ppm": recent_ppm,
        "previous_ppm": previous_ppm,
        "recent_form": " ".join(recent_df["result"].map({"W": "V", "D": "N", "L": "S"}).tolist()) or "-",
        "recent_points": int(recent_df["points"].sum()),
        "recent_goals_for": int(recent_df["goals_for"].sum()),
        "recent_goals_against": int(recent_df["goals_against"].sum()),
        "note": f"Ipotesi prudente: trend recente {trend} rispetto alla media stagionale.",
    }


def analyze_volatility(df: pd.DataFrame, team: str) -> dict[str, Any]:
    home_away = analyze_home_away_identity_shift(df, team)
    recent = analyze_recent_trend_shift(df, team)
    team_log = _build_team_log(df, team)
    if team_log.empty:
        return {"available": False, "volatility_index": 0.0, "label": "non disponibile", "drivers": []}

    points_std = _safe_std(team_log["points"])
    gd_std = _safe_std(team_log["goal_difference"])
    gf_std = _safe_std(team_log["goals_for"])
    home_away_gap = abs(float(home_away.get("ppm_gap", 0.0) or 0.0))
    recent_gap = abs(float(recent.get("recent_ppm", 0.0) or 0.0) - float(recent.get("season_ppm", 0.0) or 0.0))
    index = round(
        _clamp(
            (points_std / 1.35) * 26.0
            + (gd_std / 2.2) * 28.0
            + (gf_std / 1.6) * 18.0
            + (home_away_gap / 1.4) * 16.0
            + (recent_gap / 1.4) * 12.0
        ),
        1,
    )

    if index >= 66:
        label = "alta volatilita"
    elif index >= 42:
        label = "volatilita media"
    else:
        label = "profilo stabile"

    drivers = [
        f"Deviazione punti partita: {points_std:.2f}.",
        f"Oscillazione differenza reti: {gd_std:.2f}.",
        f"Gap PPM casa/fuori: {home_away_gap:.2f}.",
        f"Scarto forma recente vs stagione: {recent_gap:.2f} PPM.",
    ]
    return {
        "available": True,
        "volatility_index": index,
        "label": label,
        "points_std": points_std,
        "goal_difference_std": gd_std,
        "home_away_gap": home_away_gap,
        "recent_gap": recent_gap,
        "drivers": drivers,
        "note": "Indicatore interno: misura oscillazioni di risultati, gol e rendimento casa/fuori.",
    }


def infer_prudent_style_hypotheses(
    profile: dict[str, Any],
    metrics: dict[str, Any],
    matchup_data: dict[str, Any],
) -> list[str]:
    hypotheses: list[str] = []
    general = profile.get("general", {})
    home_away = profile.get("home_away", {})
    band_best = matchup_data.get("best_band")
    band_weakest = matchup_data.get("weakest_band")
    volatility = profile.get("identity_volatility", {})
    recent = profile.get("identity_recent_trend", {})

    defensive_solidity = _safe_float(metrics.get("defensive_solidity_index"))
    offensive_threat = _safe_float(metrics.get("offensive_threat_index"))
    recent_momentum = _safe_float(metrics.get("recent_momentum_index"))
    home_gap = _safe_float(home_away.get("ppm_gap")) or 0.0
    overall_ppm = _safe_float(general.get("ppm")) or 0.0

    if defensive_solidity is not None and defensive_solidity >= 60:
        hypotheses.append("Squadra solida: gli indicatori interni difensivi sono sopra media.")
    if offensive_threat is not None and offensive_threat >= 60:
        hypotheses.append("Squadra con buona pericolosita offensiva: il profilo aggregato segnala produzione sopra media.")
    if volatility.get("volatility_index", 0.0) >= 66:
        hypotheses.append("Squadra volatile: risultati e differenza reti oscillano molto nel campione disponibile.")
    if home_gap >= 0.55:
        hypotheses.append("Squadra forte in casa: il rendimento interno supera nettamente quello esterno.")
    if band_best and band_best.get("band") == "bottom" and float(band_best.get("PPM", 0.0)) >= max(2.0, overall_ppm):
        hypotheses.append("Sembra piu efficace contro avversari di fascia bassa.")
    if band_weakest and band_weakest.get("band") == "top" and int(band_weakest.get("Partite", 0)) >= 3:
        hypotheses.append("Il profilo suggerisce maggiore difficolta contro squadre di fascia alta.")
    if recent.get("trend") == "in crescita" or (recent_momentum is not None and recent_momentum >= 60):
        hypotheses.append("Squadra in crescita: la forma recente migliora la valutazione rispetto alla media stagionale.")
    if recent.get("trend") == "in calo" or (recent_momentum is not None and recent_momentum <= 40):
        hypotheses.append("Squadra in calo: il momento recente abbassa la lettura complessiva.")
    if not hypotheses:
        hypotheses.append("Squadra equilibrata o ancora in definizione: i segnali disponibili non spingono verso un'etichetta forte.")

    hypotheses.append(
        "Non abbiamo dati evento sufficienti per affermare pressing alto, costruzione dal basso o altre scelte tattiche specifiche."
    )
    return list(dict.fromkeys(hypotheses))[:8]


def build_missing_data_notes(report: dict[str, Any]) -> list[str]:
    notes = [
        "Pressing e recuperi palla: servono eventi difensivi e posizione del recupero.",
        "Costruzione dal basso: servono sequenze di passaggio, zone di inizio azione e dati possesso.",
        "Lanci lunghi e gioco diretto: servono dati passaggi con lunghezza e direzione.",
        "Possesso e passaggi progressivi: non sono disponibili nel database attuale.",
        "Lineups, assenze e rotazioni: servono dati giocatori e formazioni ufficiali.",
        "Eventi shot-by-shot: servono coordinate, tipo tiro e contesto dell'occasione.",
    ]
    if report.get("data_quality_notes"):
        notes.extend(report["data_quality_notes"])
    return list(dict.fromkeys(notes))


def build_identity_summary(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        return "Dati insufficienti per costruire una sintesi di identita squadra."

    team = str(report.get("team"))
    general = report.get("observed_data", {}).get("general", {})
    metrics = report.get("internal_indicators", {})
    volatility = report.get("volatility", {})
    recent = report.get("recent_trend", {})
    bands = report.get("opponent_bands", {})
    hypotheses = report.get("prudent_hypotheses", [])

    best_band = bands.get("best_band", {}) or {}
    weakest_band = bands.get("weakest_band", {}) or {}
    lines = [
        f"{team} ha {general.get('points', 0)} punti in {general.get('matches', 0)} partite, con differenza reti {general.get('goal_difference', 0)}.",
        f"I dati osservati dicono {general.get('goals_for', 0)} gol fatti e {general.get('goals_against', 0)} subiti.",
        (
            f"Gli indicatori interni leggono pericolosita offensiva {metrics.get('offensive_threat_index', 'n/d')}/100 "
            f"e solidita difensiva {metrics.get('defensive_solidity_index', 'n/d')}/100."
        ),
        f"La stabilita e classificata come {volatility.get('label', 'n/d')} con indice {volatility.get('volatility_index', 'n/d')}/100.",
        f"Il trend recente e {recent.get('trend', 'n/d')}: forma {recent.get('recent_form', '-')} nelle ultime cinque.",
    ]
    if best_band:
        lines.append(f"Il rendimento migliore per fascia e {best_band.get('Fascia')} con {best_band.get('PPM')} PPM.")
    if weakest_band:
        lines.append(f"La fascia piu complicata fin qui e {weakest_band.get('Fascia')} con {weakest_band.get('PPM')} PPM.")
    if hypotheses:
        lines.append("Ipotesi prudente principale: " + hypotheses[0])
    lines.append(
        "La lettura resta condizionata dal fatto che il database descrive risultati e statistiche aggregate, non dettagli tattici evento-per-evento."
    )
    return "\n".join(lines[:10])


def build_team_identity_report(df: pd.DataFrame, team: str, schedule_df: pd.DataFrame | None = None) -> dict[str, Any]:
    prepared_df = _prepare_identity_df(df)
    if prepared_df.empty:
        return {"ok": False, "message": "La stagione selezionata non contiene partite utilizzabili.", "team": team}
    if team not in get_teams(prepared_df):
        return {"ok": False, "message": "La squadra selezionata non e presente nella stagione.", "team": team}

    schedule_source_df = schedule_df if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty else prepared_df
    advanced_df = build_advanced_team_metrics(prepared_df)
    ratings_df = advanced_df.attrs.get("ratings_df")
    context = build_team_profile_context(
        prepared_df,
        ratings_df=ratings_df,
        advanced_metrics_df=advanced_df,
        schedule_df=schedule_source_df,
    )
    profile = build_team_profile_with_ratings(
        prepared_df,
        team,
        ratings_df=ratings_df,
        advanced_metrics_df=advanced_df,
        context=context,
    )
    if not profile.get("ok"):
        return {"ok": False, "message": profile.get("message", "Profilo squadra non disponibile."), "team": team}

    metrics = get_team_advanced_metrics(advanced_df, team) or {}
    win_patterns = analyze_win_patterns(prepared_df, team)
    loss_patterns = analyze_loss_patterns(prepared_df, team)
    draw_patterns = analyze_draw_patterns(prepared_df, team)
    opponent_bands = analyze_performance_by_opponent_band(prepared_df, team)
    home_away_shift = analyze_home_away_identity_shift(prepared_df, team)
    recent_trend = analyze_recent_trend_shift(prepared_df, team)
    volatility = analyze_volatility(prepared_df, team)

    profile["identity_volatility"] = volatility
    profile["identity_recent_trend"] = recent_trend
    hypotheses = infer_prudent_style_hypotheses(profile, metrics, opponent_bands)
    data_quality_notes = profile.get("notes", [])

    report = {
        "ok": True,
        "team": team,
        "observed_data": {
            "general": profile.get("general", {}),
            "home_away": profile.get("home_away", {}),
            "recent": profile.get("recent", {}),
            "rating": profile.get("rating", {}),
        },
        "internal_indicators": metrics,
        "profile": profile,
        "win_patterns": win_patterns,
        "loss_patterns": loss_patterns,
        "draw_patterns": draw_patterns,
        "opponent_bands": opponent_bands,
        "home_away_shift": home_away_shift,
        "recent_trend": recent_trend,
        "schedule_context": profile.get("schedule_context", {}),
        "volatility": volatility,
        "prudent_hypotheses": hypotheses,
        "data_quality_notes": data_quality_notes,
    }
    report["missing_data_notes"] = build_missing_data_notes(report)
    report["summary"] = build_identity_summary(report)
    return report
