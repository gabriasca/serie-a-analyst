from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics import build_standings, get_teams, prepare_matches_dataframe
from src.config import DEFAULT_COMPETITION_CODE, DEFAULT_COMPETITION_NAME, FIXTURE_SEED_PATH
from src.forecast_context import build_contextual_forecast, summarize_forecast_delta
from src.matchup_analysis import build_matchup_analysis
from src.predictor import predict_match
from src.projections import infer_remaining_fixtures


FIXTURE_COLUMNS = [
    "season",
    "match_date",
    "matchday",
    "home_team",
    "away_team",
    "competition_code",
    "competition_name",
    "source_name",
    "source_url",
]
OUTCOME_KEYS = ("1", "X", "2")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_probabilities(probabilities: dict[str, Any] | None) -> dict[str, float]:
    probabilities = probabilities or {}
    normalized = {key: max(_safe_float(probabilities.get(key)), 0.0) for key in OUTCOME_KEYS}
    total = sum(normalized.values())
    if total <= 0:
        return {key: 0.0 for key in OUTCOME_KEYS}
    return {key: normalized[key] / total for key in OUTCOME_KEYS}


def _favorite_from_probabilities(probabilities: dict[str, Any] | None) -> str | None:
    normalized = _normalize_probabilities(probabilities)
    if sum(normalized.values()) <= 0:
        return None
    return max(OUTCOME_KEYS, key=lambda key: normalized.get(key, 0.0))


def _probability_gap(probabilities: dict[str, Any] | None) -> float | None:
    normalized = _normalize_probabilities(probabilities)
    values = sorted(normalized.values(), reverse=True)
    if len(values) < 2 or values[0] <= 0:
        return None
    return values[0] - values[1]


def _match_title(home_team: str, away_team: str) -> str:
    return f"{home_team} - {away_team}"


def _format_matchday(value: Any) -> str:
    matchday = _safe_int(value)
    if matchday is None:
        return "Prossime partite"
    return f"Giornata {matchday}"


def _played_fixture_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return set()
    return {
        (str(row.home_team).strip(), str(row.away_team).strip())
        for row in prepared_df.itertuples(index=False)
        if str(row.home_team).strip() and str(row.away_team).strip()
    }


def _ensure_fixture_columns(df: pd.DataFrame) -> pd.DataFrame:
    fixtures_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    attrs = dict(getattr(fixtures_df, "attrs", {}) or {})
    for column in FIXTURE_COLUMNS:
        if column not in fixtures_df.columns:
            fixtures_df[column] = None

    optional_columns = [column for column in ["fixture_source", "fixture_note"] if column in fixtures_df.columns]
    fixtures_df = fixtures_df[FIXTURE_COLUMNS + optional_columns].copy()
    fixtures_df["home_team"] = fixtures_df["home_team"].astype(str).str.strip()
    fixtures_df["away_team"] = fixtures_df["away_team"].astype(str).str.strip()
    invalid_names = {"", "nan", "none", "<na>", "nat"}
    fixtures_df = fixtures_df[
        ~fixtures_df["home_team"].str.lower().isin(invalid_names)
        & ~fixtures_df["away_team"].str.lower().isin(invalid_names)
        & (fixtures_df["home_team"] != fixtures_df["away_team"])
    ].copy()
    fixtures_df["match_date"] = pd.to_datetime(fixtures_df["match_date"], errors="coerce")
    fixtures_df["matchday"] = pd.to_numeric(fixtures_df["matchday"], errors="coerce").astype("Int64")
    fixtures_df["competition_code"] = fixtures_df["competition_code"].fillna(DEFAULT_COMPETITION_CODE)
    fixtures_df["competition_name"] = fixtures_df["competition_name"].fillna(DEFAULT_COMPETITION_NAME)
    fixtures_df["source_name"] = fixtures_df["source_name"].fillna("fixture_seed")
    fixtures_df = fixtures_df.reset_index(drop=True)
    fixtures_df.attrs.update(attrs)
    return fixtures_df


def load_fixture_seed(path: str | Path | None = None) -> pd.DataFrame:
    seed_path = Path(path) if path is not None else FIXTURE_SEED_PATH
    if not seed_path.exists():
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    try:
        seed_df = pd.read_csv(seed_path)
    except Exception:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    if seed_df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    return _ensure_fixture_columns(seed_df)


def build_fixture_seed_report(season: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    seed_path = Path(path) if path is not None else FIXTURE_SEED_PATH
    path_exists = seed_path.exists()
    fixtures_df = load_fixture_seed(seed_path)
    if season and not fixtures_df.empty and "season" in fixtures_df.columns:
        fixtures_df = fixtures_df[
            fixtures_df["season"].isna()
            | (fixtures_df["season"].astype(str).str.strip() == str(season).strip())
        ].copy()
    fixtures_df = _sort_fixtures(fixtures_df)

    next_date = None
    if not fixtures_df.empty and "match_date" in fixtures_df.columns:
        dates = pd.to_datetime(fixtures_df["match_date"], errors="coerce").dropna()
        if not dates.empty:
            next_date = dates.min().strftime("%Y-%m-%d")

    next_matchday = None
    if not fixtures_df.empty and "matchday" in fixtures_df.columns:
        matchdays = pd.to_numeric(fixtures_df["matchday"], errors="coerce").dropna()
        if not matchdays.empty:
            next_matchday = int(matchdays.min())

    source_names: list[str] = []
    if not fixtures_df.empty and "source_name" in fixtures_df.columns:
        source_names = sorted(
            {
                str(source).strip()
                for source in fixtures_df["source_name"].dropna().tolist()
                if str(source).strip()
            }
        )

    return {
        "path": str(seed_path),
        "path_exists": bool(path_exists),
        "available": bool(path_exists and not fixtures_df.empty),
        "fixture_count": int(len(fixtures_df)),
        "next_fixture_date": next_date,
        "next_matchday": next_matchday,
        "source_names": source_names,
    }


def _sort_fixtures(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    if fixtures_df.empty:
        return fixtures_df.copy()
    sortable = fixtures_df.copy()
    sortable["_date_sort"] = pd.to_datetime(sortable["match_date"], errors="coerce")
    sortable["_matchday_sort"] = pd.to_numeric(sortable["matchday"], errors="coerce")
    sortable = sortable.sort_values(
        ["_matchday_sort", "_date_sort", "home_team", "away_team"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    return sortable.drop(columns=["_date_sort", "_matchday_sort"]).reset_index(drop=True)


def _fixture_attrs(source_mode: str, source_label: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "fixture_source": source_mode,
        "source_label": source_label,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _filter_seed_fixtures(seed_df: pd.DataFrame, df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    if seed_df.empty:
        return seed_df

    played_pairs = _played_fixture_pairs(df)
    filtered = seed_df.copy()
    if season and "season" in filtered.columns:
        filtered = filtered[
            filtered["season"].isna()
            | (filtered["season"].astype(str).str.strip() == str(season).strip())
        ].copy()
    if "competition_code" in filtered.columns:
        filtered = filtered[
            filtered["competition_code"].isna()
            | (filtered["competition_code"].astype(str).str.strip() == DEFAULT_COMPETITION_CODE)
        ].copy()

    filtered = filtered[
        ~filtered.apply(lambda row: (str(row["home_team"]), str(row["away_team"])) in played_pairs, axis=1)
    ].copy()
    return _sort_fixtures(filtered)


def _infer_round_from_missing_fixtures(df: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    remaining_df = infer_remaining_fixtures(prepared_df)
    if remaining_df.empty:
        result = pd.DataFrame(columns=FIXTURE_COLUMNS)
        result.attrs.update(
            _fixture_attrs(
                "inferred_missing",
                "Partite mancanti inferite",
                ["Non risultano partite mancanti inferibili dal calendario home/away."],
            )
        )
        return result

    teams = get_teams(prepared_df)
    target_count = max(len(teams) // 2, 1)
    selected_rows: list[dict[str, Any]] = []
    used_teams: set[str] = set()
    for row in remaining_df.to_dict(orient="records"):
        home_team = str(row.get("home_team", "")).strip()
        away_team = str(row.get("away_team", "")).strip()
        if not home_team or not away_team:
            continue
        if home_team in used_teams or away_team in used_teams:
            continue
        selected_rows.append(
            {
                "season": season,
                "match_date": None,
                "matchday": None,
                "home_team": home_team,
                "away_team": away_team,
                "competition_code": DEFAULT_COMPETITION_CODE,
                "competition_name": DEFAULT_COMPETITION_NAME,
                "source_name": "fixture_inferite",
                "source_url": None,
            }
        )
        used_teams.update([home_team, away_team])
        if len(selected_rows) >= target_count:
            break

    result = _ensure_fixture_columns(pd.DataFrame(selected_rows))
    result["fixture_source"] = "inferred_missing"
    result["fixture_note"] = "Le partite sono inferite, non confermate da calendario ufficiale."
    result.attrs.update(
        _fixture_attrs(
            "inferred_missing",
            "Partite mancanti inferite",
            ["Le partite sono inferite, non confermate da calendario ufficiale."],
        )
    )
    return result


def infer_next_round_fixtures(
    df: pd.DataFrame,
    season: str | None = None,
    fixture_seed_path: str | Path | None = None,
) -> pd.DataFrame:
    seed_df = load_fixture_seed(fixture_seed_path)
    if not seed_df.empty:
        seed_fixtures = _filter_seed_fixtures(seed_df, df, season)
        if not seed_fixtures.empty:
            seed_fixtures["fixture_source"] = "fixture_seed"
            seed_fixtures["fixture_note"] = "Partite future lette da fixture seed."
            seed_fixtures.attrs.update(
                _fixture_attrs(
                    "fixture_seed",
                    "Fixture seed",
                    [],
                )
            )
            return seed_fixtures.reset_index(drop=True)

    inferred = _infer_round_from_missing_fixtures(df, season=season)
    if seed_df.empty:
        inferred.attrs["warnings"] = list(
            dict.fromkeys(
                inferred.attrs.get("warnings", [])
                + ["Fixture seed non presente: uso partite mancanti inferite dal calendario home/away."]
            )
        )
    else:
        inferred.attrs["warnings"] = list(
            dict.fromkeys(
                inferred.attrs.get("warnings", [])
                + ["Fixture seed presente ma senza partite future utilizzabili: uso fallback inferito."]
            )
        )
    return inferred


def available_fixture_matchdays(fixtures_df: pd.DataFrame) -> list[int]:
    if fixtures_df.empty or "matchday" not in fixtures_df.columns:
        return []
    matchdays = pd.to_numeric(fixtures_df["matchday"], errors="coerce").dropna()
    return sorted({int(value) for value in matchdays.tolist()})


def select_round_fixtures(
    fixtures_df: pd.DataFrame,
    matchday: int | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    if fixtures_df.empty:
        return fixtures_df.copy()

    selected = _sort_fixtures(fixtures_df)
    if matchday is not None and "matchday" in selected.columns:
        selected = selected[pd.to_numeric(selected["matchday"], errors="coerce") == int(matchday)].copy()
    elif available_fixture_matchdays(selected):
        first_matchday = available_fixture_matchdays(selected)[0]
        selected = selected[pd.to_numeric(selected["matchday"], errors="coerce") == first_matchday].copy()
    else:
        all_teams = set(selected["home_team"].astype(str).tolist()) | set(selected["away_team"].astype(str).tolist())
        target_count = max_matches or max(len(all_teams) // 2, 1)
        selected_rows: list[dict[str, Any]] = []
        used_teams: set[str] = set()
        for row in selected.to_dict(orient="records"):
            home_team = str(row.get("home_team") or "")
            away_team = str(row.get("away_team") or "")
            if home_team in used_teams or away_team in used_teams:
                continue
            selected_rows.append(row)
            used_teams.update([home_team, away_team])
            if len(selected_rows) >= target_count:
                break
        selected = pd.DataFrame(selected_rows)

    if max_matches is not None:
        selected = selected.head(max_matches).copy()
    selected.attrs.update(fixtures_df.attrs)
    return selected.reset_index(drop=True)


def classify_match_volatility(draw_risk: float | None, upset_risk: float | None, confidence: float | None) -> str:
    draw_risk = _safe_float(draw_risk, 50.0)
    upset_risk = _safe_float(upset_risk, 50.0)
    confidence = _safe_float(confidence, 50.0)
    if confidence < 42 or draw_risk >= 68 or upset_risk >= 68:
        return "alta"
    if confidence < 58 or draw_risk >= 56 or upset_risk >= 56:
        return "media"
    return "bassa"


def _volatility_score(draw_risk: float | None, upset_risk: float | None, confidence: float | None) -> float:
    draw_risk = _safe_float(draw_risk, 50.0)
    upset_risk = _safe_float(upset_risk, 50.0)
    confidence = _safe_float(confidence, 50.0)
    return round(max(draw_risk, upset_risk) * 0.65 + (100.0 - confidence) * 0.35, 1)


def classify_match_interest(
    base_probabilities: dict[str, Any] | None,
    contextual_probabilities: dict[str, Any] | None,
    draw_risk: float | None,
    upset_risk: float | None,
    confidence: float | None,
) -> str:
    contextual_gap = _probability_gap(contextual_probabilities)
    volatility = classify_match_volatility(draw_risk, upset_risk, confidence)
    if contextual_gap is not None and contextual_gap <= 0.07:
        return "molto equilibrata"
    if _safe_float(draw_risk, 0.0) >= 62:
        return "alto rischio pareggio"
    if _safe_float(upset_risk, 0.0) >= 62:
        return "alto rischio upset"
    if volatility == "alta":
        return "partita volatile"
    if contextual_gap is not None and contextual_gap >= 0.18:
        return "favorita chiara"
    return "lettura aperta"


def classify_match_type(
    base_probabilities: dict[str, Any] | None,
    contextual_probabilities: dict[str, Any] | None,
    draw_risk: float | None,
    upset_risk: float | None,
    confidence: float | None,
) -> str:
    contextual_gap = _probability_gap(contextual_probabilities)
    if _safe_float(draw_risk, 0.0) >= 65:
        return "alto rischio pareggio"
    if _safe_float(upset_risk, 0.0) >= 65:
        return "alto rischio upset"
    if _safe_float(confidence, 50.0) < 45:
        return "bassa confidenza"
    if contextual_gap is not None and contextual_gap <= 0.08:
        return "equilibrata"
    if contextual_gap is not None and contextual_gap >= 0.20 and _safe_float(upset_risk, 0.0) >= 55:
        return "favorita chiara ma attenzione al contesto"
    if _safe_float(confidence, 50.0) >= 65 and _safe_float(draw_risk, 0.0) < 56 and _safe_float(upset_risk, 0.0) < 56:
        return "lettura stabile"
    return "matchup aperto"


def _standings_lookup(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    standings = build_standings(df)
    if standings.empty:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for position, row in standings.iterrows():
        lookup[str(row["Team"])] = {
            "position": int(position),
            "points": int(row.get("Pts", 0)),
            "matches": int(row.get("GP", 0)),
            "goal_difference": int(row.get("DR", 0)),
            "form": row.get("Forma ultime 5", "-"),
        }
    return lookup


def _context_factors_from_engine(context_engine: dict[str, Any]) -> list[str]:
    factors = context_engine.get("weighted_factors", [])
    if not isinstance(factors, list):
        return []
    lines: list[str] = []
    for factor in factors[:5]:
        if not isinstance(factor, dict):
            continue
        label = str(factor.get("label") or factor.get("factor") or "Fattore")
        impact = _safe_float(factor.get("weighted_impact"))
        note = str(factor.get("note") or "").strip()
        direction = "casa" if impact > 0.05 else "trasferta" if impact < -0.05 else "neutro"
        if note:
            lines.append(f"{label}: segnale {direction} ({impact:+.2f}). {note}")
        else:
            lines.append(f"{label}: segnale {direction} ({impact:+.2f}).")
    return lines


def _build_key_factors(match_analysis: dict[str, Any]) -> list[str]:
    matchup = match_analysis.get("matchup_analysis", {}) or {}
    contextual = match_analysis.get("contextual_forecast", {}) or {}
    prediction = match_analysis.get("prediction", {}) or {}
    home_team = str(match_analysis.get("home_team") or "Casa")
    away_team = str(match_analysis.get("away_team") or "Trasferta")

    factors: list[str] = []
    if prediction.get("ok"):
        favorite = _favorite_from_probabilities(prediction.get("probabilities"))
        if favorite == "1":
            factors.append(f"Predictor base: leggero vantaggio iniziale per {home_team}.")
        elif favorite == "2":
            factors.append(f"Predictor base: il modello vede {away_team} competitiva anche fuori casa.")
        elif favorite == "X":
            factors.append("Predictor base: il pareggio ha un peso rilevante nella distribuzione.")

    context_engine = matchup.get("context_engine", {})
    if isinstance(context_engine, dict):
        factors.extend(_context_factors_from_engine(context_engine))

    mismatches = matchup.get("mismatches", [])
    if isinstance(mismatches, list):
        factors.extend(str(item) for item in mismatches[:2])

    schedule_context = matchup.get("schedule_context", {})
    if isinstance(schedule_context, dict) and schedule_context.get("available"):
        rest_advantage = _safe_float(schedule_context.get("rest_advantage"))
        if rest_advantage > 1:
            factors.append(f"Calendario/riposo: {home_team} arriva con piu giorni di recupero disponibili.")
        elif rest_advantage < -1:
            factors.append(f"Calendario/riposo: {away_team} arriva con piu giorni di recupero disponibili.")
        else:
            factors.append("Calendario/riposo: non emerge un vantaggio netto dai dati disponibili.")

    factors.extend(str(item) for item in contextual.get("key_adjustments", [])[:2])
    unique = [item for item in dict.fromkeys(factors) if item]
    return unique[:6]


def build_match_turning_points(match_analysis: dict[str, Any]) -> list[str]:
    contextual = match_analysis.get("contextual_forecast", {}) or {}
    fixture_source = match_analysis.get("fixture_source")
    volatility = str(match_analysis.get("volatility") or "media")
    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)

    points: list[str] = []
    if confidence < 45:
        points.append("Confidenza bassa: i segnali disponibili sono contrastanti o poco stabili.")
    if draw_risk >= 58:
        points.append("Draw risk elevato: il match puo restare bloccato piu a lungo del previsto.")
    if upset_risk >= 58:
        points.append("Upset risk elevato: il favorito contestuale non ha un margine abbastanza solido.")
    if volatility == "alta":
        points.append("Profilo volatile: piccoli episodi possono cambiare molto la lettura statistica.")
    if fixture_source == "inferred_missing":
        points.append("Calendario inferito: l'ordine reale della prossima giornata potrebbe essere diverso.")

    warnings = contextual.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings[:2]:
            if "calendario" in str(warning).lower() or "rating" in str(warning).lower():
                points.append(str(warning))

    points.append("Lineup, assenze e gestione reale della partita non sono presenti nel database.")
    return list(dict.fromkeys(points))[:5]


def build_missing_data_notes(match_analysis: dict[str, Any]) -> list[str]:
    notes = [
        "Non abbiamo lineup ufficiali pre-partita.",
        "Non abbiamo infortuni o squalifiche aggiornate.",
        "Non abbiamo dati tattici live o eventi completi della partita.",
        "Non abbiamo xG reali shot-by-shot: i gol attesi mostrati sono del modello interno.",
    ]
    if match_analysis.get("fixture_source") == "inferred_missing":
        notes.append("Le partite inferite non sono confermate da calendario ufficiale.")
    matchup = match_analysis.get("matchup_analysis", {}) or {}
    schedule_context = matchup.get("schedule_context", {})
    if isinstance(schedule_context, dict):
        audit = schedule_context.get("competition_audit", {})
        if isinstance(audit, dict) and audit.get("only_league_data"):
            notes.append("Il carico calendario oggi e basato solo sulle partite disponibili, probabilmente solo campionato.")
    return list(dict.fromkeys(notes))


def build_match_narrative(match_analysis: dict[str, Any]) -> str:
    home_team = str(match_analysis.get("home_team") or "La squadra di casa")
    away_team = str(match_analysis.get("away_team") or "la squadra ospite")
    prediction = match_analysis.get("prediction", {}) or {}
    contextual = match_analysis.get("contextual_forecast", {}) or {}
    matchup = match_analysis.get("matchup_analysis", {}) or {}
    base_probabilities = prediction.get("probabilities", {})
    contextual_probabilities = contextual.get("contextual_probabilities", {})

    lines: list[str] = []
    if prediction.get("ok"):
        base_favorite = _favorite_from_probabilities(base_probabilities)
        if base_favorite == "1":
            lines.append(f"Il modello base parte da un vantaggio per {home_team}, sostenuto dal fattore campo e dalla produzione stimata.")
        elif base_favorite == "2":
            lines.append(f"Il modello base vede {away_team} leggermente avanti, nonostante giochi in trasferta.")
        else:
            lines.append("Il modello base legge una partita equilibrata, con il pareggio molto vicino agli altri esiti.")
        lines.append(
            f"I gol attesi modello sono {prediction.get('expected_goals_home', 0):.2f} per {home_team} "
            f"e {prediction.get('expected_goals_away', 0):.2f} per {away_team}."
        )
    else:
        lines.append("Il predictor base non ha abbastanza dati stabili per una previsione numerica completa.")

    contextual_favorite = _favorite_from_probabilities(contextual_probabilities)
    if contextual_favorite == "1":
        lines.append(f"La lettura contestuale conferma o avvicina il vantaggio verso {home_team}.")
    elif contextual_favorite == "2":
        lines.append(f"La lettura contestuale sposta attenzione verso {away_team}, soprattutto se i segnali recenti reggono.")
    elif contextual_favorite == "X":
        lines.append("Il contesto aumenta il peso di una gara bloccata o molto equilibrata.")

    style_advantage = matchup.get("style_advantage", {})
    if isinstance(style_advantage, dict) and style_advantage.get("label"):
        lines.append(f"Vantaggio stilistico: {style_advantage.get('label')}. {style_advantage.get('explanation', '')}")

    mismatches = matchup.get("mismatches", [])
    if isinstance(mismatches, list) and mismatches:
        lines.append(str(mismatches[0]))

    draw_risk = _safe_float(contextual.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual.get("confidence"), 50.0)
    volatility = match_analysis.get("volatility", classify_match_volatility(draw_risk, upset_risk, confidence))
    if draw_risk >= 58:
        lines.append("Il rischio pareggio e sopra la media: la partita puo svilupparsi su margini sottili.")
    if upset_risk >= 58:
        lines.append("Il rischio upset suggerisce prudenza: il favorito non ha un vantaggio blindato.")
    lines.append(f"La volatilita stimata e {volatility}, con confidenza contestuale {confidence:.1f}/100.")
    lines.append("La lettura resta condizionale: episodi, scelte iniziali e assenze non osservate possono cambiare il quadro.")
    return "\n".join(lines[:10])


def _build_contextual_fallback(
    prediction: dict[str, Any],
    matchup_analysis: dict[str, Any],
) -> dict[str, Any]:
    base_probabilities = _normalize_probabilities(prediction.get("probabilities") if prediction.get("ok") else {})
    context_engine = matchup_analysis.get("context_engine", {}) if isinstance(matchup_analysis, dict) else {}
    return {
        "ok": False,
        "base_probabilities": base_probabilities,
        "contextual_probabilities": base_probabilities,
        "probability_deltas": summarize_forecast_delta(base_probabilities, base_probabilities),
        "base_most_likely_score": prediction.get("most_likely_score"),
        "adjusted_edge": _safe_float(context_engine.get("adjusted_edge")),
        "draw_risk": _safe_float(context_engine.get("draw_risk"), 50.0),
        "upset_risk": _safe_float(context_engine.get("upset_risk"), 50.0),
        "confidence": _safe_float(context_engine.get("confidence"), 50.0),
        "confidence_label": "non disponibile",
        "key_adjustments": [],
        "warnings": [prediction.get("message") or "Predictor base non disponibile per la lettura contestuale."],
        "contextual_interpretation": "Lettura contestuale numerica non disponibile: servono probabilita base valide.",
    }


def build_round_match_analysis(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    season: str | None = None,
    schedule_df: pd.DataFrame | None = None,
    fixture_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    schedule_df = schedule_df if isinstance(schedule_df, pd.DataFrame) and not schedule_df.empty else prepared_df
    fixture_row = fixture_row or {}
    fixture_source = str(fixture_row.get("fixture_source") or "manual")

    prediction = predict_match(prepared_df, home_team, away_team, max_goals=6)
    try:
        matchup_analysis = build_matchup_analysis(prepared_df, home_team, away_team, schedule_df=schedule_df)
    except Exception as exc:
        matchup_analysis = {"ok": False, "message": f"Matchup Analysis non disponibile: {exc}"}

    if prediction.get("ok"):
        contextual_forecast = build_contextual_forecast(prediction, matchup_analysis=matchup_analysis)
    else:
        contextual_forecast = _build_contextual_fallback(prediction, matchup_analysis)

    standings = _standings_lookup(prepared_df)
    draw_risk = _safe_float(contextual_forecast.get("draw_risk"), 50.0)
    upset_risk = _safe_float(contextual_forecast.get("upset_risk"), 50.0)
    confidence = _safe_float(contextual_forecast.get("confidence"), 50.0)
    volatility = classify_match_volatility(draw_risk, upset_risk, confidence)
    match_type = classify_match_type(
        prediction.get("probabilities", {}),
        contextual_forecast.get("contextual_probabilities", {}),
        draw_risk,
        upset_risk,
        confidence,
    )
    interest = classify_match_interest(
        prediction.get("probabilities", {}),
        contextual_forecast.get("contextual_probabilities", {}),
        draw_risk,
        upset_risk,
        confidence,
    )

    result = {
        "ok": bool(prediction.get("ok")) or bool(matchup_analysis.get("ok")),
        "season": season,
        "match_date": fixture_row.get("match_date"),
        "matchday": fixture_row.get("matchday"),
        "home_team": home_team,
        "away_team": away_team,
        "match_title": _match_title(home_team, away_team),
        "fixture_source": fixture_source,
        "fixture_note": fixture_row.get("fixture_note"),
        "prediction": prediction,
        "matchup_analysis": matchup_analysis,
        "contextual_forecast": contextual_forecast,
        "standings": {
            "home": standings.get(home_team, {}),
            "away": standings.get(away_team, {}),
        },
        "volatility": volatility,
        "volatility_score": _volatility_score(draw_risk, upset_risk, confidence),
        "interest": interest,
        "match_type": match_type,
    }
    result["key_factors"] = _build_key_factors(result)
    result["turning_points"] = build_match_turning_points(result)
    result["missing_data_notes"] = build_missing_data_notes(result)
    result["narrative"] = build_match_narrative(result)
    return result


def _summary_row(match_analysis: dict[str, Any]) -> dict[str, Any]:
    prediction = match_analysis.get("prediction", {}) or {}
    contextual = match_analysis.get("contextual_forecast", {}) or {}
    base_probabilities = _normalize_probabilities(prediction.get("probabilities", {}))
    contextual_probabilities = _normalize_probabilities(contextual.get("contextual_probabilities", {}))
    return {
        "Partita": match_analysis.get("match_title"),
        "Giornata": _format_matchday(match_analysis.get("matchday")),
        "Prob base 1": base_probabilities.get("1", 0.0),
        "Prob base X": base_probabilities.get("X", 0.0),
        "Prob base 2": base_probabilities.get("2", 0.0),
        "Prob cont 1": contextual_probabilities.get("1", 0.0),
        "Prob cont X": contextual_probabilities.get("X", 0.0),
        "Prob cont 2": contextual_probabilities.get("2", 0.0),
        "Risultato piu probabile": prediction.get("most_likely_score") if prediction.get("ok") else "n/d",
        "Confidence": _safe_float(contextual.get("confidence"), 50.0),
        "Draw risk": _safe_float(contextual.get("draw_risk"), 50.0),
        "Upset risk": _safe_float(contextual.get("upset_risk"), 50.0),
        "Volatilita": match_analysis.get("volatility"),
        "Interesse match": match_analysis.get("interest"),
        "Tipo match": match_analysis.get("match_type"),
    }


def build_round_summary(match_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not match_analyses:
        return {
            "balanced_match": None,
            "highest_draw_risk_match": None,
            "highest_upset_risk_match": None,
            "highest_confidence_match": None,
            "most_volatile_match": None,
            "summary_text": "Nessuna partita disponibile per sintetizzare la giornata.",
        }

    def contextual_probs(item: dict[str, Any]) -> dict[str, float]:
        return _normalize_probabilities((item.get("contextual_forecast") or {}).get("contextual_probabilities", {}))

    def metric(item: dict[str, Any], key: str, default: float = 0.0) -> float:
        return _safe_float((item.get("contextual_forecast") or {}).get(key), default)

    balanced_match = min(
        match_analyses,
        key=lambda item: _probability_gap(contextual_probs(item)) if _probability_gap(contextual_probs(item)) is not None else 1.0,
    )
    highest_draw = max(match_analyses, key=lambda item: metric(item, "draw_risk", 0.0))
    highest_upset = max(match_analyses, key=lambda item: metric(item, "upset_risk", 0.0))
    highest_confidence = max(match_analyses, key=lambda item: metric(item, "confidence", 0.0))
    most_volatile = max(match_analyses, key=lambda item: _safe_float(item.get("volatility_score"), 0.0))

    lines = [
        f"La partita piu equilibrata nei dati e {balanced_match['match_title']}.",
        f"Il rischio pareggio piu alto emerge in {highest_draw['match_title']}.",
        f"Il rischio upset piu alto emerge in {highest_upset['match_title']}.",
        f"La lettura piu stabile per confidenza e {highest_confidence['match_title']}.",
        f"La partita piu volatile e {most_volatile['match_title']}.",
        "La sintesi resta prudente: senza calendario ufficiale, lineup e assenze, alcuni segnali possono cambiare vicino alla partita.",
    ]
    return {
        "balanced_match": balanced_match["match_title"],
        "highest_draw_risk_match": highest_draw["match_title"],
        "highest_upset_risk_match": highest_upset["match_title"],
        "highest_confidence_match": highest_confidence["match_title"],
        "most_volatile_match": most_volatile["match_title"],
        "summary_text": "\n".join(lines),
    }


def build_round_analysis(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    season: str | None = None,
    schedule_df: pd.DataFrame | None = None,
    max_matches: int | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return {"ok": False, "message": "La stagione selezionata non contiene dati utilizzabili."}

    if fixtures_df is None:
        fixtures_df = infer_next_round_fixtures(prepared_df, season=season)
    else:
        fixtures_df = _ensure_fixture_columns(fixtures_df)

    source_mode = fixtures_df.attrs.get("fixture_source")
    source_label = fixtures_df.attrs.get("source_label")
    warnings = list(fixtures_df.attrs.get("warnings", []))
    if not source_mode and "fixture_source" in fixtures_df.columns and not fixtures_df.empty:
        source_mode = str(fixtures_df["fixture_source"].iloc[0])
    if not source_label:
        source_label = "Fixture seed" if source_mode == "fixture_seed" else "Partite mancanti inferite"
    if source_mode == "inferred_missing" and "Le partite sono inferite, non confermate da calendario ufficiale." not in warnings:
        warnings.append("Le partite sono inferite, non confermate da calendario ufficiale.")

    fixtures_df = select_round_fixtures(fixtures_df, max_matches=max_matches)
    if fixtures_df.empty:
        return {
            "ok": False,
            "message": "Nessuna partita futura o mancante disponibile per costruire l'analisi giornata.",
            "fixture_source": source_mode,
            "source_label": source_label,
            "warnings": warnings,
        }

    match_analyses = [
        build_round_match_analysis(
            prepared_df,
            str(row["home_team"]),
            str(row["away_team"]),
            season=season,
            schedule_df=schedule_df,
            fixture_row=row,
        )
        for row in fixtures_df.to_dict(orient="records")
    ]
    summary_table = pd.DataFrame([_summary_row(item) for item in match_analyses])
    return {
        "ok": True,
        "season": season,
        "fixture_source": source_mode,
        "source_label": source_label,
        "warnings": list(dict.fromkeys(warnings)),
        "fixtures": fixtures_df,
        "matches": match_analyses,
        "summary_table": summary_table,
        "round_summary": build_round_summary(match_analyses),
    }
