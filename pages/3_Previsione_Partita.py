from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover
    config = None

from src.db import fetch_matches, list_seasons, list_teams
from src.narrative_match_report import build_narrative_match_report
from src.round_analysis import filter_future_fixtures, get_next_fixture_round, load_fixture_seed
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct(value: object) -> str:
    return f"{_safe_float(value) * 100:.1f}%"


def _format_score(value: object, suffix: str = "/100") -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return "n/d"


def _normalize_probs(probabilities: dict[str, object] | None) -> dict[str, float]:
    probabilities = probabilities or {}
    probs = {key: max(_safe_float(probabilities.get(key)), 0.0) for key in ["1", "X", "2"]}
    total = sum(probs.values())
    if total <= 0:
        return {key: 0.0 for key in probs}
    return {key: value / total for key, value in probs.items()}


def _favorite_label(probabilities: dict[str, object], home_team: str, away_team: str) -> str:
    probs = _normalize_probs(probabilities)
    if sum(probs.values()) <= 0:
        return "n/d"
    key = max(probs, key=probs.get)
    value = probs[key]
    if key == "X":
        return "Partita da pareggio"
    team = home_team if key == "1" else away_team
    if value >= 0.60:
        return f"{team} avanti chiaramente"
    if value >= 0.50:
        return f"{team} leggermente avanti"
    return f"{team}, ma partita aperta"


def _risk_label(value: object) -> str:
    score = _safe_float(value, 50.0)
    if score >= 60:
        return "alto"
    if score >= 45:
        return "medio"
    return "basso"


def _confidence_label(value: object) -> str:
    score = _safe_float(value, 50.0)
    if score >= 70:
        return "alta"
    if score < 45:
        return "bassa"
    return "media"


def _fixture_options(fixtures_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    options: dict[str, tuple[str, str]] = {}
    if fixtures_df.empty:
        return options
    for row in fixtures_df.to_dict(orient="records"):
        date = pd.to_datetime(row.get("match_date"), errors="coerce")
        date_label = date.strftime("%Y-%m-%d") if not pd.isna(date) else "data n/d"
        matchday = row.get("matchday")
        matchday_label = f"Giornata {int(matchday)}" if pd.notna(matchday) else "Fixture"
        home_team = str(row.get("home_team") or "")
        away_team = str(row.get("away_team") or "")
        label = f"{matchday_label} | {date_label} | {home_team} - {away_team}"
        options[label] = (home_team, away_team)
    return options


def _probability_table(base: dict[str, object], contextual: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, label in [("1", "Casa"), ("X", "Pareggio"), ("2", "Trasferta")]:
        rows.append({"Esito": label, "Base": _format_pct(base.get(key)), "Contestuale": _format_pct(contextual.get(key))})
    return pd.DataFrame(rows)


def _top_scores_table(prediction: dict[str, object]) -> pd.DataFrame:
    rows = []
    for item in prediction.get("top_scorelines", []) or []:
        if isinstance(item, dict):
            rows.append({"Risultato": item.get("score"), "Probabilità": _format_pct(item.get("probability"))})
    return pd.DataFrame(rows)


def _render_text(text: object) -> None:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        st.caption("Sezione non disponibile.")
        return
    for line in lines:
        st.write(line)


def _render_bullets(items: list[object]) -> None:
    cleaned = [str(item).strip() for item in items if str(item or "").strip()]
    if not cleaned:
        st.caption("Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in cleaned))


st.set_page_config(page_title=f"{APP_TITLE} | Previsione Partita", layout="wide")
bootstrap_database()

st.title("Previsione Partita")
st.write("Previsione base, lettura contestuale e report discorsivo per una singola gara.")
if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE) or list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Stagione", seasons)
league_df = fetch_matches(selected_season, competition_code=DEFAULT_COMPETITION_CODE)
schedule_df = fetch_matches(selected_season)
if league_df.empty:
    st.warning("La stagione selezionata non contiene partite di Serie A.")
    st.stop()
if schedule_df.empty:
    schedule_df = league_df

future_fixtures = get_next_fixture_round(filter_future_fixtures(load_fixture_seed(), schedule_df, season=selected_season))
fixture_options = _fixture_options(future_fixtures)
mode_options = ["Da fixture seed", "Scelta manuale"] if fixture_options else ["Scelta manuale"]
mode = st.radio("Scegli partita", mode_options, horizontal=True)

if mode == "Da fixture seed" and fixture_options:
    selected_fixture = st.selectbox("Partita da calendario", list(fixture_options.keys()))
    home_team, away_team = fixture_options[selected_fixture]
else:
    teams = list_teams(selected_season, competition_code=DEFAULT_COMPETITION_CODE) or list_teams(selected_season)
    if len(teams) < 2:
        st.warning("Servono almeno due squadre per generare una previsione.")
        st.stop()
    home_team = st.selectbox("Squadra casa", teams, index=0)
    away_candidates = [team for team in teams if team != home_team]
    away_team = st.selectbox("Squadra trasferta", away_candidates, index=0)

if home_team == away_team:
    st.warning("Seleziona due squadre diverse.")
    st.stop()

report = build_narrative_match_report(league_df, home_team, away_team, selected_season, schedule_df=schedule_df)
if not report.get("ok"):
    st.warning(report.get("message", "Previsione non disponibile."))
    st.stop()

prediction = report.get("prediction", {}) or {}
contextual = report.get("contextual_forecast", {}) or {}
base_probs = contextual.get("base_probabilities") or prediction.get("probabilities", {}) or {}
contextual_probs = contextual.get("contextual_probabilities") or base_probs
technical = report.get("technical", {}) or {}

st.subheader(f"{home_team} - {away_team}")
quick1, quick2, quick3, quick4 = st.columns(4)
quick1.metric("Sintesi", _favorite_label(contextual_probs, home_team, away_team))
quick2.metric("Fiducia", _confidence_label(contextual.get("confidence")))
quick3.metric("Rischio pareggio", _risk_label(contextual.get("draw_risk")))
quick4.metric("Rischio upset", _risk_label(contextual.get("upset_risk")))

st.subheader("Previsione")
col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(_probability_table(base_probs, contextual_probs), width="stretch", hide_index=True)
with col2:
    goal1, goal2, goal3 = st.columns(3)
    goal1.metric("Gol attesi modello casa", f"{_safe_float(prediction.get('expected_goals_home')):.2f}")
    goal2.metric("Gol attesi modello trasferta", f"{_safe_float(prediction.get('expected_goals_away')):.2f}")
    goal3.metric("Risultato più probabile", prediction.get("most_likely_score") or "n/d")
    st.caption("Sono gol attesi dal modello Poisson interno, non xG reali shot-by-shot.")

st.subheader("Analisi narrativa")
_render_text(report.get("brief_reading"))
_render_text(report.get("probable_match_script"))
_render_text(report.get("alternative_match_script"))
_render_text(report.get("final_analyst_take"))

st.subheader("Cosa può cambiare la partita")
change_items = list(report.get("what_could_change", []) or [])
change_items.extend(
    [
        "Formazioni ufficiali non disponibili.",
        "Assenze, squalifiche e turnover non disponibili.",
        "Dati tattici granulari e informazioni live non presenti nel database.",
        "Calendario parziale se mancano coppe o competizioni europee.",
    ]
)
_render_bullets(list(dict.fromkeys(str(item) for item in change_items if str(item).strip()))[:6])

with st.expander("Dettaglio tecnico opzionale", expanded=False):
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Adjusted edge", _format_score(technical.get("adjusted_edge"), ""))
    t2.metric("Draw risk", _format_score(technical.get("draw_risk")))
    t3.metric("Upset risk", _format_score(technical.get("upset_risk")))
    t4.metric("Confidence", _format_score(technical.get("confidence")))

    st.markdown("#### Fattori pesati")
    factors = technical.get("weighted_factors", [])
    if isinstance(factors, list) and factors:
        rows = [
            {
                "Fattore": item.get("label") or item.get("factor") or "Fattore",
                "Impatto": item.get("weighted_impact"),
                "Nota": item.get("note") or "",
            }
            for item in factors
            if isinstance(item, dict)
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("Fattori pesati non disponibili.")

    st.markdown("#### Top risultati Poisson")
    st.dataframe(_top_scores_table(prediction), width="stretch", hide_index=True)

    if st.checkbox("Mostra matrice Poisson"):
        matrix = prediction.get("score_matrix")
        if isinstance(matrix, pd.DataFrame):
            st.dataframe(matrix.map(lambda value: f"{float(value) * 100:.1f}%"), width="stretch")
        else:
            st.caption("Matrice Poisson non disponibile.")
