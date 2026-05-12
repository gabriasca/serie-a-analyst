from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - fallback difensivo per Streamlit Cloud
    config = None

from src.db import fetch_matches, list_seasons, list_teams
from src.narrative_match_report import build_narrative_match_report
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _load_report_dataframes(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if league_df.empty:
        league_df = fetch_matches(season)
    schedule_df = fetch_matches(season)
    if schedule_df.empty:
        schedule_df = league_df
    return league_df, schedule_df


def _format_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/d"


def _format_score(value: object, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return "n/d"


def _render_text_block(text: object) -> None:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        st.caption("Sezione non disponibile con i dati correnti.")
        return
    for line in lines:
        st.write(line)


def _render_bullets(items: list[object]) -> None:
    cleaned = [str(item).strip() for item in items if str(item or "").strip()]
    if not cleaned:
        st.caption("Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in cleaned))


def _technical_probabilities_table(technical: dict[str, object]) -> pd.DataFrame:
    base = technical.get("base_probabilities", {}) if isinstance(technical, dict) else {}
    contextual = technical.get("contextual_probabilities", {}) if isinstance(technical, dict) else {}
    rows = []
    for outcome, label in [("1", "Casa"), ("X", "Pareggio"), ("2", "Trasferta")]:
        rows.append(
            {
                "Esito": label,
                "Base": _format_pct((base or {}).get(outcome)),
                "Contestuale v2": _format_pct((contextual or {}).get(outcome)),
            }
        )
    return pd.DataFrame(rows)


def _technical_factors_table(technical: dict[str, object]) -> pd.DataFrame:
    factors = technical.get("weighted_factors", []) if isinstance(technical, dict) else []
    if not isinstance(factors, list) or not factors:
        return pd.DataFrame()
    rows = []
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        rows.append(
            {
                "Fattore": factor.get("label") or factor.get("factor") or "Fattore",
                "Impatto": _format_score(factor.get("weighted_impact")),
                "Nota": factor.get("note") or "",
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title=f"{APP_TITLE} | Report Discorsivo Partita", layout="wide")

bootstrap_database()

st.title("Report Discorsivo Partita")
st.caption("Analisi basata sui dati disponibili. Non è una certezza e non usa quote.")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    seasons = list_seasons()

if not seasons:
    st.warning("Nessuna stagione disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season, competition_code=DEFAULT_COMPETITION_CODE)
if len(teams) < 2:
    teams = list_teams(selected_season)

if len(teams) < 2:
    st.warning("Servono almeno due squadre nella stagione selezionata.")
    st.stop()

home_team = st.selectbox("Squadra casa", teams, index=0)
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox("Squadra trasferta", away_options, index=0)

if st.button("Genera report discorsivo"):
    league_df, schedule_df = _load_report_dataframes(selected_season)
    with st.spinner("Costruisco una lettura discorsiva usando predictor, matchup, identità, metriche e calendario..."):
        report = build_narrative_match_report(
            league_df,
            home_team,
            away_team,
            selected_season,
            schedule_df=schedule_df,
        )
    st.session_state["narrative_match_report_result"] = {
        "season": selected_season,
        "home_team": home_team,
        "away_team": away_team,
        "report": report,
    }

stored_report = st.session_state.get("narrative_match_report_result")
if not stored_report:
    st.info("Seleziona due squadre e premi 'Genera report discorsivo'.")
    st.stop()

if (
    stored_report.get("season") != selected_season
    or stored_report.get("home_team") != home_team
    or stored_report.get("away_team") != away_team
):
    st.info("Premi di nuovo 'Genera report discorsivo' per aggiornare il contenuto con la selezione corrente.")
    st.stop()

report = stored_report.get("report", {})
if not report.get("ok"):
    st.warning(report.get("message", "Impossibile generare il report discorsivo con i dati disponibili."))
    st.stop()

st.header(report.get("match_title", f"{home_team} - {away_team}"))

for warning in report.get("warnings", []):
    st.info(warning)

st.subheader("Lettura in breve")
_render_text_block(report.get("brief_reading"))

st.subheader("Trama probabile della partita")
_render_text_block(report.get("probable_match_script"))

st.subheader("Scenario alternativo")
_render_text_block(report.get("alternative_match_script"))

st.subheader("Interazione tra identità delle squadre")
_render_text_block(report.get("team_identity_interaction"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dati più affidabili")
    _render_bullets(report.get("reliable_data", []))
with col2:
    st.subheader("Dati più fragili")
    _render_bullets(report.get("fragile_data", []))

st.subheader("Cosa può cambiare la partita")
_render_bullets(report.get("what_could_change", []))

st.subheader("Cosa servirebbe sapere prima del match")
_render_bullets(report.get("data_gaps", []))

st.subheader("Conclusione dell'analista")
_render_text_block(report.get("final_analyst_take"))

with st.expander("Sezione tecnica opzionale", expanded=False):
    technical = report.get("technical", {}) or {}
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    tech_col1.metric("Adjusted edge", _format_score(technical.get("adjusted_edge")))
    tech_col2.metric("Draw risk", _format_score(technical.get("draw_risk"), "/100"))
    tech_col3.metric("Upset risk", _format_score(technical.get("upset_risk"), "/100"))
    tech_col4.metric("Confidence", _format_score(technical.get("confidence"), "/100"))

    goals_col1, goals_col2 = st.columns(2)
    goals_col1.metric("Gol attesi modello casa", _format_score(technical.get("expected_goals_home")))
    goals_col2.metric("Gol attesi modello trasferta", _format_score(technical.get("expected_goals_away")))

    st.markdown("#### Probabilità")
    st.dataframe(_technical_probabilities_table(technical), use_container_width=True)

    factors_table = _technical_factors_table(technical)
    st.markdown("#### Fattori pesati")
    if factors_table.empty:
        st.caption("Fattori pesati non disponibili.")
    else:
        st.dataframe(factors_table, use_container_width=True)

st.caption(
    "Il report distingue dati osservati, indicatori interni, ipotesi prudenti e dati mancanti. "
    "Non usa quote e non inventa informazioni tattiche, giocatori disponibili o assenze."
)
