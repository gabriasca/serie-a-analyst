from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback
    config = None

from src.analytics import get_teams
from src.db import fetch_matches, list_seasons
from src.seed_data import bootstrap_database
from src.team_identity import build_team_identity_report


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _format_number(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value.is_integer():
        return f"{int(numeric_value)}{suffix}"
    return f"{numeric_value:.{digits}f}{suffix}"


def _render_bullets(items: list[str]) -> None:
    if not items:
        st.write("- Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in items))


def _safe_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.caption("Tabella non disponibile con i dati correnti.")
        return
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.write(df.to_dict(orient="records"))


def _load_season_dataframe(season: str) -> pd.DataFrame:
    season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
    if season_df.empty:
        season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if season_df.empty:
        season_df = fetch_matches(season)
    return season_df


def _render_result_block(title: str, block: dict[str, Any]) -> None:
    st.subheader(title)
    observed_col, internal_col, hypothesis_col = st.columns(3)
    with observed_col:
        st.markdown("### Dati osservati")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Partite", block.get("matches", 0))
        metric2.metric("GF medi", _format_number(block.get("goals_for_avg")))
        metric3.metric("GA medi", _format_number(block.get("goals_against_avg")))
        st.caption(f"Casa: {block.get('home_count', 0)} | Trasferta: {block.get('away_count', 0)}")
    with internal_col:
        st.markdown("### Indicatori interni")
        st.write(f"Tiri medi: {_format_number(block.get('shots_for_avg'))}")
        st.write(f"Tiri concessi medi: {_format_number(block.get('shots_against_avg'))}")
        st.caption("Questi valori dipendono dalla disponibilita di tiri/corner nel database.")
    with hypothesis_col:
        st.markdown("### Ipotesi prudenti")
        _render_bullets(block.get("observed_patterns", []))

    with st.expander(f"Dettaglio ultime partite nel blocco: {title}"):
        _safe_dataframe(block.get("table", pd.DataFrame()))


st.set_page_config(page_title=f"{APP_TITLE} | Studio Squadra", layout="wide")

bootstrap_database()

st.title("Studio Squadra")
st.write(
    "Studio dell'identita stagionale di una squadra: separa dati osservati, indicatori interni "
    "e ipotesi prudenti costruite solo sui dati disponibili."
)
st.caption(
    "Non usa API esterne, scraping o dati tattici non presenti. Le ipotesi sono letture condizionali, non certezze."
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    seasons = list_seasons()

if not seasons:
    st.warning("Database vuoto o nessuna stagione disponibile.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
season_df = _load_season_dataframe(selected_season)
if season_df.empty:
    st.warning("La stagione selezionata non contiene dati utilizzabili.")
    st.stop()

teams = get_teams(season_df)
if not teams:
    st.warning("Nessuna squadra disponibile nella stagione selezionata.")
    st.stop()

selected_team = st.selectbox("Seleziona squadra", teams)
report = build_team_identity_report(season_df, selected_team)
if not report.get("ok"):
    st.warning(report.get("message", "Impossibile costruire lo studio squadra con i dati disponibili."))
    st.stop()

observed = report["observed_data"]
general = observed.get("general", {})
rating = observed.get("rating", {})
recent = observed.get("recent", {})
metrics = report.get("internal_indicators", {})

st.subheader("Carta identita")
top_col1, top_col2, top_col3, top_col4, top_col5, top_col6 = st.columns(6)
top_col1.metric("Posizione", general.get("position", "n/d"))
top_col2.metric("Punti", general.get("points", 0))
top_col3.metric("Partite", general.get("matches", 0))
top_col4.metric("Gol fatti", general.get("goals_for", 0))
top_col5.metric("Gol subiti", general.get("goals_against", 0))
top_col6.metric("Forma ultime 5", recent.get("form_string", "-"))

rating_col1, rating_col2, rating_col3, rating_col4 = st.columns(4)
rating_col1.metric("Elo", _format_number(rating.get("rating_value"), digits=0))
rating_col2.metric("Fascia forza", rating.get("strength_band") or "n/d")
rating_col3.metric("Pericolosita offensiva", _format_number(metrics.get("offensive_threat_index")))
rating_col4.metric("Solidita difensiva", _format_number(metrics.get("defensive_solidity_index")))

st.caption(
    "Carta identita: posizione, punti e gol sono dati osservati; Elo e indici sono layer informativi o indicatori interni."
)

_render_result_block("Come tende a vincere", report.get("win_patterns", {}))
_render_result_block("Come tende a perdere", report.get("loss_patterns", {}))
_render_result_block("Quando pareggia", report.get("draw_patterns", {}))

st.subheader("Contro chi rende meglio o peggio")
opponent_bands = report.get("opponent_bands", {})
band_rows = opponent_bands.get("rows", [])
if band_rows:
    band_df = pd.DataFrame(band_rows).drop(columns=["band"], errors="ignore")
    _safe_dataframe(band_df)
    best_band = opponent_bands.get("best_band")
    weakest_band = opponent_bands.get("weakest_band")
    if best_band:
        st.success(f"Rendimento migliore osservato: {best_band.get('Fascia')} ({best_band.get('PPM')} PPM).")
    if weakest_band:
        st.info(f"Fascia piu complicata osservata: {weakest_band.get('Fascia')} ({weakest_band.get('PPM')} PPM).")
    st.caption(
        "Le fasce usano Elo se disponibile per tutte le squadre; altrimenti usano la classifica corrente."
    )
else:
    st.caption("Dati per fascia avversaria non disponibili.")

st.subheader("Casa/Fuori e andamento recente")
home_away_shift = report.get("home_away_shift", {})
recent_trend = report.get("recent_trend", {})
ha_col, recent_col = st.columns(2)
with ha_col:
    st.markdown("### Dati osservati casa/fuori")
    _safe_dataframe(pd.DataFrame(home_away_shift.get("rows", [])))
    st.caption(home_away_shift.get("note", "Lettura casa/fuori non disponibile."))
with recent_col:
    st.markdown("### Trend recente")
    r1, r2, r3 = st.columns(3)
    r1.metric("Trend", recent_trend.get("trend", "n/d"))
    r2.metric("PPM stagione", _format_number(recent_trend.get("season_ppm")))
    r3.metric("PPM ultime 5", _format_number(recent_trend.get("recent_ppm")))
    st.write(
        f"Forma: {recent_trend.get('recent_form', '-')} | "
        f"GF {recent_trend.get('recent_goals_for', 0)} | GA {recent_trend.get('recent_goals_against', 0)}"
    )
    st.caption(recent_trend.get("note", "Trend recente non disponibile."))

st.subheader("Stabilita / volatilita")
volatility = report.get("volatility", {})
vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
vol_col1.metric("Indice volatilita", _format_number(volatility.get("volatility_index"), suffix="/100"))
vol_col2.metric("Lettura", volatility.get("label", "n/d"))
vol_col3.metric("Oscillazione DR", _format_number(volatility.get("goal_difference_std")))
vol_col4.metric("Gap casa/fuori", _format_number(volatility.get("home_away_gap")))
st.caption(volatility.get("note", "Indicatore interno non disponibile."))
_render_bullets(volatility.get("drivers", []))

st.subheader("Ipotesi di identita")
st.caption("Etichette prudenti derivate da dati osservati e indicatori interni. Non sono certezze tattiche.")
_render_bullets(report.get("prudent_hypotheses", []))

st.subheader("Cosa non sappiamo ancora")
st.caption("Dati mancanti per analisi tattica avanzata.")
_render_bullets(report.get("missing_data_notes", []))

st.subheader("Sintesi finale")
st.markdown(str(report.get("summary", "")).replace("\n", "  \n"))
