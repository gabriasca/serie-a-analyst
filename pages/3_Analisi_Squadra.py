from __future__ import annotations

import streamlit as st

from src.analytics import compute_team_stats
from src.config import APP_TITLE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import fetch_matches, list_seasons, list_teams
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Analisi Squadra", layout="wide")

bootstrap_database()

st.title("Analisi Squadra")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season)

if not teams:
    st.warning("Nessuna squadra disponibile per la stagione selezionata.")
    st.stop()

selected_team = st.selectbox("Seleziona squadra", teams)
season_df = fetch_matches(selected_season)
team_stats = compute_team_stats(season_df, selected_team)

if not team_stats:
    st.warning("Nessun dato disponibile per la squadra selezionata.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Punti", team_stats["points"])
col2.metric("Partite", team_stats["matches_played"])
col3.metric("Media gol fatti", team_stats["avg_goals_for"])
col4.metric("Media gol subiti", team_stats["avg_goals_against"])

col5, col6 = st.columns(2)
col5.metric("Differenza reti", team_stats["goal_difference"])
col6.metric("Forma ultime 5", team_stats["form_last_5"])

st.subheader("Rendimento casa / trasferta")
st.dataframe(team_stats["home_away_split"], use_container_width=True)

st.subheader("Partite recenti")
st.dataframe(team_stats["recent_matches"], use_container_width=True)

st.subheader("Andamento punti nel tempo")
progression_df = team_stats["points_progression"]
chart_df = progression_df.set_index("match_number")["cumulative_points"]
st.line_chart(chart_df)
st.dataframe(progression_df, use_container_width=True)
