from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import build_comparison_summary, compare_teams
from src.config import APP_TITLE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import fetch_matches, list_seasons, list_teams
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Confronto Squadre", layout="wide")

bootstrap_database()

st.title("Confronto Squadre")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season)

if len(teams) < 2:
    st.warning("Servono almeno due squadre per effettuare un confronto.")
    st.stop()

team_a = st.selectbox("Squadra A", teams, index=0)
team_b_options = [team for team in teams if team != team_a]
team_b = st.selectbox("Squadra B", team_b_options, index=0)

comparison = compare_teams(fetch_matches(selected_season), team_a, team_b)
stats_a = comparison["team_a"]
stats_b = comparison["team_b"]

left_col, right_col = st.columns(2)

with left_col:
    st.subheader(team_a)
    st.metric("Punti", stats_a["points"])
    st.metric("Forma ultime 5", stats_a["form_last_5"])
    st.metric("Gol fatti medi", stats_a["avg_goals_for"])
    st.metric("Gol subiti medi", stats_a["avg_goals_against"])
    st.dataframe(stats_a["home_away_split"], use_container_width=True)

with right_col:
    st.subheader(team_b)
    st.metric("Punti", stats_b["points"])
    st.metric("Forma ultime 5", stats_b["form_last_5"])
    st.metric("Gol fatti medi", stats_b["avg_goals_for"])
    st.metric("Gol subiti medi", stats_b["avg_goals_against"])
    st.dataframe(stats_b["home_away_split"], use_container_width=True)

comparison_table = pd.DataFrame(
    [
        {
            "Metrica": "Punti",
            team_a: stats_a["points"],
            team_b: stats_b["points"],
        },
        {
            "Metrica": "Forma ultime 5 (punti)",
            team_a: stats_a["form_points_last_5"],
            team_b: stats_b["form_points_last_5"],
        },
        {
            "Metrica": "Attacco (gol medi)",
            team_a: stats_a["avg_goals_for"],
            team_b: stats_b["avg_goals_for"],
        },
        {
            "Metrica": "Difesa (gol subiti medi)",
            team_a: stats_a["avg_goals_against"],
            team_b: stats_b["avg_goals_against"],
        },
    ]
)

st.subheader("Confronto diretto delle metriche")
st.dataframe(comparison_table, use_container_width=True)

st.subheader("Sintesi")
st.write(build_comparison_summary(comparison))
