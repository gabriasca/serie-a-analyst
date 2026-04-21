from __future__ import annotations

import streamlit as st

from src.analytics import build_home_away_table, build_standings
from src.config import APP_TITLE
from src.db import fetch_matches, list_seasons
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Dashboard", layout="wide")

bootstrap_database()

st.title("Dashboard Serie A")

seasons = list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
season_df = fetch_matches(selected_season)

standings = build_standings(season_df)
home_away_table = build_home_away_table(season_df)

st.subheader("Classifica")
st.dataframe(standings, use_container_width=True)

if not standings.empty:
    st.subheader("Punti per squadra")
    st.bar_chart(standings.set_index("Team")["Pts"])

st.subheader("Rendimento casa / trasferta")
st.dataframe(home_away_table, use_container_width=True)

if not home_away_table.empty:
    st.bar_chart(
        home_away_table.set_index("Team")[["Punti Casa", "Punti Trasferta"]],
        use_container_width=True,
    )
