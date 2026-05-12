from __future__ import annotations

import streamlit as st

from src.analytics import build_home_away_table, build_standings
from src.config import APP_TITLE, DEFAULT_COMPETITION_CODE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import fetch_matches, list_seasons
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Dashboard", layout="wide")

bootstrap_database()

st.title("Dashboard Serie A")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    st.warning("Nessuna stagione Serie A disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
season_df = fetch_matches(selected_season, competition_code=DEFAULT_COMPETITION_CODE)

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
