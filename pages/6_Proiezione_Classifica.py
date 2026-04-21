from __future__ import annotations

import streamlit as st

from src.analytics import build_standings, get_teams
from src.config import APP_TITLE
from src.db import fetch_matches, list_seasons
from src.projections import expected_total_matches, infer_remaining_fixtures, run_projection_simulations
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Proiezione Classifica", layout="wide")

bootstrap_database()

st.title("Proiezione Classifica")
st.write(
    "Proietta la classifica finale combinando classifica attuale, partite mancanti inferite "
    "e simulazioni Monte Carlo basate sul predictor Poisson gia presente."
)
st.warning("Le proiezioni sono simulazioni statistiche basate sui dati disponibili, non certezze.")

seasons = list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
season_df = fetch_matches(selected_season)

if season_df.empty:
    st.warning("La stagione selezionata non contiene partite.")
    st.stop()

teams = get_teams(season_df)
if len(teams) < 2:
    st.warning("Servono almeno due squadre per costruire una proiezione di classifica.")
    st.stop()

played_matches = len(season_df)
expected_matches = expected_total_matches(len(teams))
remaining_fixtures = infer_remaining_fixtures(season_df)
current_table = build_standings(season_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Squadre", len(teams))
col2.metric("Partite giocate", played_matches)
col3.metric("Partite attese totali", expected_matches)
col4.metric("Partite mancanti", len(remaining_fixtures))

st.subheader("Classifica attuale")
st.dataframe(current_table, use_container_width=True)

st.subheader("Partite mancanti inferite")
if remaining_fixtures.empty:
    st.info("La stagione risulta completa: non ci sono partite mancanti da inferire.")
else:
    st.dataframe(
        remaining_fixtures.rename(columns={"home_team": "Casa", "away_team": "Trasferta"}),
        use_container_width=True,
    )

simulation_options = [100, 500, 1000, 2000]
default_index = simulation_options.index(1000)
simulation_count = st.selectbox("Numero simulazioni", simulation_options, index=default_index)

if st.button("Esegui simulazione"):
    with st.spinner("Simulazione in corso..."):
        st.session_state["projection_classification_result"] = {
            "season": selected_season,
            "simulation_count": simulation_count,
            "result": run_projection_simulations(season_df, simulation_count),
        }

stored_projection = st.session_state.get("projection_classification_result")
if not stored_projection:
    st.info("Premi 'Esegui simulazione' per calcolare la classifica finale proiettata.")
    st.stop()

if (
    stored_projection["season"] != selected_season
    or stored_projection["simulation_count"] != simulation_count
):
    st.info("Premi di nuovo 'Esegui simulazione' per aggiornare la proiezione con la selezione corrente.")
    st.stop()

projection_result = stored_projection["result"]
if not projection_result["ok"]:
    st.warning(projection_result["message"])
    st.stop()

if projection_result["complete_season"]:
    st.info(
        "La stagione appare gia completa. La classifica proiettata coincide con quella attuale "
        "e le probabilita diventano deterministiche."
    )

if projection_result["fallback_count"] > 0:
    st.info(
        f"Per {projection_result['fallback_count']} partite mancanti il predictor non aveva dati sufficienti: "
        "e stato usato un fallback prudente basato sulla media gol campionato."
    )

summary_table = projection_result["summary_table"].copy()

st.subheader("Classifica finale proiettata")
display_summary = summary_table.copy()
for column in [
    "Prob. scudetto",
    "Prob. top 4",
    "Prob. top 6",
    "Prob. salvezza",
    "Prob. retrocessione",
]:
    display_summary[column] = display_summary[column].map(lambda value: f"{value * 100:.1f}%")

st.dataframe(display_summary, use_container_width=True)

with st.expander("Dettaglio stima partite mancanti"):
    fixture_table = projection_result["fixture_table"].copy()
    if fixture_table.empty:
        st.write("Nessuna partita mancante da stimare.")
    else:
        for column in ["Prob. 1", "Prob. X", "Prob. 2"]:
            fixture_table[column] = fixture_table[column].map(lambda value: f"{value * 100:.1f}%")
        st.dataframe(fixture_table, use_container_width=True)

st.subheader("Dettaglio squadra")
team_options = summary_table["Squadra"].tolist()
selected_team = st.selectbox("Seleziona squadra", team_options)
team_row = summary_table.loc[summary_table["Squadra"] == selected_team].iloc[0]

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Punti attuali", int(team_row["Punti attuali"]))
metric2.metric("Punti medi finali", f"{team_row['Punti medi finali']:.2f}")
metric3.metric("Posizione media", f"{team_row['Posizione media']:.2f}")
metric4.metric("Prob. scudetto", f"{team_row['Prob. scudetto'] * 100:.1f}%")

metric5, metric6, metric7, metric8 = st.columns(4)
metric5.metric("Prob. top 4", f"{team_row['Prob. top 4'] * 100:.1f}%")
metric6.metric("Prob. top 6", f"{team_row['Prob. top 6'] * 100:.1f}%")
metric7.metric("Prob. salvezza", f"{team_row['Prob. salvezza'] * 100:.1f}%")
metric8.metric("Prob. retrocessione", f"{team_row['Prob. retrocessione'] * 100:.1f}%")

distribution_df = projection_result["position_distributions"][selected_team].copy()
st.write("Distribuzione delle posizioni finali")
st.bar_chart(distribution_df.set_index("Posizione")["Probabilita"], use_container_width=True)

display_distribution = distribution_df.copy()
display_distribution["Probabilita"] = display_distribution["Probabilita"].map(
    lambda value: f"{value * 100:.1f}%"
)
st.dataframe(display_distribution, use_container_width=True)
