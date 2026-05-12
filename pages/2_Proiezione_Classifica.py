from __future__ import annotations

import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover
    config = None

from src.analytics import build_standings, get_teams
from src.db import fetch_matches, list_seasons
from src.projections import expected_total_matches, infer_remaining_fixtures, run_projection_simulations
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


st.set_page_config(page_title=f"{APP_TITLE} | Proiezione Classifica", layout="wide")
bootstrap_database()

st.title("Proiezione Classifica")
st.write("Simulazione della classifica finale basata sul predictor base Poisson.")
st.caption("La proiezione usa il predictor base. La lettura contestuale è disponibile nella pagina Previsione Partita.")
if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE) or list_seasons()
if not seasons:
    st.warning("Nessuna stagione Serie A disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Stagione", seasons)
season_df = fetch_matches(selected_season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
if season_df.empty:
    st.warning("La stagione selezionata non contiene partite di Serie A.")
    st.stop()

teams = get_teams(season_df)
if len(teams) < 2:
    st.warning("Servono almeno due squadre per costruire una proiezione.")
    st.stop()

played_matches = len(season_df)
expected_matches = expected_total_matches(len(teams))
remaining_fixtures = infer_remaining_fixtures(season_df)
current_table = build_standings(season_df).reset_index().rename(columns={"Team": "Squadra", "GP": "Partite", "Pts": "Punti", "DR": "Diff. reti"})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Squadre", len(teams))
col2.metric("Partite giocate", played_matches)
col3.metric("Partite attese", expected_matches)
col4.metric("Partite mancanti", len(remaining_fixtures))

st.subheader("Classifica attuale")
st.dataframe(current_table, width="stretch", hide_index=True)

with st.expander("Partite mancanti inferite", expanded=False):
    if remaining_fixtures.empty:
        st.info("La stagione risulta completa: non ci sono partite mancanti da inferire.")
    else:
        st.dataframe(remaining_fixtures.rename(columns={"home_team": "Casa", "away_team": "Trasferta"}), width="stretch", hide_index=True)

simulation_options = [100, 500, 1000, 2000]
simulation_count = st.selectbox("Numero simulazioni", simulation_options, index=2)

if st.button("Esegui simulazione", type="primary"):
    with st.spinner("Simulazione in corso..."):
        st.session_state["public_projection_result"] = {
            "season": selected_season,
            "simulation_count": simulation_count,
            "result": run_projection_simulations(season_df, simulation_count),
        }

stored_projection = st.session_state.get("public_projection_result")
if not stored_projection:
    st.info("Premi 'Esegui simulazione' per calcolare la classifica finale proiettata.")
    st.stop()

if stored_projection["season"] != selected_season or stored_projection["simulation_count"] != simulation_count:
    st.info("Premi di nuovo 'Esegui simulazione' per aggiornare la proiezione.")
    st.stop()

projection_result = stored_projection["result"]
if not projection_result.get("ok"):
    st.warning(projection_result.get("message", "Proiezione non disponibile."))
    st.stop()

if projection_result.get("complete_season"):
    st.info("La stagione appare già completa: la classifica proiettata coincide con quella attuale.")
if projection_result.get("fallback_count", 0) > 0:
    st.info(
        f"Per {projection_result['fallback_count']} partite mancanti è stato usato un fallback prudente "
        "basato sulla media gol campionato."
    )

summary_table = projection_result["summary_table"].copy()
display_summary = summary_table.copy()
for column in ["Prob. scudetto", "Prob. top 4", "Prob. top 6", "Prob. salvezza", "Prob. retrocessione"]:
    display_summary[column] = display_summary[column].map(lambda value: f"{value * 100:.1f}%")

st.subheader("Classifica finale proiettata")
st.dataframe(display_summary, width="stretch", hide_index=True)

st.subheader("Dettaglio squadra")
team_options = summary_table["Squadra"].tolist()
selected_team = st.selectbox("Squadra", team_options)
team_row = summary_table.loc[summary_table["Squadra"] == selected_team].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Punti attuali", int(team_row["Punti attuali"]))
m2.metric("Punti medi finali", f"{team_row['Punti medi finali']:.2f}")
m3.metric("Posizione media", f"{team_row['Posizione media']:.2f}")
m4.metric("Prob. scudetto", f"{team_row['Prob. scudetto'] * 100:.1f}%")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Prob. top 4", f"{team_row['Prob. top 4'] * 100:.1f}%")
m6.metric("Prob. top 6", f"{team_row['Prob. top 6'] * 100:.1f}%")
m7.metric("Prob. salvezza", f"{team_row['Prob. salvezza'] * 100:.1f}%")
m8.metric("Prob. retrocessione", f"{team_row['Prob. retrocessione'] * 100:.1f}%")

distribution_df = projection_result["position_distributions"][selected_team].copy()
st.write("Distribuzione delle posizioni finali")
st.bar_chart(distribution_df.set_index("Posizione")["Probabilita"], width="stretch")
