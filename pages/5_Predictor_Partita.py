from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import APP_TITLE
from src.db import fetch_matches, list_seasons, list_teams
from src.explain import build_prediction_explanation
from src.predictor import predict_match
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Predictor", layout="wide")

bootstrap_database()

st.title("Predictor Partita")
st.write("Stima statistica semplice del match usando dati della stagione selezionata.")

seasons = list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season)

if len(teams) < 2:
    st.warning("Servono almeno due squadre per calcolare una previsione.")
    st.stop()

home_team = st.selectbox("Squadra casa", teams, index=0)
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox("Squadra trasferta", away_options, index=0)

if st.button("Calcola previsione"):
    prediction = predict_match(fetch_matches(selected_season), home_team, away_team, max_goals=6)

    if not prediction["ok"]:
        st.warning(prediction["message"])
        st.stop()

    probs = prediction["probabilities"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("xG casa", round(prediction["expected_goals_home"], 2))
    col2.metric("xG trasferta", round(prediction["expected_goals_away"], 2))
    col3.metric("1", f"{probs['1'] * 100:.1f}%")
    col4.metric("X", f"{probs['X'] * 100:.1f}%")
    col5.metric("2", f"{probs['2'] * 100:.1f}%")

    st.subheader("Risultato piu probabile")
    st.write(f"{home_team} {prediction['most_likely_score']} {away_team}")

    st.subheader("Top risultati possibili")
    top_scores_df = pd.DataFrame(prediction["top_scorelines"])
    top_scores_df["probability"] = top_scores_df["probability"].map(lambda value: f"{value * 100:.1f}%")
    st.dataframe(top_scores_df.rename(columns={"score": "Risultato", "probability": "Probabilita"}))

    st.subheader("Matrice Poisson (0-6 gol)")
    display_matrix = prediction["score_matrix"].copy()
    display_matrix = display_matrix.mul(100).round(2)
    st.dataframe(display_matrix, use_container_width=True)

    st.subheader("Spiegazione del modello")
    st.write(build_prediction_explanation(prediction))
    st.info("Questa e una stima statistica basata sui dati disponibili, non una certezza.")
