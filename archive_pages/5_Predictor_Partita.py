from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, DEFAULT_COMPETITION_CODE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import fetch_matches, list_seasons, list_teams
from src.explain import build_prediction_explanation
from src.forecast_context import build_contextual_forecast
from src.matchup_analysis import build_matchup_analysis
from src.predictor import predict_match
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Predictor", layout="wide")


def _load_prediction_dataframes(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    schedule_df = fetch_matches(season)
    if schedule_df.empty:
        schedule_df = league_df
    return league_df, schedule_df


def _format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:.1f}%"


def _format_delta(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:+.1f} pp"


def _render_bullets(items: list[str]) -> None:
    if not items:
        st.write("- Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in items))

bootstrap_database()

st.title("Predictor Partita")
st.write("Stima statistica semplice del match usando dati della stagione selezionata.")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    st.warning("Nessuna stagione Serie A disponibile nel database. Vai in Import Dati per caricare un CSV o il dataset demo.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season, competition_code=DEFAULT_COMPETITION_CODE)

if len(teams) < 2:
    st.warning("Servono almeno due squadre per calcolare una previsione.")
    st.stop()

home_team = st.selectbox("Squadra casa", teams, index=0)
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox("Squadra trasferta", away_options, index=0)

if st.button("Calcola previsione"):
    season_df, schedule_df = _load_prediction_dataframes(selected_season)
    if season_df.empty:
        st.warning("La stagione selezionata non contiene partite di Serie A utilizzabili per il predictor.")
        st.stop()

    prediction = predict_match(
        season_df,
        home_team,
        away_team,
        max_goals=6,
    )

    if not prediction["ok"]:
        st.warning(prediction["message"])
        st.stop()

    matchup_analysis = build_matchup_analysis(season_df, home_team, away_team, schedule_df=schedule_df)

    contextual_forecast = build_contextual_forecast(prediction, matchup_analysis=matchup_analysis)

    st.subheader("Previsione base")
    probs = prediction["probabilities"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Gol attesi modello casa", round(prediction["expected_goals_home"], 2))
    col2.metric("Gol attesi modello trasferta", round(prediction["expected_goals_away"], 2))
    col3.metric("1", f"{probs['1'] * 100:.1f}%")
    col4.metric("X", f"{probs['X'] * 100:.1f}%")
    col5.metric("2", f"{probs['2'] * 100:.1f}%")
    st.caption("Questi sono gol attesi dal modello Poisson interno, non metriche shot-by-shot.")

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

    st.subheader("Lettura contestuale")
    contextual_probs = contextual_forecast["contextual_probabilities"]
    delta_rows = contextual_forecast["probability_deltas"]
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    ctx_col1.metric("1 contestuale", _format_pct(contextual_probs.get("1")), delta=_format_delta(delta_rows[0]["delta"]))
    ctx_col2.metric("X contestuale", _format_pct(contextual_probs.get("X")), delta=_format_delta(delta_rows[1]["delta"]))
    ctx_col3.metric("2 contestuale", _format_pct(contextual_probs.get("2")), delta=_format_delta(delta_rows[2]["delta"]))

    probability_table = pd.DataFrame(delta_rows).rename(
        columns={
            "label": "Esito",
            "base_probability": "Base",
            "contextual_probability": "Contestuale",
            "delta": "Differenza",
        }
    )
    probability_table["Base"] = probability_table["Base"].map(_format_pct)
    probability_table["Contestuale"] = probability_table["Contestuale"].map(_format_pct)
    probability_table["Differenza"] = probability_table["Differenza"].map(_format_delta)
    st.dataframe(probability_table[["Esito", "Base", "Contestuale", "Differenza"]], use_container_width=True)

    context_col1, context_col2, context_col3, context_col4 = st.columns(4)
    context_col1.metric("Adjusted edge", f"{contextual_forecast['adjusted_edge']:.2f}")
    context_col2.metric("Draw risk", f"{contextual_forecast['draw_risk']:.1f}/100")
    context_col3.metric("Upset risk", f"{contextual_forecast['upset_risk']:.1f}/100")
    context_col4.metric("Confidence", f"{contextual_forecast['confidence']:.1f}/100")
    st.caption(f"Livello confidenza contestuale: {contextual_forecast['confidence_label']}.")

    st.markdown("### Fattori principali")
    _render_bullets(contextual_forecast.get("key_adjustments", []))
    st.write(contextual_forecast.get("contextual_interpretation", "Lettura contestuale non disponibile."))

    for warning in contextual_forecast.get("warnings", []):
        st.info(warning)

    st.subheader("Come leggere questa previsione")
    _render_bullets(
        [
            "La previsione base e il modello numerico stabile: Poisson interno su gol, forza attacco/difesa, vantaggio casa e forma recente.",
            "La lettura contestuale usa Matchup Analysis, Elo, metriche avanzate, context_engine e calendario/riposo per correggere poco e in modo spiegabile.",
            "Il Predictor contestuale v2 e sperimentale: serve a leggere meglio il contesto, non a produrre certezze.",
            "Non usa quote e non e un consiglio operativo: resta un supporto analitico basato sui dati disponibili.",
        ]
    )
