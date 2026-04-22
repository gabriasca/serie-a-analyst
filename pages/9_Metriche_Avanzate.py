from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback
    config = None

from src.advanced_metrics import (
    DISPLAY_METRIC_LABELS,
    build_advanced_team_metrics,
    build_metric_explanations,
    build_metric_strengths_and_weaknesses,
    build_metric_summary,
    get_team_advanced_metrics,
)
from src.db import fetch_matches, list_seasons
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _format_number(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _load_season_dataframe(season: str) -> pd.DataFrame:
    season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
    if season_df.empty:
        season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if season_df.empty:
        season_df = fetch_matches(season)
    return season_df


def _safe_dataframe(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.write(df.to_dict(orient="records"))


def _safe_bar_chart(df: pd.DataFrame, message: str) -> None:
    try:
        st.bar_chart(df)
    except Exception:
        st.caption(message)


def _render_bullets(items: list[str]) -> None:
    if not items:
        st.write("- Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in items))


def _render_top_five(metrics_df: pd.DataFrame, column: str, title: str, ascending: bool = False) -> None:
    st.markdown(f"### {title}")
    ranking_df = metrics_df.dropna(subset=[column]).sort_values([column, "team"], ascending=[ascending, True]).head(5)
    if ranking_df.empty:
        st.caption("Classifica non disponibile.")
        return
    for idx, row in enumerate(ranking_df.itertuples(index=False), start=1):
        st.write(f"{idx}. {row.team} - {getattr(row, column):.1f}")


st.set_page_config(page_title=f"{APP_TITLE} | Metriche Avanzate", layout="wide")

bootstrap_database()

st.title("Metriche Avanzate")
st.caption("Questi sono indicatori interni basati su dati aggregati disponibili. Non sono xG reali.")

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
    st.warning("La stagione selezionata non contiene dati sufficienti per calcolare metriche avanzate.")
    st.stop()

metrics_df = build_advanced_team_metrics(season_df)
if metrics_df.empty:
    st.warning("Non ci sono abbastanza partite o squadre per costruire il layer Metriche Avanzate.")
    st.stop()

if metrics_df["shots_avg"].isna().all():
    st.info("I dati tiri non sono presenti in modo affidabile: alcuni indici si appoggiano soprattutto a gol e punti.")
if metrics_df["elo_rating"].isna().all():
    st.info("Rating Elo non disponibile: la forza calendario usa la classifica corrente oppure resta non disponibile.")

st.subheader("Panoramica squadre")
table_df = metrics_df[
    [
        "team",
        "position",
        "points",
        "elo_rating",
        "offensive_threat_index",
        "defensive_solidity_index",
        "offensive_volume_index",
        "defensive_risk_index",
        "finishing_efficiency_index",
        "home_dependency_index",
        "recent_momentum_index",
        "schedule_strength_index",
    ]
].rename(
    columns={
        "team": "Squadra",
        "position": "Posizione",
        "points": "Punti",
        "elo_rating": "Rating Elo",
        "offensive_threat_index": "Pericolosita offensiva",
        "defensive_solidity_index": "Solidita difensiva",
        "offensive_volume_index": "Volume offensivo",
        "defensive_risk_index": "Rischio difensivo",
        "finishing_efficiency_index": "Efficienza realizzativa",
        "home_dependency_index": "Dipendenza casa",
        "recent_momentum_index": "Momento recente",
        "schedule_strength_index": "Forza calendario",
    }
)
for column in table_df.columns:
    if column in {"Squadra", "Posizione", "Punti"}:
        continue
    table_df[column] = pd.to_numeric(table_df[column], errors="coerce").round(1)
_safe_dataframe(table_df)

selected_team = st.selectbox("Seleziona squadra per il dettaglio", metrics_df["team"].tolist())
team_metrics = get_team_advanced_metrics(metrics_df, selected_team)
if not team_metrics:
    st.warning("Impossibile leggere il dettaglio della squadra selezionata.")
    st.stop()

metric_explanations = build_metric_explanations(team_metrics)
metric_feedback = build_metric_strengths_and_weaknesses(team_metrics)
metric_summary = build_metric_summary(team_metrics)

st.subheader(f"Dettaglio squadra: {selected_team}")
head_col1, head_col2, head_col3, head_col4 = st.columns(4)
head_col1.metric("Posizione", team_metrics.get("position") or "n/d")
head_col2.metric("Punti", team_metrics.get("points") or 0)
head_col3.metric("Partite", team_metrics.get("matches") or 0)
head_col4.metric("Rating Elo", _format_number(team_metrics.get("elo_rating"), digits=0))

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric(
    "Pericolosita offensiva",
    _format_number(team_metrics.get("offensive_threat_index")),
)
metric_col2.metric(
    "Solidita difensiva",
    _format_number(team_metrics.get("defensive_solidity_index")),
)
metric_col3.metric(
    "Volume offensivo",
    _format_number(team_metrics.get("offensive_volume_index")),
)
metric_col4.metric(
    "Rischio difensivo",
    _format_number(team_metrics.get("defensive_risk_index")),
)

metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
metric_col5.metric(
    "Efficienza realizzativa",
    _format_number(team_metrics.get("finishing_efficiency_index")),
)
metric_col6.metric(
    "Dipendenza casa",
    _format_number(team_metrics.get("home_dependency_index")),
)
metric_col7.metric(
    "Momento recente",
    _format_number(team_metrics.get("recent_momentum_index")),
)
metric_col8.metric(
    "Forza calendario",
    _format_number(team_metrics.get("schedule_strength_index")),
)
st.caption(team_metrics.get("schedule_strength_note") or "Forza calendario non disponibile.")

detail_chart = pd.DataFrame(
    {
        "Indice": [DISPLAY_METRIC_LABELS[column] for column in DISPLAY_METRIC_LABELS],
        "Valore": [team_metrics.get(column) for column in DISPLAY_METRIC_LABELS],
    }
).dropna(subset=["Valore"]).set_index("Indice")
if not detail_chart.empty:
    _safe_bar_chart(detail_chart, "Grafico indici non disponibile in questo ambiente Streamlit.")

st.subheader("Spiegazione degli indici")
for label, explanation in metric_explanations.items():
    st.write(f"**{label}**: {explanation}")

strength_col, weakness_col = st.columns(2)
with strength_col:
    st.subheader("Punti forti rilevati")
    _render_bullets(metric_feedback["strengths"])

with weakness_col:
    st.subheader("Punti deboli rilevati")
    _render_bullets(metric_feedback["weaknesses"])

st.subheader("Sintesi finale")
st.markdown(metric_summary.replace("\n", "  \n"))

st.subheader("Classifiche rapide")
rank_col1, rank_col2 = st.columns(2)
with rank_col1:
    _render_top_five(metrics_df, "offensive_threat_index", "Squadre piu pericolose offensivamente")
    _render_top_five(metrics_df, "home_dependency_index", "Squadre piu dipendenti dalla casa")
with rank_col2:
    _render_top_five(metrics_df, "defensive_solidity_index", "Squadre piu solide difensivamente")
    _render_top_five(metrics_df, "recent_momentum_index", "Squadre con miglior momento recente")

if metrics_df["schedule_strength_index"].notna().any():
    _render_top_five(
        metrics_df,
        "schedule_strength_index",
        "Squadre con calendario affrontato piu difficile",
    )

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.caption("Top 5 per pericolosita offensiva.")
    top_attack_df = (
        metrics_df.dropna(subset=["offensive_threat_index"])
        .sort_values(["offensive_threat_index", "team"], ascending=[False, True])
        .head(5)
        .set_index("team")[["offensive_threat_index"]]
        .rename(columns={"offensive_threat_index": "Pericolosita offensiva"})
    )
    if not top_attack_df.empty:
        _safe_bar_chart(top_attack_df, "Grafico top 5 offensivo non disponibile in questo ambiente Streamlit.")

with chart_col2:
    st.caption("Top 5 per solidita difensiva.")
    top_defense_df = (
        metrics_df.dropna(subset=["defensive_solidity_index"])
        .sort_values(["defensive_solidity_index", "team"], ascending=[False, True])
        .head(5)
        .set_index("team")[["defensive_solidity_index"]]
        .rename(columns={"defensive_solidity_index": "Solidita difensiva"})
    )
    if not top_defense_df.empty:
        _safe_bar_chart(top_defense_df, "Grafico top 5 difensivo non disponibile in questo ambiente Streamlit.")
