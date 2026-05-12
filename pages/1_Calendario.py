from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover
    config = None

from src.db import fetch_matches, list_seasons
from src.projections import infer_remaining_fixtures
from src.round_analysis import (
    build_fixture_seed_report,
    build_round_analysis,
    filter_future_fixtures,
    get_next_fixture_round,
    load_fixture_seed,
)
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _format_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/d"


def _format_score(value: object, suffix: str = "/100") -> str:
    try:
        if value is None or pd.isna(value):
            return "n/d"
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return "n/d"


def _display_fixture_table(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    if fixtures_df.empty:
        return pd.DataFrame()
    display_df = fixtures_df.copy()
    display_df["match_date"] = pd.to_datetime(display_df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    columns = ["matchday", "match_date", "home_team", "away_team", "source_name"]
    existing = [column for column in columns if column in display_df.columns]
    return display_df[existing].rename(
        columns={
            "matchday": "Giornata",
            "match_date": "Data",
            "home_team": "Casa",
            "away_team": "Trasferta",
            "source_name": "Fonte",
        }
    )


def _display_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    display_df = summary_df.copy()
    for column in ["Prob. 1 base", "Prob. X base", "Prob. 2 base", "Prob. 1 v2", "Prob. X v2", "Prob. 2 v2"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(_format_pct)
    for column in ["draw_risk", "upset_risk", "confidence"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: _format_score(value))
    return display_df


st.set_page_config(page_title=f"{APP_TITLE} | Calendario", layout="wide")
bootstrap_database()

st.title("Calendario")
st.write("Prossima giornata da fixture seed e sintesi statistica dei match disponibili.")
if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE) or list_seasons()
if not seasons:
    st.warning("Nessuna stagione disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Stagione", seasons)
season_df = fetch_matches(selected_season, competition_code=DEFAULT_COMPETITION_CODE)
schedule_df = fetch_matches(selected_season)
if season_df.empty:
    st.warning("La stagione selezionata non contiene partite di Serie A.")
    st.stop()
if schedule_df.empty:
    schedule_df = season_df

fixture_report = build_fixture_seed_report(season=selected_season, results_df=schedule_df)
future_fixtures = filter_future_fixtures(load_fixture_seed(), schedule_df, season=selected_season)
next_round = get_next_fixture_round(future_fixtures)

source_col1, source_col2, source_col3 = st.columns(3)
source_col1.metric("Fonte usata", "Fixture seed" if not next_round.empty else "Non disponibile")
source_col2.metric("Fixture future valide", int(fixture_report.get("future_fixture_count", 0) or 0))
source_col3.metric("Prossima data", fixture_report.get("next_fixture_date") or "n/d")

if next_round.empty:
    st.warning(
        fixture_report.get("message")
        or "Fixture seed assente o senza partite future valide. Aggiorna il fixture seed per mostrare il calendario reale."
    )
else:
    matchday = next_round["matchday"].dropna().iloc[0] if "matchday" in next_round.columns and next_round["matchday"].dropna().any() else None
    st.subheader(f"Prossima giornata{f' {int(matchday)}' if matchday is not None else ''}")
    st.dataframe(_display_fixture_table(next_round), width="stretch", hide_index=True)

    analysis = build_round_analysis(
        season_df,
        fixtures_df=next_round,
        season=selected_season,
        schedule_df=schedule_df,
    )
    if not analysis.get("ok"):
        st.warning(analysis.get("message", "Analisi giornata non disponibile."))
    else:
        headline = analysis.get("headline_summary", {}) or {}
        cards = headline.get("cards", {}) or {}
        st.subheader("Sintesi giornata")
        st.write(headline.get("headline") or analysis.get("round_summary") or "Sintesi non disponibile.")

        card1, card2, card3, card4 = st.columns(4)
        card1.metric("Più equilibrata", cards.get("partita_piu_equilibrata") or "n/d")
        card2.metric("Rischio pareggio relativo", cards.get("piu_alto_draw_risk") or "n/d")
        card3.metric("Rischio upset relativo", cards.get("piu_alto_upset_risk") or "n/d")
        card4.metric("Più stabile", cards.get("partita_piu_stabile") or "n/d")

        st.subheader("Riepilogo partite")
        st.dataframe(_display_summary_table(analysis.get("summary_table", pd.DataFrame())), width="stretch", hide_index=True)

        sources = next_round.get("source_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        st.caption("Fixture source: " + (", ".join(sources) if sources else "fixture_seed"))

st.divider()
with st.expander("Simulazione partite mancanti inferite", expanded=False):
    st.warning("Questa non è la prossima giornata ufficiale: sono partite mancanti inferite dal calendario home/away.")
    inferred_df = infer_remaining_fixtures(season_df)
    if inferred_df.empty:
        st.info("Nessuna partita mancante inferibile.")
    else:
        st.dataframe(
            inferred_df.head(10).rename(columns={"home_team": "Casa", "away_team": "Trasferta"}),
            width="stretch",
            hide_index=True,
        )
