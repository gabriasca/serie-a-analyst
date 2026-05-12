from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - fallback difensivo per Streamlit Cloud
    config = None

from src.analytics import build_standings
from src.data_freshness import build_data_freshness_report
from src.db import fetch_matches, get_database_status, list_seasons
from src.round_analysis import build_fixture_seed_report
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _format_sources(sources: object) -> str:
    names: list[str] = []
    for source in _safe_list(sources):
        if isinstance(source, dict):
            name = source.get("source_name") or "Fonte dati"
            if name:
                names.append(str(name))
        elif source:
            names.append(str(source))
    return ", ".join(dict.fromkeys(names)) if names else "seed_csv"


def _standings_for_display(df: pd.DataFrame) -> pd.DataFrame:
    table = build_standings(df)
    if table.empty:
        return table
    return table.reset_index().rename(
        columns={
            "Team": "Squadra",
            "GP": "Partite",
            "Pts": "Punti",
            "DR": "Diff. reti",
        }
    )


def _freshness_message(status: str, message: str) -> None:
    if status == "ok":
        st.success(message)
    elif status == "database_vuoto":
        st.warning(message)
    elif status == "dati_parziali":
        st.info(message)
    else:
        st.warning(message)


st.set_page_config(page_title=f"{APP_TITLE} | Home / Classifica", layout="wide")

status_error = None
try:
    bootstrap_database()
    db_status = get_database_status()
    seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE) or list_seasons()
    all_matches_df = fetch_matches()
    freshness_report = build_data_freshness_report(all_matches_df)
    fixture_report = build_fixture_seed_report(results_df=all_matches_df)
except Exception as exc:  # pragma: no cover - difesa per runtime cloud/cache
    db_status = {}
    seasons = []
    all_matches_df = pd.DataFrame()
    freshness_report = build_data_freshness_report(pd.DataFrame())
    fixture_report = {
        "path_exists": False,
        "future_fixture_count": 0,
        "next_fixture_date": None,
        "next_matchday": None,
        "source_names": [],
    }
    status_error = str(exc)

st.title("Home / Classifica")
st.caption(APP_TITLE)
st.write(
    "Classifica, calendario, proiezioni e previsione partita in una versione pubblica più pulita. "
    "I motori analitici avanzati restano attivi dietro le quinte."
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)
if status_error:
    st.warning("Alcune informazioni di stato non sono disponibili in questo momento.")

if not seasons:
    st.warning("Nessuna stagione disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Stagione", seasons)
season_df = fetch_matches(selected_season, competition_code=DEFAULT_COMPETITION_CODE)
if season_df.empty:
    season_df = fetch_matches(selected_season)

st.subheader("Classifica Serie A")
standings_df = _standings_for_display(season_df)
if standings_df.empty:
    st.info("Classifica non disponibile: la stagione selezionata non contiene partite.")
else:
    st.dataframe(standings_df, width="stretch", hide_index=True)

st.subheader("Stato aggiornamento dati")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Partite caricate", _safe_int(freshness_report.get("match_count")))
metric2.metric("Ultima data partita", freshness_report.get("latest_match_date") or "n/d")
metric3.metric("Squadre", _safe_int(freshness_report.get("team_count")))
metric4.metric("Stagioni", _safe_int(freshness_report.get("season_count")))

_freshness_message(
    str(freshness_report.get("freshness_status") or "attenzione"),
    str(freshness_report.get("freshness_message") or freshness_report.get("freshness_summary") or "Stato dati non disponibile."),
)
st.caption("Fonti dati: " + _format_sources(db_status.get("sources", [])))

st.subheader("Prossima giornata")
fix1, fix2, fix3, fix4 = st.columns(4)
fix1.metric("Fixture seed", "Presente" if fixture_report.get("path_exists") else "Assente")
fix2.metric("Fixture future valide", _safe_int(fixture_report.get("future_fixture_count")))
fix3.metric("Prossima data", fixture_report.get("next_fixture_date") or "n/d")
fix4.metric("Giornata", fixture_report.get("next_matchday") or "n/d")

fixture_sources = fixture_report.get("source_names") or []
if fixture_sources:
    st.caption("Fonte fixture: " + ", ".join(str(source) for source in fixture_sources))
if fixture_report.get("path_exists") and not fixture_report.get("future_fixture_count"):
    st.warning("Fixture seed presente ma senza fixture future valide.")
elif not fixture_report.get("path_exists"):
    st.info("Fixture seed non presente: il Calendario ufficiale non è disponibile.")

st.subheader("Navigazione")
st.write("Usa le pagine pubbliche principali:")
st.markdown(
    "- **Calendario**: prossima giornata da fixture seed e sintesi dei match.\n"
    "- **Proiezione Classifica**: simulazione finale basata sul predictor base.\n"
    "- **Previsione Partita**: previsione numerica, lettura contestuale e report discorsivo."
)

st.caption("Le previsioni sono stime statistiche basate sui dati disponibili, non certezze.")
