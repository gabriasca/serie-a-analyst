from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback for cloud/runtime issues
    config = None

from src.data_freshness import build_data_freshness_report
from src.db import fetch_matches, get_database_status
from src.round_analysis import build_fixture_seed_report
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def safe_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_status(status: object) -> dict[str, object]:
    if not isinstance(status, dict):
        status = {}
    return {
        "database_ready": bool(status.get("database_ready", False)),
        "match_count": safe_int(status.get("match_count", 0)),
        "team_count": safe_int(status.get("team_count", 0)),
        "season_count": safe_int(status.get("season_count", 0)),
        "seasons": safe_list(status.get("seasons", [])),
        "sources": safe_list(status.get("sources", [])),
        "competitions": safe_list(status.get("competitions", [])),
    }


def format_competitions(competitions: object) -> str:
    labels: list[str] = []
    for comp in safe_list(competitions):
        if isinstance(comp, dict):
            name = comp.get("competition_name") or comp.get("competition_code") or "Competizione"
            count = comp.get("match_count")
            labels.append(f"{name} ({count})" if count is not None else str(name))
        else:
            labels.append(str(comp))
    return ", ".join(labels) if labels else "nessuna"


def format_sources(sources: object) -> str:
    labels: list[str] = []
    for source in safe_list(sources):
        if isinstance(source, dict):
            name = source.get("source_name") or "Fonte dati"
            count = source.get("match_count")
            labels.append(f"{name} ({count})" if count is not None else str(name))
        else:
            labels.append(str(source))
    return ", ".join(labels) if labels else "nessuna"


def render_freshness_message(status: object, message: object) -> None:
    status_text = str(status or "attenzione")
    message_text = str(message or "Stato aggiornamento dati non disponibile.")
    if status_text == "ok":
        st.success(message_text)
    elif status_text == "database_vuoto":
        st.warning(message_text)
    elif status_text == "dati_parziali":
        st.info(message_text)
    else:
        st.warning(message_text)


st.set_page_config(page_title=APP_TITLE, layout="wide")

status_error = None
try:
    bootstrap_database()
    db_status = safe_status(get_database_status())
    freshness_report = build_data_freshness_report(fetch_matches())
    fixture_seed_report = build_fixture_seed_report()
except Exception as exc:  # pragma: no cover - defensive fallback for cloud/runtime issues
    db_status = safe_status({})
    freshness_report = build_data_freshness_report(pd.DataFrame())
    fixture_seed_report = {
        "available": False,
        "path_exists": False,
        "fixture_count": 0,
        "next_fixture_date": None,
        "next_matchday": None,
        "source_names": [],
    }
    status_error = str(exc)

seasons = safe_list(db_status.get("seasons", []))

st.title(APP_TITLE)
st.write(
    """
    Una web app locale per importare dati della Serie A da CSV, salvarli in SQLite,
    calcolare statistiche di campionato e generare previsioni semplici e spiegabili.
    """
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)
if status_error:
    st.warning("Stato database non completamente disponibile in questo momento.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Database", "Pronto" if db_status.get("database_ready", False) else "Non inizializzato")
col2.metric("Partite caricate", db_status.get("match_count", 0))
col3.metric("Squadre", db_status.get("team_count", 0))
col4.metric("Stagioni", len(seasons))

st.subheader("Stato del progetto")
st.write(f"Stagioni disponibili: {', '.join(str(season) for season in seasons) if seasons else 'nessuna'}")
st.write(
    "Usa il menu laterale per importare nuovi CSV, esplorare la dashboard del campionato, "
    "analizzare una squadra, confrontarne due o stimare una partita."
)

if db_status.get("match_count", 0) == 0:
    st.info(
        "Il database e vuoto. Vai alla pagina Import Dati per caricare un CSV reale "
        "oppure il dataset demo di test."
    )

st.write("Competizioni presenti: " + format_competitions(db_status.get("competitions", [])))
st.write("Fonti dati rilevate: " + format_sources(db_status.get("sources", [])))

st.subheader("Stato aggiornamento dati")
fresh_col1, fresh_col2, fresh_col3, fresh_col4 = st.columns(4)
fresh_col1.metric("Ultima data partita", freshness_report.get("latest_match_date") or "n/d")
fresh_col2.metric("Partite caricate", freshness_report.get("match_count", 0))
fresh_col3.metric("Partite teoriche totali", freshness_report.get("expected_total_matches", 0))
fresh_col4.metric("Mancanti stimate", freshness_report.get("missing_matches_estimate", 0))
st.write("Fonti dati freshness: " + ", ".join(str(source) for source in freshness_report.get("source_names", [])))
render_freshness_message(
    freshness_report.get("freshness_status"),
    freshness_report.get("freshness_message") or freshness_report.get("freshness_summary"),
)

st.subheader("Fixture prossima giornata")
fixture_col1, fixture_col2, fixture_col3 = st.columns(3)
fixture_col1.metric("Fixture seed", "Presente" if fixture_seed_report.get("available") else "Assente")
fixture_col2.metric("Prossima data fixture", fixture_seed_report.get("next_fixture_date") or "n/d")
fixture_col3.metric("Fixture nel seed", fixture_seed_report.get("fixture_count", 0))
if fixture_seed_report.get("next_matchday"):
    st.write(f"Prossima giornata fixture: {fixture_seed_report.get('next_matchday')}")
sources = fixture_seed_report.get("source_names") or []
if sources:
    st.write("Fonte fixture: " + ", ".join(str(source) for source in sources))
else:
    st.caption("Fixture seed non presente: Analisi Giornata usera il fallback inferito.")

st.warning(
    "Le previsioni mostrate nell'app sono stime statistiche basate sui dati disponibili, "
    "non certezze."
)

st.caption(
    "MVP locale costruito con Streamlit, Pandas e SQLite. La sezione futura "
    '"chiedi all\'analista" potra riusare la stessa base dati e gli stessi moduli analitici.'
)
