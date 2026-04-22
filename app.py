from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import get_database_status
from src.seed_data import bootstrap_database


st.set_page_config(page_title=APP_TITLE, layout="wide")

bootstrap_database()
db_status = get_database_status()

st.title(APP_TITLE)
st.write(
    """
    Una web app locale per importare dati della Serie A da CSV, salvarli in SQLite,
    calcolare statistiche di campionato e generare previsioni semplici e spiegabili.
    """
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Database", "Pronto" if db_status["database_ready"] else "Non inizializzato")
col2.metric("Partite caricate", db_status["match_count"])
col3.metric("Squadre", db_status["team_count"])
col4.metric("Stagioni", len(db_status.get("seasons", [])))

st.subheader("Stato del progetto")
st.write(
    f"Stagioni disponibili: "
    f"{', '.join(db_status.get('seasons', [])) if db_status.get('seasons', []) else 'nessuna'}"
)
st.write(
    "Usa il menu laterale per importare nuovi CSV, esplorare la dashboard del campionato, "
    "analizzare una squadra, confrontarne due o stimare una partita."
)

if db_status["match_count"] == 0:
    st.info(
        "Il database e vuoto. Vai alla pagina Import Dati per caricare un CSV reale "
        "oppure il dataset demo di test."
    )

if db_status.get("competitions", []):
    st.write("Competizioni presenti:")
    st.dataframe(pd.DataFrame(db_status.get("competitions", [])), use_container_width=True)
else:
    st.write("Competizioni presenti: nessuna")

if db_status.get("sources", []):
    sources_df = pd.DataFrame(db_status.get("sources", []))
    st.write("Fonti dati rilevate:")
    st.dataframe(sources_df, use_container_width=True)
else:
    st.write("Fonti dati rilevate: nessuna")

st.warning(
    "Le previsioni mostrate nell'app sono stime statistiche basate sui dati disponibili, "
    "non certezze."
)

st.caption(
    "MVP locale costruito con Streamlit, Pandas e SQLite. La sezione futura "
    '"chiedi all\'analista" potra riusare la stessa base dati e gli stessi moduli analitici.'
)
