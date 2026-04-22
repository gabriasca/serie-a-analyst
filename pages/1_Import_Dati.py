from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    APP_TITLE,
    DEFAULT_COMPETITION_CODE,
    DEFAULT_COMPETITION_NAME,
    DEFAULT_COMPETITION_TYPE,
    PUBLIC_DEMO_BANNER,
    PUBLIC_DEMO_MODE,
)
from src.data_import import (
    clean_match_data,
    load_csv_to_dataframe,
    save_dataframe_to_sqlite,
    validate_required_columns,
)
from src.db import delete_all_matches, delete_matches_by_season, get_database_status
from src.demo_data import load_demo_data
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Import Dati", layout="wide")

bootstrap_database()
db_status = get_database_status()

st.title("Import Dati")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)
    st.write("Questa pagina e informativa nella versione pubblica.")
    st.subheader("Stato database")

    col1, col2, col3 = st.columns(3)
    col1.metric("Partite caricate", db_status["match_count"])
    col2.metric("Squadre", db_status["team_count"])
    col3.metric("Stagioni", len(db_status["seasons"]))

    st.write(
        f"Stagioni presenti: {', '.join(db_status['seasons']) if db_status['seasons'] else 'nessuna'}"
    )

    if db_status["competitions"]:
        st.write("Competizioni presenti:")
        st.dataframe(pd.DataFrame(db_status["competitions"]), use_container_width=True)
    else:
        st.write("Competizioni presenti: nessuna")

    if db_status["sources"]:
        st.write("Fonti dati:")
        st.dataframe(pd.DataFrame(db_status["sources"]), use_container_width=True)
    else:
        st.write("Fonti dati: nessuna")

    st.info(
        "Questa versione pubblica e consultabile. "
        "Gli aggiornamenti dati vengono fatti dall'autore aggiornando il CSV seed."
    )

    if db_status["match_count"] == 0:
        st.warning("Il database e vuoto. La versione pubblica richiede una snapshot seed pubblicata dall'autore.")

    st.stop()

st.write("Gestisci il database locale, carica il dataset demo per test oppure importa un CSV reale.")

page_message = st.session_state.pop("import_data_message", None)
if page_message:
    getattr(st, page_message["level"])(page_message["text"])

st.subheader("Gestione database")
col1, col2, col3 = st.columns(3)
col1.metric("Partite totali", db_status["match_count"])
col2.metric("Squadre", db_status["team_count"])
col3.metric("Stagioni", len(db_status["seasons"]))

st.write(
    f"Stagioni presenti: {', '.join(db_status['seasons']) if db_status['seasons'] else 'nessuna'}"
)

if db_status["competitions"]:
    st.write("Competizioni presenti:")
    st.dataframe(pd.DataFrame(db_status["competitions"]), use_container_width=True)
else:
    st.write("Competizioni presenti: nessuna")

if db_status["sources"]:
    st.write("Fonti dati presenti:")
    st.dataframe(pd.DataFrame(db_status["sources"]), use_container_width=True)
else:
    st.write("Fonti dati presenti: nessuna")

st.info(
    "Il dataset demo serve solo per testare l'app localmente. "
    "Per evitare mix con dati reali, viene caricato solo con un'azione manuale."
)

if st.button("Carica dataset demo"):
    if db_status["match_count"] > 0:
        st.session_state["import_data_message"] = {
            "level": "warning",
            "text": (
                "Il database contiene gia dati. Per evitare di mischiare demo e dati reali, "
                "svuota il database o elimina prima le stagioni presenti."
            ),
        }
    else:
        demo_stats = load_demo_data(force=False)
        st.session_state["import_data_message"] = {
            "level": "success",
            "text": (
                f"Dataset demo caricato: {demo_stats['inserted']} partite inserite, "
                f"{demo_stats['duplicates']} duplicati ignorati."
            ),
        }
    st.rerun()

st.markdown("---")
st.write("Svuota solo i dati della tabella `matches`, senza toccare file o struttura del progetto.")
confirm_delete_all = st.checkbox(
    "Confermo di voler cancellare tutte le partite dal database",
    key="confirm_delete_all",
)
if st.button("Svuota database", disabled=not confirm_delete_all):
    deleted_count = delete_all_matches()
    st.session_state["import_data_message"] = {
        "level": "success",
        "text": f"Database svuotato: eliminate {deleted_count} partite dalla tabella matches.",
    }
    st.rerun()

st.markdown("---")
if db_status["seasons"]:
    season_to_delete = st.selectbox("Seleziona stagione da eliminare", db_status["seasons"])
else:
    season_to_delete = None
    st.selectbox("Seleziona stagione da eliminare", ["Nessuna stagione disponibile"], disabled=True)

confirm_delete_season = st.checkbox(
    "Confermo di voler eliminare la stagione selezionata",
    key="confirm_delete_season",
)
if st.button(
    "Elimina stagione selezionata",
    disabled=not (season_to_delete and confirm_delete_season and db_status["seasons"]),
):
    deleted_count = delete_matches_by_season(season_to_delete)
    st.session_state["import_data_message"] = {
        "level": "success",
        "text": (
            f"Stagione {season_to_delete} eliminata dal database: "
            f"{deleted_count} partite rimosse."
        ),
    }
    st.rerun()

st.markdown("---")
st.subheader("Import CSV reale")
st.write("Carica un CSV locale, verifica le colonne disponibili e salva le partite in SQLite.")

season_fallback = st.text_input(
    "Stagione da applicare se manca nel CSV",
    value="",
    placeholder="Esempio: 2024-2025",
)

comp_col1, comp_col2, comp_col3 = st.columns(3)
competition_code_input = comp_col1.text_input(
    "competition_code di default",
    value=DEFAULT_COMPETITION_CODE,
)
competition_name_input = comp_col2.text_input(
    "competition_name di default",
    value=DEFAULT_COMPETITION_NAME,
)
competition_type_input = comp_col3.text_input(
    "competition_type di default",
    value=DEFAULT_COMPETITION_TYPE,
)

uploaded_file = st.file_uploader("Seleziona un file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = load_csv_to_dataframe(uploaded_file)
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"Errore durante la lettura del CSV: {exc}")
    else:
        validation = validate_required_columns(raw_df, provided_season=season_fallback)

        st.subheader("Anteprima dati originali")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("Mappa colonne rilevata")
        st.json(validation["rename_map"])

        if validation["missing_columns"]:
            st.error("Colonne obbligatorie mancanti: " + ", ".join(validation["missing_columns"]))
        else:
            cleaned_df = clean_match_data(
                raw_df,
                default_season=season_fallback,
                source_name=uploaded_file.name,
                default_competition_code=competition_code_input.strip() or DEFAULT_COMPETITION_CODE,
                default_competition_name=competition_name_input.strip() or DEFAULT_COMPETITION_NAME,
                default_competition_type=competition_type_input.strip() or DEFAULT_COMPETITION_TYPE,
            )
            st.success("Validazione completata con successo.")
            st.subheader("Anteprima dati puliti")
            st.dataframe(cleaned_df.head(10), use_container_width=True)

            detected_seasons = sorted(cleaned_df["season"].dropna().astype(str).unique().tolist())
            detected_competitions = (
                cleaned_df[["competition_code", "competition_name", "competition_type"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            if detected_seasons:
                season_label = (
                    "Stagione che verra salvata"
                    if len(detected_seasons) == 1
                    else "Stagioni che verranno salvate"
                )
                st.info(f"{season_label}: {', '.join(detected_seasons)}")

            if not detected_competitions.empty:
                st.write("Competizione/i che verranno salvate:")
                st.dataframe(detected_competitions, use_container_width=True)

            overlapping_seasons = [
                season for season in detected_seasons if season in db_status["seasons"]
            ]
            if overlapping_seasons:
                st.warning(
                    "Nel database esistono gia dati per: "
                    + ", ".join(overlapping_seasons)
                    + ". I nuovi record verranno aggiunti e gli eventuali duplicati saranno ignorati "
                    "grazie al vincolo UNIQUE."
                )
            elif detected_seasons:
                st.success("La stagione rilevata non e ancora presente nel database.")

            if cleaned_df.empty:
                st.warning("Dopo la pulizia non sono rimaste righe valide da importare.")
            elif st.button("Salva in SQLite"):
                save_stats = save_dataframe_to_sqlite(cleaned_df, source_name=uploaded_file.name)
                st.session_state["import_data_message"] = {
                    "level": "success",
                    "text": (
                        f"Import completato: {save_stats['inserted']} partite inserite, "
                        f"{save_stats['duplicates']} duplicati ignorati."
                    ),
                }
                st.rerun()
