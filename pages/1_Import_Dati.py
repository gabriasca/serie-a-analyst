from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback for cloud/runtime issues
    config = None

from src.data_import import (
    clean_match_data,
    load_csv_to_dataframe,
    save_dataframe_to_sqlite,
    validate_required_columns,
)
from src.data_freshness import build_data_freshness_report
from src.db import delete_all_matches, delete_matches_by_season, fetch_matches, get_database_status
from src.demo_data import load_demo_data
from src.round_analysis import build_fixture_seed_report
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
DEFAULT_COMPETITION_NAME = getattr(config, "DEFAULT_COMPETITION_NAME", "Serie A")
DEFAULT_COMPETITION_TYPE = getattr(config, "DEFAULT_COMPETITION_TYPE", "league")


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


st.set_page_config(page_title=f"{APP_TITLE} | Import Dati", layout="wide")

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

st.title("Import Dati")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)
    st.write("Questa pagina e informativa nella versione pubblica.")
    st.subheader("Stato database")
    if status_error:
        st.warning("Stato database non completamente disponibile in questo momento.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Partite caricate", db_status.get("match_count", 0))
    col2.metric("Squadre", db_status.get("team_count", 0))
    col3.metric("Stagioni", len(seasons))

    st.write(f"Stagioni presenti: {', '.join(str(season) for season in seasons) if seasons else 'nessuna'}")
    st.write("Competizioni presenti: " + format_competitions(db_status.get("competitions", [])))
    st.write("Fonti dati: " + format_sources(db_status.get("sources", [])))

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
    fixture_sources = fixture_seed_report.get("source_names") or []
    if fixture_sources:
        st.write("Fonte fixture: " + ", ".join(str(source) for source in fixture_sources))
    else:
        st.caption("Fixture seed non presente: Analisi Giornata usera il fallback inferito.")

    st.subheader("Ultimi match caricati")
    latest_matches = freshness_report.get("latest_matches")
    if isinstance(latest_matches, pd.DataFrame) and not latest_matches.empty:
        st.dataframe(latest_matches, width="stretch")
    else:
        st.caption("Nessun match disponibile da mostrare.")

    st.info(
        "Questa versione pubblica e consultabile. "
        "Gli aggiornamenti dati vengono fatti dall'autore aggiornando il CSV seed."
    )

    if db_status.get("match_count", 0) == 0:
        st.warning("Il database e vuoto. La versione pubblica richiede una snapshot seed pubblicata dall'autore.")

    st.stop()

st.write("Gestisci il database locale, carica il dataset demo per test oppure importa un CSV reale.")
if status_error:
    st.warning("Stato database non completamente disponibile in questo momento.")

page_message = st.session_state.pop("import_data_message", None)
if isinstance(page_message, dict):
    level = page_message.get("level", "info")
    text = page_message.get("text", "")
    getattr(st, level if hasattr(st, level) else "info")(text)

st.subheader("Gestione database")
col1, col2, col3 = st.columns(3)
col1.metric("Partite totali", db_status.get("match_count", 0))
col2.metric("Squadre", db_status.get("team_count", 0))
col3.metric("Stagioni", len(seasons))

st.write(f"Stagioni presenti: {', '.join(str(season) for season in seasons) if seasons else 'nessuna'}")

competitions = safe_list(db_status.get("competitions", []))
if competitions:
    st.write("Competizioni presenti:")
    st.dataframe(pd.DataFrame(competitions), use_container_width=True)
else:
    st.write("Competizioni presenti: nessuna")

sources = safe_list(db_status.get("sources", []))
if sources:
    st.write("Fonti dati presenti:")
    st.dataframe(pd.DataFrame(sources), use_container_width=True)
else:
    st.write("Fonti dati presenti: nessuna")

st.info(
    "Il dataset demo serve solo per testare l'app localmente. "
    "Per evitare mix con dati reali, viene caricato solo con un'azione manuale."
)

if st.button("Carica dataset demo"):
    if db_status.get("match_count", 0) > 0:
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
if seasons:
    season_to_delete = st.selectbox("Seleziona stagione da eliminare", seasons)
else:
    season_to_delete = None
    st.selectbox("Seleziona stagione da eliminare", ["Nessuna stagione disponibile"], disabled=True)

confirm_delete_season = st.checkbox(
    "Confermo di voler eliminare la stagione selezionata",
    key="confirm_delete_season",
)
if st.button(
    "Elimina stagione selezionata",
    disabled=not (season_to_delete and confirm_delete_season and seasons),
):
    deleted_count = delete_matches_by_season(str(season_to_delete))
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

            overlapping_seasons = [season for season in detected_seasons if season in seasons]
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
