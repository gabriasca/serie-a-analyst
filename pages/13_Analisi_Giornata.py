from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - fallback difensivo per Streamlit Cloud
    config = None

from src.db import fetch_matches, list_seasons
from src.round_analysis import (
    available_fixture_matchdays,
    build_round_analysis,
    infer_missing_fixtures_simulation,
    infer_next_round_fixtures,
    select_round_fixtures,
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


def _load_league_dataframe(season: str) -> pd.DataFrame:
    league_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
    if league_df.empty:
        league_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if league_df.empty:
        league_df = fetch_matches(season)
    return league_df


def _load_schedule_dataframe(season: str, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    schedule_df = fetch_matches(season)
    if schedule_df.empty and isinstance(fallback_df, pd.DataFrame):
        return fallback_df
    return schedule_df


def _format_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:.1f}%"


def _format_pp(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:+.1f} pp"


def _format_score_value(value: object, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _render_bullets(items: list[str]) -> None:
    if not items:
        st.write("- Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in items))


def _fixture_signature(fixtures_df: pd.DataFrame) -> str:
    if fixtures_df.empty:
        return "empty"
    parts: list[str] = []
    for row in fixtures_df.to_dict(orient="records"):
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        date_label = match_date.strftime("%Y-%m-%d") if pd.notna(match_date) else ""
        parts.append(
            "|".join(
                [
                    str(row.get("season") or ""),
                    date_label,
                    str(row.get("matchday") or ""),
                    str(row.get("home_team") or ""),
                    str(row.get("away_team") or ""),
                    str(row.get("fixture_source") or ""),
                ]
            )
        )
    return "||".join(parts)


def _display_fixture_table(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    display_df = fixtures_df.copy()
    if display_df.empty:
        return display_df
    display_df["match_date"] = pd.to_datetime(display_df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    columns = [
        column
        for column in ["match_date", "matchday", "competition_name", "home_team", "away_team", "source_name"]
        if column in display_df.columns
    ]
    return display_df[columns].rename(
        columns={
            "match_date": "Data",
            "matchday": "Giornata",
            "competition_name": "Competizione",
            "home_team": "Casa",
            "away_team": "Trasferta",
            "source_name": "Fonte",
        }
    )


def _display_summary_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_table.copy()
    if display_df.empty:
        return display_df
    pct_columns = ["Prob base 1", "Prob base X", "Prob base 2", "Prob cont 1", "Prob cont X", "Prob cont 2"]
    for column in pct_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(_format_pct)
    for column in ["Confidence", "Draw risk", "Upset risk"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: _format_score_value(value, digits=1, suffix="/100"))
    preferred_columns = [
        "Partita",
        "Giornata",
        "livello_lettura",
        "Prob base 1",
        "Prob base X",
        "Prob base 2",
        "Prob cont 1",
        "Prob cont X",
        "Prob cont 2",
        "Risultato piu probabile",
        "Confidence",
        "Draw risk",
        "Upset risk",
        "Volatilita",
        "Scenario principale breve",
        "Dato affidabile",
        "Dato fragile",
        "Interesse match",
        "Tipo match",
    ]
    columns = [column for column in preferred_columns if column in display_df.columns]
    return display_df[columns]


def _display_rank_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if "valore" in table.columns:
        table["valore"] = table["valore"].map(lambda value: _format_score_value(value, digits=1))
    return table.rename(
        columns={
            "partita": "Partita",
            "metrica": "Metrica",
            "valore": "Valore",
            "livello_lettura": "Livello lettura",
            "scenario": "Scenario",
        }
    )


def _render_overview_card(label: str, value: object, help_text: str = "") -> None:
    st.markdown(f"**{label}**")
    st.write(value or "n/d")
    if help_text:
        st.caption(help_text)


def _render_round_overview(analysis: dict[str, object]) -> None:
    headline_summary = analysis.get("headline_summary", {}) or {}
    cards = headline_summary.get("cards", {}) if isinstance(headline_summary, dict) else {}
    tables = headline_summary.get("tables", {}) if isinstance(headline_summary, dict) else {}

    st.subheader("Panoramica giornata")
    st.write(headline_summary.get("headline", "Panoramica non disponibile."))

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        _render_overview_card("Piu equilibrata", cards.get("partita_piu_equilibrata"), "Probabilita piu vicine tra loro.")
    with card2:
        _render_overview_card("Draw risk", cards.get("piu_alto_draw_risk"), "Rischio pareggio piu alto.")
    with card3:
        _render_overview_card("Upset risk", cards.get("piu_alto_upset_risk"), "Favorito piu esposto al contesto.")
    with card4:
        _render_overview_card("Confidence alta", cards.get("confidence_piu_alta"), "Lettura piu coerente.")

    card5, card6, card7 = st.columns(3)
    with card5:
        _render_overview_card("Confidence bassa", cards.get("confidence_piu_bassa"), "Match da leggere con piu prudenza.")
    with card6:
        _render_overview_card("Piu volatile", cards.get("partita_piu_volatile"), "Rischio di cambio lettura piu alto.")
    with card7:
        _render_overview_card("Piu stabile", cards.get("partita_piu_stabile"), "Segnali piu ordinati.")

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.markdown("#### Partite piu aperte")
        st.dataframe(_display_rank_table(tables.get("partite_piu_aperte", [])), use_container_width=True)
        st.markdown("#### Alto draw risk")
        st.dataframe(_display_rank_table(tables.get("alto_draw_risk", [])), use_container_width=True)
    with table_col2:
        st.markdown("#### Alto upset risk")
        st.dataframe(_display_rank_table(tables.get("alto_upset_risk", [])), use_container_width=True)
        st.markdown("#### Bassa confidence")
        st.dataframe(_display_rank_table(tables.get("bassa_confidence", [])), use_container_width=True)


def _render_probability_metrics(title: str, probabilities: dict[str, object]) -> None:
    st.markdown(f"**{title}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("1", _format_pct(probabilities.get("1")))
    col2.metric("X", _format_pct(probabilities.get("X")))
    col3.metric("2", _format_pct(probabilities.get("2")))


def _render_delta_table(delta_rows: list[dict[str, object]]) -> None:
    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        st.caption("Differenze non disponibili.")
        return
    delta_df = delta_df.rename(
        columns={
            "label": "Esito",
            "base_probability": "Base",
            "contextual_probability": "Contestuale",
            "delta": "Differenza",
        }
    )
    delta_df["Base"] = delta_df["Base"].map(_format_pct)
    delta_df["Contestuale"] = delta_df["Contestuale"].map(_format_pct)
    delta_df["Differenza"] = delta_df["Differenza"].map(_format_pp)
    st.dataframe(delta_df[["Esito", "Base", "Contestuale", "Differenza"]], use_container_width=True)


def _render_match_detail(match: dict[str, object]) -> None:
    prediction = match.get("prediction", {}) or {}
    contextual = match.get("contextual_forecast", {}) or {}

    st.markdown("#### Previsione base")
    if prediction.get("ok"):
        base_col1, base_col2, base_col3 = st.columns(3)
        base_col1.metric("Gol attesi modello casa", _format_score_value(prediction.get("expected_goals_home"), digits=2))
        base_col2.metric("Gol attesi modello trasferta", _format_score_value(prediction.get("expected_goals_away"), digits=2))
        base_col3.metric("Risultato piu probabile", prediction.get("most_likely_score") or "n/d")
        _render_probability_metrics("Probabilita base 1/X/2", prediction.get("probabilities", {}))
        st.caption("Sono gol attesi dal modello Poisson interno, non xG reali shot-by-shot.")
    else:
        st.warning(prediction.get("message") or "Predictor base non disponibile per questa partita.")

    st.markdown("#### Lettura contestuale v2")
    _render_probability_metrics("Probabilita contestuali 1/X/2", contextual.get("contextual_probabilities", {}))
    _render_delta_table(contextual.get("probability_deltas", []))
    edge_col1, edge_col2, edge_col3, edge_col4 = st.columns(4)
    edge_col1.metric("Adjusted edge", _format_score_value(contextual.get("adjusted_edge"), digits=2))
    edge_col2.metric("Draw risk", _format_score_value(contextual.get("draw_risk"), digits=1, suffix="/100"))
    edge_col3.metric("Upset risk", _format_score_value(contextual.get("upset_risk"), digits=1, suffix="/100"))
    edge_col4.metric("Confidence", _format_score_value(contextual.get("confidence"), digits=1, suffix="/100"))

    st.markdown("#### Scenario principale")
    st.write(match.get("scenario_principale") or "Scenario principale non disponibile.")

    st.markdown("#### Scenario alternativo")
    st.write(match.get("scenario_alternativo") or "Scenario alternativo non disponibile.")

    st.markdown("#### Cosa puo cambiare la partita")
    _render_bullets(match.get("cosa_puo_cambiare", []))

    st.markdown("#### Dato piu affidabile")
    st.info(match.get("dato_piu_affidabile") or "Dato affidabile non disponibile.")

    st.markdown("#### Dato piu fragile")
    st.warning(match.get("dato_piu_fragile") or "Dato fragile non disponibile.")

    if match.get("key_factors"):
        st.markdown("#### Fattori chiave di supporto")
        _render_bullets(match.get("key_factors", []))

    st.markdown("#### Dati mancanti")
    _render_bullets(match.get("missing_data_notes", []))

    for warning in contextual.get("warnings", []):
        st.info(warning)


st.set_page_config(page_title=f"{APP_TITLE} | Analisi Giornata", layout="wide")

bootstrap_database()

st.title("Analisi Giornata")
st.caption("Analisi statistica basata sui dati disponibili. Non e una certezza e non usa quote.")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    seasons = list_seasons()

if not seasons:
    st.warning("Database vuoto o nessuna stagione disponibile.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
league_df = _load_league_dataframe(selected_season)
if league_df.empty:
    st.warning("La stagione selezionata non contiene partite Serie A utilizzabili.")
    st.stop()

analysis_mode = st.radio(
    "Modalita analisi",
    ["Prossima giornata da fixture seed", "Simulazione partite mancanti inferite"],
    index=0,
)

if analysis_mode == "Prossima giornata da fixture seed":
    candidate_fixtures = infer_next_round_fixtures(league_df, season=selected_season)
else:
    candidate_fixtures = infer_missing_fixtures_simulation(league_df, season=selected_season)

source_label = candidate_fixtures.attrs.get("source_label", "Partite mancanti inferite")
source_mode = candidate_fixtures.attrs.get("fixture_source", "inferred_missing")
fixture_status = candidate_fixtures.attrs.get("fixture_status", source_mode)
fixture_message = candidate_fixtures.attrs.get("message")
warnings = candidate_fixtures.attrs.get("warnings", [])

st.subheader("Fonte partite")
if analysis_mode == "Prossima giornata da fixture seed" and source_mode == "fixture_seed":
    st.success("Fonte partite: fixture seed disponibile.")
elif analysis_mode == "Prossima giornata da fixture seed":
    st.error("Fixture seed non disponibile o senza fixture future valide.")
else:
    st.warning("Modalita simulazione: partite mancanti inferite, non calendario ufficiale.")
st.write(f"Fonte usata: **{source_label}**")
if fixture_message:
    st.info(str(fixture_message))
for warning in warnings:
    st.info(warning)

if candidate_fixtures.empty:
    if analysis_mode == "Prossima giornata da fixture seed":
        st.warning("Aggiorna fixture seed da GitHub Actions oppure controlla data/raw/serie_a_fixtures_seed.csv.")
        if fixture_status == "no_valid_future_fixtures":
            st.error("Fixture seed presente ma senza partite future utilizzabili. Non mostro partite inferite in modalita giornata reale.")
        else:
            st.error("Fixture seed assente o non valido. Non mostro partite inferite in modalita giornata reale.")
    else:
        st.warning("Nessuna partita mancante inferibile disponibile per costruire la simulazione.")
    st.stop()

if analysis_mode == "Simulazione partite mancanti inferite":
    st.error("Questa non e la prossima giornata ufficiale. Sono partite mancanti inferite.")

matchdays = available_fixture_matchdays(candidate_fixtures)
selected_matchday: int | None = None
if matchdays:
    selected_matchday = st.selectbox("Seleziona giornata", matchdays, format_func=lambda value: f"Giornata {value}")
    round_fixtures = select_round_fixtures(candidate_fixtures, matchday=selected_matchday)
else:
    if source_mode == "fixture_seed":
        st.info("Matchday non disponibile nel fixture seed: mostro il primo blocco di fixture ordinate per data.")
    else:
        st.warning("Matchday non disponibile: mostro partite mancanti inferite nella simulazione.")
    round_fixtures = select_round_fixtures(candidate_fixtures)

if round_fixtures.empty:
    st.warning("Nessuna partita disponibile per la selezione corrente.")
    st.stop()

st.subheader("Partite selezionate")
st.dataframe(_display_fixture_table(round_fixtures), use_container_width=True)

fixture_signature = _fixture_signature(round_fixtures)
if st.button("Genera analisi giornata"):
    schedule_df = _load_schedule_dataframe(selected_season, fallback_df=league_df)
    with st.spinner("Genero predictor base, lettura contestuale e narrativa partita per partita..."):
        analysis = build_round_analysis(
            league_df,
            fixtures_df=round_fixtures,
            season=selected_season,
            schedule_df=schedule_df,
        )
    st.session_state["round_analysis_result"] = {
        "season": selected_season,
        "mode": analysis_mode,
        "signature": fixture_signature,
        "analysis": analysis,
    }

stored_result = st.session_state.get("round_analysis_result")
if not stored_result:
    st.info("Seleziona la giornata e premi 'Genera analisi giornata' per creare la lettura completa.")
    st.stop()
    analysis = {}
else:
    if (
        stored_result.get("season") != selected_season
        or stored_result.get("mode") != analysis_mode
        or stored_result.get("signature") != fixture_signature
    ):
        st.info("Premi di nuovo 'Genera analisi giornata' per aggiornare il contenuto con la selezione corrente.")
        st.stop()
        analysis = {}
    else:
        analysis = stored_result.get("analysis", {})

if not analysis.get("ok"):
    st.warning(analysis.get("message", "Impossibile costruire l'analisi giornata con i dati disponibili."))
    st.stop()

for warning in analysis.get("warnings", []):
    st.info(warning)

_render_round_overview(analysis)

st.subheader("Tabella riepilogo giornata")
summary_table = analysis.get("summary_table", pd.DataFrame())
if isinstance(summary_table, pd.DataFrame) and not summary_table.empty:
    st.dataframe(_display_summary_table(summary_table), use_container_width=True)
else:
    st.caption("Riepilogo giornata non disponibile.")

st.subheader("Dettaglio partita per partita")
for match in analysis.get("matches", []):
    with st.expander(str(match.get("match_title") or "Partita"), expanded=False):
        _render_match_detail(match)

st.subheader("Sintesi giornata")
round_summary = analysis.get("round_summary", {})
summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Piu equilibrata", round_summary.get("balanced_match") or "n/d")
summary_col2.metric("Rischio pareggio", round_summary.get("highest_draw_risk_match") or "n/d")
summary_col3.metric("Rischio upset", round_summary.get("highest_upset_risk_match") or "n/d")
summary_col4, summary_col5 = st.columns(2)
summary_col4.metric("Maggiore confidence", round_summary.get("highest_confidence_match") or "n/d")
summary_col5.metric("Piu volatile", round_summary.get("most_volatile_match") or "n/d")
summary_col6, summary_col7 = st.columns(2)
summary_col6.metric("Confidence piu bassa", round_summary.get("lowest_confidence_match") or "n/d")
summary_col7.metric("Piu stabile", round_summary.get("most_stable_match") or "n/d")
st.write(round_summary.get("summary_text", "Sintesi non disponibile."))

st.subheader("Conclusione giornata")
analyst_notes = analysis.get("analyst_notes", {}) or {}
counts = analyst_notes.get("counts", {})
if isinstance(counts, dict) and counts:
    count_col1, count_col2, count_col3, count_col4 = st.columns(4)
    count_col1.metric("Partite equilibrate", counts.get("balanced_matches", 0))
    count_col2.metric("Alto draw risk", counts.get("high_draw_risk_matches", 0))
    count_col3.metric("Alto upset risk", counts.get("high_upset_risk_matches", 0))
    count_col4.metric("Confidence media", _format_score_value(counts.get("average_confidence"), digits=1, suffix="/100"))
st.write(analyst_notes.get("conclusion", "Conclusione giornata non disponibile."))
limits = analyst_notes.get("limits", [])
if limits:
    st.markdown("#### Limiti principali della lettura")
    _render_bullets(limits)
st.caption(
    "Limiti: la pagina non usa quote, non inventa lineup o infortuni e non modifica Proiezione Classifica, che resta basata sul predictor base."
)
