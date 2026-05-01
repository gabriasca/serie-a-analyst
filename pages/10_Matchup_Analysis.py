from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback
    config = None

from src.db import fetch_matches, list_seasons, list_teams
from src.matchup_analysis import build_matchup_analysis
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _format_number(value: float | int | None, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _load_season_dataframe(season: str) -> pd.DataFrame:
    season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
    if season_df.empty:
        season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if season_df.empty:
        season_df = fetch_matches(season)
    return season_df


def _load_schedule_dataframe(season: str, fallback_df: pd.DataFrame | None = None) -> pd.DataFrame:
    schedule_df = fetch_matches(season)
    if schedule_df.empty and isinstance(fallback_df, pd.DataFrame):
        return fallback_df
    return schedule_df


def _load_season_teams(season: str) -> list[str]:
    teams = list_teams(season, competition_code=DEFAULT_COMPETITION_CODE)
    if not teams:
        teams = list_teams(season)
    return teams


def _safe_dataframe(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.write(df.to_dict(orient="records"))


def _render_bullets(items: list[str]) -> None:
    if not items:
        st.write("- Nessun elemento disponibile.")
        return
    st.markdown("\n".join(f"- {item}" for item in items))


def _format_form_block(form: dict[str, object]) -> str:
    if not form or not form.get("matches"):
        return "n/d"
    return (
        f"{form.get('form_string', '-')} "
        f"({form.get('points', 0)} pt, GF {form.get('goals_for', 0)}, GA {form.get('goals_against', 0)})"
    )


st.set_page_config(page_title=f"{APP_TITLE} | Matchup Analysis", layout="wide")

bootstrap_database()

st.title("Matchup Analysis")
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
teams = _load_season_teams(selected_season)
if len(teams) < 2:
    st.warning("Servono almeno due squadre nella stagione selezionata.")
    st.stop()

home_team = st.selectbox("Squadra casa", teams, index=0)
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox("Squadra trasferta", away_options, index=0)

if st.button("Analizza matchup"):
    season_df = _load_season_dataframe(selected_season)
    schedule_df = _load_schedule_dataframe(selected_season, fallback_df=season_df)
    st.session_state["matchup_analysis_result"] = {
        "season": selected_season,
        "home_team": home_team,
        "away_team": away_team,
        "analysis": build_matchup_analysis(season_df, home_team, away_team, schedule_df=schedule_df),
    }

analysis: dict[str, object] | None = None
stored_result = st.session_state.get("matchup_analysis_result")
if not stored_result:
    st.info("Seleziona due squadre e premi 'Analizza matchup' per generare la lettura del confronto.")
    st.stop()
else:
    if (
        stored_result.get("season") != selected_season
        or stored_result.get("home_team") != home_team
        or stored_result.get("away_team") != away_team
    ):
        st.info("Premi di nuovo 'Analizza matchup' per aggiornare il contenuto con la selezione corrente.")
        st.stop()
    candidate_analysis = stored_result.get("analysis") or {}
    if not candidate_analysis.get("ok"):
        st.warning(candidate_analysis.get("message", "Impossibile costruire il matchup con i dati disponibili."))
        st.stop()
    else:
        analysis = candidate_analysis

if analysis:
    for warning in analysis.get("warnings", []):
        st.info(warning)

    home_profile = analysis["home_profile"]
    away_profile = analysis["away_profile"]
    predictor_context = analysis.get("predictor_context", {})
    style_advantage = analysis.get("style_advantage", {})
    context_engine = analysis.get("context_engine", {})

    st.subheader("Riepilogo partita")
    st.write(f"**{analysis['home_team']} vs {analysis['away_team']}**")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(f"### {analysis['home_team']}")
        row1, row2, row3, row4 = st.columns(4)
        row1.metric("Posizione", home_profile["general"]["position"])
        row2.metric("Punti", home_profile["general"]["points"])
        row3.metric("Rating Elo", _format_number(home_profile["rating"].get("rating_value"), digits=0))
        row4.metric("Forma ultime 5", home_profile["recent"]["form_string"])
    with summary_col2:
        st.markdown(f"### {analysis['away_team']}")
        row1, row2, row3, row4 = st.columns(4)
        row1.metric("Posizione", away_profile["general"]["position"])
        row2.metric("Punti", away_profile["general"]["points"])
        row3.metric("Rating Elo", _format_number(away_profile["rating"].get("rating_value"), digits=0))
        row4.metric("Forma ultime 5", away_profile["recent"]["form_string"])

    st.subheader("Predictor sintetico")
    predictor = analysis.get("predictor", {})
    if predictor.get("ok"):
        pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(5)
        pred_col1.metric("Prob 1", f"{predictor_context.get('home_probability', 0.0) * 100:.1f}%")
        pred_col2.metric("Prob X", f"{predictor_context.get('draw_probability', 0.0) * 100:.1f}%")
        pred_col3.metric("Prob 2", f"{predictor_context.get('away_probability', 0.0) * 100:.1f}%")
        pred_col4.metric("xG predictor casa", f"{predictor_context.get('home_xg', 0.0):.2f}")
        pred_col5.metric("xG predictor trasferta", f"{predictor_context.get('away_xg', 0.0):.2f}")
        st.write(
            f"Risultato piu probabile: **{analysis['home_team']} {predictor_context.get('most_likely_score')} {analysis['away_team']}**"
        )
        top_scores_df = pd.DataFrame(predictor_context.get("top_scores", [])).rename(
            columns={"score": "Risultato", "probability": "Probabilita"}
        )
        if not top_scores_df.empty:
            top_scores_df["Probabilita"] = top_scores_df["Probabilita"].map(lambda value: f"{value * 100:.1f}%")
            _safe_dataframe(top_scores_df)
    else:
        st.warning(
            predictor_context.get("message") or predictor.get("message") or "Predictor non disponibile per questa sfida."
        )

    if predictor_context.get("bullets"):
        st.caption("Perche il predictor spinge in questa direzione:")
        _render_bullets(predictor_context["bullets"])

    schedule_context = analysis.get("schedule_context", {})
    if isinstance(schedule_context, dict):
        st.subheader("Calendario e riposo")
        st.caption(
            schedule_context.get(
                "calendar_context_note",
                "Contesto calendario basato sulle partite disponibili nel database.",
            )
        )
        if schedule_context.get("available"):
            home_load = schedule_context.get("home_schedule_load", {})
            away_load = schedule_context.get("away_schedule_load", {})
            sched_col1, sched_col2 = st.columns(2)
            with sched_col1:
                st.markdown(f"### {analysis['home_team']}")
                row1, row2, row3, row4 = st.columns(4)
                row1.metric("Riposo", _format_number(schedule_context.get("rest_days_home"), digits=0, suffix=" gg"))
                row2.metric("Carico", home_load.get("load_label", "n/d"))
                row3.metric("Partite 14 gg", home_load.get("matches_last_14", 0))
                row4.metric("Competizioni recenti", home_load.get("recent_competitions_count", 0))
                st.write(f"Forma all competitions: {_format_form_block(schedule_context.get('home_recent_all_comp_form', {}))}")
                st.write(f"Forma campionato: {_format_form_block(schedule_context.get('home_recent_league_form', {}))}")
            with sched_col2:
                st.markdown(f"### {analysis['away_team']}")
                row1, row2, row3, row4 = st.columns(4)
                row1.metric("Riposo", _format_number(schedule_context.get("rest_days_away"), digits=0, suffix=" gg"))
                row2.metric("Carico", away_load.get("load_label", "n/d"))
                row3.metric("Partite 14 gg", away_load.get("matches_last_14", 0))
                row4.metric("Competizioni recenti", away_load.get("recent_competitions_count", 0))
                st.write(f"Forma all competitions: {_format_form_block(schedule_context.get('away_recent_all_comp_form', {}))}")
                st.write(f"Forma campionato: {_format_form_block(schedule_context.get('away_recent_league_form', {}))}")
            st.info(schedule_context.get("summary", "Contesto calendario disponibile sulle partite presenti."))
        else:
            st.caption("Contesto calendario non disponibile in modo completo per questo matchup.")

    st.subheader("Confronto metriche avanzate")
    comparison_rows = analysis.get("metric_comparison", [])
    comparison_df = pd.DataFrame(comparison_rows).rename(
        columns={
            "label": "Metrica",
            "home_value": analysis["home_team"],
            "away_value": analysis["away_team"],
            "edge": "Lettura",
            "reading": "Nota",
        }
    )
    if not comparison_df.empty:
        comparison_df[analysis["home_team"]] = pd.to_numeric(comparison_df[analysis["home_team"]], errors="coerce").round(1)
        comparison_df[analysis["away_team"]] = pd.to_numeric(comparison_df[analysis["away_team"]], errors="coerce").round(1)
        _safe_dataframe(comparison_df[["Metrica", analysis["home_team"], analysis["away_team"], "Lettura", "Nota"]])
    else:
        st.caption("Confronto metriche non disponibile in modo completo.")

    st.subheader("Mismatch principali")
    _render_bullets(analysis.get("mismatches", []))

    st.subheader("Vantaggio stilistico")
    adv_col1, adv_col2 = st.columns([1, 2])
    with adv_col1:
        st.metric("Lettura matchup", style_advantage.get("label", "matchup equilibrato"))
    with adv_col2:
        st.write(style_advantage.get("explanation", "Non emerge un vantaggio stilistico netto."))

    if context_engine:
        st.subheader("Peso del dato e contesto")
        ctx_col1, ctx_col2, ctx_col3, ctx_col4 = st.columns(4)
        ctx_col1.metric("Edge corretto", _format_number(context_engine.get("adjusted_edge"), digits=2))
        ctx_col2.metric("Rischio pareggio", _format_number(context_engine.get("draw_risk"), digits=1, suffix="/100"))
        ctx_col3.metric("Rischio upset", _format_number(context_engine.get("upset_risk"), digits=1, suffix="/100"))
        ctx_col4.metric("Confidenza", _format_number(context_engine.get("confidence"), digits=1, suffix="/100"))
        st.caption(context_engine.get("textual_explanation", "Il motore contestuale non ha prodotto una lettura completa."))

        top_weighted = context_engine.get("weighted_factors", [])[:4]
        if top_weighted:
            st.caption("Fattori che pesano di piu nel contesto:")
            factor_lines = []
            for factor in top_weighted:
                factor_lines.append(
                    f"{factor.get('label')}: impatto {factor.get('weighted_impact', 0.0):.2f}, "
                    f"affidabilita {factor.get('reliability', 0.0):.2f}, rilevanza {factor.get('context_relevance', 0.0):.2f}. "
                    f"{factor.get('note', '')}"
                )
            _render_bullets(factor_lines)

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.subheader(f"Rischi per {analysis['home_team']}")
        _render_bullets(analysis.get("home_risks", []))
    with risk_col2:
        st.subheader(f"Rischi per {analysis['away_team']}")
        _render_bullets(analysis.get("away_risks", []))

    st.subheader("Cosa guardare nella partita")
    _render_bullets(analysis.get("tactical_questions", []))

    st.subheader("Sintesi finale")
    st.markdown(analysis.get("summary", "").replace("\n", "  \n"))
