from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, DEFAULT_COMPETITION_CODE, PUBLIC_DEMO_BANNER, PUBLIC_DEMO_MODE
from src.db import fetch_matches, list_seasons, list_teams
from src.reporting import build_match_report_data
from src.seed_data import bootstrap_database


st.set_page_config(page_title=f"{APP_TITLE} | Report Partita", layout="wide")


def _format_schedule_value(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _format_form_block(form: dict[str, object]) -> str:
    if not form or not form.get("matches"):
        return "n/d"
    return (
        f"{form.get('form_string', '-')} "
        f"({form.get('points', 0)} pt, GF {form.get('goals_for', 0)}, GA {form.get('goals_against', 0)})"
    )


def _load_report_dataframes(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    schedule_df = fetch_matches(season)
    if schedule_df.empty:
        schedule_df = league_df
    return league_df, schedule_df

bootstrap_database()

st.title("Report Partita")
st.write("Genera un report pre-partita leggibile usando solo i dati disponibili nell'app.")
st.caption("Per una lettura dei mismatch, usa la pagina Matchup Analysis.")

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    st.warning("Nessuna stagione Serie A disponibile nel database.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
teams = list_teams(selected_season, competition_code=DEFAULT_COMPETITION_CODE)

if len(teams) < 2:
    st.warning("Servono almeno due squadre nella stagione selezionata per generare un report.")
    st.stop()

home_team = st.selectbox("Squadra casa", teams, index=0)
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox("Squadra trasferta", away_options, index=0)

if st.button("Genera report"):
    league_df, schedule_df = _load_report_dataframes(selected_season)
    st.session_state["match_report_result"] = {
        "season": selected_season,
        "home_team": home_team,
        "away_team": away_team,
        "report": build_match_report_data(
            league_df,
            selected_season,
            home_team,
            away_team,
            schedule_df=schedule_df,
        ),
    }

stored_report = st.session_state.get("match_report_result")
if not stored_report:
    st.info("Seleziona due squadre e premi 'Genera report' per costruire l'analisi.")
    st.stop()

if (
    stored_report["season"] != selected_season
    or stored_report["home_team"] != home_team
    or stored_report["away_team"] != away_team
):
    st.info("Premi di nuovo 'Genera report' per aggiornare il contenuto con la selezione corrente.")
    st.stop()

report = stored_report["report"]
if not report.get("ok"):
    st.warning(report.get("message", "Impossibile generare il report con i dati disponibili."))
    st.stop()

home_general = report["general_performance"]["home"]
away_general = report["general_performance"]["away"]
home_recent = report["recent_form"]["home"]
away_recent = report["recent_form"]["away"]
prediction = report["prediction"]
table_context = report["table_context"]
ratings = report.get("ratings", {})
home_rating = ratings.get("home")
away_rating = ratings.get("away")
advanced = report.get("advanced_metrics", {})
home_advanced = advanced.get("home") or {}
away_advanced = advanced.get("away") or {}
schedule_context = report.get("schedule_context", {})

st.header(report["match_title"])

st.subheader("Stato dati")
state_col1, state_col2, state_col3 = st.columns(3)
state_col1.metric("Stagione usata", report["season"])
state_col2.metric("Partite disponibili", report["match_count"])
state_col3.metric("Squadre in stagione", report["team_count"])
st.info("Analisi statistica basata sui dati disponibili, non una certezza.")

st.subheader("Rating Elo")
rating_col1, rating_col2 = st.columns(2)
with rating_col1:
    st.metric(
        f"Elo {report['home_team']}",
        f"{home_rating['rating_value']:.0f}" if home_rating else "n/d",
    )
with rating_col2:
    st.metric(
        f"Elo {report['away_team']}",
        f"{away_rating['rating_value']:.0f}" if away_rating else "n/d",
    )
st.caption(ratings.get("note", "Il rating e usato come indicatore di forza storica/recente, non come certezza."))

st.subheader("Calendario e riposo")
if isinstance(schedule_context, dict) and schedule_context.get("available"):
    st.caption(
        schedule_context.get(
            "calendar_context_note",
            "Il contesto calendario e basato sulle partite disponibili nel database.",
        )
    )
    home_load = schedule_context.get("home_schedule_load", {})
    away_load = schedule_context.get("away_schedule_load", {})
    sched_left, sched_right = st.columns(2)
    with sched_left:
        st.markdown(f"### {report['home_team']}")
        rest_col, load_col, games_col = st.columns(3)
        rest_col.metric("Riposo", _format_schedule_value(schedule_context.get("rest_days_home"), " gg"))
        load_col.metric("Carico", home_load.get("load_label", "n/d"))
        games_col.metric("Partite 14 gg", home_load.get("matches_last_14", 0))
        st.caption(f"Forma tutte le competizioni: {_format_form_block(schedule_context.get('home_recent_all_comp_form', {}))}")
    with sched_right:
        st.markdown(f"### {report['away_team']}")
        rest_col, load_col, games_col = st.columns(3)
        rest_col.metric("Riposo", _format_schedule_value(schedule_context.get("rest_days_away"), " gg"))
        load_col.metric("Carico", away_load.get("load_label", "n/d"))
        games_col.metric("Partite 14 gg", away_load.get("matches_last_14", 0))
        st.caption(f"Forma tutte le competizioni: {_format_form_block(schedule_context.get('away_recent_all_comp_form', {}))}")
    st.info(schedule_context.get("summary", "Contesto calendario letto sulle partite disponibili."))
else:
    st.caption("Contesto calendario non disponibile in modo completo per questa partita.")

st.subheader("Metriche avanzate interne")
st.caption(advanced.get("note", "Questi indicatori interni aiutano a leggere il match, ma non sono xG reali."))
if advanced.get("available"):
    adv_left, adv_right = st.columns(2)
    with adv_left:
        st.markdown(f"### {report['home_team']}")
        adv_metric1, adv_metric2, adv_metric3 = st.columns(3)
        adv_metric1.metric(
            "Pericolosita offensiva",
            f"{home_advanced.get('offensive_threat_index', 0):.1f}" if home_advanced.get("offensive_threat_index") is not None else "n/d",
        )
        adv_metric2.metric(
            "Solidita difensiva",
            f"{home_advanced.get('defensive_solidity_index', 0):.1f}" if home_advanced.get("defensive_solidity_index") is not None else "n/d",
        )
        adv_metric3.metric(
            "Momento recente",
            f"{home_advanced.get('recent_momentum_index', 0):.1f}" if home_advanced.get("recent_momentum_index") is not None else "n/d",
        )
    with adv_right:
        st.markdown(f"### {report['away_team']}")
        adv_metric1, adv_metric2, adv_metric3 = st.columns(3)
        adv_metric1.metric(
            "Pericolosita offensiva",
            f"{away_advanced.get('offensive_threat_index', 0):.1f}" if away_advanced.get("offensive_threat_index") is not None else "n/d",
        )
        adv_metric2.metric(
            "Solidita difensiva",
            f"{away_advanced.get('defensive_solidity_index', 0):.1f}" if away_advanced.get("defensive_solidity_index") is not None else "n/d",
        )
        adv_metric3.metric(
            "Momento recente",
            f"{away_advanced.get('recent_momentum_index', 0):.1f}" if away_advanced.get("recent_momentum_index") is not None else "n/d",
        )
else:
    st.caption("Metriche avanzate non disponibili in modo completo per questa partita.")

st.subheader("Forma recente")
left_form, right_form = st.columns(2)

with left_form:
    st.markdown(f"### {report['home_team']}")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Forma ultime 5", home_recent["form_string"])
    metric2.metric("Punti ultime 5", home_recent["points"])
    metric3.metric("Gol fatti", home_recent["goals_for"])
    metric4.metric("Gol subiti", home_recent["goals_against"])
    if not home_recent["table"].empty:
        st.dataframe(home_recent["table"], use_container_width=True)
    else:
        st.caption("Nessuna partita recente disponibile.")

with right_form:
    st.markdown(f"### {report['away_team']}")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Forma ultime 5", away_recent["form_string"])
    metric2.metric("Punti ultime 5", away_recent["points"])
    metric3.metric("Gol fatti", away_recent["goals_for"])
    metric4.metric("Gol subiti", away_recent["goals_against"])
    if not away_recent["table"].empty:
        st.dataframe(away_recent["table"], use_container_width=True)
    else:
        st.caption("Nessuna partita recente disponibile.")

st.subheader("Rendimento generale")
left_general, right_general = st.columns(2)

with left_general:
    st.markdown(f"### {report['home_team']}")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Punti attuali", home_general["points"])
    metric2.metric("Posizione", home_general["position"] if home_general["position"] is not None else "-")
    metric3.metric("Differenza reti", home_general["goal_difference"])
    metric4, metric5, metric6 = st.columns(3)
    metric4.metric("Gol fatti medi", f"{home_general['avg_goals_for']:.2f}")
    metric5.metric("Gol subiti medi", f"{home_general['avg_goals_against']:.2f}")
    metric6.metric("Rendimento casa", f"{home_general['relevant_split']['PPM']:.2f} PPM")
    st.caption(
        f"Casa: {home_general['relevant_split']['Pts']} punti in {home_general['relevant_split']['GP']} gare, "
        f"{home_general['relevant_split']['GF']} gol fatti e {home_general['relevant_split']['GA']} subiti."
    )

with right_general:
    st.markdown(f"### {report['away_team']}")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Punti attuali", away_general["points"])
    metric2.metric("Posizione", away_general["position"] if away_general["position"] is not None else "-")
    metric3.metric("Differenza reti", away_general["goal_difference"])
    metric4, metric5, metric6 = st.columns(3)
    metric4.metric("Gol fatti medi", f"{away_general['avg_goals_for']:.2f}")
    metric5.metric("Gol subiti medi", f"{away_general['avg_goals_against']:.2f}")
    metric6.metric("Rendimento trasferta", f"{away_general['relevant_split']['PPM']:.2f} PPM")
    st.caption(
        f"Trasferta: {away_general['relevant_split']['Pts']} punti in {away_general['relevant_split']['GP']} gare, "
        f"{away_general['relevant_split']['GF']} gol fatti e {away_general['relevant_split']['GA']} subiti."
    )

st.subheader("Predictor")
if prediction.get("ok"):
    probs = prediction["probabilities"]
    pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(5)
    pred_col1.metric("xG casa", f"{prediction['expected_goals_home']:.2f}")
    pred_col2.metric("xG trasferta", f"{prediction['expected_goals_away']:.2f}")
    pred_col3.metric("1", f"{probs['1'] * 100:.1f}%")
    pred_col4.metric("X", f"{probs['X'] * 100:.1f}%")
    pred_col5.metric("2", f"{probs['2'] * 100:.1f}%")

    st.write(
        f"Risultato piu probabile: **{report['home_team']} {prediction['most_likely_score']} {report['away_team']}**"
    )

    top_scores_df = pd.DataFrame(prediction["top_scorelines"]).rename(
        columns={"score": "Risultato", "probability": "Probabilita"}
    )
    top_scores_df["Probabilita"] = top_scores_df["Probabilita"].map(lambda value: f"{value * 100:.1f}%")
    st.write("Top 5 risultati possibili")
    st.dataframe(top_scores_df, use_container_width=True)
else:
    st.warning(prediction.get("message", "Predictor non disponibile per questa partita."))

st.subheader("Fattori chiave")
for item in report["key_factors"]:
    st.markdown(f"- {item}")

st.subheader("Sintesi finale")
st.write(report["summary"])

st.subheader("Impatto classifica")
impact_col1, impact_col2 = st.columns(2)
impact_col1.metric(
    f"Posizione {report['home_team']}",
    table_context["home_position"] if table_context["home_position"] is not None else "-",
    delta=f"{table_context['home_points']} punti",
)
impact_col2.metric(
    f"Posizione {report['away_team']}",
    table_context["away_position"] if table_context["away_position"] is not None else "-",
    delta=f"{table_context['away_points']} punti",
)
st.write(table_context["note"])
