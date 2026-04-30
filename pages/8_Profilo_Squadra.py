from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback
    config = None

from src.analytics import get_teams
from src.db import fetch_matches, list_seasons
from src.seed_data import bootstrap_database
from src.team_profiles import build_team_profile


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


def _format_number(value: float | int | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


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


def _safe_bar_chart(data: pd.DataFrame, fallback_message: str) -> None:
    try:
        st.bar_chart(data)
    except Exception:
        st.caption(fallback_message)


def _render_vs_strength(rows: list[dict[str, object]]) -> None:
    if not rows:
        st.write("Nessun dato disponibile contro le diverse fasce avversarie.")
        return
    for row in rows:
        st.write(
            f"{row['bucket_label']}: {row['points']} punti in {row['matches']} partite "
            f"({row['ppm']:.2f} PPM, GF {row['goals_for']}, GA {row['goals_against']}, DR {row['goal_difference']})."
        )


def _load_season_dataframe(season: str) -> pd.DataFrame:
    season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE, competition_type="league")
    if season_df.empty:
        season_df = fetch_matches(season, competition_code=DEFAULT_COMPETITION_CODE)
    if season_df.empty:
        season_df = fetch_matches(season)
    return season_df


st.set_page_config(page_title=f"{APP_TITLE} | Profilo Squadra", layout="wide")

bootstrap_database()

st.title("Profilo Squadra")
st.write(
    "Descrive il DNA stagionale di una squadra con indicatori interni basati su dati aggregati "
    "di gol, tiri, punti, casa e trasferta, forma recente e rendimento contro fasce di classifica."
)
st.caption(
    "Gli indici mostrati sotto sono indicatori interni dell'app: aiutano a leggere tendenze di gioco "
    "e rendimento, ma non rappresentano metriche proprietarie esterne."
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    seasons = list_seasons()

if not seasons:
    st.warning("Database vuoto o nessuna stagione disponibile. Pubblica prima una stagione valida di Serie A.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)
season_df = _load_season_dataframe(selected_season)

if season_df.empty:
    st.warning("La stagione selezionata non contiene partite utilizzabili per costruire un profilo squadra.")
    st.stop()

teams = get_teams(season_df)
if not teams:
    st.warning("Nessuna squadra disponibile per la stagione selezionata.")
    st.stop()

selected_team = st.selectbox("Seleziona squadra", teams)
profile = build_team_profile(season_df, selected_team)

if not profile.get("ok"):
    st.warning(profile.get("message", "Dati insufficienti per costruire il profilo squadra."))
    st.stop()

for note in profile.get("notes", []):
    st.info(note)

general = profile["general"]
offensive = profile["offensive"]
defensive = profile["defensive"]
home_away = profile["home_away"]
recent = profile["recent"]
indicators = profile["indicators"]
rating = profile["rating"]
advanced = profile.get("advanced_metrics", {})
schedule_context = profile.get("schedule_context", {})
vs_strength_rows = profile["vs_strength_buckets"]

st.subheader("Riepilogo generale")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Posizione", general["position"])
col2.metric("Punti", general["points"])
col3.metric("Partite", general["matches"])
col4.metric("Gol fatti", general["goals_for"])
col5.metric("Gol subiti", general["goals_against"])
col6.metric("Differenza reti", general["goal_difference"])

rating_col1, rating_col2, rating_col3 = st.columns(3)
rating_col1.metric("Rating Elo", _format_number(rating.get("rating_value"), digits=0))
rating_col2.metric("Fascia forza", rating.get("strength_band") or "n/d")
rating_col3.metric("Rank Elo", rating.get("rating_rank") or "n/d")
if rating.get("available"):
    st.caption(
        f"Rating informativo da {rating.get('source_name') or 'fonte esterna'} "
        f"({rating.get('rating_date') or 'data n/d'})."
    )
else:
    st.caption("Rating Elo non disponibile per questa squadra: il profilo resta basato solo sui dati partita.")

st.subheader("Identita offensiva")
off_col1, off_col2, off_col3, off_col4, off_col5 = st.columns(5)
off_col1.metric("Gol medi", _format_number(offensive["goals_avg"]))
off_col2.metric("Tiri medi", _format_number(offensive["shots_avg"]))
off_col3.metric("Tiri in porta medi", _format_number(offensive["shots_on_target_avg"]))
off_col4.metric(
    "Efficienza realizzativa",
    _format_number(offensive["efficienza_realizzativa"], suffix="%"),
)
off_col5.metric(
    "Indice pericolosita offensiva",
    _format_number(offensive["indice_pericolosita_offensiva"], digits=1),
)

st.subheader("Identita difensiva")
def_col1, def_col2, def_col3, def_col4 = st.columns(4)
def_col1.metric("Gol subiti medi", _format_number(defensive["goals_against_avg"]))
def_col2.metric("Tiri concessi medi", _format_number(defensive["shots_conceded_avg"]))
def_col3.metric("Tiri in porta concessi medi", _format_number(defensive["shots_on_target_conceded_avg"]))
def_col4.metric(
    "Indice solidita difensiva",
    _format_number(defensive["indice_solidita_difensiva"], digits=1),
)

st.subheader("Casa / Fuori")
home_col1, home_col2, home_col3, home_col4 = st.columns(4)
home_col1.metric("Punti casa", home_away["points_home"])
home_col2.metric("Punti trasferta", home_away["points_away"])
home_col3.metric("PPM casa", _format_number(home_away["ppm_home"]))
home_col4.metric("PPM trasferta", _format_number(home_away["ppm_away"]))
st.write(home_away["note"])

st.subheader("Forma recente")
recent_col1, recent_col2, recent_col3, recent_col4 = st.columns(4)
recent_col1.metric("Ultime 5", recent["form_string"])
recent_col2.metric("Punti ultime 5", recent["points"])
recent_col3.metric("Gol fatti ultime 5", recent["goals_for"])
recent_col4.metric("Gol subiti ultime 5", recent["goals_against"])

st.subheader("Forma multi-competizione e calendario")
if isinstance(schedule_context, dict) and schedule_context:
    schedule_load = schedule_context.get("schedule_load", {})
    form_comparison = schedule_context.get("form_comparison", {})
    league_form = form_comparison.get("league_form", {})
    all_comp_form = form_comparison.get("all_competition_form", {})
    sched_col1, sched_col2, sched_col3, sched_col4 = st.columns(4)
    sched_col1.metric("Riposo ultimo match", _format_number(schedule_load.get("rest_days"), digits=0, suffix=" gg"))
    sched_col2.metric("Carico calendario", schedule_load.get("load_label", "n/d"))
    sched_col3.metric("Partite 14 gg", schedule_load.get("matches_last_14", 0))
    sched_col4.metric("Competizioni recenti", schedule_load.get("recent_competitions_count", 0))
    st.write(f"Forma campionato: {_format_form_block(league_form)}")
    st.write(f"Forma tutte le competizioni disponibili: {_format_form_block(all_comp_form)}")
    st.caption(schedule_context.get("note") or "Contesto calendario basato sulle partite disponibili.")
else:
    st.caption("Dati calendario non disponibili per questa squadra.")

if advanced:
    st.subheader("Metriche avanzate v1")
    st.caption("Indicatori interni 0-100 basati sui dati aggregati disponibili. Non sono xG reali.")
    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    adv_col1.metric(
        "Pericolosita offensiva",
        _format_number(advanced.get("offensive_threat_index"), digits=1),
    )
    adv_col2.metric(
        "Solidita difensiva",
        _format_number(advanced.get("defensive_solidity_index"), digits=1),
    )
    adv_col3.metric(
        "Momento recente",
        _format_number(advanced.get("recent_momentum_index"), digits=1),
    )
    adv_col4.metric(
        "Forza calendario",
        _format_number(advanced.get("schedule_strength_index"), digits=1),
    )
    st.caption(advanced.get("schedule_strength_note") or "Forza calendario non disponibile.")

st.subheader("Rendimento per tipo avversario")
if profile.get("strength_bucket_source") == "elo":
    st.caption("Le fasce forti/medie/deboli sono costruite con il ranking Elo attuale disponibile nel seed.")
else:
    st.caption("Le fasce forti/medie/deboli sono costruite con la classifica corrente della stagione.")
_render_vs_strength(vs_strength_rows)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.caption("Indici interni: 100 rappresenta circa la media del campionato.")
    indices_chart = pd.DataFrame(
        {
            "Indice": [
                "Pericolosita offensiva",
                "Solidita difensiva",
                "Intensita",
                "Dipendenza casa",
            ],
            "Valore": [
                indicators["indice_pericolosita_offensiva"],
                indicators["indice_solidita_difensiva"],
                indicators["indice_intensita"],
                indicators["indice_dipendenza_casa"],
            ],
        }
    ).set_index("Indice")
    _safe_bar_chart(indices_chart, "Grafico indici non disponibile in questo ambiente Streamlit.")

with chart_col2:
    home_away_chart = pd.DataFrame(
        {
            "Contesto": ["Casa", "Trasferta"],
            "PPM": [home_away["ppm_home"], home_away["ppm_away"]],
            "Punti": [home_away["points_home"], home_away["points_away"]],
        }
    ).set_index("Contesto")
    st.caption("Confronto tra rendimento interno ed esterno.")
    _safe_bar_chart(home_away_chart[["PPM"]], "Grafico casa/trasferta non disponibile in questo ambiente Streamlit.")

st.caption("PPM per fascia avversaria.")
vs_chart = pd.DataFrame(vs_strength_rows).rename(columns={"bucket_label": "Fascia avversaria"}).set_index("Fascia avversaria")[["ppm"]]
vs_chart = vs_chart.rename(columns={"ppm": "PPM"})
_safe_bar_chart(vs_chart, "Grafico per fascia avversaria non disponibile in questo ambiente Streamlit.")

st.subheader("Archetipi squadra")
st.caption("Etichette assegnate con regole semplici e leggibili sui dati stagionali.")
_render_bullets(profile["archetypes"])

strength_col, weakness_col = st.columns(2)
with strength_col:
    st.subheader("Punti forti")
    _render_bullets(profile["strengths"])

with weakness_col:
    st.subheader("Punti deboli")
    _render_bullets(profile["weaknesses"])

st.subheader("Sintesi finale")
st.markdown(profile["summary"].replace("\n", "  \n"))
