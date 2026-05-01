from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from src import config
except Exception:  # pragma: no cover - defensive fallback
    config = None

from src.db import fetch_matches, list_seasons
from src.model_review import build_model_review
from src.seed_data import bootstrap_database


APP_TITLE = getattr(config, "APP_TITLE", "Serie A Analyst")
DEFAULT_COMPETITION_CODE = getattr(config, "DEFAULT_COMPETITION_CODE", "ITA_SERIE_A")
PUBLIC_DEMO_MODE = getattr(config, "PUBLIC_DEMO_MODE", True)
PUBLIC_DEMO_BANNER = getattr(
    config,
    "PUBLIC_DEMO_BANNER",
    "Versione pubblica demo: dati snapshot, previsioni statistiche non certe.",
)


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


@st.cache_data(show_spinner=False)
def _build_review_for_season(season: str) -> dict[str, object]:
    season_df = _load_season_dataframe(season)
    schedule_df = _load_schedule_dataframe(season, fallback_df=season_df)
    return build_model_review(season_df, minimum_team_history=1, schedule_df=schedule_df)


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


def _format_rate(hits: int, total: int) -> tuple[str, str]:
    if total <= 0:
        return "n/d", "0/0"
    rate = hits / total * 100.0
    return f"{rate:.1f}%", f"{hits}/{total}"


def _render_metric_block(label: str, hits: int, total: int) -> None:
    rate_text, count_text = _format_rate(hits, total)
    st.metric(label, rate_text)
    st.caption(count_text)


st.set_page_config(page_title=f"{APP_TITLE} | Model Review", layout="wide")

bootstrap_database()

st.title("Model Review")
st.caption("Backtest storico: ogni partita viene analizzata usando solo i dati disponibili prima del match.")
st.caption(
    "Serve per capire se `base_edge`, `adjusted_edge`, `draw_risk`, `upset_risk` e `confidence` stanno davvero aggiungendo valore."
)
st.caption(
    "Il Predictor contestuale v2 e sperimentale; la calibrazione principale resta monitorata qui in Model Review."
)

if PUBLIC_DEMO_MODE:
    st.caption(PUBLIC_DEMO_BANNER)

seasons = list_seasons(competition_code=DEFAULT_COMPETITION_CODE)
if not seasons:
    seasons = list_seasons()

if not seasons:
    st.warning("Database vuoto o nessuna stagione disponibile.")
    st.stop()

selected_season = st.selectbox("Seleziona stagione", seasons)

if st.button("Esegui model review"):
    with st.spinner("Sto costruendo il backtest storico partita per partita..."):
        st.session_state["model_review_result"] = {
            "season": selected_season,
            "review": _build_review_for_season(selected_season),
        }

review: dict[str, object] | None = None
stored_result = st.session_state.get("model_review_result")
if not stored_result:
    st.info("Seleziona una stagione e premi 'Esegui model review' per generare il backtest.")
else:
    if stored_result.get("season") != selected_season:
        st.info("Premi di nuovo 'Esegui model review' per aggiornare il backtest con la stagione corrente.")
    else:
        candidate_review = stored_result.get("review") or {}
        if not candidate_review.get("ok"):
            st.warning(candidate_review.get("message", "Impossibile costruire la review del modello con i dati disponibili."))
        else:
            review = candidate_review

if review:
    backtest_df = review["backtest_df"]
    ratings_audit = review.get("ratings_audit", {})
    schedule_audit = review.get("schedule_audit", {})
    general_review = review["general_review"]
    diagnostic_tables = review["diagnostic_tables"]
    bucket_review = review["bucket_review"]
    factor_review = review["factor_review"]
    calibration_guidance = review.get("calibration_guidance", {})
    conclusions = review["conclusions"]

    if int(general_review.get("matches_analyzed", 0) or 0) < 20:
        st.warning("Il numero di partite analizzate e ancora limitato: la review e utile, ma va letta con prudenza.")

    predictor_coverage = 0.0
    if not backtest_df.empty and "predictor_available" in backtest_df.columns:
        predictor_coverage = float(backtest_df["predictor_available"].fillna(False).mean() * 100.0)

    st.subheader("Metriche generali")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Partite analizzate", int(general_review.get("matches_analyzed", 0) or 0))
    metric_col2.metric("Copertura predictor", f"{predictor_coverage:.1f}%")
    metric_col3.metric(
        "Confronto edge medio",
        f"{float(general_review.get('mean_base_edge_abs', 0.0) or 0.0):.2f} -> "
        f"{float(general_review.get('mean_adjusted_edge_abs', 0.0) or 0.0):.2f}",
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        _render_metric_block(
            "Favorito base non perde",
            int(general_review.get("base_favorite_non_loss_hits", 0) or 0),
            int(general_review.get("base_favorite_non_loss_total", 0) or 0),
        )
    with summary_col2:
        _render_metric_block(
            "Favorito contestuale non perde",
            int(general_review.get("adjusted_favorite_non_loss_hits", 0) or 0),
            int(general_review.get("adjusted_favorite_non_loss_total", 0) or 0),
        )
    with summary_col3:
        _render_metric_block(
            "Alto draw_risk finisce in pari",
            int(general_review.get("high_draw_hits", 0) or 0),
            int(general_review.get("high_draw_total", 0) or 0),
        )
    with summary_col4:
        _render_metric_block(
            "Alto upset_risk: favorito non vince",
            int(general_review.get("high_upset_hits", 0) or 0),
            int(general_review.get("high_upset_total", 0) or 0),
        )

    st.caption(
        f"Contesto che migliora il giudizio base: {int(general_review.get('context_helped', 0) or 0)} partite. "
        f"Contesto che lo peggiora: {int(general_review.get('context_hurt', 0) or 0)}."
    )

    st.subheader("Tabelle diagnostiche")
    diagnostic_labels = {
        "context_helped": "Partite dove il contesto ha corretto bene il giudizio base",
        "context_hurt": "Partite dove il contesto ha peggiorato il giudizio",
        "high_uncertainty": "Partite ad alta incertezza",
        "high_draw_risk": "Partite ad alto rischio pareggio",
        "high_upset_risk": "Partite ad alto rischio upset",
    }
    for key, label in diagnostic_labels.items():
        st.markdown(f"### {label}")
        table = diagnostic_tables.get(key, pd.DataFrame())
        if isinstance(table, pd.DataFrame) and not table.empty:
            _safe_dataframe(table)
        else:
            st.caption("Nessun caso rilevante disponibile per questo blocco.")

    st.subheader("Analisi per bucket")
    bucket_labels = {
        "confidence": "Confidence bassa / media / alta",
        "draw_risk": "Draw risk basso / medio / alto",
        "upset_risk": "Upset risk basso / medio / alto",
        "adjusted_edge": "Adjusted edge basso / medio / alto",
    }
    bucket_col1, bucket_col2 = st.columns(2)
    bucket_items = list(bucket_labels.items())
    for index, (key, label) in enumerate(bucket_items):
        target_column = bucket_col1 if index % 2 == 0 else bucket_col2
        with target_column:
            st.markdown(f"### {label}")
            bucket_df = bucket_review.get(key, pd.DataFrame())
            if isinstance(bucket_df, pd.DataFrame) and not bucket_df.empty:
                _safe_dataframe(bucket_df)
            else:
                st.caption("Bucket non disponibile.")

    st.subheader("Valutazione fattori")
    if isinstance(factor_review, pd.DataFrame) and not factor_review.empty:
        factor_review_display = factor_review.drop(
            columns=[column for column in ["factor_key", "use_for_calibration", "sort_priority"] if column in factor_review.columns],
            errors="ignore",
        )
        _safe_dataframe(factor_review_display)
    else:
        st.caption("Nessuna valutazione fattori disponibile con i dati correnti.")

    st.subheader("Indicazioni di calibrazione")
    if ratings_audit:
        st.caption(
            f"Audit Elo: {ratings_audit.get('status', 'n/d')}. "
            f"{ratings_audit.get('note', '')}"
        )
    if schedule_audit:
        st.caption(f"Audit calendario: {schedule_audit.get('note', 'Contesto calendario non disponibile.')}")

    guidance_col1, guidance_col2 = st.columns(2)
    with guidance_col1:
        st.markdown("### Fattori da valorizzare")
        _render_bullets(calibration_guidance.get("factors_to_value", []))
        st.markdown("### Metriche promettenti")
        _render_bullets(calibration_guidance.get("promising_metrics", []))

    with guidance_col2:
        st.markdown("### Fattori da ridurre")
        _render_bullets(calibration_guidance.get("factors_to_reduce", []))
        st.markdown("### Metriche da rivedere")
        _render_bullets(calibration_guidance.get("metrics_to_review", []))

    sample_warnings = calibration_guidance.get("sample_warnings", [])
    if sample_warnings:
        st.markdown("### Avvisi campioni")
        _render_bullets(sample_warnings)

    st.subheader("Conclusioni automatiche")
    _render_bullets(conclusions)

    with st.expander("Dettaglio partite backtestate"):
        raw_columns = [
            "match_date",
            "home_team",
            "away_team",
            "score",
            "actual_outcome",
            "base_edge",
            "adjusted_edge",
            "draw_risk",
            "upset_risk",
            "confidence",
            "top_context_factors",
        ]
        raw_df = backtest_df[raw_columns].copy()
        if not raw_df.empty:
            raw_df["match_date"] = pd.to_datetime(raw_df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            _safe_dataframe(raw_df)
        else:
            st.caption("Nessuna partita disponibile nel backtest.")
