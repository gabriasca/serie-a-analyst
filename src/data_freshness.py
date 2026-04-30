from __future__ import annotations

from typing import Any

import pandas as pd


LATEST_MATCH_COLUMNS = [
    "match_date",
    "competition_name",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "source_name",
]


def _prepare_freshness_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=LATEST_MATCH_COLUMNS)

    prepared_df = df.copy()
    defaults = {
        "match_date": None,
        "competition_name": "Serie A",
        "home_team": "",
        "away_team": "",
        "home_goals": 0,
        "away_goals": 0,
        "source_name": "seed_csv",
        "season": "",
    }
    for column, default in defaults.items():
        if column not in prepared_df.columns:
            prepared_df[column] = default

    prepared_df["match_date"] = pd.to_datetime(prepared_df["match_date"], errors="coerce")
    prepared_df["competition_name"] = (
        prepared_df["competition_name"].replace("", pd.NA).fillna("Serie A").astype(str)
    )
    prepared_df["source_name"] = prepared_df["source_name"].replace("", pd.NA).fillna("seed_csv").astype(str)
    prepared_df["home_goals"] = pd.to_numeric(prepared_df["home_goals"], errors="coerce").fillna(0).astype(int)
    prepared_df["away_goals"] = pd.to_numeric(prepared_df["away_goals"], errors="coerce").fillna(0).astype(int)
    return prepared_df


def _iso_date(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _get_teams(df: pd.DataFrame) -> list[str]:
    if df.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return []
    teams = set(df["home_team"].dropna().astype(str).tolist()) | set(df["away_team"].dropna().astype(str).tolist())
    return sorted(team for team in teams if team.strip())


def _get_seasons(df: pd.DataFrame) -> list[str]:
    if df.empty or "season" not in df.columns:
        return []
    return sorted({str(season) for season in df["season"].dropna().tolist() if str(season).strip()}, reverse=True)


def get_latest_match_date(df: pd.DataFrame) -> str | None:
    prepared_df = _prepare_freshness_dataframe(df)
    if prepared_df.empty or prepared_df["match_date"].dropna().empty:
        return None
    return _iso_date(prepared_df["match_date"].max())


def get_recent_loaded_matches(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    prepared_df = _prepare_freshness_dataframe(df)
    if prepared_df.empty:
        return pd.DataFrame(columns=LATEST_MATCH_COLUMNS)

    sort_columns = ["match_date"]
    if "id" in prepared_df.columns:
        sort_columns.append("id")
    latest_df = prepared_df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(n).copy()
    latest_df = latest_df[LATEST_MATCH_COLUMNS]
    latest_df["match_date"] = latest_df["match_date"].apply(_iso_date)
    return latest_df.reset_index(drop=True)


def expected_total_matches(team_count: int) -> int:
    if team_count < 2:
        return 0
    return int(team_count * (team_count - 1))


def estimate_missing_matches(df: pd.DataFrame) -> int:
    prepared_df = _prepare_freshness_dataframe(df)
    team_count = len(_get_teams(prepared_df))
    expected_matches = expected_total_matches(team_count)
    return max(expected_matches - int(len(prepared_df)), 0)


def build_freshness_status(df: pd.DataFrame) -> dict[str, str]:
    prepared_df = _prepare_freshness_dataframe(df)
    if prepared_df.empty:
        return {
            "freshness_status": "database_vuoto",
            "freshness_message": "Database vuoto: nessuna partita caricata.",
        }

    latest_match_date = get_latest_match_date(prepared_df)
    if latest_match_date is None:
        return {
            "freshness_status": "attenzione",
            "freshness_message": "Partite presenti, ma nessuna data valida per capire l'aggiornamento.",
        }

    missing_matches = estimate_missing_matches(prepared_df)
    latest_date = pd.to_datetime(latest_match_date, errors="coerce")
    today = pd.Timestamp.today().normalize()
    days_since_latest = None if pd.isna(latest_date) else int((today - latest_date.normalize()).days)

    if days_since_latest is not None and days_since_latest > 21 and missing_matches > 0:
        return {
            "freshness_status": "attenzione",
            "freshness_message": (
                f"Ultima partita al {latest_match_date}: dataset potenzialmente vecchio "
                f"o stagione ancora non completa ({missing_matches} partite mancanti stimate)."
            ),
        }

    if missing_matches > 0:
        return {
            "freshness_status": "dati_parziali",
            "freshness_message": (
                f"Ultima partita al {latest_match_date}. Dataset parziale: "
                f"mancano circa {missing_matches} partite rispetto al calendario teorico."
            ),
        }

    return {
        "freshness_status": "ok",
        "freshness_message": f"Dataset completo rispetto al calendario teorico. Ultima partita: {latest_match_date}.",
    }


def build_data_freshness_report(df: pd.DataFrame) -> dict[str, Any]:
    prepared_df = _prepare_freshness_dataframe(df)
    teams = _get_teams(prepared_df)
    seasons = _get_seasons(prepared_df)
    match_count = int(len(prepared_df))
    team_count = int(len(teams))
    latest_match_date = get_latest_match_date(prepared_df)
    latest_matches = get_recent_loaded_matches(prepared_df, n=10)
    expected_matches = expected_total_matches(team_count)
    missing_matches = estimate_missing_matches(prepared_df)
    source_names = sorted(
        {str(source) for source in prepared_df.get("source_name", pd.Series(dtype=str)).dropna().tolist() if str(source).strip()}
    )

    if latest_match_date:
        latest_date = pd.to_datetime(latest_match_date, errors="coerce")
        matches_on_latest_date = int((prepared_df["match_date"].dt.normalize() == latest_date.normalize()).sum())
    else:
        matches_on_latest_date = 0

    status = build_freshness_status(prepared_df)
    report = {
        "match_count": match_count,
        "team_count": team_count,
        "season_count": len(seasons),
        "seasons": seasons,
        "latest_match_date": latest_match_date,
        "latest_matches": latest_matches,
        "matches_on_latest_date": matches_on_latest_date,
        "expected_total_matches": expected_matches,
        "missing_matches_estimate": missing_matches,
        "source_names": source_names or ["seed_csv"],
        "freshness_status": status["freshness_status"],
        "freshness_message": status["freshness_message"],
    }
    report["freshness_summary"] = build_freshness_summary(report)
    return report


def build_freshness_summary(report: dict[str, Any]) -> str:
    status = str(report.get("freshness_status") or "attenzione")
    latest_date = report.get("latest_match_date") or "n/d"
    match_count = int(report.get("match_count", 0) or 0)
    expected_matches = int(report.get("expected_total_matches", 0) or 0)
    missing_matches = int(report.get("missing_matches_estimate", 0) or 0)

    if status == "database_vuoto":
        return "Database vuoto: non e possibile valutare l'aggiornamento dati."
    if status == "ok":
        return f"Dati allineati al calendario teorico: {match_count}/{expected_matches} partite, ultima data {latest_date}."
    if status == "dati_parziali":
        return f"Dati parziali: {match_count}/{expected_matches} partite, circa {missing_matches} mancanti, ultima data {latest_date}."
    return f"Controllo consigliato: ultima data {latest_date}, {match_count} partite caricate, {missing_matches} mancanti stimate."
