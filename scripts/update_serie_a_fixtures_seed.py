from __future__ import annotations

import io
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.config import (
    DEFAULT_COMPETITION_CODE,
    DEFAULT_COMPETITION_NAME,
    FIXTURE_SEED_PATH,
    FOOTBALL_DATA_SERIE_A_FIXTURES_URL,
    SEED_CSV_PATH,
)
from src.data_import import load_csv_to_dataframe, normalize_columns, normalize_team_name, parse_match_dates


DEFAULT_SOURCE_NAME = "football-data.co.uk fixtures"
STRICT_UPDATE_ENV = "FOOTBALL_DATA_FIXTURES_STRICT_UPDATE"
FIXTURE_COLUMNS = [
    "season",
    "match_date",
    "matchday",
    "home_team",
    "away_team",
    "competition_code",
    "competition_name",
    "source_name",
    "source_url",
]


def infer_season_from_url(url: str) -> str | None:
    match = re.search(r"/(\d{4})/", url)
    if not match:
        return None

    season_code = match.group(1)
    start_year = 2000 + int(season_code[:2])
    end_year = 2000 + int(season_code[2:])
    return f"{start_year}-{end_year}"


def download_csv_dataframe(url: str) -> pd.DataFrame:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SerieAAnalystFixturesUpdater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read().decode("utf-8-sig")
    return load_csv_to_dataframe(io.StringIO(content))


def _safe_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _prepare_source_rows(raw_df: pd.DataFrame, url: str) -> pd.DataFrame:
    normalized_df, _ = normalize_columns(raw_df)
    required_columns = {"match_date", "home_team", "away_team"}
    missing_columns = required_columns - set(normalized_df.columns)
    if missing_columns:
        raise ValueError(f"CSV fixture non valido. Colonne mancanti: {', '.join(sorted(missing_columns))}")

    fixtures_df = normalized_df.copy()
    fixtures_df["_source_order"] = range(len(fixtures_df))
    fixtures_df["match_date"] = parse_match_dates(fixtures_df["match_date"])
    fixtures_df["home_team"] = fixtures_df["home_team"].apply(normalize_team_name)
    fixtures_df["away_team"] = fixtures_df["away_team"].apply(normalize_team_name)
    fixtures_df = fixtures_df.dropna(subset=["match_date", "home_team", "away_team"]).copy()
    fixtures_df = fixtures_df[fixtures_df["home_team"] != fixtures_df["away_team"]].copy()
    if fixtures_df.empty:
        raise ValueError("Il CSV scaricato non contiene righe fixture valide.")

    if "season" not in fixtures_df.columns:
        fixtures_df["season"] = infer_season_from_url(url)
    fixtures_df["season"] = fixtures_df["season"].fillna(infer_season_from_url(url))

    teams = sorted(
        set(fixtures_df["home_team"].dropna().astype(str).tolist())
        | set(fixtures_df["away_team"].dropna().astype(str).tolist())
    )
    matches_per_round = max(len(teams) // 2, 1)
    if "matchday" in fixtures_df.columns and fixtures_df["matchday"].notna().any():
        fixtures_df["matchday"] = pd.to_numeric(fixtures_df["matchday"], errors="coerce")
    else:
        fixtures_df["matchday"] = (fixtures_df["_source_order"] // matches_per_round) + 1

    for goal_column in ["home_goals", "away_goals"]:
        if goal_column not in fixtures_df.columns:
            fixtures_df[goal_column] = pd.NA
        fixtures_df[goal_column] = pd.to_numeric(fixtures_df[goal_column], errors="coerce")

    if "full_time_result" not in fixtures_df.columns:
        fixtures_df["full_time_result"] = pd.NA

    fixtures_df["competition_code"] = DEFAULT_COMPETITION_CODE
    fixtures_df["competition_name"] = DEFAULT_COMPETITION_NAME
    fixtures_df["source_name"] = DEFAULT_SOURCE_NAME
    fixtures_df["source_url"] = url
    return fixtures_df


def _fixture_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
    date_label = match_date.strftime("%Y-%m-%d") if pd.notna(match_date) else ""
    return (
        date_label,
        _safe_string(row.get("home_team")),
        _safe_string(row.get("away_team")),
    )


def _load_seed_match_keys() -> set[tuple[str, str, str]]:
    if not SEED_CSV_PATH.exists() or SEED_CSV_PATH.stat().st_size <= 0:
        return set()

    try:
        seed_df = pd.read_csv(SEED_CSV_PATH)
    except Exception:
        return set()

    if seed_df.empty or not {"match_date", "home_team", "away_team"}.issubset(seed_df.columns):
        return set()

    seed_df = seed_df.copy()
    seed_df["match_date"] = pd.to_datetime(seed_df["match_date"], errors="coerce")
    seed_df["home_team"] = seed_df["home_team"].apply(normalize_team_name)
    seed_df["away_team"] = seed_df["away_team"].apply(normalize_team_name)
    return {_fixture_key(row) for _, row in seed_df.dropna(subset=["match_date", "home_team", "away_team"]).iterrows()}


def _select_fixture_candidates(source_df: pd.DataFrame) -> pd.DataFrame:
    seed_match_keys = _load_seed_match_keys()
    has_seed_reference = bool(seed_match_keys)

    unfinished_mask = (
        source_df["home_goals"].isna()
        | source_df["away_goals"].isna()
        | source_df["full_time_result"].isna()
        | (source_df["full_time_result"].astype(str).str.strip() == "")
    )
    if has_seed_reference:
        missing_from_seed_mask = ~source_df.apply(lambda row: _fixture_key(row) in seed_match_keys, axis=1)
        selected = source_df[unfinished_mask | missing_from_seed_mask].copy()
    else:
        selected = source_df[unfinished_mask].copy()

    selected = selected.sort_values(["matchday", "match_date", "_source_order"], na_position="last").reset_index(drop=True)
    return selected


def build_fixture_seed(url: str = FOOTBALL_DATA_SERIE_A_FIXTURES_URL) -> tuple[pd.DataFrame, list[str]]:
    raw_df = download_csv_dataframe(url)
    source_df = _prepare_source_rows(raw_df, url)
    selected_df = _select_fixture_candidates(source_df)
    warnings: list[str] = []

    if selected_df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS), ["Warning: nessuna fixture futura o non ancora presente nel seed partite."]

    teams = sorted(
        set(source_df["home_team"].dropna().astype(str).tolist())
        | set(source_df["away_team"].dropna().astype(str).tolist())
    )
    expected_round_size = max(len(teams) // 2, 1)
    first_matchday = pd.to_numeric(selected_df["matchday"], errors="coerce").dropna()
    if not first_matchday.empty:
        first_matchday_value = int(first_matchday.min())
        first_round_size = int((pd.to_numeric(selected_df["matchday"], errors="coerce") == first_matchday_value).sum())
        if first_round_size < expected_round_size:
            warnings.append(
                f"Warning: prima giornata fixture con solo {first_round_size} partite; il calendario potrebbe essere parziale."
            )

    fixture_df = selected_df[FIXTURE_COLUMNS].copy()
    fixture_df["match_date"] = pd.to_datetime(fixture_df["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fixture_df["matchday"] = pd.to_numeric(fixture_df["matchday"], errors="coerce").astype("Int64")
    fixture_df = fixture_df.dropna(subset=["season", "match_date", "home_team", "away_team"])
    fixture_df = fixture_df.drop_duplicates(subset=["season", "match_date", "home_team", "away_team"], keep="last")
    fixture_df = fixture_df.sort_values(["matchday", "match_date", "home_team", "away_team"]).reset_index(drop=True)
    return fixture_df, warnings


def update_fixture_seed(url: str = FOOTBALL_DATA_SERIE_A_FIXTURES_URL) -> dict[str, object]:
    fixture_df, warnings = build_fixture_seed(url=url)
    if fixture_df.empty:
        raise ValueError("Nessuna fixture valida trovata: impossibile aggiornare il fixture seed.")

    FIXTURE_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture_df.to_csv(FIXTURE_SEED_PATH, index=False)
    return {
        "url": url,
        "rows": int(len(fixture_df)),
        "path": str(FIXTURE_SEED_PATH),
        "first_date": str(fixture_df["match_date"].iloc[0]) if not fixture_df.empty else None,
        "first_matchday": int(fixture_df["matchday"].dropna().iloc[0]) if not fixture_df["matchday"].dropna().empty else None,
        "warnings": warnings,
        "updated": True,
    }


if __name__ == "__main__":
    strict_update = os.getenv(STRICT_UPDATE_ENV, "").strip() == "1"
    try:
        result = update_fixture_seed()
    except Exception as exc:
        fallback_exists = FIXTURE_SEED_PATH.exists() and FIXTURE_SEED_PATH.stat().st_size > 0
        if not strict_update and fallback_exists:
            print("Warning: fixture update failed, keeping existing serie_a_fixtures_seed.csv")
            print(f"Cause: {exc}")
            sys.exit(0)
        if strict_update and fallback_exists:
            print("Error: fixture strict update failed; existing serie_a_fixtures_seed.csv was not modified.")
            print(f"Cause: {exc}")
            sys.exit(1)
        print("Error: fixture update failed and no existing serie_a_fixtures_seed.csv fallback is available.")
        print(f"Cause: {exc}")
        sys.exit(1)

    print(f"Fixture seed aggiornato da: {result['url']}")
    print(f"Righe esportate: {result['rows']}")
    print(f"Prima data fixture: {result.get('first_date') or 'n/d'}")
    print(f"Prima giornata fixture: {result.get('first_matchday') or 'n/d'}")
    print(f"Output: {result['path']}")
    for warning in result["warnings"]:
        print(warning)
