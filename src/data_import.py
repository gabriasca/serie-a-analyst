from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.config import CANONICAL_COLUMNS, COLUMN_ALIASES, NUMERIC_COLUMNS, REQUIRED_BASE_COLUMNS, TEAM_NAME_ALIASES
from src.db import insert_matches


def standardize_column_name(column_name: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower())
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rename_map: dict[str, str] = {}
    for original_name in df.columns:
        normalized_name = standardize_column_name(original_name)
        rename_map[original_name] = COLUMN_ALIASES.get(normalized_name, normalized_name)

    normalized_df = df.rename(columns=rename_map)
    return normalized_df, rename_map


def validate_required_columns(
    df: pd.DataFrame,
    provided_season: str | None = None,
) -> dict[str, Any]:
    normalized_df, rename_map = normalize_columns(df)
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in normalized_df.columns]

    if "season" not in normalized_df.columns and not (provided_season or "").strip():
        missing.append("season")

    return {
        "normalized_df": normalized_df,
        "rename_map": rename_map,
        "missing_columns": missing,
        "valid": len(missing) == 0,
    }


def normalize_team_name(team_name: Any) -> str | None:
    if pd.isna(team_name):
        return None

    cleaned = re.sub(r"\s+", " ", str(team_name).strip())
    if not cleaned:
        return None

    if cleaned.isupper():
        cleaned = cleaned.title()

    alias = TEAM_NAME_ALIASES.get(cleaned.lower())
    return alias or cleaned


def normalize_result(home_goals: Any, away_goals: Any, current_value: Any = None) -> str | None:
    mapping = {
        "H": "H",
        "1": "H",
        "HOME": "H",
        "D": "D",
        "X": "D",
        "DRAW": "D",
        "A": "A",
        "2": "A",
        "AWAY": "A",
    }

    if pd.notna(current_value):
        normalized = mapping.get(str(current_value).strip().upper())
        if normalized:
            return normalized

    if pd.isna(home_goals) or pd.isna(away_goals):
        return None

    if int(home_goals) > int(away_goals):
        return "H"
    if int(home_goals) < int(away_goals):
        return "A"
    return "D"


def parse_match_dates(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, errors="coerce", format="%Y-%m-%d")
    unresolved = parsed.isna()

    if unresolved.any():
        dayfirst_parse = pd.to_datetime(date_series[unresolved], errors="coerce", dayfirst=True)
        parsed.loc[unresolved] = dayfirst_parse

    unresolved = parsed.isna()
    if unresolved.any():
        fallback_parse = pd.to_datetime(date_series[unresolved], errors="coerce", dayfirst=False)
        parsed.loc[unresolved] = fallback_parse

    return parsed.dt.strftime("%Y-%m-%d")


def clean_match_data(
    df: pd.DataFrame,
    default_season: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    normalized_df, _ = normalize_columns(df)
    cleaned_df = normalized_df.copy()

    if "season" not in cleaned_df.columns:
        cleaned_df["season"] = (default_season or "").strip() or None
    elif default_season:
        cleaned_df["season"] = cleaned_df["season"].fillna(default_season)
        cleaned_df["season"] = cleaned_df["season"].replace("", default_season)

    cleaned_df["match_date"] = parse_match_dates(cleaned_df["match_date"])
    cleaned_df["home_team"] = cleaned_df["home_team"].apply(normalize_team_name)
    cleaned_df["away_team"] = cleaned_df["away_team"].apply(normalize_team_name)

    for column in NUMERIC_COLUMNS:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        else:
            cleaned_df[column] = pd.NA

    if "full_time_result" not in cleaned_df.columns:
        cleaned_df["full_time_result"] = pd.NA

    cleaned_df["full_time_result"] = cleaned_df.apply(
        lambda row: normalize_result(
            row.get("home_goals"),
            row.get("away_goals"),
            row.get("full_time_result"),
        ),
        axis=1,
    )

    cleaned_df["season"] = cleaned_df["season"].astype("string").str.strip()
    cleaned_df["source_name"] = source_name

    for column in CANONICAL_COLUMNS:
        if column not in cleaned_df.columns:
            cleaned_df[column] = pd.NA

    cleaned_df = cleaned_df[CANONICAL_COLUMNS]

    required_final_columns = REQUIRED_BASE_COLUMNS + ["season", "full_time_result"]
    cleaned_df = cleaned_df.dropna(subset=required_final_columns)
    cleaned_df = cleaned_df[cleaned_df["home_team"] != cleaned_df["away_team"]]

    cleaned_df["season"] = cleaned_df["season"].astype(str).str.strip()
    cleaned_df["match_date"] = cleaned_df["match_date"].astype(str)
    cleaned_df["full_time_result"] = cleaned_df["full_time_result"].astype(str).str.upper()

    for required_numeric in ["home_goals", "away_goals"]:
        cleaned_df[required_numeric] = (
            pd.to_numeric(cleaned_df[required_numeric], errors="coerce")
            .astype("Int64")
        )

    for optional_numeric in [
        column for column in NUMERIC_COLUMNS if column not in {"home_goals", "away_goals"}
    ]:
        cleaned_df[optional_numeric] = (
            pd.to_numeric(cleaned_df[optional_numeric], errors="coerce")
            .astype("Int64")
        )

    cleaned_df = cleaned_df.dropna(subset=["home_goals", "away_goals"])
    cleaned_df = cleaned_df.where(pd.notna(cleaned_df), None)
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


def load_csv_to_dataframe(file_source: Any) -> pd.DataFrame:
    return pd.read_csv(file_source, sep=None, engine="python")


def save_dataframe_to_sqlite(df: pd.DataFrame, source_name: str | None = None) -> dict[str, int]:
    return insert_matches(df, source_name=source_name)
