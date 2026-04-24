from __future__ import annotations

import sqlite3
from typing import Any
from pathlib import Path

import pandas as pd

from src.config import TEAM_RATINGS_SEED_PATH
from src.db import get_connection, init_db


DEFAULT_RATING_TYPE = "elo"
DEFAULT_SOURCE_NAME = "ClubElo"
STRENGTH_BAND_MOLTO_ALTA = "molto alta"
STRENGTH_BAND_ALTA = "alta"
STRENGTH_BAND_MEDIA = "media"
STRENGTH_BAND_BASSA = "bassa"

REQUIRED_SEED_COLUMNS = [
    "team_name",
    "rating_date",
    "rating_type",
    "rating_value",
    "source_name",
    "source_url",
]


def _normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _fetch_team_alias_map() -> dict[str, str]:
    init_db()
    alias_map: dict[str, str] = {}
    with get_connection() as conn:
        if not _table_exists(conn, "team_aliases"):
            return alias_map
        try:
            rows = conn.execute(
                "SELECT canonical_team_name, alias FROM team_aliases WHERE canonical_team_name IS NOT NULL AND alias IS NOT NULL"
            ).fetchall()
        except sqlite3.Error:
            return alias_map

    for canonical_team_name, alias in rows:
        alias_map[_normalize_name(alias)] = str(canonical_team_name)
    return alias_map


def _canonical_team_name(team_name: str) -> str:
    alias_map = _fetch_team_alias_map()
    normalized = _normalize_name(team_name)
    return alias_map.get(normalized, str(team_name).strip())


def load_team_ratings_seed(path: str | None = None) -> pd.DataFrame:
    seed_path = TEAM_RATINGS_SEED_PATH if path is None else Path(path)
    if not seed_path.exists():
        return pd.DataFrame(columns=REQUIRED_SEED_COLUMNS)

    raw_df = pd.read_csv(seed_path)
    if raw_df.empty:
        return pd.DataFrame(columns=REQUIRED_SEED_COLUMNS)

    for column in REQUIRED_SEED_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = None

    seed_df = raw_df[REQUIRED_SEED_COLUMNS].copy()
    seed_df["team_name"] = seed_df["team_name"].astype(str).str.strip()
    seed_df["rating_type"] = seed_df["rating_type"].fillna(DEFAULT_RATING_TYPE).astype(str).str.strip().str.lower()
    seed_df["source_name"] = seed_df["source_name"].fillna(DEFAULT_SOURCE_NAME).astype(str).str.strip()
    seed_df["source_url"] = seed_df["source_url"].where(seed_df["source_url"].notna(), None)
    seed_df["rating_date"] = pd.to_datetime(seed_df["rating_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    seed_df["rating_value"] = pd.to_numeric(seed_df["rating_value"], errors="coerce")

    seed_df = seed_df.dropna(subset=["team_name", "rating_date", "rating_type", "rating_value", "source_name"])
    seed_df = seed_df[
        (seed_df["team_name"].astype(str).str.len() > 0)
        & (seed_df["team_name"].astype(str).str.lower() != "nan")
    ]
    seed_df = seed_df.drop_duplicates(
        subset=["team_name", "rating_date", "rating_type", "source_name"],
        keep="last",
    ).reset_index(drop=True)
    return seed_df


def insert_team_ratings_from_seed(seed_df: pd.DataFrame | None = None) -> dict[str, Any]:
    init_db()
    if seed_df is None:
        seed_df = load_team_ratings_seed()

    if seed_df.empty:
        return {"loaded": False, "inserted": 0, "rows": 0, "reason": "missing_or_empty_seed"}

    rows = [
        (
            str(row["team_name"]).strip(),
            str(row["rating_date"]),
            str(row["rating_type"]).strip().lower(),
            float(row["rating_value"]),
            str(row["source_name"]).strip(),
            row["source_url"] if pd.notna(row["source_url"]) else None,
        )
        for row in seed_df.to_dict(orient="records")
    ]

    with get_connection() as conn:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO team_ratings (
                team_name,
                rating_date,
                rating_type,
                rating_value,
                source_name,
                source_url
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        inserted = conn.total_changes - before

    return {
        "loaded": inserted > 0,
        "inserted": int(inserted),
        "rows": int(len(rows)),
        "reason": "seed_loaded" if inserted > 0 else "no_new_rows",
    }


def fetch_latest_team_ratings(
    rating_type: str = DEFAULT_RATING_TYPE,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        if not _table_exists(conn, "team_ratings"):
            return pd.DataFrame()
        try:
            ratings_df = pd.read_sql_query(
                """
                SELECT team_name, rating_date, rating_type, rating_value, source_name, source_url
                FROM team_ratings
                WHERE rating_type = ?
                ORDER BY rating_date DESC, rowid DESC
                """,
                conn,
                params=(rating_type,),
            )
        except sqlite3.Error:
            return pd.DataFrame()

    if ratings_df.empty:
        return ratings_df

    ratings_df["rating_date"] = pd.to_datetime(ratings_df["rating_date"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["team_name", "rating_date", "rating_value"])
    ratings_df = ratings_df.sort_values(["team_name", "rating_date"], ascending=[True, False])
    ratings_df = ratings_df.drop_duplicates(subset=["team_name"], keep="first").reset_index(drop=True)

    if teams is not None:
        ratings_df = ratings_df[ratings_df["team_name"].isin(teams)].reset_index(drop=True)

    return ratings_df


def get_team_rating(
    team_name: str,
    rating_date: str | None = None,
    rating_type: str = DEFAULT_RATING_TYPE,
) -> dict[str, Any] | None:
    canonical_name = _canonical_team_name(team_name)
    init_db()
    with get_connection() as conn:
        if not _table_exists(conn, "team_ratings"):
            return None

        params: list[Any] = [canonical_name, rating_type]
        query = """
            SELECT team_name, rating_date, rating_type, rating_value, source_name, source_url
            FROM team_ratings
            WHERE team_name = ? AND rating_type = ?
        """
        if rating_date:
            query += " AND rating_date <= ?"
            params.append(rating_date)
        query += " ORDER BY rating_date DESC, rowid DESC LIMIT 1"

        try:
            row = conn.execute(query, tuple(params)).fetchone()
        except sqlite3.Error:
            return None

    if row is None:
        return None

    return {
        "team_name": str(row[0]),
        "rating_date": str(row[1]),
        "rating_type": str(row[2]),
        "rating_value": float(row[3]),
        "source_name": str(row[4]),
        "source_url": row[5],
    }


def _strength_band_from_rank(rank: int, total: int) -> str | None:
    if total <= 0:
        return None
    if total == 1:
        return STRENGTH_BAND_MEDIA

    percentile = (rank - 1) / max(total - 1, 1)
    if percentile <= 0.2:
        return STRENGTH_BAND_MOLTO_ALTA
    if percentile <= 0.45:
        return STRENGTH_BAND_ALTA
    if percentile <= 0.75:
        return STRENGTH_BAND_MEDIA
    return STRENGTH_BAND_BASSA


def enrich_standings_with_ratings(
    standings_df: pd.DataFrame,
    ratings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if standings_df.empty:
        return standings_df.copy()

    working_df = standings_df.reset_index()
    index_name = standings_df.index.name or working_df.columns[0]
    rating_df = ratings_df.copy() if ratings_df is not None else fetch_latest_team_ratings(teams=working_df["Team"].astype(str).tolist())
    if not rating_df.empty:
        rating_df["team_name"] = rating_df["team_name"].astype(str)
        rating_df["rating_value"] = pd.to_numeric(rating_df["rating_value"], errors="coerce")
        rating_df["rating_date"] = pd.to_datetime(rating_df["rating_date"], errors="coerce")
        rating_df = rating_df.dropna(subset=["team_name", "rating_value"])
        rating_df = rating_df.sort_values(["team_name", "rating_date"], ascending=[True, False])
        rating_df = rating_df.drop_duplicates(subset=["team_name"], keep="first").reset_index(drop=True)
        rating_df = rating_df[rating_df["team_name"].isin(working_df["Team"].astype(str).tolist())].reset_index(drop=True)

    if rating_df.empty:
        enriched_df = working_df.copy()
        enriched_df["Elo"] = pd.NA
        enriched_df["Elo Rank"] = pd.NA
        enriched_df["Fascia forza"] = pd.NA
        enriched_df["Rating Date"] = pd.NA
        enriched_df["Rating Source"] = pd.NA
        return enriched_df.set_index(index_name)

    ranked_ratings = rating_df.sort_values(["rating_value", "team_name"], ascending=[False, True]).reset_index(drop=True)
    ranked_ratings["Elo Rank"] = range(1, len(ranked_ratings) + 1)
    ranked_ratings["Fascia forza"] = ranked_ratings["Elo Rank"].apply(
        lambda rank: _strength_band_from_rank(int(rank), len(ranked_ratings))
    )
    ranked_ratings = ranked_ratings.rename(
        columns={
            "team_name": "Team",
            "rating_value": "Elo",
            "rating_date": "Rating Date",
            "source_name": "Rating Source",
        }
    )

    enriched_df = working_df.merge(
        ranked_ratings[["Team", "Elo", "Elo Rank", "Fascia forza", "Rating Date", "Rating Source"]],
        on="Team",
        how="left",
    )
    return enriched_df.set_index(index_name)


def build_strength_bucket_map(standings_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]], str]:
    if standings_df.empty:
        return {}, {}, "classifica"

    if "Team" not in standings_df.columns:
        return {}, {}, "classifica"

    teams = standings_df["Team"].astype(str).tolist()
    use_rating_buckets = "Elo" in standings_df.columns and standings_df["Elo"].notna().all()
    if use_rating_buckets:
        ordered_teams = (
            standings_df.sort_values(["Elo", "Pts", "DR", "GF", "Team"], ascending=[False, False, False, False, True])[
                "Team"
            ]
            .astype(str)
            .tolist()
        )
        source = "elo"
    else:
        ordered_teams = standings_df.sort_index()["Team"].astype(str).tolist()
        source = "classifica"

    team_count = len(ordered_teams)
    top_limit = min(6, team_count)
    bottom_start = max(top_limit + 1, team_count - 5)

    bucket_map: dict[str, str] = {}
    bucket_teams = {
        "top": [],
        "middle": [],
        "bottom": [],
    }

    for position, team_name in enumerate(ordered_teams, start=1):
        if position <= top_limit:
            bucket_key = "top"
        elif position >= bottom_start:
            bucket_key = "bottom"
        else:
            bucket_key = "middle"
        bucket_map[team_name] = bucket_key
        bucket_teams[bucket_key].append(team_name)

    return bucket_map, bucket_teams, source
