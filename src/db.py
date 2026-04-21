from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.config import CANONICAL_COLUMNS, DB_PATH, NUMERIC_COLUMNS, RAW_DATA_DIR


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    match_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    full_time_result TEXT NOT NULL,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_cards INTEGER,
    away_cards INTEGER,
    source_name TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, match_date, home_team, away_team)
);
"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);",
    "CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches(match_date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team);",
]

INSERT_COLUMNS = CANONICAL_COLUMNS


def ensure_data_paths() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    ensure_data_paths()
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        conn.commit()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def database_has_matches() -> bool:
    return count_matches() > 0


def count_matches(season: str | None = None) -> int:
    init_db()
    query = "SELECT COUNT(*) FROM matches"
    params: tuple[Any, ...] = ()

    if season:
        query += " WHERE season = ?"
        params = (season,)

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return int(cursor.fetchone()[0])


def insert_matches(df: pd.DataFrame, source_name: str | None = None) -> dict[str, int]:
    init_db()
    if df.empty:
        return {"inserted": 0, "duplicates": 0}

    insert_df = df.copy()
    if "source_name" not in insert_df.columns:
        insert_df["source_name"] = source_name
    elif source_name:
        insert_df["source_name"] = insert_df["source_name"].fillna(source_name)

    for column in INSERT_COLUMNS:
        if column not in insert_df.columns:
            insert_df[column] = None

    for column in NUMERIC_COLUMNS:
        insert_df[column] = pd.to_numeric(insert_df[column], errors="coerce").astype("Int64")

    insert_df = insert_df[INSERT_COLUMNS].where(pd.notna(insert_df[INSERT_COLUMNS]), None)

    rows = [
        tuple(record[column] for column in INSERT_COLUMNS)
        for record in insert_df.to_dict(orient="records")
    ]

    placeholders = ", ".join(["?"] * len(INSERT_COLUMNS))
    column_sql = ", ".join(INSERT_COLUMNS)
    query = f"INSERT OR IGNORE INTO matches ({column_sql}) VALUES ({placeholders})"

    with get_connection() as conn:
        before = conn.total_changes
        conn.executemany(query, rows)
        conn.commit()
        inserted = conn.total_changes - before

    return {"inserted": inserted, "duplicates": len(rows) - inserted}


def fetch_matches(season: str | None = None) -> pd.DataFrame:
    init_db()
    query = "SELECT * FROM matches"
    params: list[Any] = []

    if season:
        query += " WHERE season = ?"
        params.append(season)

    query += " ORDER BY match_date ASC, id ASC"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["match_date"])

    if df.empty:
        return df

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def list_seasons() -> list[str]:
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT season FROM matches ORDER BY season DESC")
        return [row[0] for row in cursor.fetchall()]


def list_teams(season: str | None = None) -> list[str]:
    init_db()
    if season:
        query = """
        SELECT team FROM (
            SELECT home_team AS team FROM matches WHERE season = ?
            UNION
            SELECT away_team AS team FROM matches WHERE season = ?
        ) ORDER BY team ASC
        """
        params = (season, season)
    else:
        query = """
        SELECT team FROM (
            SELECT home_team AS team FROM matches
            UNION
            SELECT away_team AS team FROM matches
        ) ORDER BY team ASC
        """
        params = ()

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


def list_data_sources() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(source_name, '(non specificata)') AS source_name,
                COUNT(*) AS match_count
            FROM matches
            GROUP BY COALESCE(source_name, '(non specificata)')
            ORDER BY match_count DESC, source_name ASC
            """
        ).fetchall()

    return [
        {"source_name": row[0], "match_count": int(row[1])}
        for row in rows
    ]


def delete_all_matches() -> int:
    init_db()
    deleted_count = count_matches()

    with get_connection() as conn:
        conn.execute("DELETE FROM matches")
        conn.commit()

    return deleted_count


def delete_matches_by_season(season: str) -> int:
    init_db()
    deleted_count = count_matches(season=season)

    with get_connection() as conn:
        conn.execute("DELETE FROM matches WHERE season = ?", (season,))
        conn.commit()

    return deleted_count


def get_database_status() -> dict[str, Any]:
    init_db()

    with get_connection() as conn:
        database_ready = _table_exists(conn, "matches")

        match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        team_count = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT home_team AS team FROM matches
                UNION
                SELECT away_team AS team FROM matches
            )
            """
        ).fetchone()[0]
        seasons = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT season FROM matches ORDER BY season DESC"
            ).fetchall()
        ]
        sources = [
            {"source_name": row[0], "match_count": int(row[1])}
            for row in conn.execute(
                """
                SELECT
                    COALESCE(source_name, '(non specificata)') AS source_name,
                    COUNT(*) AS match_count
                FROM matches
                GROUP BY COALESCE(source_name, '(non specificata)')
                ORDER BY match_count DESC, source_name ASC
                """
            ).fetchall()
        ]

    return {
        "database_ready": database_ready,
        "db_path": str(DB_PATH),
        "match_count": match_count,
        "team_count": team_count,
        "seasons": seasons,
        "sources": sources,
    }
