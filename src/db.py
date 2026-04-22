from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.config import (
    CANONICAL_COLUMNS,
    DB_PATH,
    DEFAULT_COMPETITION_CODE,
    DEFAULT_COMPETITION_NAME,
    DEFAULT_COMPETITION_TYPE,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    RAW_DATA_DIR,
)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    competition_code TEXT,
    competition_name TEXT,
    competition_type TEXT,
    matchday INTEGER,
    round TEXT,
    stage TEXT,
    external_match_id TEXT,
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
    source_url TEXT,
    updated_at TEXT,
    xg_home REAL,
    xg_away REAL,
    proxy_xg_home REAL,
    proxy_xg_away REAL,
    proxy_xg_model_version TEXT,
    elo_home_pre REAL,
    elo_away_pre REAL,
    rest_days_home REAL,
    rest_days_away REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, match_date, home_team, away_team)
);
"""

MATCH_MIGRATION_COLUMNS = {
    "competition_code": "TEXT",
    "competition_name": "TEXT",
    "competition_type": "TEXT",
    "matchday": "INTEGER",
    "round": "TEXT",
    "stage": "TEXT",
    "external_match_id": "TEXT",
    "source_url": "TEXT",
    "updated_at": "TEXT",
    "xg_home": "REAL",
    "xg_away": "REAL",
    "proxy_xg_home": "REAL",
    "proxy_xg_away": "REAL",
    "proxy_xg_model_version": "TEXT",
    "elo_home_pre": "REAL",
    "elo_away_pre": "REAL",
    "rest_days_home": "REAL",
    "rest_days_away": "REAL",
}

CREATE_COMPETITIONS_SQL = """
CREATE TABLE IF NOT EXISTS competitions (
    competition_code TEXT PRIMARY KEY,
    competition_name TEXT NOT NULL,
    competition_type TEXT NOT NULL,
    country TEXT,
    provider TEXT,
    provider_competition_id TEXT,
    active INTEGER DEFAULT 1
);
"""

CREATE_TEAM_ALIASES_SQL = """
CREATE TABLE IF NOT EXISTS team_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_team_name TEXT NOT NULL,
    alias TEXT NOT NULL,
    provider TEXT,
    UNIQUE(alias, provider)
);
"""

CREATE_DATA_SOURCES_SQL = """
CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    source_url TEXT,
    source_type TEXT,
    last_successful_update TEXT,
    last_error TEXT,
    notes TEXT
);
"""

CREATE_TEAM_RATINGS_SQL = """
CREATE TABLE IF NOT EXISTS team_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    rating_date TEXT NOT NULL,
    rating_type TEXT NOT NULL,
    rating_value REAL NOT NULL,
    source_name TEXT,
    UNIQUE(team_name, rating_date, rating_type, source_name)
);
"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);",
    "CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches(match_date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_competition_code ON matches(competition_code);",
    "CREATE INDEX IF NOT EXISTS idx_matches_competition_type ON matches(competition_type);",
]

INSERT_COLUMNS = CANONICAL_COLUMNS


def ensure_data_paths() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    ensure_data_paths()
    return sqlite3.connect(DB_PATH)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _get_existing_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    if not _table_exists(conn, table_name):
        return []

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def get_existing_columns(table_name: str) -> list[str]:
    with get_connection() as conn:
        return _get_existing_columns(conn, table_name)


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> bool:
    existing_columns = _get_existing_columns(conn, table_name)
    if column_name in existing_columns:
        return False

    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
    return True


def ensure_column(table_name: str, column_name: str, column_definition: str) -> bool:
    with get_connection() as conn:
        changed = _ensure_column(conn, table_name, column_name, column_definition)
        conn.commit()
    return changed


def _seed_default_competitions(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO competitions (
            competition_code,
            competition_name,
            competition_type,
            country,
            provider,
            provider_competition_id,
            active
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            DEFAULT_COMPETITION_CODE,
            DEFAULT_COMPETITION_NAME,
            DEFAULT_COMPETITION_TYPE,
            "Italy",
            "internal_default",
            None,
            1,
        ),
    )


def seed_default_competitions() -> None:
    with get_connection() as conn:
        _seed_default_competitions(conn)
        conn.commit()


def _backfill_serie_a_competition_fields(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE matches
        SET competition_code = ?,
            competition_name = ?,
            competition_type = ?
        WHERE COALESCE(TRIM(competition_code), '') = ''
        """,
        (DEFAULT_COMPETITION_CODE, DEFAULT_COMPETITION_NAME, DEFAULT_COMPETITION_TYPE),
    )
    conn.execute(
        """
        UPDATE matches
        SET competition_name = ?
        WHERE competition_code = ? AND COALESCE(TRIM(competition_name), '') = ''
        """,
        (DEFAULT_COMPETITION_NAME, DEFAULT_COMPETITION_CODE),
    )
    conn.execute(
        """
        UPDATE matches
        SET competition_type = ?
        WHERE competition_code = ? AND COALESCE(TRIM(competition_type), '') = ''
        """,
        (DEFAULT_COMPETITION_TYPE, DEFAULT_COMPETITION_CODE),
    )


def backfill_serie_a_competition_fields() -> None:
    with get_connection() as conn:
        _backfill_serie_a_competition_fields(conn)
        conn.commit()


def _sync_competitions_from_matches(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "matches"):
        return

    conn.execute(
        """
        INSERT OR IGNORE INTO competitions (
            competition_code,
            competition_name,
            competition_type,
            active
        )
        SELECT DISTINCT
            competition_code,
            COALESCE(NULLIF(TRIM(competition_name), ''), competition_code),
            COALESCE(NULLIF(TRIM(competition_type), ''), ?),
            1
        FROM matches
        WHERE COALESCE(TRIM(competition_code), '') <> ''
        """,
        (DEFAULT_COMPETITION_TYPE,),
    )


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    for column_name, column_definition in MATCH_MIGRATION_COLUMNS.items():
        _ensure_column(conn, "matches", column_name, column_definition)

    conn.execute(CREATE_COMPETITIONS_SQL)
    conn.execute(CREATE_TEAM_ALIASES_SQL)
    conn.execute(CREATE_DATA_SOURCES_SQL)
    conn.execute(CREATE_TEAM_RATINGS_SQL)

    _seed_default_competitions(conn)
    _backfill_serie_a_competition_fields(conn)
    _sync_competitions_from_matches(conn)


def run_schema_migrations() -> None:
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        _run_schema_migrations(conn)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        conn.commit()


def init_db() -> None:
    run_schema_migrations()


def _build_match_filters(
    season: str | None = None,
    competition_code: str | None = None,
    competition_type: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []

    if season:
        clauses.append("season = ?")
        params.append(season)
    if competition_code:
        clauses.append("competition_code = ?")
        params.append(competition_code)
    if competition_type:
        clauses.append("competition_type = ?")
        params.append(competition_type)

    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, tuple(params)


def database_has_matches() -> bool:
    return count_matches() > 0


def count_matches(
    season: str | None = None,
    competition_code: str | None = None,
    competition_type: str | None = None,
) -> int:
    init_db()
    where_sql, params = _build_match_filters(season, competition_code, competition_type)
    query = f"SELECT COUNT(*) FROM matches{where_sql}"

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return int(cursor.fetchone()[0])


def insert_matches(
    df: pd.DataFrame,
    source_name: str | None = None,
    source_url: str | None = None,
) -> dict[str, int]:
    init_db()
    if df.empty:
        return {"inserted": 0, "duplicates": 0}

    insert_df = df.copy()

    if "source_name" not in insert_df.columns:
        insert_df["source_name"] = source_name
    elif source_name:
        insert_df["source_name"] = insert_df["source_name"].fillna(source_name)

    if "source_url" not in insert_df.columns:
        insert_df["source_url"] = source_url
    elif source_url:
        insert_df["source_url"] = insert_df["source_url"].fillna(source_url)

    default_text_columns = {
        "competition_code": DEFAULT_COMPETITION_CODE,
        "competition_name": DEFAULT_COMPETITION_NAME,
        "competition_type": DEFAULT_COMPETITION_TYPE,
    }
    for column_name, default_value in default_text_columns.items():
        if column_name not in insert_df.columns:
            insert_df[column_name] = default_value
        else:
            insert_df[column_name] = insert_df[column_name].replace("", pd.NA).fillna(default_value)

    for column in INSERT_COLUMNS:
        if column not in insert_df.columns:
            insert_df[column] = None

    for column in INTEGER_COLUMNS:
        insert_df[column] = pd.to_numeric(insert_df[column], errors="coerce").astype("Int64")

    for column in FLOAT_COLUMNS:
        insert_df[column] = pd.to_numeric(insert_df[column], errors="coerce")

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
        _sync_competitions_from_matches(conn)
        conn.commit()
        inserted = conn.total_changes - before

    return {"inserted": inserted, "duplicates": len(rows) - inserted}


def fetch_matches(
    season: str | None = None,
    competition_code: str | None = None,
    competition_type: str | None = None,
) -> pd.DataFrame:
    init_db()
    where_sql, params = _build_match_filters(season, competition_code, competition_type)
    query = f"SELECT * FROM matches{where_sql} ORDER BY match_date ASC, id ASC"

    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["match_date"])

    if df.empty:
        return df

    for column in INTEGER_COLUMNS + FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def list_seasons(
    competition_code: str | None = None,
    competition_type: str | None = None,
) -> list[str]:
    init_db()
    where_sql, params = _build_match_filters(None, competition_code, competition_type)
    query = f"SELECT DISTINCT season FROM matches{where_sql} ORDER BY season DESC"

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


def list_teams(
    season: str | None = None,
    competition_code: str | None = None,
    competition_type: str | None = None,
) -> list[str]:
    init_db()
    where_sql, params = _build_match_filters(season, competition_code, competition_type)

    home_query = f"SELECT home_team AS team FROM matches{where_sql}"
    away_query = f"SELECT away_team AS team FROM matches{where_sql}"
    query = f"SELECT team FROM ({home_query} UNION {away_query}) ORDER BY team ASC"
    combined_params = params + params

    with get_connection() as conn:
        cursor = conn.execute(query, combined_params)
        return [row[0] for row in cursor.fetchall()]


def list_data_sources() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(source_name, '(non specificata)') AS source_name,
                NULLIF(MAX(COALESCE(source_url, '')), '') AS source_url,
                COUNT(*) AS match_count
            FROM matches
            GROUP BY COALESCE(source_name, '(non specificata)')
            ORDER BY match_count DESC, source_name ASC
            """
        ).fetchall()

    return [
        {
            "source_name": row[0],
            "source_url": row[1],
            "match_count": int(row[2]),
        }
        for row in rows
    ]


def seed_data_source(
    source_name: str,
    source_url: str | None = None,
    source_type: str | None = None,
    notes: str | None = None,
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO data_sources (source_name, source_url, source_type, notes)
            SELECT ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM data_sources
                WHERE source_name = ? AND COALESCE(source_url, '') = COALESCE(?, '')
            )
            """,
            (source_name, source_url, source_type, notes, source_name, source_url),
        )
        conn.commit()


def list_competitions() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT competition_code, competition_name, competition_type, country, provider, provider_competition_id, active
            FROM competitions
            ORDER BY active DESC, competition_name ASC, competition_code ASC
            """
        ).fetchall()

    return [
        {
            "competition_code": row[0],
            "competition_name": row[1],
            "competition_type": row[2],
            "country": row[3],
            "provider": row[4],
            "provider_competition_id": row[5],
            "active": int(row[6]) if row[6] is not None else 0,
        }
        for row in rows
    ]


def get_competition_summary() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                COALESCE(competition_code, ?) AS competition_code,
                COALESCE(competition_name, ?) AS competition_name,
                COALESCE(competition_type, ?) AS competition_type,
                COUNT(*) AS match_count,
                COUNT(DISTINCT season) AS season_count
            FROM matches
            GROUP BY COALESCE(competition_code, ?), COALESCE(competition_name, ?), COALESCE(competition_type, ?)
            ORDER BY match_count DESC, competition_name ASC
            """,
            (
                DEFAULT_COMPETITION_CODE,
                DEFAULT_COMPETITION_NAME,
                DEFAULT_COMPETITION_TYPE,
                DEFAULT_COMPETITION_CODE,
                DEFAULT_COMPETITION_NAME,
                DEFAULT_COMPETITION_TYPE,
            ),
        ).fetchall()

    return [
        {
            "competition_code": row[0],
            "competition_name": row[1],
            "competition_type": row[2],
            "match_count": int(row[3]),
            "season_count": int(row[4]),
        }
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
            {
                "source_name": row[0],
                "source_url": row[1],
                "match_count": int(row[2]),
            }
            for row in conn.execute(
                """
                SELECT
                    COALESCE(source_name, '(non specificata)') AS source_name,
                    NULLIF(MAX(COALESCE(source_url, '')), '') AS source_url,
                    COUNT(*) AS match_count
                FROM matches
                GROUP BY COALESCE(source_name, '(non specificata)')
                ORDER BY match_count DESC, source_name ASC
                """
            ).fetchall()
        ]

    competitions = get_competition_summary()
    return {
        "database_ready": database_ready,
        "db_path": str(DB_PATH),
        "match_count": match_count,
        "team_count": team_count,
        "seasons": seasons,
        "sources": sources,
        "competitions": competitions,
    }


