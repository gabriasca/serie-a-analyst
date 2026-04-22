from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import CANONICAL_COLUMNS, DB_PATH, SEED_CSV_PATH
from src.db import init_db


EXPORT_COLUMNS = CANONICAL_COLUMNS


def export_seed() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database non trovato: {DB_PATH}")

    init_db()
    SEED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT {", ".join(EXPORT_COLUMNS)}
            FROM matches
            ORDER BY season, competition_code, match_date, home_team, away_team, id
            """,
            conn,
        )

    df.to_csv(SEED_CSV_PATH, index=False)
    print(f"Seed esportato: {SEED_CSV_PATH}")
    print(f"Righe esportate: {len(df)}")


if __name__ == "__main__":
    export_seed()
