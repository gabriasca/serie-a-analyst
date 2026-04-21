from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "serie_a.db"
SEED_CSV_PATH = BASE_DIR / "data" / "raw" / "serie_a_seed.csv"

EXPORT_COLUMNS = [
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "full_time_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_cards",
    "away_cards",
    "source_name",
]


def export_seed() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database non trovato: {DB_PATH}")

    SEED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT {", ".join(EXPORT_COLUMNS)}
            FROM matches
            ORDER BY season, match_date, home_team, away_team, id
            """,
            conn,
        )

    df.to_csv(SEED_CSV_PATH, index=False)
    print(f"Seed esportato: {SEED_CSV_PATH}")
    print(f"Righe esportate: {len(df)}")


if __name__ == "__main__":
    export_seed()
