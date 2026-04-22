from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DB_PATH, SEED_CSV_PATH, TEAM_RATINGS_SEED_PATH
from src.data_import import clean_match_data, load_csv_to_dataframe
from src.db import database_has_matches, init_db, insert_matches, seed_data_source
from src.ratings import insert_team_ratings_from_seed


SEED_SOURCE_NAME = "seed_csv"


def get_seed_dataset() -> pd.DataFrame:
    if not SEED_CSV_PATH.exists():
        return pd.DataFrame()

    raw_df = load_csv_to_dataframe(SEED_CSV_PATH)
    return clean_match_data(
        raw_df,
        source_name=SEED_SOURCE_NAME,
        source_url=str(SEED_CSV_PATH),
    )


def bootstrap_database() -> dict[str, Any]:
    db_file_already_present = DB_PATH.exists()
    init_db()

    result: dict[str, Any] = {
        "seed_loaded": False,
        "reason": "existing_db_file" if db_file_already_present else "not_loaded",
        "inserted": 0,
        "duplicates": 0,
        "seed_path": str(SEED_CSV_PATH),
        "team_ratings_loaded": False,
        "team_ratings_inserted": 0,
        "team_ratings_seed_path": str(TEAM_RATINGS_SEED_PATH),
    }

    if not db_file_already_present and SEED_CSV_PATH.exists() and not database_has_matches():
        seed_df = get_seed_dataset()
        if not seed_df.empty:
            save_stats = insert_matches(seed_df, source_name=SEED_SOURCE_NAME, source_url=str(SEED_CSV_PATH))
            seed_data_source(
                source_name=SEED_SOURCE_NAME,
                source_url=str(SEED_CSV_PATH),
                source_type="seed_csv",
                notes="Seed pubblico usato per bootstrap dell'app read-only.",
            )
            result.update(
                {
                    "seed_loaded": save_stats["inserted"] > 0,
                    "reason": "seed_loaded",
                    "inserted": save_stats["inserted"],
                    "duplicates": save_stats["duplicates"],
                }
            )
        else:
            result["reason"] = "empty_seed_dataset"
    elif not SEED_CSV_PATH.exists():
        result["reason"] = "missing_seed_file"
    elif database_has_matches():
        result["reason"] = "database_not_empty"

    try:
        ratings_result = insert_team_ratings_from_seed()
    except Exception:  # pragma: no cover - bootstrap must stay resilient in public mode
        ratings_result = {"loaded": False, "inserted": 0}

    result["team_ratings_loaded"] = bool(ratings_result.get("loaded", False))
    result["team_ratings_inserted"] = int(ratings_result.get("inserted", 0) or 0)
    return result
