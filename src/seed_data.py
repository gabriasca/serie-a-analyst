from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DB_PATH, SEED_CSV_PATH
from src.data_import import clean_match_data, load_csv_to_dataframe
from src.db import database_has_matches, init_db, insert_matches


def get_seed_dataset() -> pd.DataFrame:
    if not SEED_CSV_PATH.exists():
        return pd.DataFrame()

    raw_df = load_csv_to_dataframe(SEED_CSV_PATH)
    return clean_match_data(raw_df, source_name="seed_csv")


def bootstrap_database() -> dict[str, Any]:
    db_file_already_present = DB_PATH.exists()
    init_db()

    if db_file_already_present:
        return {"seed_loaded": False, "reason": "existing_db_file"}

    if not SEED_CSV_PATH.exists():
        return {"seed_loaded": False, "reason": "missing_seed_file"}

    if database_has_matches():
        return {"seed_loaded": False, "reason": "database_not_empty"}

    seed_df = get_seed_dataset()
    if seed_df.empty:
        return {"seed_loaded": False, "reason": "empty_seed_dataset"}

    save_stats = insert_matches(seed_df, source_name="seed_csv")
    return {
        "seed_loaded": save_stats["inserted"] > 0,
        "reason": "seed_loaded",
        "inserted": save_stats["inserted"],
        "duplicates": save_stats["duplicates"],
        "seed_path": str(SEED_CSV_PATH),
    }
