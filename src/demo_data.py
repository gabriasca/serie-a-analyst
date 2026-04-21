from __future__ import annotations

import pandas as pd

from src.config import DEMO_CSV_PATH
from src.data_import import clean_match_data
from src.db import database_has_matches, init_db, insert_matches


def get_demo_dataset() -> pd.DataFrame:
    if DEMO_CSV_PATH.exists():
        raw_df = pd.read_csv(DEMO_CSV_PATH)
    else:
        raw_df = pd.DataFrame()

    return clean_match_data(raw_df, default_season="2024-2025", source_name="demo_dataset")


def load_demo_data(force: bool = False) -> dict[str, int | str | bool]:
    init_db()
    if database_has_matches() and not force:
        return {
            "loaded": False,
            "inserted": 0,
            "duplicates": 0,
            "reason": "database_not_empty",
        }

    demo_df = get_demo_dataset()
    save_stats = insert_matches(demo_df, source_name="demo_dataset")
    return {
        "loaded": save_stats["inserted"] > 0,
        "inserted": save_stats["inserted"],
        "duplicates": save_stats["duplicates"],
        "reason": "loaded",
    }


def ensure_demo_data_loaded() -> dict[str, int | str | bool]:
    init_db()
    if database_has_matches():
        return {
            "loaded": False,
            "inserted": 0,
            "duplicates": 0,
            "reason": "database_not_empty",
        }

    return load_demo_data(force=False)
