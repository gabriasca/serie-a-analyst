from __future__ import annotations

import io
import re
import sys
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.config import (
    DEFAULT_COMPETITION_CODE,
    DEFAULT_COMPETITION_NAME,
    DEFAULT_COMPETITION_TYPE,
    FOOTBALL_DATA_SERIE_A_URL,
    SEED_CSV_PATH,
)
from src.data_import import clean_match_data, load_csv_to_dataframe, validate_required_columns


DEFAULT_SOURCE_NAME = "football-data.co.uk"


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
        headers={"User-Agent": "SerieAAnalystDataUpdater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read().decode("utf-8-sig")

    return load_csv_to_dataframe(io.StringIO(content))


def update_seed_csv(url: str = FOOTBALL_DATA_SERIE_A_URL) -> dict[str, str | int]:
    raw_df = download_csv_dataframe(url)
    inferred_season = infer_season_from_url(url)

    validation = validate_required_columns(raw_df, provided_season=inferred_season)
    if not validation["valid"]:
        missing = ", ".join(validation["missing_columns"])
        raise ValueError(f"CSV non valido. Colonne obbligatorie mancanti: {missing}")

    cleaned_df = clean_match_data(
        raw_df,
        default_season=inferred_season,
        source_name=DEFAULT_SOURCE_NAME,
        source_url=url,
        default_competition_code=DEFAULT_COMPETITION_CODE,
        default_competition_name=DEFAULT_COMPETITION_NAME,
        default_competition_type=DEFAULT_COMPETITION_TYPE,
    )

    if cleaned_df.empty:
        raise ValueError("Il CSV scaricato non contiene righe valide dopo la pulizia.")

    SEED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(SEED_CSV_PATH, index=False)

    return {
        "url": url,
        "rows": len(cleaned_df),
        "season": ", ".join(sorted(cleaned_df["season"].dropna().astype(str).unique().tolist())),
        "competition": ", ".join(sorted(cleaned_df["competition_code"].dropna().astype(str).unique().tolist())),
        "path": str(SEED_CSV_PATH),
    }


if __name__ == "__main__":
    result = update_seed_csv()
    print(f"Seed aggiornato da: {result['url']}")
    print(f"Stagione rilevata: {result['season']}")
    print(f"Competizione rilevata: {result['competition']}")
    print(f"Righe esportate: {result['rows']}")
    print(f"Output: {result['path']}")
