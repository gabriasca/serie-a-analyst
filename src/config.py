from __future__ import annotations

import os
from pathlib import Path


APP_TITLE = "Serie A Analyst"
APP_MODE = os.getenv("SERIE_A_ANALYST_MODE", "public").lower()
PUBLIC_DEMO_MODE = APP_MODE != "local"
PUBLIC_DEMO_BANNER = "Versione pubblica demo: dati snapshot, previsioni statistiche non certe."
DEFAULT_COMPETITION_CODE = "ITA_SERIE_A"
DEFAULT_COMPETITION_NAME = "Serie A"
DEFAULT_COMPETITION_TYPE = "league"
DEFAULT_FOOTBALL_DATA_SERIE_A_URL = "https://www.football-data.co.uk/mmz4281/2526/I1.csv"
DEFAULT_CLUBELO_RATINGS_URL = "https://clubelo.com/ITA"
FOOTBALL_DATA_SERIE_A_URL = (
    os.getenv("FOOTBALL_DATA_SERIE_A_URL", DEFAULT_FOOTBALL_DATA_SERIE_A_URL).strip()
    or DEFAULT_FOOTBALL_DATA_SERIE_A_URL
)
CLUBELO_RATINGS_URL = (
    os.getenv("CLUBELO_RATINGS_URL", DEFAULT_CLUBELO_RATINGS_URL).strip()
    or DEFAULT_CLUBELO_RATINGS_URL
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "serie_a.db"
DEMO_CSV_PATH = RAW_DATA_DIR / "serie_a_matches.csv"
SEED_CSV_PATH = RAW_DATA_DIR / "serie_a_seed.csv"
TEAM_RATINGS_SEED_PATH = RAW_DATA_DIR / "team_ratings_seed.csv"

REQUIRED_BASE_COLUMNS = [
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

CANONICAL_COLUMNS = [
    "season",
    "competition_code",
    "competition_name",
    "competition_type",
    "matchday",
    "round",
    "stage",
    "external_match_id",
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
    "source_url",
    "updated_at",
    "xg_home",
    "xg_away",
    "proxy_xg_home",
    "proxy_xg_away",
    "proxy_xg_model_version",
    "elo_home_pre",
    "elo_away_pre",
    "rest_days_home",
    "rest_days_away",
]

INTEGER_COLUMNS = [
    "home_goals",
    "away_goals",
    "matchday",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_cards",
    "away_cards",
]

FLOAT_COLUMNS = [
    "xg_home",
    "xg_away",
    "proxy_xg_home",
    "proxy_xg_away",
    "elo_home_pre",
    "elo_away_pre",
    "rest_days_home",
    "rest_days_away",
]

NUMERIC_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS

COLUMN_ALIASES = {
    "season": "season",
    "stagione": "season",
    "competition_code": "competition_code",
    "competition_name": "competition_name",
    "competition_type": "competition_type",
    "matchday": "matchday",
    "giornata": "matchday",
    "round": "round",
    "stage": "stage",
    "fase": "stage",
    "external_match_id": "external_match_id",
    "match_id": "external_match_id",
    "date": "match_date",
    "match_date": "match_date",
    "data": "match_date",
    "hometeam": "home_team",
    "home_team": "home_team",
    "casa": "home_team",
    "awayteam": "away_team",
    "away_team": "away_team",
    "trasferta": "away_team",
    "fthg": "home_goals",
    "home_goals": "home_goals",
    "goals_home": "home_goals",
    "ftag": "away_goals",
    "away_goals": "away_goals",
    "goals_away": "away_goals",
    "ftr": "full_time_result",
    "result": "full_time_result",
    "full_time_result": "full_time_result",
    "hs": "home_shots",
    "home_shots": "home_shots",
    "as": "away_shots",
    "away_shots": "away_shots",
    "hst": "home_shots_on_target",
    "home_shots_on_target": "home_shots_on_target",
    "ast": "away_shots_on_target",
    "away_shots_on_target": "away_shots_on_target",
    "hc": "home_corners",
    "home_corners": "home_corners",
    "ac": "away_corners",
    "away_corners": "away_corners",
    "hy": "home_cards",
    "home_cards": "home_cards",
    "ay": "away_cards",
    "away_cards": "away_cards",
    "source_name": "source_name",
    "source_url": "source_url",
    "updated_at": "updated_at",
    "last_updated": "updated_at",
    "xg_home": "xg_home",
    "xg_away": "xg_away",
    "proxy_xg_home": "proxy_xg_home",
    "proxy_xg_away": "proxy_xg_away",
    "proxy_xg_model_version": "proxy_xg_model_version",
    "elo_home_pre": "elo_home_pre",
    "elo_away_pre": "elo_away_pre",
    "rest_days_home": "rest_days_home",
    "rest_days_away": "rest_days_away",
}

TEAM_NAME_ALIASES = {
    "internazionale": "Inter",
    "inter milan": "Inter",
    "fc inter": "Inter",
    "ac milan": "Milan",
    "milan fc": "Milan",
    "juventus fc": "Juventus",
    "juve": "Juventus",
    "ssc napoli": "Napoli",
    "as roma": "Roma",
    "atalanta bc": "Atalanta",
}
