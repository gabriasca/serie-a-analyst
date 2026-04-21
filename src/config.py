from __future__ import annotations

from pathlib import Path


APP_TITLE = "Serie A Analyst"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "serie_a.db"
DEMO_CSV_PATH = RAW_DATA_DIR / "serie_a_matches.csv"
SEED_CSV_PATH = RAW_DATA_DIR / "serie_a_seed.csv"

REQUIRED_BASE_COLUMNS = [
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

CANONICAL_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_cards",
    "away_cards",
]

COLUMN_ALIASES = {
    "season": "season",
    "stagione": "season",
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
