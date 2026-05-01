from __future__ import annotations

import os
import sys
import unicodedata
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.config import CLUBELO_RATINGS_URL, SEED_CSV_PATH, TEAM_NAME_ALIASES, TEAM_RATINGS_SEED_PATH


DEFAULT_SOURCE_NAME = "ClubElo"
DEFAULT_RATING_TYPE = "elo"
STRICT_UPDATE_ENV = "CLUBELO_STRICT_UPDATE"
DEFAULT_SERIE_A_TEAMS = [
    "Atalanta",
    "Bologna",
    "Cagliari",
    "Como",
    "Cremonese",
    "Fiorentina",
    "Genoa",
    "Inter",
    "Juventus",
    "Lazio",
    "Lecce",
    "Milan",
    "Napoli",
    "Parma",
    "Pisa",
    "Roma",
    "Sassuolo",
    "Torino",
    "Udinese",
    "Verona",
]

MANUAL_CLUBELO_ALIASES = {
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "juventus turin": "Juventus",
    "as roma": "Roma",
    "ssc napoli": "Napoli",
    "atalanta bc": "Atalanta",
    "bologna fc 1909": "Bologna",
    "genoa cfc": "Genoa",
    "torino fc": "Torino",
    "udinese calcio": "Udinese",
    "hellas verona": "Verona",
    "parma calcio 1913": "Parma",
    "parma calcio": "Parma",
    "cagliari calcio": "Cagliari",
    "us lecce": "Lecce",
    "ss lazio": "Lazio",
    "acf fiorentina": "Fiorentina",
    "us cremonese": "Cremonese",
    "pisa sporting club": "Pisa",
    "sassuolo calcio": "Sassuolo",
}


class SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            cleaned = " ".join(data.split())
            if cleaned:
                self._current_cell.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._current_row.append(" ".join(self._current_cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._in_row = False
        elif tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(text.strip().lower().split())


def download_html(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SerieAAnalystRatingsUpdater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_ranking_table(html: str) -> pd.DataFrame:
    parser = SimpleHTMLTableParser()
    parser.feed(html)

    for table in parser.tables:
        if not table:
            continue
        header = [str(cell).strip() for cell in table[0]]
        if {"Rank", "Club", "Elo"}.issubset(set(header)):
            header_index = {name: idx for idx, name in enumerate(header)}
            rows: list[dict[str, str]] = []
            for raw_row in table[1:]:
                if len(raw_row) < len(header):
                    continue
                club = raw_row[header_index["Club"]].strip()
                elo = raw_row[header_index["Elo"]].strip()
                if not club or not elo:
                    continue
                rows.append(
                    {
                        "club_name": club,
                        "elo": elo,
                    }
                )
            if rows:
                ranking_df = pd.DataFrame(rows)
                ranking_df["rating_value"] = pd.to_numeric(ranking_df["elo"], errors="coerce")
                ranking_df = ranking_df.dropna(subset=["rating_value"]).reset_index(drop=True)
                return ranking_df

    raise ValueError("Impossibile trovare una tabella ranking valida nella pagina ClubElo.")


def load_app_teams_from_seed() -> list[str]:
    if SEED_CSV_PATH.exists():
        seed_df = pd.read_csv(SEED_CSV_PATH)
        if not seed_df.empty and {"home_team", "away_team"}.issubset(seed_df.columns):
            teams = sorted(
                set(seed_df["home_team"].dropna().astype(str).tolist())
                | set(seed_df["away_team"].dropna().astype(str).tolist())
            )
            if teams:
                return teams
    return DEFAULT_SERIE_A_TEAMS.copy()


def build_canonical_team_map(app_teams: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for team in app_teams:
        mapping[normalize_name(team)] = team

    for alias, canonical in TEAM_NAME_ALIASES.items():
        if canonical in app_teams:
            mapping[normalize_name(alias)] = canonical

    for alias, canonical in MANUAL_CLUBELO_ALIASES.items():
        if canonical in app_teams:
            mapping[normalize_name(alias)] = canonical

    return mapping


def build_team_ratings_seed(url: str = CLUBELO_RATINGS_URL) -> tuple[pd.DataFrame, list[str]]:
    html = download_html(url)
    ranking_df = parse_ranking_table(html)
    app_teams = load_app_teams_from_seed()
    canonical_map = build_canonical_team_map(app_teams)

    matched_rows: list[dict[str, object]] = []
    matched_teams: set[str] = set()
    warnings: list[str] = []
    rating_date = date.today().isoformat()

    for row in ranking_df.to_dict(orient="records"):
        canonical_team = canonical_map.get(normalize_name(row["club_name"]))
        if canonical_team is None or canonical_team in matched_teams:
            continue
        matched_rows.append(
            {
                "team_name": canonical_team,
                "rating_date": rating_date,
                "rating_type": DEFAULT_RATING_TYPE,
                "rating_value": float(row["rating_value"]),
                "source_name": DEFAULT_SOURCE_NAME,
                "source_url": url,
            }
        )
        matched_teams.add(canonical_team)

    missing_teams = [team for team in app_teams if team not in matched_teams]
    for team in missing_teams:
        warnings.append(f"Warning: rating non trovato per {team}")

    seed_df = pd.DataFrame(matched_rows, columns=[
        "team_name",
        "rating_date",
        "rating_type",
        "rating_value",
        "source_name",
        "source_url",
    ]).sort_values(["rating_value", "team_name"], ascending=[False, True]).reset_index(drop=True)

    return seed_df, warnings


def _existing_seed_row_count() -> int:
    if not TEAM_RATINGS_SEED_PATH.exists() or TEAM_RATINGS_SEED_PATH.stat().st_size <= 0:
        return 0
    try:
        existing_df = pd.read_csv(TEAM_RATINGS_SEED_PATH)
    except Exception:
        return 0
    if existing_df.empty or "rating_value" not in existing_df.columns:
        return 0
    valid_df = existing_df.dropna(subset=["rating_value"])
    return int(len(valid_df))


def update_team_ratings_seed(url: str = CLUBELO_RATINGS_URL) -> dict[str, object]:
    seed_df, warnings = build_team_ratings_seed(url=url)
    if seed_df.empty:
        raise ValueError("Nessun rating valido trovato: impossibile aggiornare il seed ClubElo.")

    existing_rows = _existing_seed_row_count()
    if existing_rows > 0 and len(seed_df) < existing_rows:
        warnings.append(
            "Warning: nuovo seed ClubElo meno completo del seed esistente; file esistente preservato."
        )
        return {
            "url": url,
            "rows": existing_rows,
            "new_rows": int(len(seed_df)),
            "path": str(TEAM_RATINGS_SEED_PATH),
            "warnings": warnings,
            "updated": False,
            "reason": "incomplete_new_seed",
        }

    TEAM_RATINGS_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(TEAM_RATINGS_SEED_PATH, index=False)
    return {
        "url": url,
        "rows": int(len(seed_df)),
        "new_rows": int(len(seed_df)),
        "path": str(TEAM_RATINGS_SEED_PATH),
        "warnings": warnings,
        "updated": True,
        "reason": "updated",
    }


if __name__ == "__main__":
    strict_update = os.getenv(STRICT_UPDATE_ENV, "").strip() == "1"
    try:
        result = update_team_ratings_seed()
    except Exception as exc:
        fallback_exists = TEAM_RATINGS_SEED_PATH.exists() and TEAM_RATINGS_SEED_PATH.stat().st_size > 0
        if not strict_update and fallback_exists:
            print("Warning: ClubElo update failed, keeping existing team_ratings_seed.csv")
            print(f"Cause: {exc}")
            sys.exit(0)
        if strict_update and fallback_exists:
            print("Error: ClubElo strict update failed; existing team_ratings_seed.csv was not modified.")
            print(f"Cause: {exc}")
            sys.exit(1)
        print("Error: ClubElo update failed and no existing team_ratings_seed.csv fallback is available.")
        print(f"Cause: {exc}")
        sys.exit(1)

    if not result.get("updated") and strict_update:
        print("Error: ClubElo strict update enabled and seed was not updated.")
        print(f"Reason: {result.get('reason')}")
        sys.exit(1)

    if result.get("updated"):
        print(f"Seed rating aggiornato da: {result['url']}")
        print(f"Righe esportate: {result['rows']}")
    else:
        print("Warning: ClubElo update skipped, keeping existing team_ratings_seed.csv")
        print(f"Righe nuovo seed valide: {result.get('new_rows', 0)}")
        print(f"Righe seed esistente preservate: {result['rows']}")
    print(f"Output: {result['path']}")
    for warning in result["warnings"]:
        print(warning)
