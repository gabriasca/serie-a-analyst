from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_freshness import build_data_freshness_report  # noqa: E402
from src.db import fetch_matches  # noqa: E402
from src.seed_data import bootstrap_database  # noqa: E402


def _print_latest_matches(latest_matches: object) -> None:
    if not isinstance(latest_matches, pd.DataFrame) or latest_matches.empty:
        print("Ultimi 10 match: nessun match disponibile")
        return

    print("Ultimi 10 match:")
    print(latest_matches.to_string(index=False))


def main() -> None:
    bootstrap_database()
    report = build_data_freshness_report(fetch_matches())

    print("Stato aggiornamento dati")
    print(f"Status: {report['freshness_status']}")
    print(f"Messaggio: {report['freshness_message']}")
    print(f"Partite caricate: {report['match_count']}")
    print(f"Squadre: {report['team_count']}")
    print(f"Stagioni: {report['season_count']} ({', '.join(report['seasons']) if report['seasons'] else 'nessuna'})")
    print(f"Ultima data partita: {report['latest_match_date'] or 'n/d'}")
    print(f"Partite teoriche totali: {report['expected_total_matches']}")
    print(f"Partite mancanti stimate: {report['missing_matches_estimate']}")
    print(f"Fonti dati: {', '.join(report['source_names']) if report['source_names'] else 'seed_csv'}")
    print()
    _print_latest_matches(report["latest_matches"])


if __name__ == "__main__":
    main()
