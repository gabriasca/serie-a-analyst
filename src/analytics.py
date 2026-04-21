from __future__ import annotations

from typing import Any

import pandas as pd


RESULT_LABELS = {"W": "V", "D": "N", "L": "S"}


def prepare_matches_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    prepared_df = df.copy()
    prepared_df["match_date"] = pd.to_datetime(prepared_df["match_date"], errors="coerce")
    prepared_df = prepared_df.sort_values(["match_date", "id"], ascending=[True, True])
    prepared_df = prepared_df.reset_index(drop=True)
    return prepared_df


def get_teams(df: pd.DataFrame) -> list[str]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return []

    return sorted(
        set(prepared_df["home_team"].dropna().tolist()) | set(prepared_df["away_team"].dropna().tolist())
    )


def _result_and_points(goals_for: int, goals_against: int) -> tuple[str, int]:
    if goals_for > goals_against:
        return "W", 3
    if goals_for < goals_against:
        return "L", 0
    return "D", 1


def get_team_match_log(df: pd.DataFrame, team: str) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    records: list[dict[str, Any]] = []

    for row in prepared_df.itertuples(index=False):
        if row.home_team == team:
            result, points = _result_and_points(int(row.home_goals), int(row.away_goals))
            records.append(
                {
                    "match_date": row.match_date,
                    "team": team,
                    "opponent": row.away_team,
                    "venue": "Casa",
                    "goals_for": int(row.home_goals),
                    "goals_against": int(row.away_goals),
                    "result": result,
                    "display_result": RESULT_LABELS[result],
                    "points": points,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "score": f"{int(row.home_goals)}-{int(row.away_goals)}",
                }
            )
        elif row.away_team == team:
            result, points = _result_and_points(int(row.away_goals), int(row.home_goals))
            records.append(
                {
                    "match_date": row.match_date,
                    "team": team,
                    "opponent": row.home_team,
                    "venue": "Trasferta",
                    "goals_for": int(row.away_goals),
                    "goals_against": int(row.home_goals),
                    "result": result,
                    "display_result": RESULT_LABELS[result],
                    "points": points,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "score": f"{int(row.home_goals)}-{int(row.away_goals)}",
                }
            )

    if not records:
        return pd.DataFrame()

    match_log = pd.DataFrame(records).sort_values("match_date").reset_index(drop=True)
    return match_log


def compute_recent_form(
    df: pd.DataFrame,
    team: str | None = None,
    last_n: int = 5,
) -> dict[str, Any] | pd.DataFrame:
    if team is not None:
        match_log = get_team_match_log(df, team)
        if match_log.empty:
            return {"form_string": "-", "points": 0, "matches": 0}

        recent = match_log.tail(last_n)
        return {
            "form_string": " ".join(recent["display_result"].tolist()) or "-",
            "points": int(recent["points"].sum()),
            "matches": int(len(recent)),
        }

    rows = []
    for team_name in get_teams(df):
        form = compute_recent_form(df, team_name, last_n=last_n)
        rows.append(
            {
                "Team": team_name,
                "Forma ultime 5": form["form_string"],
                "Punti ultime 5": form["points"],
            }
        )

    return pd.DataFrame(rows)


def build_standings(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in prepared_df.itertuples(index=False):
        home_result, home_points = _result_and_points(int(row.home_goals), int(row.away_goals))
        away_result, away_points = _result_and_points(int(row.away_goals), int(row.home_goals))

        rows.append(
            {
                "Team": row.home_team,
                "GP": 1,
                "V": 1 if home_result == "W" else 0,
                "N": 1 if home_result == "D" else 0,
                "S": 1 if home_result == "L" else 0,
                "GF": int(row.home_goals),
                "GA": int(row.away_goals),
                "Pts": home_points,
            }
        )
        rows.append(
            {
                "Team": row.away_team,
                "GP": 1,
                "V": 1 if away_result == "W" else 0,
                "N": 1 if away_result == "D" else 0,
                "S": 1 if away_result == "L" else 0,
                "GF": int(row.away_goals),
                "GA": int(row.home_goals),
                "Pts": away_points,
            }
        )

    standings = pd.DataFrame(rows).groupby("Team", as_index=False).sum()
    standings["DR"] = standings["GF"] - standings["GA"]
    standings["Forma ultime 5"] = standings["Team"].apply(
        lambda team_name: compute_recent_form(prepared_df, team_name, last_n=5)["form_string"]
    )
    standings = standings.sort_values(
        by=["Pts", "DR", "GF", "Team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    standings.index = standings.index + 1
    standings.index.name = "Pos"
    return standings


def compute_home_away_split(df: pd.DataFrame, team: str) -> pd.DataFrame:
    match_log = get_team_match_log(df, team)
    if match_log.empty:
        return pd.DataFrame()

    rows = []
    for venue in ["Casa", "Trasferta"]:
        venue_log = match_log[match_log["venue"] == venue]
        if venue_log.empty:
            continue

        wins = int((venue_log["result"] == "W").sum())
        draws = int((venue_log["result"] == "D").sum())
        losses = int((venue_log["result"] == "L").sum())
        games = int(len(venue_log))
        goals_for = int(venue_log["goals_for"].sum())
        goals_against = int(venue_log["goals_against"].sum())
        points = int(venue_log["points"].sum())

        rows.append(
            {
                "Contesto": venue,
                "GP": games,
                "V": wins,
                "N": draws,
                "S": losses,
                "GF": goals_for,
                "GA": goals_against,
                "Pts": points,
                "PPM": round(points / games, 2) if games else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_home_away_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in get_teams(df):
        split = compute_home_away_split(df, team)
        if split.empty:
            continue

        home_row = split[split["Contesto"] == "Casa"]
        away_row = split[split["Contesto"] == "Trasferta"]

        rows.append(
            {
                "Team": team,
                "Punti Casa": int(home_row["Pts"].iloc[0]) if not home_row.empty else 0,
                "Punti Trasferta": int(away_row["Pts"].iloc[0]) if not away_row.empty else 0,
                "PPM Casa": float(home_row["PPM"].iloc[0]) if not home_row.empty else 0.0,
                "PPM Trasferta": float(away_row["PPM"].iloc[0]) if not away_row.empty else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["Punti Casa", "Punti Trasferta", "Team"],
        ascending=[False, False, True],
    )


def get_recent_matches(df: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    match_log = get_team_match_log(df, team)
    if match_log.empty:
        return pd.DataFrame()

    recent = match_log.tail(n).copy()
    recent["Data"] = recent["match_date"].dt.strftime("%Y-%m-%d")
    recent["Partita"] = recent["home_team"] + " " + recent["score"] + " " + recent["away_team"]
    recent["Esito"] = recent["display_result"]
    return recent[["Data", "Partita", "venue", "Esito", "points"]].rename(
        columns={"venue": "Contesto", "points": "Punti"}
    )


def get_points_progression(df: pd.DataFrame, team: str) -> pd.DataFrame:
    match_log = get_team_match_log(df, team)
    if match_log.empty:
        return pd.DataFrame()

    progression = match_log.copy()
    progression["match_number"] = range(1, len(progression) + 1)
    progression["cumulative_points"] = progression["points"].cumsum()
    progression["date_label"] = progression["match_date"].dt.strftime("%Y-%m-%d")
    return progression[
        [
            "match_number",
            "date_label",
            "points",
            "cumulative_points",
            "display_result",
            "opponent",
            "venue",
        ]
    ]


def compute_team_stats(df: pd.DataFrame, team: str) -> dict[str, Any]:
    match_log = get_team_match_log(df, team)
    if match_log.empty:
        return {}

    wins = int((match_log["result"] == "W").sum())
    draws = int((match_log["result"] == "D").sum())
    losses = int((match_log["result"] == "L").sum())
    goals_for = int(match_log["goals_for"].sum())
    goals_against = int(match_log["goals_against"].sum())
    games = int(len(match_log))
    points = int(match_log["points"].sum())
    recent_form = compute_recent_form(df, team, last_n=5)

    return {
        "team": team,
        "matches_played": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "points": points,
        "avg_goals_for": round(goals_for / games, 2) if games else 0.0,
        "avg_goals_against": round(goals_against / games, 2) if games else 0.0,
        "form_last_5": recent_form["form_string"],
        "form_points_last_5": recent_form["points"],
        "home_away_split": compute_home_away_split(df, team),
        "recent_matches": get_recent_matches(df, team, n=5),
        "points_progression": get_points_progression(df, team),
    }


def compare_teams(df: pd.DataFrame, team_a: str, team_b: str) -> dict[str, Any]:
    stats_a = compute_team_stats(df, team_a)
    stats_b = compute_team_stats(df, team_b)

    return {
        "team_a": stats_a,
        "team_b": stats_b,
        "teams": [team_a, team_b],
    }


def build_comparison_summary(comparison: dict[str, Any]) -> str:
    team_a = comparison["team_a"]
    team_b = comparison["team_b"]

    if not team_a or not team_b:
        return "Dati insufficienti per costruire una sintesi del confronto."

    attack_leader = (
        team_a["team"] if team_a["avg_goals_for"] >= team_b["avg_goals_for"] else team_b["team"]
    )
    defense_leader = (
        team_a["team"]
        if team_a["avg_goals_against"] <= team_b["avg_goals_against"]
        else team_b["team"]
    )
    form_leader = (
        team_a["team"]
        if team_a["form_points_last_5"] >= team_b["form_points_last_5"]
        else team_b["team"]
    )

    return (
        f"{attack_leader} arriva con l'attacco medio piu produttivo, "
        f"{defense_leader} concede meno gol in media, "
        f"mentre la forma recente favorisce {form_leader}."
    )
