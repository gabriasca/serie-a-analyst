from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.analytics import compute_recent_form, get_team_match_log, prepare_matches_dataframe


PRIOR_MATCHES = 5.0
MIN_MATCHES_PER_TEAM = 3
MIN_LEAGUE_MATCHES = 6


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _smoothed_rate(total: float, matches: int, baseline_rate: float, prior_matches: float = PRIOR_MATCHES) -> float:
    return (total + baseline_rate * prior_matches) / max(matches + prior_matches, 1.0)


def compute_home_advantage(df: pd.DataFrame) -> float:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return 1.0

    avg_home_goals = float(prepared_df["home_goals"].mean())
    avg_away_goals = float(prepared_df["away_goals"].mean())
    if avg_away_goals <= 0:
        return 1.0

    return _clamp(avg_home_goals / avg_away_goals, 0.9, 1.35)


def compute_form_factor(df: pd.DataFrame, team: str, last_n: int = 5) -> dict[str, float | int | str]:
    recent_form = compute_recent_form(df, team, last_n=last_n)
    matches = int(recent_form["matches"])
    if matches == 0:
        return {"factor": 1.0, "points": 0, "matches": 0, "form_string": "-"}

    points = int(recent_form["points"])
    points_per_game = points / matches
    factor = 1.0 + ((points_per_game - 1.4) * 0.08)
    return {
        "factor": _clamp(factor, 0.85, 1.15),
        "points": points,
        "matches": matches,
        "form_string": recent_form["form_string"],
    }


def estimate_team_strengths(df: pd.DataFrame) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return {"league": {}, "teams": pd.DataFrame()}

    league_avg_goals = float(
        (prepared_df["home_goals"].sum() + prepared_df["away_goals"].sum()) / (len(prepared_df) * 2)
    )
    home_advantage = compute_home_advantage(prepared_df)

    team_rows = []
    teams = sorted(
        set(prepared_df["home_team"].dropna().tolist()) | set(prepared_df["away_team"].dropna().tolist())
    )

    for team in teams:
        match_log = get_team_match_log(prepared_df, team)
        matches = len(match_log)
        goals_for = int(match_log["goals_for"].sum()) if matches else 0
        goals_against = int(match_log["goals_against"].sum()) if matches else 0

        smoothed_for = _smoothed_rate(goals_for, matches, league_avg_goals)
        smoothed_against = _smoothed_rate(goals_against, matches, league_avg_goals)
        form = compute_form_factor(prepared_df, team, last_n=5)

        team_rows.append(
            {
                "team": team,
                "matches": matches,
                "goals_for_total": goals_for,
                "goals_against_total": goals_against,
                "goals_for_per_match": round(goals_for / matches, 2) if matches else 0.0,
                "goals_against_per_match": round(goals_against / matches, 2) if matches else 0.0,
                "attack_strength": round(smoothed_for / league_avg_goals, 3) if league_avg_goals else 1.0,
                "defense_strength": round(smoothed_against / league_avg_goals, 3) if league_avg_goals else 1.0,
                "form_factor": round(float(form["factor"]), 3),
                "form_points_last_5": int(form["points"]),
                "form_string": str(form["form_string"]),
            }
        )

    team_df = pd.DataFrame(team_rows).set_index("team")
    return {
        "league": {
            "match_count": len(prepared_df),
            "avg_goals_per_team": round(league_avg_goals, 3),
            "avg_home_goals": round(float(prepared_df["home_goals"].mean()), 3),
            "avg_away_goals": round(float(prepared_df["away_goals"].mean()), 3),
            "home_advantage": round(home_advantage, 3),
        },
        "teams": team_df,
    }


def poisson_probability_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 6,
) -> pd.DataFrame:
    rows = []
    for home_goals in range(max_goals + 1):
        row = []
        for away_goals in range(max_goals + 1):
            home_prob = math.exp(-lambda_home) * (lambda_home**home_goals) / math.factorial(home_goals)
            away_prob = math.exp(-lambda_away) * (lambda_away**away_goals) / math.factorial(away_goals)
            row.append(home_prob * away_prob)
        rows.append(row)

    matrix = pd.DataFrame(rows, index=range(max_goals + 1), columns=range(max_goals + 1))
    total_probability = float(matrix.to_numpy().sum())
    if total_probability > 0:
        matrix = matrix / total_probability

    return matrix


def match_outcome_probabilities(score_matrix: pd.DataFrame) -> dict[str, Any]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    top_scores = []

    for home_goals in score_matrix.index:
        for away_goals in score_matrix.columns:
            probability = float(score_matrix.loc[home_goals, away_goals])
            top_scores.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": probability,
                }
            )
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

    top_scores = sorted(top_scores, key=lambda item: item["probability"], reverse=True)[:5]

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "top_scores": top_scores,
        "most_likely_score": top_scores[0]["score"] if top_scores else None,
    }


def expected_goals(df: pd.DataFrame, home_team: str, away_team: str) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    strengths = estimate_team_strengths(prepared_df)
    league = strengths["league"]
    teams_df = strengths["teams"]

    if prepared_df.empty or len(prepared_df) < MIN_LEAGUE_MATCHES:
        return {
            "ok": False,
            "message": "Dati insufficienti: servono almeno alcune partite di campionato per stimare il match.",
        }

    if home_team not in teams_df.index or away_team not in teams_df.index:
        return {
            "ok": False,
            "message": "Una o entrambe le squadre selezionate non sono presenti nel dataset.",
        }

    home_metrics = teams_df.loc[home_team]
    away_metrics = teams_df.loc[away_team]

    if int(home_metrics["matches"]) < MIN_MATCHES_PER_TEAM or int(away_metrics["matches"]) < MIN_MATCHES_PER_TEAM:
        return {
            "ok": False,
            "message": "Dati insufficienti: servono almeno 3 partite per ciascuna squadra.",
        }

    base_rate = float(league["avg_goals_per_team"])
    home_advantage = float(league["home_advantage"])

    lambda_home = (
        base_rate
        * home_advantage
        * float(home_metrics["attack_strength"])
        * float(away_metrics["defense_strength"])
        * float(home_metrics["form_factor"])
    )
    lambda_away = (
        base_rate
        * (1 / home_advantage)
        * float(away_metrics["attack_strength"])
        * float(home_metrics["defense_strength"])
        * float(away_metrics["form_factor"])
    )

    lambda_home = round(_clamp(lambda_home, 0.2, 3.5), 3)
    lambda_away = round(_clamp(lambda_away, 0.2, 3.5), 3)

    return {
        "ok": True,
        "expected_goals_home": lambda_home,
        "expected_goals_away": lambda_away,
        "league": league,
        "home_team_metrics": home_metrics.to_dict(),
        "away_team_metrics": away_metrics.to_dict(),
    }


def predict_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int = 6,
) -> dict[str, Any]:
    xg_result = expected_goals(df, home_team, away_team)
    if not xg_result["ok"]:
        return xg_result

    score_matrix = poisson_probability_matrix(
        xg_result["expected_goals_home"],
        xg_result["expected_goals_away"],
        max_goals=max_goals,
    )
    outcome = match_outcome_probabilities(score_matrix)

    return {
        "ok": True,
        "home_team": home_team,
        "away_team": away_team,
        "expected_goals_home": xg_result["expected_goals_home"],
        "expected_goals_away": xg_result["expected_goals_away"],
        "probabilities": {
            "1": outcome["home_win"],
            "X": outcome["draw"],
            "2": outcome["away_win"],
        },
        "most_likely_score": outcome["most_likely_score"],
        "top_scorelines": outcome["top_scores"],
        "score_matrix": score_matrix,
        "factors": {
            "league": xg_result["league"],
            "home_team": xg_result["home_team_metrics"],
            "away_team": xg_result["away_team_metrics"],
        },
    }
