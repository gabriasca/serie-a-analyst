from __future__ import annotations

import random
from typing import Any

import pandas as pd

from src.analytics import build_standings, get_teams, prepare_matches_dataframe
from src.predictor import poisson_probability_matrix, predict_match


DEFAULT_HOME_XG = 1.35
DEFAULT_AWAY_XG = 1.15


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def expected_total_matches(team_count: int) -> int:
    return team_count * max(team_count - 1, 0)


def infer_remaining_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = prepare_matches_dataframe(df)
    teams = get_teams(prepared_df)

    if not teams:
        return pd.DataFrame(columns=["home_team", "away_team"])

    played_fixtures = {
        (str(row.home_team), str(row.away_team))
        for row in prepared_df.itertuples(index=False)
    }

    remaining_rows = []
    for home_team in teams:
        for away_team in teams:
            if home_team == away_team:
                continue
            if (home_team, away_team) in played_fixtures:
                continue
            remaining_rows.append({"home_team": home_team, "away_team": away_team})

    return pd.DataFrame(remaining_rows, columns=["home_team", "away_team"])


def _fallback_expected_goals(df: pd.DataFrame) -> tuple[float, float]:
    prepared_df = prepare_matches_dataframe(df)
    if prepared_df.empty:
        return DEFAULT_HOME_XG, DEFAULT_AWAY_XG

    avg_home_goals = float(prepared_df["home_goals"].mean())
    avg_away_goals = float(prepared_df["away_goals"].mean())

    if avg_home_goals <= 0:
        avg_home_goals = DEFAULT_HOME_XG
    if avg_away_goals <= 0:
        avg_away_goals = DEFAULT_AWAY_XG

    return round(_clamp(avg_home_goals, 0.2, 3.5), 3), round(_clamp(avg_away_goals, 0.2, 3.5), 3)


def _flatten_score_matrix(score_matrix: pd.DataFrame) -> tuple[list[tuple[int, int]], list[float]]:
    scorelines: list[tuple[int, int]] = []
    probabilities: list[float] = []

    for home_goals in score_matrix.index:
        for away_goals in score_matrix.columns:
            scorelines.append((int(home_goals), int(away_goals)))
            probabilities.append(float(score_matrix.loc[home_goals, away_goals]))

    return scorelines, probabilities


def _extract_outcome_probabilities(score_matrix: pd.DataFrame) -> tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    for home_goals in score_matrix.index:
        for away_goals in score_matrix.columns:
            probability = float(score_matrix.loc[home_goals, away_goals])
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

    return home_win, draw, away_win


def build_fixture_models(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    max_goals: int = 6,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    fixture_models: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []

    fallback_home_xg, fallback_away_xg = _fallback_expected_goals(df)

    for row in fixtures_df.itertuples(index=False):
        prediction = predict_match(df, row.home_team, row.away_team, max_goals=max_goals)
        method = "predictor"
        warning = ""

        if prediction["ok"]:
            score_matrix = prediction["score_matrix"]
            expected_goals_home = float(prediction["expected_goals_home"])
            expected_goals_away = float(prediction["expected_goals_away"])
        else:
            method = "fallback_media_campionato"
            warning = prediction.get("message", "Fallback sulla media gol campionato.")
            expected_goals_home = fallback_home_xg
            expected_goals_away = fallback_away_xg
            score_matrix = poisson_probability_matrix(
                expected_goals_home,
                expected_goals_away,
                max_goals=max_goals,
            )

        scorelines, probabilities = _flatten_score_matrix(score_matrix)
        prob_1, prob_x, prob_2 = _extract_outcome_probabilities(score_matrix)

        fixture_models.append(
            {
                "home_team": row.home_team,
                "away_team": row.away_team,
                "expected_goals_home": expected_goals_home,
                "expected_goals_away": expected_goals_away,
                "scorelines": scorelines,
                "probabilities": probabilities,
                "method": method,
                "warning": warning,
            }
        )
        fixture_rows.append(
            {
                "Casa": row.home_team,
                "Trasferta": row.away_team,
                "xG casa": round(expected_goals_home, 2),
                "xG trasferta": round(expected_goals_away, 2),
                "Prob. 1": prob_1,
                "Prob. X": prob_x,
                "Prob. 2": prob_2,
                "Metodo": method,
                "Nota": warning or "-",
            }
        )

    fixture_table = pd.DataFrame(fixture_rows)
    return fixture_models, fixture_table


def _build_table_state(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    prepared_df = prepare_matches_dataframe(df)
    state = {
        team: {"Team": team, "GP": 0, "V": 0, "N": 0, "S": 0, "GF": 0, "GA": 0, "Pts": 0}
        for team in get_teams(prepared_df)
    }

    for row in prepared_df.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        _apply_match_result(state, home_team, away_team, home_goals, away_goals)

    return state


def _apply_match_result(
    state: dict[str, dict[str, Any]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    for team in [home_team, away_team]:
        if team not in state:
            state[team] = {"Team": team, "GP": 0, "V": 0, "N": 0, "S": 0, "GF": 0, "GA": 0, "Pts": 0}

    state[home_team]["GP"] += 1
    state[away_team]["GP"] += 1

    state[home_team]["GF"] += home_goals
    state[home_team]["GA"] += away_goals
    state[away_team]["GF"] += away_goals
    state[away_team]["GA"] += home_goals

    if home_goals > away_goals:
        state[home_team]["V"] += 1
        state[away_team]["S"] += 1
        state[home_team]["Pts"] += 3
    elif home_goals < away_goals:
        state[away_team]["V"] += 1
        state[home_team]["S"] += 1
        state[away_team]["Pts"] += 3
    else:
        state[home_team]["N"] += 1
        state[away_team]["N"] += 1
        state[home_team]["Pts"] += 1
        state[away_team]["Pts"] += 1


def _state_to_table(state: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not state:
        return pd.DataFrame()

    table = pd.DataFrame(state.values()).copy()
    table["DR"] = table["GF"] - table["GA"]
    table = table.sort_values(
        by=["Pts", "DR", "GF", "Team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "Pos", range(1, len(table) + 1))
    return table


def simulate_single_match(fixture_model: dict[str, Any], rng: random.Random) -> tuple[int, int]:
    sampled_score = rng.choices(
        fixture_model["scorelines"],
        weights=fixture_model["probabilities"],
        k=1,
    )[0]
    return int(sampled_score[0]), int(sampled_score[1])


def _deterministic_projection(current_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if current_table.empty:
        return pd.DataFrame(), {}

    team_count = len(current_table)
    relegated_slots = min(3, max(team_count - 1, 0))
    safe_cutoff = max(team_count - relegated_slots, 0)

    summary_rows = []
    position_distributions: dict[str, pd.DataFrame] = {}

    for row in current_table.itertuples(index=False):
        top4 = 1.0 if row.Pos <= min(4, team_count) else 0.0
        top6 = 1.0 if row.Pos <= min(6, team_count) else 0.0
        relegation = 1.0 if row.Pos > safe_cutoff and relegated_slots > 0 else 0.0
        safety = 1.0 - relegation

        summary_rows.append(
            {
                "Posizione media": float(row.Pos),
                "Squadra": row.Team,
                "Punti attuali": int(row.Pts),
                "Punti medi finali": float(row.Pts),
                "Differenza reti media finale": float(row.DR),
                "Prob. scudetto": 1.0 if row.Pos == 1 else 0.0,
                "Prob. top 4": top4,
                "Prob. top 6": top6,
                "Prob. salvezza": safety,
                "Prob. retrocessione": relegation,
                "Migliore posizione": int(row.Pos),
                "Peggiore posizione": int(row.Pos),
            }
        )
        position_distributions[row.Team] = pd.DataFrame(
            [{"Posizione": int(row.Pos), "Probabilita": 1.0}]
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Posizione media", "Punti medi finali", "Squadra"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return summary_df, position_distributions


def summarize_projection_results(
    current_table: pd.DataFrame,
    simulation_tables: list[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if current_table.empty:
        return pd.DataFrame(), {}

    team_count = len(current_table)
    current_points = {row.Team: int(row.Pts) for row in current_table.itertuples(index=False)}
    relegated_slots = min(3, max(team_count - 1, 0))
    safe_cutoff = max(team_count - relegated_slots, 0)

    summary_rows = []
    position_distributions: dict[str, pd.DataFrame] = {}

    for team in current_points:
        positions: list[int] = []
        final_points: list[int] = []
        final_goal_diff: list[int] = []

        for table in simulation_tables:
            team_row = table.loc[table["Team"] == team].iloc[0]
            positions.append(int(team_row["Pos"]))
            final_points.append(int(team_row["Pts"]))
            final_goal_diff.append(int(team_row["DR"]))

        distribution_series = pd.Series(positions).value_counts(normalize=True).sort_index()
        position_distributions[team] = pd.DataFrame(
            {
                "Posizione": distribution_series.index.astype(int),
                "Probabilita": distribution_series.values,
            }
        )

        summary_rows.append(
            {
                "Posizione media": round(sum(positions) / len(positions), 2),
                "Squadra": team,
                "Punti attuali": current_points[team],
                "Punti medi finali": round(sum(final_points) / len(final_points), 2),
                "Differenza reti media finale": round(sum(final_goal_diff) / len(final_goal_diff), 2),
                "Prob. scudetto": distribution_series.get(1, 0.0),
                "Prob. top 4": float(
                    sum(prob for pos, prob in distribution_series.items() if pos <= min(4, team_count))
                ),
                "Prob. top 6": float(
                    sum(prob for pos, prob in distribution_series.items() if pos <= min(6, team_count))
                ),
                "Prob. salvezza": float(
                    sum(prob for pos, prob in distribution_series.items() if pos <= safe_cutoff)
                ) if safe_cutoff > 0 else 0.0,
                "Prob. retrocessione": float(
                    sum(prob for pos, prob in distribution_series.items() if pos > safe_cutoff)
                ) if relegated_slots > 0 else 0.0,
                "Migliore posizione": min(positions),
                "Peggiore posizione": max(positions),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Posizione media", "Punti medi finali", "Squadra"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return summary_df, position_distributions


def run_projection_simulations(
    df: pd.DataFrame,
    n_simulations: int,
    max_goals: int = 6,
    seed: int | None = None,
) -> dict[str, Any]:
    prepared_df = prepare_matches_dataframe(df)
    teams = get_teams(prepared_df)

    if prepared_df.empty:
        return {"ok": False, "message": "Nessun dato disponibile per la stagione selezionata."}

    if len(teams) < 2:
        return {"ok": False, "message": "Servono almeno due squadre per costruire una proiezione."}

    current_table = _state_to_table(_build_table_state(prepared_df))
    current_display = build_standings(prepared_df)
    remaining_fixtures = infer_remaining_fixtures(prepared_df)
    expected_matches = expected_total_matches(len(teams))

    fixture_models, fixture_table = build_fixture_models(prepared_df, remaining_fixtures, max_goals=max_goals)
    fallback_count = int((fixture_table["Metodo"] == "fallback_media_campionato").sum()) if not fixture_table.empty else 0

    if remaining_fixtures.empty:
        summary_df, position_distributions = _deterministic_projection(current_table)
        return {
            "ok": True,
            "teams": teams,
            "team_count": len(teams),
            "played_matches": len(prepared_df),
            "expected_matches": expected_matches,
            "missing_matches": 0,
            "current_table": current_display,
            "remaining_fixtures": remaining_fixtures,
            "fixture_table": fixture_table,
            "summary_table": summary_df,
            "position_distributions": position_distributions,
            "fallback_count": fallback_count,
            "complete_season": True,
            "n_simulations": n_simulations,
        }

    rng = random.Random(seed)
    base_state = _build_table_state(prepared_df)
    simulation_tables: list[pd.DataFrame] = []

    for _ in range(n_simulations):
        simulation_state = {team: values.copy() for team, values in base_state.items()}

        for fixture_model in fixture_models:
            home_goals, away_goals = simulate_single_match(fixture_model, rng)
            _apply_match_result(
                simulation_state,
                fixture_model["home_team"],
                fixture_model["away_team"],
                home_goals,
                away_goals,
            )

        simulation_tables.append(_state_to_table(simulation_state))

    summary_df, position_distributions = summarize_projection_results(current_table, simulation_tables)

    return {
        "ok": True,
        "teams": teams,
        "team_count": len(teams),
        "played_matches": len(prepared_df),
        "expected_matches": expected_matches,
        "missing_matches": len(remaining_fixtures),
        "current_table": current_display,
        "remaining_fixtures": remaining_fixtures,
        "fixture_table": fixture_table,
        "summary_table": summary_df,
        "position_distributions": position_distributions,
        "fallback_count": fallback_count,
        "complete_season": False,
        "n_simulations": n_simulations,
    }
