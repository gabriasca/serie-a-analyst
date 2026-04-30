from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_LEAGUE_CODE = "ITA_SERIE_A"
FORM_LABELS = {"W": "V", "D": "N", "L": "S"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> str | None:
    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return None
    return date_value.strftime("%Y-%m-%d")


def _result_and_points(goals_for: int, goals_against: int) -> tuple[str, int]:
    if goals_for > goals_against:
        return "W", 3
    if goals_for < goals_against:
        return "L", 0
    return "D", 1


def _prepare_schedule_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    prepared_df = df.copy()
    required_defaults = {
        "season": "",
        "match_date": None,
        "home_team": "",
        "away_team": "",
        "home_goals": 0,
        "away_goals": 0,
        "competition_code": DEFAULT_LEAGUE_CODE,
        "competition_name": "Serie A",
        "competition_type": "league",
        "stage": "",
        "round": "",
        "matchday": None,
    }
    for column, default in required_defaults.items():
        if column not in prepared_df.columns:
            prepared_df[column] = default

    prepared_df["match_date"] = pd.to_datetime(prepared_df["match_date"], errors="coerce")
    prepared_df["home_team"] = prepared_df["home_team"].fillna("").astype(str)
    prepared_df["away_team"] = prepared_df["away_team"].fillna("").astype(str)
    prepared_df["competition_code"] = prepared_df["competition_code"].fillna("").astype(str)
    prepared_df["competition_name"] = prepared_df["competition_name"].fillna("").astype(str)
    prepared_df["competition_type"] = prepared_df["competition_type"].fillna("").astype(str)
    prepared_df["home_goals"] = pd.to_numeric(prepared_df["home_goals"], errors="coerce").fillna(0).astype(int)
    prepared_df["away_goals"] = pd.to_numeric(prepared_df["away_goals"], errors="coerce").fillna(0).astype(int)

    sort_columns = ["match_date"]
    if "id" in prepared_df.columns:
        sort_columns.append("id")
    prepared_df = prepared_df.sort_values(sort_columns, ascending=True).reset_index(drop=True)
    return prepared_df


def _competition_label(row: dict[str, Any]) -> str:
    name = str(row.get("competition_name") or "").strip()
    code = str(row.get("competition_code") or "").strip()
    return name or code or "Competizione"


def _is_league_row(row: dict[str, Any]) -> bool:
    competition_type = str(row.get("competition_type") or "").strip().lower()
    competition_code = str(row.get("competition_code") or "").strip()
    if competition_type:
        return competition_type == "league"
    if competition_code:
        return competition_code == DEFAULT_LEAGUE_CODE
    return True


def _count_recent_matches(log_df: pd.DataFrame, reference_date: pd.Timestamp, days: int) -> int:
    if log_df.empty or pd.isna(reference_date):
        return 0
    start_date = reference_date - pd.Timedelta(days=days)
    window_df = log_df.loc[(log_df["match_date"] >= start_date) & (log_df["match_date"] <= reference_date)]
    return int(len(window_df))


def _form_from_log(log_df: pd.DataFrame, last_n: int = 5) -> dict[str, Any]:
    if log_df.empty:
        return {
            "matches": 0,
            "last_n": int(last_n),
            "form_string": "-",
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "ppm": 0.0,
            "competition_count": 0,
            "competitions": [],
        }

    recent_df = log_df.sort_values("match_date").tail(last_n)
    competitions = sorted({str(value) for value in recent_df["competition_label"].dropna().tolist() if str(value).strip()})
    matches = int(len(recent_df))
    points = int(recent_df["points"].sum())
    goals_for = int(recent_df["goals_for"].sum())
    goals_against = int(recent_df["goals_against"].sum())
    return {
        "matches": matches,
        "last_n": int(last_n),
        "form_string": " ".join(recent_df["display_result"].tolist()) or "-",
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "ppm": round(points / matches, 2) if matches else 0.0,
        "competition_count": int(len(competitions)),
        "competitions": competitions,
    }


def _filter_before_reference(df: pd.DataFrame, reference_date: pd.Timestamp | None, strict_before: bool = False) -> pd.DataFrame:
    if df.empty or reference_date is None or pd.isna(reference_date):
        return df.copy()
    if strict_before:
        return df.loc[df["match_date"] < reference_date].copy()
    return df.loc[df["match_date"] <= reference_date].copy()


def build_schedule_data_audit(df: pd.DataFrame) -> dict[str, Any]:
    prepared_df = _prepare_schedule_df(df)
    if prepared_df.empty:
        return {
            "available": False,
            "competition_count": 0,
            "competitions": [],
            "only_league_data": True,
            "multi_competition_available": False,
            "note": "Nessuna partita disponibile per leggere il contesto calendario.",
        }

    competition_rows = []
    for _, group_df in prepared_df.groupby(["competition_code", "competition_name", "competition_type"], dropna=False):
        first_row = group_df.iloc[0].to_dict()
        competition_rows.append(
            {
                "competition_code": str(first_row.get("competition_code") or ""),
                "competition_name": str(first_row.get("competition_name") or ""),
                "competition_type": str(first_row.get("competition_type") or ""),
                "match_count": int(len(group_df)),
            }
        )

    non_empty_competitions = [
        row
        for row in competition_rows
        if row["competition_code"].strip() or row["competition_name"].strip() or row["competition_type"].strip()
    ]
    competition_count = len(non_empty_competitions) if non_empty_competitions else 1
    non_league_count = sum(1 for row in non_empty_competitions if row["competition_type"].strip().lower() not in {"", "league"})
    only_league_data = competition_count <= 1 and non_league_count == 0

    note = (
        "Contesto calendario basato solo sulle partite disponibili. Dati multi-competizione non ancora presenti."
        if only_league_data
        else "Contesto calendario basato sulle competizioni disponibili nel database."
    )
    return {
        "available": True,
        "competition_count": int(competition_count),
        "competitions": non_empty_competitions,
        "only_league_data": bool(only_league_data),
        "multi_competition_available": bool(not only_league_data),
        "note": note,
    }


def build_team_match_log_all_competitions(df: pd.DataFrame, team: str) -> pd.DataFrame:
    prepared_df = _prepare_schedule_df(df)
    if prepared_df.empty or not team:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in prepared_df.to_dict(orient="records"):
        home_team = str(row.get("home_team") or "")
        away_team = str(row.get("away_team") or "")
        if team not in {home_team, away_team}:
            continue

        is_home = home_team == team
        goals_for = _safe_int(row.get("home_goals") if is_home else row.get("away_goals"))
        goals_against = _safe_int(row.get("away_goals") if is_home else row.get("home_goals"))
        result, points = _result_and_points(goals_for, goals_against)
        records.append(
            {
                "season": row.get("season"),
                "match_date": row.get("match_date"),
                "team": team,
                "opponent": away_team if is_home else home_team,
                "venue": "Casa" if is_home else "Trasferta",
                "home_team": home_team,
                "away_team": away_team,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goals_for - goals_against,
                "result": result,
                "display_result": FORM_LABELS[result],
                "points": points,
                "score": f"{_safe_int(row.get('home_goals'))}-{_safe_int(row.get('away_goals'))}",
                "competition_code": str(row.get("competition_code") or ""),
                "competition_name": str(row.get("competition_name") or ""),
                "competition_type": str(row.get("competition_type") or ""),
                "competition_label": _competition_label(row),
                "stage": row.get("stage"),
                "round": row.get("round"),
                "matchday": row.get("matchday"),
                "is_league": _is_league_row(row),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("match_date").reset_index(drop=True)


def compute_rest_days_for_matches(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _prepare_schedule_df(df)
    if prepared_df.empty:
        return prepared_df

    previous_dates: dict[str, pd.Timestamp] = {}
    rest_home: list[int | None] = []
    rest_away: list[int | None] = []
    for row in prepared_df.to_dict(orient="records"):
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        home_team = str(row.get("home_team") or "")
        away_team = str(row.get("away_team") or "")

        home_previous = previous_dates.get(home_team)
        away_previous = previous_dates.get(away_team)
        rest_home.append(None if home_previous is None or pd.isna(match_date) else int((match_date - home_previous).days))
        rest_away.append(None if away_previous is None or pd.isna(match_date) else int((match_date - away_previous).days))

        if not pd.isna(match_date):
            previous_dates[home_team] = match_date
            previous_dates[away_team] = match_date

    result_df = prepared_df.copy()
    result_df["rest_days_home"] = rest_home
    result_df["rest_days_away"] = rest_away
    return result_df


def classify_schedule_load(
    matches_last_7: int,
    matches_last_14: int,
    matches_last_30: int,
    recent_competitions_count: int = 1,
) -> str:
    if matches_last_7 >= 2 or matches_last_14 >= 4 or matches_last_30 >= 7:
        return "alto"
    if recent_competitions_count >= 2 and matches_last_14 >= 3:
        return "alto"
    if matches_last_7 >= 1 or matches_last_14 >= 2 or matches_last_30 >= 4:
        return "medio"
    return "basso"


def _load_score(load_label: str) -> int:
    return {"basso": 0, "medio": 1, "alto": 2}.get(load_label, 0)


def compute_team_schedule_load(
    df: pd.DataFrame,
    team: str,
    reference_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    prepared_df = _prepare_schedule_df(df)
    if prepared_df.empty:
        return {
            "team": team,
            "available": False,
            "rest_days": None,
            "last_match_date": None,
            "matches_last_7": 0,
            "matches_last_14": 0,
            "matches_last_30": 0,
            "recent_competitions_count": 0,
            "recent_competitions": [],
            "load_label": "n/d",
            "load_score": 0,
            "note": "Nessuna partita disponibile per calcolare il carico calendario.",
        }

    reference = pd.to_datetime(reference_date, errors="coerce") if reference_date is not None else prepared_df["match_date"].max()
    if pd.isna(reference):
        reference = None

    team_log = build_team_match_log_all_competitions(prepared_df, team)
    if team_log.empty:
        return {
            "team": team,
            "available": False,
            "reference_date": _iso_date(reference),
            "rest_days": None,
            "last_match_date": None,
            "matches_last_7": 0,
            "matches_last_14": 0,
            "matches_last_30": 0,
            "recent_competitions_count": 0,
            "recent_competitions": [],
            "load_label": "n/d",
            "load_score": 0,
            "note": "Nessuna partita precedente disponibile per questa squadra.",
        }

    scoped_log = _filter_before_reference(team_log, reference)
    if scoped_log.empty:
        return {
            "team": team,
            "available": False,
            "reference_date": _iso_date(reference),
            "rest_days": None,
            "last_match_date": None,
            "matches_last_7": 0,
            "matches_last_14": 0,
            "matches_last_30": 0,
            "recent_competitions_count": 0,
            "recent_competitions": [],
            "load_label": "n/d",
            "load_score": 0,
            "note": "Nessuna partita precedente alla data di riferimento.",
        }

    last_match_date = scoped_log["match_date"].max()
    rest_days = None if reference is None or pd.isna(last_match_date) else int((reference - last_match_date).days)
    matches_last_7 = _count_recent_matches(scoped_log, reference, 7) if reference is not None else 0
    matches_last_14 = _count_recent_matches(scoped_log, reference, 14) if reference is not None else 0
    matches_last_30 = _count_recent_matches(scoped_log, reference, 30) if reference is not None else 0
    recent_30_df = scoped_log.loc[scoped_log["match_date"] >= reference - pd.Timedelta(days=30)] if reference is not None else scoped_log.tail(5)
    recent_competitions = sorted({str(value) for value in recent_30_df["competition_label"].dropna().tolist() if str(value).strip()})
    load_label = classify_schedule_load(matches_last_7, matches_last_14, matches_last_30, len(recent_competitions))

    return {
        "team": team,
        "available": True,
        "reference_date": _iso_date(reference),
        "rest_days": rest_days,
        "last_match_date": _iso_date(last_match_date),
        "matches_last_7": matches_last_7,
        "matches_last_14": matches_last_14,
        "matches_last_30": matches_last_30,
        "recent_competitions_count": int(len(recent_competitions)),
        "recent_competitions": recent_competitions,
        "load_label": load_label,
        "load_score": _load_score(load_label),
        "note": f"Carico calendario {load_label} sulle partite disponibili.",
    }


def compute_recent_all_competition_form(df: pd.DataFrame, team: str, last_n: int = 5) -> dict[str, Any]:
    log_df = build_team_match_log_all_competitions(df, team)
    return _form_from_log(log_df, last_n=last_n)


def compute_recent_league_only_form(df: pd.DataFrame, team: str, last_n: int = 5) -> dict[str, Any]:
    log_df = build_team_match_log_all_competitions(df, team)
    if log_df.empty:
        return _form_from_log(log_df, last_n=last_n)
    league_df = log_df.loc[log_df["is_league"].fillna(True)].copy()
    return _form_from_log(league_df, last_n=last_n)


def compare_league_vs_all_competition_form(df: pd.DataFrame, team: str) -> dict[str, Any]:
    audit = build_schedule_data_audit(df)
    league_form = compute_recent_league_only_form(df, team, last_n=5)
    all_comp_form = compute_recent_all_competition_form(df, team, last_n=5)
    points_gap = int(all_comp_form.get("points", 0) or 0) - int(league_form.get("points", 0) or 0)
    gd_gap = (
        int(all_comp_form.get("goals_for", 0) or 0)
        - int(all_comp_form.get("goals_against", 0) or 0)
        - int(league_form.get("goals_for", 0) or 0)
        + int(league_form.get("goals_against", 0) or 0)
    )
    note = (
        "Dati multi-competizione non ancora presenti: forma all competitions coincide di fatto con la forma campionato."
        if audit.get("only_league_data", True)
        else "Confronto tra forma campionato e forma su tutte le competizioni disponibili."
    )
    return {
        "team": team,
        "league_form": league_form,
        "all_competition_form": all_comp_form,
        "points_gap": points_gap,
        "goal_difference_gap": gd_gap,
        "multi_competition_available": bool(audit.get("multi_competition_available", False)),
        "note": note,
        "audit": audit,
    }


def build_schedule_context_summary(context: dict[str, Any]) -> str:
    if not context or not context.get("available"):
        return "Contesto calendario non disponibile con i dati correnti."

    home_team = str(context.get("home_team") or "Casa")
    away_team = str(context.get("away_team") or "Trasferta")
    rest_home = context.get("rest_days_home")
    rest_away = context.get("rest_days_away")
    rest_advantage = context.get("rest_advantage")
    home_load = context.get("home_schedule_load", {})
    away_load = context.get("away_schedule_load", {})

    if rest_advantage is None:
        rest_note = "Il vantaggio riposo non e calcolabile in modo affidabile."
    elif rest_advantage > 0:
        rest_note = f"{home_team} ha {rest_advantage} giorni di riposo in piu rispetto a {away_team}."
    elif rest_advantage < 0:
        rest_note = f"{away_team} ha {abs(rest_advantage)} giorni di riposo in piu rispetto a {home_team}."
    else:
        rest_note = "Le due squadre hanno un riposo simile sulle partite disponibili."

    return (
        f"Calendario: {home_team} riposo {rest_home if rest_home is not None else 'n/d'} giorni "
        f"e carico {home_load.get('load_label', 'n/d')}; {away_team} riposo "
        f"{rest_away if rest_away is not None else 'n/d'} giorni e carico "
        f"{away_load.get('load_label', 'n/d')}. {rest_note}"
    )


def build_match_schedule_context(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    prepared_df = _prepare_schedule_df(df)
    audit = build_schedule_data_audit(prepared_df)
    if prepared_df.empty:
        return {
            "available": False,
            "home_team": home_team,
            "away_team": away_team,
            "rest_days_home": None,
            "rest_days_away": None,
            "rest_advantage": None,
            "home_schedule_load": {},
            "away_schedule_load": {},
            "home_recent_all_comp_form": compute_recent_all_competition_form(prepared_df, home_team),
            "away_recent_all_comp_form": compute_recent_all_competition_form(prepared_df, away_team),
            "home_recent_league_form": compute_recent_league_only_form(prepared_df, home_team),
            "away_recent_league_form": compute_recent_league_only_form(prepared_df, away_team),
            "calendar_context_note": audit["note"],
            "warnings": [audit["note"]],
            "competition_audit": audit,
            "schedule_factor_available": False,
            "summary": "Contesto calendario non disponibile con i dati correnti.",
        }

    reference = pd.to_datetime(match_date, errors="coerce") if match_date is not None else prepared_df["match_date"].max()
    strict_before = match_date is not None
    scoped_df = _filter_before_reference(prepared_df, reference, strict_before=strict_before)
    if scoped_df.empty:
        scoped_df = prepared_df.iloc[:0].copy()

    home_load = compute_team_schedule_load(scoped_df, home_team, reference_date=reference)
    away_load = compute_team_schedule_load(scoped_df, away_team, reference_date=reference)
    rest_home = home_load.get("rest_days")
    rest_away = away_load.get("rest_days")
    rest_advantage = None
    if rest_home is not None and rest_away is not None:
        rest_advantage = int(rest_home) - int(rest_away)

    warnings: list[str] = []
    if audit.get("only_league_data", True):
        warnings.append("Contesto calendario basato solo sulle partite disponibili: dati coppe/europee non ancora presenti.")
    if not home_load.get("available") or not away_load.get("available"):
        warnings.append("Storico calendario incompleto per una o entrambe le squadre.")

    context = {
        "available": bool(home_load.get("available") or away_load.get("available")),
        "home_team": home_team,
        "away_team": away_team,
        "reference_date": _iso_date(reference),
        "rest_days_home": rest_home,
        "rest_days_away": rest_away,
        "rest_advantage": rest_advantage,
        "home_schedule_load": home_load,
        "away_schedule_load": away_load,
        "home_recent_all_comp_form": compute_recent_all_competition_form(scoped_df, home_team),
        "away_recent_all_comp_form": compute_recent_all_competition_form(scoped_df, away_team),
        "home_recent_league_form": compute_recent_league_only_form(scoped_df, home_team),
        "away_recent_league_form": compute_recent_league_only_form(scoped_df, away_team),
        "calendar_context_note": audit["note"],
        "warnings": warnings,
        "competition_audit": audit,
        "schedule_factor_available": bool(home_load.get("available") and away_load.get("available")),
        "schedule_factor_weight_modifier": 0.55 if audit.get("only_league_data", True) else 1.0,
    }
    context["summary"] = build_schedule_context_summary(context)
    return context
