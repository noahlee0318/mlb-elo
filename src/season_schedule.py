"""Remaining regular-season schedule and the league/division structure.

Both come from the same MLB Stats API the rest of the project uses — the
teams endpoint is the source of truth for divisions (never a hardcoded
name dict), and the schedule pull reuses ingest.py's raw-endpoint helper.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.games_data import load_elo_games
from src.ingest import _schedule_json
from src.mlb_api import season_dates, session

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"


@lru_cache(maxsize=None)
def team_structure(season=None):
    """{team_id: (league_name, division_name, team_name)} from the teams
    endpoint, cached per process."""
    season = season or date.today().year
    resp = session.get(TEAMS_URL, params={"sportId": 1, "season": season},
                       timeout=30)
    resp.raise_for_status()
    return {int(t["id"]): (t["league"]["name"], t["division"]["name"], t["name"])
            for t in resp.json()["teams"]}


def remaining_schedule(season=None):
    """DataFrame of not-yet-Final regular-season games from today through
    the season's end date: game_id, game_date, home_id, away_id.

    Games already Final in the unified table (load_elo_games) are excluded
    (deduped on game_id) so nothing is double-counted against the current
    records; in-progress and delayed games count as remaining since they have
    no decided result yet.
    """
    season = season or date.today().year
    _start, end = season_dates(season)
    today = date.today().isoformat()
    if today > end:
        return pd.DataFrame(columns=["game_id", "game_date", "home_id", "away_id"])
    data = _schedule_json(today, end)

    finals = set(load_elo_games()["game_id"].astype(int))

    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            if g.get("status", {}).get("detailedState") == "Final":
                continue
            if g["gamePk"] in finals:
                continue
            rows.append({
                "game_id": int(g["gamePk"]),
                "game_date": g["gameDate"][:10],
                "home_id": int(g["teams"]["home"]["team"]["id"]),
                "away_id": int(g["teams"]["away"]["team"]["id"]),
            })
    return pd.DataFrame(rows, columns=["game_id", "game_date", "home_id", "away_id"])
