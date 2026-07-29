"""MLB Stats API schedule-endpoint helpers.

Once the builder of data/games.csv, this module is now RETIRED as a table
writer — the games.csv migration made games_full.csv (built by
src/ingest_history.py, kept current by scripts/backfill_season.py) the single
source of truth. What survives here are the raw-schedule pull helpers other
modules still import: `_schedule_json` (used by src/season_schedule.py) and the
Final-game parsers. Running `python -m src.ingest` no longer writes any file.
"""

import time
from pathlib import Path

import requests

from src.mlb_api import season_dates, session

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

COLUMNS = ["game_datetime", "game_id", "season", "home_id", "away_id",
           "home_name", "away_name", "home_score", "away_score"]


def _schedule_json(start, end, retries=3):
    """One season-range schedule call. Uses the raw endpoint with a light
    hydrate — the wrapper's default hydrate (broadcasts/media/etc.) is so
    heavy that full-season pulls 503 on the backend."""
    params = {"sportId": 1, "startDate": start, "endDate": end,
              "gameType": "R", "hydrate": "linescore"}
    for attempt in range(retries):
        try:
            resp = session.get(SCHEDULE_URL, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def parse_finals(data, season=None):
    """games.csv-shaped rows for every Final regular-season game in one
    schedule JSON payload. season=None derives it from each game's date."""
    rows = {}
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            if g.get("status", {}).get("detailedState") != "Final":
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            hs, aw = home.get("score"), away.get("score")
            if hs is None or aw is None:
                continue
            if int(hs) == int(aw):
                continue  # a tie can't teach a win/loss model anything
            rows[g["gamePk"]] = {  # keyed on gamePk to dedupe; doubleheaders have distinct ids
                "game_datetime": g["gameDate"],
                "game_id": int(g["gamePk"]),
                "season": season if season is not None else int(g["gameDate"][:4]),
                "home_id": int(home["team"]["id"]),
                "away_id": int(away["team"]["id"]),
                "home_name": home["team"]["name"],
                "away_name": away["team"]["name"],
                "home_score": int(hs),
                "away_score": int(aw),
            }
    return list(rows.values())


def fetch_season_finals(year, end_cap=None):
    """All Final regular-season games for one season, one schedule call."""
    start, end = season_dates(year)
    if end_cap is not None and end_cap < end:
        end = end_cap
    if end < start:
        return []  # season hasn't started yet
    return parse_finals(_schedule_json(start, end), season=year)


def main():
    raise SystemExit(
        "src.ingest is retired as a table writer: data/games.csv has been "
        "replaced by data/games_full.csv (the games.csv migration). Rebuild "
        "the full history with `python src/ingest_history.py`, or bring the "
        "current season current with `python scripts/backfill_season.py`. "
        "The schedule helpers (_schedule_json, parse_finals, fetch_season_finals) "
        "remain importable for the live schedule path."
    )


if __name__ == "__main__":
    main()
