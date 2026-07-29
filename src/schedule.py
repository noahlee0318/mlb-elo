"""One date's MLB slate from the Stats API schedule endpoint.

Probable pitchers are carried for DISPLAY ONLY — they never enter any
calculation. An empty slate is a normal answer (off-days, All-Star
break), not an error.
"""

from datetime import date
from functools import lru_cache

from src.mlb_api import session, statsapi


def todays_date_str():
    """Today per the system clock — never hardcoded."""
    return date.today().isoformat()


@lru_cache(maxsize=None)
def venue_city(venue_id):
    """'Boston MA' for a venue id, cached per venue. Display-only enrichment,
    so any lookup failure degrades to '' rather than breaking the slate."""
    try:
        resp = session.get(f"https://statsapi.mlb.com/api/v1/venues/{venue_id}",
                           params={"hydrate": "location"}, timeout=30)
        resp.raise_for_status()
        loc = resp.json()["venues"][0].get("location", {})
    except Exception:
        return ""
    city = loc.get("city")
    # international venues (London, Tokyo, ...) have no state — use country
    region = loc.get("stateAbbrev") or loc.get("state") or loc.get("country")
    return " ".join(p for p in (city, region) if p) if city else ""


def _probable_pitcher_ids(date_str):
    """{game_pk: (home_pp_id, away_pp_id)} for a date, from the RAW hydrated
    schedule. The statsapi.schedule() wrapper below keeps only the probable
    pitcher's NAME and drops the id; this supplemental read recovers the
    canonical person.id that pitcher_games.csv is keyed on, so the live feature
    path can look up the starter's trailing stats instead of imputing. A truly
    absent starter (none posted yet) flows through as None. Any failure
    degrades to {} — the caller then sees None and correctly imputes+flags."""
    try:
        data = statsapi.get("schedule",
                            {"sportId": 1, "startDate": date_str,
                             "endDate": date_str, "hydrate": "probablePitcher"})
    except Exception:
        return {}

    def _pid(team):
        pp = (team or {}).get("probablePitcher") or {}
        v = pp.get("id")
        return int(v) if v is not None else None

    out = {}
    for day in data.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {}) or {}
            out[g.get("gamePk")] = (_pid(teams.get("home")),
                                    _pid(teams.get("away")))
    return out


def fetch_slate(date_str=None):
    """Regular-season games for one date (default: today) as list of dicts."""
    ds = date_str or todays_date_str()
    games = statsapi.schedule(date=ds, sportId=1)
    pp_ids = _probable_pitcher_ids(ds)   # game_pk -> (home_pp_id, away_pp_id)
    slate = []
    for g in games:
        if g.get("game_type") != "R":
            continue  # the model covers regular-season games only
        venue = g.get("venue_name") or "TBD"
        city = venue_city(g["venue_id"]) if g.get("venue_id") else ""
        slate.append({
            "game_id": g["game_id"],
            "game_datetime": g["game_datetime"],
            "status": g["status"],
            "home_id": int(g["home_id"]),
            "away_id": int(g["away_id"]),
            "home_name": g["home_name"],
            "away_name": g["away_name"],
            "venue_name": f"{venue}, {city}" if city else venue,
            # 'N' = single game; 'Y'/'S' = (split) doubleheader, game_num 1 or 2
            "doubleheader": g.get("doubleheader") or "N",
            "game_num": int(g.get("game_num") or 1),
            "home_probable_pitcher": g.get("home_probable_pitcher") or "TBD",
            "away_probable_pitcher": g.get("away_probable_pitcher") or "TBD",
            # canonical person.id for the probable starter (None if unposted);
            # recovered from the raw schedule since the wrapper drops it.
            "home_probable_pitcher_id": pp_ids.get(g["game_id"], (None, None))[0],
            "away_probable_pitcher_id": pp_ids.get(g["game_id"], (None, None))[1],
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
        })
    return slate


if __name__ == "__main__":
    slate = fetch_slate()
    if not slate:
        print("No regular-season games on this date (off-day or All-Star break).")
    for g in slate:
        print(f'{g["away_name"]} @ {g["home_name"]}  '
              f'[{g["away_probable_pitcher"]} vs {g["home_probable_pitcher"]}]  ({g["status"]})')
