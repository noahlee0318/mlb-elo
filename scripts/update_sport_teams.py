"""Refresh the checked-in NFL/NBA team directory from ESPN's public API.

Run from the repository root: python -m scripts.update_sport_teams --as-of YYYY-MM-DD
Only metadata is saved; the dashboard makes no API requests for these pages.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path

import requests


def fetch_json(url, **params):
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_team(base, summary):
    source = f"{base}/{summary['id']}"
    team = fetch_json(source)["team"]
    venue = team["franchise"]["venue"]
    address = venue.get("address", {})
    return {
        "id": team["id"], "name": team["displayName"],
        "abbreviation": team["abbreviation"],
        "logo": team["logos"][0]["href"],
        "venue": venue["fullName"],
        "city": address.get("city", ""),
        "region": address.get("state", address.get("country", "")),
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "data" / "teams"
    output.mkdir(exist_ok=True)
    # ESPN franchise metadata can retain a previous home venue. Preserve
    # reviewed corrections and their primary-source URLs across refreshes.
    overrides = json.loads((output / "venue_overrides.json").read_text(encoding="utf-8"))
    for sport, league, expected in [("football", "nfl", 32), ("basketball", "nba", 30)]:
        base = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams"
        summaries = fetch_json(base, limit=100)["sports"][0]["leagues"][0]["teams"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            teams = list(pool.map(lambda entry: fetch_team(base, entry["team"]), summaries))
        teams.sort(key=lambda team: team["name"])
        for team in teams:
            team.update(overrides.get(league, {}).get(team["id"], {}))
        if len(teams) != expected or len({team["id"] for team in teams}) != expected:
            raise ValueError(f"Unexpected {league} team count; existing snapshot preserved")
        if not all(team["name"] and team["venue"] and team["logo"].startswith("https://")
                   for team in teams):
            raise ValueError(f"Incomplete {league} team metadata; existing snapshot preserved")
        snapshot = {"league": league.upper(), "as_of": args.as_of.isoformat(),
                    "source": base, "teams": teams}
        (output / f"{league}.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{league.upper()}: {len(teams)} teams saved")


if __name__ == "__main__":
    main()
