"""Shared MLB Stats API plumbing.

One polite HTTP session (User-Agent set), a throttle helper for
multi-season pulls, and live season start/end date lookups. Every module
that touches the network imports from here so all requests share the
same client configuration.
"""

import random
import time

import requests
import statsapi

USER_AGENT = "mlb-elo/1.0 (hobby Elo rating project; python-requests)"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT

# MLB-StatsAPI calls the module-level name `requests`; pointing it at our
# session makes every wrapper call carry the User-Agent above.
statsapi.requests = session


def polite_sleep(lo=0.2, hi=0.5):
    """Small delay between consecutive pulls — be a polite client."""
    time.sleep(random.uniform(lo, hi))


def season_dates(year):
    """Regular-season (start, end) as 'YYYY-MM-DD' strings, looked up live
    so nothing is hardcoded (openers abroad shift start dates)."""
    resp = session.get(
        "https://statsapi.mlb.com/api/v1/seasons",
        params={"sportId": 1, "season": year},
        timeout=30,
    )
    resp.raise_for_status()
    seasons = resp.json().get("seasons", [])
    if not seasons:
        raise ValueError(f"MLB Stats API returned no season info for {year}")
    return seasons[0]["regularSeasonStartDate"], seasons[0]["regularSeasonEndDate"]
