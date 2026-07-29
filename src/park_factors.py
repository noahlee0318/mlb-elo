"""Chunk 10, Deliverable 1: leakage-clean park factors.

A park factor is a single flat run-environment multiplier per venue per season,
FROZEN from completed prior seasons only (N-3..N-1, 2020 excluded). Because it
uses only fully-completed seasons strictly before N, no game in season N can
influence its own factor — the as-of property is trivial and does not depend on
within-season ordering.

Formula (the simple, documented version): for the prior-season window,
    park_factor(V) = (runs/game in games played at V) / (league runs/game)
Both teams' runs count (home_score + away_score). ~1.0 is neutral; Coors > 1,
pitcher parks < 1. KNOWN LIMITATION (accepted for this validation chunk): the
simple form lets a venue's home team's own offense/defense contaminate its
factor — the team-vs-road-split form controls for that. It doesn't matter here:
park effects largely cancel in a win-probability differential, so the feature is
expected to be inert, and the simple form is defensible and standard.

2020 is excluded from the computation (short, fan-less, atypical run
environment), matching its exclusion from every model split.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

PARK_WINDOW = 3          # N-3 .. N-1
EXCLUDE_SEASONS = {2020}
NEUTRAL = 1.0


def compute_park_factors(games, up_to_season):
    """Frozen {venue_id: factor} for `up_to_season`, from seasons in
    [up_to_season-PARK_WINDOW, up_to_season-1] minus EXCLUDE_SEASONS.

    NEVER reads a game from season >= up_to_season (asserted). Venues absent
    from the prior window are simply not in the returned dict; the caller
    (park_factor_for_game) maps their absence to the neutral factor."""
    up_to_season = int(up_to_season)
    window = [s for s in range(up_to_season - PARK_WINDOW, up_to_season)
              if s not in EXCLUDE_SEASONS]
    prior = games[games["season"].isin(window)].copy()

    # leakage guard — the core promise of this module
    assert not (prior["season"] >= up_to_season).any(), \
        f"season >= {up_to_season} leaked into its own park-factor window"

    prior = prior[prior["venue_id"].notna()]
    if prior.empty:
        return {}

    runs = (prior["home_score"].astype(float)
            + prior["away_score"].astype(float))
    league_rpg = float(runs.sum()) / float(len(prior))
    if league_rpg <= 0:
        return {}

    tmp = pd.DataFrame({"venue_id": prior["venue_id"].astype(int).to_numpy(),
                        "runs": runs.to_numpy()})
    agg = tmp.groupby("venue_id")["runs"].agg(["sum", "count"])
    factors = {int(v): float((r["sum"] / r["count"]) / league_rpg)
               for v, r in agg.iterrows()}
    return factors


def park_factor_for_game(season, venue_id, factors_by_season):
    """(factor, imputed) for one game. factors_by_season is
    {season: {venue_id: factor}}. A null venue or a venue absent from that
    season's frozen factors -> (NEUTRAL, True)."""
    fac = factors_by_season.get(int(season), {})
    if pd.isna(venue_id):
        return NEUTRAL, True
    v = int(venue_id)
    if v not in fac:
        return NEUTRAL, True
    return fac[v], False


def factors_by_season_for(games, seasons):
    """Convenience: {season: compute_park_factors(games, season)} for a set of
    seasons, sharing one games frame. Used by the feature builder and audit."""
    return {int(N): compute_park_factors(games, int(N)) for N in seasons}


if __name__ == "__main__":
    from src.games_data import load_games
    g = load_games()
    seasons = sorted(int(s) for s in g["season"].unique())
    fbs = factors_by_season_for(g, seasons)
    for N in seasons:
        fac = fbs[N]
        if fac:
            vals = sorted(fac.values())
            print(f"{N}: {len(fac)} venues  min={vals[0]:.3f} "
                  f"median={vals[len(vals)//2]:.3f} max={vals[-1]:.3f}")
        else:
            print(f"{N}: no prior-window data -> all games neutral (imputed)")
