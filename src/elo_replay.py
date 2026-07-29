"""Chunk 5, Deliverable 2: replay Elo from scratch and record a LEAKAGE-FREE
pre-game probability for every game.

THIS MODULE MUST NEVER READ data/ratings.csv. Ratings are rebuilt from
BASE (1500) inside replay_elo, game by game. ratings.csv is a snapshot of
end-of-data ratings; using it to score historical games would leak the
future. This module does not import, open, or reference that file anywhere.

The season-boundary one-third regression is imported from src.elo (the same
regress_to_mean that build_ratings.py imports), so the two cannot drift.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.elo import BASE, expected_home, regress_to_mean, update
from src.games_data import load_games


def replay_elo(start_season, end_season):
    """One row per game in [start_season, end_season]: game_pk, date, season,
    elo_prob_home, home_rating_pre, away_rating_pre.

    Predict-before-update: the recorded probability uses ratings as they
    stand BEFORE the game, then the actual result updates them. Iterates
    EVERY game load_games returns for the range — including 2020, every
    team's first 15 games, and the postseason — because the rating state
    needs them all; downstream scoring does the filtering, not this function.
    A game with a null result (tie/missing home_win) still gets a recorded
    pre-game probability but does not update ratings.
    """
    games = load_games(include_postseason=True, exclude_seasons=(),
                       seasons=list(range(start_season, end_season + 1)))
    # deterministic order; doubleheaders never reorder between runs
    sort_cols = (["date", "start_time_utc", "game_pk"]
                 if "start_time_utc" in games.columns else ["date", "game_pk"])
    games = games.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    ratings = {}
    prev_season = None
    rows = []
    for g in games.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            ratings = regress_to_mean(ratings)   # once per season boundary
        prev_season = g.season

        h, a = int(g.home_team_id), int(g.away_team_id)
        ratings.setdefault(h, BASE)
        ratings.setdefault(a, BASE)
        rh, ra = ratings[h], ratings[a]

        # PREDICT first, from pre-game ratings
        rows.append({
            "game_pk": int(g.game_pk), "date": g.date, "season": int(g.season),
            "elo_prob_home": expected_home(rh, ra),
            "home_rating_pre": rh, "away_rating_pre": ra,
        })

        # UPDATE after, from the actual result (skip null/tie results)
        hw = g.home_win
        if pd.isna(hw):
            continue
        ratings[h], ratings[a] = update(rh, ra, home_won=bool(int(hw)))

    return pd.DataFrame(rows, columns=["game_pk", "date", "season",
                                       "elo_prob_home", "home_rating_pre",
                                       "away_rating_pre"])


if __name__ == "__main__":
    df = replay_elo(2015, 2025)
    print(f"replayed {len(df)} games; elo_prob_home range "
          f"[{df['elo_prob_home'].min():.4f}, {df['elo_prob_home'].max():.4f}]")
