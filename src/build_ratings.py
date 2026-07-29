"""Replay the unified Elo game window chronologically through elo.py to get
current team ratings, keyed on stable team_id (never team name).

Source: src.games_data.load_elo_games() (games_full.csv restricted to the Elo
window) — the retired data/games.csv is no longer read. The Elo math, K, HFA,
season-boundary regression, and game ordering are unchanged; only the data
source moved (see the games.csv migration).

Run from the project root to print the table:  python -m src.build_ratings
"""

from pathlib import Path

import pandas as pd

from src.elo import BASE, regress_to_mean, update
from src.games_data import load_elo_games


def build_ratings(games=None, use_mov=False):
    """Returns (ratings, names): {team_id: rating} and {team_id: latest name}.

    games: optional pre-shaped input — a DataFrame with the legacy columns, or
    a CSV path (used by the backtest / tests). Default None loads the unified
    Elo window via load_elo_games().

    use_mov stays False in v1 — the margin-of-victory plumbing exists but
    the default prediction path does not use it.
    """
    if games is None:
        df = load_elo_games()
    elif isinstance(games, (str, Path)):
        df = pd.read_csv(games)
    else:
        df = games.copy()
    df = df.sort_values("game_datetime", kind="stable")

    ratings, names = {}, {}
    prev_season = None
    for g in df.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            ratings = regress_to_mean(ratings)  # once per season boundary
        prev_season = g.season

        home, away = int(g.home_id), int(g.away_id)
        ratings.setdefault(home, BASE)
        ratings.setdefault(away, BASE)
        names[home], names[away] = g.home_name, g.away_name

        ratings[home], ratings[away] = update(
            ratings[home], ratings[away],
            home_won=g.home_score > g.away_score,
            run_diff=(g.home_score - g.away_score) if use_mov else None,
        )
    return ratings, names


def main():
    ratings, names = build_ratings()
    print(f"{'rank':>4}  {'team':<26} {'elo':>7}")
    for i, (tid, r) in enumerate(sorted(ratings.items(), key=lambda kv: -kv[1]), 1):
        print(f"{i:>4}  {names[tid]:<26} {r:>7.1f}")


if __name__ == "__main__":
    main()
