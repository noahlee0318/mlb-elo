"""Walk-forward backtest: warm up on prior seasons, evaluate the test season
by predicting each game before updating ratings on it (leakage-free).

Elo stays online through the test season — ratings keep updating after each
game, and each prediction uses only games that already happened. Elo
constants (K, HFA, regression fraction) are the live app's values via
src/elo.py; do not tune them against TEST_SEASON — point TEST_SEASON at an
earlier season for any tuning and keep the current season sealed.

Games come from the unified Elo window (load_elo_games, from games_full.csv)
— the retired data/games.csv is no longer read.

Run from the project root:
    python src/backtest.py            # test season = TEST_SEASON (2026)
    python src/backtest.py 2025       # any other season in the Elo window
(python -m src.backtest also works)
"""

import math
import sys
from pathlib import Path

# Allow `python src/backtest.py` from the project root, where sys.path[0]
# is src/ and the package-style imports below would otherwise fail.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.elo import BASE, expected_home, regress_to_mean, update
from src.games_data import load_elo_games
from src.predict import BASELINE_PROB_HOME as BASELINE_HOME_PROB  # 0.540

TEST_SEASON = 2026


def load_games(path=None):
    """Chronological game table for the backtest. Default None reads the
    unified Elo window (load_elo_games, from games_full.csv, already sorted
    by game_datetime); a CSV path reads a legacy-shaped table instead (tests)."""
    if path is None:
        return load_elo_games()
    df = pd.read_csv(path, parse_dates=["game_datetime"])
    return df.sort_values("game_datetime").reset_index(drop=True)


def log_loss_one(prob, actual, eps=1e-15):
    p = min(max(prob, eps), 1 - eps)
    return -(actual * math.log(p) + (1 - actual) * math.log(1 - p))


def run_backtest(df, test_season=TEST_SEASON):
    ratings, prev_season, scored = {}, None, []
    for g in df.itertuples(index=False):
        h, a = g.home_id, g.away_id
        ratings.setdefault(h, BASE)
        ratings.setdefault(a, BASE)

        if prev_season is not None and g.season != prev_season:
            ratings = regress_to_mean(ratings)
            ratings.setdefault(h, BASE)
            ratings.setdefault(a, BASE)
        prev_season = g.season

        home_won = 1 if g.home_score > g.away_score else 0

        if g.season == test_season:  # PREDICT before updating
            scored.append({
                "elo_prob": expected_home(ratings[h], ratings[a]),
                "baseline_prob": BASELINE_HOME_PROB,
                "actual": home_won,
            })

        ratings[h], ratings[a] = update(ratings[h], ratings[a], home_won)  # UPDATE after
    return pd.DataFrame(scored)


def summarize(results):
    def metrics(col):
        picks = (results[col] >= 0.5).astype(int)
        acc = (picks == results["actual"]).mean()
        ll = results.apply(lambda r: log_loss_one(r[col], r["actual"]), axis=1).mean()
        return acc, ll

    elo_acc, elo_ll = metrics("elo_prob")
    base_acc, base_ll = metrics("baseline_prob")
    print(f"Test games scored: {len(results)}\n")
    print(f"{'model':<10}{'accuracy':>10}{'log_loss':>12}")
    print(f"{'Elo':<10}{elo_acc:>10.3f}{elo_ll:>12.4f}")
    print(f"{'baseline':<10}{base_acc:>10.3f}{base_ll:>12.4f}")
    print(f"\nElo edge vs baseline: {elo_acc - base_acc:+.3f} accuracy, "
          f"{base_ll - elo_ll:+.4f} log-loss  (positive = Elo better)")
    if elo_ll >= base_ll:
        print("\nWARNING: Elo did not beat baseline on log-loss. Suspect the "
              "season-regression step or the Final/regular-season filter.")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else TEST_SEASON
    df = load_games()
    results = run_backtest(df, test_season=season)
    if results.empty:
        print(f"No games found for test season {season}. Check games_full.csv and the year.")
    else:
        summarize(results)
