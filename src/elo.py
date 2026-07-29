"""Pure Elo rating math for MLB teams.

Hard rule: this module does NO file I/O and NO network calls — pure
functions in, numbers out. All I/O lives in the other src modules.
"""

import math

BASE = 1500.0
K    = 4.0    # low on purpose: 162 games/season, so each game moves little
HFA  = 25.0   # home edge in Elo pts (~54% home win rate — baseball's edge is SMALL)


def expected_home(r_home, r_away, hfa=HFA):
    """Win probability for the home team."""
    return 1.0 / (1.0 + 10 ** (-(r_home + hfa - r_away) / 400.0))


def update(r_home, r_away, home_won, run_diff=None, hfa=HFA, k=K):
    """One game's rating update; returns (new_r_home, new_r_away).

    run_diff=None keeps the margin-of-victory multiplier OFF (the v1
    default). Pass the home-minus-away run differential to turn it on.
    """
    p = expected_home(r_home, r_away, hfa)
    s = 1.0 if home_won else 0.0
    mult = 1.0 if run_diff is None else _mov_mult(run_diff, r_home + hfa - r_away, home_won)
    delta = k * mult * (s - p)
    return r_home + delta, r_away - delta


def _mov_mult(run_diff, elo_diff_home, home_won):
    # FiveThirtyEight-style margin-of-victory multiplier; damps blowout autocorrelation
    rd = abs(run_diff)
    winner_elo_diff = elo_diff_home if home_won else -elo_diff_home
    return math.log(rd + 1.0) * (2.2 / (winner_elo_diff * 0.001 + 2.2))


def regress_to_mean(ratings, frac=1 / 3, base=BASE):
    # call ONCE at each season boundary; otherwise new-season predictions are stale
    return {t: base + (r - base) * (1 - frac) for t, r in ratings.items()}
