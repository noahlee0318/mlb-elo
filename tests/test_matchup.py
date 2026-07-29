"""Tests for the head-to-head matchup feature. The Elo adapter is used
with synthetic ratings; format math routes through playoffs.py's DP."""

import numpy as np
import pytest

from src.matchup import (FORMATS, check_matchup, simulate_outcomes,
                         win_probability)
from src.prob_model import EloProbabilityModel

A, B = 1, 2
EQUAL = EloProbabilityModel({A: 1500.0, B: 1500.0})
FAVORED = EloProbabilityModel({A: 1560.0, B: 1500.0})

# minimal structure: A and C share a league (different divisions),
# B is in the other league; D shares A's division
C, D = 3, 4
STRUCTURE = {A: ("NL", "NL East"), B: ("AL", "AL East"),
             C: ("NL", "NL West"), D: ("NL", "NL East")}
NAMES = {A: "Alpha", B: "Bravo", C: "Charlie", D: "Delta"}


class FlatModel:
    """One team is a constant p favorite regardless of venue — isolates
    the series math from home-field effects for the anchor checks."""

    def __init__(self, p, favorite=A):
        self.p, self.favorite = p, favorite

    def matchup_prob(self, home_id, away_id, context=None):
        return self.p if home_id == self.favorite else 1.0 - self.p


def test_equal_ratings_neutral_single_game_is_exactly_half():
    p = win_probability(EQUAL, A, B, "Single game", host=A, neutral=True)
    assert p == 0.5


def test_equal_ratings_home_single_game_shows_hfa():
    p = win_probability(EQUAL, A, B, "Single game", host=A)
    assert p == pytest.approx(0.536, abs=0.002)


def test_probabilities_sum_to_one_for_every_format():
    for fmt in FORMATS:
        p_a = win_probability(FAVORED, A, B, fmt, host=A)
        p_b = win_probability(FAVORED, B, A, fmt, host=A)
        assert p_a + p_b == pytest.approx(1.0, abs=1e-12), fmt
        assert 0.0 < p_a < 1.0


def test_home_swap_moves_single_game_probability_about_seven_points():
    p_home = win_probability(EQUAL, A, B, "Single game", host=A)
    p_away = win_probability(EQUAL, A, B, "Single game", host=B)
    assert 0.05 < p_home - p_away < 0.09  # zero would mean HFA is not wired in


def test_series_amplify_the_favorite_anchor_values():
    flat = FlatModel(0.55)
    bo3 = win_probability(flat, A, B, "Wild Card Series (best of 3)", host=A)
    bo7 = win_probability(flat, A, B, "World Series (best of 7)", host=A)
    assert bo3 == pytest.approx(0.575, abs=0.002)
    assert bo7 == pytest.approx(0.608, abs=0.002)
    assert 0.55 < bo3 < bo7  # longer series push further from a coin flip


def test_series_never_closer_to_half_than_single_game():
    single = win_probability(FAVORED, A, B, "Single game", host=A, neutral=True)
    for fmt, pattern in FORMATS.items():
        if pattern is None:
            continue
        series = win_probability(FAVORED, A, B, fmt, host=A)
        assert abs(series - 0.5) > abs(single - 0.5) - 1e-9, fmt


def test_impossibility_flags():
    ws = "World Series (best of 7)"
    lcs = "League Championship Series (best of 7)"
    # same league in the WS -> flagged
    assert check_matchup(A, C, ws, STRUCTURE, NAMES) is not None
    # cross-league in WCS / DS / LCS -> flagged
    for fmt in ("Wild Card Series (best of 3)", "Division Series (best of 5)", lcs):
        assert check_matchup(A, B, fmt, STRUCTURE, NAMES) is not None, fmt
    # legitimate pairings -> no flag
    assert check_matchup(A, B, ws, STRUCTURE, NAMES) is None
    assert check_matchup(A, D, lcs, STRUCTURE, NAMES) is None  # same division: fine
    assert check_matchup(A, C, lcs, STRUCTURE, NAMES) is None
    # single game is never flagged, even cross-league
    assert check_matchup(A, B, "Single game", STRUCTURE, NAMES) is None


def test_flagged_matchup_still_returns_probability_and_simulates():
    ws = "World Series (best of 7)"
    assert check_matchup(A, C, ws, STRUCTURE, NAMES) is not None
    p = win_probability(FAVORED, A, C, ws, host=A)
    assert 0.0 < p < 1.0
    wins = simulate_outcomes(p, n=1000, rng=np.random.default_rng(0))
    assert 0 < wins < 1000


def test_simulate_outcomes_matches_probability_in_bulk():
    wins = simulate_outcomes(0.55, n=10000, rng=np.random.default_rng(1))
    assert wins == pytest.approx(5500, abs=150)
