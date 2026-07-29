"""Unit tests for the pure Elo math in src/elo.py.

Run from the project root:  python -m pytest
"""

import pytest

from src.elo import BASE, K, _mov_mult, expected_home, regress_to_mean, update


def test_even_teams_no_hfa_is_coin_flip():
    assert expected_home(1500.0, 1500.0, hfa=0) == 0.5


def test_even_teams_default_hfa_is_small_home_edge():
    p = expected_home(1500.0, 1500.0, hfa=25)
    assert 0.53 < p < 0.54  # ~0.536, the real MLB home-win rate


def test_update_conserves_total_rating():
    for home_won in (True, False):
        r_h, r_a = update(1543.7, 1481.2, home_won)
        assert r_h + r_a == pytest.approx(1543.7 + 1481.2, abs=1e-9)


def test_update_with_mov_still_conserves_total_rating():
    r_h, r_a = update(1543.7, 1481.2, True, run_diff=6)
    assert r_h + r_a == pytest.approx(1543.7 + 1481.2, abs=1e-9)


def test_higher_rated_home_team_favored():
    assert expected_home(1560.0, 1500.0) > 0.5
    assert expected_home(1560.0, 1500.0, hfa=0) > 0.5  # even without the home bump


def test_winner_gains_loser_drops():
    r_h, r_a = update(1500.0, 1500.0, home_won=True)
    assert r_h > 1500.0 > r_a
    r_h, r_a = update(1500.0, 1500.0, home_won=False)
    assert r_h < 1500.0 < r_a


def test_k_is_small_so_one_game_moves_little():
    r_h, _ = update(1500.0, 1500.0, home_won=True)
    assert 0.0 < r_h - 1500.0 <= K  # never more than K per game with MOV off


def test_probabilities_symmetric_without_hfa():
    p = expected_home(1520.0, 1490.0, hfa=0) + expected_home(1490.0, 1520.0, hfa=0)
    assert p == pytest.approx(1.0)


def test_regress_to_mean_pulls_one_third_back():
    out = regress_to_mean({1: 1560.0, 2: 1440.0})
    assert out[1] == pytest.approx(BASE + 60.0 * (2 / 3))
    assert out[2] == pytest.approx(BASE - 60.0 * (2 / 3))


def test_mov_multiplier_positive_and_grows_with_margin():
    assert 0.0 < _mov_mult(1, 0.0, True) < _mov_mult(8, 0.0, True)


def test_mov_off_by_default():
    assert update(1500.0, 1500.0, True) == update(1500.0, 1500.0, True, run_diff=None)
