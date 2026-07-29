"""Identity tests for the Monte Carlo season engine.

These are hard invariants, not statistical hopes — any failure is a real
bug. The synthetic league is exact (every team ends at 162 games) so the
identities have no data noise; the engine is exercised through the same
ProbabilityModel seam the app uses, with models defined here so the
engine tests stay independent of Elo.
"""

import copy
from math import comb

import numpy as np
import pandas as pd
import pytest

from src.playoffs import series_win_prob
from src.prob_model import EloProbabilityModel
from src.simulate import simulate_season

N_SIMS = 2000
SEED = 7


# --- synthetic 30-team league: 2 leagues x 3 divisions x 5 teams -------------

def league_structure():
    structure = {}
    tid = 1
    for lg in ("L1", "L2"):
        for dv in ("E", "C", "W"):
            for _ in range(5):
                structure[tid] = (lg, f"{lg}-{dv}")
                tid += 1
    return structure


def synthetic_remaining(structure, rounds=62):
    """Round-robin rotation: every team plays exactly `rounds` games."""
    teams = sorted(structure)
    rows, gid = [], 1
    rest = teams[1:]
    for r in range(rounds):
        arr = [teams[0]] + rest
        for i in range(len(teams) // 2):
            a, b = arr[i], arr[-1 - i]
            home, away = (a, b) if r % 2 == 0 else (b, a)
            rows.append({"game_id": gid, "game_date": "2026-08-01",
                         "home_id": home, "away_id": away})
            gid += 1
        rest = rest[-1:] + rest[:-1]
    return pd.DataFrame(rows)


STRUCTURE = league_structure()
REMAINING = synthetic_remaining(STRUCTURE)
RECORDS = {t: (50, 50) for t in STRUCTURE}


class FlatModel:
    """Fixed probability for every game — (G,) return shape."""

    def __init__(self, p=0.5):
        self.p = p

    def schedule_probs(self, games, n_sims=None):
        return np.full(len(games), self.p)

    def matchup_prob(self, home_id, away_id, context=None):
        return self.p


class PerSimModel(FlatModel):
    """(n_sims, G) return shape — models strength uncertainty per sim."""

    def schedule_probs(self, games, n_sims=None):
        rng = np.random.default_rng(123)
        return np.clip(rng.normal(0.5, 0.05, (n_sims, len(games))), 0.01, 0.99)


@pytest.fixture(scope="module")
def result():
    return simulate_season(REMAINING, RECORDS, STRUCTURE, FlatModel(0.55),
                           n_sims=N_SIMS, seed=SEED)


def test_berth_probabilities_sum_to_twelve(result):
    total = sum(t["p_berth"] for t in result["teams"].values())
    assert total == pytest.approx(12.0, abs=1e-9)


def test_title_probabilities_sum_to_one(result):
    total = sum(t["p_title"] for t in result["teams"].values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_pennant_probabilities_sum_to_one_per_league(result):
    for lg in ("L1", "L2"):
        total = sum(result["teams"][t]["p_pennant"] for t in STRUCTURE
                    if STRUCTURE[t][0] == lg)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_mean_wins_match_schedule_expectation(result):
    p = 0.55
    for t, row in result["teams"].items():
        exp_home = p * (REMAINING["home_id"] == t).sum()
        exp_away = (1 - p) * (REMAINING["away_id"] == t).sum()
        expected = RECORDS[t][0] + exp_home + exp_away
        assert row["mean_wins"] == pytest.approx(expected, abs=0.5)


def test_league_wins_equal_losses_every_sim(result):
    tw = np.asarray(result["total_wins"]).sum(axis=1)
    tl = np.asarray(result["total_losses"]).sum(axis=1)
    assert (tw == tl).all()  # every game hands out exactly one of each


def test_every_team_projects_162_games(result):
    totals = {t: row["total_games"] for t, row in result["teams"].items()}
    off = {t: n for t, n in totals.items() if n != 162}
    # synthetic league is exact; with real data a postponed-unmade-up game
    # should be FLAGGED by the caller, never silently forced to 162
    assert off == {}, f"teams not projecting 162 games: {off}"


def test_fixed_seed_is_reproducible():
    a = simulate_season(REMAINING, RECORDS, STRUCTURE, FlatModel(0.55),
                        n_sims=300, seed=11)
    b = simulate_season(REMAINING, RECORDS, STRUCTURE, FlatModel(0.55),
                        n_sims=300, seed=11)
    assert a["teams"] == b["teams"]
    assert np.array_equal(a["total_wins"], b["total_wins"])


def test_per_sim_probability_shape_is_accepted():
    out = simulate_season(REMAINING, RECORDS, STRUCTURE, PerSimModel(),
                          n_sims=300, seed=3)
    assert sum(t["p_title"] for t in out["teams"].values()) == pytest.approx(1.0)


def test_ratings_are_not_mutated_by_a_simulation_run():
    """Guardrail for the frozen-ratings constraint: the ratings object the
    Elo adapter wraps must be byte-identical after a full simulation."""
    ratings = {t: 1500.0 + (t * 7 % 90) - 45 for t in STRUCTURE}
    snapshot = copy.deepcopy(ratings)
    model = EloProbabilityModel(ratings)
    simulate_season(REMAINING, RECORDS, STRUCTURE, model,
                    n_sims=300, seed=5)
    assert ratings == snapshot
    assert model._ratings == snapshot


def test_playoff_labels_field_shape(result):
    from src.standings import assign_playoff_labels
    wins = {t: r["mean_wins"] for t, r in result["teams"].items()}
    labels = assign_playoff_labels(wins, STRUCTURE)
    assert len(labels) == 12                       # exactly the playoff field
    counts = {v: sum(1 for x in labels.values() if x == v) for v in "zyx"}
    assert counts == {"z": 2, "y": 4, "x": 6}
    for lg in ("L1", "L2"):
        members = {t for t in STRUCTURE if STRUCTURE[t][0] == lg}
        assert sum(1 for t in members if labels.get(t) == "z") == 1
        assert sum(1 for t in members if labels.get(t) == "y") == 2
        assert sum(1 for t in members if labels.get(t) == "x") == 3
    assert "e" not in labels.values()              # never an elimination marker


def test_playoff_labels_z_and_y_lead_their_divisions(result):
    from src.standings import assign_playoff_labels
    wins = {t: r["mean_wins"] for t, r in result["teams"].items()}
    labels = assign_playoff_labels(wins, STRUCTURE)
    for t, letter in labels.items():
        if letter in ("z", "y"):
            division = [o for o in STRUCTURE if STRUCTURE[o][1] == STRUCTURE[t][1]]
            assert wins[t] == max(wins[o] for o in division), (t, letter)


def test_series_win_prob_identities():
    assert series_win_prob([0.5] * 7) == pytest.approx(0.5)
    assert series_win_prob([1.0] * 3) == pytest.approx(1.0)
    assert series_win_prob([0.0] * 5) == pytest.approx(0.0)
    # constant p best-of-7 has a closed form: sum over losses before the 4th win
    p = 0.6
    closed = sum(comb(3 + l, l) * p**4 * (1 - p)**l for l in range(4))
    assert series_win_prob([p] * 7) == pytest.approx(closed)
