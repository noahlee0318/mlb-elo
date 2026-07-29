"""The model seam for the season simulator.

ProbabilityModel is the ONLY interface the simulation engine knows about:
models produce home-win probabilities, the engine consumes them. All Elo
knowledge lives in EloProbabilityModel below; a future model (pitchers,
lineups, park factors, ...) replaces that adapter and nothing in
simulate.py / playoffs.py changes.
"""

from typing import Protocol

import numpy as np
import pandas as pd

from src.elo import BASE, expected_home


class ProbabilityModel(Protocol):
    def schedule_probs(self, games: pd.DataFrame,
                       n_sims: int | None = None) -> np.ndarray:
        """Home-team win probabilities for a set of scheduled games.

        games: one row per game, with at minimum home_id, away_id,
        game_date. Extra context columns may be present; simpler models
        ignore them.

        Returns shape (G,) — one fixed probability per game — or shape
        (n_sims, G) for models that draw team-strength uncertainty per
        simulation. The engine broadcasts over either shape.
        """
        ...

    def matchup_prob(self, home_id: int, away_id: int,
                     context: dict | None = None) -> float:
        """Home win probability for a hypothetical game not on the
        schedule (playoff series, where opponents aren't known ahead)."""
        ...


class EloProbabilityModel:
    """Thin adapter translating (home_id, away_id) into the existing Elo
    call using a ratings snapshot. Ratings are copied at construction and
    never change afterwards — the simulator must see a frozen model."""

    def __init__(self, ratings: dict[int, float]):
        self._ratings = dict(ratings)

    def _prob(self, home_id, away_id):
        r = self._ratings
        return expected_home(r.get(int(home_id), BASE), r.get(int(away_id), BASE))

    def schedule_probs(self, games, n_sims=None):
        return np.array([self._prob(h, a) for h, a in
                         zip(games["home_id"], games["away_id"])], dtype=float)

    def matchup_prob(self, home_id, away_id, context=None):
        if context and context.get("neutral"):
            r = self._ratings
            return float(expected_home(r.get(int(home_id), BASE),
                                       r.get(int(away_id), BASE), hfa=0))
        return float(self._prob(home_id, away_id))
