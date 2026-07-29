"""Weakness 2, Deliverable 1: ML ProbabilityModel adapters for the season sim
and matchup tool.

Implements the ProbabilityModel protocol (src/prob_model.py) — schedule_probs
+ matchup_prob — over the trained logreg / calibrated-histgb models, so the
UNCHANGED simulation engine can drive them exactly as it drives Elo. This module
NEVER imports elo.py and NEVER edits the engine; a model swap is a new adapter
plus a one-line change at the construction point (app.py).

Static team strength (inherited, not chosen — Phase 2 report §4.1/§8): each
team's feature components are built ONCE, as of today, and applied to all its
remaining games; a simulated result never updates them. So schedule_probs builds
features per (team, side) — ~60 build_live_features calls, cached — not once per
remaining game, and predicts the whole grid in one model call.

Two deliberate differences from the live single-game UI, both documented:
  * PITCHING is rotation-average: the sim can't know each future game's starter,
    so a team's starter FIP/K-rate is the mean over its recent distinct starters'
    trailing-60-IP stats, computed by REUSING the builder's PitcherIndex (no
    forked window logic). The live UI uses the actual announced starter.
  * away_tz_delta is a game-level travel term that depends on the specific host
    park; under one-vector-per-team caching it is set to 0 (no-travel). It is a
    near-zero-signal feature (logreg |coef| ~0.007), so this is immaterial.

Determinism: build_live_features is pure, rotation-average is a fixed reduction,
predict is deterministic -> same as_of_date + data => identical probabilities.
"""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.features import PitcherIndex, canonical_order
from src.features_ml import FEATURE_COLUMNS
from src.games_data import load_games
from src.live_features import build_live_features
from src.schedule import todays_date_str

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PITCHER_CSV = ROOT / "data" / "pitcher_games.csv"

ROTATION_SIZE = 5          # recent distinct starters blended into the rotation avg
NEUTRAL_TZ = 0.0           # away_tz_delta for sim games (see module docstring)

# where each side's own columns live in the 11-feature vector
_HOME_COLS = [c for c in FEATURE_COLUMNS if c.startswith("home_")]
_AWAY_COLS = [c for c in FEATURE_COLUMNS if c.startswith("away_")]

_MODEL_FILES = {
    "logreg": ("logreg.joblib", "logreg_meta.json"),
    "histgb_cal": ("histgb_calibrated.joblib", "histgb_calibrated_meta.json"),
}


def _load_scorer(model_key):
    """(sklearn_estimator, feature_columns) for a deployed model. The estimator
    predicts a whole (G, n_feat) matrix in one call; its meta pins the column
    order it was trained on (11 features — the with-park model is not used here)."""
    if model_key not in _MODEL_FILES:
        raise ValueError(f"model must be one of {list(_MODEL_FILES)}, "
                         f"got {model_key!r}")
    jbl, meta_f = _MODEL_FILES[model_key]
    est = joblib.load(MODELS / jbl)
    cols = json.loads((MODELS / meta_f).read_text())["FEATURE_COLUMNS"]
    assert cols == FEATURE_COLUMNS, "deployed model column order drifted"
    return est, cols


# ---------------------------------------------------------------------------
# Feature cache — built once, as of today, shared across models for comparison
# ---------------------------------------------------------------------------

class TeamFeatureCache:
    """Per-team, as-of-today feature components under static strength. Lazily
    builds each team's home-side and away-side vectors (form / run-diff / rest
    from build_live_features; starter FIP/K-rate from the rotation average) and
    caches them. Model-independent, so one cache serves the 3-model comparison."""

    def __init__(self, as_of_date=None):
        self.as_of = as_of_date or todays_date_str()
        games = load_games()
        co = canonical_order(games)
        self._order_of_pk = dict(zip(co["game_pk"].astype(int), co["game_order"]))
        date_of_order = dict(zip(co["game_order"], co["date"]))
        self._today_order = int(co["game_order"].max()) + 1   # after all history
        logs = pd.read_csv(PITCHER_CSV, low_memory=False)
        logs = logs[logs["game_type"] == "R"].copy()
        self._logs = logs
        self._pindex = PitcherIndex(logs, self._order_of_pk, date_of_order)
        self._team_ids = sorted(set(games["home_team_id"].astype(int))
                                | set(games["away_team_id"].astype(int)))
        self._comp = {}          # team_id -> {feature: value}

    def _ref_opp(self, team_id):
        for t in self._team_ids:
            if t != team_id:
                return t
        return team_id

    def _rotation_avg_sp(self, team_id):
        """(fip, krate) averaged over the team's recent distinct starters'
        trailing-60-IP stats, via the builder's PitcherIndex (reused, not
        reimplemented). Falls back to the as-of league-average if the team has
        no usable recent starter."""
        st = self._logs[(self._logs["team_id"] == team_id)
                        & (self._logs["is_starter"].astype(str).isin(["True", "true"]))
                        & (self._logs["date"] < self.as_of)]
        st = st.sort_values("date")
        recent = list(dict.fromkeys(st["pitcher_id"].astype(int).tolist()[::-1]))
        recent = recent[:ROTATION_SIZE]
        fips, krs = [], []
        for pid in recent:
            fip, kr, _outs, _enough = self._pindex.trailing(pid, self._today_order)
            if fip is not None and np.isfinite(fip):
                fips.append(fip)
            if kr is not None and np.isfinite(kr):
                krs.append(kr)
        if not fips or not krs:
            lf, lk = self._pindex.league_asof(self._today_order, self.as_of)
            return (float(lf) if lf is not None else np.nan,
                    float(lk) if lk is not None else np.nan)
        return float(np.mean(fips)), float(np.mean(krs))

    def components(self, team_id):
        """{the team's home-side and away-side feature columns}. Cached."""
        team_id = int(team_id)
        if team_id in self._comp:
            return self._comp[team_id]
        opp = self._ref_opp(team_id)
        # form / run-diff / rest are opponent-independent; two calls extract the
        # home-labelled and away-labelled versions from the shared builder.
        h = build_live_features(team_id, opp, self.as_of, "home", team_pp_id=None)
        a = build_live_features(team_id, opp, self.as_of, "away", team_pp_id=None)
        fip, kr = self._rotation_avg_sp(team_id)
        comp = {
            "home_form_15": h["home_form_15"],
            "home_run_diff_pg": h["home_run_diff_pg"],
            "home_rest_days": h["home_rest_days"],
            "home_sp_fip_60": fip, "home_sp_k_rate": kr,
            "away_form_15": a["away_form_15"],
            "away_run_diff_pg": a["away_run_diff_pg"],
            "away_rest_days": a["away_rest_days"],
            "away_sp_fip_60": fip, "away_sp_k_rate": kr,
            "games_played": int(h["games_played"]),
        }
        self._comp[team_id] = comp
        return comp

    def prewarm(self, team_ids):
        for t in team_ids:
            self.components(t)


# ---------------------------------------------------------------------------
# The ProbabilityModel adapter
# ---------------------------------------------------------------------------

class MLProbabilityModel:
    """ProbabilityModel over a deployed ML model + the as-of-today feature cache.
    schedule_probs returns (G,) — matching EloProbabilityModel so the engine
    needs no branching. Never imports elo.py; never touches the engine."""

    def __init__(self, model_key="histgb_cal", as_of_date=None,
                 feature_cache=None):
        self.model_key = model_key
        self._est, self._cols = _load_scorer(model_key)
        self.cache = feature_cache or TeamFeatureCache(as_of_date)

    def _row(self, home_id, away_id):
        """The 11-feature vector for one game, from cached team components."""
        h = self.cache.components(home_id)
        a = self.cache.components(away_id)
        row = {c: h[c] for c in _HOME_COLS}
        # away_tz_delta is set below (game-level, not a cached team component)
        row.update({c: a[c] for c in _AWAY_COLS if c != "away_tz_delta"})
        row["away_tz_delta"] = NEUTRAL_TZ
        return row

    def _predict_matrix(self, rows):
        X = np.array([[float(r[c]) for c in self._cols] for r in rows],
                     dtype=float)
        p = self._est.predict_proba(X)[:, 1]
        return np.clip(p, 1e-9, 1 - 1e-9)      # guard the open interval (0,1)

    def schedule_probs(self, games, n_sims=None):
        """(G,) home-win probabilities for the remaining games, one model call.
        Shape matches EloProbabilityModel; the engine tiles across n_sims."""
        homes = games["home_id"].astype(int).tolist()
        aways = games["away_id"].astype(int).tolist()
        # prewarm the cache (all distinct teams) before assembling the grid
        self.cache.prewarm(set(homes) | set(aways))
        rows = [self._row(h, a) for h, a in zip(homes, aways)]
        if not rows:
            return np.zeros(0, dtype=float)
        return self._predict_matrix(rows)

    def matchup_prob(self, home_id, away_id, context=None):
        """Home win probability for a hypothetical matchup. A neutral single
        game is symmetrized (no host advantage), since the ML model has no HFA
        knob to zero out the way Elo does."""
        if context and context.get("neutral"):
            p_home = self._predict_matrix([self._row(home_id, away_id)])[0]
            p_away = self._predict_matrix([self._row(away_id, home_id)])[0]
            return float(0.5 * (p_home + (1.0 - p_away)))
        return float(self._predict_matrix([self._row(home_id, away_id)])[0])


def build_models(model_keys, as_of_date=None):
    """Construct several MLProbabilityModels sharing ONE feature cache — so the
    ~60 build_live_features calls happen once for the whole comparison, and only
    the predict step repeats per model. Elo is constructed separately by the
    caller (it needs no features)."""
    cache = TeamFeatureCache(as_of_date)
    return {k: MLProbabilityModel(k, feature_cache=cache) for k in model_keys}
