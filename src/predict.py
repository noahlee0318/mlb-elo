"""Elo and naive-baseline home-win probabilities for a date's slate,
side by side, logged to data/predictions.csv for later scoring.

Run from the project root:  python -m src.predict

Chunk 9 adds `predict_game` below: one function that serves Elo / logistic
regression / calibrated gradient boosting from ALREADY-BUILT feature states
(see src/live_features.py). It consumes built features, it never builds them —
so exactly one feature builder (the Chunk 3 code, reused by live_features)
feeds both training and serving. The Chunk-5 `predict_for_date` slate logger
above is untouched.
"""

from pathlib import Path

import pandas as pd

from src.build_ratings import build_ratings
from src.elo import BASE, HFA, expected_home
from src.features_ml import FEATURE_COLUMNS
from src.schedule import fetch_slate, todays_date_str

# home-/away-side split of the ML feature vector (away owns away_tz_delta)
_HOME_COLS = [c for c in FEATURE_COLUMNS if c.startswith("home_")]
_AWAY_COLS = [c for c in FEATURE_COLUMNS if c.startswith("away_")]
ML_MODELS = ("logreg", "histgb_cal")
ALL_MODELS = ("elo",) + ML_MODELS
COLD_START_MIN_GAMES = 15

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"

# League home-win rate; "always pick home" is just this thresholded at 0.5.
BASELINE_PROB_HOME = 0.540

PRED_COLUMNS = ["date", "game_id", "home_id", "away_id",
                "elo_prob_home", "baseline_prob_home", "home_won"]


def predict_for_date(date_str=None, log=True, ratings=None):
    """Both probabilities for every game on the slate; logs them by default.

    ratings=None replays the unified Elo window from scratch (build_ratings);
    pass a {team_id: rating} dict (e.g. the data/ratings.csv snapshot) to skip
    the replay."""
    date_str = date_str or todays_date_str()
    if ratings is None:
        ratings, _names = build_ratings()
    preds = []
    for g in fetch_slate(date_str):
        elo_p = expected_home(ratings.get(g["home_id"], BASE),
                              ratings.get(g["away_id"], BASE))
        assert 0.0 <= elo_p <= 1.0
        preds.append({**g, "date": date_str,
                      "elo_prob_home": elo_p,
                      "baseline_prob_home": BASELINE_PROB_HOME})
    if log and preds:
        try:
            log_predictions(preds)
        except OSError:
            # OneDrive can briefly lock predictions.csv mid-sync; showing
            # the slate matters more than logging this run's predictions
            pass
    return preds


def log_predictions(preds):
    """Append to predictions.csv, deduped on game_id so re-running the same
    day never double-logs. home_won stays blank here; score.py fills it
    once games go Final. Returns how many new rows were written."""
    new = pd.DataFrame([{c: p.get(c, pd.NA) for c in PRED_COLUMNS} for p in preds],
                       columns=PRED_COLUMNS)
    if PREDICTIONS_CSV.exists():
        old = pd.read_csv(PREDICTIONS_CSV)
        new = new[~new["game_id"].isin(set(old["game_id"]))]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_CSV, index=False)
    return len(new)


# ---------------------------------------------------------------------------
# Chunk 9 — predict_game over built feature states
# ---------------------------------------------------------------------------

# lazily-loaded model scorers, so importing this module stays cheap and does
# not require the joblibs to exist until a model is actually requested
_SCORERS = {}


def _scorer(model):
    """The predict_proba_home(feature_row)->float callable for an ML model."""
    if model not in _SCORERS:
        if model == "logreg":
            from src.model_logreg import predict_proba_home as f
        elif model == "histgb_cal":
            from src.calibration import predict_proba_home as f
        else:
            raise ValueError(f"no scorer for model {model!r}")
        _SCORERS[model] = f
    return _SCORERS[model]


def _assemble_features(home_state, away_state):
    """The 11-column FEATURE_COLUMNS dict from the two side-states. This is a
    MERGE of already-built values, not a feature computation."""
    row = {}
    for c in _HOME_COLS:
        row[c] = float(home_state[c])
    for c in _AWAY_COLS:
        row[c] = float(away_state[c])
    assert set(row) == set(FEATURE_COLUMNS), "assembled features != FEATURE_COLUMNS"
    return row


def predict_game(home_state, away_state, context=None, model="histgb_cal"):
    """Home/away win probability for one game from already-built feature states.

    home_state / away_state: dicts carrying this side's FEATURE_COLUMNS values,
    `games_played` (int), and `elo_rating` (float) — as produced by
    live_features.build_game_state, or hand-constructed for user input.
    context: optional dict; `low_confidence_reasons` is read from it if present.
    model: one of {"elo", "logreg", "histgb_cal"}.

    Cold-start rule: if EITHER team has games_played < 15 the ML vector is
    undefined on a side, so logreg/histgb_cal are NOT run — the result falls
    back to Elo (always defined) with a populated fallback_reason. Returns a
    dict: requested_model, served_model, p_home, p_away, low_confidence,
    low_confidence_reasons, fallback_reason, games_played_home,
    games_played_away.
    """
    if model not in ALL_MODELS:
        raise ValueError(f"model must be one of {ALL_MODELS}, got {model!r}")
    context = context or {}
    gp_home = int(home_state["games_played"])
    gp_away = int(away_state["games_played"])
    reasons = list(context.get("low_confidence_reasons", []))
    low_conf = bool(context.get("low_confidence", bool(reasons)))

    requested = model
    fallback_reason = None
    served = model

    # cold-start gate: ML models undefined if either side < 15 games
    if model in ML_MODELS and min(gp_home, gp_away) < COLD_START_MIN_GAMES:
        team = "home" if gp_home < COLD_START_MIN_GAMES else "away"
        n = min(gp_home, gp_away)
        served = "elo"
        fallback_reason = f"cold_start: {team} has {n}<{COLD_START_MIN_GAMES} games"

    if served == "elo":
        r_home = home_state.get("elo_rating")
        r_away = away_state.get("elo_rating")
        if r_home is None or r_away is None:
            raise ValueError("elo path needs elo_rating in both states "
                             "(pass elo_ratings to build_game_state)")
        p_home = float(expected_home(float(r_home), float(r_away)))
    else:
        row = _assemble_features(home_state, away_state)
        p_home = float(_scorer(served)(row))

    assert 0.0 < p_home < 1.0, f"p_home {p_home} outside (0,1) for {served}"
    return {
        "requested_model": requested,
        "served_model": served,
        "p_home": p_home,
        "p_away": 1.0 - p_home,
        "low_confidence": low_conf,
        "low_confidence_reasons": reasons,
        "fallback_reason": fallback_reason,
        "games_played_home": gp_home,
        "games_played_away": gp_away,
    }


def predict_all_models(home_state, away_state, context=None):
    """Run all three models for the same game (the UI shows them side by side).
    Returns {model: predict_game(...result...)}."""
    return {m: predict_game(home_state, away_state, context, model=m)
            for m in ALL_MODELS}


# NOTE: the Chunk 9 shape-only MLProbabilityModel stub that lived here was
# removed in Weakness 2 — src/ml_prob_model.py now provides the real,
# batched/cached ProbabilityModel adapter the simulator uses.


if __name__ == "__main__":
    rows = predict_for_date()
    if not rows:
        print("No regular-season games today (off-day or All-Star break) — nothing logged.")
    for g in rows:
        print(f'{g["away_name"]} @ {g["home_name"]}:  '
              f'Elo P(home)={g["elo_prob_home"]:.3f}   baseline={g["baseline_prob_home"]:.3f}')
