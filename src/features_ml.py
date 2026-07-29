"""Chunk 6, Deliverable 1: the shared ML feature matrix.

A tiny seam so this chunk and later ML chunks agree on exactly which columns
enter the model and in what order. The order is frozen here; downstream code
(serialized model metadata, coefficient audits) relies on it.

FEATURE_COLUMNS is the ordered list load_split() returns (minus the key/label
columns), with any zero / near-zero variance column removed. Phase 0 measured
every column's train-split variance well above 1e-12 (smallest ~2.4e-3, the
starter K-rates), so NOTHING is dropped in the current data. home_field is not
a candidate here at all — the Chunk 3 feature table omits it by construction
(constant offset; the model intercept carries it), so there is no constant-1
column to exclude. The drop machinery below is kept live and honest anyway:
if a future rebuild produced a degenerate column it would be removed and
reported rather than silently fed to the scaler.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.splits import load_split

# The ordered candidate list = load_split's feature columns (Phase 0 order).
# Kept as a literal so this module does not silently follow an upstream
# reordering without a code change and a re-audit.
_CANDIDATE_COLUMNS = [
    "home_form_15", "away_form_15",
    "home_run_diff_pg", "away_run_diff_pg",
    "home_rest_days", "away_rest_days",
    "home_sp_fip_60", "away_sp_fip_60",
    "home_sp_k_rate", "away_sp_k_rate",
    "away_tz_delta",
]

_VAR_FLOOR = 1e-12


def _select_columns(verbose=True):
    """Return the kept columns after dropping any with train-split variance
    below the floor. Prints what was dropped and why."""
    train = load_split("train")
    missing = [c for c in _CANDIDATE_COLUMNS if c not in train.columns]
    assert not missing, f"candidate columns absent from load_split: {missing}"

    kept, dropped = [], []
    for c in _CANDIDATE_COLUMNS:
        v = float(np.var(train[c].to_numpy(dtype=float)))
        if v < _VAR_FLOOR:
            dropped.append((c, v))
        else:
            kept.append(c)
    if verbose:
        if dropped:
            for c, v in dropped:
                print(f"[features_ml] DROPPED {c!r}: train variance {v:.3g} "
                      f"< {_VAR_FLOOR:g} (degenerate; intercept absorbs any "
                      f"constant offset)")
        else:
            print(f"[features_ml] no columns dropped — all "
                  f"{len(_CANDIDATE_COLUMNS)} candidates exceed the "
                  f"{_VAR_FLOOR:g} variance floor on train")
    return kept


FEATURE_COLUMNS = _select_columns(verbose=False)

# Chunk 10 (evaluation-only): the deployed models and the whole serving path
# (live_features, predict_game, the calibrated joblib, the skew fixture) are
# frozen on the 11-column FEATURE_COLUMNS above. park_factor is a game-level
# feature that does not fit the home/away split those modules assume, and the
# with-park model is NOT deployed this chunk — so it lives here as a SEPARATE
# list used only by the Chunk 10 retrain/compare. Serving never sees it.
FEATURE_COLUMNS_WITHPARK = FEATURE_COLUMNS + ["park_factor"]


def get_xy(df, columns=None):
    """(X, y) for a split frame: X selects `columns` (default FEATURE_COLUMNS)
    in fixed order as a float ndarray, y is home_win as an int ndarray. Pass
    FEATURE_COLUMNS_WITHPARK for the Chunk 10 with-park models."""
    cols = list(columns) if columns is not None else FEATURE_COLUMNS
    X = df[cols].to_numpy(dtype=float)
    y = df["home_win"].to_numpy(dtype=int)
    assert not np.isnan(X).any(), "NaN in feature matrix X"
    assert X.shape[1] == len(cols), \
        f"X has {X.shape[1]} cols, expected {len(cols)}"
    return X, y


if __name__ == "__main__":
    _select_columns(verbose=True)
    print(f"[features_ml] FEATURE_COLUMNS ({len(FEATURE_COLUMNS)}):")
    for i, c in enumerate(FEATURE_COLUMNS):
        print(f"   [{i:>2}] {c}")
