"""Chunk 7: one HistGradientBoostingClassifier on the same features/splits as
Chunk 6, tuned against the 2023 val slice only, evaluated on 2024-2025, with a
pre-committed val-gated rule deciding whether it beats logistic regression.

No calibration here (HistGB's mild overconfidence is Chunk 8's input, not a bug
to fix). No scaler — trees are scale-invariant, so standardizing buys nothing.

Leakage discipline: val and test are scored only, never fit. Fitting happens on
`tune` (during the grid search) and `train` (the single final refit). test is
not loaded until the final model exists.

Run:  python src/model_histgb.py [--force-winner {histgb|logreg}]
Deterministic: random_state=42 on every fit -> two runs produce byte-identical
preds and search CSVs.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from src.evaluate import compare, evaluate
from src.features_ml import FEATURE_COLUMNS, get_xy
from src.splits import TRAIN_SEASONS, load_split

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
MODELS_DIR = ROOT / "models"
BASELINES_CSV = EVAL_DIR / "baselines_test.csv"
LOGREG_CSV = EVAL_DIR / "logreg_test.csv"
LOGREG_MODEL = MODELS_DIR / "logreg.joblib"
SEARCH_CSV = EVAL_DIR / "histgb_search.csv"
HISTGB_CSV = EVAL_DIR / "histgb_test.csv"
PREDS_CSV = EVAL_DIR / "preds_histgb_test.csv"
DECISION_JSON = EVAL_DIR / "model_decision.json"
MODEL_PATH = MODELS_DIR / "histgb.joblib"
META_PATH = MODELS_DIR / "histgb_meta.json"

RANDOM_STATE = 42
NOISE_FLOOR = 0.002        # same width as Chunk 6; justified by ~2.4k val games
BIG_IMPROVEMENT = 0.02     # implausibly large val gain -> flag for leak review

# The deliberate grid. DO NOT expand it to chase a win — the tripwire below
# hashes this exact structure and fails if it is altered.
GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 1.0],
}
GRID_KEYS = ["learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"]
# Canonical fingerprint of the specified grid (tripwire reference).
EXPECTED_GRID_SIG = (
    ("l2_regularization", (0.0, 1.0)),
    ("learning_rate", (0.03, 0.05, 0.1)),
    ("max_iter", (100, 200, 400)),
    ("max_leaf_nodes", (15, 31, 63)),
)


def _grid_signature():
    return tuple(sorted((k, tuple(v)) for k, v in GRID.items()))


def _make_model(cfg):
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **cfg)


def run_search(tune_df, val_df):
    """Fit every grid config on `tune`, score log loss on `val`. Returns the
    full results DataFrame (deterministically ordered)."""
    X_tune, y_tune = get_xy(tune_df)
    X_val, y_val = get_xy(val_df)

    rows = []
    for lr, mi, mln, l2 in itertools.product(
            GRID["learning_rate"], GRID["max_iter"],
            GRID["max_leaf_nodes"], GRID["l2_regularization"]):
        cfg = {"learning_rate": lr, "max_iter": mi,
               "max_leaf_nodes": mln, "l2_regularization": l2}
        model = _make_model(cfg).fit(X_tune, y_tune)
        p = model.predict_proba(X_val)[:, 1]
        rows.append({**cfg, "val_log_loss": float(log_loss(y_val, p,
                                                           labels=[0, 1]))})
    df = pd.DataFrame(rows, columns=GRID_KEYS + ["val_log_loss"])
    # deterministic order: by the grid axes, so the CSV is byte-stable
    df = df.sort_values(GRID_KEYS, kind="stable").reset_index(drop=True)
    return df


def select_config(search_df):
    """Simplicity-biased selection (scheme b): among configs within
    ll_min + NOISE_FLOOR, pick the simplest by the fixed lexical preference —
    fewest max_iter, smallest max_leaf_nodes, largest l2, smallest
    learning_rate. Returns (cfg_final, histgb_val_ll, info dict)."""
    ll_min = float(search_df["val_log_loss"].min())
    argmin_row = search_df.loc[search_df["val_log_loss"].idxmin()]
    argmin_cfg = {k: _clean(argmin_row[k]) for k in GRID_KEYS}

    band = search_df[search_df["val_log_loss"] <= ll_min + NOISE_FLOOR].copy()
    # lexical simplicity: max_iter asc, max_leaf_nodes asc, l2 DESC, lr asc
    band = band.sort_values(
        by=["max_iter", "max_leaf_nodes", "l2_regularization", "learning_rate"],
        ascending=[True, True, False, True], kind="stable")
    sel = band.iloc[0]
    cfg_final = {k: _clean(sel[k]) for k in GRID_KEYS}
    histgb_val_ll = float(sel["val_log_loss"])
    info = {"ll_min": ll_min, "argmin_cfg": argmin_cfg,
            "band_size": int(len(band)),
            "differs_from_argmin": cfg_final != argmin_cfg}
    return cfg_final, histgb_val_ll, info


def _clean(v):
    """Cast a numpy scalar back to a plain int/float for JSON + sklearn."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return int(f) if f.is_integer() and v in (100, 200, 400, 15, 31, 63) \
            else f
    return v


def _logreg_val_ll(val_df):
    """logreg's val log loss — recomputed from the serialized pipeline (Chunk 6
    did not persist it to a file)."""
    pipe = joblib.load(LOGREG_MODEL)
    Xv, yv = get_xy(val_df)
    pv = pipe.predict_proba(Xv)[:, 1]
    return float(log_loss(yv, pv, labels=[0, 1]))


def _eval_row(name, subset, df, p):
    m = evaluate(df["home_win"].to_numpy(dtype=int), p, name)
    return {"name": name, "subset": subset, "n": m["n"],
            "accuracy": m["accuracy"], "log_loss": m["log_loss"],
            "brier": m["brier"], "auc": m["auc"],
            "mean_pred": m["mean_pred"], "base_rate": m["base_rate"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-winner", choices=["histgb", "logreg"],
                    default=None)
    args = ap.parse_args(argv)

    MODELS_DIR.mkdir(exist_ok=True)

    # --- tripwire: grid untouched ----------------------------------------
    if _grid_signature() != tuple(sorted(EXPECTED_GRID_SIG)):
        print("TRIPWIRE FAILURE: search grid was altered from the spec",
              file=sys.stderr)
        sys.exit(1)

    tune_df = load_split("tune")
    val_df = load_split("val")

    # --- search (fit on tune, score on val) ------------------------------
    search_df = run_search(tune_df, val_df)
    search_df.to_csv(SEARCH_CSV, index=False)

    cfg_final, histgb_val_ll, info = select_config(search_df)
    logreg_val_ll = _logreg_val_ll(val_df)
    improvement = logreg_val_ll - histgb_val_ll

    print("\n" + "=" * 70)
    print("SEARCH — HistGB val (2023) log loss, simplicity-biased selection")
    print("=" * 70)
    print(f"  grid size                 = {len(search_df)} configs")
    print(f"  ll_min                    = {info['ll_min']:.6f}")
    print(f"  config at ll_min (argmin) = {info['argmin_cfg']}")
    print(f"  tie band (<= ll_min+{NOISE_FLOOR}) = {info['band_size']} config(s)")
    print(f"  SELECTED cfg_final        = {cfg_final}")
    print(f"  histgb_val_ll (selected)  = {histgb_val_ll:.6f}")
    if info["differs_from_argmin"]:
        print("  NOTE: selected config differs from the raw argmin — this is "
              "the\n        simplicity-bias scheme working as intended, not a bug.")
    else:
        print("  (selected config == raw argmin; the simplest tie-band member "
              "was\n        also the lowest-loss one.)")
    print("=" * 70)

    # --- winner gate (pre-committed, decided on val) ---------------------
    auto_winner = "histgb" if improvement >= NOISE_FLOOR else "logreg"
    overridden = args.force_winner is not None
    winner = args.force_winner if overridden else auto_winner

    print("\n" + "=" * 70)
    print("WINNER GATE — decided on VAL, before test is touched")
    print("=" * 70)
    print(f"  logreg_val_ll             = {logreg_val_ll:.6f}")
    print(f"  histgb_val_ll             = {histgb_val_ll:.6f}")
    print(f"  improvement (lr - hgb)    = {improvement:.6f}")
    print(f"  threshold                 = {NOISE_FLOOR:.6f}")
    print(f"  automatic winner          = {auto_winner}")
    if overridden:
        print(f"  OVERRIDDEN via --force-winner -> winner = {winner}")
    else:
        print(f"  winner                    = {winner}")
    if improvement > BIG_IMPROVEMENT:
        print("\n  *** WARNING: val improvement exceeds "
              f"{BIG_IMPROVEMENT} — implausibly large for MLB game\n"
              "      prediction. This often means the tree exploits a feature "
              "the linear\n      model can't — frequently a subtle leak the "
              "sign audit missed. FLAGGED\n      FOR MANUAL REVIEW (not a "
              "failure).", file=sys.stderr)
    print("=" * 70)

    decision = {
        "logreg_val_ll": logreg_val_ll,
        "histgb_val_ll": histgb_val_ll,
        "improvement": improvement,
        "threshold": NOISE_FLOOR,
        "winner": winner,
        "overridden": overridden,
        "automatic_winner": auto_winner,
        "cfg_final": cfg_final,
    }
    DECISION_JSON.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # --- final fit on FULL train, once -----------------------------------
    train_df = load_split("train")
    X_train, y_train = get_xy(train_df)
    model = _make_model(cfg_final).fit(X_train, y_train)
    train_home_rate = float(y_train.mean())

    # --- test loaded only now; scored, never fit -------------------------
    test_df = load_split("test")
    X_test, y_test = get_xy(test_df)
    p_test = model.predict_proba(X_test)[:, 1]

    lc = test_df["low_confidence"].to_numpy(dtype=bool)
    hi_df = test_df[~lc]
    p_hi = p_test[~lc]

    row_all = _eval_row("histgb", "all", test_df, p_test)
    row_hi = _eval_row("histgb", "high_conf", hi_df, p_hi)

    # --- tripwires -------------------------------------------------------
    baselines = pd.read_csv(BASELINES_CSV)
    base_all_n = int(baselines[(baselines["name"] == "elo_replay")
                               & (baselines["subset"] == "all")]["n"].iloc[0])
    fails = []
    if row_all["accuracy"] >= 0.62:
        fails.append(f"histgb test accuracy {row_all['accuracy']:.4f} >= 0.62 "
                     f"(leakage suspected)")
    if not ((p_test > 0.0).all() and (p_test < 1.0).all()):
        fails.append("p_histgb outside open interval (0,1)")
    if row_all["n"] != base_all_n:
        fails.append(f"test row count {row_all['n']} != baseline/logreg row "
                     f"count {base_all_n}")
    if not DECISION_JSON.exists():
        fails.append("model_decision.json was not written")
    if fails:
        print("\nTRIPWIRE FAILURE:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    # --- write eval outputs ----------------------------------------------
    cols = ["name", "subset", "n", "accuracy", "log_loss", "brier", "auc",
            "mean_pred", "base_rate"]
    pd.DataFrame([row_all, row_hi])[cols].to_csv(HISTGB_CSV, index=False)

    preds = pd.DataFrame({
        "game_pk": test_df["game_pk"].to_numpy(),
        "date": test_df["date"].to_numpy(),
        "season": test_df["season"].to_numpy(),
        "home_win": test_df["home_win"].to_numpy(dtype=int),
        "p_histgb": p_test,
        "low_confidence": lc,
    })
    preds.to_csv(PREDS_CSV, index=False)

    # --- serialize -------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    meta = {
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
        "cfg_final": cfg_final,
        "train_seasons": list(TRAIN_SEASONS),
        "sklearn_version": sklearn.__version__,
        "train_home_win_rate": train_home_rate,
        "random_state": RANDOM_STATE,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- console summary: all four on subset=all -------------------------
    logreg_all = pd.read_csv(LOGREG_CSV)
    lr_all = logreg_all[logreg_all["subset"] == "all"].iloc[0]
    base_cmp = baselines[(baselines["subset"] == "all")
                         & baselines["name"].isin(["constant_home",
                                                   "elo_replay"])]
    keep = ["name", "n", "accuracy", "log_loss", "brier", "auc",
            "mean_pred", "base_rate"]
    rows_cmp = base_cmp[keep].to_dict("records")
    rows_cmp.append({k: (lr_all[k] if k != "name" else "logreg") for k in keep})
    rows_cmp.append({k: row_all[k] for k in keep})
    cmp = compare(rows_cmp)

    print("\n" + "=" * 74)
    print("COMBINED COMPARISON — subset=all, ascending log loss")
    print("=" * 74)
    print(cmp.to_string(index=False))
    print("=" * 74)

    print("\nhistgb vs logreg on TEST (subset=all), judged independently:")
    for metric, lower_better in (("log_loss", True), ("brier", True),
                                 ("accuracy", False)):
        hv, lv = row_all[metric], float(lr_all[metric])
        won = (hv < lv) if lower_better else (hv > lv)
        verb = "BEAT" if won else "did NOT beat"
        print(f"  {metric:<9}: histgb {hv:.4f} vs logreg {lv:.4f} "
              f"-> histgb {verb} logreg")

    print(f"\nGate winner: {winner}"
          f"{' (OVERRIDDEN)' if overridden else ' (automatic)'}")
    if auto_winner == "logreg":
        print("Honest finding: the boosted model did NOT clear the val gate — "
              "on held-out\n2023 data it did not measurably improve on the "
              "linear model. This is a\nlegitimate and expected result for MLB "
              "game prediction, where signal is\nthin and largely linear — not "
              "a failure.")
    else:
        print("Honest finding: boosting cleared the val gate, measurably "
              "improving on the\nlinear model on held-out 2023 data.")

    print("\nTo override the gate, re-run with --force-winner {histgb|logreg}.")

    print(f"\nArtifacts written:")
    for p in (SEARCH_CSV, HISTGB_CSV, PREDS_CSV, DECISION_JSON, MODEL_PATH,
              META_PATH):
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Phase-2 ProbabilityModel-compatible single-row scorer (NOT wired in Chunk 7)
# ---------------------------------------------------------------------------

_LOADED = {}


def _load():
    if "model" not in _LOADED:
        _LOADED["model"] = joblib.load(MODEL_PATH)
        _LOADED["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    return _LOADED["model"], _LOADED["meta"]


def predict_proba_home(feature_row: dict) -> float:
    """Home win probability for one game given a dict of feature values. Reads
    FEATURE_COLUMNS from the row in serialized order and applies the loaded
    model. Satisfies the phase-2 ProbabilityModel shape; intentionally not
    imported by simulate.py / matchup.py in this chunk."""
    model, meta = _load()
    cols = meta["FEATURE_COLUMNS"]
    x = np.array([[float(feature_row[c]) for c in cols]], dtype=float)
    return float(model.predict_proba(x)[0, 1])


if __name__ == "__main__":
    main()
