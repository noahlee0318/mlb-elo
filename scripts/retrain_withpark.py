"""Chunk 10, Deliverable 4 — retrain the ML models WITH park_factor on the
EXACT same splits, hyperparameters, and test rows as the without-park models,
and compare. Evaluation only: nothing here is deployed.

The frozen hyperparameters (C_final, cfg_final) are READ from the existing
without-park meta files, so "same hyperparameters" is guaranteed, not retyped.
No re-tuning — re-tuning would confound the feature's effect with a
hyperparameter change. Elo/constant are unchanged (they use no features).

Writes data/eval/*_test_withpark.csv (never overwrites the without-park files).
Inverted tripwire: a LARGE gain is the alarm, not a win — for a feature that
should cancel in win probability, > 1.0 pp accuracy or > 0.005 log-loss
improvement signals leakage/bug and is flagged loudly for manual review.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate
from src.features_ml import (FEATURE_COLUMNS, FEATURE_COLUMNS_WITHPARK, get_xy)
from src.splits import load_split

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
MODELS = ROOT / "models"
RANDOM_STATE = 42
MAX_ITER_LR = 5000

# inverted tripwire thresholds (a gain beyond these = alarm)
ACC_ALARM = 0.010          # 1.0 percentage point
LL_ALARM = 0.005


def _row(name, subset, df, p):
    m = evaluate(df["home_win"].to_numpy(int), p, name)
    return {"name": name, "subset": subset, "n": m["n"], "accuracy": m["accuracy"],
            "log_loss": m["log_loss"], "brier": m["brier"], "auc": m["auc"],
            "mean_pred": m["mean_pred"], "base_rate": m["base_rate"]}


def main():
    # frozen hyperparameters from the without-park metas (not retyped)
    C_final = json.loads((MODELS / "logreg_meta.json").read_text())["C_final"]
    cfg_final = json.loads((MODELS / "histgb_meta.json").read_text())["cfg_final"]
    cal_meta = json.loads((MODELS / "histgb_calibrated_meta.json").read_text())
    print("Frozen hyperparameters (read from without-park metas):")
    print(f"  logreg C_final = {C_final}")
    print(f"  histgb cfg_final = {cfg_final}")
    print(f"  calibrated: base on {cal_meta['base_trained_on']}, "
          f"{cal_meta['method']} on {cal_meta['calibrator_fit_on']}")
    print(f"  with-park feature count = {len(FEATURE_COLUMNS_WITHPARK)} "
          f"(without = {len(FEATURE_COLUMNS)})")

    tune = load_split("tune"); val = load_split("val")
    train = load_split("train"); test = load_split("test")

    COLS = FEATURE_COLUMNS_WITHPARK
    Xtr, ytr = get_xy(train, COLS)
    Xte, yte = get_xy(test, COLS)
    Xtu, ytu = get_xy(tune, COLS)
    Xva, yva = get_xy(val, COLS)

    # test-row identity vs without-park (tripwire)
    wp_test_pks = set(pd.read_csv(EVAL / "preds_logreg_test.csv")["game_pk"])
    this_test_pks = set(test["game_pk"])
    fails = []
    if wp_test_pks != this_test_pks:
        fails.append("with-park test rows differ from without-park test rows")
    print(f"\nTest rows identical to without-park: "
          f"{wp_test_pks == this_test_pks} (n={len(this_test_pks)})")

    results = {}          # name -> (row_all, extra)

    # --- logreg ----------------------------------------------------------
    lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(C=C_final, max_iter=MAX_ITER_LR,
                                              random_state=RANDOM_STATE))]).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    results["logreg"] = _row("logreg", "all", test, p)
    coef = lr.named_steps["clf"].coef_.ravel()
    pf_idx = COLS.index("park_factor")
    park_coef = float(coef[pf_idx])

    # --- histgb ----------------------------------------------------------
    hg = HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                        **cfg_final).fit(Xtr, ytr)
    results["histgb"] = _row("histgb", "all", test, hg.predict_proba(Xte)[:, 1])

    # --- histgb_calibrated (base on tune, sigmoid on val; FrozenEstimator) --
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                          **cfg_final).fit(Xtu, ytu)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(Xva, yva)
    results["histgb_calibrated"] = _row("histgb_calibrated", "all", test,
                                        cal.predict_proba(Xte)[:, 1])

    # --- write with-park eval files --------------------------------------
    cols = ["name", "subset", "n", "accuracy", "log_loss", "brier", "auc",
            "mean_pred", "base_rate"]
    name_to_file = {"logreg": "logreg_test_withpark.csv",
                    "histgb": "histgb_test_withpark.csv",
                    "histgb_calibrated": "histgb_calibrated_test_withpark.csv"}
    for nm, r in results.items():
        pd.DataFrame([r])[cols].to_csv(EVAL / name_to_file[nm], index=False)

    # --- comparison table (without vs with park, identical test rows) -----
    def without(nm):
        f = {"logreg": "logreg_test.csv", "histgb": "histgb_test.csv",
             "histgb_calibrated": "histgb_calibrated_test.csv"}[nm]
        d = pd.read_csv(EVAL / f)
        return d[d["subset"] == "all"].iloc[0]

    print("\n" + "=" * 92)
    print("WITHOUT-PARK vs WITH-PARK on IDENTICAL test rows (subset=all). "
          "Delta = with - without.")
    print("=" * 92)
    hdr = f"  {'model':<18}{'metric':<10}{'without':>11}{'with':>11}{'delta':>11}"
    print(hdr); print("  " + "-" * 88)
    alarms = []
    for nm in ("logreg", "histgb", "histgb_calibrated"):
        wo = without(nm); wp = results[nm]
        for metric in ("log_loss", "brier", "accuracy", "auc"):
            wov = float(wo[metric]); wpv = float(wp[metric])
            d = wpv - wov
            print(f"  {nm:<18}{metric:<10}{wov:>11.5f}{wpv:>11.5f}{d:>+11.5f}")
            # inverted alarm: IMPROVEMENT beyond threshold
            if metric == "accuracy" and (wpv - wov) > ACC_ALARM:
                alarms.append(f"{nm} accuracy improved {wpv-wov:+.4f} > {ACC_ALARM}")
            if metric == "log_loss" and (wov - wpv) > LL_ALARM:
                alarms.append(f"{nm} log_loss improved {wov-wpv:+.4f} > {LL_ALARM}")
        print("  " + "-" * 88)

    print(f"\nlogreg park_factor STANDARDIZED coefficient: {park_coef:+.5f}")
    print(f"  (|other standardized coefs| range: "
          f"{np.abs(coef[:-1]).min():.5f}..{np.abs(coef[:-1]).max():.5f}; "
          f"park_factor |coef| rank: "
          f"{int(np.sum(np.abs(coef) >= abs(park_coef)))}/{len(coef)} by magnitude)")

    if alarms:
        print("\n*** INVERTED TRIPWIRE — LARGE GAIN FLAGGED FOR MANUAL REVIEW ***",
              file=sys.stderr)
        for a in alarms:
            print(f"    {a}", file=sys.stderr)
        print("    A feature expected to cancel in win probability should NOT "
              "improve the model this much — investigate as a probable "
              "leak/bug before trusting.", file=sys.stderr)
        fails.append("inverted tripwire: implausibly large improvement")

    if fails:
        print("\nTRIPWIRE FAILURE:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("\nEval files written:", ", ".join(name_to_file.values()))


if __name__ == "__main__":
    main()
