"""Chunk 6, Deliverable 2 & 3: train ONE logistic regression, evaluate it on
the 2024-2025 test set against the Chunk 5 baselines, and audit its
coefficient signs against baseball sense.

Leakage discipline (the one rule that matters here): the StandardScaler is fit
exactly once per model, on the rows that model is fit on, and applied unchanged
everywhere else. Tuning fits on `tune` and scores `val`; the final model refits
the scaler on full `train` and transforms `test`. The scaler+classifier are
bundled in a single sklearn Pipeline so the fit boundary cannot be crossed by
accident, and `.fit` is never called on anything derived from `test`.

Run:  python src/model_logreg.py
Deterministic: LogisticRegression(random_state=42) with the default lbfgs
solver is deterministic, so two runs produce byte-identical predictions.
"""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import compare, evaluate
from src.features_ml import FEATURE_COLUMNS, get_xy
from src.splits import TRAIN_SEASONS, load_split

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
MODELS_DIR = ROOT / "models"
BASELINES_CSV = EVAL_DIR / "baselines_test.csv"
LOGREG_CSV = EVAL_DIR / "logreg_test.csv"
PREDS_CSV = EVAL_DIR / "preds_logreg_test.csv"
MODEL_PATH = MODELS_DIR / "logreg.joblib"
META_PATH = MODELS_DIR / "logreg_meta.json"

RANDOM_STATE = 42
MAX_ITER = 5000
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
NOISE_FLOOR = 0.002        # fixed; justified for ~2k-2.5k val games. Do not tune.

# Asserted coefficient signs (Deliverable 3). Concept -> (column, expected
# sign). Lower FIP is better pitching, so a lower home FIP should RAISE the
# home win prob => the coefficient on raw (pre-standardization sign-preserving)
# home FIP is negative; away FIP is the mirror. A positive home FIP coefficient
# means a swapped column, which is why this fails loudly.
ASSERTED_SIGNS = {
    "home_form_15": +1,
    "away_form_15": -1,
    "home_run_diff_pg": +1,
    "away_run_diff_pg": -1,
    "home_sp_fip_60": -1,
    "away_sp_fip_60": +1,
    "home_sp_k_rate": +1,
    "away_sp_k_rate": -1,
}
# Report-only (rest small / possibly negative; travel direction ambiguous).
REPORT_ONLY = ["home_rest_days", "away_rest_days", "away_tz_delta"]


def _make_pipeline(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=MAX_ITER,
                                   random_state=RANDOM_STATE)),
    ])


def tune_C(tune_df, val_df):
    """Choose C by held-out (val) log loss, gated on the fixed noise floor.
    Returns (C_final, table_rows)."""
    X_tune, y_tune = get_xy(tune_df)
    X_val, y_val = get_xy(val_df)

    def val_ll(C):
        pipe = _make_pipeline(C).fit(X_tune, y_tune)          # fit on TUNE only
        p = pipe.predict_proba(X_val)[:, 1]
        return float(log_loss(y_val, p, labels=[0, 1]))

    ll_default = val_ll(1.0)
    rows = [(C, val_ll(C)) for C in C_GRID]
    C_best, ll_best = min(rows, key=lambda r: r[1])
    improvement = ll_default - ll_best
    took_tuned = improvement >= NOISE_FLOOR
    C_final = C_best if took_tuned else 1.0

    print("\n" + "=" * 68)
    print("TUNING — val (2023) log loss per C, pipeline fit on `tune` only")
    print("=" * 68)
    print(f"  {'C':>8}   {'val_log_loss':>14}")
    for C, ll in rows:
        star = "  *best" if (C == C_best and ll == ll_best) else ""
        print(f"  {C:>8}   {ll:>14.6f}{star}")
    print("-" * 68)
    print(f"  ll_default (C=1.0)        = {ll_default:.6f}")
    print(f"  ll_best    (C={C_best})   = {ll_best:.6f}")
    print(f"  improvement               = {improvement:.6f}")
    print(f"  noise-floor threshold     = {NOISE_FLOOR:.6f}")
    if took_tuned:
        print(f"  BRANCH: improvement >= threshold -> C_final = C_best = {C_final}")
    else:
        print(f"  BRANCH: improvement <  threshold -> C_final = 1.0 "
              f"(tuning below noise floor)")
    print("=" * 68)
    return C_final, rows


def _eval_row(name, subset, df, p):
    """A metric dict shaped to the baselines_test.csv schema."""
    m = evaluate(df["home_win"].to_numpy(dtype=int), p, name)
    return {"name": name, "subset": subset, "n": m["n"],
            "accuracy": m["accuracy"], "log_loss": m["log_loss"],
            "brier": m["brier"], "auc": m["auc"],
            "mean_pred": m["mean_pred"], "base_rate": m["base_rate"]}


def _univariate_signs(X_train, y_train):
    """Standardized single-feature logreg coefficient per column, fit on the
    same rows the final model uses. The univariate sign is the collinearity-
    ROBUST swap-detector: a swapped/mis-derived column flips it regardless of
    which other features are present. (The multivariate PARTIAL coefficient is
    not a reliable swap-detector when two asserted features are collinear —
    e.g. home_form_15 and home_run_diff_pg at r~0.59, which drags home_form's
    partial coefficient slightly negative even though the column is correct.)"""
    uni = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        x = X_train[:, [i]]
        pipe = _make_pipeline(1.0).fit(x, y_train)
        uni[col] = float(pipe.named_steps["clf"].coef_.ravel()[0])
    return uni


def audit_signs(pipe, X_train, y_train):
    """Deliverable 3: standardized-coefficient sign audit.

    Reports the FINAL model's multivariate standardized coefficients sorted by
    |value| (directly comparable because features were standardized). ASSERTS
    on each feature's UNIVARIATE standardized sign — the collinearity-robust
    swap-detector (see _univariate_signs). Fails loudly on any asserted
    univariate mismatch, since that genuinely indicates a swapped/mis-derived
    column. A row where the multivariate partial sign disagrees with the
    univariate sign is flagged 'collin' — expected for correlated features,
    not a failure. Returns the coefficient table (list of dicts)."""
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    coef_std = clf.coef_.ravel()          # multivariate, per standardized feat
    scale = scaler.scale_
    uni = _univariate_signs(X_train, y_train)

    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        rows.append({"feature": col, "coef_std": float(coef_std[i]),
                     "uni_coef": uni[col], "abs": abs(float(coef_std[i])),
                     "scale": float(scale[i]),
                     "expected": ASSERTED_SIGNS.get(col),
                     "asserted": col in ASSERTED_SIGNS})
    rows.sort(key=lambda r: r["abs"], reverse=True)

    print("\n" + "=" * 86)
    print("COEFFICIENT SIGN AUDIT")
    print("  reported: FINAL multivariate std coef (sorted by |value|)")
    print("  asserted on: UNIVARIATE std sign (collinearity-robust swap check)")
    print("=" * 86)
    print(f"  {'feature':<20}{'multivar':>10}{'univar':>10}{'m_sign':>7}"
          f"{'u_sign':>7}{'expect':>8}{'assert':>8}{'result':>9}")
    print("-" * 86)
    failures = []
    for r in rows:
        m_sign = "+" if r["coef_std"] >= 0 else "-"
        u_sign = "+" if r["uni_coef"] >= 0 else "-"
        if r["asserted"]:
            exp = "+" if r["expected"] > 0 else "-"
            ok = (r["uni_coef"] > 0) == (r["expected"] > 0)
            if not ok:
                result = "MISMATCH"
                failures.append((r["feature"], exp, u_sign))
            elif m_sign != u_sign:
                result = "OK/collin"   # partial sign flipped by collinearity
            else:
                result = "OK"
        else:
            exp, result = "n/a", "report"
        print(f"  {r['feature']:<20}{r['coef_std']:>10.5f}{r['uni_coef']:>10.5f}"
              f"{m_sign:>7}{u_sign:>7}{exp:>8}{str(r['asserted']):>8}"
              f"{result:>9}")
    print("=" * 86)

    if failures:
        print("\nSIGN AUDIT FAILED — a univariate sign flip means a swapped or "
              "mis-derived column (leakage/bug signal):", file=sys.stderr)
        for feat, exp, got in failures:
            print(f"  {feat}: expected {exp}, got univariate {got}",
                  file=sys.stderr)
        sys.exit(1)
    collin = [r["feature"] for r in rows if r["asserted"]
              and (r["coef_std"] >= 0) != (r["uni_coef"] >= 0)]
    print("sign audit OK: all asserted univariate signs match expected.")
    if collin:
        print(f"  note: {collin} have a multivariate partial sign flipped by "
              f"collinearity with a stronger correlate (expected, not a bug).")
    return rows


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    tune_df = load_split("tune")
    val_df = load_split("val")
    train_df = load_split("train")
    test_df = load_split("test")

    # --- Tuning (fit on tune, score on val) -------------------------------
    C_final, _ = tune_C(tune_df, val_df)

    # --- Final fit: refit scaler + clf on FULL train, once -----------------
    X_train, y_train = get_xy(train_df)
    pipe = _make_pipeline(C_final).fit(X_train, y_train)
    train_home_rate = float(y_train.mean())

    # --- Evaluate on test (never fit on test) -----------------------------
    X_test, y_test = get_xy(test_df)
    p_test = pipe.predict_proba(X_test)[:, 1]

    lc = test_df["low_confidence"].to_numpy(dtype=bool)
    hi_df = test_df[~lc]
    p_hi = p_test[~lc]

    row_all = _eval_row("logreg", "all", test_df, p_test)
    row_hi = _eval_row("logreg", "high_conf", hi_df, p_hi)

    # --- Coefficient sign audit (fails loudly) ----------------------------
    audit_signs(pipe, X_train, y_train)

    # --- Tripwires --------------------------------------------------------
    # (sign audit already exited nonzero above on any mismatch.)
    baselines = pd.read_csv(BASELINES_CSV)
    base_all_n = int(baselines[(baselines["name"] == "elo_replay")
                               & (baselines["subset"] == "all")]["n"].iloc[0])
    fails = []
    if row_all["accuracy"] >= 0.62:
        fails.append(f"logreg test accuracy {row_all['accuracy']:.4f} >= 0.62 "
                     f"(leakage suspected)")
    if not ((p_test > 0.0).all() and (p_test < 1.0).all()):
        fails.append("p_logreg outside open interval (0,1)")
    if row_all["n"] != base_all_n:
        fails.append(f"test row count {row_all['n']} != baseline row count "
                     f"{base_all_n}")
    gap = abs(row_all["mean_pred"] - train_home_rate)
    if gap > 0.05:
        fails.append(f"mean_pred {row_all['mean_pred']:.4f} differs from train "
                     f"home rate {train_home_rate:.4f} by {gap:.4f} > 0.05")
    if fails:
        print("\nTRIPWIRE FAILURE:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    # --- Write eval outputs -----------------------------------------------
    cols = ["name", "subset", "n", "accuracy", "log_loss", "brier", "auc",
            "mean_pred", "base_rate"]
    pd.DataFrame([row_all, row_hi])[cols].to_csv(LOGREG_CSV, index=False)

    preds = pd.DataFrame({
        "game_pk": test_df["game_pk"].to_numpy(),
        "date": test_df["date"].to_numpy(),
        "season": test_df["season"].to_numpy(),
        "home_win": test_df["home_win"].to_numpy(dtype=int),
        "p_logreg": p_test,
        "low_confidence": lc,
    })
    preds.to_csv(PREDS_CSV, index=False)

    # --- Serialize model + metadata ---------------------------------------
    joblib.dump(pipe, MODEL_PATH)
    meta = {
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
        "C_final": C_final,
        "train_seasons": list(TRAIN_SEASONS),
        "sklearn_version": sklearn.__version__,
        "train_home_win_rate": train_home_rate,
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Console summary: combined comparison -----------------------------
    base_cmp = baselines[baselines["subset"] == "all"]
    base_cmp = base_cmp[base_cmp["name"].isin(["constant_home", "elo_replay"])]
    rows_cmp = base_cmp[["name", "n", "accuracy", "log_loss", "brier", "auc",
                         "mean_pred", "base_rate"]].to_dict("records")
    rows_cmp.append({k: row_all[k] for k in ("name", "n", "accuracy",
                     "log_loss", "brier", "auc", "mean_pred", "base_rate")})
    cmp = compare(rows_cmp)

    print("\n" + "=" * 74)
    print("COMBINED COMPARISON — subset=all, sorted ascending by log loss")
    print("=" * 74)
    print(cmp.to_string(index=False))
    print("=" * 74)

    elo = base_cmp[base_cmp["name"] == "elo_replay"].iloc[0]
    print("\nlogreg vs elo_replay (subset=all), judged independently:")
    for metric, lower_better in (("log_loss", True), ("brier", True),
                                 ("accuracy", False)):
        lv, ev_ = row_all[metric], float(elo[metric])
        if lower_better:
            won = lv < ev_
        else:
            won = lv > ev_
        verb = "BEAT" if won else "did NOT beat"
        print(f"  {metric:<9}: logreg {lv:.4f} vs elo {ev_:.4f} "
              f"-> logreg {verb} Elo")

    print(f"\nArtifacts written:")
    print(f"  {LOGREG_CSV}")
    print(f"  {PREDS_CSV}")
    print(f"  {MODEL_PATH}")
    print(f"  {META_PATH}")


# ---------------------------------------------------------------------------
# Phase-2 ProbabilityModel-compatible single-row scorer (NOT wired in Chunk 6)
# ---------------------------------------------------------------------------

_LOADED = {}


def _load():
    if "pipe" not in _LOADED:
        _LOADED["pipe"] = joblib.load(MODEL_PATH)
        _LOADED["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    return _LOADED["pipe"], _LOADED["meta"]


def predict_proba_home(feature_row: dict) -> float:
    """Home win probability for one game given a dict of feature values.
    Reads FEATURE_COLUMNS from the row in the serialized order and applies the
    loaded pipeline. Intended to satisfy the phase-2 ProbabilityModel protocol;
    intentionally not imported by simulate.py / matchup.py in this chunk."""
    pipe, meta = _load()
    cols = meta["FEATURE_COLUMNS"]
    x = np.array([[float(feature_row[c]) for c in cols]], dtype=float)
    return float(pipe.predict_proba(x)[0, 1])


if __name__ == "__main__":
    main()
