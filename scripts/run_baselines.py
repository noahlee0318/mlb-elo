"""Chunk 5, Deliverable 4: establish every baseline on the 2024-2025 test set.

No models are trained. Three baselines, each scored on the full test split
and on the low_confidence==False subset:

  constant_home     — constant P = the TRAIN home-win rate
  elo_replay        — leakage-free from-scratch Elo replay (Deliverable 2)
  elo_current_LEAKY — the ratings.csv end-of-data snapshot; NOT a valid
                      baseline, reported only to show the size of the gap

Deterministic: no RNG anywhere. Two runs produce byte-identical
data/eval/preds_test.csv.

Run from the project root:  python scripts/run_baselines.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.elo import BASE, expected_home
from src.elo_replay import replay_elo
from src.evaluate import compare, evaluate
from src.games_data import load_games
from src.splits import (TEST_SEASONS, TRAIN_SEASONS, eval_row_count,
                        load_split, validate_splits)

EVAL_DIR = ROOT / "data" / "eval"
RATINGS_CSV = ROOT / "data" / "ratings.csv"

TRIP = {"const_acc": (0.51, 0.57), "elo_acc": (0.52, 0.60)}


def fail(msg):
    print(f"\nTRIPWIRE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    validate_splits()
    train = load_split("train")
    test = load_split("test")

    # row-count tripwire: the test split must equal the feature table's
    # 2024-2025 count after the same eval-row filter
    expected = eval_row_count(TEST_SEASONS)
    print(f"\ntest rows: {len(test)}  (feature-table 2024-25 post-filter: "
          f"{expected})")
    if len(test) != expected:
        fail(f"test row count {len(test)} != feature-table count {expected}")

    # team ids for the leaky snapshot baseline (load_split drops them)
    ids = load_games(seasons=TEST_SEASONS)[["game_pk", "home_team_id",
                                            "away_team_id"]]
    test = test.merge(ids, on="game_pk", how="left")
    assert test[["home_team_id", "away_team_id"]].notna().all().all()

    # ---- constant_home: TRAIN home-win rate ----
    p_const = float(train["home_win"].mean())
    print(f"constant_home probability (train home-win rate): {p_const:.6f}")
    test["p_constant"] = p_const

    # ---- elo_replay: leakage-free, joined 1:1 ----
    replay = replay_elo(2015, 2025)
    if (replay["elo_prob_home"] <= 0).any() or (replay["elo_prob_home"] >= 1).any():
        fail("elo_prob_home outside the open interval (0, 1)")
    merged = test.merge(replay[["game_pk", "elo_prob_home"]], on="game_pk",
                        how="left", validate="one_to_one")
    missing = merged["elo_prob_home"].isna()
    if missing.any():
        ex = merged.loc[missing, "game_pk"].head(3).tolist()
        fail(f"{int(missing.sum())} test game_pk(s) absent from elo replay; "
             f"examples: {ex}")
    test["p_elo_replay"] = merged["elo_prob_home"].to_numpy()

    # ---- elo_current_LEAKY: end-of-data ratings.csv snapshot ----
    ratings = _load_ratings_snapshot()
    test["p_elo_leaky"] = [
        expected_home(ratings.get(int(h), BASE), ratings.get(int(a), BASE))
        for h, a in zip(test["home_team_id"], test["away_team_id"])]

    # -------------------- evaluate --------------------
    models = [("constant_home", "p_constant"),
              ("elo_replay", "p_elo_replay"),
              ("elo_current_LEAKY", "p_elo_leaky")]
    hc = ~test["low_confidence"].astype(bool)
    rows = []
    for subset_name, mask in (("all", pd.Series(True, index=test.index)),
                              ("high_conf", hc)):
        sub = test[mask]
        for model_name, col in models:
            r = evaluate(sub["home_win"], sub[col], model_name)
            r["subset"] = subset_name
            rows.append(r)

    all_tbl = compare([r for r in rows if r["subset"] == "all"])
    hc_tbl = compare([r for r in rows if r["subset"] == "high_conf"])
    print("\n=== test set (all rows) — sorted by log loss ===")
    print(all_tbl.to_string(index=False))
    print("\n=== test set (high_conf: both starters >=60 trailing IP) ===")
    print(hc_tbl.to_string(index=False))

    # -------------------- tripwires --------------------
    def metric(model, subset, key):
        return next(r[key] for r in rows
                    if r["name"] == model and r["subset"] == subset)

    ca = metric("constant_home", "all", "accuracy")
    if not TRIP["const_acc"][0] <= ca <= TRIP["const_acc"][1]:
        fail(f"constant_home accuracy {ca:.4f} outside {TRIP['const_acc']}")
    ea = metric("elo_replay", "all", "accuracy")
    if not TRIP["elo_acc"][0] <= ea <= TRIP["elo_acc"][1]:
        fail(f"elo_replay accuracy {ea:.4f} outside {TRIP['elo_acc']} "
             "(>0.60 means the replay is leaking)")
    if not metric("elo_replay", "all", "log_loss") \
            < metric("constant_home", "all", "log_loss"):
        fail("elo_replay log loss is not lower than constant_home")

    # -------------------- write outputs --------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metric_cols = ["name", "subset", "n", "accuracy", "log_loss", "brier",
                   "auc", "mean_pred", "base_rate"]
    baselines = pd.DataFrame(rows)[metric_cols].sort_values(
        ["subset", "log_loss"], kind="stable")
    baselines.to_csv(EVAL_DIR / "baselines_test.csv", index=False)

    preds = test[["game_pk", "date", "season", "home_win", "p_constant",
                  "p_elo_replay", "low_confidence"]].sort_values(
        "game_pk", kind="stable").reset_index(drop=True)
    preds.to_csv(EVAL_DIR / "preds_test.csv", index=False)
    print(f"\nwrote {EVAL_DIR / 'baselines_test.csv'}")
    print(f"wrote {EVAL_DIR / 'preds_test.csv'}")
    print("\nall tripwires passed")


def _load_ratings_snapshot():
    """End-of-data ratings snapshot (LEAKY-baseline only). Prefers the
    ratings.csv snapshot; falls back to rebuilding via build_ratings."""
    if RATINGS_CSV.exists():
        r = pd.read_csv(RATINGS_CSV)
        return {int(t): float(v) for t, v in zip(r["team_id"], r["rating"])}
    from src.build_ratings import build_ratings
    ratings, _names = build_ratings()
    return {int(t): float(v) for t, v in ratings.items()}


if __name__ == "__main__":
    main()
