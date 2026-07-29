"""Settle logged predictions against final scores and report accuracy and
log-loss for Elo vs the naive home baseline — the proof (or not) that the
model beats a coin-lean.

Run from the project root:  python -m src.score
"""

import math
from pathlib import Path

import pandas as pd

from src.mlb_api import polite_sleep
from src.schedule import fetch_slate

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"

EPS = 1e-12


def settle(df):
    """Fill home_won for any logged game that has since gone Final."""
    pending_dates = sorted(df.loc[df["home_won"].isna(), "date"].unique())
    for d in pending_dates:
        finals = {g["game_id"]: g for g in fetch_slate(str(d))
                  if g["status"] == "Final"
                  and g["home_score"] not in (None, "")
                  and g["away_score"] not in (None, "")}
        polite_sleep()
        for idx in df.index[(df["date"] == d) & df["home_won"].isna()]:
            g = finals.get(df.at[idx, "game_id"])
            if g is None:
                continue
            hs, aw = int(g["home_score"]), int(g["away_score"])
            if hs != aw:
                df.at[idx, "home_won"] = 1 if hs > aw else 0
    return df


def report(df):
    settled = df[df["home_won"].notna()]
    print(f"{len(settled)} settled of {len(df)} logged predictions")
    if settled.empty:
        print("Nothing to score yet — run again after games finish.")
        return
    y = settled["home_won"].astype(float)
    for label, col in (("Elo", "elo_prob_home"), ("Baseline 0.540", "baseline_prob_home")):
        p = settled[col].astype(float).clip(EPS, 1 - EPS)
        acc = float(((p >= 0.5).astype(float) == y).mean())
        logloss = float(-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log)).mean())
        print(f"{label:<15}  accuracy={acc:.3f}  log-loss={logloss:.4f}")


def _grade_day_elo(day_rows):
    """(correct, total, pushes) for one day's settled Elo predictions.

    A pick is the side given probability strictly above 0.500. Exactly
    0.500 is a no-pick ("push") and is excluded from both counts —
    counting it for either side would misstate accuracy.
    """
    p = day_rows["elo_prob_home"].astype(float)
    y = day_rows["home_won"].astype(float)
    push = p == 0.5
    correct = int((((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))).sum())
    return correct, int((~push).sum()), int(push.sum())


def score_latest_day():
    """Grade the most recent date whose logged Elo predictions have Final
    results — the dashboard's "how did yesterday go" line.

    Not "today minus 1": the date graded is the newest one with at least
    one settled prediction, however far back that is. Returns a dict:
      {"status": "no_log"}                              nothing logged yet
      {"status": "pending", "date": d}                  logged, none Final yet
      {"status": "ok", "date": d, "correct": c,
       "total": t, "pushes": p, "pending": k}           graded; k games that
                                                        day not Final yet
    Settles in place first (one API call per pending date); if that fails
    (offline) it grades whatever settled previously instead of crashing.
    """
    if not PREDICTIONS_CSV.exists():
        return {"status": "no_log"}
    df = pd.read_csv(PREDICTIONS_CSV)
    if df.empty:
        return {"status": "no_log"}
    df["home_won"] = pd.to_numeric(df["home_won"], errors="coerce").astype("Int64")
    try:
        df = settle(df)
        tmp = PREDICTIONS_CSV.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(PREDICTIONS_CSV)
    except Exception:
        pass  # offline — grade what was already settled
    settled = df[df["home_won"].notna()]
    if settled.empty:
        return {"status": "pending", "date": str(df["date"].max())}
    d = str(settled["date"].max())
    day = df[df["date"] == d]
    correct, total, pushes = _grade_day_elo(day[day["home_won"].notna()])
    if total == 0:  # every settled game that day was a push
        return {"status": "pending", "date": d}
    return {"status": "ok", "date": d, "correct": correct, "total": total,
            "pushes": pushes, "pending": int(day["home_won"].isna().sum())}


def main():
    if not PREDICTIONS_CSV.exists():
        print(f"No predictions logged yet ({PREDICTIONS_CSV} not found) — "
              "run `python -m src.predict` on a game day first.")
        return
    df = pd.read_csv(PREDICTIONS_CSV)
    if df.empty:
        print("predictions.csv is empty — nothing to score.")
        return
    df["home_won"] = pd.to_numeric(df["home_won"], errors="coerce").astype("Int64")
    df = settle(df)
    df.to_csv(PREDICTIONS_CSV, index=False)
    report(df)


if __name__ == "__main__":
    main()
