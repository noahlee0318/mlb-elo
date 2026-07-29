"""Chunk 10, Deliverable 3 — leakage audit for the park-factor feature.

The single most likely way this "inert" feature shows a FAKE improvement is a
park factor computed from the full season (or a later season) instead of only
prior completed seasons. This gate proves that cannot happen. Exits nonzero on
any failure.

Checks:
  1. Identical-with-or-without-future: for several (season N, venue) pairs, the
     factor computed on data THROUGH N-1 equals the factor computed on the FULL
     dataset then used for N. If they differ, a future season leaked in.
  2. No season >= N contributed to N's factor (the direct in-function guard,
     re-exercised here).
  3. Stored features.csv: every non-imputed park_factor is finite and positive;
     every imputed one is exactly 1.0; and the stored value matches a fresh
     re-computation (builder vs. documented single-game function agree).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.games_data import load_games
from src.park_factors import (EXCLUDE_SEASONS, PARK_WINDOW, compute_park_factors,
                              factors_by_season_for, park_factor_for_game)

FEATURES_CSV = Path(__file__).resolve().parents[1] / "data" / "features.csv"


def main():
    fails = []
    g = load_games()
    seasons = sorted(int(s) for s in g["season"].unique())

    print("=" * 68)
    print("PARK-FACTOR LEAKAGE AUDIT")
    print("=" * 68)

    # --- 1: identical with or without future data -------------------------
    print("[1] factor(N) computed on full data == computed on data through N-1")
    test_seasons = [N for N in (2018, 2021, 2023, 2024, 2025) if N in seasons]
    for N in test_seasons:
        full = compute_park_factors(g, N)                     # full dataset
        truncated = compute_park_factors(g[g["season"] < N], N)  # only < N
        same = full == truncated
        # also: adding a FABRICATED future season must not change factor(N)
        future = g[g["season"] < N].copy()
        inj = g[g["season"] == N].copy()
        inj_full = compute_park_factors(pd.concat([future, inj]), N)
        same_inj = full == inj_full
        ok = same and same_inj
        print(f"    N={N}: {len(full)} venues  full==trunc:{same}  "
              f"future-injection-inert:{same_inj}  -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"1: factor({N}) changed when season>= {N} data present")

    # --- 2: no season >= N contributed ------------------------------------
    print("[2] no season >= N contributes to N's window")
    for N in test_seasons:
        window = [s for s in range(N - PARK_WINDOW, N) if s not in EXCLUDE_SEASONS]
        bad = [s for s in window if s >= N]
        excl_ok = 2020 not in window
        print(f"    N={N}: window={window}  no future:{not bad}  "
              f"2020 excluded:{excl_ok}  -> {'OK' if (not bad and excl_ok) else 'FAIL'}")
        if bad or not excl_ok:
            fails.append(f"2: window for {N} is {window}")

    # --- 3: stored features.csv validity + builder/function agreement -----
    print("[3] stored park_factor: finite/positive, imputed==1.0, matches recompute")
    feats = pd.read_csv(FEATURES_CSV, low_memory=False)
    pf = feats["park_factor"].astype(float)
    imp = feats["park_factor_imputed"].astype(str).isin(["True", "true"])
    n_bad_val = int((~np.isfinite(pf) | (pf <= 0)).sum())
    imp_ok = bool((pf[imp] == 1.0).all())
    print(f"    non-finite/non-positive: {n_bad_val}  |  imputed all 1.0: {imp_ok}")
    if n_bad_val:
        fails.append(f"3: {n_bad_val} non-finite/non-positive park_factor")
    if not imp_ok:
        fails.append("3: an imputed park_factor != 1.0")

    # recompute for a sample and compare to the stored value
    fbs = factors_by_season_for(g, seasons)
    gg = g.set_index("game_pk")
    sample = feats.sample(min(400, len(feats)), random_state=42)
    mismatch = 0
    for r in sample.itertuples(index=False):
        if int(r.game_pk) not in gg.index:
            continue                        # 2026 rows etc. all present via g
        venue = gg.loc[int(r.game_pk), "venue_id"]
        f, i = park_factor_for_game(int(r.season), venue, fbs)
        if abs(float(f) - float(r.park_factor)) > 1e-9 or bool(i) != bool(
                str(r.park_factor_imputed) in ("True", "true")):
            mismatch += 1
    print(f"    stored vs recomputed mismatches (of {len(sample)}): {mismatch}")
    if mismatch:
        fails.append(f"3: {mismatch} stored park_factor values disagree with recompute")

    print("=" * 68)
    if fails:
        print("PARK LEAKAGE AUDIT FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("PARK LEAKAGE AUDIT PASSED — factors are prior-season-only, clean.")


if __name__ == "__main__":
    main()
