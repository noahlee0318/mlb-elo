"""Standalone verification for data/features.csv:

    python src/verify_features.py

Prints the report and exits nonzero on failure. Rebuilds features in both
modes in-process for the train/predict agreement check.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features import (FEATURE_COLUMNS, FEATURES_CSV, build_features,
                          canonical_order, load_inputs, load_venue_offsets,
                          team_long_frame)

EXPECTED_RANGES = {
    "home_form_15": (0.0, 1.0), "away_form_15": (0.0, 1.0),
    "home_rest_days": (0, 5), "away_rest_days": (0, 5),
    # away_tz_delta: [0,3] for DOMESTIC pairs; international specials
    # (London, Tokyo, Seoul series) legitimately exceed it and are
    # checked by name in tz_exceptions() instead of by range.
}


def tz_exceptions(df, games):
    """Rows with away_tz_delta > 3 must involve an international venue
    (current or previous, UTC offset >= 0). Returns (n_checked, offenders)."""
    offsets = load_venue_offsets()
    g = canonical_order(games)
    long = team_long_frame(g)
    long["prev_venue"] = long.groupby(["team_id", "season"],
                                      sort=False)["venue_id"].shift(1)
    away = long[~long["is_home"]][["game_pk", "venue_id", "prev_venue"]]
    big = df[pd.to_numeric(df["away_tz_delta"], errors="coerce") > 3]
    big = big.merge(away, on="game_pk", how="left")
    offenders = []
    for r in big.itertuples(index=False):
        cur = offsets.get(int(r.venue_id)) if pd.notna(r.venue_id) else None
        prv = offsets.get(int(r.prev_venue)) if pd.notna(r.prev_venue) else None
        intl = (cur is not None and cur >= 0) or (prv is not None and prv >= 0)
        if not intl:
            offenders.append((int(r.game_pk), r.away_tz_delta))
    return len(big), offenders


def main():
    ok = True
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    for c in df.columns:
        if c.endswith(("low_confidence", "is_opener")) or c == "in_burn_in":
            df[c] = df[c].map({True: True, False: False, "True": True,
                               "False": False})

    n_burn = int(df["in_burn_in"].sum())
    print(f"rows: total={len(df)}  excluded by burn-in={n_burn}  "
          f"remaining={len(df) - n_burn}\n")
    post = df[~df["in_burn_in"].astype(bool)]

    print("null counts per feature (after burn-in | of those, rows without "
          "the matching low-confidence flag):")
    null_ok = True
    for c in FEATURE_COLUMNS:
        n = int(post[c].isna().sum())
        side = c.split("_")[0]
        if "sp_" in c:
            flag = post[f"{side}_sp_low_confidence"]
        elif "form" in c or "run_diff" in c:
            flag = post[f"{side}_form_low_confidence"]
        else:
            flag = pd.Series(False, index=post.index)
        unflagged = int((post[c].isna() & ~flag.astype(bool)).sum())
        note = ""
        if c == "away_tz_delta" and unflagged <= 4:
            note = "  (null-venue Field of Dreams games — named exception)"
        elif unflagged:
            null_ok = False
            note = "  <-- UNEXPLAINED NULLS"
        print(f"  {c:<20} {n:>5} | {unflagged}{note}")
    ok &= null_ok

    print("\nSP low-confidence share by season (walk-back correctness is "
          "proven by Defense 1; this checks coverage. Steady-state seasons "
          "must be <15%. Named structural exemptions: 2015 (logs start "
          "2015 — no prior history exists), 2016 (one prior season), 2021 "
          "(2020 excluded from the log scope by convention)):")
    sp_ok = True
    exempt = {2015, 2016, 2021}
    for season, grp in post.groupby("season"):
        h = grp["home_sp_low_confidence"].astype(bool).mean()
        a = grp["away_sp_low_confidence"].astype(bool).mean()
        mark = ""
        if season in exempt:
            mark = "  (boundary season — structural, exempt)"
        elif h > 0.15 or a > 0.15:
            sp_ok, mark = False, "  <-- TOO HIGH: walk-back suspect"
        print(f"  {season}: home {h:.1%}  away {a:.1%}{mark}")
    ok &= sp_ok

    print("\nform low-confidence share by season (either team; expect ~9%, "
          "concentrated in the first weeks — ~0% means the season reset "
          "is broken):")
    form_ok = True
    first_season = int(df["season"].min())
    for season, grp in post.groupby("season"):
        e = (grp["home_form_low_confidence"].astype(bool)
             | grp["away_form_low_confidence"].astype(bool)).mean()
        mark = ""
        if season == first_season and e < 0.02:
            mark = ("  (first season: its cold-start weeks ARE the burn-in "
                    "window, so ~0% here is by construction)")
        elif e < 0.02:
            form_ok, mark = False, "  <-- NEAR ZERO: reset broken?"
        print(f"  {season}: {e:.1%}{mark}")
    ok &= form_ok

    # season-boundary reset test
    print("\nseason-boundary reset test:")
    d = df.copy()
    reset_bad = 0
    carry_seen_by_season = {}
    # a team's true first game of a season can be home OR away, so build
    # one long frame across both sides before taking the first row
    sides = []
    for side in ("home", "away"):
        sub = d[[f"{side}_team_id", "season", "game_order",
                 f"{side}_games_played", f"{side}_form_15",
                 f"{side}_run_diff_pg", f"{side}_sp_low_confidence",
                 f"{side}_sp_trailing_ip"]].rename(
            columns=lambda c: c.replace(f"{side}_", ""))
        sides.append(sub)
    both = pd.concat(sides, ignore_index=True)
    # .nth(0), NOT .first(): first() skips NaN per column and would fetch
    # the first non-null form instead of the opener's null
    d2 = both.sort_values("game_order").groupby(["team_id", "season"],
                                                sort=False).nth(0)
    d2 = d2.set_index(["team_id", "season"])
    for (team, season), r in d2.iterrows():
        if season == 2015:
            continue
        if r["games_played"] != 0 or pd.notna(r["form_15"]) \
                or (pd.notna(r["run_diff_pg"]) and r["run_diff_pg"] != 0):
            reset_bad += 1
        if not bool(r["sp_low_confidence"]) and r["sp_trailing_ip"] >= 60:
            carry_seen_by_season[season] = carry_seen_by_season.get(season, 0) + 1
    print(f"  first-game team-state violations (games_played!=0 or non-null "
          f"form/run_diff): {reset_bad} (must be 0)")
    carry_ok = all(carry_seen_by_season.get(s, 0) > 0
                   for s in sorted(d["season"].unique()) if s != 2015)
    print("  season-openers with a FULL-window (>=60 IP) SP feature — proves "
          "pitcher state carries across the boundary while team state resets:")
    for s in sorted(carry_seen_by_season):
        print(f"    {s}: {carry_seen_by_season[s]} opener sides")
    ok &= (reset_bad == 0) and carry_ok

    print("\nfeature distributions (post-burn-in):")
    range_ok = True
    for c in FEATURE_COLUMNS:
        x = pd.to_numeric(post[c], errors="coerce").dropna()
        q = x.quantile([0.01, 0.5, 0.99])
        print(f"  {c:<20} min={x.min():+.3f} p1={q[0.01]:+.3f} "
              f"med={q[0.5]:+.3f} p99={q[0.99]:+.3f} max={x.max():+.3f}")
        if c in EXPECTED_RANGES:
            lo, hi = EXPECTED_RANGES[c]
            if x.min() < lo or x.max() > hi:
                range_ok = False
                print(f"    <-- OUTSIDE EXPECTED [{lo}, {hi}]")
    ok &= range_ok

    print("\ncorrelation with home_win (|r|>0.20 is a leakage signature):")
    corr_ok = True
    y = pd.to_numeric(post["home_win"], errors="coerce")
    for c in FEATURE_COLUMNS:
        x = pd.to_numeric(post[c], errors="coerce")
        m = x.notna() & y.notna()
        r = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() else np.nan
        mark = ""
        if abs(r) > 0.20:
            corr_ok, mark = False, "  <-- LEAKAGE SIGNATURE"
        elif abs(r) > 0.12:
            mark = "  (above the ~0.12 guideline — inspect)"
        print(f"  {c:<20} r={r:+.4f}{mark}")
    ok &= corr_ok

    print("\nhome_sp_is_opener count by season (expect ~0 pre-2018, rising):")
    openers = df[df["home_sp_is_opener"] == True]  # noqa: E712
    counts = openers.groupby("season").size()
    for season in sorted(df["season"].unique()):
        print(f"  {season}: {int(counts.get(season, 0))}")

    games, logs = load_inputs()
    n_intl, tz_bad = tz_exceptions(df, games)
    print(f"\naway_tz_delta > 3: {n_intl} rows — every one must involve an "
          f"international venue (London/Tokyo/Seoul series); "
          f"unexplained: {len(tz_bad)} (must be 0)"
          + (f"  {tz_bad[:10]}" if tz_bad else ""))
    ok &= len(tz_bad) == 0

    print("\nmode='train' vs mode='predict' non-pitcher feature agreement:")
    tr = build_features(games, logs, mode="train").set_index("game_pk")
    pr = build_features(games, logs, mode="predict").set_index("game_pk")
    non_pitcher = ["home_form_15", "away_form_15", "home_run_diff_pg",
                   "away_run_diff_pg", "home_rest_days", "away_rest_days",
                   "away_tz_delta", "home_games_played", "away_games_played"]
    diff = 0
    for c in non_pitcher:
        a, b = tr[c], pr[c].reindex(tr.index)
        neq = ~((a == b) | (a.isna() & b.isna()))
        diff += int(neq.sum())
    print(f"  mismatching non-pitcher values: {diff} (must be 0)")
    ok &= diff == 0

    print("\n" + "=" * 60)
    print("VERIFY FEATURES:", "PASS" if ok else "FAIL")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
