"""The three leakage defenses for src/features.py. Standalone:

    python src/test_leakage.py

Defense 1 — slow reference implementation: per row, physically slice the
input frames to game_order < i and recompute every feature from the slice
alone, sharing NO windowing code with the fast path. Exact-equality
comparison on >=3000 random rows (all aggregates are integer sums, so the
two paths must agree to full float precision — any mismatch is a fast-path
bug; the reference is never adjusted to match).

Defense 2 — future-deletion: for 200 random games, rebuild features from
input frames with every OTHER game at game_order >= i physically deleted
(the game's own rows are kept — the starter's identity is an input, not a
feature; its stats never enter any strict-< window). Output for that game
must be identical to the full-data run.

Defense 3 — shuffled-label: shuffle home_win, rebuild, and require every
feature's correlation with the shuffled label to be |r| < 0.03.

Exit code 0 only if all three pass. A failure is reported and NOT worked
around here.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features import (FEATURE_COLUMNS, LEAGUE_LOOKBACK_DAYS,
                          FORM_MIN_GAMES, FORM_WINDOW, REST_CAP,
                          SP_WINDOW_OUTS, build_features, canonical_order,
                          load_inputs, load_venue_offsets)

REF_SAMPLE = 3000
DELETE_SAMPLE = 200
SEED = 20260719

CMP_COLS = FEATURE_COLUMNS + [
    "home_sp_low_confidence", "away_sp_low_confidence",
    "home_sp_trailing_ip", "away_sp_trailing_ip",
    "home_form_low_confidence", "away_form_low_confidence",
    "home_games_played", "away_games_played",
]


# --------------------------------------------------------------------------
# Defense 1 — the slow reference
# --------------------------------------------------------------------------

def reference_row(g, logs_o, venue_offsets, i):
    """Every feature for the game at game_order == i, computed from
    physical slices g[g.game_order < i] / logs_o[logs_o.game_order < i]."""
    row = g.loc[g["game_order"] == i].iloc[0]
    prior_g = g[g["game_order"] < i]
    prior_l = logs_o[logs_o["game_order"] < i]
    out = {}

    for side in ("home", "away"):
        team = row[f"{side}_team_id"]
        # this team's prior games this season, most recent last
        mine = prior_g[((prior_g["home_team_id"] == team)
                        | (prior_g["away_team_id"] == team))
                       & (prior_g["season"] == row["season"])]
        mine = mine.sort_values("game_order")
        gp = len(mine)
        out[f"{side}_games_played"] = gp
        out[f"{side}_form_low_confidence"] = gp < FORM_WINDOW

        last15 = mine.tail(FORM_WINDOW)
        wons = []
        for rr in last15.itertuples(index=False):
            hw = rr.home_win
            if pd.isna(hw):
                continue                      # tie: occupies a slot, no decision
            wons.append(int(hw) if rr.home_team_id == team else 1 - int(hw))
        out[f"{side}_form_15"] = (sum(wons) / len(wons)
                                  if len(wons) >= FORM_MIN_GAMES else np.nan)

        if gp >= FORM_MIN_GAMES:
            rd = 0
            for rr in mine.itertuples(index=False):
                d = int(rr.home_score) - int(rr.away_score)
                rd += d if rr.home_team_id == team else -d
            out[f"{side}_run_diff_pg"] = rd / gp
        else:
            out[f"{side}_run_diff_pg"] = np.nan

        # rest: previous game ANY season (cap swallows the offseason)
        mine_all = prior_g[(prior_g["home_team_id"] == team)
                           | (prior_g["away_team_id"] == team)]
        if len(mine_all):
            prev = mine_all.sort_values("game_order").iloc[-1]
            days = (pd.Timestamp(row["date"]) - pd.Timestamp(prev["date"])).days
            out[f"{side}_rest_days"] = min(days, REST_CAP)
        else:
            out[f"{side}_rest_days"] = REST_CAP

        if side == "away":
            if pd.isna(row["venue_id"]):
                out["away_tz_delta"] = np.nan
            elif len(mine) == 0:
                out["away_tz_delta"] = 0
            else:
                pv = mine.iloc[-1]["venue_id"]
                out["away_tz_delta"] = (np.nan if pd.isna(pv) else
                                        abs(venue_offsets[int(row["venue_id"])]
                                            - venue_offsets[int(pv)]))

        # starting pitcher (train mode: the actual starter of THIS game)
        me_start = logs_o[(logs_o["game_pk"] == row["game_pk"])
                          & (logs_o["team_id"] == team)
                          & (logs_o["is_starter"])]
        pid = int(me_start.iloc[0]["pitcher_id"]) if len(me_start) else None

        fip = kr = np.nan
        outs_avail, enough = 0, False
        if pid is not None:
            hist = prior_l[prior_l["pitcher_id"] == pid].sort_values("game_order")
            outs_avail = int(hist["ip_outs"].sum())
            if outs_avail >= SP_WINDOW_OUTS:
                acc, rows_used = 0, []
                for rr in reversed(list(hist.itertuples(index=False))):
                    rows_used.append(rr)
                    acc += int(rr.ip_outs)
                    if acc >= SP_WINDOW_OUTS:
                        break
                outs = sum(int(x.ip_outs) for x in rows_used)
                hr = sum(int(x.home_runs) for x in rows_used)
                bb = sum(int(x.walks) for x in rows_used)
                hbp = sum(int(x.hit_by_pitch) for x in rows_used)
                k = sum(int(x.strikeouts) for x in rows_used)
                bf = sum(int(x.batters_faced) for x in rows_used)
                fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / (outs / 3.0)
                kr = k / bf
                enough = True
                outs_avail = outs
            elif outs_avail > 0:
                outs = int(hist["ip_outs"].sum())
                fip = ((13 * int(hist["home_runs"].sum())
                        + 3 * (int(hist["walks"].sum())
                               + int(hist["hit_by_pitch"].sum()))
                        - 2 * int(hist["strikeouts"].sum())) / (outs / 3.0))
                kr = int(hist["strikeouts"].sum()) / int(hist["batters_faced"].sum())
        if not enough:
            # as-of league average: trailing 365 days of prior STARTER rows
            cutoff = (pd.Timestamp(row["date"])
                      - pd.Timedelta(days=LEAGUE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            lg = prior_l[(prior_l["is_starter"]) & (prior_l["date"] >= cutoff)]
            if len(lg):
                outs = int(lg["ip_outs"].sum())
                fip = ((13 * int(lg["home_runs"].sum())
                        + 3 * (int(lg["walks"].sum())
                               + int(lg["hit_by_pitch"].sum()))
                        - 2 * int(lg["strikeouts"].sum())) / (outs / 3.0))
                kr = int(lg["strikeouts"].sum()) / int(lg["batters_faced"].sum())
            else:
                fip = kr = np.nan
        out[f"{side}_sp_fip_60"] = fip
        out[f"{side}_sp_k_rate"] = kr
        out[f"{side}_sp_trailing_ip"] = outs_avail / 3.0
        out[f"{side}_sp_low_confidence"] = not enough
    return out


def _eq(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        return float(a) == float(b)      # exact — both paths sum integers
    except (TypeError, ValueError):
        return a == b


def defense1(games, logs, fast):
    rng = np.random.default_rng(SEED)
    g = canonical_order(games)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    logs_o = logs.copy()
    logs_o["game_order"] = logs_o["game_pk"].map(order_of_pk)
    logs_o = logs_o.dropna(subset=["game_order"])
    logs_o["game_order"] = logs_o["game_order"].astype("int64")
    venue_offsets = load_venue_offsets()

    sample = rng.choice(len(g), size=min(REF_SAMPLE, len(g)), replace=False)
    fast_by_order = fast.set_index("game_order")
    bad = 0
    for n, i in enumerate(sorted(int(x) for x in sample), 1):
        ref = reference_row(g, logs_o, venue_offsets, i)
        frow = fast_by_order.loc[i]
        for col, val in ref.items():
            if not _eq(val, frow[col]):
                bad += 1
                if bad <= 10:
                    print(f"  MISMATCH order={i} game_pk={int(frow['game_pk'])} "
                          f"{col}: ref={val!r} fast={frow[col]!r}")
        if n % 500 == 0:
            print(f"  reference check {n}/{len(sample)} rows...", flush=True)
    print(f"Defense 1 (reference implementation, {len(sample)} rows): "
          + ("PASS" if bad == 0 else f"FAIL — {bad} mismatching values"))
    return bad == 0


# --------------------------------------------------------------------------
# Defense 2 — future deletion
# --------------------------------------------------------------------------

def defense2(games, logs, fast):
    rng = np.random.default_rng(SEED + 1)
    g = canonical_order(games)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    fast_by_order = fast.set_index("game_order")
    sample = sorted(int(x) for x in
                    rng.choice(len(g), size=min(DELETE_SAMPLE, len(g)),
                               replace=False))
    bad = 0
    logs_order = logs["game_pk"].map(order_of_pk)
    for n, i in enumerate(sample, 1):
        pk_i = int(g.loc[g["game_order"] == i, "game_pk"].iloc[0])
        # physically delete every OTHER game at game_order >= i
        g_cut = g[(g["game_order"] < i) | (g["game_pk"] == pk_i)]
        l_cut = logs[(logs_order < i) | (logs["game_pk"] == pk_i)]
        out = build_features(g_cut.drop(columns=["game_order"]), l_cut,
                             mode="train")
        orow = out[out["game_pk"] == pk_i].iloc[0]
        frow = fast_by_order.loc[i]
        for col in CMP_COLS:
            if not _eq(orow[col], frow[col]):
                bad += 1
                if bad <= 10:
                    print(f"  MISMATCH game_pk={pk_i} {col}: "
                          f"truncated={orow[col]!r} full={frow[col]!r}")
        if n % 50 == 0:
            print(f"  future-deletion check {n}/{len(sample)} games...", flush=True)
    print(f"Defense 2 (future deletion, {len(sample)} games): "
          + ("PASS" if bad == 0 else f"FAIL — {bad} mismatching values"))
    return bad == 0


# --------------------------------------------------------------------------
# Defense 3 — shuffled labels
# --------------------------------------------------------------------------

def defense3(games, logs):
    rng = np.random.default_rng(SEED + 2)
    shuffled = games.copy().reset_index(drop=True)
    shuffled["home_win"] = rng.permutation(shuffled["home_win"].to_numpy())
    out = build_features(shuffled, logs, mode="train")
    y = pd.to_numeric(out["home_win"], errors="coerce")
    worst, bad = 0.0, []
    for col in FEATURE_COLUMNS:
        x = pd.to_numeric(out[col], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 100 or x[m].std() == 0:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        worst = max(worst, abs(r))
        if abs(r) >= 0.03:
            bad.append((col, r))
        print(f"  shuffled-label corr {col:<22} r={r:+.4f}")
    print(f"Defense 3 (shuffled labels): worst |r|={worst:.4f} — "
          + ("PASS" if not bad else f"FAIL — {bad}"))
    return not bad


def main():
    games, logs = load_inputs()
    print("building fast-path features once for comparison...", flush=True)
    fast = build_features(games, logs, mode="train")

    ok1 = defense1(games, logs, fast)
    ok2 = defense2(games, logs, fast)
    ok3 = defense3(games, logs)
    print("\n" + "=" * 60)
    for name, ok in (("Defense 1 — reference implementation", ok1),
                     ("Defense 2 — future deletion", ok2),
                     ("Defense 3 — shuffled labels", ok3)):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
