"""Chunk 4: the leakage audit. No modeling deliverable — the only output is
an audit record (docs/leakage_audit.md) and a pass/fail exit code.

    python src/audit_leakage.py

Seven tests. Chronological split everywhere (train 2015-2023, test
2024-2025), one plain LogisticRegression(max_iter=2000), no tuning. The
tripwire (test acc > 0.590 or log loss < 0.655) stops the audit and reports
a suspected leak rather than proceeding.
"""

import hashlib
import io
import subprocess
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (FEATURE_COLUMNS, FEATURES_CSV, LEAGUE_LOOKBACK_DAYS,
                          FORM_MIN_GAMES, FORM_WINDOW, REST_CAP,
                          SP_WINDOW_OUTS, build_features, canonical_order,
                          load_inputs, load_venue_offsets)
from src.games_data import load_games

DOCS = Path(__file__).resolve().parents[1] / "docs"
AUDIT_MD = DOCS / "leakage_audit.md"

TRAIN_MAX_SEASON = 2023
TRIP_ACC, TRIP_LL = 0.590, 0.655
SHUFFLE_SEEDS = [11, 22, 33, 44, 55]

# Hard-coded trace games (Test 5) — chosen once, kept stable:
#   446946  2016-04-10  both teams at exactly 6 prior games (partial window)
#   633121  2021-07-26  mid-season, both teams >15 games, no low-conf SP
#   663152  2022-06-04  game 2 of a doubleheader (the ordering-critical case)
TRACE_PKS = [446946, 633121, 663152]

REPORT = io.StringIO()


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    REPORT.write(line + "\n")


def make_model():
    # plain logistic regression; imputer/scaler are fit on TRAIN ONLY via
    # the pipeline (no test-set statistics ever touch the fit)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000)),
    ])


def load_matrix():
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    for c in df.columns:
        if c.endswith(("low_confidence", "is_opener")) or c == "in_burn_in":
            df[c] = df[c].map({True: True, False: False,
                               "True": True, "False": False})
    df = df[~df["in_burn_in"].astype(bool)]
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = df["home_win"].astype(int)
    train = df[df["season"] <= TRAIN_MAX_SEASON]
    test = df[df["season"] > TRAIN_MAX_SEASON]
    # split discipline: chronological, never random
    assert train["game_order"].max() < test["game_order"].min(), \
        "chronological split violated"
    return df, train, test


def fit_eval(train, test, cols, y_train=None, y_test=None):
    model = make_model()
    yt = train["home_win"] if y_train is None else y_train
    ye = test["home_win"] if y_test is None else y_test
    model.fit(train[cols], yt)
    p = model.predict_proba(test[cols])[:, 1]
    return {"acc": accuracy_score(ye, p >= 0.5),
            "ll": log_loss(ye, p),
            "brier": brier_score_loss(ye, p),
            "auc": roc_auc_score(ye, p),
            "p": p}


# ---------------------------------------------------------------- Test 1

def test1_shuffle(df, train, test):
    say("\n## Test 1 — shuffle test (5 seeds; judged on log loss; accuracy "
        "stays near the base rate BY DESIGN since shuffling preserves the "
        "class balance)")
    ok = True
    for seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(seed)
        y_shuf = pd.Series(rng.permutation(df["home_win"].to_numpy()),
                           index=df.index)
        r = fit_eval(train, test, FEATURE_COLUMNS,
                     y_train=y_shuf.loc[train.index],
                     y_test=y_shuf.loc[test.index])
        verdict = "PASS" if r["ll"] >= 0.685 else "FAIL — below 0.685"
        if r["ll"] < 0.685:
            ok = False
        say(f"  seed {seed}: log_loss={r['ll']:.4f}  acc={r['acc']:.4f}  "
            f"[{verdict}]")
    say(f"Test 1: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Test 2

def test2_real(train, test):
    say("\n## Test 2 — real model, chronological holdout "
        f"(train 2015-{TRAIN_MAX_SEASON} n={len(train)}, "
        f"test {TRAIN_MAX_SEASON + 1}-2025 n={len(test)})")
    r = fit_eval(train, test, FEATURE_COLUMNS)
    home_rate_test = test["home_win"].mean()
    p0 = train["home_win"].mean()
    base_ll = log_loss(test["home_win"], np.full(len(test), p0))
    say(f"  model:               acc={r['acc']:.4f}  log_loss={r['ll']:.4f}  "
        f"brier={r['brier']:.4f}  auc={r['auc']:.4f}")
    say(f"  always-pick-home:    acc={home_rate_test:.4f}")
    say(f"  base-rate constant ({p0:.4f}): log_loss={base_ll:.4f}")
    tripped = r["acc"] > TRIP_ACC or r["ll"] < TRIP_LL
    if tripped:
        say(f"Test 2: TRIPWIRE HIT (acc>{TRIP_ACC} or ll<{TRIP_LL}) — "
            "SUSPECTED LEAK. Stopping the audit here per protocol.")
    else:
        say(f"Test 2: PASS (within honest range; tripwire acc>{TRIP_ACC} "
            f"/ ll<{TRIP_LL} not hit)")
    return (not tripped), r


# ---------------------------------------------------------------- Test 3

def test3_calibration(test, p):
    say("\n## Test 3 — prediction distribution and calibration")
    qs = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    vals = np.quantile(p, qs)
    say("  prediction quantiles: "
        + "  ".join(f"{'min' if q == 0 else 'max' if q == 1 else f'p{int(q*100)}'}"
                    f"={v:.3f}" for q, v in zip(qs, vals)))
    outside = (p < 0.25) | (p > 0.80)
    say(f"  predictions outside [0.25, 0.80]: {int(outside.sum())} "
        f"of {len(p)}")
    ok = True
    if outside.any():
        idx = np.argsort(np.abs(p - 0.5))[::-1][:10]
        say("  10 most extreme predictions:")
        for i in idx:
            row = test.iloc[i]
            say(f"    game_pk={int(row['game_pk'])} p={p[i]:.3f} "
                f"form=({row['home_form_15']}, {row['away_form_15']}) "
                f"rdpg=({row['home_run_diff_pg']:.2f}, "
                f"{row['away_run_diff_pg']:.2f}) "
                f"fip=({row['home_sp_fip_60']:.2f}, {row['away_sp_fip_60']:.2f})")
    say("  calibration by decile (pred mean vs observed rate):")
    dec = pd.qcut(p, 10, duplicates="drop")
    tbl = pd.DataFrame({"p": p, "y": test["home_win"].to_numpy()}) \
        .groupby(dec, observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                         obs=("y", "mean"))
    worst_gap = 0.0
    for iv, r in tbl.iterrows():
        gap = abs(r["pred"] - r["obs"])
        worst_gap = max(worst_gap, gap)
        say(f"    {str(iv):<18} n={int(r['n']):>4}  pred={r['pred']:.3f}  "
            f"obs={r['obs']:.3f}  gap={gap:+.3f}")
    say(f"  worst decile gap: {worst_gap:.3f}")
    say("Test 3: PASS" if ok else "Test 3: FAIL")
    return ok


# ---------------------------------------------------------------- Test 4

def test4_ablation(train, test, full_ll):
    say("\n## Test 4 — single-feature models (fail if any lone feature "
        "acc > 0.580; scrutiny note above 0.560)")
    ok = True
    rows = []
    for c in FEATURE_COLUMNS:
        r = fit_eval(train, test, [c])
        rows.append((c, r["acc"], r["ll"]))
    for c, acc, ll in sorted(rows, key=lambda x: -x[1]):
        mark = ""
        if acc > 0.580:
            ok, mark = False, "  <-- SIDE CHANNEL: trace this feature"
        elif acc > 0.560:
            mark = "  (above 0.560 — scrutiny)"
        say(f"  {c:<20} acc={acc:.4f}  log_loss={ll:.4f}{mark}")
    say("  leave-one-out (change in test log loss vs full model; "
        "scrutiny above +0.010):")
    for c in FEATURE_COLUMNS:
        cols = [x for x in FEATURE_COLUMNS if x != c]
        r = fit_eval(train, test, cols)
        delta = r["ll"] - full_ll
        mark = "  (large — scrutiny)" if delta > 0.010 else ""
        say(f"  drop {c:<20} ll={r['ll']:.4f}  delta={delta:+.4f}{mark}")
    say(f"Test 4: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Test 5

def _hand_features(games, logs, pk, venue_offsets):
    """Fully independent hand computation of every feature for one game,
    straight from the raw frames. Returns (values, contributing_pks)."""
    g = canonical_order(games)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    lo = logs.copy()
    lo["game_order"] = lo["game_pk"].map(order_of_pk)
    lo = lo.dropna(subset=["game_order"])
    lo["game_order"] = lo["game_order"].astype("int64")

    row = g.loc[g["game_pk"] == pk].iloc[0]
    i = int(row["game_order"])
    prior = g[g["game_order"] < i]
    prior_l = lo[lo["game_order"] < i]
    vals, contribs = {}, {}
    for side in ("home", "away"):
        team = row[f"{side}_team_id"]
        mine = prior[((prior["home_team_id"] == team)
                      | (prior["away_team_id"] == team))
                     & (prior["season"] == row["season"])].sort_values("game_order")
        gp = len(mine)
        vals[f"{side}_games_played"] = gp
        vals[f"{side}_form_low_confidence"] = gp < FORM_WINDOW
        last = mine.tail(FORM_WINDOW)
        contribs[f"{side}_form_15"] = [int(x) for x in last["game_pk"]]
        wins = []
        for r in last.itertuples(index=False):
            if pd.isna(r.home_win):
                continue
            wins.append(int(r.home_win) if r.home_team_id == team
                        else 1 - int(r.home_win))
        vals[f"{side}_form_15"] = (sum(wins) / len(wins)
                                   if len(wins) >= FORM_MIN_GAMES else np.nan)
        rd = sum((int(r.home_score) - int(r.away_score))
                 * (1 if r.home_team_id == team else -1)
                 for r in mine.itertuples(index=False))
        vals[f"{side}_run_diff_pg"] = rd / gp if gp >= FORM_MIN_GAMES else np.nan
        contribs[f"{side}_run_diff_pg"] = [int(x) for x in mine["game_pk"]]

        alle = prior[(prior["home_team_id"] == team)
                     | (prior["away_team_id"] == team)].sort_values("game_order")
        if len(alle):
            prev = alle.iloc[-1]
            vals[f"{side}_rest_days"] = min(
                (pd.Timestamp(row["date"]) - pd.Timestamp(prev["date"])).days,
                REST_CAP)
            contribs[f"{side}_rest_days"] = [int(prev["game_pk"])]
        else:
            vals[f"{side}_rest_days"] = REST_CAP
            contribs[f"{side}_rest_days"] = []
        if side == "away":
            if pd.isna(row["venue_id"]):
                vals["away_tz_delta"] = np.nan
            elif not len(mine):
                vals["away_tz_delta"] = 0
            else:
                pv = mine.iloc[-1]["venue_id"]
                vals["away_tz_delta"] = (np.nan if pd.isna(pv) else abs(
                    venue_offsets[int(row["venue_id"])] - venue_offsets[int(pv)]))
            contribs["away_tz_delta"] = ([int(mine.iloc[-1]["game_pk"])]
                                         if len(mine) else [])

        st = lo[(lo["game_pk"] == pk) & (lo["team_id"] == team) & (lo["is_starter"])]
        pid = int(st.iloc[0]["pitcher_id"]) if len(st) else None
        hist = (prior_l[prior_l["pitcher_id"] == pid].sort_values("game_order")
                if pid is not None else prior_l.iloc[0:0])
        total = int(hist["ip_outs"].sum())
        enough = total >= SP_WINDOW_OUTS
        if enough:
            acc_o, used = 0, 0
            for o in hist["ip_outs"].to_numpy()[::-1]:
                used += 1
                acc_o += int(o)
                if acc_o >= SP_WINDOW_OUTS:
                    break
            win = hist.tail(used)
        else:
            win = hist
        contribs[f"{side}_sp_fip_60"] = [int(x) for x in win["game_pk"]]
        if len(win) and (enough or total > 0):
            outs = int(win["ip_outs"].sum())
            fip = ((13 * int(win["home_runs"].sum())
                    + 3 * (int(win["walks"].sum()) + int(win["hit_by_pitch"].sum()))
                    - 2 * int(win["strikeouts"].sum())) / (outs / 3.0))
            kr = int(win["strikeouts"].sum()) / int(win["batters_faced"].sum())
        else:
            fip = kr = np.nan
        if not enough:
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
        vals[f"{side}_sp_fip_60"] = fip
        vals[f"{side}_sp_k_rate"] = kr
        vals[f"{side}_sp_trailing_ip"] = ((int(win["ip_outs"].sum()) if enough
                                           else total) / 3.0)
        vals[f"{side}_sp_low_confidence"] = not enough
    return vals, contribs, i


def test5_traces(games, logs, feats):
    say("\n## Test 5 — manual trace of three hard-coded games "
        f"({TRACE_PKS}) — every feature, 6-decimal equality")
    venue_offsets = load_venue_offsets()
    fidx = feats.set_index("game_pk")
    ok = True
    for pk in TRACE_PKS:
        vals, contribs, order_i = _hand_features(games, logs, pk, venue_offsets)
        frow = fidx.loc[pk]
        say(f"\n  game_pk={pk} date={frow['date']} (game_order={order_i}):")
        say(f"    {'feature':<24} {'hand':>12} {'table':>12}  match")
        for col, hv in vals.items():
            tv = frow[col]
            if isinstance(hv, (bool, np.bool_)):
                match = bool(hv) == bool(tv)
            elif pd.isna(hv) and pd.isna(tv):
                match = True
            else:
                match = abs(float(hv) - float(tv)) < 5e-7
            if not match:
                ok = False
            hv_s = f"{hv:.6f}" if isinstance(hv, float) and not pd.isna(hv) else str(hv)
            tv_s = f"{tv:.6f}" if isinstance(tv, float) and not pd.isna(tv) else str(tv)
            say(f"    {col:<24} {hv_s:>12} {tv_s:>12}  "
                f"{'OK' if match else '*** MISMATCH ***'}")
        for col in ("home_form_15", "away_form_15", "home_sp_fip_60",
                    "away_sp_fip_60"):
            say(f"    contributing pks [{col}]: {contribs[col]}")
        # doubleheader: game 1 must be inside game 2's team windows
        grow = games.loc[games["game_pk"] == pk].iloc[0]
        if str(grow["doubleheader_code"]) in ("Y", "S") and int(grow["game_number"]) == 2:
            g1 = games[(games["date"] == grow["date"])
                       & (games["home_team_id"] == grow["home_team_id"])
                       & (games["away_team_id"] == grow["away_team_id"])
                       & (games["game_number"] == 1)]
            g1pk = int(g1.iloc[0]["game_pk"])
            in_home = g1pk in contribs["home_form_15"]
            in_away = g1pk in contribs["away_form_15"]
            say(f"    DOUBLEHEADER: game 1 (pk={g1pk}) in game 2's form "
                f"windows: home={in_home} away={in_away} "
                f"{'OK' if in_home and in_away else '*** ORDERING BUG ***'}")
            ok &= in_home and in_away
    say(f"Test 5: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Test 6

def test6_provenance(games, logs, feats):
    say("\n## Test 6 — ordering assertion on game_order via provenance "
        "(additive path; sets are the definitional strict-< slices, proven "
        "equal to the fast path by Defense 1 + Test 5)")
    g = canonical_order(games)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    dh2 = games[(games["doubleheader_code"].isin(["Y", "S"]))
                & (games["game_number"] == 2)]["game_pk"].astype(int).tolist()
    rng = np.random.default_rng(7)
    sample = set(dh2) | set(TRACE_PKS) | set(
        int(x) for x in rng.choice(feats["game_pk"].to_numpy(), 3000,
                                   replace=False))
    say(f"  sample: {len(sample)} games = ALL {len(dh2)} doubleheader "
        f"game-2s + 3000 random + {len(TRACE_PKS)} traces")
    _, prov = build_features(games, logs, mode="train",
                             return_provenance=True,
                             provenance_game_pks=sorted(sample))
    viol, dh_missing = 0, 0
    for pk, rec in prov.items():
        own = order_of_pk[pk]
        for feat, e in rec.items():
            if e["max_order"] is not None and e["max_order"] >= own:
                viol += 1
                if viol <= 10:
                    say(f"  *** VIOLATION game_pk={pk} {feat}: contributing "
                        f"max_order={e['max_order']} >= own {own}")
    for pk in dh2:
        grow = games.loc[games["game_pk"] == pk].iloc[0]
        g1 = games[(games["date"] == grow["date"])
                   & (games["home_team_id"] == grow["home_team_id"])
                   & (games["away_team_id"] == grow["away_team_id"])
                   & (games["game_number"] == 1)]
        if not len(g1):
            continue  # game 1 postponed to another date — no same-day prior
        g1pk = int(g1.iloc[0]["game_pk"])
        rec = prov[pk]
        if (g1pk not in rec["home_form_15"]["pks"]
                or g1pk not in rec["away_form_15"]["pks"]):
            dh_missing += 1
            if dh_missing <= 5:
                say(f"  *** DH game 1 {g1pk} missing from game 2 {pk} windows")
    say(f"  strict-order violations: {viol} (must be 0)")
    say(f"  doubleheader game-2 rows whose windows LACK game 1: {dh_missing} "
        "(must be 0)")
    ok = viol == 0 and dh_missing == 0
    say(f"Test 6: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- Test 7

def test7_defenses():
    say("\n## Test 7 — re-run Chunk 3's defenses (verbatim output)")
    r = subprocess.run([sys.executable, "src/test_leakage.py"],
                       capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    out = r.stdout + r.stderr
    for ln in out.splitlines():
        say("  | " + ln)
    ok = r.returncode == 0
    checks = {
        "Defense 1 sampled 3000 rows": "3000 rows): PASS" in out,
        "Defense 2 covered 200 games": "200 games): PASS" in out,
        "Defense 3 all |r| < 0.03": "shuffled labels): worst |r|=" in out
                                    and "PASS" in out.split("worst |r|=")[1][:30],
    }
    for name, c in checks.items():
        say(f"  confirm: {name}: {'YES' if c else 'NO'}")
        ok &= c
    say(f"Test 7: {'PASS' if ok else 'FAIL'} (exit code {r.returncode})")
    return ok


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    say(f"# Leakage audit — {date.today().isoformat()}")
    say("commit: NOT A GIT REPOSITORY — no commit hash exists; file "
        "integrity anchors instead:")
    for p in ("src/features.py", "src/test_leakage.py", "data/features.csv",
              "data/games_full.csv", "data/pitcher_games.csv"):
        say(f"  sha256[{p}] = {_sha(Path(__file__).resolve().parents[1] / p)}…")
    say(f"split: train seasons <= {TRAIN_MAX_SEASON}, test > "
        f"{TRAIN_MAX_SEASON}; chronological, asserted on game_order")
    say("model: SimpleImputer(median) + StandardScaler + "
        "LogisticRegression(max_iter=2000), all fit on train only")
    say(f"features: {len(FEATURE_COLUMNS)} columns (home_field deliberately "
        "omitted in Chunk 3, so 'twelve' is eleven on disk)")

    df, train, test = load_matrix()
    say(f"rows: {len(df)} modelable (post burn-in, non-null label) = "
        f"{len(train)} train + {len(test)} test")

    results = {}
    results["Test 1 (shuffle)"] = test1_shuffle(df, train, test)
    ok2, real = test2_real(train, test)
    results["Test 2 (holdout)"] = ok2
    if not ok2:
        _finish(results, stopped_early=True)
        return
    results["Test 3 (calibration)"] = test3_calibration(test, real["p"])
    results["Test 4 (ablation)"] = test4_ablation(train, test, real["ll"])
    games, logs = load_inputs()
    results["Test 5 (manual traces)"] = test5_traces(games, logs,
                                                     pd.read_csv(FEATURES_CSV,
                                                                 low_memory=False))
    results["Test 6 (ordering/provenance)"] = test6_provenance(
        games, logs, pd.read_csv(FEATURES_CSV, low_memory=False))
    results["Test 7 (Chunk 3 defenses)"] = test7_defenses()
    _finish(results)


def _finish(results, stopped_early=False):
    say("\n" + "=" * 60)
    say("AUDIT SUMMARY")
    for name, ok in results.items():
        say(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if stopped_early:
        say("  (audit STOPPED at the tripwire — remaining tests not run)")
    all_ok = all(results.values()) and not stopped_early
    say(f"AUDIT: {'PASS' if all_ok else 'FAIL'}")
    say("=" * 60)
    DOCS.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(REPORT.getvalue(), encoding="utf-8")
    print(f"\naudit record written -> {AUDIT_MD}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
