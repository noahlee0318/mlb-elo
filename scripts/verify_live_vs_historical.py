"""Chunk 9, Deliverable 4 — the skew test (this chunk's headline verification).

Proves the LIVE feature path (src/live_features.build_game_state) reproduces the
Chunk 3 historical feature vector (data/features.csv) for the same game as of
the same date, column by column, to 1e-9 on floats and exactly on ints/bools.
A single mismatch fails the run: a systematic mismatch is training/serving
skew, which silently invalidates every live prediction.

Reproducibility: on first run it samples games, reconstructs them, asserts, and
freezes the sample (inputs + expected historical vectors) to
data/eval/skew_fixture.json. On later runs it asserts build_game_state against
that FIXED fixture — a regression weeks from now is reproducible against a fixed
input, not a schedule that has since changed. It ALSO runs the live path
against today's actual slate when games exist (structural checks only, since an
unplayed game has no historical row yet); that branch is the one non-reproducible
part and is quarantined here.

Usage:
  python scripts/verify_live_vs_historical.py            # fixture (build if absent) + today
  python scripts/verify_live_vs_historical.py --rebuild  # force re-sample the fixture
  python scripts/verify_live_vs_historical.py --n 250    # sample size on (re)build
  python scripts/verify_live_vs_historical.py --no-slate # skip the network slate branch
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features_ml import FEATURE_COLUMNS
from src.games_data import load_games
from src.live_features import build_game_state, COLD_START_MIN_GAMES

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "features.csv"
FIXTURE = ROOT / "data" / "eval" / "skew_fixture.json"

FLOAT_TOL = 1e-9
SAMPLE_SEASON = 2024
RNG_SEED = 42


def _feats_table():
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    for c in FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _sample_games(n):
    """n games from SAMPLE_SEASON, stratified early/mid/late, with some
    low_confidence. Deterministic given RNG_SEED."""
    games = load_games()
    feats = _feats_table()
    g = games[games["season"] == SAMPLE_SEASON].merge(
        feats[["game_pk", "home_sp_low_confidence", "away_sp_low_confidence"]],
        on="game_pk", how="inner")
    g["low_conf"] = (g["home_sp_low_confidence"].astype(str).isin(["True", "true"])
                     | g["away_sp_low_confidence"].astype(str).isin(["True", "true"]))
    g = g.sort_values(["date", "game_number", "game_pk"]).reset_index(drop=True)

    rng = np.random.default_rng(RNG_SEED)
    # stratify by month buckets so early/mid/late are all represented
    g["month"] = g["date"].str.slice(5, 7)
    buckets = {"early": ["03", "04", "05"], "mid": ["06", "07"],
               "late": ["08", "09", "10"]}
    per = max(1, n // 3)
    picks = []
    for names in buckets.values():
        sub = g[g["month"].isin(names)]
        take = min(per, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    chosen = pd.concat(picks)
    # guarantee some low_confidence games are in the set
    lc = g[g["low_conf"] & ~g["game_pk"].isin(set(chosen["game_pk"]))]
    if len(lc):
        add = lc.iloc[rng.choice(len(lc), min(max(10, n // 10), len(lc)),
                                 replace=False)]
        chosen = pd.concat([chosen, add])
    chosen = chosen.drop_duplicates("game_pk").sort_values(
        ["date", "game_number", "game_pk"]).reset_index(drop=True)
    return chosen


def _expected_vector(feats, game_pk):
    row = feats[feats["game_pk"] == game_pk].iloc[0]
    return {c: (None if pd.isna(row[c]) else float(row[c]))
            for c in FEATURE_COLUMNS}


def _compare(live_feats, expected, game_pk):
    """Return list of (col, live, exp, delta) mismatches beyond tolerance."""
    bad = []
    for c in FEATURE_COLUMNS:
        lv = live_feats[c]
        ev = expected[c]
        lv_na = lv is None or (isinstance(lv, float) and np.isnan(lv))
        ev_na = ev is None
        if lv_na and ev_na:
            continue
        if lv_na != ev_na:
            bad.append((c, lv, ev, "NA-mismatch"))
            continue
        d = abs(float(lv) - float(ev))
        if d > FLOAT_TOL:
            bad.append((c, float(lv), float(ev), d))
    return bad


def build_fixture(n):
    games = load_games()
    feats = _feats_table()
    chosen = _sample_games(n)
    print(f"[fixture] sampling {len(chosen)} games from {SAMPLE_SEASON} "
          f"(early/mid/late; {int(chosen['low_conf'].sum())} low_confidence)")
    entries, mismatches = [], 0
    for i, r in enumerate(chosen.itertuples(index=False), 1):
        pk = int(r.game_pk)
        st = build_game_state(int(r.home_team_id), int(r.away_team_id),
                              r.date, game_number=int(r.game_number))
        live = st["context"]["features"]
        exp = _expected_vector(feats, pk)
        bad = _compare(live, exp, pk)
        if bad:
            mismatches += 1
            for col, lv, ev, d in bad:
                print(f"  MISMATCH pk={pk} {col}: live={lv} hist={ev} delta={d}")
        entries.append({
            "game_pk": pk, "home_id": int(r.home_team_id),
            "away_id": int(r.away_team_id), "date": r.date,
            "game_number": int(r.game_number),
            "expected": exp,
            "games_played_home": st["home_state"]["games_played"],
            "games_played_away": st["away_state"]["games_played"],
            "low_confidence": st["context"]["low_confidence"],
        })
        if i % 25 == 0:
            print(f"  ...{i}/{len(chosen)} reconstructed")
    if mismatches:
        print(f"\nFIXTURE BUILD FAILED: {mismatches} game(s) mismatched — skew.",
              file=sys.stderr)
        sys.exit(1)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({
        "created_season": SAMPLE_SEASON, "n": len(entries),
        "tolerance": FLOAT_TOL, "feature_columns": FEATURE_COLUMNS,
        "rng_seed": RNG_SEED, "games": entries,
    }, indent=1), encoding="utf-8")
    print(f"[fixture] wrote {FIXTURE} ({len(entries)} games, all match to "
          f"{FLOAT_TOL})")
    return len(entries)


def check_against_fixture():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fx["feature_columns"] == FEATURE_COLUMNS, \
        "fixture FEATURE_COLUMNS differ from current features_ml order"
    games = fx["games"]
    print(f"[fixture] checking {len(games)} frozen games (tol {fx['tolerance']})")
    mismatches, gp_bad = 0, 0
    for i, e in enumerate(games, 1):
        st = build_game_state(e["home_id"], e["away_id"], e["date"],
                              game_number=e["game_number"])
        live = st["context"]["features"]
        bad = _compare(live, e["expected"], e["game_pk"])
        if bad:
            mismatches += 1
            for col, lv, ev, d in bad:
                print(f"  MISMATCH pk={e['game_pk']} {col}: live={lv} "
                      f"hist={ev} delta={d}")
        # ints exact
        if (st["home_state"]["games_played"] != e["games_played_home"]
                or st["away_state"]["games_played"] != e["games_played_away"]):
            gp_bad += 1
            print(f"  GP-MISMATCH pk={e['game_pk']}: "
                  f"live=({st['home_state']['games_played']},"
                  f"{st['away_state']['games_played']}) "
                  f"fixture=({e['games_played_home']},{e['games_played_away']})")
        if i % 50 == 0:
            print(f"  ...{i}/{len(games)} checked")
    ok = mismatches == 0 and gp_bad == 0
    print(f"[fixture] result: {'PASS' if ok else 'FAIL'} — "
          f"{len(games)-mismatches}/{len(games)} vectors match, "
          f"{gp_bad} games_played mismatches")
    return ok, len(games), mismatches, gp_bad


def check_today_slate():
    """Live path against today's real slate — structural checks only (an
    unplayed game has no historical row). Best-effort: no network / no games is
    a normal skip, not a failure."""
    try:
        from src.schedule import fetch_slate, todays_date_str
        ds = todays_date_str()
        slate = fetch_slate(ds)
    except Exception as exc:                       # offline, API down, etc.
        print(f"[slate] skipped (could not fetch today's slate: {exc})")
        return True, 0
    if not slate:
        print("[slate] no regular-season games today — nothing to check")
        return True, 0
    print(f"[slate] {ds}: {len(slate)} game(s), live structural checks")
    bad = 0
    for g in slate:
        try:
            st = build_game_state(g["home_id"], g["away_id"], ds)
        except Exception as exc:
            print(f"  {g['away_name']} @ {g['home_name']}: build failed ({exc})")
            bad += 1
            continue
        feats = st["context"]["features"]
        # structural: right columns, and any NaN must be flagged low_confidence
        if list(feats.keys()) != FEATURE_COLUMNS:
            print(f"  column-order mismatch for {g['home_id']}"); bad += 1; continue
        nan_cols = [c for c in FEATURE_COLUMNS
                    if feats[c] is None or (isinstance(feats[c], float)
                                            and np.isnan(feats[c]))]
        gp_h = st["home_state"]["games_played"]
        gp_a = st["away_state"]["games_played"]
        cold = min(gp_h, gp_a) < COLD_START_MIN_GAMES
        # NaN features are only acceptable when cold-start (form/run_diff
        # undefined < 5-15 games) or a flagged low_confidence pitcher
        if nan_cols and not (cold or st["context"]["low_confidence"]):
            print(f"  unexpected NaN {nan_cols} for {g['home_id']} "
                  f"(gp={gp_h},{gp_a}) not flagged"); bad += 1; continue
        print(f"  {g['away_name']} @ {g['home_name']}: gp=({gp_h},{gp_a}) "
              f"cold={cold} low_conf={st['context']['low_confidence_reasons']}")
    return bad == 0, len(slate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-slate", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("SKEW TEST — live feature path vs Chunk 3 historical table")
    print("=" * 74)

    if args.rebuild or not FIXTURE.exists():
        n_fixture = build_fixture(args.n)
        fixture_ok, n_checked, mism, gp_bad = True, n_fixture, 0, 0
        # immediately verify the freshly-written fixture round-trips
        fixture_ok, n_checked, mism, gp_bad = check_against_fixture()
    else:
        fixture_ok, n_checked, mism, gp_bad = check_against_fixture()

    slate_ok, n_slate = (True, 0) if args.no_slate else check_today_slate()

    print("\n" + "=" * 74)
    print("REPORT")
    print("=" * 74)
    print(f"  fixture skew test : {'PASS' if fixture_ok else 'FAIL'} "
          f"({n_checked} games checked, {mism} vector mismatches, "
          f"{gp_bad} games_played mismatches)")
    print(f"  today's slate     : {n_slate} game(s) "
          f"{'checked' if n_slate else 'none/skipped'}, "
          f"{'ok' if slate_ok else 'PROBLEMS'}")
    if not (fixture_ok and slate_ok):
        print("\nTRIPWIRE FAILURE: skew or structural check failed",
              file=sys.stderr)
        sys.exit(1)
    print("  ALL CHECKS PASS")


if __name__ == "__main__":
    main()
