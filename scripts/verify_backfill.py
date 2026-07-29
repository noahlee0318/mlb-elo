"""A — verification GATE for the 2026 backfill. Exits nonzero if any check
fails. Run after scripts/backfill_season.py; downstream work (re-deriving
features/models) must not proceed until this is green.

  python scripts/verify_backfill.py --season 2026 [--expect-pre 24295]

Checks (all via the sanctioned load_games accessor except where noted):
  1. No null-result rows entered — the corruption check. 2026 rows > 0 and
     null home_win / null scores both 0.
  2. Historical seasons untouched — pre-<season> load_games count unchanged.
  3. game_pk globally unique across the whole raw file (idempotent-append proof).
  4. A team spot-count is plausible for the date (not 0, not a full 162).
  5. The live feature path yields a real current-season games_played (>=15), so
     the model actually runs instead of cold-starting to Elo.
  6. Boxscore coverage — the season's final game_pks have pitcher-log rows, so
     2026 feature rows aren't all forced to low-confidence imputation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.games_data import GAMES_FULL_CSV, load_games

PITCHER_CSV = Path(__file__).resolve().parents[1] / "data" / "pitcher_games.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--expect-pre", type=int, default=24295,
                    help="expected pre-season load_games row count (unchanged)")
    ap.add_argument("--spot-team", type=int, default=134, help="Pirates")
    args = ap.parse_args()
    S = args.season

    g = load_games()
    fails = []
    print("=" * 66)
    print(f"BACKFILL VERIFICATION GATE — season {S}")
    print("=" * 66)

    # 1 — no null-result rows
    n = g[g["season"] == S]
    nw = int(n["home_win"].isna().sum())
    ns = int(n[["home_score", "away_score"]].isna().any(axis=1).sum())
    ok1 = len(n) > 0 and nw == 0 and ns == 0
    print(f"[1] null-result check: {S} rows={len(n)}  null_home_win={nw}  "
          f"null_scores={ns}  -> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        fails.append("1: null-result rows entered (scheduled game slipped in)")

    # 2 — historical untouched
    pre = int((g["season"] < S).sum())
    ok2 = pre == args.expect_pre
    print(f"[2] historical untouched: pre-{S} rows={pre} "
          f"(expected {args.expect_pre}) -> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        fails.append(f"2: pre-{S} count {pre} != {args.expect_pre}")

    # 3 — game_pk globally unique (raw file, no filtering)
    raw = pd.read_csv(GAMES_FULL_CSV, low_memory=False)
    dup = int(raw["game_pk"].duplicated().sum())
    ok3 = dup == 0
    print(f"[3] game_pk unique (raw {len(raw)} rows): dup={dup} "
          f"-> {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        fails.append(f"3: {dup} duplicate game_pk")

    # 4 — team spot-count plausible
    t = args.spot_team
    ts = g[((g.home_team_id == t) | (g.away_team_id == t)) & (g.season == S)]
    ok4 = 0 < len(ts) < 162
    rng = (f"{ts['date'].min()}..{ts['date'].max()}" if len(ts) else "n/a")
    print(f"[4] team {t} {S} spot-count: {len(ts)} games ({rng}) "
          f"-> {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        fails.append(f"4: team {t} has implausible {len(ts)} games")

    # 5 — live feature path yields real games_played
    from src.live_features import build_live_features
    import datetime
    y = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    # pick a team that actually has season-S games for a meaningful count
    tid = int(ts.iloc[0]["home_team_id"]) if len(ts) else t
    opp = int(ts.iloc[0]["away_team_id"]) if len(ts) else 111
    f = build_live_features(tid, opp, y, "home")
    ok5 = f["games_played"] >= 15
    print(f"[5] live games_played (team {tid} as of {y}): {f['games_played']} "
          f">=15 -> {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        fails.append(f"5: games_played {f['games_played']} < 15 — still cold-start")

    # 6 — boxscore coverage for season-S final game_pks
    reg = g[(g.season == S)]
    season_pks = set(reg["game_pk"].astype(int))
    if PITCHER_CSV.exists():
        pit = pd.read_csv(PITCHER_CSV, usecols=["game_pk", "season"],
                          low_memory=False)
        covered = set(pit.loc[pit["season"] == S, "game_pk"].astype(int))
        missing = season_pks - covered
        frac = 1 - len(missing) / len(season_pks) if season_pks else 0
        ok6 = frac >= 0.98
        print(f"[6] boxscore coverage {S}: {len(season_pks)-len(missing)}/"
              f"{len(season_pks)} final games have pitcher logs "
              f"({frac:.1%}) -> {'PASS' if ok6 else 'FAIL'}")
        if not ok6:
            fails.append(f"6: only {frac:.1%} boxscore coverage "
                         f"({len(missing)} missing)")
    else:
        print("[6] boxscore coverage: pitcher_games.csv missing -> FAIL")
        fails.append("6: pitcher_games.csv missing")

    print("=" * 66)
    if fails:
        print("GATE FAILED:")
        for f_ in fails:
            print(f"  - {f_}")
        sys.exit(1)
    print("GATE PASSED — all checks green.")


if __name__ == "__main__":
    main()
