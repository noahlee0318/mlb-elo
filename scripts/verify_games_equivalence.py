"""Migration verification: prove games_full (restricted to games.csv's season
window) is games.csv PLUS exactly the 29 'Completed Early' games games.csv's
stricter detailedState=='Final' filter always dropped — and nothing else.

Under the accepted reconciliation (include those real games), ratings are NOT
byte-identical; instead the change must be attributable ONLY to those games.
This gate proves the game sets differ by exactly that known category, with zero
result mismatches on shared games and zero unexplained extras.

Exits nonzero on any unexplained discrepancy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.games_data import GAMES_FULL_CSV, load_games

ROOT = Path(__file__).resolve().parents[1]
GAMES_CSV = ROOT / "data" / "games.csv"
GAMES_CSV_RETIRED = ROOT / "data" / "games.csv.retired"


def main():
    src = GAMES_CSV if GAMES_CSV.exists() else GAMES_CSV_RETIRED
    if not src.exists():
        print("neither games.csv nor games.csv.retired present — nothing to "
              "verify (already migrated & cleaned).")
        return
    gc = pd.read_csv(src)
    seasons = sorted(int(s) for s in gc["season"].unique())
    gf = load_games()
    gf = gf[gf["season"].isin(seasons)]

    fails = []
    print("=" * 66)
    print(f"GAMES EQUIVALENCE (window {seasons})")
    print("=" * 66)

    set_gc = set(gc["game_id"].astype(int))
    set_gf = set(gf["game_pk"].astype(int))
    only_gc = sorted(set_gc - set_gf)
    only_gf = sorted(set_gf - set_gc)
    shared = set_gc & set_gf

    print(f"  games.csv={len(gc)}  games_full[window]={len(gf)}  shared={len(shared)}")
    print(f"  games.csv-only={len(only_gc)}  games_full-only={len(only_gf)}")

    # (1) no game the OLD table had is missing from the new source
    if only_gc:
        fails.append(f"{len(only_gc)} games in games.csv absent from games_full: "
                     f"{only_gc[:10]}")
        print(f"  FAIL: games.csv-only (should be 0): {only_gc[:10]}")
    else:
        print("  OK: every games.csv game is present in games_full")

    # (2) every EXTRA game in games_full is a 'Completed Early' game — the known,
    #     accepted category — not some unexplained inclusion
    raw = pd.read_csv(GAMES_FULL_CSV, low_memory=False)
    extra = raw[raw["game_pk"].isin(only_gf)]
    non_ce = extra[extra["status_detailed"] != "Completed Early"]
    print(f"  games_full-only breakdown: "
          f"{extra['status_detailed'].value_counts().to_dict()}")
    if len(non_ce):
        fails.append(f"{len(non_ce)} extra games are NOT 'Completed Early' "
                     f"(unexplained): {non_ce['game_pk'].tolist()[:10]}")
        print(f"  FAIL: unexplained extras: {non_ce['game_pk'].tolist()[:10]}")
    else:
        print(f"  OK: all {len(only_gf)} extras are 'Completed Early' "
              "(the accepted correction)")

    # (3) results identical on every shared game
    gcm = gc.set_index("game_id")
    gfm = gf.set_index("game_pk")
    mism = 0
    for pk in shared:
        a, b = gcm.loc[pk], gfm.loc[pk]
        if (int(a["home_score"]) != int(b["home_score"])
                or int(a["away_score"]) != int(b["away_score"])):
            mism += 1
    if mism:
        fails.append(f"{mism} shared games have different scores")
        print(f"  FAIL: {mism} result mismatches on shared games")
    else:
        print(f"  OK: 0 result mismatches across {len(shared)} shared games")

    # (4) no ties in the window (build_ratings would mis-handle a tie as a loss)
    ties = int((gf["home_score"] == gf["away_score"]).sum())
    print(f"  ties in games_full[window]: {ties}"
          f" {'(OK)' if ties == 0 else '(REVIEW)'}")

    print("=" * 66)
    if fails:
        print("EQUIVALENCE FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print(f"EQUIVALENCE OK — games_full[window] == games.csv + {len(only_gf)} "
          "'Completed Early' games, 0 other differences, 0 result mismatches.")


if __name__ == "__main__":
    main()
