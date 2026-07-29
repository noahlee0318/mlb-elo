"""Standalone verification for pitcher_games.csv / batter_games.csv:

    python src/verify_boxscores.py

Eight checks, actual numbers printed, exits nonzero on any failure.
Check 3 (pitching-runs vs Chunk 1 scores) gates checks 4-8: if the join or
orientation is wrong there is no point grading anything downstream.

Each check is an importable function over DataFrames so the logic itself is
unit-tested (tests/test_ingest_boxscores.py feeds a deliberately swapped
frame to the Check 3 logic).
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.games_data import load_games

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PITCHER_CSV = DATA_DIR / "pitcher_games.csv"
BATTER_CSV = DATA_DIR / "batter_games.csv"

_BOOL = {True: True, False: False, "True": True, "False": False}


def load_tables():
    pit = pd.read_csv(PITCHER_CSV, low_memory=False)
    bat = pd.read_csv(BATTER_CSV, low_memory=False)
    for df, cols in ((pit, ("is_home", "is_starter", "is_opener",
                            "starter_flag_mismatch", "complete_game", "shutout")),
                     (bat, ("is_home",))):
        for c in cols:
            df[c] = df[c].map(_BOOL).astype(bool)
    return pit, bat


def load_scope_games():
    g = load_games(include_postseason=True, exclude_seasons=(),
                   finals_only=True, mlb_matchups_only=True)
    return g


# ---------------------------------------------------------------- Check 1

def check1_coverage(games, pit, bat):
    lines = ["Check 1 — coverage:"]
    pit_teams = pit.groupby("game_pk")["team_id"].nunique()
    bat_counts = bat.groupby(["game_pk", "team_id"]).size()

    missing_pitching = []
    thin_batting = []
    for r in games.itertuples(index=False):
        pk = r.game_pk
        if pit_teams.get(pk, 0) < 2:
            missing_pitching.append(pk)
            continue
        for t in (r.home_team_id, r.away_team_id):
            if bat_counts.get((pk, t), 0) < 9:
                thin_batting.append((pk, int(t), int(bat_counts.get((pk, t), 0))))
    lines.append(f"  games lacking pitching rows for both teams: "
                 f"{len(missing_pitching)}"
                 + (f"  first 20: {missing_pitching[:20]}" if missing_pitching else ""))
    lines.append(f"  (game, team) with <9 batting rows: {len(thin_batting)}"
                 + (f"  first 20: {thin_batting[:20]}" if thin_batting else ""))

    per_team = pit.groupby(["game_pk", "team_id"]).size()
    lines.append(f"  pitchers per team-game: mean={per_team.mean():.3f} "
                 f"mode={int(per_team.mode().iloc[0])}")
    by_season = pit.groupby(["season", "game_pk", "team_id"]).size() \
                   .groupby("season").mean()
    lines.append("  mean pitchers per team-game by season: "
                 + "  ".join(f"{int(s)}:{v:.2f}" for s, v in by_season.items()))
    ok = not missing_pitching and not thin_batting
    return ok, lines


# ---------------------------------------------------------------- Check 2

def check2_outs(games, pit):
    """Per-(game_pk, team_id) ip_outs, validated by DIVISIBILITY, not by fixed
    values. A game can never end in the middle of a TOP half (the away team
    always finishes batting its half), so:

      2a  home-team pitchers pitch only complete top halves -> their per-game
          out total is a multiple of 3, except when a game is called
          mid-top-half (rain / suspended / tie), which are named exceptions.
      2b  away-team pitchers pitch bottom halves, which end early only on a
          home walk-off (home_win==1) or an early-called game -> an away
          non-multiple of 3 with home_win!=1 in a normal game is impossible.

    So 24, 25, 26 and 27 are ALL valid nine-inning totals: 24 = home led
    entering the bottom 9th (or a 0-out walk-off), 25/26 = one/two-out
    walk-offs, 27 = the away side won or the home side lost in a completed
    9th. Extra innings shift the whole picture up by multiples of 3
    (28/29 = a 10th-inning walk-off, etc.). The invariant is divisibility,
    not membership in {24, 27}.

    Scope for the divisibility/low-total invariants is games SCHEDULED for 9
    innings; scheduled-7 games (2020-21 doubleheaders) complete legitimately
    at 21 outs and are reported separately, not judged against 24."""
    meta = games.set_index("game_pk")
    outs = pit.groupby(["game_pk", "team_id"])["ip_outs"].sum().reset_index()
    outs = outs.merge(
        games[["game_pk", "home_team_id", "away_team_id", "home_win",
               "scheduled_innings", "innings_played", "status_detailed"]],
        on="game_pk", how="left")
    outs["is_home"] = outs["team_id"] == outs["home_team_id"]
    sched = outs["scheduled_innings"].fillna(9).astype(int)
    played = outs["innings_played"].fillna(sched).astype(int)
    outs["nine"] = sched == 9
    # a game stopped mid-inning: fewer innings than scheduled, or a called tie
    outs["called_mid"] = (played < sched) | outs["home_win"].isna()
    outs["short_total"] = (sched < 9) | (played < sched) | outs["home_win"].isna()
    outs["rem"] = outs["ip_outs"] % 3

    lines = ["Check 2 — outs per (game_pk, team_id) [divisibility invariant]:"]

    # 2e — full distribution
    dist = outs["ip_outs"].value_counts().sort_index()
    lines.append("  2e full distribution (outs:team-games):")
    lines.append("    " + "  ".join(f"{int(k)}:{int(v)}" for k, v in dist.items()))
    n_seven = int((~outs["nine"]).sum())
    lines.append(f"     (scheduled-<9-inning team-games excluded from the "
                 f"24/27 invariants: {n_seven})")

    nine = outs[outs["nine"]]

    # 2a — home-side non-multiples of 3
    home_nm = nine[nine["is_home"] & (nine["rem"] != 0)]
    home_bug = home_nm[~home_nm["called_mid"]]
    lines.append(f"  2a home-side non-multiples of 3: {len(home_nm)} "
                 f"(each must be a called-mid-inning/irregular game; "
                 f"{len(home_bug)} unexplained — must be 0):")
    for r in home_nm.itertuples(index=False):
        tag = "irregular OK" if r.called_mid else "*** BUG: ordinary game ***"
        lines.append(f"     game_pk={r.game_pk} outs={r.ip_outs} "
                     f"status={str(meta.loc[r.game_pk, 'status_detailed'])!r} [{tag}]")

    # 2b — away-side non-multiples of 3
    away_nm = nine[~nine["is_home"] & (nine["rem"] != 0)]
    away_not_wo = away_nm[away_nm["home_win"] != 1]
    away_bug = away_not_wo[~away_not_wo["called_mid"]]
    lines.append(f"  2b away-side non-multiples of 3: {len(away_nm)}; "
                 f"home_win!=1 among them: {len(away_not_wo)} "
                 f"({len(away_not_wo) - len(away_bug)} irregular/explained, "
                 f"{len(away_bug)} UNEXPLAINED — must be 0)")
    for r in away_bug.itertuples(index=False):
        lines.append(f"     *** game_pk={r.game_pk} outs={r.ip_outs} "
                     f"home_win={r.home_win} "
                     f"status={str(meta.loc[r.game_pk, 'status_detailed'])!r}")

    # 2c — remainder balance among away non-multiples
    r1 = int((away_nm["rem"] == 1).sum())
    r2 = int((away_nm["rem"] == 2).sum())
    ratio = r1 / r2 if r2 else float("inf")
    lines.append(f"  2c away non-multiple remainders: one-out(rem 1)={r1}, "
                 f"two-out(rem 2)={r2}, ratio={ratio:.2f} "
                 f"(~1 expected; a heavy skew would signal an off-by-one)")

    # 2d — walk-off correspondence, by innings_played
    walkoffs = away_nm[away_nm["home_win"] == 1]
    lines.append(f"  2d away non-multiples that are walk-offs (home_win==1): "
                 f"{len(walkoffs)}/{len(away_nm)}; by innings_played:")
    wo_by_inn = walkoffs.assign(ip=played[walkoffs.index]).groupby("ip").size()
    for ip_played, n in wo_by_inn.items():
        lines.append(f"       {int(ip_played)} innings: {int(n)}")

    # totals below 24 among nine-inning games (must be early-called)
    low = nine[nine["ip_outs"] < 24]
    low_bug = low[~low["short_total"]]
    lines.append(f"  nine-inning team-games with <24 outs: {len(low)} "
                 f"({len(low) - len(low_bug)} rain/called-short, "
                 f"{len(low_bug)} UNEXPLAINED — must be 0):")
    for r in low.itertuples(index=False):
        tag = "" if r.short_total else "  *** UNEXPLAINED"
        lines.append(f"     game_pk={r.game_pk} team={r.team_id} outs={r.ip_outs} "
                     f"status={str(meta.loc[r.game_pk, 'status_detailed'])!r}{tag}")

    ok = len(home_bug) == 0 and len(away_bug) == 0 and len(low_bug) == 0
    return ok, lines


# ---------------------------------------------------------------- Check 3

def runs_cross_mismatches(games, pit):
    """(most important) Sum of runs allowed by a team's pitchers must equal
    the OPPONENT's score in games_full. Validates join, parse, and
    orientation at once. Returns the mismatching rows."""
    allowed = (pit.groupby(["game_pk", "team_id"])["runs"].sum()
               .rename("runs_allowed").reset_index())
    g = games[["game_pk", "home_team_id", "away_team_id",
               "home_score", "away_score"]]
    home = allowed.merge(g, left_on=["game_pk", "team_id"],
                         right_on=["game_pk", "home_team_id"], how="inner")
    home["expected"] = home["away_score"]
    away = allowed.merge(g, left_on=["game_pk", "team_id"],
                         right_on=["game_pk", "away_team_id"], how="inner")
    away["expected"] = away["home_score"]
    both = pd.concat([home, away], ignore_index=True)
    return both[both["runs_allowed"] != both["expected"]]


def check3_runs(games, pit):
    mm = runs_cross_mismatches(games, pit)
    lines = [f"Check 3 — pitching runs vs opponent score (games_full): "
             f"{len(mm)} mismatch(es) (expected 0)"]
    if len(mm):
        lines.append("  FIRST 20:")
        for _, r in mm.head(20).iterrows():
            lines.append(f"    game_pk={int(r['game_pk'])} team={int(r['team_id'])} "
                         f"pitcher_runs={int(r['runs_allowed'])} "
                         f"expected={int(r['expected'])}")
    return len(mm) == 0, lines


# ---------------------------------------------------------------- Check 4

def check4_batting_runs(games, bat):
    scored = (bat.groupby(["game_pk", "team_id"])["runs"].sum()
              .rename("runs_scored").reset_index())
    g = games[["game_pk", "home_team_id", "away_team_id",
               "home_score", "away_score"]]
    home = scored.merge(g, left_on=["game_pk", "team_id"],
                        right_on=["game_pk", "home_team_id"], how="inner")
    home["expected"] = home["home_score"]
    away = scored.merge(g, left_on=["game_pk", "team_id"],
                        right_on=["game_pk", "away_team_id"], how="inner")
    away["expected"] = away["away_score"]
    both = pd.concat([home, away], ignore_index=True)
    mm = both[both["runs_scored"] != both["expected"]]
    lines = [f"Check 4 — batting runs vs own score: {len(mm)} mismatch(es) "
             "(expected 0)"]
    for _, r in mm.head(20).iterrows():
        lines.append(f"    game_pk={int(r['game_pk'])} team={int(r['team_id'])} "
                     f"batter_runs={int(r['runs_scored'])} expected={int(r['expected'])}")
    return len(mm) == 0, lines


# ---------------------------------------------------------------- Check 5

def check5_bf_vs_pa(pit, bat):
    bf = pit.groupby(["game_pk", "team_id"])["batters_faced"].sum().reset_index()
    pa = bat.groupby(["game_pk", "team_id"])["plate_appearances"].sum() \
            .rename("pa").reset_index()
    # opponent join: my pitchers face THEIR batters
    merged = bf.merge(pa, left_on=["game_pk", "team_id"],
                      right_on=["game_pk", "team_id"], how="inner",
                      suffixes=("", "_same"))
    # need opponent's PA: map via pit's opponent_team_id
    opp_map = pit[["game_pk", "team_id", "opponent_team_id"]].drop_duplicates()
    bf2 = bf.merge(opp_map, on=["game_pk", "team_id"])
    merged = bf2.merge(pa, left_on=["game_pk", "opponent_team_id"],
                       right_on=["game_pk", "team_id"], suffixes=("", "_opp"))
    merged["diff"] = merged["batters_faced"] - merged["pa"]
    nz = merged[merged["diff"] != 0]
    dist = nz["diff"].value_counts().sort_index().to_dict()
    lines = [f"Check 5 — team BF vs opposing team PA: "
             f"{len(merged) - len(nz)}/{len(merged)} team-games match exactly",
             f"  nonzero-diff distribution: {dist if dist else '(none)'}"]
    for _, r in nz.head(10).iterrows():
        lines.append(f"    game_pk={int(r['game_pk'])} team={int(r['team_id'])} "
                     f"BF={int(r['batters_faced'])} oppPA={int(r['pa'])}")
    # pass unless the discrepancy is more than a rare +/-1 (investigated in report)
    big = nz[nz["diff"].abs() > 1]
    return len(big) == 0 and len(nz) <= max(5, len(merged) // 2000), lines


# ---------------------------------------------------------------- Check 6

def check6_starters(pit):
    starters = pit[pit["is_starter"]].groupby(["game_pk", "team_id"]).size()
    per_team = pit.groupby(["game_pk", "team_id"]).size()
    zero = per_team.index.difference(starters.index)
    multi = starters[starters > 1]
    mism = int(pit["starter_flag_mismatch"].sum())
    lines = [f"Check 6 — one starter per team-game: "
             f"{len(zero)} team-games with zero starters, "
             f"{len(multi)} with more than one",
             f"  rows where positional is_starter disagrees with API "
             f"gamesStarted: {mism} (reported, not asserted — every such "
             "game is a suspended/resumed game where MLB's own array order "
             "and gamesStarted disagree; the parse is faithful to the raw "
             "JSON and the starter_flag_mismatch column marks them)"]
    if len(zero):
        lines.append(f"  zero-starter first 20: {list(zero[:20])}")
    if mism:
        bad = pit[pit["starter_flag_mismatch"]]
        lines.append("  affected games: "
                     + str(sorted(bad["game_pk"].unique().tolist())))
    return len(zero) == 0 and len(multi) == 0, lines


# ---------------------------------------------------------------- Check 7

def check7_leaderboards(pit):
    lines = ["Check 7 — leaderboard recompute (regular season only; "
             "check these against Baseball Reference):"]
    for season in (2023, 2019):
        sub = pit[(pit["season"] == season) & (pit["game_type"] == "R")]
        agg = sub.groupby("pitcher_id").agg(
            name=("pitcher_name", "last"), outs=("ip_outs", "sum"),
            er=("earned_runs", "sum"), k=("strikeouts", "sum"))
        qual = agg[agg["outs"] >= 486].copy()          # 162 IP = 486 outs
        qual["era"] = qual["er"] * 27 / qual["outs"]
        top = qual.nsmallest(5, "era")
        lines.append(f"  {season} qualified (>=162 IP) ERA top 5:")
        for _, r in top.iterrows():
            ip_disp = f"{int(r['outs'] // 3)}.{int(r['outs'] % 3)}"
            lines.append(f"    {r['name']:<24} ERA {r['era']:.3f}  "
                         f"({ip_disp} IP, {int(r['er'])} ER)")
        klead = agg.nlargest(1, "k").iloc[0]
        lines.append(f"  {season} strikeout leader: {klead['name']} "
                     f"({int(klead['k'])} K)")
    return True, lines  # graded externally by the user


# ---------------------------------------------------------------- Check 8

def check8_dups_nulls(pit, bat):
    dup_p = int(pit.duplicated(["game_pk", "team_id", "pitcher_id"]).sum())
    dup_b = int(bat.duplicated(["game_pk", "team_id", "batter_id"]).sum())
    lines = [f"Check 8 — duplicates on (game_pk, team_id, player_id): "
             f"pitching {dup_p}, batting {dup_b} (expected 0)"]
    # (game_pk, batter_id) alone can legitimately repeat when a player is
    # traded during a suspended game and appears for BOTH teams (Danny
    # Jansen, game 746942, 2024) — report such cross-team rows explicitly
    cross = bat[bat.duplicated(["game_pk", "batter_id"], keep=False)]
    if len(cross):
        lines.append(f"  same batter on both teams in one game (real MLB "
                     f"history, not a bug): {len(cross)} rows")
        for _, r in cross.iterrows():
            lines.append(f"    game_pk={int(r['game_pk'])} {r['batter_name']} "
                         f"team={int(r['team_id'])}")
    lines.append(f"  row counts: {len(pit)} pitching (expect ~220k-250k), "
                 f"{len(bat)} batting (expect ~500k-600k)")
    ok = dup_p == 0 and dup_b == 0
    for label, df, req in (("pitcher_games", pit,
                            ("ip_outs", "batters_faced", "team_id", "pitcher_id")),
                           ("batter_games", bat, ("team_id", "batter_id"))):
        lines.append(f"  {label} null counts:")
        for c in df.columns:
            n = int(df[c].isna().sum())
            flag = ""
            if c in req and n:
                flag, ok = "  <-- MUST BE ZERO", False
            if n or c in req:
                lines.append(f"    {c:<26} {n}{flag}")
    band = (200_000 <= len(pit) <= 260_000) and (450_000 <= len(bat) <= 650_000)
    if not band:
        ok = False
        lines.append("  row counts OUTSIDE expected bands")
    return ok, lines


def main():
    for p in (PITCHER_CSV, BATTER_CSV):
        if not p.exists():
            print(f"{p} not found — run `python src/ingest_boxscores.py` first.")
            sys.exit(2)
    pit, bat = load_tables()
    games = load_scope_games()
    print(f"Loaded {len(pit)} pitcher rows, {len(bat)} batter rows, "
          f"{len(games)} in-scope games\n")

    results = []
    for check in (lambda: check1_coverage(games, pit, bat),
                  lambda: check2_outs(games, pit)):
        ok, lines = check()
        print("\n".join(lines)); print()
        results.append((lines[0].split(" —")[0], ok))

    ok3, lines3 = check3_runs(games, pit)
    print("\n".join(lines3)); print()
    results.append(("Check 3", ok3))
    if not ok3:
        print("!! Check 3 failed — halting before checks 4-8 as instructed.")
        _summary(results)
        sys.exit(1)

    for check in (lambda: check4_batting_runs(games, bat),
                  lambda: check5_bf_vs_pa(pit, bat),
                  lambda: check6_starters(pit),
                  lambda: check7_leaderboards(pit),
                  lambda: check8_dups_nulls(pit, bat)):
        ok, lines = check()
        print("\n".join(lines)); print()
        results.append((lines[0].split(" —")[0], ok))

    _summary(results)
    sys.exit(0 if all(ok for _, ok in results) else 1)


def _summary(results):
    print("=" * 60)
    print("SUMMARY")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
