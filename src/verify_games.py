"""Standalone verification for data/games_full.csv — re-runnable without
re-ingesting:

    python src/verify_games.py

Prints a full report and exits nonzero if any check fails. Each check is an
importable function taking a DataFrame, so the logic (notably the Check 4
isWinner cross-check) is unit-tested directly.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FULL_CSV = DATA_DIR / "games_full.csv"
GAMES_CSV = DATA_DIR / "games.csv"

_INT_COLS = [
    "game_pk", "season", "home_team_id", "away_team_id", "home_score",
    "away_score", "home_win", "home_wins_after", "home_losses_after",
    "away_wins_after", "away_losses_after", "game_number", "venue_id",
    "scheduled_innings", "innings_played", "home_probable_pitcher_id",
    "away_probable_pitcher_id", "winning_pitcher_id", "losing_pitcher_id",
]
_BOOL = {True: True, False: False, "True": True, "False": False}


def read_games_full(path=GAMES_FULL_CSV):
    """Read games_full.csv back with the dtypes the checks expect."""
    df = pd.read_csv(path, low_memory=False)
    for c in _INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["home_is_winner_flag"] = df["home_is_winner_flag"].map(_BOOL).astype("boolean")
    for c in ("is_final", "is_anomalous_season", "is_mlb_matchup"):
        if c in df.columns:
            df[c] = df[c].map(_BOOL).fillna(False).astype(bool)
    return df


def _final_reg(df):
    return df[(df["game_type"] == "R") & (df["is_final"])]


# --------------------------------------------------------------------------
# Check 1 — regular-season row count per season
# --------------------------------------------------------------------------

def check1_regular_counts(df):
    """Evaluates PLAYED (final) regular-season games against the band — that
    is what '~2430' and '~898 (2020)' mean. Raw game_type=='R' totals also
    include a handful of Cancelled-and-never-made-up games (shown as orphans)
    which were never played, so they are reported but not scored."""
    reg = df[df["game_type"] == "R"]
    lines = ["Check 1 — regular-season games per season "
             "(played = final R; band applies to played):"]
    ok = True
    for season in range(2015, 2026):
        sub = reg[reg["season"] == season]
        total = len(sub)
        played = int(sub["is_final"].sum())
        orphans = total - played
        lo, hi = (890, 905) if season == 2020 else (2425, 2431)
        in_band = lo <= played <= hi
        ok = ok and in_band
        note = "  (COVID 60-game season)" if season == 2020 else ""
        lines.append(f"  {season}: played={played:>5}  total_R={total:>5}  "
                     f"cancelled_orphans={orphans}  expect played {lo}-{hi}  "
                     f"[{'OK' if in_band else 'OUT OF RANGE'}]{note}")
    return ok, lines


# --------------------------------------------------------------------------
# Check 2 — games per team per season
# --------------------------------------------------------------------------

def check2_games_per_team(df):
    reg = _final_reg(df)  # played games; a cancelled orphan was never played
    lines, ok, offenders = ["Check 2 — games per team per season (played R):"], True, []
    for season in range(2015, 2026):
        sub = reg[reg["season"] == season]
        played = pd.concat([sub["home_team_id"], sub["away_team_id"]]).value_counts()
        expected = 60 if season == 2020 else 162
        lo, hi = (56, 62) if season == 2020 else (160, 164)
        for team_id, n in played.items():
            if not lo <= n <= hi:
                offenders.append((season, int(team_id), int(n), expected))
                ok = False
    if offenders:
        lines.append("  team-seasons OUTSIDE tolerance:")
        for season, team_id, n, expected in offenders:
            lines.append(f"    {season} team {team_id}: {n} (expected ~{expected})")
    else:
        lines.append("  all 30 teams within tolerance every season (162+-2; 2020 60+-2)")
    return ok, lines


# --------------------------------------------------------------------------
# Check 3 — home win rate
# --------------------------------------------------------------------------

def check3_home_win_rate(df):
    fr = _final_reg(df)
    decided = fr[fr["home_win"].notna()]
    lines, ok = ["Check 3 — home win rate (final R games, non-null home_win):"], True
    for season in range(2015, 2026):
        sub = decided[decided["season"] == season]
        if len(sub):
            rate = float(sub["home_win"].astype(float).mean())
            tag = "  <-- 2020 empty stadiums" if season == 2020 else ""
            lines.append(f"  {season}: {rate:.4f}  (n={len(sub)}){tag}")
    overall = float(decided["home_win"].astype(float).mean())
    ok = 0.52 <= overall <= 0.55
    lines.append(f"  OVERALL: {overall:.4f}  expected 0.52-0.55  "
                 f"[{'OK' if ok else 'OUT OF RANGE — possible home/away swap'}]")
    return ok, lines


# --------------------------------------------------------------------------
# Check 4 — isWinner row-level cross-check (the important one)
# --------------------------------------------------------------------------

def iswinner_mismatches(df):
    """Final R rows with non-null scores and a non-null isWinner flag where
    the score-derived home win disagrees with the raw flag. Empty == clean."""
    fr = _final_reg(df)
    sub = fr[fr["home_score"].notna() & fr["away_score"].notna()
             & fr["home_is_winner_flag"].notna()
             & (fr["home_score"] != fr["away_score"])].copy()
    derived = (sub["home_score"] > sub["away_score"])
    flag = sub["home_is_winner_flag"].astype(bool)
    return sub[derived.values != flag.values]


def check4_iswinner(df):
    mm = iswinner_mismatches(df)
    lines = [f"Check 4 — isWinner cross-check: {len(mm)} mismatch(es) "
             "(expected 0)"]
    if len(mm):
        lines.append("  FIRST 20 MISMATCHES:")
        cols = ["game_pk", "date", "home_team_id", "away_team_id",
                "home_score", "away_score", "home_win", "home_is_winner_flag"]
        for _, r in mm.head(20).iterrows():
            lines.append("    " + "  ".join(f"{c}={r[c]}" for c in cols))
    return len(mm) == 0, lines


# --------------------------------------------------------------------------
# Check 5 — duplicate game_pk / chunk-overlap artifacts
# --------------------------------------------------------------------------

def check5_duplicates(df):
    dup_pk = int(df["game_pk"].duplicated().sum())
    lines, ok = [f"Check 5 — duplicate game_pk (all types): {dup_pk} "
                 "(expected 0)"], dup_pk == 0
    # Triple test scoped to REGULAR season: only there must a same-day repeat
    # of the same teams be a doubleheader. Spring training runs split-squad
    # games (same teams, same day, doubleHeader='N', distinct gamePk) which
    # are legitimate and would false-positive here.
    reg = df[df["game_type"] == "R"]
    bad = []
    for (date, h, a), grp in reg.groupby(["date", "home_team_id", "away_team_id"]):
        if len(grp) > 1:
            codes = set(grp["doubleheader_code"])
            nums = set(grp["game_number"].dropna())
            if not codes <= {"S", "Y"} or len(nums) != len(grp):
                bad.append((date, int(h), int(a), len(grp), sorted(codes)))
    if bad:
        ok = False
        lines.append("  regular-season repeated (date,home,away) triples that "
                     "are NOT clean doubleheaders:")
        for date, h, a, n, codes in bad[:20]:
            lines.append(f"    {date} {h} vs {a}: {n} rows, codes={codes}")
    else:
        lines.append("  every repeated regular-season (date,home,away) triple "
                     "is a real doubleheader (distinct game_number, code in S/Y)")
    return ok, lines


# --------------------------------------------------------------------------
# Check 6 — null audit on finals
# --------------------------------------------------------------------------

def check6_nulls(df):
    """Structural columns (scores, team ids) must be zero-null on played
    games. Two columns get principled exemptions verified against the data:
    a tie has no winner (isWinner null iff home_win null), and MLB's API
    omits venue_id for neutral-site special events (Field of Dreams). Both
    are reported explicitly, never silently."""
    fr = _final_reg(df)
    required = ["home_score", "away_score", "home_team_id", "away_team_id"]
    lines = [f"Check 6 — null counts on final R rows (n={len(fr)}):"]
    ok = True
    for c in df.columns:
        nulls = int(fr[c].isna().sum())
        flag = ""
        if c in required and nulls != 0:
            flag, ok = "  <-- MUST BE ZERO (structural)", False
        elif c in required:
            flag = "  (required zero: OK)"
        lines.append(f"  {c:<26} {nulls}{flag}")

    # isWinner: nulls are allowed ONLY on ties (home_win also null)
    unexplained = fr[fr["home_is_winner_flag"].isna() & fr["home_win"].notna()]
    if len(unexplained):
        ok = False
        lines.append(f"  home_is_winner_flag: {len(unexplained)} null(s) on "
                     "DECIDED finals (unexplained) — MUST BE ZERO")
    else:
        n_tie = int(fr["home_is_winner_flag"].isna().sum())
        lines.append(f"  home_is_winner_flag null only on {n_tie} tie game(s) "
                     "(no winner exists) — OK")

    # venue_id: name every null; neutral-site specials legitimately lack one
    vnull = fr[fr["venue_id"].isna()]
    if len(vnull):
        lines.append(f"  venue_id null on {len(vnull)} final(s) — MLB API omits "
                     "venue for neutral-site special events:")
        for _, r in vnull.iterrows():
            lines.append(f"    game_pk={r['game_pk']} {r['date']} "
                         f"{r['home_team_name']} vs {r['away_team_name']}")
    return ok, lines


# --------------------------------------------------------------------------
# Check 7 — leagueRecord reconciliation (+ convention determination)
# --------------------------------------------------------------------------

def _team_season_facts(df):
    """Per (team_id, season) over final R games: derived wins, the team's
    wins_after in its chronologically last game (+ whether it won it, for the
    convention vote), and the MAX wins_after seen. Returns dict keyed
    (team, season).

    max_after is the robust season-ending win total: under the inclusive
    convention wins_after is monotonic over the true game order, so the
    maximum equals the final total even when a suspended/rescheduled game's
    officialDate sorts its snapshot out of order (which leaves a couple of
    teams' literal last-game snapshot one behind)."""
    fr = _final_reg(df).sort_values(["date", "game_pk"], kind="stable")
    facts = {}
    for _, r in fr.iterrows():
        hw = r["home_win"]
        for team, is_home in ((int(r["home_team_id"]), True),
                              (int(r["away_team_id"]), False)):
            key = (team, int(r["season"]))
            f = facts.setdefault(key, {"derived": 0, "last_after": None,
                                       "last_won": False, "max_after": None})
            won = pd.notna(hw) and ((is_home and hw == 1) or (not is_home and hw == 0))
            if won:
                f["derived"] += 1
            after = r["home_wins_after"] if is_home else r["away_wins_after"]
            if pd.notna(after):
                a = int(after)
                f["last_after"] = a
                f["last_won"] = bool(won)
                f["max_after"] = a if f["max_after"] is None else max(f["max_after"], a)
    return facts


def determine_convention(df):
    """Empirically: does leagueRecord include the game in question or not?

    Only team-seasons whose LAST game was a win discriminate the two: there
    inclusive predicts last wins_after == season wins, exclusive predicts
    == wins - 1. Seasons ending on a loss/tie satisfy both and are ignored
    for the vote."""
    facts = _team_season_facts(df)
    inc = exc = neither = 0
    for f in facts.values():
        if f["last_after"] is None or not f["last_won"]:
            continue
        if f["last_after"] == f["derived"]:
            inc += 1
        elif f["last_after"] == f["derived"] - 1:
            exc += 1
        else:
            neither += 1
    convention = "inclusive" if inc >= exc else "exclusive"
    return convention, inc, exc, neither, facts


def check7_leaguerecord(df):
    convention, inc, exc, neither, facts = determine_convention(df)
    lines = [f"Check 7 — leagueRecord reconciliation:",
             f"  convention determined EMPIRICALLY: {convention}  "
             f"(of win-ending team-seasons: inclusive-consistent={inc}, "
             f"exclusive-consistent={exc}, neither={neither})"]
    if neither:
        lines.append(f"  WARNING: {neither} win-ending team-seasons match "
                     "neither convention — inconsistent bookkeeping?")
    lines.append("  reconciling derived wins vs MLB season total = "
                 "max(wins_after) (robust to suspended-game snapshot ordering)")
    ok = True
    per_season = {}
    nonzero = []
    for (team, season), f in facts.items():
        if f["max_after"] is None:
            continue
        api_total = f["max_after"]
        diff = f["derived"] - api_total
        per_season.setdefault(season, [0, 0])
        per_season[season][0] += 1
        if diff == 0:
            per_season[season][1] += 1
        else:
            nonzero.append((season, team, f["derived"], api_total, diff))
            ok = False
    for season in sorted(per_season):
        total, zero = per_season[season]
        lines.append(f"  {season}: {zero}/{total} team-seasons reconcile "
                     f"exactly (diff=0)")
    if nonzero:
        lines.append("  NONZERO DIFFERENCES:")
        for season, team, d, a, diff in nonzero:
            lines.append(f"    {season} team {team}: derived={d} api={a} diff={diff}")
    else:
        lines.append("  all team-seasons reconcile against MLB bookkeeping "
                     "(diff=0 everywhere)")
    return ok, lines


# --------------------------------------------------------------------------
# Check 8 — reconciliation with existing games.csv (read-only)
# --------------------------------------------------------------------------

def check8_reconcile_games(df, games_csv=GAMES_CSV):
    lines = ["Check 8 — reconciliation with data/games.csv (read-only):"]
    if not Path(games_csv).exists():
        return True, lines + ["  games.csv not found — skipped."]
    g = pd.read_csv(games_csv)
    full = df.set_index("game_pk")
    ids_g, ids_f = set(g["game_id"]), set(full.index)
    absent = ids_g - ids_f
    both = ids_g & ids_f
    absent_seasons = g[g["game_id"].isin(absent)]["season"].value_counts().to_dict()
    disagree = 0
    for _, row in g[g["game_id"].isin(both)].iterrows():
        fr = full.loc[row["game_id"]]
        if (int(fr["home_score"]) != int(row["home_score"])
                or int(fr["away_score"]) != int(row["away_score"])
                or int(fr["home_team_id"]) != int(row["home_id"])
                or int(fr["away_team_id"]) != int(row["away_id"])):
            disagree += 1
    lines += [
        f"  game_ids in games.csv absent from games_full.csv: {len(absent)}",
        f"    (by season: {absent_seasons} — 2026 is expected, "
        "games_full stops at 2025)",
        f"  rows present in both: {len(both)}",
        f"  of those, disagreements on score/team assignment: {disagree}",
    ]
    return disagree == 0, lines


ALL_CHECKS = [check1_regular_counts, check2_games_per_team, check3_home_win_rate]


def main():
    if not GAMES_FULL_CSV.exists():
        print(f"{GAMES_FULL_CSV} not found — run `python src/ingest_history.py` first.")
        sys.exit(2)
    df = read_games_full()
    print(f"Loaded {len(df)} rows from {GAMES_FULL_CSV}\n")

    results = []
    for check in ALL_CHECKS:
        ok, lines = check(df)
        print("\n".join(lines)); print()
        results.append((check.__name__, ok))

    # Check 4 gates the rest: if isWinner disagrees with scores, stop here.
    ok4, lines4 = check4_iswinner(df)
    print("\n".join(lines4)); print()
    results.append(("check4_iswinner", ok4))
    if not ok4:
        print("!! Check 4 failed — halting before checks 5-8 as instructed.")
        _summary(results)
        sys.exit(1)

    for check in (check5_duplicates, check6_nulls, check7_leaguerecord,
                  check8_reconcile_games):
        ok, lines = check(df)
        print("\n".join(lines)); print()
        results.append((check.__name__, ok))

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
