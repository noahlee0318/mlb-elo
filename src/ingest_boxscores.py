"""Chunk 2: per-game boxscores -> data/pitcher_games.csv + data/batter_games.csv.

Scope: every FINAL regular-season and postseason game 2015-2025, INCLUDING
2020 — the log tables are complete on purpose; the modeling-time exclusion
of 2020/postseason stays in games_data.load_games(). The game list comes
from load_games() (the sanctioned games_full.csv reader), never from a
direct read.

Archive design: one ZIP per season (data/raw/boxscores_{season}.zip), one
entry per game named {game_pk}.json, deflate, allowZip64 — the ZIP central
directory gives O(1) "do I already have this game?" lookups. Incoming games
land in data/raw/_staging/{season}/ first; a season's staging directory is
rolled into its ZIP only when the season completes, so an interrupt can
never corrupt a ZIP central directory. A game counts as already-pulled if
it is in the season ZIP or in staging (resume is the default behavior).

Run from the project root:
    python src/ingest_boxscores.py                  # pull whatever's missing, then build CSVs
    python src/ingest_boxscores.py --seasons 2023   # one season
    python src/ingest_boxscores.py --force          # ignore archives, re-pull
    python src/ingest_boxscores.py --build-only     # skip network, rebuild CSVs from archives
"""

import argparse
import json
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.games_data import load_games
from src.mlb_api import session

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = RAW_DIR / "_staging"
PITCHER_CSV = DATA_DIR / "pitcher_games.csv"
BATTER_CSV = DATA_DIR / "batter_games.csv"

BOX_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"

N_WORKERS = 3
PER_WORKER_SPACING = 0.25  # seconds between requests, per worker

PITCHER_COLUMNS = [
    "game_pk", "season", "date", "game_type", "team_id", "opponent_team_id",
    "is_home", "pitcher_id", "pitcher_name",
    "ip_outs", "batters_faced", "pitches_thrown", "strikes",
    "hits", "runs", "earned_runs", "home_runs", "strikeouts", "walks",
    "intentional_walks", "hit_by_pitch", "ground_outs", "air_outs",
    "wild_pitches", "balks",
    "inherited_runners", "inherited_runners_scored",
    "appearance_order", "is_starter", "is_opener", "games_started",
    "starter_flag_mismatch", "complete_game", "shutout",
    "decision",
]

BATTER_COLUMNS = [
    "game_pk", "season", "date", "game_type", "team_id", "opponent_team_id",
    "is_home", "batter_id", "batter_name", "batting_order",
    "batting_order_slot", "position",
    "plate_appearances", "at_bats", "runs", "hits", "doubles", "triples",
    "home_runs", "rbi", "walks", "intentional_walks", "strikeouts",
    "hit_by_pitch", "sac_bunts", "sac_flies", "stolen_bases",
    "caught_stealing", "gidp", "left_on_base", "total_bases",
]


# --------------------------------------------------------------------------
# Pure parsing — unit-tested in tests/test_ingest_boxscores.py, no I/O here
# --------------------------------------------------------------------------

def parse_ip_to_outs(ip):
    """MLB thirds-notation innings string -> integer outs. '6.1' means six
    and ONE-THIRD innings -> 19 outs. Parsing it as a float and summing
    produces plausible-looking wrong totals, so outs are the stored unit.
    None/'' -> None. A fractional digit outside {0,1,2} is malformed."""
    if ip is None or ip == "":
        return None
    s = str(ip)
    whole, dot, frac = s.partition(".")
    frac = frac or "0"
    if not whole.lstrip("-").isdigit() or frac not in ("0", "1", "2"):
        raise ValueError(f"malformed inningsPitched: {ip!r}")
    return int(whole) * 3 + int(frac)


def _to_int(v):
    return None if v is None or v == "" else int(v)


def _decision_token(note):
    """Leading token of the pitching note: '(H, 12)' -> 'H'. Used only for
    S/H/BS (and W/L fallback when Chunk 1 carries no decision ids)."""
    if not note:
        return None
    m = re.match(r"\(\s*([A-Z]+)", str(note))
    tok = m.group(1) if m else None
    return tok if tok in {"W", "L", "S", "H", "BS"} else None


def parse_boxscore(meta, box):
    """One boxscore payload -> (pitcher_rows, batter_rows).

    meta: dict with game_pk, season, date, game_type, winning_pitcher_id,
    losing_pitcher_id (from games_full via load_games). Team ids and
    orientation come from the boxscore payload itself, so the Check 3/4
    cross-checks against games_full validate the join independently.
    """
    sides = box.get("teams", {})
    ids = {s: _to_int(sides.get(s, {}).get("team", {}).get("id")) for s in ("home", "away")}
    win_pid = meta.get("winning_pitcher_id")
    lose_pid = meta.get("losing_pitcher_id")

    keys = {"game_pk": meta["game_pk"], "season": meta["season"],
            "date": meta["date"], "game_type": meta["game_type"]}

    pitcher_rows, batter_rows = [], []
    for side in ("home", "away"):
        sd = sides.get(side, {})
        players = sd.get("players", {})
        base = dict(keys, team_id=ids[side],
                    opponent_team_id=ids["away" if side == "home" else "home"],
                    is_home=(side == "home"))

        for order, pid in enumerate(sd.get("pitchers", []), 1):
            p = players.get(f"ID{pid}", {})
            st = (p.get("stats", {}) or {}).get("pitching", {}) or {}
            ip_outs = parse_ip_to_outs(st.get("inningsPitched"))
            bf = _to_int(st.get("battersFaced"))
            gs = _to_int(st.get("gamesStarted"))
            is_starter = order == 1
            if pd.notna(win_pid) and win_pid is not None and int(win_pid) == pid:
                decision = "W"
            elif pd.notna(lose_pid) and lose_pid is not None and int(lose_pid) == pid:
                decision = "L"
            else:
                tok = _decision_token(st.get("note"))
                decision = tok if tok in {"S", "H", "BS"} else None
            pitcher_rows.append(dict(
                base,
                pitcher_id=pid,
                pitcher_name=p.get("person", {}).get("fullName"),
                ip_outs=ip_outs,
                batters_faced=bf,
                pitches_thrown=_to_int(st.get("numberOfPitches")),
                strikes=_to_int(st.get("strikes")),
                hits=_to_int(st.get("hits")),
                runs=_to_int(st.get("runs")),
                earned_runs=_to_int(st.get("earnedRuns")),
                home_runs=_to_int(st.get("homeRuns")),
                strikeouts=_to_int(st.get("strikeOuts")),
                walks=_to_int(st.get("baseOnBalls")),
                intentional_walks=_to_int(st.get("intentionalWalks")),
                hit_by_pitch=_to_int(st.get("hitByPitch")),
                ground_outs=_to_int(st.get("groundOuts")),
                air_outs=_to_int(st.get("airOuts")),
                wild_pitches=_to_int(st.get("wildPitches")),
                balks=_to_int(st.get("balks")),
                inherited_runners=_to_int(st.get("inheritedRunners")),
                inherited_runners_scored=_to_int(st.get("inheritedRunnersScored")),
                appearance_order=order,
                is_starter=is_starter,
                # heuristic for the post-2018 opener strategy (nominal starter
                # who isn't the primary pitcher) — NOT ground truth
                is_opener=bool(is_starter and ip_outs is not None and ip_outs <= 6
                               and bf is not None and bf <= 9),
                games_started=gs,
                # is_starter is positional (pitchers-array order); the API's own
                # gamesStarted disagreeing with it indicates a parse bug
                starter_flag_mismatch=(gs is not None
                                       and bool(gs == 1) != is_starter),
                complete_game=bool(_to_int(st.get("completeGames")) or 0),
                shutout=bool(_to_int(st.get("shutouts")) or 0),
                decision=decision,
            ))

        for key, p in players.items():
            bst = (p.get("stats", {}) or {}).get("batting", {}) or {}
            border = p.get("battingOrder")
            if not bst and border is None:
                continue  # e.g. relief pitcher who never batted
            batter_rows.append(dict(
                base,
                batter_id=_to_int(p.get("person", {}).get("id")),
                batter_name=p.get("person", {}).get("fullName"),
                batting_order=(str(border) if border is not None else None),
                batting_order_slot=(int(str(border)) // 100
                                    if border is not None else None),
                position=(p.get("position", {}) or {}).get("abbreviation"),
                plate_appearances=_to_int(bst.get("plateAppearances")),
                at_bats=_to_int(bst.get("atBats")),
                runs=_to_int(bst.get("runs")),
                hits=_to_int(bst.get("hits")),
                doubles=_to_int(bst.get("doubles")),
                triples=_to_int(bst.get("triples")),
                home_runs=_to_int(bst.get("homeRuns")),
                rbi=_to_int(bst.get("rbi")),
                walks=_to_int(bst.get("baseOnBalls")),
                intentional_walks=_to_int(bst.get("intentionalWalks")),
                strikeouts=_to_int(bst.get("strikeOuts")),
                hit_by_pitch=_to_int(bst.get("hitByPitch")),
                sac_bunts=_to_int(bst.get("sacBunts")),
                sac_flies=_to_int(bst.get("sacFlies")),
                stolen_bases=_to_int(bst.get("stolenBases")),
                caught_stealing=_to_int(bst.get("caughtStealing")),
                gidp=_to_int(bst.get("groundIntoDoublePlay")),
                left_on_base=_to_int(bst.get("leftOnBase")),
                total_bases=_to_int(bst.get("totalBases")),
            ))
    return pitcher_rows, batter_rows


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------

def scope_games():
    """Meta frame of every final R + postseason game 2015-2025 incl. 2020,
    via the sanctioned accessor (pull wide; load_games' defaults stay narrow
    for modeling)."""
    g = load_games(include_postseason=True, exclude_seasons=(),
                   finals_only=True, mlb_matchups_only=True)
    cols = ["game_pk", "season", "date", "game_type",
            "home_team_id", "away_team_id",
            "winning_pitcher_id", "losing_pitcher_id"]
    return g[cols].sort_values(["season", "date", "game_pk"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Archive plumbing
# --------------------------------------------------------------------------

def season_zip(season):
    return RAW_DIR / f"boxscores_{season}.zip"


def zip_members(season):
    zp = season_zip(season)
    if not zp.exists():
        return set()
    try:
        with zipfile.ZipFile(zp) as z:
            return {int(Path(n).stem) for n in z.namelist() if n.endswith(".json")}
    except zipfile.BadZipFile:
        print(f"  !! {zp.name} is corrupt — ignoring it (will re-pull)", flush=True)
        return set()


def staging_members(season):
    d = STAGING_DIR / str(season)
    if not d.exists():
        return set()
    return {int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()}


def _stage_write(season, pk, text):
    d = STAGING_DIR / str(season)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{pk}.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(d / f"{pk}.json")


def roll_season_zip(season):
    """Fold staging/{season} into boxscores_{season}.zip atomically: build a
    fresh ZIP (old entries not superseded by staging + all staging entries),
    replace, then delete staging. Never appends into a live ZIP."""
    stage = STAGING_DIR / str(season)
    staged = sorted(stage.glob("*.json")) if stage.exists() else []
    if not staged:
        return 0
    old = season_zip(season)
    tmp = old.with_suffix(".zip.tmp")
    staged_names = {p.name for p in staged}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as znew:
        if old.exists():
            try:
                with zipfile.ZipFile(old) as zold:
                    for name in zold.namelist():
                        if name not in staged_names:
                            znew.writestr(name, zold.read(name))
            except zipfile.BadZipFile:
                pass  # corrupt old zip: rebuild from staging only
        for p in staged:
            znew.write(p, arcname=p.name)
    tmp.replace(old)
    for p in staged:
        p.unlink()
    try:
        stage.rmdir()
    except OSError:
        # OneDrive briefly locks freshly-emptied dirs; an empty leftover
        # staging dir is harmless, so never let cleanup kill the run
        pass
    return len(staged)


def read_game_json(season, pk, zf_cache):
    """Boxscore JSON for one game, staging first, else the season ZIP."""
    sp = STAGING_DIR / str(season) / f"{pk}.json"
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    if season not in zf_cache:
        zf_cache[season] = zipfile.ZipFile(season_zip(season))
    return json.loads(zf_cache[season].read(f"{pk}.json"))


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------

_thread_local = threading.local()


def _fetch_one(season, pk):
    """Fetch + stage one game. Returns None on success, error string on
    persistent failure. Runs inside a worker thread."""
    delay = 1.0
    for attempt in range(1, 4):
        try:
            resp = session.get(BOX_URL.format(pk=pk), timeout=60)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                return f"HTTP {resp.status_code}"  # 404 won't heal — no retry
            json.loads(resp.text)  # validate before staging
            _stage_write(season, pk, resp.text)
            time.sleep(PER_WORKER_SPACING)
            return None
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == 3:
                return str(exc)
            time.sleep(delay)
            delay *= 2
    return "unreachable"


def pull_season(season, pks, force=False):
    """Pull all missing games for one season. Returns list of (pk, error)."""
    have = set() if force else (zip_members(season) | staging_members(season))
    todo = [pk for pk in pks if pk not in have]
    print(f"[{season}] scope={len(pks)} already={len(pks) - len(todo)} "
          f"to_pull={len(todo)}", flush=True)
    if not todo:
        return []
    failed = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, season, pk): pk for pk in todo}
        for fut in as_completed(futures):
            pk = futures[fut]
            err = fut.result()
            if err:
                print(f"  !! FAILED game_pk={pk}: {err}", flush=True)
                failed.append((pk, err))
            done += 1
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                rem = len(todo) - done
                eta = rem / rate if rate > 0 else float("inf")
                print(f"[{season}] {done}/{len(todo)} done, {rem} left, "
                      f"{el:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)
    return failed


# --------------------------------------------------------------------------
# Build CSVs
# --------------------------------------------------------------------------

def build_csvs(games):
    pitcher_rows, batter_rows, missing = [], [], []
    zf_cache = {}
    try:
        for meta in games.to_dict("records"):
            season, pk = int(meta["season"]), int(meta["game_pk"])
            try:
                box = read_game_json(season, pk, zf_cache)
            except (KeyError, FileNotFoundError):
                missing.append(pk)
                continue
            pr, br = parse_boxscore(meta, box)
            pitcher_rows.extend(pr)
            batter_rows.extend(br)
    finally:
        for z in zf_cache.values():
            z.close()

    pit = pd.DataFrame(pitcher_rows, columns=PITCHER_COLUMNS)
    bat = pd.DataFrame(batter_rows, columns=BATTER_COLUMNS)
    pit = pit.sort_values(["date", "game_pk", "team_id", "appearance_order"],
                          kind="stable").reset_index(drop=True)
    bat["_bo"] = bat["batting_order"].fillna("999")
    bat = bat.sort_values(["date", "game_pk", "team_id", "_bo"],
                          kind="stable").drop(columns="_bo").reset_index(drop=True)

    for df, path in ((pit, PITCHER_CSV), (bat, BATTER_CSV)):
        tmp = path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    return pit, bat, missing


def main():
    ap = argparse.ArgumentParser(description="Pull boxscores and build per-player game logs.")
    ap.add_argument("--seasons", default="2015-2025")
    ap.add_argument("--force", action="store_true",
                    help="re-pull everything, ignoring archives")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted pull (this is the default behavior)")
    ap.add_argument("--build-only", action="store_true",
                    help="skip the network entirely; rebuild CSVs from archives")
    args = ap.parse_args()
    if "-" in args.seasons:
        lo, hi = args.seasons.split("-", 1)
        seasons = list(range(int(lo), int(hi) + 1))
    else:
        seasons = [int(args.seasons)]

    games = scope_games()
    games = games[games["season"].isin(seasons)]
    print(f"scope: {len(games)} final R+postseason games, "
          f"seasons {seasons[0]}-{seasons[-1]}", flush=True)

    all_failed = []
    if not args.build_only:
        for season in seasons:
            pks = games.loc[games["season"] == season, "game_pk"].astype(int).tolist()
            failed = pull_season(season, pks, force=args.force)
            all_failed.extend((season, pk, err) for pk, err in failed)
            rolled = roll_season_zip(season)
            if rolled:
                print(f"[{season}] rolled {rolled} staged game(s) into "
                      f"{season_zip(season).name}", flush=True)

    pit, bat, missing = build_csvs(games)
    print(f"\nwrote {len(pit)} pitcher rows -> {PITCHER_CSV}", flush=True)
    print(f"wrote {len(bat)} batter rows -> {BATTER_CSV}", flush=True)
    if missing:
        print(f"{len(missing)} games in scope had NO archived boxscore "
              f"(first 20): {missing[:20]}", flush=True)
    if all_failed:
        print(f"\n{len(all_failed)} FAILED pulls — data is INCOMPLETE:", flush=True)
        for season, pk, err in all_failed:
            print(f"  {season} game_pk={pk}: {err}", flush=True)
    else:
        print("no failed pulls", flush=True)


if __name__ == "__main__":
    main()
