"""Build data/games_full.csv — the canonical one-row-per-game history for
seasons 2015-2025 from the MLB Stats API schedule endpoint.

This is a NEW, standalone table. It does not touch data/games.csv or any
existing module — the live Elo pipeline keeps reading games.csv untouched.

Pulls one calendar month at a time (Feb-Nov) with probablePitcher +
decisions + linescore hydrated, saves every raw response verbatim under
data/raw/, and caches on disk so re-runs never re-hit the API. Parsing is
a pure function (parse_game) so it is unit-testable without a network.

Run from the project root:
    python src/ingest_history.py                    # 2015-2025, cached
    python src/ingest_history.py --seasons 2022     # one season
    python src/ingest_history.py --force            # ignore the disk cache
"""

import argparse
import calendar
import json
import sys
import time
from pathlib import Path

# Allow `python src/ingest_history.py` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from src.mlb_api import session  # shared User-Agent session; not modified

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HYDRATE = "probablePitcher,decisions,linescore"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
GAMES_FULL_CSV = DATA_DIR / "games_full.csv"

MONTHS = range(2, 12)  # February through November spans spring training -> WS

# Output column order (see the task table). game_pk is the only never-null key.
COLUMNS = [
    "game_pk", "season", "date", "start_time_utc", "game_type",
    "status_detailed", "is_final", "home_team_id", "away_team_id",
    "home_team_name", "away_team_name", "home_score", "away_score",
    "home_win", "home_is_winner_flag", "home_wins_after", "home_losses_after",
    "away_wins_after", "away_losses_after", "doubleheader_code", "game_number",
    "venue_id", "scheduled_innings", "innings_played",
    "home_probable_pitcher_id", "away_probable_pitcher_id",
    "winning_pitcher_id", "losing_pitcher_id", "is_anomalous_season",
    "rescheduled_from",
]

# Integer columns that may legitimately be null -> pandas nullable Int64 so
# the CSV renders "123" or "" rather than "123.0".
NULLABLE_INT_COLS = [
    "season", "home_team_id", "away_team_id", "home_score", "away_score",
    "home_win", "home_wins_after", "home_losses_after", "away_wins_after",
    "away_losses_after", "game_number", "venue_id", "scheduled_innings",
    "innings_played", "home_probable_pitcher_id", "away_probable_pitcher_id",
    "winning_pitcher_id", "losing_pitcher_id",
]


# --------------------------------------------------------------------------
# Parsing — pure functions, no I/O, unit-tested in tests/test_ingest_history.py
# --------------------------------------------------------------------------

def _nested(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _to_int(v):
    if v is None or v == "":
        return None
    return int(v)


def _is_cancelled_or_postponed(detailed_state):
    if not detailed_state:
        return False
    d = detailed_state.lower()
    return "cancel" in d or "postpon" in d


def _home_win_from_scores(home_score, away_score):
    """1/0/None. None on a missing score or an exact tie — a tie is never
    coerced to a loss."""
    if home_score is None or away_score is None:
        return None
    if home_score == away_score:
        return None
    return 1 if home_score > away_score else 0


def parse_game(g):
    """One schedule 'game' dict -> one canonical row dict.

    home_win is derived from the SCORES; home_is_winner_flag is read raw
    from teams.home.isWinner. They are deliberately independent so Check 4
    can cross-validate them.
    """
    home = _nested(g, ["teams", "home"], {}) or {}
    away = _nested(g, ["teams", "away"], {}) or {}
    status = g.get("status", {}) or {}
    linescore = g.get("linescore", {}) or {}
    decisions = g.get("decisions", {}) or {}

    detailed = status.get("detailedState")
    is_final = (status.get("abstractGameState") == "Final"
                and not _is_cancelled_or_postponed(detailed))

    home_score = _to_int(home.get("score"))
    away_score = _to_int(away.get("score"))
    # only a finished game contributes a win/loss; leaves postponed/tie -> None
    home_win = _home_win_from_scores(home_score, away_score) if is_final else None

    season = _to_int(g.get("season"))

    return {
        "game_pk": int(g["gamePk"]),
        "season": season,
        "date": g.get("officialDate"),          # NOT gameDate's date — TZ safe
        "start_time_utc": g.get("gameDate"),
        "game_type": g.get("gameType"),
        "status_detailed": detailed,
        "is_final": bool(is_final),
        "home_team_id": _to_int(_nested(home, ["team", "id"])),
        "away_team_id": _to_int(_nested(away, ["team", "id"])),
        "home_team_name": _nested(home, ["team", "name"]),
        "away_team_name": _nested(away, ["team", "name"]),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "home_is_winner_flag": home.get("isWinner"),   # raw bool / None
        "home_wins_after": _to_int(_nested(home, ["leagueRecord", "wins"])),
        "home_losses_after": _to_int(_nested(home, ["leagueRecord", "losses"])),
        "away_wins_after": _to_int(_nested(away, ["leagueRecord", "wins"])),
        "away_losses_after": _to_int(_nested(away, ["leagueRecord", "losses"])),
        "doubleheader_code": g.get("doubleHeader"),
        "game_number": _to_int(g.get("gameNumber")),
        "venue_id": _to_int(_nested(g, ["venue", "id"])),
        "scheduled_innings": _to_int(linescore.get("scheduledInnings")),
        "innings_played": _to_int(linescore.get("currentInning")),
        "home_probable_pitcher_id": _to_int(_nested(home, ["probablePitcher", "id"])),
        "away_probable_pitcher_id": _to_int(_nested(away, ["probablePitcher", "id"])),
        "winning_pitcher_id": _to_int(_nested(decisions, ["winner", "id"])),
        "losing_pitcher_id": _to_int(_nested(decisions, ["loser", "id"])),
        "is_anomalous_season": season == 2020,
        "rescheduled_from": g.get("rescheduledFrom"),  # raw (date string) / None
    }


def parse_payload(data):
    """Every game in one schedule JSON payload -> list of row dicts."""
    rows = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            rows.append(parse_game(g))
    return rows


def dedupe_snapshots(df):
    """Collapse the API's multiple status snapshots of ONE game to a single
    row. A postponed/suspended game comes back both as a placeholder (status
    Postponed, null scores) and, once played, as its Final row — under the
    SAME gamePk. Keep the most-resolved snapshot per game_pk: Final over
    non-final, scored over unscored, latest start time as the final tiebreak.

    This is NOT the chunk-overlap dedup Check 5 guards against (month chunks
    are date-disjoint); it is collapsing real MLB status history. Doubleheader
    games have DISTINCT gamePks and are never merged here.
    """
    df = df.copy()
    df["_final_rank"] = df["is_final"].astype(int)
    df["_score_rank"] = df["home_score"].notna().astype(int)
    df = df.sort_values(["game_pk", "_final_rank", "_score_rank", "start_time_utc"],
                        kind="stable")
    df = df.drop_duplicates("game_pk", keep="last")
    return df.drop(columns=["_final_rank", "_score_rank"])


def to_frame(rows):
    """Rows -> typed, de-duplicated, sorted DataFrame ready to write."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["game_pk"] = df["game_pk"].astype("int64")
    for c in NULLABLE_INT_COLS:
        df[c] = df[c].astype("Int64")
    df["home_is_winner_flag"] = df["home_is_winner_flag"].astype("boolean")
    df["is_final"] = df["is_final"].astype(bool)
    df["is_anomalous_season"] = df["is_anomalous_season"].astype(bool)
    df = dedupe_snapshots(df)

    # is_mlb_matchup: both clubs are among the 30 that play regular-season
    # games. Derived from the data (never a hardcoded id/name list) so spring
    # split-squads and non-MLB opponents (college / national / WBC) come out
    # False. Lets the accessor drop non-MLB matchups without guessing ids.
    reg = df[df["game_type"] == "R"]
    mlb_ids = (set(reg["home_team_id"].dropna().astype("int64"))
               | set(reg["away_team_id"].dropna().astype("int64")))
    df["is_mlb_matchup"] = (df["home_team_id"].isin(mlb_ids)
                            & df["away_team_id"].isin(mlb_ids)).astype(bool)
    return df.sort_values(["date", "game_pk"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Pull + disk cache
# --------------------------------------------------------------------------

def _raw_path(season, month):
    return RAW_DIR / f"schedule_{season}_{month:02d}.json"


def _fetch_chunk(start, end, retries=3):
    """One month's raw schedule JSON text, with exponential-backoff retry on
    network errors and 5xx. Raises on persistent failure."""
    params = {"sportId": 1, "startDate": start, "endDate": end, "hydrate": HYDRATE}
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(SCHEDULE_URL, params=params, timeout=120)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            json.loads(resp.text)  # validate before we trust it
            return resp.text
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries - 1} after error: {exc}")
            time.sleep(delay)
            delay *= 2


def load_or_fetch(season, month, force=False):
    """Return (raw_text, source) for one month, reading the cached file when
    present and valid unless force. source is 'cached' or 'live'."""
    path = _raw_path(season, month)
    if path.exists() and not force:
        try:
            text = path.read_text(encoding="utf-8")
            json.loads(text)
            return text, "cached"
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/partial cache -> re-pull

    last_day = calendar.monthrange(season, month)[1]
    start = f"{season}-{month:02d}-01"
    end = f"{season}-{month:02d}-{last_day:02d}"
    text = _fetch_chunk(start, end)  # may raise

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atomic
    return text, "live"


def build(seasons, force=False):
    """Pull/cache every month for the given seasons, parse, and write
    data/games_full.csv atomically. Returns (df, failed_chunks)."""
    all_rows = []
    failed = []
    for season in seasons:
        for month in MONTHS:
            try:
                text, source = load_or_fetch(season, month, force=force)
            except Exception as exc:  # loud, non-fatal — record and continue
                print(f"  !! FAILED {season}-{month:02d}: {exc}")
                failed.append((season, month, str(exc)))
                continue
            all_rows.extend(parse_payload(json.loads(text)))
            if source == "live":
                print(f"  pulled {season}-{month:02d} (live)")
                time.sleep(0.5)
    if not all_rows:
        raise SystemExit("No rows parsed — every chunk failed. Refusing to write.")

    df = to_frame(all_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GAMES_FULL_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(GAMES_FULL_CSV)  # atomic — never a truncated CSV under OneDrive
    return df, failed


def _parse_seasons_arg(s):
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


def main():
    ap = argparse.ArgumentParser(description="Build data/games_full.csv (2015-2025).")
    ap.add_argument("--seasons", default="2015-2025",
                    help="'2015-2025' range or a single year like '2022'")
    ap.add_argument("--force", action="store_true",
                    help="re-pull every chunk, ignoring the disk cache")
    args = ap.parse_args()
    seasons = _parse_seasons_arg(args.seasons)

    print(f"Building games_full.csv for seasons {seasons[0]}-{seasons[-1]} "
          f"({'force re-pull' if args.force else 'cache-first'})")
    df, failed = build(seasons, force=args.force)

    # summary
    print(f"\nwrote {len(df)} rows -> {GAMES_FULL_CSV}")
    print(f"duplicate game_pk after dedup: {int(df['game_pk'].duplicated().sum())}")
    reg = df[df["game_type"] == "R"]
    print("regular-season games per season (played / total_R):")
    for season in sorted(reg["season"].dropna().unique()):
        sub = reg[reg["season"] == season]
        print(f"  {int(season)}: played={int(sub['is_final'].sum())}  total_R={len(sub)}")
    all_ties = df[df["is_final"] & (df["home_score"] == df["away_score"])]
    reg_ties = reg[reg["is_final"] & (reg["home_score"] == reg["away_score"])]
    print(f"final games with equal scores (home_win=null ties): "
          f"{len(all_ties)} all types ({len(reg_ties)} regular-season; "
          "the rest are spring-training games, which legitimately tie)")
    if failed:
        print(f"\n{len(failed)} FAILED chunk(s) — data is INCOMPLETE:")
        for season, month, err in failed:
            print(f"  {season}-{month:02d}: {err}")
    else:
        print("\nall chunks pulled/cached successfully")


if __name__ == "__main__":
    main()
