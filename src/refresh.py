"""Ratings-snapshot primitives for the dashboard refresh.

refresh_data() replays the unified game table (games_full.csv, via
build_ratings/load_elo_games) through elo.py and snapshots data/ratings.csv.
Writes are write-to-temp-then-replace, and any failure leaves the existing
files untouched and returns a status dict the app can display.

Since the games.csv migration this module does NOT pull games or maintain a
table (games_full.csv is the single source). Since chunk B, the staleness GATE
and the full games -> ratings -> predictions -> boxscores chain live in
src/daily_refresh.py; the functions here are the ratings step + the shared
primitives (load_ratings, is_stale, _atomic_write) that daily_refresh and the
app reuse. There is exactly one gate, and it is daily_refresh.refresh_if_stale.

Run from the project root to re-snapshot ratings only:  python -m src.refresh
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.build_ratings import build_ratings
from src.games_data import load_games

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RATINGS_CSV = DATA_DIR / "ratings.csv"

STALE_AFTER_HOURS = 3  # games finalize once a day; 3h keeps reopens cheap

log = logging.getLogger(__name__)


def _atomic_write(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _newest_game_date():
    """Date (YYYY-MM-DD) of the newest game in the unified table, or None."""
    try:
        d = load_games()["date"]
        return None if d.empty else str(d.max())[:10]
    except Exception:
        return None


def load_ratings():
    """(ratings {team_id: rating}, names {team_id: name}, updated_at str)
    from the snapshot, or (None, None, None) if no snapshot exists yet."""
    if not RATINGS_CSV.exists():
        return None, None, None
    df = pd.read_csv(RATINGS_CSV)
    ratings = {int(r.team_id): float(r.rating) for r in df.itertuples(index=False)}
    names = {int(r.team_id): r.team_name for r in df.itertuples(index=False)}
    return ratings, names, str(df["updated_at"].max())


def is_stale(max_age_hours=STALE_AFTER_HOURS):
    """A real refresh is due if there's no snapshot yet or it's old."""
    _, _, updated_at = load_ratings()
    if updated_at is None:
        return True
    try:
        age = datetime.now() - datetime.fromisoformat(updated_at)
    except ValueError:
        return True  # unparseable stamp — treat as stale
    return age > timedelta(hours=max_age_hours)


def refresh_data(lookback_days=5):
    """Re-snapshot ratings.csv by replaying the unified games_full table.

    Since the games.csv migration this does NOT pull games or maintain a table
    (games_full.csv is the single source, kept current by backfill_season /
    chunk B). It only rebuilds Elo ratings from whatever games_full holds and
    writes the snapshot atomically. Returns a status dict:
    ok, games_added, timestamp, data_as_of, error. `lookback_days` is retained
    for signature compatibility and is unused here.
    """
    import pandas as pd
    status = {"ok": False, "games_added": 0, "timestamp": None,
              "data_as_of": _newest_game_date(), "error": None}
    try:
        ratings, names = build_ratings()  # from games_full via load_elo_games
        now = datetime.now().isoformat(timespec="seconds")
        snapshot = pd.DataFrame(
            [{"team_id": tid, "team_name": names[tid],
              "rating": round(r, 2), "updated_at": now}
             for tid, r in sorted(ratings.items(), key=lambda kv: -kv[1])])
        _atomic_write(snapshot, RATINGS_CSV)
        status.update(ok=True, games_added=0, timestamp=now,
                      data_as_of=_newest_game_date())
        log.info("ratings refresh ok: data through %s", status["data_as_of"])
    except Exception as exc:
        # ratings.csv was never half-written; report and carry on stale
        status["error"] = str(exc)
        log.warning("ratings refresh failed, keeping data as of %s: %s",
                    status["data_as_of"], exc)
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = refresh_data()
    print(s)
