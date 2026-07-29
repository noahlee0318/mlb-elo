"""Chunk B — the single on-open refresh gate for the unified pipeline.

`refresh_if_stale(now, force, with_boxscores)` is THE gate the app-open path
hits. If the ratings snapshot is younger than the staleness window (and not
forced), it returns immediately without touching disk. Otherwise it runs the
refresh chain against the one unified game table (games_full.csv) — there is
no second games table after the games.csv migration.

The chain, and why it is split into a strict core + a best-effort tail:

    CORE (strictly atomic, fail-closed):
      1. games        scripts.backfill_season.backfill_games  (final-only,
                        idempotent, atomic; dedupes on game_pk)
      3. ratings       src.refresh.refresh_data                (build_ratings
                        replays games_full through elo.py, atomic snapshot)

    TAIL (best-effort, each independently guarded, never corrupts):
      4. predictions   src.predict.predict_for_date            (today's slate
                        only; immutability of prior days is asserted)
      2. boxscores     backfill_season.backfill_boxscores      (feeds only the
                        on-demand ML views; slow, so deferred)

Ratings (step 3) and today's logged Elo predictions (step 4) depend ONLY on
the games table, never on boxscores — so the fast, user-visible path is
1 -> 3 -> 4 and boxscores (2) runs last. The app-open call passes
with_boxscores=False so a reopen never blocks on the minutes-long box pull;
the "Refresh now" button and scripts/run_refresh.py pass True.

Fail-closed contract: a FAILED refresh means the CORE failed, and then every
CSV is restored byte-for-byte to its pre-run state — indistinguishable from no
refresh. The tail steps can only ever succeed-or-warn: a network hiccup logging
today's slate never discards the good games+ratings the core just committed,
and it can never rewrite a prior day's prediction (asserted). All five mutable
CSVs are snapshotted to *.dref.bak before any write, and a cross-process
lockfile stops two app opens (or a scheduler) from running the chain at once.

Run manually / from a scheduler:  python scripts/run_refresh.py [--force]
"""

import logging
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.games_data import GAMES_FULL_CSV, load_games
from src.ingest_boxscores import BATTER_CSV, PITCHER_CSV
from src.predict import PREDICTIONS_CSV, predict_for_date
from src.refresh import (DATA_DIR, RATINGS_CSV, STALE_AFTER_HOURS, load_ratings,
                         refresh_data)
from src.schedule import todays_date_str

# backfill_season lives in scripts/ (not an installable package); put it on the
# path and import it as a module so we CALL its verified functions rather than
# reimplement the ingest / boxscore rebuild.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import backfill_season as _bf  # noqa: E402

log = logging.getLogger(__name__)

# The five model-visible mutable files. games.csv is retired — deliberately not
# here. The boxscore ZIP/staging under data/raw are internal, atomic, and
# self-healing caches (roll_season_zip writes a .tmp then replaces; pull_season
# skips already-archived game_pks), so they are not part of the CSV rollback set.
CORE_FILES = [GAMES_FULL_CSV, RATINGS_CSV]
TAIL_PRED_FILES = [PREDICTIONS_CSV]
TAIL_BOX_FILES = [PITCHER_CSV, BATTER_CSV]
ALL_MUTABLE = CORE_FILES + TAIL_PRED_FILES + TAIL_BOX_FILES

LOCK_PATH = DATA_DIR / ".refresh.lock"
LOCK_STALE_SECONDS = 1800  # 30 min — longer than any real refresh; steal if older


# --------------------------------------------------------------------------
# snapshot / restore  (fail-closed .bak discipline)
# --------------------------------------------------------------------------

def _bak(path):
    return path.with_suffix(path.suffix + ".dref.bak")


def _snapshot(paths):
    """Copy each existing file to <path>.dref.bak; record which existed so a
    restore can also DELETE files the chain newly created. Returns the record."""
    existed = {}
    for p in paths:
        bak = _bak(p)
        if p.exists():
            shutil.copy2(p, bak)
            existed[p] = True
        else:
            if bak.exists():
                bak.unlink()  # never leave a stale .bak that would mis-restore
            existed[p] = False
    return existed


def _restore(existed, subset):
    """Return `subset` to its snapshotted state: copy the .bak back over a file
    that existed, or delete a file that did not exist before the run."""
    for p in subset:
        was = existed.get(p, False)
        bak = _bak(p)
        if was and bak.exists():
            tmp = p.with_suffix(p.suffix + ".restore.tmp")
            shutil.copy2(bak, tmp)
            tmp.replace(p)  # atomic swap back to the pre-run bytes
        elif not was and p.exists():
            p.unlink()      # the chain created it; pre-run state was "absent"


# --------------------------------------------------------------------------
# cross-process lock
# --------------------------------------------------------------------------

def _acquire_lock():
    """True if we now hold the lock. O_EXCL create is atomic; a lock older than
    LOCK_STALE_SECONDS is treated as a crashed run and stolen."""
    for _ in range(2):
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - LOCK_PATH.stat().st_mtime
            except FileNotFoundError:
                continue  # vanished between create and stat — retry
            if age > LOCK_STALE_SECONDS:
                try:
                    LOCK_PATH.unlink()
                except FileNotFoundError:
                    pass
                continue  # stole a stale lock — retry the create
            return False  # held by a live run
    return False


def _release_lock():
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------

def _age_hours(now):
    """Hours since the ratings snapshot's updated_at, or None if there is no
    (parseable) snapshot — the freshness signal the gate keys on. Wall-clock
    'time since last refresh', NOT newest-game date (which would read stale for
    days over an off-day break and refresh on every open)."""
    _, _, updated_at = load_ratings()
    if updated_at is None:
        return None
    try:
        return (now - datetime.fromisoformat(updated_at)).total_seconds() / 3600.0
    except ValueError:
        return None  # unparseable stamp — treat as stale


def _assert_games_clean(season):
    """Re-assert A's corruption guard on the freshly-written table: no null
    scores / null home_win in the current season. A scheduled or in-progress
    game must never have reached the finals-only table."""
    g = load_games(seasons=[season])
    bad_score = g["home_score"].isna() | g["away_score"].isna()
    assert not bad_score.any(), (
        f"{int(bad_score.sum())} {season} row(s) with null score after games step")
    assert g["home_win"].notna().all(), (
        f"null home_win in {season} rows after games step")


def _prior_pred_fingerprint(today):
    """(row_count, value_hash) of predictions dated before `today`, compared at
    the VALUE level. predict.log_predictions rewrites the whole file each call,
    and pandas' float repr round-trips can nudge the last digit of a probability
    STRING (e.g. ...90143 -> ...9014) with no change to the float64 value — a
    raw-string check would false-trip on that (Phase 0 item 5). Reading with
    normal type inference hashes the parsed values, so only a genuine change to
    a prior prediction (flipped home_won, different prob, dropped row) trips the
    guard, while harmless reformatting does not."""
    if not PREDICTIONS_CSV.exists():
        return (0, 0)
    df = pd.read_csv(PREDICTIONS_CSV)
    prior = df[df["date"].astype(str) < today]
    if prior.empty:
        return (0, 0)
    return (len(prior), int(pd.util.hash_pandas_object(prior, index=False).sum()))


def _pred_row_count():
    return len(pd.read_csv(PREDICTIONS_CSV)) if PREDICTIONS_CSV.exists() else 0


def _assert_predictions_unique():
    if PREDICTIONS_CSV.exists():
        p = pd.read_csv(PREDICTIONS_CSV)
        assert p["game_id"].is_unique, "duplicate game_id in predictions.csv"


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def _status(**kw):
    """A status dict whose keys are a superset of what app.py's slate header
    reads (ok, timestamp, games_added, data_as_of, error). timestamp/data_as_of
    are read fresh from disk so they reflect the post-op state (the restored
    old snapshot on a core failure, the new one on success)."""
    _, _, ts = load_ratings()
    try:
        data_as_of = None if not GAMES_FULL_CSV.exists() else \
            str(load_games()["date"].max())[:10]
    except Exception:
        data_as_of = None
    base = {
        "ok": False, "refreshed": False, "reason": None, "age_hours": None,
        "timestamp": ts, "games_added": 0, "predictions_logged": 0,
        "data_as_of": data_as_of, "predictions": None, "boxscores": None,
        "error": None,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def refresh_if_stale(now=None, force=False, with_boxscores=False, season=None):
    """THE single refresh gate. See module docstring for the chain.

    now: injectable wall clock (tests). force: run even if fresh.
    with_boxscores: run the deferred step-2 boxscore rebuild after the fast
    path commits (False for the non-blocking app-open path; True for the button
    and scripts/run_refresh.py). season: schedule season to pull (default: the
    current calendar year).
    """
    now = now or datetime.now()
    season = season or now.year

    age = _age_hours(now)
    if not force and age is not None and age < STALE_AFTER_HOURS:
        return _status(ok=True, refreshed=False, reason="fresh", age_hours=age)

    if not _acquire_lock():
        # another open / a scheduler is mid-refresh; serve current data
        return _status(ok=True, refreshed=False, reason="locked", age_hours=age)

    try:
        # snapshot only the files this run can write — the fast (app-open) path
        # never touches the large boxscore CSVs, so don't copy them
        targets = CORE_FILES + TAIL_PRED_FILES + \
            (TAIL_BOX_FILES if with_boxscores else [])
        existed = _snapshot(targets)
        through = _bf._clamp_through(None)  # yesterday; never today/future
        today = todays_date_str()

        # ---- CORE: games -> ratings (strictly atomic) --------------------
        try:
            new_pks = _bf.backfill_games(season, through, dry_run=False)
            _assert_games_clean(season)
            rstatus = refresh_data()  # build_ratings from games_full + snapshot
            if not rstatus["ok"]:
                raise RuntimeError(f"ratings step failed: {rstatus['error']}")
        except (Exception, SystemExit) as exc:
            _restore(existed, CORE_FILES)
            log.warning("refresh CORE failed, rolled back, serving prior data: %s",
                        exc)
            return _status(ok=False, refreshed=False,
                           reason=f"error: {exc}", error=str(exc), age_hours=age)

        # ---- TAIL step 4: today's predictions (best-effort, immutable) ---
        pred_note, pred_n = None, 0
        try:
            before = _prior_pred_fingerprint(today)
            n_before = _pred_row_count()
            ratings, _n, _t = load_ratings()          # the just-written snapshot
            predict_for_date(today, log=True, ratings=ratings)
            after = _prior_pred_fingerprint(today)
            if before != after:
                raise RuntimeError("a prior-date prediction row changed")
            _assert_predictions_unique()
            pred_n = _pred_row_count() - n_before      # net-new rows, not slate size
        except (Exception, SystemExit) as exc:
            _restore(existed, TAIL_PRED_FILES)
            pred_note = f"predictions skipped ({exc})"
            log.warning("refresh predictions step failed (non-fatal): %s", exc)

        # ---- TAIL step 2: boxscores (deferred, best-effort) --------------
        box_note = "deferred (not run this pass)"
        if with_boxscores:
            try:
                _bf.backfill_boxscores(season)
                box_note = "rebuilt"
            except (Exception, SystemExit) as exc:
                _restore(existed, TAIL_BOX_FILES)
                box_note = f"boxscores skipped ({exc})"
                log.warning("refresh boxscore step failed (non-fatal): %s", exc)

        return _status(ok=True, refreshed=True, reason="refreshed",
                       age_hours=age, games_added=len(new_pks),
                       predictions_logged=pred_n, predictions=pred_note,
                       boxscores=box_note)
    finally:
        _release_lock()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(refresh_if_stale(force=True, with_boxscores=True))
