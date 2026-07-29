"""Chunk B — on-demand / scheduler-ready entry point for the daily refresh.

Runs the ONE gate (src.daily_refresh.refresh_if_stale) and prints its status.
This is the exact command a future Windows Task Scheduler job would call — the
on-open gate and any scheduler share this single code path. No OS scheduler is
created here; this just keeps the entry point scheduler-ready.

Unlike the app-open path, this runs WITH the boxscore rebuild by default (the
scheduler is where the slow step-2 pull belongs); pass --no-boxscores to skip
it. Exits 0 when the refresh is ok (ran or cleanly skipped as fresh), 1 on a
core failure (rolled back, prior data intact).

Usage:
  python scripts/run_refresh.py            # refresh if stale (incl. boxscores)
  python scripts/run_refresh.py --force    # refresh even if fresh
  python scripts/run_refresh.py --no-boxscores
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_refresh import refresh_if_stale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="run the chain even if the snapshot is within the window")
    ap.add_argument("--no-boxscores", action="store_true",
                    help="skip the deferred step-2 boxscore rebuild")
    args = ap.parse_args()

    status = refresh_if_stale(force=args.force,
                              with_boxscores=not args.no_boxscores)

    print("=" * 66)
    print(f"DAILY REFRESH — {status['reason']}")
    print("=" * 66)
    print(f"  ok:                {status['ok']}")
    print(f"  refreshed:         {status['refreshed']}")
    age = status["age_hours"]
    print(f"  age (hours):       {age:.2f}" if age is not None else
          "  age (hours):       n/a (no prior snapshot)")
    print(f"  games added:       {status['games_added']}")
    print(f"  predictions logged:{status['predictions_logged']}")
    print(f"  ratings updated_at:{status['timestamp']}")
    print(f"  results through:   {status['data_as_of']}")
    if status["predictions"]:
        print(f"  predictions note:  {status['predictions']}")
    if status["boxscores"]:
        print(f"  boxscores:         {status['boxscores']}")
    if status["error"]:
        print(f"  error:             {status['error']}")
    print("=" * 66)

    sys.exit(0 if status["ok"] else 1)


if __name__ == "__main__":
    main()
