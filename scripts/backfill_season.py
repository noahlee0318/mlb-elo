"""A — re-runnable, idempotent, FINAL-GAMES-ONLY season backfill for
data/games_full.csv (the source of truth every model reads).

Why this is gated: the historical ingest (src/ingest_history.py) writes EVERY
game, final or not — it only ever ran on complete seasons, so the non-final
rows it produced were harmless cancelled games that load_games(finals_only)
drops. A mid-season pull returns scheduled/in-progress games too; writing one
with a null score would poison the feature builder's trailing windows. This
script therefore filters to terminal-final, both-scores-present,
home_win-derivable games and nothing else, and proves it (scripts/verify_backfill.py).

Re-runnable + idempotent by design (it is the core of the eventual daily
refresh): it pulls the season, keeps only new final games (dedupe on game_pk),
and a second run appends zero rows and leaves the file byte-identical.

Reuses src/ingest_history's PURE parsers (parse_payload, to_frame) — no parsing
reimplementation, so the 2026 rows are built by the exact code that built
2015-2025. Boxscores for the new game_pks go through the existing Chunk 2
ingest (src/ingest_boxscores) — also not reimplemented.

Usage:
  python scripts/backfill_season.py --season 2026 [--through YYYY-MM-DD] [--dry-run]
  python scripts/backfill_season.py --season 2026 --no-boxscores
"""

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingest_history import (GAMES_FULL_CSV, NULLABLE_INT_COLS,
                                load_or_fetch, parse_payload, to_frame)

# season start month for the schedule pull (spring training onward, matching
# the historical Feb-Nov structure). Final-only + date<=through do the trimming.
SEASON_START_MONTH = 2
_BOOL_COLS = ["is_final", "is_anomalous_season", "is_mlb_matchup"]


def read_typed(path):
    """Read games_full.csv re-applying the EXACT dtype schema src/ingest_history
    wrote it with, so a read -> concat -> write round-trip renders historical
    rows byte-identically (nullable Int64 -> '123'/'' not '123.0'). This is the
    load-bearing guarantee behind the idempotency + untouched-history checks."""
    df = pd.read_csv(path, low_memory=False)
    df["game_pk"] = df["game_pk"].astype("int64")
    for c in NULLABLE_INT_COLS:
        df[c] = df[c].astype("Int64")
    df["home_is_winner_flag"] = df["home_is_winner_flag"].astype("boolean")
    for c in _BOOL_COLS:
        df[c] = df[c].astype(bool)
    return df


def _clamp_through(through):
    """Never today or the future — today's games may be in progress."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    if through is None:
        return yesterday
    return through if through <= yesterday else yesterday


def fetch_season(season, force=True):
    """Every scheduled game for `season`, Feb-Nov, via the historical ingest's
    cached month puller (force=True so recent months always refresh). Returns a
    typed, deduped, sorted frame (to_frame) with all columns + is_mlb_matchup."""
    rows = []
    for month in range(SEASON_START_MONTH, 12):
        try:
            text, source = load_or_fetch(season, month, force=force)
        except Exception as exc:
            print(f"  !! fetch failed {season}-{month:02d}: {exc}",
                  file=sys.stderr)
            continue
        import json
        rows.extend(parse_payload(json.loads(text)))
    if not rows:
        raise SystemExit(f"No {season} games returned — refusing to proceed.")
    return to_frame(rows)


def filter_final(df, through):
    """Keep only terminal-final, both-scores-present, home_win-derivable games
    dated <= through. Returns (kept, report_dict, tie_pks). Ties (final, scored,
    but home_win null) are DROPPED and reported by game_pk, loudly."""
    fetched = len(df)
    dated = df[df["date"] <= through]
    future = df[df["date"] > through]

    final = dated[dated["is_final"]]
    not_final = dated[~dated["is_final"]]

    scored = final[final["home_score"].notna() & final["away_score"].notna()]
    no_score = final[final["home_score"].isna() | final["away_score"].isna()]

    ties = scored[scored["home_win"].isna()]           # final + scored, no W/L
    kept = scored[scored["home_win"].notna()]

    report = {
        "fetched": fetched,
        "future_dropped": len(future),
        "not_final_dropped": len(not_final),
        "no_score_dropped": len(no_score),
        "tie_dropped": len(ties),
        "kept_final": len(kept),
    }
    tie_pks = ties["game_pk"].tolist()
    return kept, report, ties


def backfill_games(season, through, dry_run):
    """The games_full.csv backfill. Returns the list of newly-appended game_pks
    (empty if none / dry-run)."""
    print(f"[games] fetching {season} schedule (Feb-Nov, through {through})…")
    df = fetch_season(season)
    kept, report, ties = filter_final(df, through)

    print(f"[games] fetched={report['fetched']}  "
          f"kept_final={report['kept_final']}")
    print(f"[games] dropped: future={report['future_dropped']}  "
          f"not_final={report['not_final_dropped']}  "
          f"no_score={report['no_score_dropped']}  "
          f"tie={report['tie_dropped']}")
    if len(ties):
        # LOUD, per-game_pk — a tie is dropped deliberately (no W/L signal),
        # but never silently: print each so a surprising count is auditable.
        print(f"[games] TIES DROPPED (final+scored but home_win null), "
              f"{len(ties)} game(s):", file=sys.stderr)
        for r in ties.itertuples(index=False):
            print(f"    game_pk={r.game_pk}  {r.date}  "
                  f"{r.away_team_name} @ {r.home_team_name}  "
                  f"{r.away_score}-{r.home_score}", file=sys.stderr)

    existing = read_typed(GAMES_FULL_CSV)
    existing_pks = set(existing["game_pk"].tolist())
    new = kept[~kept["game_pk"].isin(existing_pks)].copy()
    print(f"[games] already present: {len(kept) - len(new)}  |  new to append: "
          f"{len(new)}")

    if new.empty:
        print("[games] 0 new games appended (idempotent — file untouched).")
        return []

    if dry_run:
        print("[games] --dry-run: would append these game_pks (first 20): "
              f"{new['game_pk'].tolist()[:20]}")
        print(f"[games] --dry-run: would append {len(new)} rows spanning "
              f"{new['date'].min()}..{new['date'].max()}. No file written.")
        return []

    # backup before any write (OneDrive-synced path — a corrupt write is worse)
    bak = GAMES_FULL_CSV.with_suffix(".csv.bak")
    shutil.copy2(GAMES_FULL_CSV, bak)
    print(f"[games] backed up -> {bak.name}")

    # full read-concat-sort-atomic-write (per the confirmed design). Columns of
    # `new` already match `existing` (to_frame order == file order).
    new = new[existing.columns]
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.sort_values(["date", "game_pk"], kind="stable") \
                       .reset_index(drop=True)

    # invariants before committing the write
    assert combined["game_pk"].is_unique, "duplicate game_pk after append!"
    n_old = len(existing)
    # the historical (<= season-1) portion must be preserved row-for-row: since
    # every new row is dated after all existing rows, the sort places them last
    # and the leading n_old rows must equal `existing` exactly.
    head = combined.iloc[:n_old].reset_index(drop=True)
    assert head["game_pk"].tolist() == existing["game_pk"].tolist(), \
        "historical row order changed — refusing to write"

    tmp = GAMES_FULL_CSV.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(GAMES_FULL_CSV)                        # atomic
    print(f"[games] wrote {len(combined)} rows ({n_old} + {len(new)} new) "
          f"-> {GAMES_FULL_CSV.name}")
    return new["game_pk"].astype(int).tolist()


def backfill_boxscores(season):
    """Pull boxscores for the season's final game_pks via the EXISTING Chunk 2
    ingest (idempotent season ZIP), then rebuild the log CSVs over the FULL
    scope so 2015-(season) are all present. Not reimplemented."""
    from src import ingest_boxscores as ib
    scope = ib.scope_games()                       # load_games(finals) all seasons
    pks = scope.loc[scope["season"] == season, "game_pk"].astype(int).tolist()
    if not pks:
        print(f"[box] no {season} final games in scope yet — skipping boxscores")
        return
    print(f"[box] pulling boxscores for {len(pks)} {season} final game(s) "
          "(idempotent)…")
    failed = ib.pull_season(season, pks)
    rolled = ib.roll_season_zip(season)
    if rolled:
        print(f"[box] rolled {rolled} staged game(s) into "
              f"{ib.season_zip(season).name}")
    if failed:
        print(f"[box] {len(failed)} boxscore pull(s) FAILED (data incomplete):",
              file=sys.stderr)
        for pk, err in failed[:20]:
            print(f"    game_pk={pk}: {err}", file=sys.stderr)
    # rebuild the FULL log CSVs (all seasons) — NOT a per-season subset, which
    # would wipe other seasons. 2015-(season-1) rebuild identically from their
    # archives; the new season appends (sorted by date).
    print("[box] rebuilding pitcher_games.csv + batter_games.csv over full "
          "scope…")
    pit, bat, missing = ib.build_csvs(scope)
    print(f"[box] wrote {len(pit)} pitcher rows, {len(bat)} batter rows")
    if missing:
        print(f"[box] {len(missing)} scope games had no archived boxscore "
              f"(first 20): {missing[:20]}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--through", default=None,
                    help="YYYY-MM-DD; default yesterday. Never today/future.")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + report, write nothing")
    ap.add_argument("--no-boxscores", action="store_true",
                    help="skip the Chunk 2 boxscore pull/rebuild")
    args = ap.parse_args()

    through = _clamp_through(args.through)
    if args.through and through != args.through:
        print(f"[games] --through {args.through} is today/future; clamped to "
              f"{through}", file=sys.stderr)

    print("=" * 70)
    print(f"BACKFILL season={args.season} through={through} "
          f"{'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 70)

    new_pks = backfill_games(args.season, through, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] no games written; skipping boxscores.")
        return
    if args.no_boxscores:
        print("\n[box] skipped (--no-boxscores).")
        return
    # boxscores rebuild even when 0 new games appended: it is idempotent and
    # cheap-to-skip (already-archived pks aren't re-pulled), and it heals a
    # prior partial run.
    backfill_boxscores(args.season)


if __name__ == "__main__":
    main()
