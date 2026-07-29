"""The single sanctioned reader of data/games_full.csv.

`load_games()` is the ONLY function any modeling / feature / rating code
should use to read the history. Its defaults are the modeling-safe filters:
regular-season, final, MLB-vs-MLB games, with the anomalous 2020 season
excluded. Reading games_full.csv directly anywhere else is a bug — the
whole point of this module is that the correct filtering is the default and
bypassing it takes a visible, explicit argument.

Why spring training / exhibition / All-Star games are excluded by default
(recorded here so the reasoning survives):

    Those game types (S, E, A) include opponents that are NOT the 30 MLB
    clubs — college teams, national teams, World Baseball Classic squads —
    and "split squads" that field a franchise's identity in two simultaneous
    lineups. They carry no predictive signal for regular-season outcomes.
    Feeding them into Elo or any per-team-per-date feature silently corrupts
    ratings (a rout of a college team looks like a dominant win) and inflates
    a team's game count. They are kept in games_full.csv for completeness but
    must never reach a calculation, so they are off by default and turning
    them on is loud.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FULL_CSV = DATA_DIR / "games_full.csv"

EXHIBITION_TYPES = {"S", "E", "A"}          # spring, exhibition, All-Star
POSTSEASON_TYPES = {"F", "D", "L", "W"}     # wild card, division, league, world series
REGULAR_TYPE = {"R"}

_BOOL = {True: True, False: False, "True": True, "False": False}

log = logging.getLogger(__name__)


def load_games(
    include_regular_season=True,
    include_postseason=False,
    include_exhibition=False,   # spring training (S), exhibition (E), All-Star (A)
    finals_only=True,
    mlb_matchups_only=True,
    exclude_seasons=(2020,),
    seasons=None,               # optional explicit season list/range; None = all
):
    """Load data/games_full.csv with the modeling-safe filters applied.

    THIS IS THE ONLY SANCTIONED READER of games_full.csv. Every consumer
    (Elo, features, the future ML model, simulation) must come through here
    rather than calling pd.read_csv on the file, so the exclusion of
    non-predictive game types can never be forgotten.

    Defaults are the safe modeling set: regular-season, final, MLB-vs-MLB
    games, excluding the anomalous 2020 season. The three game-type switches
    are deliberately independent — excluding spring training and excluding
    the postseason are separate decisions, so they are separate arguments.

    Returns a COPY (never a view into cached state).
    """
    if not GAMES_FULL_CSV.exists():
        raise FileNotFoundError(
            f"{GAMES_FULL_CSV} does not exist. Build it first by running "
            "`python src/ingest_history.py` (see src/ingest_history.py)."
        )

    if include_exhibition:
        print("WARNING: load_games called with include_exhibition=True; "
              "spring/exhibition/All-Star games included. This is almost "
              "certainly not what you want for modeling.", file=sys.stderr)

    df = pd.read_csv(GAMES_FULL_CSV, low_memory=False)
    for c in ("is_final", "is_mlb_matchup"):
        df[c] = df[c].map(_BOOL).fillna(False).astype(bool)

    allowed = set()
    if include_regular_season:
        allowed |= REGULAR_TYPE
    if include_postseason:
        allowed |= POSTSEASON_TYPES
    if include_exhibition:
        allowed |= EXHIBITION_TYPES
    df = df[df["game_type"].isin(allowed)]

    if finals_only:
        df = df[df["is_final"]]
    if mlb_matchups_only:
        df = df[df["is_mlb_matchup"]]
    if exclude_seasons:
        df = df[~df["season"].isin(set(exclude_seasons))]
    if seasons is not None:
        df = df[df["season"].isin(set(seasons))]

    df = df.copy()  # never hand back a view into a filtered frame

    assert not df.empty, (
        "load_games returned zero rows — the filters excluded everything. "
        f"allowed types={sorted(allowed)}, finals_only={finals_only}, "
        f"mlb_matchups_only={mlb_matchups_only}, exclude_seasons={exclude_seasons}, "
        f"seasons={seasons}"
    )

    breakdown = df["game_type"].value_counts().to_dict()
    # every caller leaves a trace of exactly what it loaded
    log.info("load_games: %d rows | game_type breakdown: %s", len(df), breakdown)
    print(f"load_games: {len(df)} rows | game_type breakdown: {breakdown}",
          file=sys.stderr)
    return df


# Elo rating window: last ELO_WINDOW_SEASONS completed seasons + the current
# one (season >= current_year - ELO_WINDOW_SEASONS), reproducing the span the
# retired data/games.csv carried (ingest.py used N_COMPLETED_SEASONS = 5).
ELO_WINDOW_SEASONS = 5


def load_elo_games(seasons=None):
    """games_full restricted to the Elo rating window, mapped to the legacy
    games.csv column names. THE single Elo game source after the games.csv
    migration — build_ratings, standings, and remaining_schedule read this
    instead of the retired data/games.csv.

    Columns: game_datetime, game_id, season, home_id, away_id, home_name,
    away_name, home_score, away_score (sorted by game_datetime). Includes the
    'Completed Early' finals the old detailedState=='Final' filter dropped —
    real, decided games; finals-only / regular-season / MLB-matchups / 2020
    excluded all inherited from load_games(). See docs and
    scripts/verify_games_equivalence.py for the accepted difference from the
    old games.csv (exactly those 'Completed Early' games).
    """
    from datetime import date
    import pandas as pd
    g = load_games()
    if seasons is None:
        floor = date.today().year - ELO_WINDOW_SEASONS
        seasons = sorted(int(s) for s in g["season"].unique() if int(s) >= floor)
    g = g[g["season"].isin(list(seasons))]
    # True-chronological replay order. start_time_utc is the real game clock;
    # game_number then game_pk break ties deterministically for doubleheaders
    # that share a start time (game 1 before game 2). Sorting on the native
    # columns HERE — before renaming to the legacy schema, which drops
    # game_number — is what keeps the Elo replay order (and the ratings)
    # matching the pre-migration baseline. build_ratings' later stable sort on
    # game_datetime only preserves this order; it cannot recover it.
    g = g.sort_values(["start_time_utc", "game_number", "game_pk"], kind="stable")
    out = pd.DataFrame({
        "game_datetime": g["start_time_utc"].to_numpy(),
        "game_id": g["game_pk"].astype("int64").to_numpy(),
        "season": g["season"].astype("int64").to_numpy(),
        "home_id": g["home_team_id"].astype("int64").to_numpy(),
        "away_id": g["away_team_id"].astype("int64").to_numpy(),
        "home_name": g["home_team_name"].to_numpy(),
        "away_name": g["away_team_name"].to_numpy(),
        "home_score": g["home_score"].astype("int64").to_numpy(),
        "away_score": g["away_score"].astype("int64").to_numpy(),
    })
    return out.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = load_games()
    print(g.groupby("season").size())
