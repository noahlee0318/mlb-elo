"""Chunk 9, Deliverable 1: the LIVE feature path.

The one rule this module exists to enforce: the live path computes features by
calling the SAME Chunk 3 builder (`src.features.build_features`) that produced
the training table — never a reimplementation. Training/serving skew is a
silent failure, so there is exactly one function that computes a feature and
both paths use it, differing only in the data cutoff.

HOW (the approved design — option A): the Chunk 3 builder is a whole-frame
batch builder with no per-game/by-date entry point, but it already carries a
`mode="predict"` serving path. So the live path reuses it verbatim: build a
frame of [every historical FINAL game strictly before the target in canonical
order] + [the target game row], call build_features, and read the target's row.
No forked logic, no modification to the frozen Chunk 3 builder.

THE CUTOFF IS CANONICAL POSITION, NOT RAW DATE. The Chunk 3 "as-of" is defined
by game_order = position in the (date, game_number, game_pk) sort, and a game's
features draw only on strictly-earlier positions. A naive `date < as_of_date`
cutoff DROPS same-day-earlier games that legitimately contribute — doubleheader
game 1 feeding game 2's team form, and same-day-earlier starts feeding the
365-day league-average pool for a low-confidence pitcher — and was measured to
disagree with the stored table by ~3e-4 on exactly those games. The
position cutoff reproduces the historical vector to 1e-9 on single, doubleheader
AND low-confidence games (verified in scripts/verify_live_vs_historical.py).
Consequently the leakage guard here is POSITION-based ("no contributing game at
or after the target's canonical position"), which is the honest "no future
game leaks" check; a strict date>=as_of check would false-positive on the
legitimate doubleheader-1 -> doubleheader-2 contribution.

train vs predict starter: reconstructing a PLAYED game uses the actual boxscore
starter (mode="train"), reproducing the stored table exactly — this isolates
the feature-math equality (the skew we are hunting) from the orthogonal
"did we guess the right starter" question. A genuinely unplayed game uses the
listed probable (mode="predict") or, if none is announced, the same
league-average imputation the builder uses, flagged low_confidence.

Determinism: given fixed underlying game data and a fixed as_of_date this is a
pure function. The only wall-clock dependency is resolving "today" in the live
slate branch of the verifier, which is deliberately quarantined there.
"""

import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features import build_features, canonical_order
from src.features_ml import FEATURE_COLUMNS
from src.games_data import load_games

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PITCHER_CSV = DATA_DIR / "pitcher_games.csv"

# home-side / away-side split of FEATURE_COLUMNS (away owns away_tz_delta)
HOME_COLS = [c for c in FEATURE_COLUMNS if c.startswith("home_")]
AWAY_COLS = [c for c in FEATURE_COLUMNS if c.startswith("away_")]
assert set(HOME_COLS) | set(AWAY_COLS) == set(FEATURE_COLUMNS), \
    "a FEATURE_COLUMN is neither home_ nor away_ prefixed"

COLD_START_MIN_GAMES = 15


def _as_iso(d) -> str:
    """Normalize a date-like (datetime.date, datetime, pd.Timestamp, or ISO
    str) to a 'YYYY-MM-DD' string, matching the string `date` column that
    load_games() returns. The project keeps dates as ISO strings by
    convention (they sort correctly lexically, which is why the convention
    exists); this obeys it at the module boundary so every internal comparison
    against the `date` column is string-vs-string, never str-vs-date."""
    if isinstance(d, str):
        return d[:10]
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Cached history (read once; pure w.r.t. the on-disk data)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _history():
    """(games_all, logs_all, rank_of_pk, pos_of_pk) for all sanctioned finals.

    rank_of_pk maps game_pk -> its integer position in canonical order;
    pos_of_pk maps game_pk -> the (date, game_number, game_pk) tuple. Both are
    computed once from the same canonical_order the Chunk 3 builder uses."""
    games = load_games()
    logs = pd.read_csv(PITCHER_CSV, low_memory=False)
    logs = logs[logs["game_type"] == "R"].copy()
    co = canonical_order(games)
    rank_of_pk = {int(r.game_pk): int(r.game_order)
                  for r in co.itertuples(index=False)}
    pos_of_pk = {int(r.game_pk): (r.date, int(r.game_number), int(r.game_pk))
                 for r in co.itertuples(index=False)}
    return games, logs, rank_of_pk, pos_of_pk


def _find_historical_pk(games, home_id, away_id, as_of_date, game_number=None):
    """The game_pk of a FINAL game matching this matchup/date, or None. When a
    doubleheader yields two, game_number disambiguates (else the earliest)."""
    m = games[(games["home_team_id"] == home_id)
              & (games["away_team_id"] == away_id)
              & (games["date"] == as_of_date)]
    if m.empty:
        return None
    if game_number is not None and (m["game_number"] == game_number).any():
        m = m[m["game_number"] == game_number]
    return int(m.sort_values("game_number").iloc[0]["game_pk"])


def _venue_for(games, team_id, as_of_date):
    """The team's most recent home venue strictly before as_of_date (its home
    park). Used only to populate a synthetic live row's venue_id."""
    prior = games[(games["home_team_id"] == team_id)
                  & (games["date"] < as_of_date)]
    if prior.empty:
        return np.nan
    return prior.sort_values("date").iloc[-1]["venue_id"]


# ---------------------------------------------------------------------------
# The single builder-calling core
# ---------------------------------------------------------------------------

def _reconstruct_game(home_id, away_id, as_of_date, *,
                      home_pp_id=None, away_pp_id=None, game_number=None):
    """Full feature row for one game as of as_of_date, via the Chunk 3 builder.

    Returns a dict with the 11 FEATURE_COLUMNS, per-side games_played and
    sp_low_confidence flags, whether the target was a played game, and the
    starter mode used. Raises on any positional leakage."""
    # boundary contract: the public entry points normalize as_of_date to an ISO
    # string via _as_iso. Fail loudly here if a caller reached this core with a
    # raw date object, rather than deep inside a pandas str-vs-date comparison.
    assert type(as_of_date) is str, \
        f"as_of_date must be an ISO string here, got {type(as_of_date).__name__}"
    games, logs, rank_of_pk, pos_of_pk = _history()

    hist_pk = _find_historical_pk(games, home_id, away_id, as_of_date,
                                  game_number)
    if hist_pk is not None:
        # played game: reproduce the stored table exactly (actual starter)
        target_pos = pos_of_pk[hist_pk]
        target_row = games[games["game_pk"] == hist_pk]
        mode = "train"
        played = True
    else:
        # unplayed game: synthesize a target row after the same-day earlier
        # finals, use the listed probables (mode="predict"; missing -> the
        # builder's own league-average imputation, flagged low_confidence)
        gnum = int(game_number) if game_number is not None else 1
        fake_pk = max(rank_of_pk) + 1  # any id after all real games
        # LIVE cutoff is strict date < as_of_date: exclude the whole target day
        # so the prediction does not depend on which same-day games have gone
        # final by wall-clock (the live branch is the non-reproducible one and
        # is kept deterministic w.r.t. the historical data this way). The -inf
        # sentinels make (d, gnum, pk) < (as_of, -inf, -inf) hold iff d<as_of.
        target_pos = (as_of_date, float("-inf"), float("-inf"))
        season = int(as_of_date[:4])
        target_row = pd.DataFrame([{
            "game_pk": fake_pk, "season": season, "date": as_of_date,
            "game_type": "R", "is_final": False,
            "home_team_id": home_id, "away_team_id": away_id,
            "home_score": np.nan, "away_score": np.nan, "home_win": np.nan,
            "game_number": gnum,
            "venue_id": _venue_for(games, home_id, as_of_date),
            "home_probable_pitcher_id": home_pp_id,
            "away_probable_pitcher_id": away_pp_id,
        }])
        # align columns to games_all so concat is clean
        for c in games.columns:
            if c not in target_row.columns:
                target_row[c] = np.nan
        target_row = target_row[games.columns]
        mode = "predict"
        played = False

    # frame = every FINAL game strictly canonically before the target, + target
    def _pos(pk):
        return pos_of_pk[int(pk)]
    before_mask = games["game_pk"].map(
        lambda pk: _pos(pk) < target_pos)
    before = games[before_mask]

    # LEAKAGE GUARD (positional — the honest "no future game" check). Every
    # contributing game must sit strictly before the target in canonical order,
    # and none may post-date as_of_date. Same-day-earlier games (doubleheader
    # game 1) are legitimately < target and are NOT leaks.
    assert bool((before["date"] <= as_of_date).all()), \
        "a pre-target game post-dates as_of_date"
    future = before[before["date"] > as_of_date]
    assert future.empty, f"future leakage: {len(future)} games date>as_of"

    frame = pd.concat([before, target_row], ignore_index=True)
    tgt_pk = int(target_row.iloc[0]["game_pk"])
    logs_sub = logs[logs["game_pk"].isin(set(frame["game_pk"]))].copy()

    built = build_features(frame, logs_sub, mode=mode)
    row = built[built["game_pk"] == tgt_pk]
    assert len(row) == 1, f"expected exactly one built target row, got {len(row)}"
    row = row.iloc[0]

    feats = {c: (float(row[c]) if pd.notna(row[c]) else np.nan)
             for c in FEATURE_COLUMNS}
    return {
        "features": feats,
        "games_played_home": int(row["home_games_played"]),
        "games_played_away": int(row["away_games_played"]),
        "home_sp_low_confidence": bool(row["home_sp_low_confidence"]),
        "away_sp_low_confidence": bool(row["away_sp_low_confidence"]),
        "home_sp_trailing_ip": float(row["home_sp_trailing_ip"]),
        "away_sp_trailing_ip": float(row["away_sp_trailing_ip"]),
        "played": played, "mode": mode,
        "target_game_pk": tgt_pk,
    }


def _low_conf_reason(side_low, played, trailing_ip, pp_id):
    """Reason string for a low-confidence starter, or None."""
    if not side_low:
        return None
    if not played and pp_id is None:
        return "starter_not_announced"
    return "starter_low_ip"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_live_features(team_id, opponent_id, as_of_date, side,
                        *, team_pp_id=None, opp_pp_id=None, game_number=None):
    """Feature vector for a game with `team_id` on `side` ('home'|'away') vs
    `opponent_id`, as of as_of_date, via the shared Chunk 3 builder.

    Returns a dict with the 11 FEATURE_COLUMNS (game-level vector), plus
    `low_confidence` (bool, for this team's starter), `low_confidence_reason`,
    and `games_played` (int, for this team as of the date)."""
    if side not in ("home", "away"):
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")
    as_of_date = _as_iso(as_of_date)      # normalize at the module boundary
    if side == "home":
        rec = _reconstruct_game(team_id, opponent_id, as_of_date,
                                home_pp_id=team_pp_id, away_pp_id=opp_pp_id,
                                game_number=game_number)
        gp = rec["games_played_home"]
        low = rec["home_sp_low_confidence"]
        reason = _low_conf_reason(low, rec["played"],
                                  rec["home_sp_trailing_ip"], team_pp_id)
    else:
        rec = _reconstruct_game(opponent_id, team_id, as_of_date,
                                home_pp_id=opp_pp_id, away_pp_id=team_pp_id,
                                game_number=game_number)
        gp = rec["games_played_away"]
        low = rec["away_sp_low_confidence"]
        reason = _low_conf_reason(low, rec["played"],
                                  rec["away_sp_trailing_ip"], team_pp_id)

    out = dict(rec["features"])                      # 11 FEATURE_COLUMNS
    # column-order guard: the live path must key features in features_ml order
    assert list(out.keys()) == FEATURE_COLUMNS, \
        "live feature dict key order differs from FEATURE_COLUMNS"
    out["low_confidence"] = bool(low)
    out["low_confidence_reason"] = reason
    out["games_played"] = int(gp)
    out["played"] = rec["played"]
    return out


def build_game_state(home_id, away_id, as_of_date,
                     *, home_pp_id=None, away_pp_id=None, game_number=None,
                     elo_ratings=None):
    """Assemble both sides into the state dict predict_game consumes. Uses the
    SAME _reconstruct_game as build_live_features (one builder path, built
    once here rather than per-side).

    elo_ratings: optional {team_id: rating} snapshot; each side's rating is
    carried into its state so predict_game's elo path is self-contained."""
    as_of_date = _as_iso(as_of_date)      # normalize at the module boundary
    rec = _reconstruct_game(home_id, away_id, as_of_date,
                            home_pp_id=home_pp_id, away_pp_id=away_pp_id,
                            game_number=game_number)
    f = rec["features"]
    er = elo_ratings or {}

    home_reason = _low_conf_reason(rec["home_sp_low_confidence"], rec["played"],
                                   rec["home_sp_trailing_ip"], home_pp_id)
    away_reason = _low_conf_reason(rec["away_sp_low_confidence"], rec["played"],
                                   rec["away_sp_trailing_ip"], away_pp_id)

    home_state = {
        **{c: f[c] for c in HOME_COLS},
        "team_id": int(home_id),
        "games_played": rec["games_played_home"],
        "low_confidence": rec["home_sp_low_confidence"],
        "low_confidence_reason": home_reason,
        "elo_rating": er.get(int(home_id)),
    }
    away_state = {
        **{c: f[c] for c in AWAY_COLS},                # incl away_tz_delta
        "team_id": int(away_id),
        "games_played": rec["games_played_away"],
        "low_confidence": rec["away_sp_low_confidence"],
        "low_confidence_reason": away_reason,
        "elo_rating": er.get(int(away_id)),
    }
    reasons = []
    if home_reason:
        reasons.append(f"home:{home_reason}")
    if away_reason:
        reasons.append(f"away:{away_reason}")

    context = {
        "home_id": int(home_id), "away_id": int(away_id),
        "as_of_date": as_of_date,
        "played": rec["played"], "mode": rec["mode"],
        "target_game_pk": rec["target_game_pk"],
        "low_confidence": bool(reasons),
        "low_confidence_reasons": reasons,
        # the fully-assembled 11-col vector, for the model path / skew test
        "features": {c: f[c] for c in FEATURE_COLUMNS},
    }
    return {"home_state": home_state, "away_state": away_state,
            "context": context}


if __name__ == "__main__":
    # tiny smoke test against a known 2024 game
    st = build_game_state(147, 111, "2024-04-05")   # example matchup/date
    print("features:", st["context"]["features"])
    print("gp home/away:", st["home_state"]["games_played"],
          st["away_state"]["games_played"],
          "| low_conf:", st["context"]["low_confidence_reasons"])
