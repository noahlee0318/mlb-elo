"""Chunk 5, Deliverable 1: chronological train / tune / val / test splits over
the Chunk 3 feature table (data/features.csv).

2020 is excluded from every model split (it is already absent from the
feature table). The split is by SEASON and strictly chronological — no
random splitting anywhere in this chunk.

Eval-row filter applied inside load_split():
  * cold-start: drop a game unless BOTH teams have >= 15 prior games that
    season (the filter is NOT baked into the stored table; it is applied
    here). Equivalent to dropping rows flagged home/away_form_low_confidence.
  * postseason excluded — the feature table is regular-season-only by
    construction (built from load_games() defaults), so this is already
    satisfied; asserted defensively.
  * low_confidence rows are KEPT.

low_confidence (single derived column — the stored table has four flags, no
single one): defined as home_sp_low_confidence OR away_sp_low_confidence, i.e.
either starting pitcher had < 60 trailing IP and got a league-average
imputation. The two form-cold-start flags cannot survive the >=15 filter, so
they carry no information post-filter and are intentionally excluded from the
definition.
"""

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import FEATURE_COLUMNS, FEATURES_CSV

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
TUNE_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022]   # train minus validation
VAL_SEASONS = [2023]                                        # carved from END of train
TEST_SEASONS = [2024, 2025]

SPLIT_SEASONS = {"train": TRAIN_SEASONS, "tune": TUNE_SEASONS,
                 "val": VAL_SEASONS, "test": TEST_SEASONS}

COLD_START_MIN_GAMES = 15
_BOOL = {True: True, False: False, "True": True, "False": False}
_FLAG_COLS = ["home_sp_low_confidence", "away_sp_low_confidence",
              "home_form_low_confidence", "away_form_low_confidence",
              "home_sp_is_opener", "away_sp_is_opener", "in_burn_in"]


_FILTERED_CACHE = {}


def _load_filtered():
    """The whole feature table after the eval-row filter, with a derived
    low_confidence column and NaN-free features.

    Cached on features.csv (mtime, size): every load_split/eval_row_count
    call in one run reuses a single read+filter, and the one-time stderr
    notices print once. The key invalidates automatically if the file is
    rebuilt, so there is no staleness trap. Callers only ever slice-and-copy
    the result, never mutate it in place."""
    stat = os.stat(FEATURES_CSV)
    key = (stat.st_mtime_ns, stat.st_size)
    hit = _FILTERED_CACHE.get("frame")
    if hit is not None and hit[0] == key:
        return hit[1]
    df = _compute_filtered()
    _FILTERED_CACHE["frame"] = (key, df)
    return df


def _compute_filtered():
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    for c in _FLAG_COLS:
        df[c] = df[c].map(_BOOL).astype("boolean")

    df["low_confidence"] = (df["home_sp_low_confidence"].fillna(False)
                            | df["away_sp_low_confidence"].fillna(False)).astype(bool)

    # drop null-label rows: a tie (home_win null — one 2016 rain-shortened
    # 1-1 game in the whole table) has no binary label and cannot be a
    # modeling row. Same treatment as the Chunk 4 audit.
    n_tie = int(df["home_win"].isna().sum())
    if n_tie:
        print(f"[splits] dropping {n_tie} null-label (tie) row(s)",
              file=sys.stderr)
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = df["home_win"].astype(int)

    # cold-start: both teams must have >= 15 prior games this season
    keep = (df["home_games_played"] >= COLD_START_MIN_GAMES) \
        & (df["away_games_played"] >= COLD_START_MIN_GAMES)
    df = df[keep].copy()

    # the only feature that can still be null here is away_tz_delta, for the
    # two neutral-site "Field of Dreams" games whose venue_id is unknown
    # (Chunk 3 leaves it null on purpose). Fill those with 0 (assume no time
    # change) at LOAD time so the modeling splits are NaN-free; the stored
    # feature table keeps the honest null.
    tz_null = int(df["away_tz_delta"].isna().sum())
    if tz_null:
        bad = df.loc[df["away_tz_delta"].isna(), "game_pk"].tolist()
        print(f"[splits] filling away_tz_delta=0 for {tz_null} unknown-venue "
              f"game(s): {bad}", file=sys.stderr)
        df["away_tz_delta"] = df["away_tz_delta"].fillna(0.0)
    return df


def load_split(name):
    """One split as a DataFrame: game_pk, date, season, the feature columns,
    home_win, low_confidence. Cold-start filtered, postseason-free, NaN-free;
    low_confidence rows retained."""
    if name not in SPLIT_SEASONS:
        raise ValueError(f"unknown split {name!r}; expected one of "
                         f"{list(SPLIT_SEASONS)}")
    df = _load_filtered()
    df = df[df["season"].isin(SPLIT_SEASONS[name])].copy()
    out = df[["game_pk", "date", "season"] + FEATURE_COLUMNS
             + ["home_win", "low_confidence"]].copy()
    nan_cols = [c for c in FEATURE_COLUMNS if out[c].isna().any()]
    assert not nan_cols, f"NaN in feature columns of split {name!r}: {nan_cols}"
    assert out["home_win"].notna().all(), f"null home_win in split {name!r}"
    return out.sort_values("game_pk").reset_index(drop=True)


def eval_row_count(seasons):
    """Post-eval-filter row count for a set of seasons, computed straight from
    the feature table (used by the run_baselines row-count tripwire)."""
    df = _load_filtered()
    return int(df[df["season"].isin(seasons)].shape[0])


def validate_splits():
    """Raise on any structural violation. Note: the task's bullet 'no game_pk
    in more than one of train/val/test' is literally impossible because
    val (2023) is a subset of train (tune ∪ val == train), so it is
    implemented as the coherent set: {tune, val, test} pairwise-disjoint,
    tune ∪ val == train, and train ∩ test == ∅ (the leakage-relevant check)."""
    parts = {n: load_split(n) for n in SPLIT_SEASONS}
    pk = {n: set(parts[n]["game_pk"]) for n in parts}

    # {tune, val, test} pairwise disjoint
    for a, b in (("tune", "val"), ("tune", "test"), ("val", "test")):
        inter = pk[a] & pk[b]
        assert not inter, f"{a} and {b} overlap on {len(inter)} game_pk(s)"
    # tune ∪ val == train
    assert pk["tune"] | pk["val"] == pk["train"], "tune ∪ val != train"
    # train ∩ test empty (the real leakage check)
    assert not (pk["train"] & pk["test"]), "train and test overlap"
    # chronological: max train date strictly < min test date
    assert parts["train"]["date"].max() < parts["test"]["date"].min(), \
        "train max date not strictly before test min date"
    # no NaN features (already asserted per split, re-checked here)
    for n, p in parts.items():
        assert not p[FEATURE_COLUMNS].isna().any().any(), f"NaN features in {n}"

    print("split validation OK:")
    for n in ("train", "tune", "val", "test"):
        p = parts[n]
        print(f"  {n:<6} n={len(p):>5}  seasons={SPLIT_SEASONS[n]}  "
              f"dates {p['date'].min()}..{p['date'].max()}  "
              f"low_conf={int(p['low_confidence'].sum())}")
    return True


if __name__ == "__main__":
    validate_splits()
