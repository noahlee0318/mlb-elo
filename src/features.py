"""Chunk 3: the as-of feature builder — data/features.csv.

Every feature for game i is computed ONLY from rows with
game_order < game_order_i, where game_order is the canonical sort by
(date, game_number, game_pk). Doubleheader game 2 shares a date with game 1
but strictly follows it; sorting by date alone either drops game 1 or leaks
it, which is why game_order exists.

STATE CONVENTIONS (future features must follow the same split, or any
feature combining them is incoherent):
  * TEAM state (form, run differential, games played, tz reference venue)
    RESETS at the season boundary (team_form_cross_season=False default).
  * INDIVIDUAL PITCHER state (FIP / K-rate windows) CARRIES OVER seasons
    (pitcher_cross_season=True default).

Other deliberate choices, recorded so they aren't re-litigated:
  * home_field is OMITTED as a feature — it is a constant offset and the
    model intercept carries it.
  * sp_fip_60 is FIP WITHOUT the league constant:
    (13*HR + 3*(BB+HBP) - 2*K) / IP. The published constant is derived from
    a full completed season (leaks) and is additive (intercept absorbs it).
    Values are therefore NOT comparable to published FIP.
  * The cold-start league average is itself as-of: a trailing-365-day
    league-wide pooled mean over prior STARTER appearances with
    game_order < i. Early-1st-season rows with an empty window emit null
    and are excluded by the caller via the burn-in flag.
  * The pitcher-log input is restricted by the build script to the same
    scope as the games input (regular season; 2020 excluded by
    load_games() defaults) — so early-2021 pitcher windows draw on 2019.
    This mirrors the accessor's exclusion convention.
  * away_tz_delta uses each venue's current UTC offset from
    data/venues.json; a team's first game of a season has no prior venue
    and gets 0 (months off — no acute travel effect). A game with a null
    venue_id (the two Field of Dreams specials) yields null.
  * Rows are NEVER dropped here. in_burn_in flags date < burn_in_end; the
    exclusion belongs to the caller.

Leakage discipline: no feature is computed by an operation that can see the
row's own game or any later game. groupby().transform is forbidden in this
module entirely; every rolling/cumulative aggregate is computed on data
that is group-shifted by one game so the row's own game never enters its
own window (see the "asof-shift" comments at each site). _leak_guard()
enforces both rules against this module's own source at import time.

Build:  python src/features.py     (writes data/venues.json if absent,
                                    then data/features.csv)
"""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.games_data import load_games
from src.park_factors import factors_by_season_for, park_factor_for_game

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PITCHER_CSV = DATA_DIR / "pitcher_games.csv"
VENUES_JSON = DATA_DIR / "venues.json"
FEATURES_CSV = DATA_DIR / "features.csv"

FORM_WINDOW = 15
FORM_MIN_GAMES = 5
REST_CAP = 5
SP_WINDOW_OUTS = 180          # 60 IP
LEAGUE_LOOKBACK_DAYS = 365

FEATURE_COLUMNS = [
    "home_form_15", "away_form_15",
    "home_run_diff_pg", "away_run_diff_pg",
    "home_rest_days", "away_rest_days",
    "home_sp_fip_60", "away_sp_fip_60",
    "home_sp_k_rate", "away_sp_k_rate",
    "away_tz_delta",
    # Chunk 10: game-level venue run-environment multiplier, frozen from prior
    # completed seasons (see src/park_factors.py). One column — NOT a
    # home/away pair — both teams share the venue, so a differential is
    # identically zero. Its near-zero model coefficient is the validation result.
    "park_factor",
]
FLAG_COLUMNS = [
    "home_sp_low_confidence", "away_sp_low_confidence",
    "home_sp_is_opener", "away_sp_is_opener",
    "home_sp_trailing_ip", "away_sp_trailing_ip",
    "home_form_low_confidence", "away_form_low_confidence",
    "home_games_played", "away_games_played",
    "park_factor_imputed",          # metadata (like low_confidence), not a model input
]
KEY_COLUMNS = ["game_pk", "date", "season", "game_order",
               "home_team_id", "away_team_id", "home_win", "in_burn_in"]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_inputs():
    """(games, pitcher_logs) in the canonical modeling scope. Logs are cut
    to the games frame's game_pks so both honor the same exclusions."""
    games = load_games()
    logs = pd.read_csv(PITCHER_CSV, low_memory=False)
    logs = logs[logs["game_type"] == "R"]
    logs = logs[logs["game_pk"].isin(set(games["game_pk"]))].copy()
    logs["is_starter"] = logs["is_starter"].map(
        {True: True, False: False, "True": True, "False": False}).astype(bool)
    logs["is_opener"] = logs["is_opener"].map(
        {True: True, False: False, "True": True, "False": False}).astype(bool)
    return games, logs


def load_venue_offsets():
    if not VENUES_JSON.exists():
        raise FileNotFoundError(
            f"{VENUES_JSON} missing — run `python src/features.py` once "
            "online to fetch venue timezones.")
    raw = json.loads(VENUES_JSON.read_text(encoding="utf-8"))
    return {int(k): int(v["offset"]) for k, v in raw.items()}


def ensure_venue_map(games):
    """Fetch venue -> timezone once and cache to data/venues.json. Fails
    loudly on any venue_id present in the data but absent from the API."""
    if VENUES_JSON.exists():
        offsets = load_venue_offsets()
    else:
        offsets = None
    ids = sorted(int(v) for v in games["venue_id"].dropna().unique())
    if offsets is not None and all(i in offsets for i in ids):
        return
    from src.mlb_api import session
    resp = session.get("https://statsapi.mlb.com/api/v1/venues",
                       params={"venueIds": ",".join(map(str, ids)),
                               "hydrate": "timezone"}, timeout=60)
    resp.raise_for_status()
    got = {}
    for v in resp.json().get("venues", []):
        tz = v.get("timeZone", {}) or {}
        if "offset" in tz:
            got[str(v["id"])] = {"name": v.get("name"),
                                 "tz_id": tz.get("id"),
                                 "offset": int(tz["offset"])}
    missing = [i for i in ids if str(i) not in got]
    if missing:
        raise RuntimeError(f"venues endpoint returned no timezone for "
                           f"venue_ids {missing} — refusing to default to 0")
    tmp = VENUES_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(got, indent=1), encoding="utf-8")
    tmp.replace(VENUES_JSON)


# --------------------------------------------------------------------------
# Canonical ordering
# --------------------------------------------------------------------------

def canonical_order(games):
    """Sort by (date, game_number, game_pk) — NOT date alone: doubleheader
    game 2 shares its date with game 1 and must strictly follow it — and
    assign the monotonic integer game_order used by every strict `<` window."""
    g = games.sort_values(["date", "game_number", "game_pk"],
                          kind="stable").reset_index(drop=True).copy()
    g["game_order"] = np.arange(len(g), dtype=np.int64)
    return g


def _group_shift1(df, keys, col):
    # asof-shift: group-shift by one game so a row's own game never enters
    # its own window; everything downstream of this column is strictly prior
    return df.groupby(keys, sort=False)[col].shift(1)


# --------------------------------------------------------------------------
# Team-state features
# --------------------------------------------------------------------------

def team_long_frame(g):
    """One row per (game, team), in game_order, with won/run_diff/venue."""
    home = pd.DataFrame({
        "game_pk": g["game_pk"], "game_order": g["game_order"],
        "season": g["season"], "date": g["date"], "team_id": g["home_team_id"],
        "is_home": True, "won": g["home_win"].astype("Float64"),
        "run_diff": (g["home_score"] - g["away_score"]).astype("Int64"),
        "venue_id": g["venue_id"],
    })
    away = pd.DataFrame({
        "game_pk": g["game_pk"], "game_order": g["game_order"],
        "season": g["season"], "date": g["date"], "team_id": g["away_team_id"],
        "is_home": False, "won": (1 - g["home_win"]).astype("Float64"),
        "run_diff": (g["away_score"] - g["home_score"]).astype("Int64"),
        "venue_id": g["venue_id"],
    })
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values("game_order", kind="stable").reset_index(drop=True)


def _rolling_prior_sum_count(long, keys, window):
    """Sum and count of NON-NULL `won` over each row's PRIOR `window` games
    within its group. Built as cumulative/rolling aggregates over the raw
    columns, then group-shifted by one game (asof-shift) so the row's own
    result is never inside its own window. Ties (won null) occupy a window
    slot but contribute to neither sum nor count."""
    won0 = long["won"].fillna(0).astype("float64")          # int-valued -> exact
    dec = long["won"].notna().astype("float64")
    grp_w = won0.groupby([long[k] for k in keys], sort=False)
    grp_d = dec.groupby([long[k] for k in keys], sort=False)
    sum_incl = grp_w.rolling(window, min_periods=1).sum()   # window incl. self
    cnt_incl = grp_d.rolling(window, min_periods=1).sum()
    sum_incl.index = sum_incl.index.get_level_values(-1)
    cnt_incl.index = cnt_incl.index.get_level_values(-1)
    tmp = pd.DataFrame({"s": sum_incl.sort_index(), "c": cnt_incl.sort_index()})
    for k in keys:
        tmp[k] = long[k]
    # asof-shift: slide the inclusive window back one game
    s_prior = tmp.groupby(keys, sort=False)["s"].shift(1)
    c_prior = tmp.groupby(keys, sort=False)["c"].shift(1)
    return s_prior, c_prior


def add_team_features(long, cross_season, venue_offsets):
    form_keys = ["team_id"] if cross_season else ["team_id", "season"]

    s_prior, c_prior = _rolling_prior_sum_count(long, form_keys, FORM_WINDOW)
    long["form_15"] = np.where(c_prior >= FORM_MIN_GAMES,
                               s_prior / c_prior, np.nan)

    # games_played / run_diff always reset per season regardless of the
    # form window's cross-season switch
    long["games_played"] = long.groupby(["team_id", "season"],
                                        sort=False).cumcount()
    long["form_low_confidence"] = long["games_played"] < FORM_WINDOW

    rd = long["run_diff"].astype("float64")                  # int-valued -> exact
    csum_incl = rd.groupby([long["team_id"], long["season"]], sort=False).cumsum()
    long["_csum_incl"] = csum_incl
    # asof-shift: cumulative run diff through the PREVIOUS game only
    csum_prior = _group_shift1(long, ["team_id", "season"], "_csum_incl")
    gp = long["games_played"].to_numpy()
    long["run_diff_pg"] = np.where(gp >= FORM_MIN_GAMES,
                                   csum_prior / np.where(gp == 0, 1, gp),
                                   np.nan)
    long.drop(columns="_csum_incl", inplace=True)

    # rest days: previous game across seasons; cap handles the offseason
    long["_d"] = pd.to_datetime(long["date"])
    prev_date = _group_shift1(long, ["team_id"], "_d")
    rest = (long["_d"] - prev_date).dt.days
    long["rest_days"] = rest.clip(upper=REST_CAP).fillna(REST_CAP).astype("int64")
    long.drop(columns="_d", inplace=True)

    # tz delta: previous venue WITHIN the season (team state resets);
    # first game of season -> 0; unknown venue -> null
    prev_venue = _group_shift1(long, ["team_id", "season"], "venue_id")
    off_cur = long["venue_id"].map(venue_offsets)
    off_prev = prev_venue.map(venue_offsets)
    tz = (off_cur - off_prev).abs()
    tz[prev_venue.isna() & long["venue_id"].notna()] = 0     # season opener
    long["tz_delta"] = tz.astype("Float64")                  # null iff venue unknown
    return long


# --------------------------------------------------------------------------
# Pitcher-state features
# --------------------------------------------------------------------------

class PitcherIndex:
    """Per-pitcher prefix sums over appearances in game_order, plus a
    league-wide starter index for the as-of trailing-365-day average."""

    STAT_COLS = ["ip_outs", "home_runs", "walks", "hit_by_pitch",
                 "strikeouts", "batters_faced"]

    def __init__(self, logs, order_of_pk, date_of_order):
        lg = logs.copy()
        lg["game_order"] = lg["game_pk"].map(order_of_pk)
        lg = lg.dropna(subset=["game_order"])
        lg["game_order"] = lg["game_order"].astype("int64")
        lg = lg.sort_values(["pitcher_id", "game_order"],
                            kind="stable").reset_index(drop=True)
        self.by_pitcher = {}
        for pid, grp in lg.groupby("pitcher_id", sort=False):
            arrs = {c: grp[c].to_numpy(dtype="int64") for c in self.STAT_COLS}
            cums = {c: np.concatenate(([0], np.cumsum(a)))
                    for c, a in arrs.items()}
            self.by_pitcher[int(pid)] = (grp["game_order"].to_numpy(), cums)

        st = lg[lg["is_starter"]].sort_values("game_order",
                                              kind="stable").reset_index(drop=True)
        self.lg_orders = st["game_order"].to_numpy()
        self.lg_dates = pd.to_datetime(
            st["game_order"].map(date_of_order)).to_numpy(dtype="datetime64[D]")
        self.lg_cums = {c: np.concatenate(
            ([0], np.cumsum(st[c].to_numpy(dtype="int64"))))
            for c in self.STAT_COLS}
        # starter identity per (game_pk, team): id, is_opener
        self.starter_of = {(int(r.game_pk), int(r.team_id)):
                           (int(r.pitcher_id), bool(r.is_opener))
                           for r in st.itertuples(index=False)}

    @staticmethod
    def _fip_krate(sums):
        outs, hr = sums["ip_outs"], sums["home_runs"]
        bb, hbp = sums["walks"], sums["hit_by_pitch"]
        k, bf = sums["strikeouts"], sums["batters_faced"]
        fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / (outs / 3.0)
               if outs > 0 else np.nan)
        krate = k / bf if bf > 0 else np.nan
        return fip, krate

    def league_asof(self, order_i, date_i):
        """Pooled league mean over starter appearances in the trailing 365
        days with game_order < order_i. (None, None) if the window is empty."""
        hi = int(np.searchsorted(self.lg_orders, order_i, side="left"))
        cutoff = np.datetime64(date_i, "D") - np.timedelta64(LEAGUE_LOOKBACK_DAYS, "D")
        lo = int(np.searchsorted(self.lg_dates, cutoff, side="left"))
        if lo >= hi:
            return None, None
        sums = {c: int(self.lg_cums[c][hi] - self.lg_cums[c][lo])
                for c in self.STAT_COLS}
        return self._fip_krate(sums)

    def trailing(self, pid, order_i):
        """(fip, krate, trailing_ip_outs, enough) over the pitcher's prior
        appearances walked back until SP_WINDOW_OUTS is reached."""
        entry = self.by_pitcher.get(int(pid)) if pd.notna(pid) else None
        if entry is None:
            return None, None, 0, False
        orders, cums = entry
        m = int(np.searchsorted(orders, order_i, side="left"))
        total = int(cums["ip_outs"][m])
        if total < SP_WINDOW_OUTS:
            if total == 0:
                return None, None, 0, False
            sums = {c: int(cums[c][m]) for c in self.STAT_COLS}
            fip, kr = self._fip_krate(sums)
            return fip, kr, total, False
        j = int(np.searchsorted(cums["ip_outs"][:m + 1],
                                total - SP_WINDOW_OUTS, side="right")) - 1
        sums = {c: int(cums[c][m] - cums[c][j]) for c in self.STAT_COLS}
        fip, kr = self._fip_krate(sums)
        return fip, kr, total - int(cums["ip_outs"][j]), True


def add_pitcher_features(g, pindex, mode):
    """The ONLY thing mode changes is which pitcher id is looked up:
    'train' = the actual starter (known post-game), 'predict' = the listed
    probable. Window logic is identical."""
    out = {side: {"fip": [], "kr": [], "ip": [], "low": [], "opener": []}
           for side in ("home", "away")}
    for r in g.itertuples(index=False):
        for side in ("home", "away"):
            team = getattr(r, f"{side}_team_id")
            if mode == "train":
                sid_op = pindex.starter_of.get((int(r.game_pk), int(team)))
                pid, opener = (sid_op if sid_op else (None, None))
            else:
                pid = getattr(r, f"{side}_probable_pitcher_id")
                pid = int(pid) if pd.notna(pid) else None
                opener = None      # unknowable pre-game
            fip, kr, outs, enough = (pindex.trailing(pid, r.game_order)
                                     if pid is not None
                                     else (None, None, 0, False))
            if not enough:
                lf, lk = pindex.league_asof(r.game_order, r.date)
                fip = lf if lf is not None else np.nan
                kr = lk if lk is not None else np.nan
            o = out[side]
            o["fip"].append(fip if fip is not None else np.nan)
            o["kr"].append(kr if kr is not None else np.nan)
            o["ip"].append(outs / 3.0)
            o["low"].append(not enough)
            o["opener"].append(opener)
    for side in ("home", "away"):
        o = out[side]
        g[f"{side}_sp_fip_60"] = o["fip"]
        g[f"{side}_sp_k_rate"] = o["kr"]
        g[f"{side}_sp_trailing_ip"] = o["ip"]
        g[f"{side}_sp_low_confidence"] = o["low"]
        g[f"{side}_sp_is_opener"] = pd.array(o["opener"], dtype="boolean")
    return g


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------

def build_features(games_df, pitcher_logs_df, mode="train",
                   team_form_cross_season=False, pitcher_cross_season=True,
                   burn_in_end="2015-05-15", return_provenance=False,
                   provenance_game_pks=None):
    """One feature row per game in games_df. See module docstring for the
    conventions. Emits every row; in_burn_in marks rows the CALLER should
    exclude from training (exclusion is not performed here).

    return_provenance=True (ADDITIVE audit path, Chunk 4) additionally
    returns feature_provenance(...) for provenance_game_pks (default: all
    games). It changes no computation — see feature_provenance's docstring
    for exactly what the provenance sets are and are not."""
    assert mode in ("train", "predict")
    venue_offsets = load_venue_offsets()

    g = canonical_order(games_df)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    date_of_order = dict(zip(g["game_order"], g["date"]))

    long = team_long_frame(g)
    long = add_team_features(long, team_form_cross_season, venue_offsets)

    def side_merge(is_home, prefix, cols):
        sub = long[long["is_home"] == is_home][["game_pk"] + cols]
        sub = sub.rename(columns={c: f"{prefix}_{c}" for c in cols})
        return sub

    team_cols = ["form_15", "run_diff_pg", "rest_days", "games_played",
                 "form_low_confidence", "tz_delta"]
    g = g.merge(side_merge(True, "home", team_cols), on="game_pk")
    g = g.merge(side_merge(False, "away", team_cols), on="game_pk")
    g = g.drop(columns=["home_tz_delta"])          # only the away delta is a feature

    if not pitcher_cross_season:
        pitcher_logs_df = pitcher_logs_df.merge(
            g[["game_pk", "season"]], on="game_pk", how="inner",
            suffixes=("", "_g"))
        # (non-default path) restrict each pitcher's window to his current
        # season by splitting pitcher identity per season
        pitcher_logs_df = pitcher_logs_df.assign(
            pitcher_id=pitcher_logs_df["pitcher_id"].astype("int64")
            * 10000 + pitcher_logs_df["season"].astype("int64") % 10000)

    pindex = PitcherIndex(pitcher_logs_df, order_of_pk, date_of_order)
    g = add_pitcher_features(g, pindex, mode)

    g["in_burn_in"] = g["date"] < burn_in_end
    g["home_form_15"] = g["home_form_15"].astype("Float64")
    g["away_form_15"] = g["away_form_15"].astype("Float64")

    # Chunk 10 — park_factor: each game gets its SEASON's frozen factor,
    # computed from prior completed seasons only (2020 excluded). Leakage-clean
    # by construction: factors_by_season_for computes each season N's factors
    # from g's seasons < N, so no season-N game influences its own factor. A
    # null/new venue with no prior-window history -> neutral 1.0, flagged.
    seasons_present = sorted(int(s) for s in g["season"].dropna().unique())
    factors_by_season = factors_by_season_for(g, seasons_present)
    pf, pf_imp = [], []
    for season, venue in zip(g["season"].to_numpy(), g["venue_id"].to_numpy()):
        f, imp = park_factor_for_game(season, venue, factors_by_season)
        pf.append(f)
        pf_imp.append(imp)
    g["park_factor"] = pf
    g["park_factor_imputed"] = pd.array(pf_imp, dtype="boolean")

    out = g[KEY_COLUMNS + FEATURE_COLUMNS + FLAG_COLUMNS].copy()
    if return_provenance:
        pks = (provenance_game_pks if provenance_game_pks is not None
               else out["game_pk"].tolist())
        return out, feature_provenance(games_df, pitcher_logs_df, pks,
                                       mode=mode)
    return out


# --------------------------------------------------------------------------
# Provenance (ADDITIVE audit path — Chunk 4, Test 6)
# --------------------------------------------------------------------------

def feature_provenance(games_df, pitcher_logs_df, game_pks, mode="train"):
    """Per requested game and per feature: the contributing game_pks and
    their max game_order.

    HONESTY NOTE: these sets are built from the DEFINITIONAL strict
    `game_order < i` slices, not by instrumenting the vectorized kernels
    (instrumenting them would mean restructuring feature logic, which the
    audit forbids). The definitions and the fast path are established as
    equal by Defense 1 (3,000-row exact reference comparison) and the
    audit's manual traces; provenance then verifies the definitions
    themselves — notably that a doubleheader game 1 contributes to game 2
    and nothing at or after a row's own game_order ever contributes.
    League-average fallbacks contribute hundreds of games, so they are
    summarized as {count, max_order} rather than enumerated.

    Returns {game_pk: {feature_name: {"pks": [...] | None,
                                      "count": int, "max_order": int|None}}}.
    """
    g = canonical_order(games_df)
    order_of_pk = dict(zip(g["game_pk"].astype(int), g["game_order"]))
    logs = pitcher_logs_df.copy()
    logs["game_order"] = logs["game_pk"].map(order_of_pk)
    logs = logs.dropna(subset=["game_order"])
    logs["game_order"] = logs["game_order"].astype("int64")

    def entry(pks_orders):
        pks = [int(p) for p, _ in pks_orders]
        orders = [int(o) for _, o in pks_orders]
        return {"pks": pks, "count": len(pks),
                "max_order": max(orders) if orders else None}

    out = {}
    for pk in game_pks:
        row = g.loc[g["game_pk"] == pk].iloc[0]
        i = int(row["game_order"])
        prior_g = g[g["game_order"] < i]
        prior_l = logs[logs["game_order"] < i]
        rec = {}
        for side in ("home", "away"):
            team = row[f"{side}_team_id"]
            mine = prior_g[((prior_g["home_team_id"] == team)
                            | (prior_g["away_team_id"] == team))
                           & (prior_g["season"] == row["season"])]
            mine = mine.sort_values("game_order")
            po = list(zip(mine["game_pk"], mine["game_order"]))
            rec[f"{side}_form_15"] = entry(po[-FORM_WINDOW:])
            rec[f"{side}_run_diff_pg"] = entry(po)
            mine_all = prior_g[(prior_g["home_team_id"] == team)
                               | (prior_g["away_team_id"] == team)]
            mine_all = mine_all.sort_values("game_order")
            pa = list(zip(mine_all["game_pk"], mine_all["game_order"]))
            rec[f"{side}_rest_days"] = entry(pa[-1:])
            if side == "away":
                rec["away_tz_delta"] = entry(po[-1:])

            if mode == "train":
                me = logs[(logs["game_pk"] == pk) & (logs["team_id"] == team)
                          & (logs["is_starter"])]
                pid = int(me.iloc[0]["pitcher_id"]) if len(me) else None
            else:
                pid = row[f"{side}_probable_pitcher_id"]
                pid = int(pid) if pd.notna(pid) else None
            if pid is None:
                hist = prior_l.iloc[0:0]
            else:
                hist = prior_l[prior_l["pitcher_id"] == pid].sort_values("game_order")
            outs_rev = hist["ip_outs"].to_numpy()[::-1]
            acc, used = 0, 0
            for o in outs_rev:
                used += 1
                acc += int(o)
                if acc >= SP_WINDOW_OUTS:
                    break
            enough = acc >= SP_WINDOW_OUTS
            window = hist.tail(used) if enough else hist
            sp_entry = entry(list(zip(window["game_pk"], window["game_order"])))
            if not enough:
                cutoff = (pd.Timestamp(row["date"])
                          - pd.Timedelta(days=LEAGUE_LOOKBACK_DAYS)
                          ).strftime("%Y-%m-%d")
                lg = prior_l[(prior_l["is_starter"]) & (prior_l["date"] >= cutoff)]
                sp_entry = {"pks": None, "count": int(len(lg)),
                            "max_order": (int(lg["game_order"].max())
                                          if len(lg) else None),
                            "league_fallback": True}
            rec[f"{side}_sp_fip_60"] = sp_entry
            rec[f"{side}_sp_k_rate"] = sp_entry
        out[int(pk)] = rec
    return out


# --------------------------------------------------------------------------
# Source-level leakage guard
# --------------------------------------------------------------------------

def _leak_guard():
    """Forbid, in this module's own source: any groupby-transform call
    (full-column ops can see the row's own game), and any rolling-window
    call outside the audited helper whose results are group-shifted by one
    game before any row can read them. Runs at import."""
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    # needles are split so the guard's own source never matches them
    t_needle = ".trans" + "form("
    r_needle = ".roll" + "ing("
    assert t_needle not in src, "groupby().transform is forbidden here"
    for ln in src.splitlines():
        if r_needle in ln and "min_periods=1).sum()" not in ln:
            raise AssertionError(f"unaudited rolling call: {ln.strip()}")


_leak_guard()


def main():
    games, logs = load_inputs()
    ensure_venue_map(games)
    feats = build_features(games, logs, mode="train")
    tmp = FEATURES_CSV.with_suffix(".csv.tmp")
    feats.to_csv(tmp, index=False)
    tmp.replace(FEATURES_CSV)
    n_burn = int(feats["in_burn_in"].sum())
    print(f"wrote {len(feats)} rows -> {FEATURES_CSV} "
          f"({n_burn} flagged in_burn_in, kept in file)")


if __name__ == "__main__":
    main()
