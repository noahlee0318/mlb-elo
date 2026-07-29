"""Streamlit dashboard: today's MLB slate with the home team's Elo win
chance, plus a Methods tab explaining the model.

Data freshens itself on open (see src/refresh.py) — no scheduler needed.

Run from the project root:  streamlit run app.py
"""

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from src.matchup import (FORMATS, check_matchup, season_sim_occurrences,
                         simulate_outcomes, win_probability)
from src.playoffs import series_win_prob
from src.predict import PREDICTIONS_CSV, predict_for_date
from src.prob_model import EloProbabilityModel
from src.daily_refresh import refresh_if_stale
from src.refresh import STALE_AFTER_HOURS, load_ratings
from src.schedule import todays_date_str
from src.score import score_latest_day
from src.season_schedule import remaining_schedule, team_structure
from src.simulate import simulate_season
from src.standings import assign_playoff_labels, current_records, played_h2h

N_SIMS = 10000
SIM_SEED = 42

# Official MLB team marks, served by MLB's own static CDN and keyed on the
# same stable team_id everything else in this project uses.
LOGO_URL = "https://www.mlbstatic.com/team-logos/{team_id}.svg"

CELL_STYLE = "padding:10px 12px;border-bottom:1px solid rgba(128,128,128,0.3);"

st.set_page_config(page_title="MLB Elo predictor", page_icon="⚾")

st.title("MLB Elo — today's slate")


@st.cache_data(ttl=STALE_AFTER_HOURS * 3600, show_spinner=False)
def cached_scoreline():
    """All grading lives in src/score.py; this only caches its answer so
    reruns within the staleness window never re-hit the API."""
    return score_latest_day()


def _pretty_date(d):  # "2026-07-16" -> "Jul 16"
    return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d")


scoreline = cached_scoreline()
if scoreline["status"] == "ok":
    pct = round(100 * scoreline["correct"] / scoreline["total"])
    line = (f'Predictions for {_pretty_date(scoreline["date"])}: '
            f'{scoreline["correct"]}/{scoreline["total"]} correct ({pct}%)')
    if scoreline["pending"]:
        line += f' · {scoreline["pending"]} game(s) not final yet'
    if scoreline["pushes"]:
        line += f' · {scoreline["pushes"]} exact-50% no-pick(s) excluded'
    st.caption(line)
elif scoreline["status"] == "pending":
    st.caption(f'Predictions for {_pretty_date(scoreline["date"])} are '
               "logged — no final results to grade yet.")
else:
    st.caption("No graded predictions yet — the first score appears once "
               "a logged slate's games go final.")

tab_slate, tab_proj, tab_h2h, tab_methods = st.tabs(
    ["Today's slate", "Season projection", "H2H Sim", "Methods"])


@st.cache_data(ttl=STALE_AFTER_HOURS * 3600, show_spinner=False)
def auto_refresh():
    """Streamlit reruns this script on every widget click; the TTL cache
    means at most one staleness check per window actually executes, and
    refresh_if_stale (src/daily_refresh.py — the one gate) skips the network
    when the snapshot is fresh. with_boxscores=False so a reopen never blocks
    on the minutes-long box pull; the button and scripts/run_refresh.py do the
    boxscore rebuild."""
    return refresh_if_stale()


@st.cache_data(show_spinner="Fetching today's slate…")
def cached_predictions(date_str, ratings_stamp):
    """ratings_stamp keys the cache to the ratings snapshot's timestamp, so
    a data refresh automatically invalidates cached predictions."""
    ratings, _names, _ts = load_ratings()
    return predict_for_date(date_str, ratings=ratings)


# Chunk 9: the three shipped models. Calibrated GB is the default foreground.
MODEL_KEYS = ["histgb_cal", "logreg", "elo"]
MODEL_LABELS = {"histgb_cal": "Calibrated Gradient Boosting",
                "logreg": "Logistic Regression", "elo": "Elo"}


@st.cache_data(show_spinner="Running models…")
def cached_game_models(home_id, away_id, date_str, ratings_stamp,
                       home_pp_id=None, away_pp_id=None):
    """All three models' results for one matchup as of date_str, via the shared
    live feature builder (src/live_features) + src.predict.predict_game. Cached
    per matchup + ratings snapshot + probable-starter ids, so the heavy per-game
    feature rebuild runs at most once per game per refresh. The pp ids let the
    live path use the real starter's trailing stats instead of imputing (they
    are ignored for a game that is already final — build_live_features resolves
    those to the actual starter). Returns {model_key: predict_game dict} or
    {'error': str} if features can't be built (kept out of the render)."""
    try:
        from src.live_features import build_game_state
        from src.predict import predict_all_models
        ratings, _names, _ts = load_ratings()
        state = build_game_state(int(home_id), int(away_id), date_str,
                                 home_pp_id=home_pp_id, away_pp_id=away_pp_id,
                                 elo_ratings=ratings)
        return predict_all_models(state["home_state"], state["away_state"],
                                  state["context"])
    except Exception as exc:                       # never crash the slate tab
        return {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(ttl=STALE_AFTER_HOURS * 3600, show_spinner="Fetching remaining schedule…")
def cached_remaining(ratings_stamp):
    """Remaining regular-season games; keyed to the ratings stamp so a data
    refresh (which may fold new finals in) refetches the schedule."""
    return remaining_schedule()


@st.cache_data(ttl=STALE_AFTER_HOURS * 3600, show_spinner=False)
def cached_h2h(ratings_stamp):
    """Played head-to-head results, for tiebreaks and the season-series
    line; keyed to the ratings stamp like every other data cache."""
    return played_h2h()


@st.cache_data(show_spinner=f"Simulating {N_SIMS:,} seasons…")
def cached_projection(ratings_stamp, sched_key, n_sims, seed):
    """The Monte Carlo projection, cached on ratings snapshot + schedule
    hash + n_sims + seed: Streamlit reruns (clicks, tab switches) hit this
    cache, so the simulation runs at most once per data refresh."""
    remaining = cached_remaining(ratings_stamp)
    ratings, _names, _ts = load_ratings()
    return simulate_season(remaining, current_records(), team_structure(),
                           EloProbabilityModel(ratings), n_sims=n_sims,
                           seed=seed, played_h2h=played_h2h())


# Weakness 2: sim/matchup model selector. Elo is instant; the ML models build
# each team's as-of-today features once (~a couple minutes) then drive the SAME
# unchanged engine. Default is Calibrated Gradient Boosting.
SIM_MODEL_LABELS = ["Calibrated Gradient Boosting", "Logistic Regression", "Elo"]
SIM_MODEL_KEY = {"Calibrated Gradient Boosting": "histgb_cal",
                 "Logistic Regression": "logreg", "Elo": "elo"}
SIM_KEY_LABEL = {v: k for k, v in SIM_MODEL_KEY.items()}


@st.cache_data(show_spinner=f"Simulating {N_SIMS:,} seasons per model…")
def cached_projections(model_keys, ratings_stamp, sched_key, n_sims, seed):
    """Season projection per requested model. The ML models SHARE one
    as-of-today feature cache (build_models), so the ~60 build_live_features
    calls happen once for the whole comparison and only the predict step
    repeats. The engine is untouched — each model is just a different
    ProbabilityModel handed to the unchanged simulate_season."""
    from src.ml_prob_model import build_models
    remaining = cached_remaining(ratings_stamp)
    records, structure, h2h = current_records(), team_structure(), played_h2h()
    ml_keys = [k for k in model_keys if k != "elo"]
    models = build_models(ml_keys) if ml_keys else {}
    if "elo" in model_keys:
        ratings, _n, _t = load_ratings()
        models["elo"] = EloProbabilityModel(ratings)
    return {k: simulate_season(remaining, records, structure, models[k],
                               n_sims=n_sims, seed=seed, played_h2h=h2h)
            for k in model_keys}


@st.cache_resource(show_spinner="Loading matchup model…")
def matchup_model(model_key, ratings_stamp):
    """One ProbabilityModel for the matchup tool. cache_resource (by reference,
    no copy) so the ML model's lazy 2-team feature cache persists across reruns.
    A single matchup only builds the two teams' features (~seconds), not all 30."""
    if model_key == "elo":
        ratings, _n, _t = load_ratings()
        return EloProbabilityModel(ratings)
    from src.ml_prob_model import MLProbabilityModel
    return MLProbabilityModel(model_key)


def series_pgame(model, higher, lower, pattern):
    """The reference (higher) side's per-game win probs following the hosting
    pattern — built exactly as playoffs.py builds them (model.matchup_prob)."""
    return [model.matchup_prob(higher, lower) if hosts
            else 1.0 - model.matchup_prob(lower, higher) for hosts in pattern]


def series_length_dist(p_by_game):
    """ANALYTIC (no sampling) P(reference clinches with a win in exactly game g),
    for g over the series. Same DP as playoffs.series_win_prob, additionally
    recording the clinching game — NOT a Monte Carlo series sampler. The values
    over g sum to the series win probability."""
    need = len(p_by_game) // 2 + 1
    states = {(0, 0): 1.0}
    win_in = {}
    for g, p in enumerate(p_by_game, start=1):
        nxt = {}
        for (w, l), mass in states.items():
            if mass == 0.0:
                continue
            if w + 1 == need:
                win_in[g] = win_in.get(g, 0.0) + mass * p
            else:
                nxt[(w + 1, l)] = nxt.get((w + 1, l), 0.0) + mass * p
            if l + 1 < need:
                nxt[(w, l + 1)] = nxt.get((w, l + 1), 0.0) + mass * (1.0 - p)
        states = nxt
    return win_in


def logged_fallback(date_str):
    """Offline last resort: matchups and chances replayed from
    data/predictions.csv (logged earlier today), names from the ratings
    snapshot. No pitchers or venue without the network."""
    if not PREDICTIONS_CSV.exists():
        return []
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[df["date"] == date_str]
    _ratings, names, _ts = load_ratings()
    names = names or {}
    return [{
        "away_id": int(r.away_id), "home_id": int(r.home_id),
        "away_name": names.get(int(r.away_id), f"Team {int(r.away_id)}"),
        "home_name": names.get(int(r.home_id), f"Team {int(r.home_id)}"),
        "away_probable_pitcher": "pitcher unavailable offline",
        "home_probable_pitcher": "pitcher unavailable offline",
        "elo_prob_home": float(r.elo_prob_home),
    } for r in df.itertuples(index=False)]


def team_block(team_id, name, pitcher, favored):
    """One team: logo, name (bold when Elo favors this side), pitcher beneath."""
    weight = 700 if favored else 400
    return (
        '<div style="display:flex;align-items:center;gap:8px;">'
        f'<img src="{LOGO_URL.format(team_id=team_id)}" width="26" height="26" alt="">'
        f'<div><div style="font-weight:{weight};">{html.escape(name)}</div>'
        f'<div style="font-size:0.85em;font-weight:400;opacity:0.65;">{html.escape(pitcher)}</div>'
        "</div></div>"
    )


def matchup_cell(g):
    """'away @ home' on one line, pitchers under their teams, venue above;
    doubleheaders get a Game 1/2 label above the venue line."""
    # .get with defaults: st.cache_data may replay slates fetched before
    # these fields existed, and a stale entry must not crash the page
    lines = []
    if g.get("doubleheader", "N") != "N":
        lines.append('<div style="font-size:0.8em;font-weight:600;opacity:0.7;">'
                     f'Doubleheader — Game {g.get("game_num", 1)}</div>')
    venue = g.get("venue_name")
    if venue:
        lines.append('<div style="font-size:0.8em;opacity:0.7;margin-bottom:4px;">'
                     f'{html.escape(venue)}</div>')
    home_favored = g["elo_prob_home"] >= 0.5
    lines.append(
        '<div style="display:flex;align-items:flex-start;gap:10px;flex-wrap:wrap;">'
        + team_block(g["away_id"], g["away_name"],
                     g["away_probable_pitcher"], not home_favored)
        + '<div style="align-self:center;opacity:0.65;">@</div>'
        + team_block(g["home_id"], g["home_name"],
                     g["home_probable_pitcher"], home_favored)
        + "</div>")
    return "".join(lines)


def slate_table(preds):
    rows = "".join(
        f'<tr><td style="{CELL_STYLE}">{matchup_cell(g)}</td>'
        f'<td style="{CELL_STYLE}text-align:center;vertical-align:middle;'
        f'font-size:1.15em;font-weight:600;">{g["elo_prob_home"]:.1%}</td></tr>'
        for g in preds)
    return (
        '<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr><th style="{CELL_STYLE}text-align:left;">Matchup</th>'
        f'<th style="{CELL_STYLE}text-align:center;">Home team\'s chance to win'
        "</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def model_comparison_block(preds, date_str, stamp, choice_key):
    """Three models side by side for the slate, the selected one foregrounded.
    All three stay visible regardless of selection; cold-start and low-confidence
    games are called out honestly rather than silently served as Elo."""
    header_cells = ""
    for k in MODEL_KEYS:
        sel = k == choice_key
        header_cells += (
            f'<th style="{CELL_STYLE}text-align:center;'
            f'{"background:rgba(64,140,255,0.15);" if sel else ""}">'
            f'{MODEL_LABELS[k]}{" ★" if sel else ""}</th>')
    body, caveats = "", []
    for g in preds:
        res = cached_game_models(g["home_id"], g["away_id"], date_str, stamp,
                                 home_pp_id=g.get("home_probable_pitcher_id"),
                                 away_pp_id=g.get("away_probable_pitcher_id"))
        matchup = (f'{html.escape(g["away_name"])} @ '
                   f'{html.escape(g["home_name"])}')
        if "error" in res:
            body += (f'<tr><td style="{CELL_STYLE}">{matchup}</td>'
                     f'<td style="{CELL_STYLE}text-align:center;" colspan="3">'
                     f'<span style="opacity:0.6;">model unavailable</span></td></tr>')
            continue
        cells = ""
        for k in MODEL_KEYS:
            r = res[k]
            sel = k == choice_key
            fell_back = r["served_model"] != r["requested_model"]
            star = "*" if fell_back else ""
            cells += (
                f'<td style="{CELL_STYLE}text-align:center;font-size:1.1em;'
                f'font-weight:{700 if sel else 400};'
                f'{"background:rgba(64,140,255,0.10);" if sel else ""}">'
                f'{r["p_home"]:.1%}{star}</td>')
        body += (f'<tr><td style="{CELL_STYLE}">{matchup}</td>{cells}</tr>')

        # honest caveats for the FOREGROUNDED model
        chosen = res[choice_key]
        gp_h, gp_a = chosen["games_played_home"], chosen["games_played_away"]
        if chosen["served_model"] != chosen["requested_model"]:
            team = g["home_name"] if gp_h <= gp_a else g["away_name"]
            n = min(gp_h, gp_a)
            caveats.append(
                f'⚾ **{matchup}** — You selected {MODEL_LABELS[choice_key]}, but '
                f'{html.escape(team)} has only played {n} games. Showing Elo '
                "until both teams reach 15 games — model predictions aren't "
                "reliable before then.")
        for reason in chosen.get("low_confidence_reasons", []):
            side, _, kind = reason.partition(":")
            team = g["home_name"] if side == "home" else g["away_name"]
            if kind == "starter_not_announced":
                caveats.append(
                    f'ⓘ **{matchup}** — Starter not yet announced for '
                    f'{html.escape(team)} — pitching features use team averages, '
                    "so this prediction is less reliable.")
            elif kind == "starter_low_ip":
                caveats.append(
                    f'ⓘ **{matchup}** — {html.escape(team)}\'s starter has little '
                    "recent workload — pitching features use team averages, so "
                    "this prediction is less reliable.")

    st.markdown(
        '<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr><th style="{CELL_STYLE}text-align:left;">Matchup</th>'
        f'{header_cells}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True)
    st.caption("Every number is the **home team's** win chance. ★ marks the "
               "model you selected; * marks a game served by Elo because a team "
               "hasn't reached 15 games yet (see notes below).")
    st.caption("Honest ceiling: MLB single-game win probabilities realistically "
               "top out in the high-60s percent, so a number near 68% is about "
               "as confident as the model honestly gets — that's the real edge, "
               "not hedging.")
    for c in caveats:
        st.caption(c)


with tab_slate:
    if st.button("Refresh now",
                 help="Pull recent finals, rebuild ratings + today's slate, "
                      "and refresh boxscores now (may take a minute)"):
        auto_refresh.clear()
        cached_predictions.clear()
        cached_scoreline.clear()
        cached_remaining.clear()
        cached_projection.clear()
        with st.spinner("Updating data (finals, ratings, boxscores)…"):
            status = refresh_if_stale(force=True, with_boxscores=True)
    else:
        with st.spinner("Checking for new data…"):
            status = auto_refresh()

    if status["ok"]:
        line = f"Data current as of {status['timestamp']}"
        if status["games_added"]:
            line += f" · {status['games_added']} new final(s) folded in"
        st.caption(f"{line} · results through {status['data_as_of']}")
        # surface non-fatal tail warnings honestly rather than implying all-clear
        for note in (status.get("predictions"), status.get("boxscores")):
            if note and "skip" in str(note).lower():
                st.caption(f"⚠ {note}")
    else:
        st.warning(f"Refresh failed ({status['error']}) — showing data as of "
                   f"{status['data_as_of']}.")

    today = todays_date_str()
    try:
        preds = cached_predictions(today, status["timestamp"])
        offline = False
    except Exception as exc:
        preds, offline = logged_fallback(today), True
        # name the actual error: a code/file problem must not masquerade
        # as a network outage
        st.warning(f"Couldn't fetch today's slate ({type(exc).__name__}: "
                   f"{exc}) — showing predictions logged earlier today, "
                   "if any.")

    st.caption(f"{today} · {len(preds)} regular-season game(s) · "
               "probable pitchers are context only — they are not in the model")

    if not preds:
        if offline:
            st.info("No predictions were logged for today yet — reconnect "
                    "and refresh to fetch the slate.")
        else:
            st.info("No MLB games scheduled today — off-day or All-Star "
                    "break. Check back tomorrow.")
    else:
        st.markdown(slate_table(preds), unsafe_allow_html=True)
        st.caption("Chances hugging 45%–60% is correct behavior — "
                   "single MLB games are near-coinflips; the model's edge over "
                   "a naive baseline shows up in log-loss over many games "
                   "(see src/score.py).")

        # Chunk 9: three-model comparison. The Elo table above is unchanged;
        # this adds logreg + calibrated gradient boosting alongside it.
        if not offline:
            st.divider()
            st.markdown("#### Compare models")
            labels = [MODEL_LABELS[k] for k in MODEL_KEYS]
            picked = st.radio("Foreground model", labels, index=0,
                              horizontal=True,
                              help="Foregrounds one model; all three stay "
                                   "visible below. Default: Calibrated "
                                   "Gradient Boosting.")
            choice_key = MODEL_KEYS[labels.index(picked)]
            try:
                model_comparison_block(preds, today, status["timestamp"],
                                       choice_key)
            except Exception as exc:
                st.info(f"Model comparison unavailable "
                        f"({type(exc).__name__}: {exc}).")
        else:
            st.caption("Model comparison needs the live feature path — "
                       "reconnect and refresh to compare Elo, logistic "
                       "regression, and calibrated gradient boosting.")

def pct(p):
    """Honest compact percentage: never rounds a nonzero chance to 0 or a
    non-certainty to 100."""
    if p == 0:
        return "—"
    if p < 0.001:
        return "<0.1%"
    if 0.999 < p < 1:
        return ">99.9%"
    return f"{p:.1%}"


def team_cell(team_id, name, label=None):
    # fixed-width slot whether or not a letter is present, so the left
    # edge of every row stays aligned
    tag = ('<span style="font-family:monospace;font-size:0.85em;opacity:0.55;'
           f'display:inline-block;width:1.1em;">{label or ""}</span>')
    return ('<div style="display:flex;align-items:center;gap:7px;">'
            f'{tag}<img src="{LOGO_URL.format(team_id=team_id)}" width="20" height="20" alt="">'
            f"<span>{html.escape(name)}</span></div>")


def division_table(rows):
    """rows: (team_id, name, projection dict, label) sorted best-first."""
    body = ""
    for tid, name, r, label in rows:
        proj_l = r["total_games"] - r["mean_wins"]
        body += (
            f'<tr><td style="{CELL_STYLE}">{team_cell(tid, name, label)}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{r["wins"]}–{r["losses"]}</td>'
            f'<td style="{CELL_STYLE}text-align:center;font-weight:600;">'
            f'{r["mean_wins"]:.0f}–{proj_l:.0f}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{r["p10"]:.0f}–{r["p90"]:.0f}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{pct(r["p_division"])}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{pct(r["p_berth"])}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{pct(r["p_pennant"])}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{pct(r["p_title"])}</td></tr>')
    heads = ["Team", "Now", "Proj W–L", "10–90% wins",
             "Div title", "Playoffs", "Pennant", "Title"]
    head = "".join(f'<th style="{CELL_STYLE}text-align:center;">{h}</th>'
                   for h in heads).replace("center", "left", 1)
    return ('<table style="width:100%;border-collapse:collapse;">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def projection_comparison(projs, team_names):
    """Playoff odds side by side across models, teams ordered by the LARGEST
    cross-model disagreement in title odds — that divergence is the point."""
    keys = list(projs)
    all_teams = list(next(iter(projs.values()))["teams"])

    def spread(t, field):
        vals = [projs[k]["teams"][t][field] for k in keys]
        return max(vals) - min(vals)

    ranked = sorted(all_teams, key=lambda t: spread(t, "p_title"), reverse=True)
    head = (f'<th style="{CELL_STYLE}text-align:left;">Team</th>'
            + "".join(f'<th style="{CELL_STYLE}text-align:center;" colspan="1">'
                      f'{SIM_KEY_LABEL[k].split()[0]}<br><span style="opacity:.6;'
                      'font-weight:400;">berth · title</span></th>' for k in keys)
            + f'<th style="{CELL_STYLE}text-align:center;">Δtitle</th>')
    body = ""
    for t in ranked[:10]:
        dt = spread(t, "p_title")
        hot = "background:rgba(255,170,0,0.14);" if dt >= 0.05 else ""
        cells = ""
        for k in keys:
            r = projs[k]["teams"][t]
            cells += (f'<td style="{CELL_STYLE}text-align:center;">'
                      f'{pct(r["p_berth"])} · <b>{pct(r["p_title"])}</b></td>')
        body += (f'<tr style="{hot}"><td style="{CELL_STYLE}">'
                 f'{team_cell(t, team_names.get(t, str(t)))}</td>{cells}'
                 f'<td style="{CELL_STYLE}text-align:center;font-weight:600;">'
                 f'{dt*100:.1f} pp</td></tr>')
    st.markdown(f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)
    st.caption("Rows ordered by how much the models disagree about a team's "
               "title odds (Δtitle). Highlighted rows are the biggest "
               "disagreements — the novel signal from running multiple models.")


with tab_proj:
    st.markdown("#### Season projection")
    fg_label = st.radio(
        "Model", SIM_MODEL_LABELS, index=0, horizontal=True,
        help="Foreground model driving the projection. Calibrated Gradient "
             "Boosting (default) and Logistic Regression build as-of-today team "
             "features once (~a couple minutes, then cached); Elo is instant.")
    fg_key = SIM_MODEL_KEY[fg_label]
    compare_labels = st.multiselect(
        "Also run and compare (optional)",
        [l for l in SIM_MODEL_LABELS if l != fg_label],
        help="Opt-in: run additional models and compare playoff odds side by "
             "side. Each extra model adds runtime.")
    compare_keys = [SIM_MODEL_KEY[l] for l in compare_labels]
    all_keys = tuple(dict.fromkeys([fg_key] + compare_keys))

    # Button-gated so a page load never blocks on the ~3-min ML feature build;
    # the sim runs only when explicitly requested, and its result (cached)
    # re-displays across tab switches / reruns.
    if st.button("Run projection", type="primary",
                 help="Run the season sim under the selected model(s). The ML "
                      "models build as-of-today features once (~a couple minutes, "
                      "then cached); Elo is instant."):
        st.session_state["proj_keys"] = all_keys
        st.session_state["proj_fg"] = fg_key

    ran_keys = st.session_state.get("proj_keys")
    ran_fg = st.session_state.get("proj_fg")
    proj, projs, team_names, remaining, ratings_stamp = None, {}, {}, [], None
    if ran_keys is None:
        st.info("Choose a model and click **Run projection**. Calibrated Gradient "
                "Boosting (default) and Logistic Regression build as-of-today "
                "features once (~a couple minutes, then cached); Elo is instant.")
    else:
        try:
            _ratings, team_names, ratings_stamp = load_ratings()
            remaining = cached_remaining(ratings_stamp)
            sched_key = hash(tuple(remaining["game_id"]))
            projs = cached_projections(ran_keys, ratings_stamp, sched_key,
                                       N_SIMS, SIM_SEED)
            proj = projs.get(ran_fg)
        except Exception as exc:
            st.warning(f"Season projection unavailable ({type(exc).__name__}: "
                       f"{exc}) — it needs the network to fetch the remaining "
                       "schedule.")
            proj, projs = None, {}
        if proj is not None and (ran_keys != all_keys or ran_fg != fg_key):
            st.caption(f"⚠ Selection changed — showing the previous run "
                       f"(**{SIM_KEY_LABEL[ran_fg]}**). Click **Run projection** "
                       "to update to your current selection.")

    if proj and len(ran_keys) > 1:
        st.markdown("##### Model comparison — playoff odds")
        projection_comparison(projs, team_names)
        st.divider()

    if proj:
        st.caption(f"Detail below is the foregrounded "
                   f"**{SIM_KEY_LABEL[ran_fg]}** projection. The engine is "
                   "unchanged — the model is swapped at construction only.")

    if proj:
        structure = team_structure()
        teams = proj["teams"]
        st.caption(f'{proj["n_sims"]:,} simulated seasons · seed {proj["seed"]} · '
                   f"ratings as of {ratings_stamp} · "
                   f"{len(remaining)} games left league-wide · ratings frozen "
                   "during simulation (simulated results never update them)")

        short = {t: (161, "one postponed game currently unscheduled")
                 for t, r in teams.items() if r["total_games"] != 162}
        if short:
            names_ = ", ".join(f'{team_names[t]} ({teams[t]["total_games"]})'
                               for t in short)
            st.caption(f"⚠ Not all teams project 162 games: {names_} — "
                       "shown as-is, not forced to 162.")

        h2h_played = cached_h2h(ratings_stamp)
        labels = assign_playoff_labels(
            {t: r["mean_wins"] for t, r in teams.items()}, structure,
            h2h_wins=lambda a, b: h2h_played.get((a, b), 0))

        st.markdown("#### Projected standings")
        for lg in sorted({v[0] for v in structure.values()}):
            st.markdown(f"**{lg}**")
            for dv in sorted({v[1] for v in structure.values() if v[0] == lg}):
                members = [t for t in teams if structure[t][1] == dv]
                rows = sorted(((t, team_names.get(t, str(t)), teams[t],
                                labels.get(t))
                               for t in members), key=lambda x: -x[2]["mean_wins"])
                st.caption(dv)
                st.markdown(division_table(rows), unsafe_allow_html=True)
        st.caption(f"Projected finish — z: best projected record in league · "
                   "y: projected division winner · x: projected wild card "
                   f"berth. Based on {N_SIMS:,} simulations, not clinched "
                   "status — nothing is clinched or eliminated in July.")

        st.markdown("#### Most likely bracket")
        st.caption("Per slot: the modal matchup across simulations and how "
                   "often it occurs — no slot is certain, so every matchup "
                   "carries its probability.")
        slot_labels = [("WCS 3v6", "Wild Card (3 vs 6)"),
                       ("WCS 4v5", "Wild Card (4 vs 5)"),
                       ("DS (1 side)", "Division Series, 1-seed side"),
                       ("DS (2 side)", "Division Series, 2-seed side"),
                       ("LCS", "League Championship")]
        for lg in sorted({v[0] for v in structure.values()}):
            lines = [f"**{lg}**"]
            for slot, label in slot_labels:
                tally = proj["series"].get(f"{lg} {slot}", {})
                if not tally:
                    continue
                (hi, lo), n = max(tally.items(), key=lambda kv: kv[1])
                lines.append(f"- {label}: {team_names.get(hi, hi)} vs "
                             f"{team_names.get(lo, lo)} — {pct(n / proj['n_sims'])} of sims")
            st.markdown("\n".join(lines))
        ws = proj["series"].get("WS final", {})
        if ws:
            (hi, lo), n = max(ws.items(), key=lambda kv: kv[1])
            st.markdown(f"**World Series** — most likely: {team_names.get(hi, hi)} vs "
                        f"{team_names.get(lo, lo)} ({pct(n / proj['n_sims'])} of sims)")

        st.markdown("#### Championship odds")
        top10 = sorted(teams, key=lambda t: -teams[t]["p_title"])[:10]
        body = "".join(
            f'<tr><td style="{CELL_STYLE}">{team_cell(t, team_names.get(t, str(t)))}</td>'
            f'<td style="{CELL_STYLE}text-align:center;font-weight:600;">'
            f'{pct(teams[t]["p_title"])}</td>'
            f'<td style="{CELL_STYLE}text-align:center;">{pct(teams[t]["p_pennant"])}</td>'
            f'</tr>' for t in top10)
        st.markdown(
            '<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr><th style="{CELL_STYLE}text-align:left;">Team</th>'
            f'<th style="{CELL_STYLE}text-align:center;">Wins World Series</th>'
            f'<th style="{CELL_STYLE}text-align:center;">Wins pennant</th>'
            f"</tr></thead><tbody>{body}</tbody></table>",
            unsafe_allow_html=True)
        st.caption("Single MLB playoff series are close to coinflips, so even "
                   "the best team's title odds stay modest — a favorite around "
                   "15–20% is expected, not a hedge.")

with tab_h2h:
    try:
        h2h_structure = team_structure()
        h2h_ratings, h2h_names, h2h_stamp = load_ratings()
        if not h2h_ratings:
            raise RuntimeError("no ratings snapshot yet — refresh first")
    except Exception as exc:
        st.warning(f"Head to head unavailable ({type(exc).__name__}: {exc}).")
        h2h_structure = None

    if h2h_structure:
        mk_label = st.radio(
            "Model", SIM_MODEL_LABELS, index=0, horizontal=True, key="h2h_model",
            help="Default: Calibrated Gradient Boosting. The ML models build the "
                 "two teams' as-of-today features (a few seconds); Elo is instant.")
        mk_key = SIM_MODEL_KEY[mk_label]
        try:
            model = matchup_model(mk_key, h2h_stamp)
        except Exception as exc:
            st.warning(f"{mk_label} unavailable ({type(exc).__name__}: {exc}) — "
                       "falling back to Elo.")
            model, mk_label, mk_key = EloProbabilityModel(h2h_ratings), "Elo", "elo"
        team_ids = sorted(h2h_structure, key=lambda t: h2h_names.get(t, ""))

        col_a, col_b = st.columns(2)
        team_a = col_a.selectbox("Team A", team_ids, index=0,
                                 format_func=lambda t: h2h_names.get(t, str(t)))
        team_b = col_b.selectbox("Team B", team_ids, index=1,
                                 format_func=lambda t: h2h_names.get(t, str(t)))

        if team_a == team_b:
            st.error("Pick two different teams — a team can't play itself.")
        else:
            name_a = h2h_names.get(team_a, str(team_a))
            name_b = h2h_names.get(team_b, str(team_b))
            fmt = st.selectbox("Format", list(FORMATS),
                               index=list(FORMATS).index("World Series (best of 7)"))
            single = FORMATS[fmt] is None

            host_options = [team_a, team_b] + (["neutral"] if single else [])
            host_choice = st.radio(
                "Home team" if single else "Home field advantage (higher seed)",
                host_options, horizontal=True,
                format_func=lambda v: ("Neutral site" if v == "neutral"
                                       else h2h_names.get(v, str(v))))
            neutral = host_choice == "neutral"
            host = team_a if neutral else host_choice

            notice = check_matchup(team_a, team_b, fmt, h2h_structure, h2h_names)
            if notice:
                st.info(notice)

            p_a = win_probability(model, team_a, team_b, fmt,
                                  host=host, neutral=neutral)

            # point-estimate headline (the foregrounded model)
            _fav_name = name_a if p_a >= 0.5 else name_b
            st.markdown(f"### {mk_label}: **{max(p_a, 1 - p_a):.1%}** "
                        f"{html.escape(_fav_name)}")

            def h2h_side(tid, name, p, favored):
                return ('<div style="display:flex;align-items:center;gap:10px;">'
                        f'<img src="{LOGO_URL.format(team_id=tid)}" width="34" '
                        'height="34" alt="">'
                        f'<div><div style="font-weight:{700 if favored else 400};">'
                        f"{html.escape(name)}</div>"
                        f'<div style="font-size:1.5em;font-weight:'
                        f'{700 if favored else 400};">{p:.1%}</div></div></div>')

            st.markdown(
                '<div style="display:flex;gap:36px;align-items:center;'
                'flex-wrap:wrap;margin:10px 0;">'
                + h2h_side(team_a, name_a, p_a, p_a >= 0.5)
                + '<div style="opacity:0.6;">vs</div>'
                + h2h_side(team_b, name_b, 1 - p_a, p_a < 0.5)
                + "</div>", unsafe_allow_html=True)
            _hf = (" · neutral site" if neutral
                   else f" · home field: {h2h_names.get(host, host)}")
            if mk_key == "elo":
                st.caption(f'Chance to win the {"game" if single else "series"}. '
                           f"Elo: {name_a} {h2h_ratings[team_a]:.0f} · {name_b} "
                           f"{h2h_ratings[team_b]:.0f} · gap "
                           f"{abs(h2h_ratings[team_a] - h2h_ratings[team_b]):.0f} "
                           f"points{_hf}")
            else:
                st.caption(f'{mk_label} — chance to win the '
                           f'{"game" if single else "series"}{_hf}. Features are '
                           "each team's as-of-today form, run differential, rest, "
                           "and rotation-average starter (not a single announced "
                           f"pitcher).")

            # analytic "More info" — exact series enumeration (no sampling)
            with st.expander("More info — exact series breakdown"):
                pattern = FORMATS[fmt]
                if pattern is None:
                    st.write(f"**Single game** — P({name_a} wins) = **{p_a:.1%}** "
                             "(exact, one game).")
                else:
                    higher, lower = ((team_a, team_b) if host == team_a
                                     else (team_b, team_a))
                    pg = series_pgame(model, higher, lower, pattern)
                    swp = series_win_prob(pg)
                    wl = series_length_dist(pg)
                    ref = h2h_names.get(higher, str(higher))
                    st.write(f"Reference side (host / higher seed): **{ref}**")
                    st.write("Per-game win probability by hosting pattern: "
                             + ", ".join(f"G{i} {p:.1%}" for i, p in
                                         enumerate(pg, 1)))
                    st.write(f"Enumerated series win probability (exact DP): "
                             f"**{swp:.1%}**")
                    st.write(f"How **{ref}** wins the series — exact length "
                             "distribution:")
                    st.markdown("".join(
                        f"- in **{g}** games: {wl[g]:.1%}\n" for g in sorted(wl)))
                    st.caption("Analytic enumeration (the same DP the bracket "
                               "uses), not a Monte Carlo sample. The lengths sum "
                               "to the series win probability.")

            if st.checkbox("Compare all three models' point estimate",
                           key="h2h_all3"):
                for lbl in SIM_MODEL_LABELS:
                    try:
                        m3 = matchup_model(SIM_MODEL_KEY[lbl], h2h_stamp)
                        p3 = win_probability(m3, team_a, team_b, fmt, host=host,
                                             neutral=neutral)
                        mark = " ← selected" if lbl == mk_label else ""
                        st.write(f"**{lbl}**: {max(p3,1-p3):.1%} "
                                 f"{name_a if p3>=0.5 else name_b}{mark}")
                    except Exception as exc:
                        st.write(f"{lbl}: unavailable ({type(exc).__name__})")

            season_h2h = cached_h2h(h2h_stamp)
            wa = season_h2h.get((team_a, team_b), 0)
            wb = season_h2h.get((team_b, team_a), 0)
            if wa or wb:
                st.caption(f"Season series so far: {name_a} {wa}–{wb} {name_b}")

            if not single:
                try:
                    rem = cached_remaining(h2h_stamp)
                    proj = cached_projection(h2h_stamp,
                                             hash(tuple(rem["game_id"])),
                                             N_SIMS, SIM_SEED)
                    n_occ = season_sim_occurrences(proj["series"],
                                                   team_a, team_b, fmt)
                    if n_occ is not None:
                        st.caption(f"This matchup occurred in "
                                   f"{n_occ / proj['n_sims']:.1%} of "
                                   f"{proj['n_sims']:,} simulated seasons.")
                except Exception:
                    pass  # projection unavailable (offline) — skip the context line

            unit = "game" if single else "series"
            c1, c2 = st.columns(2)
            if c1.button(f"Simulate the {unit}"):
                won = simulate_outcomes(p_a, 1)
                winner = name_a if won else name_b
                st.markdown(f"**{winner}** wins this simulated {unit}.")
                fav_p = max(p_a, 1 - p_a)
                st.caption(f"One simulated {unit}, not a prediction — a "
                           f"{fav_p:.0%} favorite still loses {1 - fav_p:.0%} "
                           "of the time. Click again to re-roll.")
            if c2.button("Simulate 1,000 times"):
                won = simulate_outcomes(p_a, 1000)
                st.markdown(f"**{name_a} won {won:,} of 1,000** simulated "
                            f"{unit}s; {name_b} won {1000 - won:,}.")

with tab_methods:
    st.markdown(f"""
### The model in one paragraph

Every team carries an **Elo rating** — a single number that goes up when the
team wins and down when it loses, with bigger moves for surprising results
than for expected ones. Before each game, the gap between the two teams'
ratings (plus a small home-field bump) converts into the home team's chance
to win. Ratings are rebuilt by replaying five-plus seasons of real results
in chronological order, so today's numbers always reflect everything that
has happened so far.

### How to read the slate

- Each matchup reads **away team @ home team**, and the venue line is the
  **home team's stadium**.
- The team Elo favors to win is shown in **bold**.
- Each team's **probable starting pitcher** appears directly under its name —
  context only, never part of the calculation.
- **Doubleheaders** are labeled Game 1 and Game 2 and predicted separately.
- The one number shown is the **home team's chance to win** per Elo. Above
  50% means Elo favors the home side; below 50% means it favors the visitor.

### Where the data comes from

All data comes from the public **MLB Stats API** (`statsapi.mlb.com`) — game
results for the five most recent completed seasons plus the current season,
today's schedule, probable pitchers, venues, and the team logos. Only
**regular-season games that went Final** count; spring training, the
All-Star Game, and the postseason are excluded. The app **refreshes itself
when opened**: recent finals are folded into the history and ratings are
rebuilt, at most once every {STALE_AFTER_HOURS} hours (or on demand via
**Refresh now**). If the network is down, it keeps showing the most recent
good data instead.

### The Elo math

- **Win chance.** Home team's chance = `1 / (1 + 10^(−(elo_home + 25 − elo_away) / 400))`.
  The **25-point home bump** gives an even matchup ≈ 53.6% for the home
  side, matching MLB's real ~54% home-win rate.
- **Rating update.** After each game the winner takes points from the loser:
  `Δ = K × (actual − expected)`, with **K = 4**. That is deliberately tiny —
  a 162-game season means any single game says very little, and textbook
  K values (20–32) would make ratings thrash.
- **Season boundaries.** Between seasons every rating regresses **one-third
  of the way back to the 1500 average**, so last year's juggernaut starts
  the new season strong but not untouchable.
- **What's *not* in the model.** Probable pitchers are shown for context
  only and never enter any calculation. Margin of victory is plumbed in but
  switched off. Injuries, travel, weather, and lineups are not modeled.

### The baseline, and how we keep score

Behind the scenes every Elo chance is compared against the simplest possible
strategy: **always give the home team a 54.0% chance** (the league-wide
home-win rate). That baseline isn't shown as a column — it never changes —
but it is the yardstick the model has to beat. Every prediction shown here
is logged, and after games finish both are scored on **accuracy** (did the
favored side win?) and **log-loss** (were the probabilities honest?).
A **walk-forward backtest** does the same over past seasons — each game is
predicted using only the games played before it, so the model never peeks at
the future. Elo beats the baseline on both measures in every tested season,
but expect single-game chances to stay in the 45%–60% band: baseball games
are close to coinflips, and the model's edge only shows up over many games.

### Three models, side by side

The **Compare models** panel on the slate shows three predictors for every
game. **Elo** is the rating model above. **Logistic Regression** and
**Calibrated Gradient Boosting** add team form, run differential, rest, and the
listed starter's recent strikeout and contact-quality rates — features computed
by the *exact same code* that built the training data, so what the models see
live is what they were trained on. The selector foregrounds one model; all
three stay visible, because the point of shipping three is to watch them agree
and disagree over a real season.

Two honesty rules are built in:

- **Cold start.** The feature models need about 15 games of a season before
  team form and run-differential mean anything. Before that, the panel shows
  **Elo** in place of the selected model and says so — it never dresses Elo up
  under another model's name.
- **Calibration and the ceiling.** The gradient-boosting model is
  **calibrated** (Platt scaling on a held-out season) so a stated 60% means
  roughly 60% in reality. Even so, honest single-game MLB win probabilities top
  out in the **high-60s percent** — a number near 68% is about as confident as
  the model gets, not a hedge. When a starter isn't announced yet, pitching
  features fall back to team averages and the game is flagged **less reliable**.
""")
