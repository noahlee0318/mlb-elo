"""Head-to-head matchup orchestration: validate a pairing, resolve the
format's hosting pattern, and get probabilities through the model seam.

Glue only. Probabilities come from ProbabilityModel.matchup_prob; series
math is playoffs.py's existing analytic DP (never reimplemented here);
no elo.py import — this module survives a model replacement untouched.
"""

import numpy as np

from src.playoffs import B7_PATTERN, DS_PATTERN, WC_PATTERN, _series_prob_cached

# format name -> hosting pattern (None = single game)
FORMATS = {
    "Single game": None,
    "Wild Card Series (best of 3)": WC_PATTERN,
    "Division Series (best of 5)": DS_PATTERN,
    "League Championship Series (best of 7)": B7_PATTERN,
    "World Series (best of 7)": B7_PATTERN,
}

# proj["series"] slot substrings per format, for the season-sim occurrence line
_SLOT_MARKERS = {
    "Wild Card Series (best of 3)": " WCS ",
    "Division Series (best of 5)": " DS ",
    "League Championship Series (best of 7)": " LCS",
    "World Series (best of 7)": "WS final",
}


def check_matchup(team_a, team_b, format_name, structure, names=None):
    """None if the pairing can occur in this round, else a factual notice.

    League-only rule: WCS/DS/LCS are played within one league, so a
    cross-league pair is impossible there; the World Series is played
    across leagues, so a same-league pair is impossible there. Single
    games are always possible (interleague play) and division rivals are
    fine in every round.
    """
    names = names or {}
    lg_a, lg_b = structure[team_a][0], structure[team_b][0]
    name_a = names.get(team_a, str(team_a))
    name_b = names.get(team_b, str(team_b))
    if format_name == "Single game":
        return None
    if format_name == "World Series (best of 7)":
        if lg_a == lg_b:
            return (f"{name_a} and {name_b} are both {lg_a} teams, so they "
                    "can't meet in the World Series. Simulating anyway.")
        return None
    if lg_a != lg_b:
        round_name = format_name.split(" (")[0]
        return (f"{name_a} ({lg_a}) and {name_b} ({lg_b}) can't meet in the "
                f"{round_name} — it is played within a single league. "
                "Simulating anyway.")
    return None


def win_probability(model, team_a, team_b, format_name, host, neutral=False):
    """P(team_a wins the chosen format). host = the team with home field
    (single game) or the higher seed (series). neutral applies to single
    games only."""
    pattern = FORMATS[format_name]
    if pattern is None:
        if neutral:
            return model.matchup_prob(team_a, team_b, context={"neutral": True})
        if host == team_a:
            return model.matchup_prob(team_a, team_b)
        return 1.0 - model.matchup_prob(team_b, team_a)
    higher, lower = (team_a, team_b) if host == team_a else (team_b, team_a)
    p_higher = _series_prob_cached(model, higher, lower, pattern, {})
    return p_higher if host == team_a else 1.0 - p_higher


def simulate_outcomes(p_a, n=1, rng=None):
    """Draw n independent outcomes; returns how many team A won. One draw
    per game or per series — a series was already resolved analytically
    into p_a, so a single draw decides it."""
    rng = rng if rng is not None else np.random.default_rng()
    return int((rng.random(n) < p_a).sum())


def season_sim_occurrences(series_tally, team_a, team_b, format_name):
    """How many simulated seasons produced this exact matchup in this
    round, from the season projection's recorded bracket slots. Purely a
    lookup — nothing here re-runs or modifies the season sim."""
    marker = _SLOT_MARKERS.get(format_name)
    if marker is None:
        return None  # single games aren't bracket slots
    total = 0
    for slot, tally in series_tally.items():
        if marker in slot:
            total += tally.get((team_a, team_b), 0) + tally.get((team_b, team_a), 0)
    return total
