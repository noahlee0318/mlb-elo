"""Postseason seeding, bracket, and series resolution.

Format verified against MLB.com's playoff FAQ (checked 2026-07-18):
12 teams, 6 per league. Seeds 1-2 (best two division winners) get
first-round byes; seed 3 is the remaining division winner; seeds 4-6 are
wild cards by record. Wild Card Series is best-of-3, every game at the
higher seed (3v6, 4v5). Division Series is best-of-5, 2-2-1, fixed
bracket with NO reseeding: 1 plays the 4/5 winner, 2 plays the 3/6
winner. LCS and World Series are best-of-7, 2-3-2. World Series home
field goes to the better regular-season record.

This module knows nothing about how probabilities are produced — it asks
the supplied model for matchup probabilities and nothing more. Each
series is resolved analytically: a DP over game sequences with the
correct home-game pattern yields the exact series-win probability, and
one draw against that number decides the series. Statistically identical
to playing out each game, minus per-game detail that nothing needs.
"""

# True = higher seed hosts that game
WC_PATTERN = (True, True, True)                              # best-of-3, all hosted
DS_PATTERN = (True, True, False, False, True)                # best-of-5, 2-2-1
B7_PATTERN = (True, True, False, False, False, True, True)   # best-of-7, 2-3-2


def series_win_prob(p_by_game):
    """P(the reference side wins the series), given that side's win
    probability for each possibly-played game. Series length is
    len(p_by_game); first to more than half wins. Exact DP, no sampling."""
    need = len(p_by_game) // 2 + 1
    states = {(0, 0): 1.0}
    won = 0.0
    for p in p_by_game:
        nxt = {}
        for (w, l), mass in states.items():
            if mass == 0.0:
                continue
            if w + 1 == need:
                won += mass * p
            else:
                nxt[(w + 1, l)] = nxt.get((w + 1, l), 0.0) + mass * p
            if l + 1 < need:
                nxt[(w, l + 1)] = nxt.get((w, l + 1), 0.0) + mass * (1.0 - p)
        states = nxt
    return won


def _series_prob_cached(model, higher, lower, pattern, cache):
    """P(higher seed wins the series) with the given hosting pattern,
    memoized per (higher, lower, pattern) since matchups repeat heavily
    across simulations."""
    key = (higher, lower, pattern)
    if key not in cache:
        p_games = [model.matchup_prob(higher, lower) if hosts
                   else 1.0 - model.matchup_prob(lower, higher)
                   for hosts in pattern]
        cache[key] = series_win_prob(p_games)
    return cache[key]


def run_bracket(seeds_by_league, wins_of, model, rng, cache, record=None):
    """One postseason. seeds_by_league: {league: [seed1..seed6 team ids]}.
    wins_of: callable team -> regular-season win total (for WS home field).
    Returns (champion, {league: pennant winner}). If record is a dict, each
    series is tallied into it as record[slot][(higher, lower)] += 1."""

    def play(league, slot, higher, lower, pattern):
        if record is not None:
            tally = record.setdefault(f"{league} {slot}", {})
            key = (higher, lower)
            tally[key] = tally.get(key, 0) + 1
        p = _series_prob_cached(model, higher, lower, pattern, cache)
        return higher if rng.random() < p else lower

    pennants = {}
    for league, seeds in seeds_by_league.items():
        s1, s2, s3, s4, s5, s6 = seeds
        seed_no = {t: i for i, t in enumerate(seeds, start=1)}
        w36 = play(league, "WCS 3v6", s3, s6, WC_PATTERN)
        w45 = play(league, "WCS 4v5", s4, s5, WC_PATTERN)
        # fixed bracket, no reseeding
        d1 = play(league, "DS (1 side)", s1, w45, DS_PATTERN)
        d2 = play(league, "DS (2 side)", s2, w36, DS_PATTERN)
        hi, lo = (d1, d2) if seed_no[d1] < seed_no[d2] else (d2, d1)
        pennants[league] = play(league, "LCS", hi, lo, B7_PATTERN)

    a, b = list(pennants.values())
    # WS home field: better regular-season record; exact tie -> random draw
    # (the official deeper tiebreaks are rare and effectively a coin here)
    if wins_of(a) != wins_of(b):
        hi, lo = (a, b) if wins_of(a) > wins_of(b) else (b, a)
    else:
        hi, lo = (a, b) if rng.random() < 0.5 else (b, a)
    champion = play("WS", "final", hi, lo, B7_PATTERN)
    return champion, pennants
