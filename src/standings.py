"""Current-season records, head-to-head results, and tiebreak ordering.

The loaders read the unified Elo game window (load_elo_games, from
games_full.csv) — the retired games.csv is no longer read. rank_league() is
pure so the simulation engine can call it once per simulated season with
simulated totals mixed in.
"""

import random

from src.games_data import load_elo_games


def current_records(games=None, season=None):
    """{team_id: (wins, losses)} for one season's decided finals."""
    df = load_elo_games() if games is None else games
    season = season or int(df["season"].max())
    df = df[df["season"] == season]
    records = {}
    for g in df.itertuples(index=False):
        h, a = int(g.home_id), int(g.away_id)
        records.setdefault(h, [0, 0])
        records.setdefault(a, [0, 0])
        if g.home_score > g.away_score:
            records[h][0] += 1
            records[a][1] += 1
        else:
            records[a][0] += 1
            records[h][1] += 1
    return {t: (w, l) for t, (w, l) in records.items()}


def played_h2h(games=None, season=None):
    """{(winner_id, loser_id): count} of decided head-to-head results —
    the played half of every tiebreak comparison."""
    df = load_elo_games() if games is None else games
    season = season or int(df["season"].max())
    df = df[df["season"] == season]
    h2h = {}
    for g in df.itertuples(index=False):
        h, a = int(g.home_id), int(g.away_id)
        w, l = (h, a) if g.home_score > g.away_score else (a, h)
        h2h[(w, l)] = h2h.get((w, l), 0) + 1
    return h2h


def assign_playoff_labels(wins_by_team, structure, h2h_wins=None,
                          intradiv_pct=None, rng=None):
    """PROJECTED playoff-field letters from projected win totals:
    z = best projected record in the league, y = other projected division
    winners, x = projected wild cards. Exactly 6 labels per league.

    These are projections over simulated seasons, NEVER clinch markers —
    no team has clinched (or been eliminated from) anything, so there is
    deliberately no 'e' and callers must present the letters as projected
    finish. Ordering and tiebreaks reuse rank_league, the same logic the
    season sim uses — no second tiebreaker.
    """
    h2h_wins = h2h_wins or (lambda a, b: 0)
    intradiv_pct = intradiv_pct or (lambda t: 0.0)
    rng = rng or random.Random(0)

    leagues = {}
    for t, info in structure.items():
        leagues.setdefault(info[0], []).append(t)

    labels = {}
    for members in leagues.values():
        order = rank_league(members, lambda t: wins_by_team[t],
                            h2h_wins, intradiv_pct, rng)
        seen_divisions = set()
        leaders = []
        for t in order:
            dv = structure[t][1]
            if dv not in seen_divisions:
                seen_divisions.add(dv)
                leaders.append(t)
        labels[leaders[0]] = "z"
        for t in leaders[1:3]:
            labels[t] = "y"
        wild = [t for t in order if t not in leaders[:3]][:3]
        for t in wild:
            labels[t] = "x"
    return labels


def rank_league(teams, wins, h2h_wins, intradiv_pct, rng):
    """Total order (best first) for one league's teams in one simulated
    season.

    Ties on wins break: head-to-head record among the tied group, then
    intradivision winning percentage, then a seeded coin flip.

    Shortcut, on purpose: MLB's official rules continue past these levels
    (intraleague record, then second-half intraleague splits). Those
    levels come up rarely enough that replacing them with a random draw
    does not meaningfully move aggregate probabilities — noted here so
    the shortcut is explicit rather than silent.

    teams: iterable of team ids
    wins: callable t -> season win total
    h2h_wins: callable (a, b) -> a's wins against b (played + simulated)
    intradiv_pct: callable t -> intradivision winning pct (played + simulated)
    rng: numpy Generator for the final-level draw
    """
    by_wins = {}
    for t in teams:
        by_wins.setdefault(wins(t), []).append(t)

    order = []
    for w in sorted(by_wins, reverse=True):
        group = by_wins[w]
        if len(group) == 1:
            order.extend(group)
            continue
        # mini-league head-to-head within the tied group
        def key(t):
            h2h = sum(h2h_wins(t, o) for o in group if o != t)
            return (-h2h, -intradiv_pct(t), rng.random())
        order.extend(sorted(group, key=key))
    return order
