"""Monte Carlo season engine: pure computation, no file access, no
network, and no knowledge of any prediction model.

It consumes a ProbabilityModel (see src/prob_model.py), the remaining
schedule, current records, and the league structure; it returns
aggregate projections. Probabilities are computed ONCE before the loop
and team strength is held constant throughout: a simulated result
carries no information (it was generated from those same probabilities),
so feeding it back into the model would be circular and would inflate
the tails of every distribution.

The game grid is fully vectorized: one (n_sims x n_games) uniform draw
against the probability vector decides every simulated game at once, and
per-team win tallies are matrix products. Only seeding and the bracket
run per-simulation in Python.
"""

import numpy as np

from src.playoffs import run_bracket
from src.standings import rank_league


def simulate_season(remaining, records, structure, model,
                    n_sims=10000, seed=0, played_h2h=None):
    """Project the rest of the season n_sims times.

    remaining: DataFrame with home_id, away_id (one row per unplayed game)
    records:   {team_id: (wins, losses)} — current decided results
    structure: {team_id: (league, division, ...)} — extra fields ignored
    model:     ProbabilityModel; schedule_probs may return (G,) or
               (n_sims, G), both broadcast against the same random grid
    played_h2h: {(winner_id, loser_id): count} for tiebreaks (optional)

    Returns a dict of aggregates (see the bottom of this function).
    """
    played_h2h = played_h2h or {}
    team_ids = sorted(structure)
    t_index = {t: i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)
    n_games = len(remaining)

    # --- the whole simulated regular season in one vectorized pass ---
    p = np.asarray(model.schedule_probs(remaining, n_sims=n_sims), dtype=float)
    rng = np.random.default_rng(seed)
    home_win = rng.random((n_sims, n_games)) < p  # broadcasts (G,) or (n_sims, G)

    home_idx = np.fromiter((t_index[int(t)] for t in remaining["home_id"]),
                           dtype=np.int64, count=n_games)
    away_idx = np.fromiter((t_index[int(t)] for t in remaining["away_id"]),
                           dtype=np.int64, count=n_games)
    home_mask = np.zeros((n_games, n_teams), dtype=np.float32)
    away_mask = np.zeros((n_games, n_teams), dtype=np.float32)
    home_mask[np.arange(n_games), home_idx] = 1.0
    away_mask[np.arange(n_games), away_idx] = 1.0

    hw = home_win.astype(np.float32)
    sim_wins = hw @ home_mask + (1.0 - hw) @ away_mask      # (n_sims, T)
    sim_losses = hw @ away_mask + (1.0 - hw) @ home_mask

    cur_w = np.array([records.get(t, (0, 0))[0] for t in team_ids])
    cur_l = np.array([records.get(t, (0, 0))[1] for t in team_ids])
    total_wins = (cur_w + sim_wins).astype(np.int16)
    total_losses = (cur_l + sim_losses).astype(np.int16)

    # --- precomputed lookups for tiebreaks ---
    leagues = {}
    division_of = {}
    for t in team_ids:
        lg, dv = structure[t][0], structure[t][1]
        leagues.setdefault(lg, []).append(t)
        division_of[t] = dv

    pair_home = {}          # (home, away) -> array of game indices
    intradiv_home = {t: [] for t in team_ids}
    intradiv_away = {t: [] for t in team_ids}
    for g, (h, a) in enumerate(zip(remaining["home_id"], remaining["away_id"])):
        h, a = int(h), int(a)
        pair_home.setdefault((h, a), []).append(g)
        if division_of[h] == division_of[a]:
            intradiv_home[h].append(g)
            intradiv_away[a].append(g)
    pair_home = {k: np.array(v) for k, v in pair_home.items()}
    intradiv_home = {t: np.array(v, dtype=np.int64) for t, v in intradiv_home.items()}
    intradiv_away = {t: np.array(v, dtype=np.int64) for t, v in intradiv_away.items()}

    played_intradiv = {}
    for t in team_ids:
        w = sum(c for (a, b), c in played_h2h.items()
                if a == t and division_of.get(b) == division_of[t])
        l = sum(c for (a, b), c in played_h2h.items()
                if b == t and division_of.get(a) == division_of[t])
        played_intradiv[t] = (w, l)

    # --- per-simulation seeding and postseason ---
    div_titles = np.zeros(n_teams)
    berths = np.zeros(n_teams)
    pennant_ct = np.zeros(n_teams)
    title_ct = np.zeros(n_teams)
    seed_tally = {lg: [dict() for _ in range(6)] for lg in leagues}
    series_tally = {}
    series_cache = {}

    for s in range(n_sims):
        row = home_win[s]
        wins_by_team = dict(zip(team_ids, total_wins[s].tolist()))

        def h2h_wins(a, b):
            won = played_h2h.get((a, b), 0)
            idx = pair_home.get((a, b))
            if idx is not None:
                won += int(row[idx].sum())
            idx = pair_home.get((b, a))
            if idx is not None:
                won += int((~row[idx]).sum())
            return won

        def intradiv_pct(t):
            w, l = played_intradiv[t]
            w += int(row[intradiv_home[t]].sum()) + int((~row[intradiv_away[t]]).sum())
            l += int((~row[intradiv_home[t]]).sum()) + int(row[intradiv_away[t]].sum())
            return w / (w + l) if w + l else 0.0

        seeds_by_league = {}
        for lg, members in leagues.items():
            order = rank_league(members, wins_by_team.__getitem__,
                                h2h_wins, intradiv_pct, rng)
            winners_seen = set()
            div_winners = []
            for t in order:
                if division_of[t] not in winners_seen:
                    winners_seen.add(division_of[t])
                    div_winners.append(t)
            wild_cards = [t for t in order if t not in div_winners][:3]
            seeds = div_winners + wild_cards
            seeds_by_league[lg] = seeds
            div_titles[t_index[seeds[0]]] += 1
            div_titles[t_index[seeds[1]]] += 1
            div_titles[t_index[seeds[2]]] += 1
            for k, t in enumerate(seeds):
                berths[t_index[t]] += 1
                tally = seed_tally[lg][k]
                tally[t] = tally.get(t, 0) + 1

        champion, pennants = run_bracket(
            seeds_by_league, wins_by_team.__getitem__, model, rng,
            series_cache, record=series_tally)
        title_ct[t_index[champion]] += 1
        for t in pennants.values():
            pennant_ct[t_index[t]] += 1

    remaining_ct = np.bincount(home_idx, minlength=n_teams) \
        + np.bincount(away_idx, minlength=n_teams)
    p10, p90 = np.percentile(total_wins, [10, 90], axis=0)
    teams = {}
    for t, i in t_index.items():
        teams[t] = {
            "wins": int(cur_w[i]), "losses": int(cur_l[i]),
            "remaining": int(remaining_ct[i]),
            "total_games": int(cur_w[i] + cur_l[i] + remaining_ct[i]),
            "mean_wins": float(total_wins[:, i].mean()),
            "p10": float(p10[i]), "p90": float(p90[i]),
            "p_division": div_titles[i] / n_sims,
            "p_berth": berths[i] / n_sims,
            "p_pennant": pennant_ct[i] / n_sims,
            "p_title": title_ct[i] / n_sims,
        }
    return {
        "n_sims": n_sims, "seed": seed,
        "team_ids": team_ids,
        "teams": teams,
        "seeds": seed_tally,          # {league: [ {team: count} x 6 ]}
        "series": series_tally,       # {slot: {(higher, lower): count}}
        "total_wins": total_wins,     # (n_sims, T) — for tests/percentiles
        "total_losses": total_losses,
    }
