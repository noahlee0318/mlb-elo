# mlb-elo — technical report

## Season simulation (Monte Carlo)

Added 2026-07-18: a Monte Carlo projection of the rest of the regular
season and the postseason — projected standings with uncertainty bands,
division/playoff/pennant/title probabilities, modal bracket matchups, and
a championship distribution. 10,000 simulated seasons by default, seeded
and reproducible.

### Frozen ratings, and why

Ratings are computed once, before the simulation loop, and held constant
through every simulated game. No Elo updates happen on simulated results —
not with a reduced K, not for "momentum."

The reasoning is circularity: an Elo update exists to extract information
from an observed game, and a simulated result contains no information —
it was *generated from* the current ratings. Feeding it back in creates a
self-reinforcing loop: a team that gets lucky early in a simulated season
gets a rating bump, which raises its win probability in later simulated
games, which fattens the tails of its win distribution and pushes playoff
odds toward the extremes. The distribution ends up wider than the
schedule's genuine randomness supports.

Guardrail: `tests/test_simulate.py::test_ratings_are_not_mutated_by_a_simulation_run`
asserts the ratings object is byte-identical after a full simulation run,
so this constraint cannot silently regress.

### The `ProbabilityModel` seam

The simulation engine (`src/simulate.py`) and the bracket
(`src/playoffs.py`) contain no model knowledge — no ratings, no
home-field constant, no win-probability formula, and no import of
`elo.py` (enforced by review: `grep -ri "elo" src/simulate.py
src/playoffs.py` returns nothing). They consume probabilities through the
interface in `src/prob_model.py`:

- `schedule_probs(games, n_sims=None)` → home-win probabilities for the
  remaining schedule, either shape `(G,)` (one fixed number per game) or
  `(n_sims, G)` (a separate draw per simulation, for models that carry
  team-strength uncertainty). The engine broadcasts `rand < p` over
  either shape with no branching.
- `matchup_prob(home_id, away_id, context=None)` → probability for a
  hypothetical game, needed for playoff series where opponents aren't
  known in advance.

`EloProbabilityModel` is a ~20-line adapter translating `(home_id,
away_id)` into the existing `expected_home()` call over the ratings
snapshot. Replacing Elo with a supervised model (pitchers, lineups, rest,
park, weather) means writing a new adapter — the simulator, bracket,
tests, and UI do not change.

### Current records and remaining schedule

Current W–L comes from `data/games_full.csv` (the project's single source of
truth, via `load_elo_games`) — never from a standings endpoint. The remaining
schedule is the Stats API schedule from today through the season end
date, `gameType=R`, minus anything already Final in the unified table (deduped
on `game_id`) and minus games the API already marks Final. Division and
league membership come from the API teams endpoint, cached per process —
no hardcoded team/division dict.

### Tiebreakers (documented shortcut)

MLB no longer plays Game 163. The implemented order is: **wins →
head-to-head among the tied group → intradivision winning percentage →
seeded random draw**. Head-to-head combines played results (from the
unified table) with the simulated remaining games in that same simulation.
MLB's official rules continue deeper (intraleague record, second-half
intraleague splits); those levels arise rarely enough that a random draw
in their place does not meaningfully move aggregate probabilities. The
shortcut is stated in `standings.rank_league`'s docstring rather than
left silent.

### Postseason format and analytic series resolution

Format verified 2026-07-18 against MLB.com's playoff-format FAQ
(mlb.com/news/mlb-playoff-format-faq): 12 teams, 6 per league; seeds 1–2
bye; Wild Card best-of-3 all at the higher seed (3v6, 4v5); Division
Series best-of-5 (2-2-1) with a **fixed bracket, no reseeding** — 1 plays
the 4/5 winner, 2 plays the 3/6 winner; LCS and World Series best-of-7
(2-3-2); WS home field to the better regular-season record.

Series are not simulated game-by-game. For each series the engine gets
`p(home wins)` for the higher seed in each scheduled game slot (correct
hosting pattern per round), runs an exact DP over win/loss sequences to
get the series-win probability, and flips **one** coin against it.
Statistically identical to per-game simulation; the only loss is
game-level detail ("wins in 6"), which nothing displays. Series
probabilities are memoized per matchup, so 10,000 postseasons cost only
a few hundred DP evaluations.

### Performance and caching

The regular season is fully vectorized: one probability vector, one
`(n_sims × games)` uniform draw, and per-team tallies as matrix products.
10,000 sims × 971 games runs in ~2 s. In the app the projection is cached
with `st.cache_data` keyed on (ratings-snapshot timestamp, schedule hash,
n_sims, seed) — Streamlit reruns reuse the cached result; only a data
refresh (new ratings timestamp) or "Refresh now" triggers a new run.

### Limitations

- **No pitcher rotation or usage** — a game against a staff ace and a
  bullpen day are priced identically.
- **No injuries**, and **no trade deadline**: a July projection has a
  short shelf life — August rosters can look materially different.
- **No September callups**, and no modeling of teams resting starters
  after clinching (real teams give away late-season games).
- **Static team strength**: the single largest simplification. The model
  assumes today's rating holds for 10+ weeks. The `(n_sims, G)` return
  shape exists precisely so a future model can draw strength uncertainty
  per simulation instead.
- Tiebreaks past intradivision record are a seeded coin flip (above).
- A couple of teams currently project 161 games (a postponed game with no
  makeup date yet); the UI flags this rather than forcing 162.
