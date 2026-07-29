# mlb-elo

Win-probability modeling for the 30 MLB teams, end to end: **Elo ratings**, two
**supervised ML models** (logistic regression and a calibrated gradient-boosting
model), a **Monte Carlo season simulator** with a full playoff bracket, and a
**Streamlit dashboard** that ties them together. Every prediction is scored
against a naive home-team baseline (a constant 0.540, the league home-win rate)
— the yardstick each model has to beat.

Single data source for everything: the public, unauthenticated MLB Stats API
(`https://statsapi.mlb.com`) — no API key exists or is needed.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## What's in the box

- **Elo pipeline** — chronological replay of the unified game table into current
  team ratings (`src/elo.py`, `src/build_ratings.py`).
- **Feature pipeline** — per-game boxscores → an as-of, leakage-safe feature
  table → chronological train/tune/val/test splits (`src/ingest_boxscores.py`,
  `src/features.py`, `src/splits.py`).
- **Three prediction models** behind one interface — Elo, logistic regression,
  and a calibrated gradient-boosting model (the default), with a cold-start
  fallback to Elo (`src/model_logreg.py`, `src/model_histgb.py`,
  `src/calibration.py`, `src/predict.py`, `src/live_features.py`).
- **Season simulation & playoffs** — a vectorized Monte Carlo engine any model
  can drive, plus an analytic 12-team bracket (`src/simulate.py`,
  `src/standings.py`, `src/season_schedule.py`, `src/playoffs.py`).
- **Head-to-head tool** — single game or any series format between two teams
  under any model (`src/matchup.py`).
- **Dashboard** — four tabs over all of the above (`app.py`).
- **Self-maintaining data** — a staleness-gated, fail-closed on-open refresh
  (`src/daily_refresh.py`).
- **A verification / leakage-audit suite** — standalone gates for every stage
  (`src/audit_leakage.py`, `src/verify_*.py`, `scripts/verify_*.py`).

## Full data build (first-time setup)

All commands run from this folder (`mlb-elo/`), with the venv active. Each stage
feeds the next.

```powershell
python src/ingest_history.py            # 1. games_full.csv  (the source of truth)
python scripts/backfill_season.py --season 2026   #    top up the current season
python src/ingest_boxscores.py          # 2. pitcher_games.csv + batter_games.csv
python src/features.py                  # 3. features.csv (as-of, leakage-safe)
python src/model_logreg.py              # 4. train logistic regression  -> models/
python src/model_histgb.py              #    train gradient boosting     -> models/
python src/calibration.py               #    calibrate the deployed model-> models/
python -m src.build_ratings             # 5. print the current Elo table (sanity)
```

After this, the dashboard runs and refreshes itself. Day to day you never re-run
the build — `scripts/run_refresh.py` (or just opening the app) keeps data current.

## Dashboard

```powershell
streamlit run app.py
```

Four tabs:

1. **Today's slate** — every game today as `(logo) away @ (logo) home`, with the
   probable pitcher under each team, the venue and its city, a Game 1/Game 2
   label for doubleheaders, and the home team's win chance. The favored team is
   bold. A "Data current as of …" line shows the last refresh; **Refresh now**
   forces a full refresh. Team logos come from MLB's static CDN (the browser
   needs internet). Click into a game to see all three models side by side.
2. **Season projection** — pick a foreground model (Calibrated Gradient Boosting
   by default, or Logistic Regression / Elo) and optionally compare others, then
   **Run projection** to Monte-Carlo the rest of the season: projected win
   totals, division/wild-card/playoff odds, pennant and title odds. The ML
   models build as-of-today team features once (~a couple of minutes, then
   cached); Elo is instant. Button-gated so a page load never blocks.
3. **H2H Sim** — pick two teams, a format (single game through best-of-7 World
   Series), and the host, and get the series win probability under the selected
   model (Elo is instant; ML builds the two teams' features in seconds).
4. **Methods** — a plain-language write-up of the model.

## Prediction models

All three implement one seam so the simulator, bracket, and UI are
model-agnostic — swapping models is a new adapter, not an engine change
(`src/prob_model.py`, `src/ml_prob_model.py`).

- **Elo** — `src/elo.py` / `src/build_ratings.py`. K = 4, home-field = 25 Elo
  points, ratings regress 1/3 toward 1500 at each season boundary. Keyed on
  `team_id`. Margin-of-victory is plumbed (`build_ratings(use_mov=True)`) but
  ships OFF.
- **Logistic regression** — `src/model_logreg.py` (Chunk 6). One
  `StandardScaler` + `LogisticRegression` pipeline; the scaler is fit once, only
  on the rows the model is fit on. Deterministic.
- **Gradient boosting** — `src/model_histgb.py` (Chunk 7). A
  `HistGradientBoostingClassifier` on the same features/splits, tuned against the
  2023 val slice, with a pre-committed rule deciding whether it beats logreg.
- **Calibrated gradient boosting (deployed default)** — `src/calibration.py`
  (Chunk 8). A base model trained on `tune` with the calibrator fit on the
  held-out 2023 `val` slice it never saw. This is the model the UI serves.
- **Serving + cold start** — `src/predict.py::predict_game` /
  `predict_all_models` score already-built feature states. If **either** team has
  fewer than 15 games that season the ML feature vector is undefined, so those
  models fall back to Elo (which is always defined) with a stated reason.

Trained artifacts live in `models/` (`*.joblib` + `*_meta.json`); test-set
scores, per-model predictions, reliability curves, and the model-selection
record live in `data/eval/`.

## Feature pipeline (leakage-safe by construction)

- **Boxscores** — `src/ingest_boxscores.py` (Chunk 2) pulls every final
  regular-season + postseason game 2015–2025 (incl. 2020) into per-season ZIPs
  (`data/raw/boxscores_*.zip`) and builds `data/pitcher_games.csv` +
  `data/batter_games.csv`. Resumable; a staging dir is rolled into a season's
  ZIP only when complete, so an interrupt can't corrupt an archive.
- **Features** — `src/features.py` (Chunk 3) builds `data/features.csv`. Every
  feature for a game uses only rows strictly earlier in the canonical
  `(date, game_number, game_pk)` order (`game_order`), so a doubleheader's game 2
  can use game 1 without leaking. Team state (form, run differential) resets at
  the season boundary; individual pitcher windows (FIP, K-rate) carry over.
- **Splits** — `src/splits.py` (Chunk 5) makes strictly chronological
  train/tune/val/test splits by season (2020 excluded), and applies the
  cold-start eval filter (both teams ≥ 15 prior games that season).
- **Live serving** — `src/live_features.py` (Chunk 9) computes today's features
  by calling the **same** builder as training (no train/serve skew), verified
  column-by-column by the skew test.
- **Park factors** — `src/park_factors.py` (Chunk 10) is a per-venue,
  per-season run-environment multiplier frozen from prior completed seasons only.
  It's a **validation** feature (expected to be inert in a win-probability
  differential), retrained and compared but not deployed.

## Season simulation & playoffs

- `src/simulate.py` — a pure, fully vectorized Monte Carlo engine: one
  `(n_sims × n_games)` draw decides every simulated game; team strength is held
  constant (a simulated result carries no new information). Returns projected win
  totals and postseason odds.
- `src/standings.py` — current records, head-to-head, and the tiebreak ordering
  (`rank_league`), pure so the sim can call it per simulated season. Documents
  the one deliberate shortcut (deep MLB tiebreak levels → a seeded random draw).
- `src/season_schedule.py` — the remaining regular-season schedule and the
  league/division structure (from the API teams endpoint, never hardcoded).
- `src/playoffs.py` — the 12-team bracket verified against MLB's format (byes,
  no reseeding in the DS, 2-3-2 LCS/WS). Each series is resolved analytically
  (a DP over game sequences), statistically identical to playing it out.

## Backtest

Walk-forward backtest over the unified Elo window (`data/games_full.csv`, via
`load_elo_games`): warm up ratings on all prior seasons, then score every game
of the test season (default 2026) — each game predicted **before** its result
updates the ratings, so the evaluation is leakage-free — comparing Elo vs the
0.540 home baseline on accuracy and log-loss:

```powershell
python src/backtest.py         # test season 2026
python src/backtest.py 2025    # or any other season in the Elo window
```

(`python -m src.backtest` works too.) Purely offline — no API calls. The Elo
constants stay at the live app's values; if you ever tune them, do it against an
earlier season and keep the current season sealed.

## Refreshing data

The app refreshes itself — no scheduler and no morning ritual. The one gate is
`src/daily_refresh.py::refresh_if_stale`; on open it runs a staleness-gated,
fail-closed chain against the single unified table:

- If the ratings snapshot is younger than ~3 hours, it **skips** — no network,
  no writes. Otherwise it runs, in order: **1** pull recent finals into
  `data/games_full.csv` (`scripts/backfill_season.py`, final-only, idempotent),
  **3** rebuild Elo ratings into `data/ratings.csv`, **4** log today's slate to
  `data/predictions.csv`. These three are fast and boxscore-independent.
- **Boxscores (step 2)** feed only the on-demand ML matchup views, so the
  minutes-long pull is **deferred** — the on-open path skips it (never blocks
  the UI); the **Refresh now** button and `scripts/run_refresh.py` run it.
- **Fail-closed:** the five mutable CSVs are snapshotted to `*.dref.bak` before
  any write. A failure in the core (games/ratings) restores every file
  byte-for-byte — a failed refresh is indistinguishable from no refresh. A
  prior day's logged prediction can never be rewritten (asserted). A
  cross-process lockfile keeps two opens from refreshing at once.
- `python scripts/run_refresh.py [--force]` is the manual / scheduler-ready
  entry point (runs the full chain incl. boxscores). `python -m src.refresh`
  re-snapshots ratings only. (`python -m src.ingest` is retired.)

## Scoring logged predictions

After games finish, settle logged predictions and compare Elo vs baseline on
accuracy and log-loss in place:

```powershell
python -m src.score
```

## Verification & leakage audits

Every stage has a standalone gate that prints real numbers and exits nonzero on
failure — the project's defense against silent leakage and data corruption:

```powershell
python src/audit_leakage.py                 # Chunk 4: 7-test leakage audit -> docs/leakage_audit.md
python src/verify_games.py                  # games_full.csv integrity
python src/verify_boxscores.py              # pitcher/batter log integrity
python src/verify_features.py               # features.csv + train/predict agreement
python src/test_leakage.py                  # 3 independent leakage defenses for features.py
python scripts/run_baselines.py             # Chunk 5 baselines on the 2024-2025 test set
python scripts/verify_backfill.py --season 2026        # gate after a season backfill
python scripts/verify_live_vs_historical.py            # Chunk 9 skew test (live == historical vector)
python scripts/audit_park_leakage.py        # Chunk 10 park-factor leakage gate
python scripts/retrain_withpark.py          # Chunk 10 with/without-park comparison (eval only)
python scripts/validate_ml_sim.py           # sim engine identities under the ML model
python scripts/verify_games_equivalence.py  # the games.csv migration proof
```

## Tests

```powershell
python -m pytest
```

Unit tests cover the Elo math, the simulation engine and bracket, the matchup
math, `games_full`/boxscore ingest, and the games-data accessors.

## Model notes

- **K = 4** — a 162-game season means each game should move ratings very
  little; tutorial values (20–32) would make ratings thrash.
- **Home-field advantage = 25 Elo points** — gives ~0.536 for even teams,
  matching MLB's real ~54% home-win rate.
- **Season boundaries** — ratings regress 1/3 of the way back to 1500 once
  per boundary during the replay, so old juggernauts don't stay overrated.
- **Margin of victory** — the multiplier is plumbed (`update(..., run_diff=...)`,
  `build_ratings(use_mov=True)`) but ships OFF; the default path ignores run
  differential.
- **Probable pitchers are display-only** in the slate. They *do* enter the ML
  models (the live path threads the announced starter's trailing stats), but
  never the Elo path.
- **Everything is keyed on `team_id`** — IDs are stable, names aren't.
- **Cold start** — an ML model needs ≥ 15 games for *both* teams that season;
  under that it falls back to Elo, so early-April predictions are Elo-driven.
- **Regular-season games only** for prediction (`game_type == 'R'`, final);
  spring training, the All-Star Game, and the postseason are excluded from the
  slate, so those days show the empty state. (Boxscore logs keep postseason and
  2020 for completeness; the modeling filters live in `games_data.load_games`.)
- **Empty slates are normal** — off-days and the All-Star break legitimately
  return zero games; the app shows a plain "no games" message.
- Expect win chances mostly in **45%–60%**: single MLB games are near-coinflips,
  and model-vs-baseline separation shows up in log-loss over many games.
- **Display-only enrichment** (venue/city, probable pitchers) degrades
  gracefully: if a lookup fails the slate still renders, and fully offline the
  app falls back to predictions already logged today.

## Project layout

Everything imports absolutely from the project root (`from src.elo import
update`) — the root `conftest.py` and `src/__init__.py` make that work for
pytest and plain scripts alike.

**App & Elo**
- `app.py` — the Streamlit dashboard (slate / projection / H2H / Methods tabs).
- `src/elo.py` — pure Elo math; no I/O, no network.
- `src/build_ratings.py` — chronological replay of the Elo window into ratings.
- `src/mlb_api.py` — shared HTTP session, throttle, season-date lookups.

**Data ingest & access**
- `src/ingest_history.py` — full-history build of `games_full.csv`.
- `scripts/backfill_season.py` — idempotent, final-only current-season top-up.
- `src/ingest_boxscores.py` — per-game boxscores → pitcher/batter logs (ZIPs).
- `src/games_data.py` — the sanctioned readers of `games_full.csv`
  (`load_games`, and `load_elo_games` for the Elo window).
- `src/ingest.py` — retired table writer; MLB schedule-endpoint helpers only.

**Refresh**
- `src/daily_refresh.py` — the one on-open refresh gate (staleness-gated,
  fail-closed games→ratings→predictions chain; boxscores deferred).
- `scripts/run_refresh.py` — manual / scheduler-ready entry point for it.
- `src/refresh.py` — ratings-snapshot + staleness primitives it reuses.

**Features & models**
- `src/features.py` — the as-of, leakage-safe feature builder (`features.csv`).
- `src/features_ml.py` — the frozen, ordered ML feature list.
- `src/splits.py` — chronological train/tune/val/test splits.
- `src/live_features.py` — the live serving path (same builder as training).
- `src/park_factors.py` — leakage-clean park factors (validation feature).
- `src/model_logreg.py` / `src/model_histgb.py` / `src/calibration.py` — train
  the three models into `models/`.
- `src/predict.py` — slate probabilities + `predict_game` model serving.
- `src/prob_model.py` / `src/ml_prob_model.py` — the model seam (Elo + ML
  adapters) the simulator and matchup tool consume.

**Simulation, playoffs, scoring**
- `src/simulate.py` — the vectorized Monte Carlo season engine.
- `src/standings.py` — records, head-to-head, tiebreak ordering.
- `src/season_schedule.py` — remaining schedule + league/division structure.
- `src/playoffs.py` — 12-team bracket with analytic series resolution.
- `src/matchup.py` — head-to-head orchestration for the H2H tab.
- `src/score.py` — settles logged predictions and scores model vs baseline.
- `src/backtest.py` — leakage-free walk-forward Elo evaluation of a season.
- `src/elo_replay.py` / `src/evaluate.py` — leakage-free Elo baseline + metrics.

**Verification / audits**
- `src/audit_leakage.py`, `src/verify_games.py`, `src/verify_boxscores.py`,
  `src/verify_features.py`, `src/test_leakage.py`, `scripts/verify_backfill.py`,
  `scripts/verify_live_vs_historical.py`, `scripts/audit_park_leakage.py`,
  `scripts/retrain_withpark.py`, `scripts/validate_ml_sim.py`,
  `scripts/run_baselines.py`, `scripts/verify_games_equivalence.py`.
- `tests/` — unit tests (Elo, simulation, matchup, ingest, games-data).

## Data files

**Tables**
- `data/games_full.csv` — the unified game table and single source of truth
  (2015– excl. 2020, all game types with modeling filters applied in
  `load_games`). `game_pk` is MLB's stable id and the dedupe key. The Elo
  pipeline reads the last-5-seasons window via `load_elo_games`. The old
  `data/games.csv` is retired to `data/games.csv.retired`; nothing reads it.
- `data/pitcher_games.csv` / `data/batter_games.csv` — per-game boxscore logs
  (2015–2025 incl. 2020, regular + postseason).
- `data/features.csv` — the as-of feature table the models train on.
- `data/ratings.csv` — `team_id, team_name, rating, updated_at`; the Elo
  snapshot every refresh writes so the app loads ratings without a full replay.
- `data/predictions.csv` — `date, game_id, home_id, away_id, elo_prob_home,
  baseline_prob_home, home_won`; `home_won` stays blank until `python -m
  src.score` settles finished games in place.

**Artifacts**
- `models/` — the trained models (`logreg`, `histgb`, `histgb_calibrated`) as
  `*.joblib` + `*_meta.json`.
- `data/eval/` — test-set scores, per-model predictions, reliability curves
  (CSV + PNG), the with/without-park comparison, the skew-test fixture, and the
  model-selection record.
- `data/raw/` — cached schedule JSON per season-month and the per-season
  boxscore ZIP archives.
- `docs/leakage_audit.md` — the Chunk 4 leakage-audit record.
