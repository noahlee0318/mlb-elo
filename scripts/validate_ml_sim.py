"""Weakness 2, Deliverable 3 — the Phase 2 §4.3 engine identities re-run with
the season sim driven by the CALIBRATED HISTGB model (not Elo), plus the
engine-integrity tripwires. These identities are properties of the ENGINE, so
they must hold under any ProbabilityModel — a failure means the ML adapter broke
the protocol contract (wrong shape, NaN probs, mutated state).

Exits nonzero on any failure.
"""

import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ml_prob_model import MLProbabilityModel
from src.season_schedule import remaining_schedule, team_structure
from src.simulate import simulate_season
from src.standings import current_records, played_h2h

ROOT = Path(__file__).resolve().parents[1]
ENGINE_FILES = ["src/simulate.py", "src/playoffs.py", "src/matchup.py"]
N_SIMS = 2000
SEED = 7
# Inverted "runaway favorite" guard. Per the spec prose ("a 40% favorite is an
# alarm, not an edge") and the Chunk 7 inverted-tripwire precedent ("do not fail
# on it, but flag hard for review"): >25% is a LOUD flag-for-review, not a hard
# fail; only a clearly-pathological >40% (a leak/broken-adapter signature) exits
# nonzero. 2026 is genuinely top-heavy — even Elo's own favorite sits ~22%.
FAVORITE_FLAG = 0.25
FAVORITE_HARDFAIL = 0.40


def check_engine_integrity():
    """Tripwires: the engine files must not import elo or any model, and must
    be unedited. (No git here, so this is import-grep + the session fact that
    these files were never written.)"""
    fails = []
    forbidden = re.compile(r"import\s+.*\b(elo|model_logreg|calibration|"
                           r"ml_prob_model)\b|from\s+src\.(elo|model_logreg|"
                           r"calibration|ml_prob_model)\b")
    for rel in ENGINE_FILES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        hits = [ln for ln in src.splitlines() if forbidden.search(ln)]
        ok = not hits
        print(f"  {rel}: no elo/model import -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{rel} imports a model/elo: {hits}")
    return fails


def main():
    fails = []
    print("=" * 66)
    print("ENGINE INTEGRITY (adapter + selector only; engine untouched)")
    print("=" * 66)
    fails += check_engine_integrity()

    print("\n" + "=" * 66)
    print(f"§4.3 IDENTITIES under CALIBRATED HISTGB  (n_sims={N_SIMS}, seed={SEED})")
    print("=" * 66)
    remaining = remaining_schedule()
    structure = team_structure()
    records = current_records()
    h2h = played_h2h()
    print(f"  remaining games: {len(remaining)} | teams: {len(structure)}")

    model = MLProbabilityModel("histgb_cal")

    # schedule_probs sanity BEFORE the sim (shape/range/NaN)
    p = np.asarray(model.schedule_probs(remaining, n_sims=N_SIMS), dtype=float)
    shape_ok = p.ndim == 1 and p.shape[0] == len(remaining)
    range_ok = bool(np.isfinite(p).all() and (p > 0).all() and (p < 1).all())
    print(f"  schedule_probs shape={p.shape} (expected ({len(remaining)},)) "
          f"-> {'OK' if shape_ok else 'FAIL'}")
    print(f"  all probs finite and in (0,1): {range_ok} "
          f"-> {'OK' if range_ok else 'FAIL'}")
    if not shape_ok:
        fails.append("schedule_probs shape differs from Elo's (G,)")
    if not range_ok:
        fails.append("schedule_probs has NaN or out-of-(0,1) values")

    # snapshot the feature cache to prove static strength (no mutation)
    cache_before = copy.deepcopy(model.cache._comp)

    res = simulate_season(remaining, records, structure, model,
                          n_sims=N_SIMS, seed=SEED, played_h2h=h2h)
    teams = res["teams"]

    berths = sum(t["p_berth"] for t in teams.values())
    titles = sum(t["p_title"] for t in teams.values())
    print(f"  berths sum = {berths:.6f} (expected 12.0) -> "
          f"{'OK' if abs(berths-12.0) < 1e-9 else 'FAIL'}")
    print(f"  titles sum = {titles:.6f} (expected 1.0)  -> "
          f"{'OK' if abs(titles-1.0) < 1e-9 else 'FAIL'}")
    if abs(berths - 12.0) >= 1e-9:
        fails.append(f"berths sum {berths} != 12")
    if abs(titles - 1.0) >= 1e-9:
        fails.append(f"titles sum {titles} != 1")

    for lg in sorted({v[0] for v in structure.values()}):
        s = sum(teams[t]["p_pennant"] for t in structure if structure[t][0] == lg)
        ok = abs(s - 1.0) < 1e-9
        print(f"  pennant sum [{lg}] = {s:.6f} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"pennant sum {lg} {s} != 1")

    tw = np.asarray(res["total_wins"]).sum(axis=1)
    tl = np.asarray(res["total_losses"]).sum(axis=1)
    wl_ok = bool((tw == tl).all())
    print(f"  league wins == losses every sim -> {'OK' if wl_ok else 'FAIL'}")
    if not wl_ok:
        fails.append("league wins != losses in some sim")

    # static-strength: feature cache unchanged by the run
    cache_ok = model.cache._comp == cache_before
    print(f"  feature cache unmutated by sim -> {'OK' if cache_ok else 'FAIL'}")
    if not cache_ok:
        fails.append("feature cache mutated during simulation")

    # reproducibility under the ML model
    res2 = simulate_season(remaining, records, structure, model,
                           n_sims=N_SIMS, seed=SEED, played_h2h=h2h)
    repro = res["teams"] == res2["teams"]
    print(f"  fixed seed reproducible under histgb -> {'OK' if repro else 'FAIL'}")
    if not repro:
        fails.append("fixed seed not reproducible under histgb")

    # inverted tripwire: flag a strong favorite, hard-fail only a runaway one
    fav_team = max(teams, key=lambda t: teams[t]["p_title"])
    fav = teams[fav_team]["p_title"]
    fav_name = structure[fav_team][2] if len(structure[fav_team]) > 2 else fav_team
    if fav > FAVORITE_HARDFAIL:
        print(f"  championship favorite: {fav_name} p_title={fav:.4f} -> "
              f"HARD FAIL (> {FAVORITE_HARDFAIL})")
        fails.append(f"runaway favorite {fav_name} p_title={fav:.4f} > "
                     f"{FAVORITE_HARDFAIL} — probable leak/broken adapter")
    elif fav > FAVORITE_FLAG:
        print(f"  championship favorite: {fav_name} p_title={fav:.4f} -> "
              f"FLAG (> {FAVORITE_FLAG}, review — not a hard fail)")
        print(f"    NOTE: {fav_name} p_title={fav:.4f} exceeds the {FAVORITE_FLAG} "
              "review line but is below the {:.2f} hard-fail line. Investigate, "
              "but a genuinely dominant team in a top-heavy season is a legitimate "
              "cause (Elo's own favorite is comparably high). Flagged, not failed."
              .format(FAVORITE_HARDFAIL), file=sys.stderr)
    else:
        print(f"  championship favorite: {fav_name} p_title={fav:.4f} -> OK "
              f"(< {FAVORITE_FLAG})")

    print("=" * 66)
    if fails:
        print("VALIDATION FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("ALL §4.3 IDENTITIES PASS UNDER HISTGB — adapter honors the "
          "protocol; engine untouched.")


if __name__ == "__main__":
    main()
