"""Boxscore parsing + verification-logic tests. No network — everything runs
against tests/fixtures/boxscore_fixture.json and hand-built frames."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingest_boxscores import parse_boxscore, parse_ip_to_outs
from src.verify_boxscores import runs_cross_mismatches

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "boxscore_fixture.json"


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["games"]


def rows_for(fx, key):
    g = fx[key]
    return parse_boxscore(g["meta"], g["box"])


# ------------------------------------------------------------ parse_ip_to_outs

@pytest.mark.parametrize("ip,outs", [
    ("0.0", 0), ("0.1", 1), ("0.2", 2), ("1.0", 3),
    ("6.1", 19), ("9.0", 27), ("2.2", 8), (None, None), ("", None),
])
def test_parse_ip_to_outs(ip, outs):
    assert parse_ip_to_outs(ip) == outs


def test_parse_ip_rejects_malformed_fraction():
    with pytest.raises(ValueError):
        parse_ip_to_outs("6.4")  # thirds notation never has .4


# ------------------------------------------------------------ complete game

def test_complete_game_and_orders(fx):
    pit, bat = rows_for(fx, "9001")
    home = [r for r in pit if r["is_home"]]
    away = [r for r in pit if not r["is_home"]]
    assert len(home) == 1 and home[0]["complete_game"] is True
    assert home[0]["ip_outs"] == 27 and home[0]["is_starter"]
    assert home[0]["decision"] == "W"          # from Chunk 1 winning_pitcher_id
    assert [r["appearance_order"] for r in away] == [1, 2]
    assert away[0]["decision"] == "L"
    assert sum(r["ip_outs"] for r in away) == 24  # home led through top 9
    assert not any(r["starter_flag_mismatch"] for r in pit)


def test_pinch_runner_with_zero_pa_is_included(fx):
    _, bat = rows_for(fx, "9001")
    pr = [b for b in bat if b["batter_id"] == 111009]
    assert len(pr) == 1
    assert pr[0]["plate_appearances"] == 0
    assert pr[0]["batting_order"] == "901"
    assert pr[0]["batting_order_slot"] == 9
    assert pr[0]["position"] == "PR"


# ------------------------------------------------------------ opener / two-way

def test_opener_heuristic_flags_short_first_pitcher(fx):
    pit, _ = rows_for(fx, "9003")
    opener = next(r for r in pit if r["pitcher_id"] == 301)
    bulk = next(r for r in pit if r["pitcher_id"] == 302)
    assert opener["is_starter"] and opener["is_opener"]     # 3 outs, 5 BF
    assert not bulk["is_starter"] and not bulk["is_opener"]
    assert bulk["decision"] == "W"


def test_normal_starter_is_not_opener(fx):
    pit, _ = rows_for(fx, "9001")
    home_starter = next(r for r in pit if r["pitcher_id"] == 101)
    assert home_starter["is_starter"] and not home_starter["is_opener"]


def test_two_way_player_in_both_tables_same_game(fx):
    pit, bat = rows_for(fx, "9003")
    assert any(r["pitcher_id"] == 660271 for r in pit)
    assert any(b["batter_id"] == 660271 for b in bat)
    tw_p = next(r for r in pit if r["pitcher_id"] == 660271)
    assert tw_p["ip_outs"] == 19                      # "6.1" -> 19, not 6.1
    tw_b = next(b for b in bat if b["batter_id"] == 660271)
    assert tw_b["batting_order_slot"] == 1


def test_blown_save_note_parsed(fx):
    pit, _ = rows_for(fx, "9003")
    closer = next(r for r in pit if r["pitcher_id"] == 402)
    # 402 is the losing pitcher per meta -> L wins over the note's BS
    assert closer["decision"] == "L"


# ------------------------------------------------------------ rain-shortened

def test_rain_shortened_outs(fx):
    pit, _ = rows_for(fx, "9004")
    home = next(r for r in pit if r["is_home"])
    away = next(r for r in pit if not r["is_home"])
    assert home["ip_outs"] == 15 and away["ip_outs"] == 12


# ------------------------------------------------------------ Check 3 logic

def _games_frame():
    return pd.DataFrame([
        {"game_pk": 9001, "home_team_id": 111, "away_team_id": 222,
         "home_score": 5, "away_score": 3},
    ])


def _pitcher_frame(home_runs_allowed, away_runs_allowed):
    return pd.DataFrame([
        {"game_pk": 9001, "team_id": 111, "runs": home_runs_allowed},
        {"game_pk": 9001, "team_id": 222, "runs": away_runs_allowed},
    ])


def test_check3_clean_when_orientation_correct():
    # home pitchers allowed the AWAY score (3); away allowed the HOME score (5)
    mm = runs_cross_mismatches(_games_frame(), _pitcher_frame(3, 5))
    assert len(mm) == 0


def test_check3_flags_deliberately_swapped_home_away():
    # swapped: home pitchers charged with the home score — orientation bug
    mm = runs_cross_mismatches(_games_frame(), _pitcher_frame(5, 3))
    assert len(mm) == 2
    assert set(mm["team_id"]) == {111, 222}
