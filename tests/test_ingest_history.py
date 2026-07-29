"""Parsing + verification-logic tests for the historical ingest. No network:
everything runs against tests/fixtures/schedule_fixture.json and small
hand-built frames.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingest_history import (dedupe_snapshots, parse_game, parse_payload,
                                to_frame)
from src.verify_games import iswinner_mismatches

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "schedule_fixture.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_pk(payload):
    return {row["game_pk"]: row for row in parse_payload(payload)}


def test_normal_final_parsed_correctly(by_pk):
    g = by_pk[1001]
    assert g["is_final"] is True
    assert g["home_win"] == 1              # home 5, away 3
    assert g["home_is_winner_flag"] is True
    assert g["season"] == 2018
    assert g["date"] == "2018-04-02"       # officialDate, not gameDate's date
    assert g["winning_pitcher_id"] == 500
    assert g["home_probable_pitcher_id"] == 700


def test_date_uses_official_date_not_gamedate(by_pk):
    # gameDate is 2016-08-21T00:10Z (UTC next day); officialDate is the 20th
    assert by_pk[1005]["date"] == "2016-08-20"


def test_postponed_game_is_not_final_and_has_null_home_win(by_pk):
    g = by_pk[1004]
    assert g["is_final"] is False
    assert g["home_win"] is None
    assert g["home_score"] is None
    assert g["home_is_winner_flag"] is None
    assert g["rescheduled_from"] == "2017-05-14T23:10:00Z"


def test_extra_innings_recorded(by_pk):
    g = by_pk[1005]
    assert g["innings_played"] == 11
    assert g["scheduled_innings"] == 9
    assert g["home_win"] == 0               # home 3, away 4


def test_doubleheader_two_distinct_games(by_pk):
    g1, g2 = by_pk[1002], by_pk[1003]
    assert g1["doubleheader_code"] == "S" and g2["doubleheader_code"] == "S"
    assert {g1["game_number"], g2["game_number"]} == {1, 2}
    assert g1["game_pk"] != g2["game_pk"]
    assert g1["home_win"] == 1 and g2["home_win"] == 0


def test_home_win_is_independent_of_iswinner_flag():
    # scores say home LOST, but the raw flag claims home won — parse_game must
    # report each field from its own source, not reconcile them.
    g = parse_game({
        "gamePk": 9, "season": "2018", "officialDate": "2018-06-01",
        "gameDate": "2018-06-01T18:00:00Z", "gameType": "R",
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "doubleHeader": "N", "gameNumber": 1, "venue": {"id": 1},
        "teams": {
            "home": {"team": {"id": 1}, "score": 2, "isWinner": True},
            "away": {"team": {"id": 2}, "score": 5, "isWinner": False},
        },
    })
    assert g["home_win"] == 0            # from the scores
    assert g["home_is_winner_flag"] is True   # from the raw flag


def test_2020_flagged_anomalous():
    g = parse_game({
        "gamePk": 7, "season": "2020", "officialDate": "2020-07-25",
        "gameDate": "2020-07-25T18:00:00Z", "gameType": "R",
        "status": {"abstractGameState": "Final", "detailedState": "Final"},
        "doubleHeader": "N", "gameNumber": 1, "venue": {"id": 1},
        "teams": {"home": {"team": {"id": 1}, "score": 1, "isWinner": True},
                  "away": {"team": {"id": 2}, "score": 0, "isWinner": False}},
    })
    assert g["is_anomalous_season"] is True


def _mini_frame(rows):
    df = pd.DataFrame(rows)
    df["home_is_winner_flag"] = df["home_is_winner_flag"].astype("boolean")
    df["is_final"] = df["is_final"].astype(bool)
    for c in ("home_score", "away_score", "home_win"):
        df[c] = df[c].astype("Int64")
    return df


def test_check4_flags_a_deliberately_swapped_row():
    # one honest row + one where the isWinner flag contradicts the scores
    df = _mini_frame([
        {"game_type": "R", "is_final": True, "date": "2018-04-02",
         "home_team_id": 1, "away_team_id": 2, "home_score": 5, "away_score": 3,
         "home_win": 1, "home_is_winner_flag": True},          # consistent
        {"game_type": "R", "is_final": True, "date": "2018-04-03",
         "home_team_id": 3, "away_team_id": 4, "home_score": 2, "away_score": 6,
         "home_win": 0, "home_is_winner_flag": True},          # SWAPPED
    ])
    mm = iswinner_mismatches(df)
    assert len(mm) == 1
    assert mm.iloc[0]["home_team_id"] == 3


def test_check4_clean_frame_has_no_mismatches():
    df = _mini_frame([
        {"game_type": "R", "is_final": True, "date": "2018-04-02",
         "home_team_id": 1, "away_team_id": 2, "home_score": 5, "away_score": 3,
         "home_win": 1, "home_is_winner_flag": True},
        {"game_type": "R", "is_final": True, "date": "2018-04-03",
         "home_team_id": 3, "away_team_id": 4, "home_score": 1, "away_score": 7,
         "home_win": 0, "home_is_winner_flag": False},
    ])
    assert len(iswinner_mismatches(df)) == 0


def test_dedupe_keeps_final_over_postponed_snapshot():
    # same game_pk twice: a Postponed placeholder and the resolved Final —
    # exactly what the MLB API returns for a made-up game. Keep the Final.
    rows = [
        parse_game({
            "gamePk": 500, "season": "2015", "officialDate": "2015-05-06",
            "gameDate": "2015-05-06T00:40:00Z", "gameType": "R",
            "status": {"abstractGameState": "Preview", "detailedState": "Postponed"},
            "doubleHeader": "N", "gameNumber": 1, "venue": {"id": 1},
            "teams": {"home": {"team": {"id": 115}}, "away": {"team": {"id": 109}}},
        }),
        parse_game({
            "gamePk": 500, "season": "2015", "officialDate": "2015-05-06",
            "gameDate": "2015-05-06T19:15:00Z", "gameType": "R",
            "status": {"abstractGameState": "Final", "detailedState": "Final"},
            "doubleHeader": "Y", "gameNumber": 2, "venue": {"id": 1},
            "teams": {"home": {"team": {"id": 115}, "score": 1, "isWinner": False},
                      "away": {"team": {"id": 109}, "score": 5, "isWinner": True}},
        }),
    ]
    out = to_frame(rows)
    assert len(out) == 1
    assert bool(out.iloc[0]["is_final"]) is True
    assert int(out.iloc[0]["home_score"]) == 1


def test_dedupe_preserves_distinct_doubleheader_gamepks():
    # two games of a doubleheader have DIFFERENT game_pks and must both survive
    rows = parse_payload(json.loads(FIXTURE.read_text(encoding="utf-8")))
    out = dedupe_snapshots(to_frame(rows))
    assert {1002, 1003}.issubset(set(out["game_pk"]))


def test_to_frame_types_and_sort():
    df = to_frame(parse_payload(json.loads(FIXTURE.read_text(encoding="utf-8"))))
    assert str(df["game_pk"].dtype) == "int64"
    assert str(df["home_score"].dtype) == "Int64"
    assert str(df["home_is_winner_flag"].dtype) == "boolean"
    assert list(df["date"]) == sorted(df["date"])   # sorted by date
