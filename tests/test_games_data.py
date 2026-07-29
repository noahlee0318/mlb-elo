"""Tests for the games_full.csv accessor. These read the real
data/games_full.csv (built by src/ingest_history.py); skipped if absent so
the suite still runs on a machine that hasn't pulled the history."""

import pytest

from src.games_data import GAMES_FULL_CSV, load_games

pytestmark = pytest.mark.skipif(
    not GAMES_FULL_CSV.exists(),
    reason="data/games_full.csv not built — run python src/ingest_history.py")


def test_default_excludes_exhibition_types():
    df = load_games()
    assert (df["game_type"].isin(["S", "E", "A"]).sum()) == 0


def test_default_excludes_2020():
    df = load_games()
    assert (df["season"] == 2020).sum() == 0


def test_default_excludes_non_mlb_matchups():
    df = load_games()
    assert (~df["is_mlb_matchup"]).sum() == 0


def test_default_is_all_regular_season_finals():
    df = load_games()
    assert set(df["game_type"].unique()) == {"R"}
    assert bool(df["is_final"].all())


def test_include_exhibition_is_larger_and_warns(capsys):
    base = load_games()
    bigger = load_games(include_exhibition=True)
    assert len(bigger) > len(base)
    err = capsys.readouterr().err
    assert "include_exhibition=True" in err


def test_include_postseason_adds_postseason_types():
    df = load_games(include_postseason=True)
    assert df["game_type"].isin(["F", "D", "L", "W"]).any()
    # regular season still present, exhibition still absent
    assert (df["game_type"] == "R").any()
    assert df["game_type"].isin(["S", "E", "A"]).sum() == 0


def test_returns_a_copy_not_a_view():
    df = load_games()
    # mutating the result must not raise SettingWithCopy / touch shared state
    df.loc[df.index[0], "home_score"] = -999
    df2 = load_games()
    assert (df2["home_score"] == -999).sum() == 0


def test_seasons_filter_selects_only_requested():
    df = load_games(seasons=[2021, 2022])
    assert set(df["season"].unique()) == {2021, 2022}


def test_missing_file_error_names_ingest(monkeypatch, tmp_path):
    import src.games_data as gd
    monkeypatch.setattr(gd, "GAMES_FULL_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="ingest_history"):
        gd.load_games()
