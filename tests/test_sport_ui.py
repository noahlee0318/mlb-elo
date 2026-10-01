"""Exercise the offline previews without importing the MLB model pipeline."""

import html
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.sport_teams import load_teams

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sport,count", [("NFL", 32), ("NBA", 30)])
def test_offline_directory_search_and_matchup(monkeypatch, sport, count):
    def no_network(*args, **kwargs):
        raise AssertionError("Preview pages must not request live data")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    teams = load_teams(sport)["teams"]
    assert len(teams) == len({team["id"] for team in teams}) == count
    assert all(team["name"] and team["venue"] and team["logo"].startswith("https://")
               for team in teams)
    app = AppTest.from_file(str(ROOT / "app_pages" / f"{sport.lower()}.py")).run()
    assert not app.exception
    assert f"{sport} is not active yet" in app.warning[0].value
    assert [tab.label for tab in app.tabs] == [
        "Today's slate", "Season results", "Season projection", "H2H Sim", "Methods"]
    assert all(button.disabled for button in app.button)
    directory = next(item.value for item in app.markdown if "<table" in item.value)
    assert directory.count("<img ") == count

    app.text_input(key=f"{sport}_team_search").set_value(teams[-1]["name"]).run()
    directory = next(item.value for item in app.markdown if "<table" in item.value)
    assert directory.count("<img ") == 1
    assert html.escape(teams[-1]["venue"]) in directory
    app.text_input(key=f"{sport}_team_search").set_value("no-such-team").run()
    assert any("No teams match" in item.value for item in app.info)

    app.selectbox(key=f"{sport}_home").select(teams[-1]["name"]).run()
    assert not app.exception
    preview = next(item.value for item in app.markdown if "flex-wrap:wrap" in item.value)
    assert html.escape(teams[-1]["venue"]) in preview
    assert teams[-1]["logo"] in preview
    app.selectbox(key=f"{sport}_away").select(teams[-1]["name"]).run()
    assert not app.exception
    assert app.selectbox(key=f"{sport}_home").value != teams[-1]["name"]
