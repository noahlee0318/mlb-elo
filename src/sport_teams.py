"""Offline team metadata and MLB-style display helpers for NFL/NBA previews."""

import html
import json
from pathlib import Path

TEAM_DIR = Path(__file__).resolve().parents[1] / "data" / "teams"
CELL_STYLE = "padding:10px 12px;border-bottom:1px solid rgba(128,128,128,0.3);"


def load_teams(sport):
    if sport not in {"NFL", "NBA"}:
        raise ValueError(f"Unsupported sport: {sport}")
    return json.loads((TEAM_DIR / f"{sport.lower()}.json").read_text(encoding="utf-8"))


def team_block(team):
    return (
        '<div style="display:flex;align-items:center;gap:8px;">'
        f'<img src="{html.escape(team["logo"], quote=True)}" '
        f'width="26" height="26" alt="{html.escape(team["abbreviation"])} logo">'
        f'<span>{html.escape(team["name"])}</span></div>'
    )


def venue_label(team):
    location = ", ".join(part for part in [team["city"], team["region"]] if part)
    return team["venue"] + (f" · {location}" if location else "")


def team_directory(teams, venue_type):
    rows = "".join(
        f'<tr><td style="{CELL_STYLE}">{team_block(team)}</td>'
        f'<td style="{CELL_STYLE}">{html.escape(venue_label(team))}</td></tr>'
        for team in teams
    )
    return (
        '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr><th scope="col" style="{CELL_STYLE}text-align:left;">Team</th>'
        f'<th scope="col" style="{CELL_STYLE}text-align:left;">Home {venue_type}</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def matchup_preview(away, home):
    return (
        '<div style="font-size:0.8em;opacity:0.7;margin-bottom:4px;">'
        f'{html.escape(venue_label(home))}</div>'
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        + team_block(away) + '<span style="opacity:0.65;">@</span>'
        + team_block(home) + '</div>'
    )
