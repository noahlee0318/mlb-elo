"""Official regular-season dates and standings for the dashboard."""

from datetime import date

from src.mlb_api import season_dates, session


def results_season(today):
    """Keep last season visible until this year's regular season starts."""
    today = date.fromisoformat(today)
    start, end = season_dates(today.year)
    if today.isoformat() < start:
        return today.year - 1, True
    return today.year, today.isoformat() > end


def official_results(season):
    """Read actual standings, including MLB's official division ordering."""
    response = session.get(
        "https://statsapi.mlb.com/api/v1/standings",
        params={"leagueId": "103,104", "season": season,
                "standingsTypes": "regularSeason", "hydrate": "division,league"},
        timeout=30,
    )
    response.raise_for_status()
    divisions = []
    for record in response.json().get("records", []):
        rows = []
        teams = sorted(record.get("teamRecords", []),
                       key=lambda team: int(team["divisionRank"]))
        for team in teams:
            rows.append({
                "Rank": int(team["divisionRank"]),
                "Team": team["team"]["name"],
                "W": team["wins"], "L": team["losses"],
                "PCT": team["winningPercentage"],
                "GB": team.get("divisionGamesBack", "—"),
                "Status": ("Division winner" if team.get("divisionChamp") else
                           "Wild card" if team.get("wildCardClinched") else
                           "Playoff berth clinched" if team.get("clinched") else "—"),
            })
        divisions.append((record["division"]["name"], rows))
    if not divisions or sum(len(rows) for _, rows in divisions) != 30:
        raise ValueError("MLB has not returned standings for all 30 teams")
    return divisions
