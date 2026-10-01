"""Shared dashboard layout for sports whose prediction pipeline is coming soon."""

import streamlit as st

from src.sport_teams import load_teams, matchup_preview, team_directory


def render_sport_page(sport):
    catalog = load_teams(sport)
    teams = catalog["teams"]
    by_name = {team["name"]: team for team in teams}
    venue_type = "stadium" if sport == "NFL" else "arena"
    st.title(f"{sport} — today's slate")
    st.warning(f"{sport} is not active yet. Schedules, results, predictions, and simulations "
               "are coming soon. Team and venue browsing is available below.")
    st.caption(f"{sport} predictions coming soon")

    slate, results, projection, h2h, methods = st.tabs(
        ["Today's slate", "Season results", "Season projection", "H2H Sim", "Methods"])

    with slate:
        st.button("Refresh now", disabled=True, key=f"{sport}_refresh")
        st.subheader("Today's matchups")
        st.caption("Away team @ home team · home venue above each matchup")
        st.info(f"The {sport} schedule and home-team win probabilities are coming soon.")
        st.subheader(f"{sport} teams & home venues")
        query = st.text_input("Find a team or venue", key=f"{sport}_team_search",
                              placeholder="Search by team, abbreviation, venue, or city")
        filtered = [team for team in teams if query.strip().casefold() in " ".join(
            team[field] for field in ("name", "abbreviation", "venue", "city", "region")
        ).casefold()]
        st.caption(f"{len(filtered)} of {len(teams)} teams · Team directory as of {catalog['as_of']}")
        if filtered:
            st.markdown(team_directory(filtered, venue_type), unsafe_allow_html=True)
        else:
            st.info("No teams match your search.")
        st.caption(f"Team marks: [ESPN](https://www.espn.com/{sport.lower()}/teams) · "
                   "Venue corrections checked against league and team sources")

    with results:
        st.subheader("Regular-season results")
        st.caption("Actual records and standings")
        st.info(f"Official {sport} standings will appear here once the results feed is connected.")

    with projection:
        st.subheader("Season projection")
        st.caption("Projected wins, playoff chances, and championship probabilities")
        st.selectbox("Model", ["Models coming soon"], disabled=True,
                     key=f"{sport}_projection_model")
        st.button("Run season simulation", disabled=True, key=f"{sport}_simulate")
        st.info(f"{sport} season simulations are coming soon.")

    with h2h:
        st.subheader("Head-to-head simulator")
        away, home = st.columns(2)
        with away:
            away_name = st.selectbox("Away team", list(by_name), key=f"{sport}_away")
        with home:
            home_name = st.selectbox("Home team", [name for name in by_name if name != away_name],
                                     key=f"{sport}_home")
        st.caption("Matchup preview · Choose teams to see their logos and the home venue")
        st.markdown(matchup_preview(by_name[away_name], by_name[home_name]),
                    unsafe_allow_html=True)
        st.button("Simulate matchup", disabled=True, key=f"{sport}_matchup")
        st.info("Matchup probabilities and simulations are coming soon.")

    with methods:
        st.subheader("Methods")
        st.markdown(
            f"This is the **{sport} dashboard preview**, using the same five-tab "
            "layout as MLB. The schedule, results feed, sport-specific models, "
            "and playoff simulations have not been connected yet."
        )
        st.markdown(
            "Once available, matchups will read **away team @ home team**, "
            "with the home venue above the teams and the home team's win "
            "probability beside the matchup. Model assumptions and validation "
            "results will be documented here before predictions go live."
        )
        st.markdown(
            f"**Team directory:** all {len(teams)} {sport} teams, with logo links and "
            f"home {venue_type} names from [ESPN](https://www.espn.com/{sport.lower()}/teams), "
            "with outdated venues corrected using league and team sources. "
            f"Saved on {catalog['as_of']}; venue names can change after this snapshot. "
            "Logos require an internet connection; the team names and venues are stored locally."
        )
