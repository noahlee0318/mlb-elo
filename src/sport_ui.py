"""Shared dashboard layout for sports whose prediction pipeline is coming soon."""

import streamlit as st


def render_sport_page(sport):
    st.title(f"{sport} — today's slate")
    st.caption(f"{sport} predictions coming soon")
    st.info("Preview: schedules, results, predictions, and simulations are not live yet.")

    slate, results, projection, h2h, methods = st.tabs(
        ["Today's slate", "Season results", "Season projection", "H2H Sim", "Methods"])

    with slate:
        st.button("Refresh now", disabled=True, key=f"{sport}_refresh")
        st.subheader("Today's matchups")
        st.caption("Away team @ home team · home venue above each matchup")
        st.info(f"The {sport} schedule and home-team win probabilities are coming soon.")

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
            st.selectbox("Away team", ["Teams coming soon"], disabled=True,
                         key=f"{sport}_away")
        with home:
            st.selectbox("Home team", ["Teams coming soon"], disabled=True,
                         key=f"{sport}_home")
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
