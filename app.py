"""Run with streamlit run app.py; each sport loads independently."""

import streamlit as st

st.set_page_config(page_title="Sports predictor", page_icon="🏟️")

page = st.navigation([
    st.Page("app_pages/mlb.py", title="MLB", icon="⚾", default=True),
    st.Page("app_pages/nfl.py", title="NFL", icon="🏈"),
], position="top")
page.run()
