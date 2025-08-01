import streamlit as st

from components.auth import login_signup_page
from components.dashboard import show_dashboard


st.set_page_config(page_title="EarthScape Climate System", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_signup_page()
else:
    show_dashboard()
