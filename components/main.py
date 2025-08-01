import streamlit as st
from components import auth, dashboard, alerts

def main():
    st.set_page_config(page_title="EarthScape Analytics", layout="centered")

    if "user" not in st.session_state:
        menu = st.sidebar.selectbox("Select", ["Login", "Sign Up"])
        if menu == "Login":
            auth.login()
        else:
            auth.signup()
    else:
        st.sidebar.write(f"👤 Logged in as: {st.session_state['user']['email']}")
        auth.logout()

        latest = dashboard.show_dashboard()
        alerts.show_alerts(latest)

if __name__ == "__main__":
    main()
