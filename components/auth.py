import streamlit as st
from components.firebase_config import auth

def login_signup_page():
    st.markdown("""
        <style>
            .login-card {
                background-color: #f9f9f9;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                max-width: 400px;
                margin: auto;
            }
            .login-title {
                text-align: center;
                font-size: 2rem;
                color: #2c3e50;
            }
            .stButton>button {
                width: 100%;
                background-color: #2ecc71 !important;
                color: white !important;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.5rem;
                margin-top: 1rem;
            }
            .stRadio > div {
                justify-content: center;
            }
        </style>
    """, unsafe_allow_html=True)

  

    st.markdown('<div class="login-title">🔐 Login / Sign Up</div>', unsafe_allow_html=True)
    choice = st.radio("", ["Login", "Sign Up"], horizontal=True)

    email = st.text_input("📧 Email", placeholder="Enter your email")
    password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

    if choice == "Sign Up":
        if st.button("🆕 Create Account"):
            try:
                auth.create_user_with_email_and_password(email, password)
                st.success("🎉 Account created successfully!")
                st.session_state.authenticated = True
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        if st.button("✅ Login"):
            try:
                auth.sign_in_with_email_and_password(email, password)
                st.success("🔓 Logged in successfully!")
                st.session_state.authenticated = True
            except Exception as e:
                st.error(f"❌ Login failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True) 