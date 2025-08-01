import streamlit as st
import datetime
import pandas as pd
import os

def save_to_csv(data, filename="data/feedback.csv"):
    df = pd.DataFrame([data])
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def feedback_form():
    st.title("📬 Feedback Form")
    st.markdown("We value your feedback! Please let us know how we can improve.")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Email Address")
        rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5, 3)
        comment = st.text_area("Comments / Suggestions")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_data = {
                "Name": name,
                "Email": email,
                "Rating": rating,
                "Comment": comment,
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            save_to_csv(feedback_data)

            # Set flag to redirect
            st.session_state["redirect_to_dashboard"] = True
            st.success("✅ Thank you for your feedback! Redirecting to dashboard...")
            st.experimental_rerun()
