import streamlit as st

def show_alerts(latest):
    st.subheader("🚨 Climate Alert System")
    co2_alert = "⚠️ High CO₂ Level" if latest['CO2(ppm)'] > 420 else "✅ CO₂ Normal"
    temp_alert = "⚠️ High Temperature" if latest['Temperature(°C)'] > 1.2 else "✅ Temperature Normal"

    st.write(f"**Year:** {latest['Year']}")
    st.write(f"**CO₂:** {latest['CO2(ppm)']} ppm → {co2_alert}")
    st.write(f"**Temperature:** {latest['Temperature(°C)']}°C → {temp_alert}")
