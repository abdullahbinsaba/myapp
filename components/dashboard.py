import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from enum import Enum

# Custom color palette
COLOR_PALETTE = {
    "primary": "#3498db",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "dark": "#2c3e50",
    "light": "#ecf0f1"
}

# Configure default matplotlib style
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    COLOR_PALETTE["primary"], 
    COLOR_PALETTE["success"], 
    COLOR_PALETTE["warning"],
    COLOR_PALETTE["danger"]
])

def load_data():
    """Load and prepare climate data"""
    df = pd.read_csv("data/climate_data_2000_2024.csv")
    df['Year'] = df['Year'].astype(int)
    
    # Add simulated region/anomaly data if not present
    if 'Region' not in df.columns:
        regions = ['North', 'South', 'East', 'West', 'Central']
        df['Region'] = pd.Series([regions[i%5] for i in range(len(df))])
    
    if 'Anomaly' not in df.columns:
        anomalies = ['None', 'Heatwave', 'Flood', 'Drought', 'Cyclone']
        df['Anomaly'] = pd.Series([anomalies[i%5] for i in range(len(df))])
    
    return df

class DashboardPage(Enum):
    DASHBOARD = "Dashboard"
    LIVE_DATA = "Live Data"
    ANALYTICS = "Advanced Analytics"
    PREDICTIONS = "AI Predictions"
    REGIONS = "Regional Insights"
    FEEDBACK = "Feedback"
    

def show_dashboard():
    """Main dashboard function"""
    df = load_data()
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = DashboardPage.DASHBOARD.value
    
    # Custom sidebar design
    with st.sidebar:
        st.markdown(f"""
            <div style="background:{COLOR_PALETTE['dark']}; padding:15px; border-radius:10px">
                <h1 style="color:white; text-align:center;">🌎 EarthScape</h1>
                <p style="color:{COLOR_PALETTE['light']}; text-align:center;">Climate Analytics Suite</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Navigation")
        
        # Navigation buttons with icons
        nav_options = {
            DashboardPage.DASHBOARD.value: "📊",
            DashboardPage.LIVE_DATA.value: "🌡️",
            DashboardPage.ANALYTICS.value: "📈",
            DashboardPage.PREDICTIONS.value: "🤖",
            DashboardPage.REGIONS.value: "🗺️",
            DashboardPage.FEEDBACK.value: "📝"
        }
        
        for page, icon in nav_options.items():
            if st.button(f"{icon} {page}", use_container_width=True, 
                        key=f"nav_{page}"):
                st.session_state.current_page = page
        
        st.markdown("---")
        st.markdown(f"""
            <p style="color:{COLOR_PALETTE['dark']}; font-size:small;">
            Data: 2000-2024 • v2.1.0
            </p>
        """, unsafe_allow_html=True)
    
    # Page content
    st.markdown(f"""
        <h1 style="color:{COLOR_PALETTE['dark']}; border-bottom:2px solid {COLOR_PALETTE['primary']}; padding-bottom:10px">
            {st.session_state.current_page}
        </h1>
    """, unsafe_allow_html=True)
    
    # Router
    if st.session_state.current_page == DashboardPage.DASHBOARD.value:
        show_overview_page(df)
    elif st.session_state.current_page == DashboardPage.LIVE_DATA.value:
        show_live_monitoring_page(df)
    elif st.session_state.current_page == DashboardPage.ANALYTICS.value:
        show_analytics_page(df)
    elif st.session_state.current_page == DashboardPage.PREDICTIONS.value:
        show_predictions_page(df)
    elif st.session_state.current_page == DashboardPage.REGIONS.value:
        show_regional_page(df)
    elif st.session_state.current_page == DashboardPage.FEEDBACK.value:
         show_feedback_page() 
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style="background:{COLOR_PALETTE['light']}; padding:10px; border-radius:5px; text-align:center">
            <p style="color:{COLOR_PALETTE['dark']}">
            © 2025 EarthScape Climate Intelligence | 
            <span style="color:{COLOR_PALETTE['primary']}">Powered by Streamlit</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ====================== PAGE COMPONENTS ======================

def show_overview_page(df):
    """Main dashboard overview"""
    # Key Metrics Row
    st.markdown("### 🌍 Climate at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df.iloc[-1]
    with col1:
        st.metric("Current Year", f"{int(latest['Year'])}", 
                 help="Most recent data year")
    with col2:
        st.metric("CO₂ Level", f"{latest['CO2(ppm)']:.1f} ppm", 
                 delta=f"{df['CO2(ppm)'].iloc[-1] - df['CO2(ppm)'].iloc[0]:.1f} since 2000",
                 help="Atmospheric CO₂ concentration")
    with col3:
        st.metric("Temperature", f"{latest['Temperature(C)']:.2f}°C", 
                 delta=f"{df['Temperature(C)'].iloc[-1] - df['Temperature(C)'].iloc[0]:.2f}°C since 2000",
                 delta_color="inverse",
                 help="Global temperature anomaly")
    with col4:
        st.metric("Data Points", len(df), 
                 help="Total observations in dataset")
    
    # Climate Trends
    st.markdown("### 📈 Climate Trends")
    tab1, tab2 = st.tabs(["Dual Axis View", "Individual Trends"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x='Year', y='CO2(ppm)', ax=ax, 
                    color=COLOR_PALETTE['primary'], label='CO₂ (ppm)')
        ax2 = ax.twinx()
        sns.lineplot(data=df, x='Year', y='Temperature(C)', ax=ax2,
                    color=COLOR_PALETTE['danger'], label='Temp (°C)')
        ax.set_title('CO₂ and Temperature Trends (2000-2024)')
        ax.figure.legend(loc='upper left', bbox_to_anchor=(0.15, 0.9))
        st.pyplot(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(data=df, x='Year', y='CO2(ppm)', 
                        color=COLOR_PALETTE['primary'])
            plt.title('CO₂ Concentration Over Time')
            st.pyplot(fig)
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.lineplot(data=df, x='Year', y='Temperature(C)',
                        color=COLOR_PALETTE['danger'])
            plt.title('Temperature Anomalies Over Time')
            st.pyplot(fig)
    
    # Data Preview
    with st.expander("🔍 Explore Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True)

def show_live_monitoring_page(df):
    """Animated live monitoring page"""
    st.markdown("""
        <div style="background:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px">
            <h3 style="color:#2c3e50;">🌐 Real-time Climate Monitoring</h3>
            <p style="color:#7f8c8d;">Simulated year-by-year progression of key metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # CO₂ Animation
    st.markdown("#### 🟢 CO₂ Concentration Live Feed")
    co2_placeholder = st.empty()
    
    if st.button("Play CO₂ Animation", key="co2_animate"):
        streamed_data = pd.DataFrame(columns=df.columns)
        for year in range(2000, 2025):
            new_data = df[df['Year'] == year]
            streamed_data = pd.concat([streamed_data, new_data])
            avg_co2 = streamed_data.groupby('Year')['CO2(ppm)'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=avg_co2, x='Year', y='CO2(ppm)', marker='o',
                        color=COLOR_PALETTE['success'], ax=ax)
            ax.set_title(f'Live CO₂ Monitoring (2000-{year})')
            ax.set_ylabel('CO₂ (ppm)')
            ax.grid(True)
            
            co2_placeholder.pyplot(fig)
            time.sleep(0.3)
    
    # Temperature Animation
    st.markdown("#### 🔴 Temperature Live Feed")
    temp_placeholder = st.empty()
    
    if st.button("Play Temperature Animation", key="temp_animate"):
        streamed_data = pd.DataFrame(columns=df.columns)
        for year in range(2000, 2025):
            new_data = df[df['Year'] == year]
            streamed_data = pd.concat([streamed_data, new_data])
            avg_temp = streamed_data.groupby('Year')['Temperature(C)'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=avg_temp, x='Year', y='Temperature(C)', marker='o',
                        color=COLOR_PALETTE['danger'], ax=ax)
            ax.set_title(f'Live Temperature Monitoring (2000-{year})')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True)
            
            temp_placeholder.pyplot(fig)
            time.sleep(0.3)

def show_analytics_page(df):
    """Advanced analytics visualizations"""
    st.markdown("### 🔍 Deep Dive Analytics")
    
    # Correlation Analysis
    st.markdown("#### 📊 Correlation Matrix")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Climate Variables Correlation')
    st.pyplot(fig)
    
    # Anomaly Analysis
    st.markdown("#### ⚠️ Climate Anomalies")
    col1, col2 = st.columns(2)
    
    with col1:
        anomaly_counts = df['Anomaly'].value_counts()
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values,
                   palette=[COLOR_PALETTE['danger'], '#f39c12', '#e67e22', '#d35400'])
        plt.title('Anomaly Frequency')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Anomaly', y='Temperature(C)',
                   palette=[COLOR_PALETTE['light'], '#f39c12', '#e67e22', '#d35400'])
        plt.title('Temperature Distribution by Anomaly')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Temporal Heatmap
    st.markdown("#### 🗓️ Annual Patterns")
    annual_avg = df.groupby('Year').mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(annual_avg.T, cmap='YlOrRd', annot=True, fmt=".1f",
                linewidths=.5, ax=ax)
    ax.set_title('Yearly Climate Indicators')
    st.pyplot(fig)

def show_predictions_page(df):
    """Machine learning predictions"""
    st.markdown("### 🤖 AI-Powered Climate Predictions")
    
    # Model Training
    st.markdown("#### 📉 CO₂ Prediction Model")
    X = df[['Year']]
    y = df['CO2(ppm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Model Performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy (MSE)", f"{mse:.2f}")
    with col2:
        st.metric("R² Score", f"{model.score(X_test, y_test):.2f}")
    
    # Prediction Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_test, y_test, color=COLOR_PALETTE['primary'], label='Actual')
    ax.plot(X_test, y_pred, color=COLOR_PALETTE['danger'], linewidth=2, label='Predicted')
    ax.set_title('CO₂ Level Predictions')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO₂ (ppm)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Future Projections
    st.markdown("#### 🔮 Future Projections (2025-2030)")
    future_years = pd.DataFrame({'Year': range(2025, 2031)})
    future_pred = model.predict(future_years)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=future_years['Year'], y=future_pred, 
                marker='o', color=COLOR_PALETTE['success'], ax=ax)
    ax.set_title('Projected CO₂ Levels')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO₂ (ppm)')
    ax.grid(True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        for year, pred in zip(future_years['Year'], future_pred):
            st.metric(f"Year {year}", f"{pred:.1f} ppm")

def show_regional_page(df):
    """Regional climate analysis"""
    st.markdown("### 🌎 Regional Climate Patterns")
    
    # Rainfall Distribution
    st.markdown("#### 🌧️ Regional Rainfall")
    rain_by_region = df.groupby('Region')['Rainfall(mm)'].sum().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=rain_by_region.values, y=rain_by_region.index,
               palette='Blues_d', ax=ax)
    ax.set_title('Total Rainfall by Region (2000-2024)')
    ax.set_xlabel('Rainfall (mm)')
    st.pyplot(fig)
    
    # Regional Temperature Trends
    st.markdown("#### 🌡️ Regional Temperature Trends")
    region = st.selectbox("Select Region", df['Region'].unique())
    
    regional_data = df[df['Region'] == region]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=regional_data, x='Year', y='Temperature(C)',
                color=COLOR_PALETTE['danger'], ax=ax)
    ax.set_title(f'Temperature Trends in {region} Region')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True)
    st.pyplot(fig)
    
    # Regional Comparison
    st.markdown("#### ↔️ Regional Comparison")
    metrics = st.selectbox("Select Metric", ['Temperature(C)', 'Rainfall(mm)', 'CO2(ppm)'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Region', y=metrics,
               palette='viridis', ax=ax)
    ax.set_title(f'Regional Distribution of {metrics}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
def show_feedback_page():
    st.markdown("### 📝 Feedback")
    
    name = st.text_input("Name")
    email = st.text_input("Email")
    rating = st.slider("Rate your experience (1 = Bad, 5 = Great)", 1, 5, 3)
    comments = st.text_area("Additional Comments")

    if st.button("Submit Feedback"):
        if name and email:
            feedback_data = pd.DataFrame([{
                "Name": name,
                "Email": email,
                "Rating": rating,
                "Comments": comments
            }])
            feedback_data.to_csv("data/feedback.csv", mode='a', header=False, index=False)
            st.success("✅ Thank you for your feedback!")
            st.switch_page("components/dashboard.py")  # Streamlit >=1.25
        else:
            st.warning("Please provide both Name and Email.")

if __name__ == "__main__":
    show_dashboard()
