import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="AgroShift - NASA Farm Advisor", layout="wide")

st.title("🌾 AgroShift: AI Climate & Soil Moisture Advisor")
st.markdown("Powered by NASA POWER Agroclimatology Data | Predicting Soil Moisture Risk & Irrigation Needs")

# Sidebar Configuration
st.sidebar.header("📍 Farm Coordinates")
lat = st.sidebar.number_input("Latitude", value=23.8103, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=90.4125, format="%.4f")

@st.cache_data
def get_data(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,GWETTOP,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": "20230101",
        "end": "20231231",
        "format": "JSON"
    }
    res = requests.get(url, params=params).json()
    p = res['properties']['parameter']
    df = pd.DataFrame({
        'Temperature (°C)': p['T2M'],
        'Rainfall (mm)': p['PRECTOTCORR'],
        'SoilMoisture': p['GWETTOP'],
        'Humidity (%)': p['RH2M']
    })
    df.index = pd.to_datetime(df.index)
    df.replace(-999, np.nan, inplace=True)
    df.ffill(inplace=True)
    return df

with st.spinner("Fetching Satellite Data from NASA API..."):
    df = get_data(lat, lon)

# Current Metrics
col1, col2, col3, col4 = st.columns(4)
latest = df.iloc[-1]

col1.metric("Current Temperature", f"{latest['Temperature (°C)']:.1f} °C")
col2.metric("Precipitation", f"{latest['Rainfall (mm)']:.1f} mm")
col3.metric("Soil Moisture", f"{latest['SoilMoisture']*100:.1f}%")
col4.metric("Relative Humidity", f"{latest['Humidity (%)']:.1f}%")

st.divider()

# ML Processing & Prediction
df['Target'] = (df['SoilMoisture'].shift(-1) < 0.35).astype(int)
df_clean = df.dropna()

features = ['Temperature (°C)', 'Rainfall (mm)', 'SoilMoisture', 'Humidity (%)']
X = df_clean[features]
y = df_clean['Target']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

prob = model.predict_proba([latest[features]])[0][1]

# Display Prediction & Advisory
st.subheader("🚨 AI Drought Stress & Soil Moisture Prediction")
col_a, col_b = st.columns([1, 2])

with col_a:
    st.metric("Predicted Drought Risk Score", f"{prob * 100:.1f}%")

with col_b:
    if prob >= 0.5:
        st.error("⚠️ **High Risk Alert:** Soil moisture is critically low. Immediate irrigation is recommended to prevent crop damage.")
    else:
        st.success("✅ **Optimal Condition:** Soil moisture is within safe levels. No additional irrigation required today.")

st.divider()

# Trend Visualizations
st.subheader("📊 Historical Trends: Soil Moisture vs Rainfall")
st.line_chart(df[['SoilMoisture', 'Rainfall (mm)']].tail(60))
