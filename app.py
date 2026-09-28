import json
import os
import math
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AgroShift | NASA-Powered Crop Rotation Engine",
    page_icon="🌾",
    layout="wide",
)

# ---------------------------------------------------------
# Load Knowledge Base
# ---------------------------------------------------------
@st.cache_data
def load_crop_database():
    crops_path = os.path.join(os.path.dirname(__file__), "crops.json")
    if not os.path.exists(crops_path):
        st.error(
            "Missing `crops.json`. Make sure the file exists in the application root directory."
        )
        return []
    with open(crops_path, "r", encoding="utf-8") as f:
        return json.load(f)

crop_db = load_crop_database()

# ---------------------------------------------------------
# NASA POWER API Data Retrieval
# ---------------------------------------------------------
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_environmental_data(lat: float, lon: float):
    end_date = datetime.now(timezone.utc) - timedelta(days=4)
    start_date = end_date - timedelta(days=89)  # Inclusive 90-day window

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M,GWETTOP",
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start_str,
        "end": end_str,
        "format": "JSON",
    }

    try:
        response = requests.get(NASA_POWER_BASE, params=params, timeout=12)
        if response.status_code != 200:
            return None, f"NASA POWER API returned HTTP status {response.status_code}."

        data = response.json()
        properties = data.get("properties", {}).get("parameter", {})

        t2m = properties.get("T2M", {})
        prec = properties.get("PRECTOTCORR", {})
        rh2m = properties.get("RH2M", {})
        gwettop = properties.get("GWETTOP", {})

        dates = sorted(t2m.keys())
        if not dates:
            return None, "No environmental time series returned for these coordinates."

        records = []
        for d in dates:
            val_t = t2m.get(d, -999)
            val_p = prec.get(d, -999)
            val_rh = rh2m.get(d, -999)
            val_gw = gwettop.get(d, -999)

            values = (val_t, val_p, val_rh, val_gw)
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v == -999 for v in values):
                continue

            records.append(
                {
                    "Date": datetime.strptime(d, "%Y%m%d"),
                    "Temperature_C": val_t,
                    "Precipitation_mm": val_p,
                    "Relative_Humidity_pct": val_rh,
                    "Surface_Soil_Wetness_0_1": val_gw,
                }
            )

        if not records:
            return None, "The coordinates returned incomplete or fill-value NASA data."

        df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
        date_range_str = f"{df['Date'].iloc[0].strftime('%b %d, %Y')} to {df['Date'].iloc[-1].strftime('%b %d, %Y')}"
        return (df, date_range_str), None

    except requests.exceptions.Timeout:
        return None, "NASA POWER API request timed out."
    except Exception as e:
        return None, f"Network or parsing error: {str(e)}"

# ---------------------------------------------------------
# Recommendation Engine: Transparent Multi-Factor Scoring
# ---------------------------------------------------------
def normalize_crop_name(name: str):
    """Match database IDs, full display names, and short sidebar names."""
    return "".join(name.split(" (")[0].split(" / ")[0].casefold().split())


def resolve_crop(crops: list, name: str):
    key = normalize_crop_name(name)
    return next(
        (crop for crop in crops if key in
         (normalize_crop_name(crop["id"]), normalize_crop_name(crop["name"]))),
        None,
    )


def compute_rotation_recommendations(
    crops: list,
    current_crop_name: str,
    soil_type: str,
    priority: str,
    env_summary: dict,
):
    mean_temp = env_summary["mean_temp"]
    total_rain = env_summary["total_rain"]
    mean_wetness = env_summary["mean_wetness"]

    current = resolve_crop(crops, current_crop_name)
    results = []

    for crop in crops:
        # Exclude the current crop before any scoring, including display-name aliases.
        if current and crop["id"] == current["id"]:
            continue
        score_breakdown = {}

        # 1. Climate Suitability (30 points)
        if crop["temp_min"] <= mean_temp <= crop["temp_max"]:
            temp_sub = 15.0
        else:
            diff = min(
                abs(mean_temp - crop["temp_min"]),
                abs(mean_temp - crop["temp_max"]),
            )
            temp_sub = max(0.0, 15.0 - (diff * 2.0))

        water_req = crop["water_need"]
        if water_req == "Low":
            moist_sub = 15.0 if mean_wetness < 0.6 else 10.0
        elif water_req == "Medium":
            moist_sub = 15.0 if 0.35 <= mean_wetness <= 0.8 else 10.0
        else:
            moist_sub = 15.0 if mean_wetness > 0.6 or total_rain > 120 else 6.0

        score_breakdown["Climate Suitability"] = round(temp_sub + moist_sub, 1)

        # 2. Soil Compatibility (25 points)
        if soil_type in crop["preferred_soils"]:
            score_breakdown["Soil Compatibility"] = 25.0
        elif any(s in soil_type for s in ["Loam", "Sandy"]):
            score_breakdown["Soil Compatibility"] = 17.0
        else:
            score_breakdown["Soil Compatibility"] = 10.0

        # 3. Rotation Fit (25 points)
        same_family = current is not None and crop["family"] == current["family"]
        is_incompatible = any(
            resolve_crop(crops, name) == current if current is not None
            else normalize_crop_name(name) == normalize_crop_name(current_crop_name)
            for name in crop.get("incompatible_preceding", [])
        )

        if same_family or is_incompatible:
            rotation_score = 4.0
        elif current and current["family"] == "Poaceae" and crop["is_legume"]:
            rotation_score = 25.0
        elif current and current["is_legume"] and not crop["is_legume"]:
            rotation_score = 24.0
        else:
            rotation_score = 18.0

        score_breakdown["Rotation Fit"] = float(rotation_score)

        # 4. Priority Alignment (20 points)
        p_score = 10.0
        if priority == "Water Conservation":
            p_score = 20.0 if crop["water_need"] == "Low" else (14.0 if crop["water_need"] == "Medium" else 5.0)
        elif priority == "Soil Health Restoration":
            p_score = 20.0 if crop["is_legume"] else (16.0 if "Biofumigation" in crop["soil_benefit"] else 8.0)
        elif priority == "Yield Potential":
            p_score = 20.0 if crop["yield_potential"].startswith("High") or crop["yield_potential"].startswith("Very high") else 14.0
        elif priority == "Climate & Drought Resilience":
            p_score = 20.0 if "drought" in crop["climate_resilience"].lower() else 12.0

        score_breakdown["Priority Alignment"] = float(p_score)

        total_score = sum(score_breakdown.values())
        results.append(
            {
                "crop": crop,
                "total_score": round(min(total_score, 100.0), 1),
                "breakdown": score_breakdown,
            }
        )

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results

def format_coordinates(lat: float, lon: float):
    return f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("🌍 Farm Profile & Settings")

coord_options = {
    "Dhaka Division, BD (23.81, 90.41)": (23.8103, 90.4125),
    "Rajshahi (Barind / Drought Prone, BD) (24.37, 88.60)": (24.3745, 88.6042),
    "Nebraska Corn Belt, US (41.49, -99.90)": (41.4925, -99.9018),
    "Punjab Agricultural Zone, IN (30.90, 75.85)": (30.9010, 75.8573),
    "Custom Coordinates": None,
}

selected_preset = st.sidebar.selectbox("Select Farm Region", list(coord_options.keys()))

if selected_preset == "Custom Coordinates":
    lat = st.sidebar.number_input("Latitude (-90 to +90)", value=23.81, min_value=-90.0, max_value=90.0, step=0.01)
    lon = st.sidebar.number_input("Longitude (-180 to +180)", value=90.41, min_value=-180.0, max_value=180.0, step=0.01)
else:
    lat, lon = coord_options[selected_preset]
    st.sidebar.caption(f"Coordinates: **{format_coordinates(lat, lon)}**")

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Current Field Agronomy")

current_crop = st.sidebar.selectbox(
    "Current Standing Crop",
    ["Rice", "Wheat", "Maize", "Mustard", "Lentil", "Fallow (Bare Soil)"],
    index=0,
)

soil_type = st.sidebar.selectbox(
    "Topsoil Classification",
    ["Loam", "Clay Loam", "Sandy Loam", "Silt Loam", "Clay"],
    index=0,
)

priority = st.sidebar.selectbox(
    "Farmer Operational Priority",
    [
        "Water Conservation",
        "Soil Health Restoration",
        "Yield Potential",
        "Climate & Drought Resilience",
    ],
    index=0,
)

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("AgroShift")
st.markdown(
    "### NASA-Assisted Crop-Rotation Decision Support System\n"
    "*Supporting next-crop choices using recent climate, soil characteristics, and crop rotation.*"
)

with st.spinner("Connecting to NASA POWER Agroclimatology API..."):
    data_tuple, err = fetch_nasa_environmental_data(lat, lon)

if err or not data_tuple:
    st.error("⚠ NASA POWER DATA UNAVAILABLE — recommendations are paused.")
    st.warning(err or "No environmental data were returned.")
    st.info("No synthetic observations are generated. Retry the NASA connection to obtain recommendations.")
    if st.button("Retry NASA POWER"):
        fetch_nasa_environmental_data.clear()
        st.rerun()
    st.stop()

df_env, date_label = data_tuple

recent_window = df_env[df_env["Date"] >= df_env["Date"].max() - timedelta(days=29)]
if len(recent_window) < 30:
    st.error("NASA POWER returned incomplete data for the latest 30-day period. Recommendations are paused.")
    if st.button("Retry NASA POWER"):
        fetch_nasa_environmental_data.clear()
        st.rerun()
    st.stop()
mean_temp = recent_window["Temperature_C"].mean()
total_rain = recent_window["Precipitation_mm"].sum()
mean_rh = recent_window["Relative_Humidity_pct"].mean()
mean_wetness = recent_window["Surface_Soil_Wetness_0_1"].mean()

env_summary = {
    "mean_temp": mean_temp,
    "total_rain": total_rain,
    "mean_rh": mean_rh,
    "mean_wetness": mean_wetness,
}

# 1. Farm Profile Summary
st.subheader("1. Active Farm Profile")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Standing Crop", current_crop)
col2.metric("Soil Texture", soil_type)
col3.metric("Farmer Primary Goal", priority)
col4.metric("Coordinates", format_coordinates(lat, lon))

# 2. Environmental Conditions
st.subheader("2. NASA POWER Regional Environmental Context")
st.caption(f"Data Source: **NASA POWER (Daily Agroclimatology)** | Observation Window: **{date_label}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recent Mean Temp (30-day)", f"{mean_temp:.1f} °C")
m2.metric("Cumulative Rain (30-day)", f"{total_rain:.1f} mm")
m3.metric("Relative Humidity", f"{mean_rh:.1f} %")
m4.metric("Top-Soil Wetness (0-1)", f"{mean_wetness:.2f}")

st.info(
    "**Measurement Clarification:** NASA POWER meteorological values are gridded reanalysis estimates (approximately 0.5° × 0.625°) "
    "providing regional environmental context, rather than field-level physical sensors.",
    icon="🛰️",
)

with st.expander("View 90-Day Climate & Soil Wetness History"):
    st.line_chart(
        df_env.set_index("Date")[
            [
                "Temperature_C",
                "Relative_Humidity_pct",
                "Surface_Soil_Wetness_0_1",
            ]
        ]
    )

# 3. Crop-Rotation Recommendations
st.subheader("3. Top Recommended Next Crops for Rotation")
st.caption("Heuristic score out of 100: climate 30 points, soil 25, rotation 25, farmer priority 20. "
           "This is not a probability or a forecast of yield or profit.")
ranked_crops = compute_rotation_recommendations(crop_db, current_crop, soil_type, priority, env_summary)
top_3 = ranked_crops[:3]

cols = st.columns(3)
for i, item in enumerate(top_3):
    crop = item["crop"]
    score = item["total_score"]
    breakdown = item["breakdown"]

    with cols[i]:
        st.markdown(f"#### #{i+1}. {crop['name']}")
        st.metric(label="Suitability Score", value=f"{score:g}/100")
        st.progress(score / 100.0)

        st.markdown("**Suitability Breakdown:**")
        for k, v in breakdown.items():
            st.write(f"- {k}: **{v} pts**")

        st.markdown(f"**Biological Benefit:**\n*{crop['soil_benefit']}*")
        st.markdown(f"**Climate Profile:**\n{crop['climate_resilience']}")

st.markdown("---")
st.subheader("Top Next-Crop Alternatives")
st.caption(f"Each option is scored as the next crop after {current_crop}. These are alternatives, not a multi-season sequence.")
for item in top_3:
    st.write(f"- {item['crop']['name']}")
