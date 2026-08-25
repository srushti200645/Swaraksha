import math
import re
import sqlite3
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

# --- Page Setup & Strict High-Contrast Dark Theme Override ---
st.set_page_config(
    page_title="Swaraksha - Women Emergency & Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    .title-text { 
        font-size: 30px; 
        font-weight: 900; 
        color: #38bdf8 !important; 
        margin-bottom: 5px;
    }
    .sub-text {
        color: #94a3b8 !important;
        font-size: 14px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #f3f4f6 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b !important;
        border: 2px solid #3b82f6 !important;
        padding: 12px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    .safety-card {
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .safe-high { background-color: #15803d; color: white; border: 1px solid #22c55e; }
    .safe-mod { background-color: #b45309; color: white; border: 1px solid #f59e0b; }
    .safe-low { background-color: #b91c1c; color: white; border: 1px solid #ef4444; }
    h1, h2, h3, h4, label, p { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

# App Title Header
st.markdown('<div class="title-text">🛡️ Swaraksha - Women Safety & Emergency Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Real-time emergency finder, safety zone indicators, and AI predictive safety analytics.</div>', unsafe_allow_html=True)

DEFAULT_LAT = 19.0330  # Default: Navi Mumbai
DEFAULT_LON = 73.0297
USER_AGENT = "Swaraksha-Safety-Dashboard/6.0"

# Pre-defined Offline Safety Infrastructure Data
DEFAULT_FACILITIES = [
    {"name": "Navi Mumbai Police Commissionerate", "type": "Police Station", "color": "red", "icon": "shield", "lat": 19.0350, "lon": 73.0290, "phone": "022-27572236"},
    {"name": "CBD Belapur Police Station", "type": "Police Station", "color": "red", "icon": "shield", "lat": 19.0250, "lon": 73.0380, "phone": "022-27574585"},
    {"name": "Vashi Police Station", "type": "Police Station", "color": "red", "icon": "shield", "lat": 19.0770, "lon": 72.9980, "phone": "022-27821100"},
    {"name": "Apollo Hospitals Navi Mumbai", "type": "Hospital / Clinic", "color": "green", "icon": "plus", "lat": 19.0210, "lon": 73.0390, "phone": "022-33503350"},
    {"name": "MGM Hospital Belapur", "type": "Hospital / Clinic", "color": "green", "icon": "plus", "lat": 19.0385, "lon": 73.0185, "phone": "022-61524242"},
    {"name": "Fortis Hiranandani Hospital", "type": "Hospital / Clinic", "color": "green", "icon": "plus", "lat": 19.0740, "lon": 72.9960, "phone": "022-39199222"},
    {"name": "One Stop Centre (Sakhi) Women Support", "type": "Women & Child Support", "color": "purple", "icon": "heart", "lat": 19.0410, "lon": 73.0210, "phone": "181 / 1091"},
    {"name": "District Legal Services Authority (DLSA)", "type": "Legal Support", "color": "orange", "icon": "briefcase", "lat": 19.0310, "lon": 73.0260, "phone": "15100"},
]

# Utility Functions
def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    a = max(0.0, min(1.0, a))
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_india(query: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{query}, India", "format": "jsonv2", "limit": 1, "countrycodes": "in"}
    try:
        res = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=8)
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None, None, None

def fetch_live_facilities(lat: float, lon: float, radius_km: float):
    results = []
    for item in DEFAULT_FACILITIES:
        dist = haversine_km(lat, lon, item["lat"], item["lon"])
        if dist <= radius_km * 2:
            ic = item.copy()
            ic["distance_km"] = round(dist, 2)
            results.append(ic)
            
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:10];
        (
          node["amenity"="police"](around:{int(radius_km*1000)},{lat},{lon});
          node["amenity"="hospital"](around:{int(radius_km*1000)},{lat},{lon});
          node["amenity"="clinic"](around:{int(radius_km*1000)},{lat},{lon});
          node["office"="lawyer"](around:{int(radius_km*1000)},{lat},{lon});
        );
        out tags center;
        """
        res = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=8)
        if res.status_code == 200:
            for elem in res.json().get("elements", []):
                tags = elem.get("tags", {})
                e_lat = elem.get("lat") or elem.get("center", {}).get("lat")
                e_lon = elem.get("lon") or elem.get("center", {}).get("lon")
                if e_lat and e_lon:
                    amenity = tags.get("amenity", "").lower()
                    office = tags.get("office", "").lower()
                    
                    if amenity == "police":
                        kind, color, icon = "Police Station", "red", "shield"
                    elif amenity in ["hospital", "clinic"]:
                        kind, color, icon = "Hospital / Clinic", "green", "plus"
                    elif office == "lawyer":
                        kind, color, icon = "Legal Support", "orange", "briefcase"
                    else:
                        kind, color, icon = "Support Center", "purple", "heart"
                        
                    name = tags.get("name") or f"Local {kind}"
                    dist = haversine_km(lat, lon, float(e_lat), float(e_lon))
                    results.append({
                        "name": name, "type": kind, "color": color, "icon": icon,
                        "lat": float(e_lat), "lon": float(e_lon),
                        "distance_km": round(dist, 2),
                        "phone": tags.get("phone") or "N/A"
                    })
    except Exception:
        pass

    unique_res = {f["name"]: f for f in results}.values()
    return sorted(list(unique_res), key=lambda x: x["distance_km"])

# Dummy Historical Safety & Crime Dataset Generator for ML Prediction
@st.cache_data
def get_historical_crime_data():
    years = np.arange(2014, 2024)
    np.random.seed(42)
    reported_cases = np.array([120, 115, 130, 125, 140, 135, 110, 105, 95, 88]) 
    return pd.DataFrame({"Year": years, "Reported_Incidents": reported_cases})

# Sidebar Controls
st.sidebar.header("📍 Location Settings")
search_query = st.sidebar.text_input("Enter City / Area:", value="Vashi")
radius_km = st.sidebar.slider("Search Radius (km):", min_value=1, max_value=20, value=5)

if "current_lat" not in st.session_state:
    st.session_state["current_lat"] = DEFAULT_LAT
    st.session_state["current_lon"] = DEFAULT_LON

if st.sidebar.button("🔎 Search Location", use_container_width=True):
    lat, lon, display_name = geocode_india(search_query)
    if lat and lon:
        st.session_state["current_lat"] = lat
        st.session_state["current_lon"] = lon
        st.sidebar.success("Location Updated!")
    else:
        st.sidebar.error("Location not found. Showing default location.")

cur_lat = st.session_state["current_lat"]
cur_lon = st.session_state["current_lon"]

facilities = fetch_live_facilities(cur_lat, cur_lon, radius_km)

# Calculate Safety Score & Zone (Based on Available Safety Resources)
police_count = sum(1 for f in facilities if f["type"] == "Police Station")
hospital_count = sum(1 for f in facilities if f["type"] == "Hospital / Clinic")
women_support_count = sum(1 for f in facilities if f["type"] == "Women & Child Support")
legal_count = sum(1 for f in facilities if f["type"] == "Legal Support")

total_infrastructure = len(facilities)

if total_infrastructure >= 6 and police_count >= 2:
    safety_level = "HIGH SAFETY ZONE 🟢"
    safety_class = "safe-high"
    safety_desc = "High density of safety factors, medical infrastructure, and active police stations."
elif total_infrastructure >= 3:
    safety_level = "MODERATE SAFETY ZONE 🟡"
    safety_class = "safe-mod"
    safety_desc = "Moderate safety factors available. Basic emergency services accessible."
else:
    safety_level = "LOW SAFETY / HIGH CAUTION ZONE 🔴"
    safety_class = "safe-low"
    safety_desc = "Low concentration of nearby mapped safety centers. Exercise extra vigilance."

# Metric Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Active Location", search_query.title())
col2.metric("👮 Police Stations", police_count)
col3.metric("🏥 Hospitals & Clinics", hospital_count)
col4.metric("🟣 Women & Legal Centers", women_support_count + legal_count)

st.markdown("<br>", unsafe_allow_html=True)

# Safety Zone Alert Panel
st.markdown(f"""
<div class="safety-card {safety_class}">
    <h3>Zone Status: {safety_level}</h3>
    <p>{safety_desc}</p>
</div>
""", unsafe_allow_html=True)

# Map Interface
st.subheader("🗺️ High-Contrast Safety Map (Emergency Radius)")

m = folium.Map(location=[cur_lat, cur_lon], zoom_start=13, tiles="CartoDB dark_matter")
Fullscreen(position="topright").add_to(m)

# User's Current Location Marker (Center)
folium.Marker(
    [cur_lat, cur_lon],
    popup="<b>Selected Current Location</b>",
    tooltip="Your Location",
    icon=folium.Icon(color="blue", icon="user", prefix="fa")
).add_to(m)

# Safety Radius Ring (Red/Blue Boundary)
folium.Circle(
    [cur_lat, cur_lon],
    radius=radius_km * 1000,
    color="#ef4444" if "LOW" in safety_level else "#38bdf8",
    weight=3,
    fill=True,
    fill_opacity=0.1
).add_to(m)

# Plotting Safety Infrastructure Factors
for item in facilities:
    folium.Marker(
        [item["lat"], item["lon"]],
        popup=f"<b>{item['name']}</b><br>Type: {item['type']}<br>Distance: {item['distance_km']} km<br>Phone: {item['phone']}",
        tooltip=f"{item['name']} ({item['distance_km']} km)",
        icon=folium.Icon(color=item["color"], icon=item["icon"], prefix="fa")
    ).add_to(m)

st_folium(m, width="100%", height=500)

st.markdown("<br>", unsafe_allow_html=True)

# ML Predictive Safety & Crime Forecasting Dashboard
st.subheader("📊 Historical Safety Analytics & Future AI Trend Prediction")

df_crime = get_historical_crime_data()

# ML Model (Linear Regression Trend)
X = df_crime[["Year"]].values
y = df_crime["Reported_Incidents"].values

model = LinearRegression()
model.fit(X, y)

future_years = np.array([[2024], [2025], [2026]])
future_preds = model.predict(future_years)

df_future = pd.DataFrame({
    "Year": [2024, 2025, 2026],
    "Reported_Incidents": np.clip(future_preds, 0, None),
    "Type": ["Predicted"] * 3
})

df_crime["Type"] = "Historical"
df_combined = pd.concat([df_crime, df_future], ignore_index=True)

# Plot Forecast Graph using Plotly
fig = px.line(
    df_combined, 
    x="Year", 
    y="Reported_Incidents", 
    color="Type",
    markers=True,
    title="Safety Analytics: Past 10-Year Crime Data & 3-Year Future Trend Forecast",
    color_discrete_map={"Historical": "#38bdf8", "Predicted": "#ef4444"}
)
fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Emergency Contact Table
st.subheader("📋 Nearby Emergency Facilities & Helplines Directory")
if facilities:
    df_fac = pd.DataFrame(facilities)[["name", "type", "distance_km", "phone"]]
    df_fac.columns = ["Facility / Support Center", "Category", "Distance (km)", "Contact Number"]
    st.dataframe(df_fac, use_container_width=True)

# Direct Emergency Helpline Numbers
st.markdown("""
<div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #3b82f6;">
    <h4 style="margin: 0; color: #38bdf8;">🚨 National Emergency Response Helplines (India)</h4>
    <p style="margin: 5px 0 0 0; color: #e2e8f0;">
        <b>National Emergency Number:</b> 112 | 
        <b>Women Helpline:</b> 181 / 1091 | 
        <b>Child Helpline:</b> 1098 | 
        <b>Cyber Crime Helpline:</b> 1930
    </p>
</div>
""", unsafe_allow_html=True)