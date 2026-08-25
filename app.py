import math
import sqlite3
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# --- Page Setup & High-Contrast Dark Theme ---
st.set_page_config(
    page_title="Swaraksha - Pan-India Emergency & Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling
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
st.markdown('<div class="title-text">🛡️ Swaraksha - Pan-India Women Safety & Emergency Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Search any city, locality, or landmark across India for real-time safety infrastructure.</div>', unsafe_allow_html=True)

DEFAULT_LAT = 19.0330  # Initial Fallback Center: Navi Mumbai
DEFAULT_LON = 73.0297
USER_AGENT = "Swaraksha-PanIndia-Safety/8.0"

INITIAL_NAVI_MUMBAI = [
    {"name": "Navi Mumbai Police Commissionerate", "type": "Police Station", "color": "red", "icon": "shield", "lat": 19.0350, "lon": 73.0290, "phone": "022-27572236"},
    {"name": "CBD Belapur Police Station", "type": "Police Station", "color": "red", "icon": "shield", "lat": 19.0250, "lon": 73.0380, "phone": "022-27574585"},
    {"name": "Apollo Hospitals Navi Mumbai", "type": "Hospital / Clinic", "color": "green", "icon": "plus", "lat": 19.0210, "lon": 73.0390, "phone": "022-33503350"},
    {"name": "MGM Hospital Belapur", "type": "Hospital / Clinic", "color": "green", "icon": "plus", "lat": 19.0385, "lon": 73.0185, "phone": "022-61524242"},
    {"name": "One Stop Centre (Sakhi) Women Support", "type": "Women & Child Support", "color": "purple", "icon": "heart", "lat": 19.0410, "lon": 73.0210, "phone": "181 / 1091"},
    {"name": "District Legal Services Authority (DLSA)", "type": "Legal Support", "color": "orange", "icon": "briefcase", "lat": 19.0310, "lon": 73.0260, "phone": "15100"},
]

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    a = max(0.0, min(1.0, a))
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# Pan-India Nominatim Geocoding API
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_india(query: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{query}, India", "format": "jsonv2", "limit": 1, "countrycodes": "in"}
    try:
        res = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
        data = res.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None, None, None

# Comprehensive Overpass Query with Endpoint Fallbacks
@st.cache_data(ttl=600, show_spinner=False)
def fetch_pan_india_facilities(lat: float, lon: float, radius_km: float):
    results = []
    radius_meters = int(radius_km * 1000)

    query = f"""
    [out:json][timeout:25];
    (
      nwr["amenity"="police"](around:{radius_meters},{lat},{lon});
      nwr["amenity"="hospital"](around:{radius_meters},{lat},{lon});
      nwr["amenity"="clinic"](around:{radius_meters},{lat},{lon});
      nwr["amenity"="doctors"](around:{radius_meters},{lat},{lon});
      nwr["office"="lawyer"](around:{radius_meters},{lat},{lon});
      nwr["amenity"="social_facility"](around:{radius_meters},{lat},{lon});
      nwr["social_facility"](around:{radius_meters},{lat},{lon});
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    ]

    for endpoint in endpoints:
        try:
            res = requests.post(endpoint, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
            if res.status_code == 200:
                elements = res.json().get("elements", [])
                for elem in elements:
                    tags = elem.get("tags", {})
                    e_lat = elem.get("lat") or (elem.get("center", {}).get("lat") if "center" in elem else None)
                    e_lon = elem.get("lon") or (elem.get("center", {}).get("lon") if "center" in elem else None)

                    if e_lat is not None and e_lon is not None:
                        amenity = tags.get("amenity", "").lower()
                        office = tags.get("office", "").lower()

                        if amenity == "police":
                            kind, color, icon = "Police Station", "red", "shield"
                        elif amenity in ["hospital", "clinic", "doctors"]:
                            kind, color, icon = "Hospital / Clinic", "green", "plus"
                        elif office == "lawyer":
                            kind, color, icon = "Legal Support", "orange", "briefcase"
                        else:
                            kind, color, icon = "Women & Child Support", "purple", "heart"

                        name = (
                            tags.get("name") 
                            or tags.get("name:en") 
                            or tags.get("official_name") 
                            or f"Local {kind}"
                        )
                        dist = haversine_km(lat, lon, float(e_lat), float(e_lon))
                        phone = (
                            tags.get("phone") 
                            or tags.get("contact:phone") 
                            or tags.get("contact:mobile") 
                            or tags.get("emergency:phone") 
                            or "N/A"
                        )

                        results.append({
                            "name": name,
                            "type": kind,
                            "color": color,
                            "icon": icon,
                            "lat": float(e_lat),
                            "lon": float(e_lon),
                            "distance_km": round(dist, 2),
                            "phone": phone
                        })
                if results:
                    break
        except Exception:
            continue

    # Fallback to hardcoded dataset if live fetch fails around Navi Mumbai
    if not results and abs(lat - DEFAULT_LAT) < 0.05 and abs(lon - DEFAULT_LON) < 0.05:
        for item in INITIAL_NAVI_MUMBAI:
            dist = haversine_km(lat, lon, item["lat"], item["lon"])
            if dist <= radius_km * 2:
                ic = item.copy()
                ic["distance_km"] = round(dist, 2)
                results.append(ic)

    # De-duplicate by coordinate/name pair
    unique_res = {}
    for f in results:
        key = (f["name"], round(f["lat"], 4), round(f["lon"], 4))
        if key not in unique_res:
            unique_res[key] = f

    return sorted(list(unique_res.values()), key=lambda x: x["distance_km"])

# Historical Safety & Crime Dataset Generator
@st.cache_data
def get_historical_crime_data():
    years = np.arange(2014, 2024)
    reported_cases = np.array([120, 115, 130, 125, 140, 135, 110, 105, 95, 88]) 
    return pd.DataFrame({"Year": years, "Reported_Incidents": reported_cases})

# Sidebar Search Controls
st.sidebar.header("📍 Pan-India Location Search")
search_query = st.sidebar.text_input("Enter Any Indian City / Address:", value="Vashi, Navi Mumbai")
radius_km = st.sidebar.slider("Search Radius (km):", min_value=1, max_value=25, value=5)

if "current_lat" not in st.session_state:
    st.session_state["current_lat"] = DEFAULT_LAT
    st.session_state["current_lon"] = DEFAULT_LON
    st.session_state["display_name"] = "Vashi, Navi Mumbai, Maharashtra, India"

if st.sidebar.button("🔎 Search Location", use_container_width=True):
    with st.spinner("Locating across India..."):
        lat, lon, display_name = geocode_india(search_query)
        if lat and lon:
            st.session_state["current_lat"] = lat
            st.session_state["current_lon"] = lon
            st.session_state["display_name"] = display_name
            st.sidebar.success(f"Loaded Location: {search_query.title()}")
        else:
            st.sidebar.error("Location not found! Try adding state name (e.g., 'Connaught Place, Delhi').")

cur_lat = st.session_state["current_lat"]
cur_lon = st.session_state["current_lon"]

# Fetch Live Infrastructure for Searched Coordinates
facilities = fetch_pan_india_facilities(cur_lat, cur_lon, radius_km)

# Calculate Infrastructure Metrics
police_count = sum(1 for f in facilities if f["type"] == "Police Station")
hospital_count = sum(1 for f in facilities if f["type"] == "Hospital / Clinic")
women_support_count = sum(1 for f in facilities if f["type"] == "Women & Child Support")
legal_count = sum(1 for f in facilities if f["type"] == "Legal Support")

total_infrastructure = len(facilities)

# Determine Zone Safety Rating dynamically
if total_infrastructure >= 5 and police_count >= 1:
    safety_level = "HIGH SAFETY ZONE 🟢"
    safety_class = "safe-high"
    safety_desc = "Good availability of police stations, emergency hospitals, and support infrastructure."
elif total_infrastructure >= 2:
    safety_level = "MODERATE SAFETY ZONE 🟡"
    safety_class = "safe-mod"
    safety_desc = "Basic emergency facilities available within radius. Exercise standard safety awareness."
else:
    safety_level = "LOW SAFETY / HIGH CAUTION ZONE 🔴"
    safety_class = "safe-low"
    safety_desc = "Few mapped safety centers found in this radius. Keep emergency contact numbers handy."

# Metric Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Active Area", search_query.title())
col2.metric("👮 Police Stations", police_count)
col3.metric("🏥 Hospitals & Clinics", hospital_count)
col4.metric("🟣 Support & Legal Aid", women_support_count + legal_count)

st.markdown("<br>", unsafe_allow_html=True)

# Safety Zone Alert Panel
st.markdown(f"""
<div class="safety-card {safety_class}">
    <h3>Zone Status: {safety_level}</h3>
    <p>{safety_desc}</p>
</div>
""", unsafe_allow_html=True)

# Dynamic Leaflet / OpenStreetMap Display
st.subheader("🗺️ Live All-India Interactive Safety Map")

m = folium.Map(location=[cur_lat, cur_lon], zoom_start=13, tiles="CartoDB dark_matter")
Fullscreen(position="topright").add_to(m)

# User's Searched Location Center Marker
folium.Marker(
    [cur_lat, cur_lon],
    popup=f"<b>Searched Location:</b><br>{st.session_state['display_name']}",
    tooltip="Search Center Location",
    icon=folium.Icon(color="blue", icon="user", prefix="fa")
).add_to(m)

# Selected Radius Ring Indicator
folium.Circle(
    [cur_lat, cur_lon],
    radius=radius_km * 1000,
    color="#ef4444" if "LOW" in safety_level else "#38bdf8",
    weight=3,
    fill=True,
    fill_opacity=0.12
).add_to(m)

# Plotting Safety Infrastructure Markers dynamically on Map
for item in facilities:
    folium.Marker(
        [item["lat"], item["lon"]],
        popup=f"<b>{item['name']}</b><br>Type: {item['type']}<br>Distance: {item['distance_km']} km<br>Phone: {item['phone']}",
        tooltip=f"{item['name']} ({item['distance_km']} km)",
        icon=folium.Icon(color=item["color"], icon=item["icon"], prefix="fa")
    ).add_to(m)

st_folium(m, width="100%", height=500, key=f"map_{cur_lat}_{cur_lon}_{radius_km}")

st.markdown("<br>", unsafe_allow_html=True)

# Predictive Safety & Analytics Engine (Linear Regression using NumPy)
st.subheader("📊 Safety Analytics & 3-Year Future Trend Forecast")

df_crime = get_historical_crime_data()

X_vals = df_crime["Year"].values
y_vals = df_crime["Reported_Incidents"].values

# Pure NumPy polynomial fit (No sklearn required)
slope, intercept = np.polyfit(X_vals, y_vals, 1)

future_years = np.array([2024, 2025, 2026])
future_preds = slope * future_years + intercept

df_future = pd.DataFrame({
    "Year": future_years,
    "Reported_Incidents": np.clip(future_preds, 0, None),
    "Type": ["Predicted"] * 3
})

df_crime["Type"] = "Historical"
df_combined = pd.concat([df_crime, df_future], ignore_index=True)

fig = px.line(
    df_combined, 
    x="Year", 
    y="Reported_Incidents", 
    color="Type",
    markers=True,
    title="Analytics: Historical Incident Patterns vs 3-Year Future Trend Model",
    color_discrete_map={"Historical": "#38bdf8", "Predicted": "#ef4444"}
)
fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Emergency Contacts Directory Table
st.subheader("📋 Nearby Infrastructure Directory")
if facilities:
    df_fac = pd.DataFrame(facilities)[["name", "type", "distance_km", "phone"]]
    df_fac.columns = ["Facility / Support Center", "Category", "Distance (km)", "Contact Number"]
    st.dataframe(df_fac, use_container_width=True)
else:
    st.info("No open infrastructure data mapped in OpenStreetMap for this specific radius. Try increasing the search radius in sidebar.")

# Direct National Emergency Helplines Panel
st.markdown("""
<div style="background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #3b82f6;">
    <h4 style="margin: 0; color: #38bdf8;">🚨 Pan-India National Emergency Helplines</h4>
    <p style="margin: 5px 0 0 0; color: #e2e8f0;">
        <b>National Emergency Service:</b> 112 | 
        <b>Women Helpline:</b> 181 / 1091 | 
        <b>Child Line:</b> 1098 | 
        <b>Cyber Crime Helpline:</b> 1930
    </p>
</div>
""", unsafe_allow_html=True)