import math
import re
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import Fullscreen
from sklearn.linear_model import LinearRegression
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

# --- Page Config & Dark Theme Setup ---
st.set_page_config(
    page_title="Swaraksha | India Emergency & Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Contrast Dark Theme CSS
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Primary Titles & Headers */
    .title-text { 
        font-size: 32px; 
        font-weight: 900; 
        color: #38bdf8 !important; 
        margin-bottom: 5px;
    }
    .sub-text {
        color: #94a3b8 !important;
        font-size: 15px;
        margin-bottom: 25px;
    }
    
    /* Custom Card Styling */
    .card {
        background: #1e293b !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
    }
    
    .pattern-low { color: #22c55e !important; font-weight: 800; font-size: 1.45rem; }
    .pattern-mod { color: #f59e0b !important; font-weight: 800; font-size: 1.45rem; }
    .pattern-high { color: #ef4444 !important; font-weight: 800; font-size: 1.45rem; }
    .small-text { color: #94a3b8 !important; font-size: 0.86rem; }
    
    .disclaimer {
        background: #241d0b !important;
        border: 1px solid #7c5b12 !important;
        border-radius: 12px !important;
        padding: 11px 14px !important;
        color: #fde68a !important;
        margin-bottom: 20px !important;
    }
    
    /* Metrics Box Custom Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
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

    h1, h2, h3, h4, label, p {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# Dataset & API Configuration
DATA_PATHS = [
    "data/raw/cleaned_community_safety_crime_2010_2022.csv",
    "data/raw/cleaned_community_safety_crime_dataset.csv",
    "cleaned_community_safety_crime_2010_2022.csv",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "Swaraksha-AllIndia-Safety-Dashboard/7.0"

# Central India Default Coordinates
DEFAULT_LAT = 20.5937  
DEFAULT_LON = 78.9629

HELPLINES = [
    ("🚨", "Emergency Support", "112", "All emergencies across India"),
    ("👩", "Women Helpline", "181", "Women in distress"),
    ("👧", "Child Helpline", "1098", "Children in distress"),
    ("💻", "Cyber Crime", "1930", "Financial cyber fraud"),
    ("⚖️", "Legal Aid", "15100", "Free legal aid"),
    ("🧠", "Tele-MANAS", "14416", "Mental health support"),
    ("🚑", "Ambulance", "102", "National ambulance"),
    ("🔥", "Fire", "101", "Fire emergency"),
    ("👮", "Police", "100", "Police hotline"),
]

CRIME_CATEGORIES = {
    "Rape": "rape",
    "Assault on women": "assault_women",
    "Insult to modesty": "insult_modesty",
    "Importation of girls": "importation_girls",
    "Immoral traffic": "immoral_traffic",
    "Procuration of minor girls": "procuration_minor_girls",
    "Human trafficking": "human_trafficking",
    "Cyber explicit material": "cyber_explicit_material",
    "Other women cyber crimes": "other_women_cyber_crimes",
}

FACILITY_CONFIG = {
    "police": {"label": "👮 Police Station", "color": "red", "icon": "shield", "queries": ['nwr["amenity"="police"]']},
    "hospital": {"label": "🏥 Hospital", "color": "green", "icon": "plus", "queries": ['nwr["amenity"="hospital"]']},
    "clinic": {"label": "🏥 Clinic / Health Center", "color": "cadetblue", "icon": "plus-sign", "queries": ['nwr["amenity"="clinic"]', 'nwr["healthcare"="centre"]']},
    "pharmacy": {"label": "💊 Pharmacy", "color": "purple", "icon": "medkit", "queries": ['nwr["amenity"="pharmacy"]']},
    "fire": {"label": "🔥 Fire Station", "color": "orange", "icon": "fire", "queries": ['nwr["amenity"="fire_station"]']},
    "legal": {"label": "⚖️ Legal Services", "color": "darkblue", "icon": "balance-scale", "queries": ['nwr["office"="lawyer"]', 'nwr["amenity"="legal_services"]']},
}

# --- Utility Functions ---
def normalize_text(value):
    if pd.isna(value): return ""
    s = str(value).upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1 - a)))

@st.cache_data(ttl=3600, show_spinner=False)
def load_crime_data():
    path = next((p for p in DATA_PATHS if __import__("os").path.exists(p)), None)
    if not path:
        return None, "Crime CSV dataset missing. Place dataset file in the app directory."
    df = pd.read_csv(path)
    required = {"state", "district", "year"}
    if not required.issubset(set(df.columns)):
        return None, f"Dataset missing required columns: {required - set(df.columns)}"
    df["state"] = df["state"].fillna("UNKNOWN").astype(str).str.strip()
    df["district"] = df["district"].fillna("UNKNOWN").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    for col in CRIME_CATEGORIES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    return df, None

@st.cache_data(ttl=3600, show_spinner=False)
def build_geo_lookup(df):
    states = sorted(df["state"].dropna().unique().tolist())
    districts = sorted(df["district"].dropna().unique().tolist())
    district_norm = {normalize_text(d): d for d in districts}
    state_norm = {normalize_text(s): s for s in states}
    return states, districts, district_norm, state_norm

def fuzzy_best(value, choices, cutoff=0.72):
    n = normalize_text(value)
    if not n: return None, 0
    if n in choices: return choices[n], 1.0
    best_name, best_score = None, 0.0
    for c_norm, original in choices.items():
        score = SequenceMatcher(None, n, c_norm).ratio()
        if score > best_score:
            best_score, best_name = score, original
    return (best_name, best_score) if best_score >= cutoff else (None, best_score)

@st.cache_data(ttl=900, show_spinner=False)
def geocode_india(query):
    params = {"q": f"{query}, India", "format": "jsonv2", "addressdetails": 1, "limit": 1, "countrycodes": "in"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data:
            item = data[0]
            return {"lat": float(item["lat"]), "lon": float(item["lon"]), "display_name": item.get("display_name", query), "address": item.get("address", {})}
    except Exception:
        pass
    return None

def match_geo_to_dataset(geo, district_norm, state_norm):
    if not geo: return None, None
    addr = geo.get("address", {})
    state_text = addr.get("state") or addr.get("state_district")
    matched_state, _ = fuzzy_best(state_text, state_norm, cutoff=0.68)
    
    candidates = [addr.get("state_district"), addr.get("county"), addr.get("district"), addr.get("city_district"), addr.get("city"), addr.get("town")]
    candidates = [x for x in candidates if x]
    
    matched_district, district_score = None, 0
    for cand in candidates:
        m, score = fuzzy_best(cand, district_norm, cutoff=0.70)
        if m and score > district_score:
            matched_district, district_score = m, score
    
    if not matched_district:
        matched_district, _ = fuzzy_best(geo.get("display_name", ""), district_norm, cutoff=0.78)
        
    return matched_state, matched_district

def overall_pattern(df, district_df):
    available = [c for c in CRIME_CATEGORIES.values() if c in df.columns]
    if district_df.empty or not available: return None
    scores = []
    years = sorted(district_df["year"].unique())
    recent = years[-3:] if len(years) >= 3 else years
    for c in available:
        ref = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        val = float(district_df[district_df["year"].isin(recent)][c].mean())
        percentile = float((ref <= val).mean() * 100) if len(ref) else 50
        scores.append(percentile)
    composite = float(np.mean(scores)) if scores else 50
    level = "🟢 LOW" if composite <= 33 else ("🟡 MODERATE" if composite <= 66 else "🔴 HIGH")
    return {"level": level, "years_used": recent}

def forecast_category(district_df, col, years_ahead=3):
    if col not in district_df.columns: return None
    ts = district_df.groupby("year", as_index=False)[col].sum().sort_values("year")
    ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0)
    if len(ts) < 4: return None
    X = ts[["year"]].values
    y = ts[col].values
    
    train, test = ts.iloc[:-2], ts.iloc[-2:]
    mae = None
    if len(train) >= 2:
        m_val = LinearRegression().fit(train[["year"]], train[col])
        pred_val = np.clip(m_val.predict(test[["year"]]), 0, None)
        mae = float(np.mean(np.abs(pred_val - test[col].values)))
        
    model = LinearRegression().fit(X, y)
    last_year = int(ts["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + years_ahead)
    future = np.clip(model.predict(future_years.reshape(-1, 1)), 0, None)
    return pd.DataFrame({"year": future_years, "estimated_cases": future}), mae, ts

@st.cache_data(ttl=900, show_spinner=False)
def get_all_india_facilities(lat, lon, radius_km, selected_types):
    results = []
    query_parts = []
    for key in selected_types:
        if key in FACILITY_CONFIG:
            query_parts.extend(f"{q}(around:{int(radius_km * 1000)},{lat},{lon});" for q in FACILITY_CONFIG[key]["queries"])
    
    if query_parts:
        query = f"[out:json][timeout:25]; ( {''.join(query_parts)} ); out center tags;"
        for endpoint in OVERPASS_URLS:
            try:
                res = requests.post(endpoint, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=15)
                if res.status_code == 200:
                    for el in res.json().get("elements", []):
                        tags = el.get("tags", {})
                        e_lat = el.get("lat") or el.get("center", {}).get("lat")
                        e_lon = el.get("lon") or el.get("center", {}).get("lon")
                        amenity = tags.get("amenity", "")
                        
                        kind = "police" if amenity == "police" else ("hospital" if amenity in ["hospital", "clinic"] else "clinic")
                        if kind in selected_types and e_lat and e_lon:
                            dist = haversine_km(lat, lon, e_lat, e_lon)
                            name_str = tags.get("name") or tags.get("official_name") or f"Local {FACILITY_CONFIG.get(kind, {}).get('label', 'Emergency Facility')}"
                            results.append({
                                "name": name_str,
                                "type": kind,
                                "type_label": FACILITY_CONFIG.get(kind, {}).get("label", kind.title()),
                                "color": FACILITY_CONFIG.get(kind, {}).get("color", "blue"),
                                "lat": float(e_lat),
                                "lon": float(e_lon),
                                "distance_km": round(dist, 2),
                                "phone": tags.get("phone") or tags.get("contact:phone") or tags.get("mobile") or "N/A",
                                "address": tags.get("addr:street") or tags.get("addr:full") or tags.get("addr:district") or "Mapped Location Area"
                            })
                    break
            except Exception:
                pass
                
    unique = {f"{f['name']}_{f['lat']}": f for f in results}.values()
    return sorted(list(unique), key=lambda x: x["distance_km"])

# Load Dataset
crime_df, data_error = load_crime_data()
if crime_df is None:
    st.error(data_error)
    st.stop()
states, districts, district_norm, state_norm = build_geo_lookup(crime_df)

# --- Header & Disclaimer ---
st.markdown('<div class="title-text">🛡️ Swaraksha | All-India Emergency Finder & Safety Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Search any Location across India to Locate Hospitals, Police Stations, Emergency Services & Safety Trends.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚠️ <b>All-India Safety Disclaimer:</b> Historical patterns indicate statistical relative reported trends in district datasets. Swaraksha displays live geographic facilities nationwide via OpenStreetMap but does not directly dispatch emergency first responders.</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.header("📍 Location & Search Settings")

selected_state = st.sidebar.selectbox("Filter State / UT:", ["All India"] + states)

search_query = st.sidebar.text_input("Enter Any Indian City / District / Area:", value="New Delhi")
radius_km = st.sidebar.slider("Search Radius (km):", min_value=1, max_value=30, value=10)

if "current_lat" not in st.session_state:
    st.session_state["current_lat"] = 28.6139  # Default: New Delhi
    st.session_state["current_lon"] = 77.2090
    st.session_state["current_geo"] = None
    st.session_state["active_location_name"] = "New Delhi, Delhi"

if st.sidebar.button("🔎 Search Location", use_container_width=True):
    geo_res = geocode_india(search_query)
    if geo_res:
        st.session_state["current_lat"] = geo_res["lat"]
        st.session_state["current_lon"] = geo_res["lon"]
        st.session_state["current_geo"] = geo_res
        st.session_state["active_location_name"] = geo_res["display_name"]
        st.sidebar.success(f"Location Found: {geo_res['display_name'].split(',')[0]}!")
    else:
        st.sidebar.error("Location not found. Try searching with city name, e.g., 'Bengaluru', 'Jaipur', 'Patna'.")

try:
    loc = streamlit_geolocation()
    if loc and loc.get("latitude") is not None:
        if st.sidebar.button("📍 Use Device GPS Location"):
            st.session_state["current_lat"] = float(loc["latitude"])
            st.session_state["current_lon"] = float(loc["longitude"])
            st.session_state["current_geo"] = {"lat": float(loc["latitude"]), "lon": float(loc["longitude"]), "display_name": "Device GPS Location", "address": {}}
            st.session_state["active_location_name"] = "Device GPS Coordinates"
            st.rerun()
except Exception:
    pass

cur_lat = st.session_state["current_lat"]
cur_lon = st.session_state["current_lon"]
cur_geo = st.session_state["current_geo"]

# Facility Type Selector
selected_types = st.sidebar.multiselect(
    "Filter Emergency Services:",
    list(FACILITY_CONFIG.keys()),
    default=["police", "hospital", "clinic", "pharmacy", "fire"],
    format_func=lambda x: FACILITY_CONFIG[x]["label"]
)

matched_state, matched_district = match_geo_to_dataset(cur_geo, district_norm, state_norm) if cur_geo else (None, None)
if not matched_district:
    # Fuzzy match search query if direct geo match is unassigned
    matched_district, _ = fuzzy_best(search_query, district_norm, cutoff=0.60)

facilities = get_all_india_facilities(cur_lat, cur_lon, radius_km, selected_types)
police_count = sum(1 for f in facilities if f["type"] == "police")
hospital_count = sum(1 for f in facilities if f["type"] in ["hospital", "clinic"])

# --- Key Dashboard Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Active Location", search_query.title())
col2.metric("🗺️ Matched District", matched_district if matched_district else "India Wide")
col3.metric("👮 Nearby Police Stations", police_count)
col4.metric("🏥 Healthcare Centers", hospital_count)

st.markdown("<br>", unsafe_allow_html=True)

# --- Safety & Pattern Assessment ---
st.subheader("🛡️ Regional Safety & Crime Pattern Analysis")
if matched_district:
    ddf = crime_df[crime_df["district"].str.upper() == matched_district.upper()].copy()
    overall = overall_pattern(crime_df, ddf)
    
    if overall:
        st.markdown(
            f'<div class="card"><div class="pattern-mod">Overall Pattern Level: {overall["level"]}</div>'
            f'<div class="small-text">Relative historical safety indicator for district <b>{matched_district}</b> based on reported historical dataset context ({overall["years_used"][0]}–{overall["years_used"][-1]}).</div></div>',
            unsafe_allow_html=True
        )
        
        # Historical Trend Chart
        trend_cols = [c for c in CRIME_CATEGORIES.values() if c in ddf.columns]
        trend = ddf.groupby("year", as_index=False)[trend_cols].sum()
        long_trend = trend.melt(id_vars="year", var_name="crime", value_name="cases")
        name_map = {v: k for k, v in CRIME_CATEGORIES.items()}
        long_trend["crime"] = long_trend["crime"].map(name_map).fillna(long_trend["crime"])
        
        fig = px.line(long_trend, x="year", y="cases", color="crime", markers=True, title=f"Historical Crime Category Trends — {matched_district}")
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Showing nationwide map view. Specific regional historical dataset record for {matched_district} is pending localized matching.")
else:
    st.info("Enter any Indian location in the sidebar to fetch district-level safety analysis.")

# --- Predictive ML Model ---
st.subheader("📈 Future Crime Trend Forecast Model")
if matched_district:
    ddf = crime_df[crime_df["district"].str.upper() == matched_district.upper()].copy()
    available_crimes = [label for label, col in CRIME_CATEGORIES.items() if col in ddf.columns and ddf[col].sum() > 0]
    
    if available_crimes:
        selected_forecast = st.selectbox("Select Crime Category for Forecast Estimation:", available_crimes)
        selected_col = CRIME_CATEGORIES[selected_forecast]
        
        f_res = forecast_category(ddf, selected_col, years_ahead=3)
        if f_res:
            forecast_df, mae, history_df = f_res
            combined = history_df[["year", selected_col]].rename(columns={selected_col: "cases"})
            combined["Type"] = "Historical Data"
            
            future_plot = forecast_df.rename(columns={"estimated_cases": "cases"})
            future_plot["Type"] = "ML Predictive Model"
            
            full_plot = pd.concat([combined, future_plot], ignore_index=True)
            
            fig_pred = px.line(full_plot, x="year", y="cases", color="Type", markers=True, title=f"3-Year Projected Trend: {selected_forecast} ({matched_district})")
            fig_pred.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_pred, use_container_width=True)
            if mae:
                st.caption(f"Model Mean Absolute Error (MAE): **{mae:.2f} cases**.")
        else:
            st.info("Insufficient timeline records for predictive linear modeling in this district.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Dynamic All-India GIS Emergency Map ---
st.subheader("🗺️ Live High-Contrast All-India Emergency Map")

m = folium.Map(location=[cur_lat, cur_lon], zoom_start=12, tiles="CartoDB dark_matter")
Fullscreen(position="topright").add_to(m)

# User location marker
folium.Marker(
    [cur_lat, cur_lon],
    popup=f"<b>Location:</b> {st.session_state['active_location_name']}",
    tooltip="Active Search Location",
    icon=folium.Icon(color="blue", icon="user", prefix="fa")
).add_to(m)

# Search Circle Radius
folium.Circle(
    [cur_lat, cur_lon],
    radius=radius_km * 1000,
    color="#38bdf8",
    fill=True,
    fill_opacity=0.12
).add_to(m)

# Emergency Facilities Pins
for item in facilities:
    cfg = FACILITY_CONFIG.get(item["type"], {"icon": "info-sign", "color": "blue"})
    folium.Marker(
        [item["lat"], item["lon"]],
        popup=f"<b>{item['name']}</b><br>Type: {item['type_label']}<br>Distance: {item['distance_km']} km<br>Phone: {item['phone']}<br>Address: {item.get('address', 'N/A')}",
        tooltip=f"{item['name']} ({item['distance_km']} km)",
        icon=folium.Icon(color=item.get("color", cfg["color"]), icon=cfg["icon"], prefix="fa")
    ).add_to(m)

st_folium(m, width="100%", height=520)

st.markdown("<br>", unsafe_allow_html=True)

# --- All India Facilities Directory ---
st.subheader("📋 Dynamic Emergency Services & Contact Directory")
if facilities:
    df_fac = pd.DataFrame(facilities)[["name", "type_label", "distance_km", "phone", "address"]]
    df_fac.columns = ["Facility Name", "Category", "Distance (km)", "Contact Number", "Address / Locality"]
    st.dataframe(df_fac, use_container_width=True, hide_index=True)
else:
    st.warning("No facilities returned within this radius. Increase search radius in the sidebar.")

# --- Helplines Grid ---
st.subheader("🚨 All-India National Emergency Helplines")
cols = st.columns(3)
for i, (emoji, name, number, desc) in enumerate(HELPLINES):
    with cols[i % 3]:
        st.markdown(
            f'<div class="card"><b>{emoji} {name}</b><br>'
            f'<span style="font-size:1.5rem;font-weight:800;color:#38bdf8">{number}</span><br>'
            f'<span class="small-text">{desc}</span></div>',
            unsafe_allow_html=True
        )

# --- Anonymous Feedback Section ---
st.subheader("👥 Community Safety Feedback")
with st.form("community_feedback"):
    fb_cat = st.selectbox("Observed Area Issue:", ["Poor Street Lighting", "Isolated Locality", "Low Police Patrol", "Safe/Active Zone", "Other"])
    fb_rating = st.slider("Area Perceived Safety Rating (1 = Low, 5 = High):", 1, 5, 3)
    fb_notes = st.text_area("Additional Notes (Anonymous):", max_chars=300)
    fb_submit = st.form_submit_button("Submit Feedback")

if fb_submit:
    try:
        __import__("os").makedirs("database", exist_ok=True)
        conn = sqlite3.connect("database/swaraksha_feedback.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                lat REAL, lon REAL, category TEXT, rating INTEGER, notes TEXT
            )
        """)
        conn.execute(
            "INSERT INTO feedback (timestamp, lat, lon, category, rating, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), cur_lat, cur_lon, fb_cat, fb_rating, fb_notes)
        )
        conn.commit()
        conn.close()
        st.success("Thank you! Your community observation has been anonymously recorded.")
    except Exception:
        st.error("Could not record feedback at this time.")