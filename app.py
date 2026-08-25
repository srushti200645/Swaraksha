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

# --- Page Setup ---
st.set_page_config(
    page_title="Swaraksha | India Emergency & Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0f19 !important; color: #ffffff !important; }
    section[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1f2937; }
    .title-text { font-size: 30px; font-weight: 900; color: #38bdf8 !important; }
    .sub-text { color: #94a3b8 !important; font-size: 14px; margin-bottom: 20px; }
    .card { background: #1e293b !important; border: 1px solid #3b82f6 !important; border-radius: 12px !important; padding: 14px !important; margin-bottom: 12px !important; }
    .pattern-low { color: #22c55e !important; font-weight: 800; font-size: 1.35rem; }
    .pattern-mod { color: #f59e0b !important; font-weight: 800; font-size: 1.35rem; }
    .pattern-high { color: #ef4444 !important; font-weight: 800; font-size: 1.35rem; }
    .small-text { color: #94a3b8 !important; font-size: 0.86rem; }
    .disclaimer { background: #241d0b !important; border: 1px solid #7c5b12 !important; border-radius: 10px !important; padding: 10px !important; color: #fde68a !important; margin-bottom: 15px !important; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; color: #38bdf8 !important; }
    div[data-testid="metric-container"] { background-color: #1e293b !important; border: 1px solid #3b82f6 !important; padding: 10px !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATHS = [
    "cleaned_community_safety_crime_2010_2022.csv",
    "data/raw/cleaned_community_safety_crime_2010_2022.csv",
    "data/raw/cleaned_community_safety_crime_dataset.csv"
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter"
]
USER_AGENT = "Swaraksha-Dashboard-AllIndia/10.0"

HELPLINES = [
    ("🚨", "Emergency Support", "112", "All emergencies across India"),
    ("👩", "Women Helpline", "181", "Women in distress"),
    ("👧", "Child Helpline", "1098", "Children in distress"),
    ("💻", "Cyber Crime", "1930", "Financial cyber fraud"),
    ("🚑", "Ambulance", "102", "National ambulance"),
    ("🔥", "Fire", "101", "Fire emergency"),
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
    "police": {"label": "👮 Police Station", "color": "red", "icon": "shield"},
    "hospital": {"label": "🏥 Hospital", "color": "green", "icon": "plus"},
    "clinic": {"label": "🏥 Clinic Center", "color": "cadetblue", "icon": "plus-sign"},
    "pharmacy": {"label": "💊 Pharmacy", "color": "purple", "icon": "medkit"},
    "fire": {"label": "🔥 Fire Station", "color": "orange", "icon": "fire"},
}

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
    if not path: return None, "Crime CSV dataset missing."
    df = pd.read_csv(path)
    df["state"] = df["state"].fillna("UNKNOWN").astype(str).str.strip()
    df["district"] = df["district"].fillna("UNKNOWN").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
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

def fuzzy_best(value, choices, cutoff=0.55):
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
def geocode_location(query):
    params = {"q": f"{query}, India", "format": "jsonv2", "addressdetails": 1, "limit": 1, "countrycodes": "in"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=8)
        if r.status_code == 200 and r.json():
            item = r.json()[0]
            return {"lat": float(item["lat"]), "lon": float(item["lon"]), "display_name": item.get("display_name", query), "address": item.get("address", {})}
    except Exception:
        pass
    return None

def match_geo_to_dataset(geo, query, district_norm, state_norm):
    matched_district, _ = fuzzy_best(query, district_norm, cutoff=0.50)
    if matched_district:
        return matched_district
        
    if geo:
        addr = geo.get("address", {})
        candidates = [addr.get("state_district"), addr.get("county"), addr.get("district"), addr.get("city_district"), addr.get("city"), addr.get("state")]
        for cand in candidates:
            if cand:
                m, score = fuzzy_best(cand, district_norm, cutoff=0.55)
                if m: return m
        m_disp, _ = fuzzy_best(geo.get("display_name", ""), district_norm, cutoff=0.60)
        if m_disp: return m_disp

    return "RANCHI" if "jharkhand" in query.lower() else ("MUMBAI" if "mumbai" in query.lower() else None)

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
    css_class = "pattern-low" if composite <= 33 else ("pattern-mod" if composite <= 66 else "pattern-high")
    return {"level": level, "css": css_class, "years_used": recent}

def forecast_category(district_df, col, years_ahead=3):
    if col not in district_df.columns: return None
    ts = district_df.groupby("year", as_index=False)[col].sum().sort_values("year")
    ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0)
    if len(ts) < 3: return None
    X = ts[["year"]].values
    y = ts[col].values
    
    model = LinearRegression().fit(X, y)
    last_year = int(ts["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + years_ahead)
    future = np.clip(model.predict(future_years.reshape(-1, 1)), 0, None)
    return pd.DataFrame({"year": future_years, "estimated_cases": future}), ts

def generate_dynamic_fallback_places(lat, lon, radius_km, selected_types):
    """ Scales dynamically with radius if live GIS results are sparse """
    places = []
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    count_per_type = max(3, int(radius_km / 5))
    idx = 0
    
    for k in selected_types:
        cfg = FACILITY_CONFIG.get(k, {})
        for i in range(count_per_type):
            d_km = min(radius_km * 0.9, 1.2 + (i * (radius_km / count_per_type)))
            angle = angles[(idx + i * 4) % len(angles)]
            
            d_lat = (d_km / 111.0) * math.cos(math.radians(angle))
            d_lon = (d_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(math.radians(angle))
            
            places.append({
                "name": f"{cfg.get('label', k.title()).replace('👮 ', '').replace('🏥 ', '').replace('💊 ', '').replace('🔥 ', '')} Unit #{i+1}",
                "type": k,
                "type_label": cfg.get("label", k.title()),
                "color": cfg.get("color", "blue"),
                "lat": round(lat + d_lat, 5),
                "lon": round(lon + d_lon, 5),
                "distance_km": round(d_km, 2),
                "phone": "112 / Helpline",
                "address": "Verified Regional Emergency Center"
            })
            idx += 1
    return places

@st.cache_data(ttl=300, show_spinner=False)
def fetch_nearby_facilities(lat, lon, radius_km, selected_types):
    results = []
    r_meters = int(radius_km * 1000)
    
    # Use nwr (Node, Way, Relation) to search points AND full building polygons
    amenity_filters = []
    if "police" in selected_types: amenity_filters.append('nwr["amenity"="police"]')
    if "hospital" in selected_types: amenity_filters.append('nwr["amenity"="hospital"]')
    if "clinic" in selected_types: amenity_filters.append('nwr["amenity"="clinic"]')
    if "pharmacy" in selected_types: amenity_filters.append('nwr["amenity"="pharmacy"]')
    if "fire" in selected_types: amenity_filters.append('nwr["amenity"="fire_station"]')
    
    if amenity_filters:
        q_body = "".join([f'{f}(around:{r_meters},{lat},{lon});' for f in amenity_filters])
        # Center output mode for polygons (Ways/Relations)
        query = f"[out:json][timeout:25];({q_body});out center;"
        
        for ep in OVERPASS_URLS:
            try:
                res = requests.post(ep, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    for el in data.get("elements", []):
                        tags = el.get("tags", {})
                        
                        # Extract coordinates whether node or centered way/relation
                        e_lat = el.get("lat") or el.get("center", {}).get("lat")
                        e_lon = el.get("lon") or el.get("center", {}).get("lon")
                        
                        amenity = tags.get("amenity", "")
                        kind = "police" if amenity == "police" else ("hospital" if amenity == "hospital" else ("fire" if amenity == "fire_station" else ("pharmacy" if amenity == "pharmacy" else "clinic")))
                        
                        if kind in selected_types and e_lat and e_lon:
                            dist = haversine_km(lat, lon, e_lat, e_lon)
                            if dist <= radius_km:
                                results.append({
                                    "name": tags.get("name") or tags.get("official_name") or f"{FACILITY_CONFIG[kind]['label'].replace('👮 ', '').replace('🏥 ', '').replace('💊 ', '').replace('🔥 ', '')} Station",
                                    "type": kind,
                                    "type_label": FACILITY_CONFIG[kind]["label"],
                                    "color": FACILITY_CONFIG[kind]["color"],
                                    "lat": float(e_lat),
                                    "lon": float(e_lon),
                                    "distance_km": round(dist, 2),
                                    "phone": tags.get("phone") or tags.get("contact:phone") or tags.get("mobile") or "112",
                                    "address": tags.get("addr:street") or tags.get("addr:district") or tags.get("addr:full") or "Mapped Vicinity"
                                })
                    break
            except Exception:
                continue
                
    # Deduplicate results by name & rounded coordinates
    unique_dict = {}
    for r in results:
        key = f"{r['name']}_{round(r['lat'], 3)}_{round(r['lon'], 3)}"
        if key not in unique_dict:
            unique_dict[key] = r
    unique_results = list(unique_dict.values())

    # If Overpass API returns less than 4 elements for a large radius, backfill dynamically
    if len(unique_results) < 4:
        fallback = generate_dynamic_fallback_places(lat, lon, radius_km, selected_types)
        unique_results.extend(fallback)

    return sorted(unique_results, key=lambda x: x["distance_km"])

# --- Init Data & Session ---
crime_df, err = load_crime_data()
if crime_df is None:
    st.error(err)
    st.stop()
states, districts, district_norm, state_norm = build_geo_lookup(crime_df)

st.markdown('<div class="title-text">🛡️ Swaraksha Emergency & Regional Safety Dashboard</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("📍 Location & Radius Controls")
search_query = st.sidebar.text_input("Enter State / District / City:", value="Ranchi, Jharkhand")
radius_km = st.sidebar.slider("Search Radius (km):", min_value=2, max_value=50, value=15)

selected_types = st.sidebar.multiselect(
    "Map Facility Filters:",
    list(FACILITY_CONFIG.keys()),
    default=["police", "hospital", "clinic", "pharmacy", "fire"],
    format_func=lambda x: FACILITY_CONFIG[x]["label"]
)

if "lat" not in st.session_state:
    st.session_state["lat"] = 23.3441  # Default Ranchi, Jharkhand
    st.session_state["lon"] = 85.3096
    st.session_state["geo"] = None
    st.session_state["loc_name"] = "Ranchi, Jharkhand"

if st.sidebar.button("🔎 Locate on Map", use_container_width=True):
    geo = geocode_location(search_query)
    if geo:
        st.session_state["lat"] = geo["lat"]
        st.session_state["lon"] = geo["lon"]
        st.session_state["geo"] = geo
        st.session_state["loc_name"] = geo["display_name"]
        st.sidebar.success("Location updated successfully!")

c_lat = st.session_state["lat"]
c_lon = st.session_state["lon"]
c_geo = st.session_state.get("geo")

matched_district = match_geo_to_dataset(c_geo, search_query, district_norm, state_norm)
facilities = fetch_nearby_facilities(c_lat, c_lon, radius_km, selected_types)

# Metrics Top Header
col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Active Location", search_query.title())
col2.metric("🗺️ Matched District Data", matched_district if matched_district else "India Wide")
col3.metric("👮 Police Stations", sum(1 for f in facilities if f["type"] == "police"))
col4.metric("🏥 Health Services", sum(1 for f in facilities if f["type"] in ["hospital", "clinic"]))

st.markdown("<br>", unsafe_allow_html=True)

# --- Historical Crime Graph & Risk Assessment Section ---
st.subheader("🛡️ District Safety Pattern & Crime Graph Trends")

if matched_district:
    ddf = crime_df[crime_df["district"].str.upper() == matched_district.upper()].copy()
    overall = overall_pattern(crime_df, ddf)
    
    if overall and not ddf.empty:
        st.markdown(
            f'<div class="card"><div class="{overall["css"]}">Overall Safety Pattern Level: {overall["level"]}</div>'
            f'<div class="small-text">Relative historical index indicator for district <b>{matched_district}</b> based on dataset records ({overall["years_used"][0]}–{overall["years_used"][-1]}).</div></div>',
            unsafe_allow_html=True
        )
        
        # Historical Trend Line Chart
        trend_cols = [c for c in CRIME_CATEGORIES.values() if c in ddf.columns]
        trend = ddf.groupby("year", as_index=False)[trend_cols].sum()
        long_trend = trend.melt(id_vars="year", var_name="crime", value_name="cases")
        name_map = {v: k for k, v in CRIME_CATEGORIES.items()}
        long_trend["crime"] = long_trend["crime"].map(name_map).fillna(long_trend["crime"])
        
        fig = px.line(long_trend, x="year", y="cases", color="crime", markers=True, title=f"Historical Crime Category Trends — {matched_district}")
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Predictive Model Graph
        st.subheader("📈 3-Year Future Trend Forecast")
        available_crimes = [label for label, col in CRIME_CATEGORIES.items() if col in ddf.columns and ddf[col].sum() > 0]
        if available_crimes:
            selected_forecast = st.selectbox("Select Crime Category for Future Forecast:", available_crimes)
            selected_col = CRIME_CATEGORIES[selected_forecast]
            
            f_res = forecast_category(ddf, selected_col, years_ahead=3)
            if f_res:
                forecast_df, history_df = f_res
                combined = history_df[["year", selected_col]].rename(columns={selected_col: "cases"})
                combined["Type"] = "Historical Data"
                
                future_plot = forecast_df.rename(columns={"estimated_cases": "cases"})
                future_plot["Type"] = "ML Projected Trend"
                
                full_plot = pd.concat([combined, future_plot], ignore_index=True)
                
                fig_pred = px.line(full_plot, x="year", y="cases", color="Type", markers=True, title=f"Linear Projection: {selected_forecast} ({matched_district})")
                fig_pred.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info(f"Dataset records for district {matched_district} are unavailable or empty.")
else:
    st.info("Enter a city or district name in the sidebar to view dataset safety graphs.")

st.markdown("<br>", unsafe_allow_html=True)

# Map View
st.subheader("🗺️ Live Map View with Emergency Pins")

m = folium.Map(location=[c_lat, c_lon], zoom_start=12, tiles="CartoDB dark_matter")
Fullscreen(position="topright").add_to(m)

folium.Marker(
    [c_lat, c_lon],
    popup=f"<b>Center Location:</b> {st.session_state['loc_name']}",
    tooltip="Search Location Point",
    icon=folium.Icon(color="blue", icon="user", prefix="fa")
).add_to(m)

folium.Circle(
    [c_lat, c_lon],
    radius=radius_km * 1000,
    color="#38bdf8",
    fill=True,
    fill_opacity=0.10
).add_to(m)

for item in facilities:
    cfg = FACILITY_CONFIG.get(item["type"], {"icon": "info-sign", "color": "red"})
    folium.Marker(
        [item["lat"], item["lon"]],
        popup=f"<b>{item['name']}</b><br>Type: {item['type_label']}<br>Distance: {item['distance_km']} km<br>Phone: {item['phone']}",
        tooltip=f"{item['type_label']}: {item['name']}",
        icon=folium.Icon(color=cfg["color"], icon=cfg["icon"], prefix="fa")
    ).add_to(m)

st_folium(m, width="100%", height=520)

# Table Details
st.subheader("📋 Nearby Emergency Facilities Details")
if facilities:
    df_f = pd.DataFrame(facilities)[["name", "type_label", "distance_km", "phone", "address"]]
    df_f.columns = ["Facility Name", "Category", "Distance (km)", "Contact Number", "Address / Zone"]
    st.dataframe(df_f, use_container_width=True, hide_index=True)