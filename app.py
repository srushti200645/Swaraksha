import math
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import Fullscreen, LocateControl
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None

# ============================================================
# SWARAKSHA - Know the Pattern. Find the Help.
# Historical district crime analysis + current GIS/GPS support
# ============================================================

st.set_page_config(
    page_title="Swaraksha | Know the Pattern. Find the Help.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#08111f; color:#f8fafc; }
section[data-testid="stSidebar"] { background:#0d1726; }
.block-container { padding-top:1.2rem; max-width:1500px; }
.hero { padding:22px 26px; border-radius:18px; background:linear-gradient(135deg,#10233d,#0b1524); border:1px solid #23405f; margin-bottom:18px; }
.hero h1 { margin:0; font-size:38px; }
.hero p { margin:7px 0 0; color:#b9c8da; }
.card { background:#101d2d; border:1px solid #243c56; border-radius:15px; padding:16px; height:100%; }
.pattern-low { color:#22c55e; font-weight:800; font-size:22px; }
.pattern-moderate { color:#facc15; font-weight:800; font-size:22px; }
.pattern-high { color:#ef4444; font-weight:800; font-size:22px; }
.small { color:#9fb0c4; font-size:13px; }
.warning { background:#291f09; border:1px solid #735f1b; border-radius:12px; padding:11px 14px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw" / "cleaned_community_safety_crime_2010_2022.csv"
DB_PATH = BASE / "database" / "swaraksha_feedback.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Swaraksha-CivicSafety/1.0 (educational project)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# The uploaded dataset's canonical columns. Duplicate source-derived columns are ignored.
CRIME_COLUMNS = {
    "Rape": "rape",
    "Assault on women": "assault_women",
    "Insult to modesty": "insult_modesty",
    "Importation of girls": "importation_girls",
    "Immoral traffic": "immoral_traffic",
    "Procuration of minor girls": "procuration_minor_girls",
    "Human trafficking": "human_trafficking",
    "Assault on women (above 18)": "assault_women_above18",
    "Assault on women (below 18)": "assault_women_below18",
    "Insult to modesty (above 18)": "insult_modesty_above18",
    "Insult to modesty (below 18)": "insult_modesty_below18",
    "Cyber explicit material": "cyber_explicit_material",
    "Other women cyber crimes": "other_women_cyber_crimes",
}

FACILITY_DEFS = {
    "👮 Police": {"queries": ['node["amenity"="police"]', 'way["amenity"="police"]', 'relation["amenity"="police"]'], "color": "red", "icon": "shield"},
    "🏥 Hospital": {"queries": ['node["amenity"="hospital"]', 'way["amenity"="hospital"]', 'relation["amenity"="hospital"]'], "color": "green", "icon": "plus"},
    "🏥 Health Centre / Clinic": {"queries": ['node["amenity"="clinic"]', 'way["amenity"="clinic"]', 'node["amenity"="doctors"]', 'way["amenity"="doctors"]'], "color": "cadetblue", "icon": "medkit"},
    "💊 Pharmacy": {"queries": ['node["amenity"="pharmacy"]', 'way["amenity"="pharmacy"]'], "color": "purple", "icon": "plus"},
    "🔥 Fire Station": {"queries": ['node["amenity"="fire_station"]', 'way["amenity"="fire_station"]'], "color": "orange", "icon": "fire"},
}

# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Crime CSV not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"state", "district", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["state"] = df["state"].fillna("").astype(str).str.strip()
    df["district"] = df["district"].fillna("").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for col in CRIME_COLUMNS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    if "total_crimes" in df.columns:
        df["total_crimes"] = pd.to_numeric(df["total_crimes"], errors="coerce").fillna(0).clip(lower=0)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def norm_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def canonical_district(name: str) -> str:
    n = norm_text(name)
    n = re.sub(r"\bdistrict\b", "", n).strip()
    return n


def find_dataset_area(df: pd.DataFrame, state: str, district: str) -> pd.DataFrame:
    s = norm_text(state)
    d = canonical_district(district)
    mask = df["state"].map(norm_text).eq(s) & df["district"].map(canonical_district).eq(d)
    return df.loc[mask].copy()


def match_district_from_name(df: pd.DataFrame, display_name: str) -> Tuple[Optional[str], Optional[str]]:
    text = norm_text(display_name)
    # Longest district/state labels first to reduce accidental short matches.
    states = sorted(df["state"].dropna().unique(), key=lambda x: len(norm_text(x)), reverse=True)
    districts = sorted(df["district"].dropna().unique(), key=lambda x: len(canonical_district(x)), reverse=True)
    state_match = next((s for s in states if norm_text(s) and norm_text(s) in text), None)
    district_match = next((d for d in districts if canonical_district(d) and canonical_district(d) in text), None)
    return state_match, district_match


def level_from_percentile(p: float) -> str:
    if pd.isna(p):
        return "UNAVAILABLE"
    if p < 33:
        return "LOW"
    if p < 67:
        return "MODERATE"
    return "HIGH"


def level_html(level: str) -> str:
    cls = {"LOW":"pattern-low", "MODERATE":"pattern-moderate", "HIGH":"pattern-high"}.get(level, "small")
    emoji = {"LOW":"🟢", "MODERATE":"🟡", "HIGH":"🔴"}.get(level, "⚪")
    return f'<div class="{cls}">{emoji} {level}</div>'


def district_category_levels(df: pd.DataFrame, district_df: pd.DataFrame) -> pd.DataFrame:
    if district_df.empty:
        return pd.DataFrame(columns=["Category", "Historical Cases", "Percentile", "Pattern"])
    rows = []
    for label, col in CRIME_COLUMNS.items():
        if col not in df.columns:
            continue
        district_total = float(district_df[col].sum())
        district_avg = float(district_df.groupby("year")[col].sum().mean())
        area_year = df.groupby(["state", "district", "year"], dropna=False)[col].sum().reset_index()
        # Compare annual district averages across all district-year observations.
        benchmark = area_year.groupby(["state", "district"])[col].mean()
        pct = float((benchmark < district_avg).mean() * 100) if len(benchmark) else np.nan
        rows.append({"Category": label, "Historical Cases": int(round(district_total)), "Percentile": round(pct, 1), "Pattern": level_from_percentile(pct)})
    return pd.DataFrame(rows)


def overall_pattern(df: pd.DataFrame, district_df: pd.DataFrame) -> Tuple[str, float, List[str]]:
    scores = []
    factors = []
    if district_df.empty:
        return "UNAVAILABLE", np.nan, []
    for label, col in CRIME_COLUMNS.items():
        if col not in df.columns:
            continue
        district_avg = float(district_df.groupby("year")[col].sum().mean())
        benchmark = df.groupby(["state", "district", "year"], dropna=False)[col].sum().groupby(level=[0,1]).mean()
        if len(benchmark) == 0:
            continue
        pct = float((benchmark < district_avg).mean() * 100)
        scores.append(pct)
    if not scores:
        return "UNAVAILABLE", np.nan, []
    score = float(np.mean(scores))
    level = level_from_percentile(score)
    # Explain with the categories having the highest percentile.
    cat = district_category_levels(df, district_df).sort_values("Percentile", ascending=False)
    for _, r in cat.head(4).iterrows():
        factors.append(f"{r['Category']}: {r['Pattern']} historical pattern ({r['Historical Cases']} reported cases in the dataset)")
    return level, score, factors

# ============================================================
# GEOCODING / GPS / GIS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_india(query: str):
    if not query.strip():
        return None
    params = {
        "q": f"{query}, India",
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in",
        "addressdetails": 1,
        "polygon_geojson": 1,
    }
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data:
            x = data[0]
            return {
                "lat": float(x["lat"]),
                "lon": float(x["lon"]),
                "display_name": x.get("display_name", query),
                "address": x.get("address", {}),
                "geojson": x.get("geojson"),
            }
    except Exception:
        return None
    return None


def get_browser_location():
    if streamlit_geolocation is None:
        return None
    try:
        loc = streamlit_geolocation()
        if isinstance(loc, dict) and loc.get("latitude") is not None and loc.get("longitude") is not None:
            return float(loc["latitude"]), float(loc["longitude"])
    except Exception:
        pass
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(max(1-a, 0)))


def overpass_query(lat: float, lon: float, radius_m: int, selected: List[str]) -> str:
    q = []
    for key in selected:
        q.extend(FACILITY_DEFS[key]["queries"])
    # Support centres/shelters are inconsistently mapped; use a name search only for explicit support terms.
    q += [
        'node["name"~"One Stop|One-Stop|Sakhi|Shakti Sadan|Women.*Centre|Women.*Center",i]',
        'way["name"~"One Stop|One-Stop|Sakhi|Shakti Sadan|Women.*Centre|Women.*Center",i]',
        'node["amenity"="social_facility"]',
        'way["amenity"="social_facility"]',
    ]
    parts = [f"{item}(around:{radius_m},{lat},{lon});" for item in q]
    return "[out:json][timeout:25];(" + "".join(parts) + ");out center tags;"


@st.cache_data(ttl=900, show_spinner=False)
def get_nearby_facilities(lat: float, lon: float, radius_km: float, selected: Tuple[str, ...]):
    query = overpass_query(lat, lon, int(radius_km*1000), list(selected))
    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            r = requests.post(endpoint, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=35)
            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}"
                continue
            elements = r.json().get("elements", [])
            results = []
            seen = set()
            for el in elements:
                tags = el.get("tags", {})
                e_lat = el.get("lat") or el.get("center", {}).get("lat")
                e_lon = el.get("lon") or el.get("center", {}).get("lon")
                if e_lat is None or e_lon is None:
                    continue
                name = tags.get("name") or tags.get("official_name")
                if not name:
                    continue
                amenity = tags.get("amenity", "")
                nlow = name.lower()
                if amenity == "police":
                    typ, color, icon = "Police", "red", "shield"
                elif amenity == "hospital":
                    typ, color, icon = "Hospital", "green", "plus"
                elif amenity in {"clinic", "doctors"}:
                    typ, color, icon = "Health Centre / Clinic", "cadetblue", "medkit"
                elif amenity == "pharmacy":
                    typ, color, icon = "Pharmacy", "purple", "plus"
                elif amenity == "fire_station":
                    typ, color, icon = "Fire Station", "orange", "fire"
                elif any(x in nlow for x in ["one stop", "one-stop", "sakhi", "shakti sadan", "women"]):
                    typ, color, icon = "Women Support", "pink", "female"
                elif any(x in nlow for x in ["child", "bal", "juvenile"]):
                    typ, color, icon = "Child Support", "darkblue", "child"
                else:
                    typ, color, icon = "Social Support", "lightpurple", "home"
                key = (norm_text(name), round(float(e_lat), 5), round(float(e_lon), 5))
                if key in seen:
                    continue
                seen.add(key)
                dist = haversine_km(lat, lon, float(e_lat), float(e_lon))
                address = ", ".join(filter(None, [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:suburb"), tags.get("addr:city")]))
                results.append({
                    "name": name, "type": typ, "lat": float(e_lat), "lon": float(e_lon),
                    "distance_km": round(dist, 2), "phone": tags.get("phone") or tags.get("contact:phone") or "Not mapped",
                    "address": address or "Address not mapped", "website": tags.get("website") or "",
                    "color": color, "icon": icon,
                })
            return sorted(results, key=lambda x: x["distance_km"]), None
        except Exception as e:
            last_error = str(e)
    return [], last_error

# ============================================================
# FORECASTING
# ============================================================

@st.cache_data(show_spinner=False)
def forecast_category(district_df: pd.DataFrame, col: str, horizon: int = 3):
    if col not in district_df.columns:
        return None
    yearly = district_df.groupby("year")[col].sum().sort_index()
    yearly = yearly[yearly.index.notna()]
    if len(yearly) < 4:
        return None
    x = yearly.index.astype(float).to_numpy()
    y = yearly.to_numpy(dtype=float)
    # Time-aware holdout: last 2 observations are validation.
    hold = min(2, max(1, len(y)//4))
    train_x, test_x = x[:-hold], x[-hold:]
    train_y, test_y = y[:-hold], y[-hold:]
    coef = np.polyfit(train_x, train_y, 1) if len(train_x) >= 2 else np.array([0, train_y.mean()])
    pred_test = np.polyval(coef, test_x)
    mae = float(np.mean(np.abs(test_y - pred_test)))
    # Fit on all available history for future estimates.
    coef_all = np.polyfit(x, y, 1)
    future_years = np.arange(int(x.max())+1, int(x.max())+horizon+1)
    future = np.maximum(0, np.polyval(coef_all, future_years))
    return yearly, pd.Series(future, index=future_years), mae

# ============================================================
# FEEDBACK
# ============================================================

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        category TEXT NOT NULL,
        description TEXT,
        rating INTEGER
    )""")
    con.commit(); con.close()

init_db()

# ============================================================
# UI HELPERS
# ============================================================

def emergency_panel():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Emergency & Helplines")
    numbers = [
        ("🚨 Emergency", "112"), ("👩 Women", "181"), ("👧 Child", "1098"),
        ("💻 Cyber Crime", "1930"), ("⚖️ Legal Aid", "15100"), ("🧠 Tele-MANAS", "14416"),
        ("🚑 Ambulance", "102"), ("🔥 Fire", "101"), ("👮 Police / legacy", "100"),
    ]
    for label, num in numbers:
        st.sidebar.markdown(f"**{label}: [{num}](tel:{num})**")
    st.sidebar.caption("Swaraksha does not dispatch emergency services. Use the official emergency service directly.")


def draw_map(lat, lon, zoom, facilities=None, geojson=None, radius_km=5, india_view=False):
    if india_view:
        m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True)
    else:
        m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    Fullscreen(position="topright").add_to(m)
    LocateControl(auto_start=False, flyTo=True, strings={"title": "Use browser location"}).add_to(m)
    folium.Marker([lat, lon], tooltip="📍 Selected location", popup="Selected location", icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(m)
    if not india_view and radius_km:
        folium.Circle([lat, lon], radius=radius_km*1000, color="#38bdf8", fill=True, fill_opacity=0.08, tooltip=f"Search radius: {radius_km} km").add_to(m)
    if geojson:
        try:
            folium.GeoJson(geojson, name="Selected boundary", style_function=lambda _: {"fillColor":"#38bdf8","color":"#38bdf8","weight":2,"fillOpacity":0.05}).add_to(m)
        except Exception:
            pass
    for f in facilities or []:
        popup = f"<b>{f['name']}</b><br>Type: {f['type']}<br>Distance: {f['distance_km']} km<br>Address: {f['address']}<br>Phone: {f['phone']}"
        folium.Marker([f["lat"], f["lon"]], tooltip=f"{f['name']} • {f['distance_km']} km", popup=popup, icon=folium.Icon(color=f["color"], icon=f["icon"], prefix="fa")).add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ============================================================
# MAIN
# ============================================================

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the crime dataset: {e}")
    st.stop()

st.markdown("""
<div class="hero">
<h1>🛡️ SWARAKSHA</h1>
<p><b>Know the Pattern. Find the Help.</b> — India-focused historical crime analysis + current GIS/GPS support discovery.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🛡️ Swaraksha")
page = st.sidebar.radio("Navigate", ["📍 Check an Area", "🗺️ GIS Safety Map", "👩 Women Safety", "🆘 Help Nearby", "📈 Future Estimates", "👥 Community Feedback", "ℹ️ Methodology"], index=0)

st.sidebar.markdown("### 📍 Location")
search = st.sidebar.text_input("Search any Indian place / address", placeholder="e.g. Thane West, Maharashtra", key="search_box")
radius = st.sidebar.slider("Nearby support radius (km)", 1, 20, 5)

if "location" not in st.session_state:
    st.session_state.location = {"lat": 22.5, "lon": 79.0, "display_name": "India", "address": {}, "geojson": None}

c1, c2 = st.sidebar.columns(2)
if c1.button("🔎 Search", use_container_width=True):
    result = geocode_india(search)
    if result:
        st.session_state.location = result
        st.session_state.location_message = f"Showing {result['display_name']}"
    else:
        st.session_state.location_message = "Location could not be found. Try a fuller Indian address/city/district name."

if c2.button("📍 My location", use_container_width=True):
    loc = get_browser_location()
    if loc:
        lat, lon = loc
        result = geocode_india(f"{lat}, {lon}")
        if result:
            st.session_state.location = result
        else:
            st.session_state.location = {"lat":lat,"lon":lon,"display_name":f"Current location ({lat:.5f}, {lon:.5f})","address":{},"geojson":None}
        st.session_state.location_message = "Current browser location received."
    else:
        st.session_state.location_message = "Location permission/service is unavailable. Allow browser location or use Search."

if st.session_state.get("location_message"):
    st.sidebar.info(st.session_state.location_message)

loc = st.session_state.location
lat, lon = float(loc["lat"]), float(loc["lon"])
display_name = loc.get("display_name", "Selected location")
state_match, district_match = match_district_from_name(df, display_name)

# Try Nominatim structured address first, then text matching.
addr = loc.get("address", {}) or {}
state_from_addr = addr.get("state") or state_match

district_from_addr = addr.get("state_district") or addr.get("district") or addr.get("county") or district_match
if district_from_addr:
    candidates = sorted(df["district"].dropna().unique(), key=lambda x: len(canonical_district(x)), reverse=True)
    district_match = next((d for d in candidates if canonical_district(d) == canonical_district(district_from_addr)), district_from_addr)
if state_from_addr:
    candidates = sorted(df["state"].dropna().unique(), key=lambda x: len(norm_text(x)), reverse=True)
    state_match = next((s for s in candidates if norm_text(s) == norm_text(state_from_addr)), state_from_addr)

area_df = find_dataset_area(df, state_match, district_match) if state_match and district_match else pd.DataFrame()

# Optional support query. It is cached for 15 minutes and only runs for pages needing it.
selected_facilities = tuple(FACILITY_DEFS.keys())
facilities, facility_error = ([], None)
if page in ["📍 Check an Area", "🗺️ GIS Safety Map", "🆘 Help Nearby"]:
    with st.spinner("Finding current mapped support services nearby…"):
        facilities, facility_error = get_nearby_facilities(lat, lon, radius, selected_facilities)

# ============================================================
# PAGE: CHECK AREA
# ============================================================
if page == "📍 Check an Area":
    st.subheader("📍 Check the selected area")
    st.caption("Search any Indian location or use browser GPS. Current support comes from geographic data; historical crime results come only from the uploaded dataset.")

    if state_match and district_match:
        st.write(f"**Matched area:** {state_match} → {district_match}")
    else:
        st.warning("This location could not be matched confidently to a district in the historical dataset. The live map and current support search can still work; no historical crime rating is fabricated.")

    if not area_df.empty:
        overall, score, factors = overall_pattern(df, area_df)
        category = district_category_levels(df, area_df)
        total_cases = int(sum(area_df[c].sum() for c in CRIME_COLUMNS.values() if c in area_df.columns))
        years = f"{area_df.year.min()}–{area_df.year.max()}"

        a,b,c,d = st.columns(4)
        a.metric("📍 District", district_match)
        b.metric("📅 Data coverage", years)
        c.metric("📊 Women-related records", f"{total_cases:,}")
        d.markdown("**Overall historical pattern**")
        d.markdown(level_html(overall), unsafe_allow_html=True)

        st.markdown('<div class="warning">⚠️ <b>Important:</b> Historical crime patterns do not guarantee present or future safety. The LOW/MODERATE/HIGH label is a historical pattern indicator, not a guarantee or live danger alert.</div>', unsafe_allow_html=True)

        st.markdown("### 🛡️ Particular crime-type historical patterns")
        show = category[["Category","Historical Cases","Percentile","Pattern"]].copy()
        show["Historical comparison"] = show["Percentile"].map(lambda x: f"{x:.1f}th percentile")
        st.dataframe(show[["Category","Historical Cases","Historical comparison","Pattern"]], use_container_width=True, hide_index=True)

        st.markdown("### 🔎 Why did the overall indicator get this pattern?")
        for f in factors:
            st.write("• " + f)

        # Historical trend
        available = [c for c in CRIME_COLUMNS.values() if c in area_df.columns]
        yearly = area_df.groupby("year")[available].sum()
        fig = go.Figure()
        for label, col in CRIME_COLUMNS.items():
            if col in yearly.columns:
                fig.add_trace(go.Scatter(x=yearly.index, y=yearly[col], mode="lines+markers", name=label))
        fig.update_layout(title="Historical women-related crime trends", xaxis_title="Year", yaxis_title="Reported cases", template="plotly_dark", height=500, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        # Category totals bar
        bar = category.sort_values("Historical Cases", ascending=False)
        fig2 = go.Figure(go.Bar(x=bar["Category"], y=bar["Historical Cases"], text=bar["Pattern"], hovertemplate="%{x}<br>Cases: %{y}<br>Pattern: %{text}<extra></extra>"))
        fig2.update_layout(title="Historical category comparison", xaxis_title="Crime category", yaxis_title="Reported cases", template="plotly_dark", height=450)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No matching historical district record was found. You can still use the current GIS support map below.")

    st.markdown("### 🗺️ Current location + support map")
    m = draw_map(lat, lon, 13 if display_name != "India" else 5, facilities, loc.get("geojson"), radius, india_view=(display_name=="India"))
    st_folium(m, use_container_width=True, height=580, key="area_map")

# ============================================================
# PAGE: GIS MAP
# ============================================================
elif page == "🗺️ GIS Safety Map":
    st.subheader("🗺️ India-wide current GIS map")
    st.write(f"**Selected location:** {display_name}")
    st.caption("OpenStreetMap/Leaflet provides the current basemap. Facility markers are based on currently mapped public geographic data and may be incomplete.")
    if facility_error:
        st.warning("The public geographic service is temporarily unavailable. The map itself can still load; try the support search again later.")
    m = draw_map(lat, lon, 13 if display_name != "India" else 5, facilities, loc.get("geojson"), radius, india_view=(display_name=="India"))
    st_folium(m, use_container_width=True, height=650, key="gis_map")
    counts = pd.Series([x["type"] for x in facilities]).value_counts() if facilities else pd.Series(dtype=int)
    cols = st.columns(5)
    for i, typ in enumerate(["Police","Hospital","Health Centre / Clinic","Pharmacy","Fire Station"]):
        cols[i].metric(typ, int(counts.get(typ,0)))
    if facilities:
        st.dataframe(pd.DataFrame(facilities)[["name","type","distance_km","address","phone"]].rename(columns={"name":"Facility","type":"Type","distance_km":"Distance (km)","address":"Address","phone":"Contact"}), use_container_width=True, hide_index=True)

# ============================================================
# PAGE: WOMEN SAFETY
# ============================================================
elif page == "👩 Women Safety":
    st.subheader("👩 Women Safety — historical pattern analysis")
    if area_df.empty:
        st.warning("Select/search a location whose district exists in the uploaded historical dataset to see the crime analysis.")
    else:
        category = district_category_levels(df, area_df)
        st.markdown("Historical pattern is calculated separately for each available women-related category. It compares the selected district's historical annual average with district-level annual averages across the dataset.")
        for _, r in category.iterrows():
            c1,c2,c3 = st.columns([3,1,2])
            c1.write(f"**{r['Category']}**")
            c2.write(level_html(r['Pattern']), unsafe_allow_html=True)
            c3.write(f"{int(r['Historical Cases']):,} reported cases")
        st.markdown("### 📈 Women-related historical trends")
        yearly = area_df.groupby("year")[[c for c in CRIME_COLUMNS.values() if c in area_df.columns]].sum()
        fig = go.Figure()
        for label,col in CRIME_COLUMNS.items():
            if col in yearly:
                fig.add_trace(go.Scatter(x=yearly.index,y=yearly[col],mode="lines+markers",name=label))
        fig.update_layout(template="plotly_dark",height=600,xaxis_title="Year",yaxis_title="Reported cases")
        st.plotly_chart(fig,use_container_width=True)

# ============================================================
# PAGE: HELP NEARBY
# ============================================================
elif page == "🆘 Help Nearby":
    st.subheader("🆘 Find help near this location")
    st.caption("Only current mapped geographic results are displayed. No facility is invented if the public map does not contain it.")
    if facility_error:
        st.error("Current support lookup is temporarily unavailable. Use the emergency numbers in the sidebar if you need immediate assistance.")
    if not facilities:
        st.info("No mapped facilities were returned within the selected radius. Try a larger radius or another location.")
    else:
        grouped = {}
        for f in facilities:
            grouped.setdefault(f["type"], []).append(f)
        for typ, items in grouped.items():
            st.markdown(f"### {items[0]['type']}")
            cols = st.columns(min(3,len(items)))
            for i, item in enumerate(items[:12]):
                with cols[i % len(cols)]:
                    st.markdown(f'<div class="card"><b>{item["name"]}</b><br>📏 {item["distance_km"]} km<br>📍 {item["address"]}<br>☎️ {item["phone"]}</div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Support map")
    m=draw_map(lat,lon,14,facilities,None,radius)
    st_folium(m,use_container_width=True,height=600,key="help_map")

# ============================================================
# PAGE: FUTURE ESTIMATES
# ============================================================
elif page == "📈 Future Estimates":
    st.subheader("📈 Past data + future crime trend estimates")
    if area_df.empty:
        st.warning("Select a district present in the historical dataset first.")
    else:
        st.markdown("Future values are **estimates**, not guarantees. The model uses only the selected district's historical yearly observations and validates on the latest historical years without random shuffling.")
        choices = {label:col for label,col in CRIME_COLUMNS.items() if col in area_df.columns}
        selected = st.selectbox("Crime category", list(choices.keys()), index=0)
        result = forecast_category(area_df, choices[selected], horizon=3)
        if result is None:
            st.warning("Not enough yearly observations for a reliable time-trend estimate for this category.")
        else:
            historical, future, mae = result
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=historical.index,y=historical.values,mode="lines+markers",name="Historical reported cases"))
            fig.add_trace(go.Scatter(x=future.index,y=future.values,mode="lines+markers",name="Future estimate",line=dict(dash="dash")))
            fig.add_vline(x=int(historical.index.max()),line_dash="dot",annotation_text="Last observed year")
            fig.update_layout(title=f"{selected}: historical data and 3-year estimate",xaxis_title="Year",yaxis_title="Reported cases",template="plotly_dark",height=500)
            st.plotly_chart(fig,use_container_width=True)
            st.metric("Time-aware validation MAE", f"{mae:.2f} cases/year")
            st.info("Interpretation: the estimate follows the historical trend. It should not be presented as a guaranteed prediction of future crime.")

        st.markdown("### 🔮 Forecast all available categories")
        summary=[]
        for label,col in choices.items():
            res=forecast_category(area_df,col,3)
            if res:
                hist,fut,mae=res
                summary.append({"Category":label,"Last observed":int(hist.iloc[-1]),"Year +1 estimate":round(float(fut.iloc[0]),1),"Year +2 estimate":round(float(fut.iloc[1]),1),"Year +3 estimate":round(float(fut.iloc[2]),1),"Validation MAE":round(mae,2)})
        if summary:
            st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)

# ============================================================
# PAGE: FEEDBACK
# ============================================================
elif page == "👥 Community Feedback":
    st.subheader("👥 Community Feedback")
    st.caption("Feedback is stored locally in SQLite for future validation/aggregation. A single report does not change the historical crime indicator.")
    categories=["💡 Poor lighting","🏚️ Isolated area","👥 Low public activity","🚌 Transport concern","⚠️ Harassment concern","🏥 Poor access to support","👮 Visible police presence","💡 Good lighting","👥 Active/public area","➕ Other"]
    with st.form("feedback_form"):
        cat=st.selectbox("Observation",categories)
        desc=st.text_area("Description (do not include names, phone numbers or other personal information)")
        rating=st.slider("Optional local-experience rating",1,5,3)
        submitted=st.form_submit_button("Submit observation")
    if submitted:
        con=sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO feedback(timestamp,latitude,longitude,category,description,rating) VALUES(datetime('now'),?,?,?,?,?)",(lat,lon,cat,desc,rating))
        con.commit();con.close()
        st.success("Observation saved. It will not directly change the crime indicator.")

# ============================================================
# PAGE: METHODOLOGY
# ============================================================
elif page == "ℹ️ Methodology":
    st.subheader("ℹ️ Methodology & limitations")
    st.markdown("""
### Data
The application reads the supplied district-level historical crime CSV and detects the available State, District, Year and crime columns. Missing numeric crime values are treated as zero only after numeric conversion; negative values are clipped to zero.

### Historical pattern
For each crime category, the selected district's historical annual average is compared with annual district averages across the dataset. The resulting percentile is grouped into:
- 🟢 **LOW:** below 33rd percentile
- 🟡 **MODERATE:** 33rd–66th percentile
- 🔴 **HIGH:** 67th percentile and above

This is a **historical reported-crime pattern indicator**, not a live safety score.

### Future estimates
A simple time-trend regression is fitted to the selected district's yearly history. The latest historical years are held out for time-aware validation. Future values are estimates and are not guaranteed predictions.

### GIS / GPS
The map uses OpenStreetMap/Leaflet. Search uses Nominatim. Current nearby support discovery uses OpenStreetMap's Overpass API and is cached to reduce requests. Browser GPS requires user permission and connectivity.

### Safety limitation
A district-level dataset cannot honestly identify a specific street as a crime hotspot. Swaraksha therefore does **not** manufacture street-level crime points.

> **Historical crime patterns do not guarantee present or future safety.**
    """)

emergency_panel()
