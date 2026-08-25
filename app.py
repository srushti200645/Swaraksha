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

# ============================================================
# SWARAKSHA — Know the Pattern. Find the Help.
# Location-first India safety application.
#
# DATA:
#   Your supplied district-level crime dataset is used only for
#   historical district patterns. It is NEVER used as a source
#   of current hospitals/police/facilities.
#
# LIVE GIS:
#   OpenStreetMap/Leaflet map tiles + Nominatim geocoding +
#   Overpass current mapped facilities.
#
# IMPORTANT:
#   No street-level crime hotspot is invented.
#   "LOW/MODERATE/HIGH" means historical pattern level only.
# ============================================================

st.set_page_config(
    page_title="Swaraksha | Know the Pattern. Find the Help.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
USER_AGENT = "Swaraksha-CivicSafety-Streamlit/1.0 (educational project)"

# Government helplines verified against India.gov / PIB sources.
HELPLINES = [
    ("🚨", "Emergency Response Support", "112", "All emergencies"),
    ("👩", "Women Helpline", "181", "Women in distress / support"),
    ("👧", "Child Helpline", "1098", "Children in distress"),
    ("💻", "National Cyber Crime Helpline", "1930", "Cyber crime / financial cyber fraud"),
    ("⚖️", "NALSA Legal Aid", "15100", "Free legal aid"),
    ("🧠", "Tele-MANAS", "14416", "Mental-health support"),
    ("🚑", "National Ambulance Service", "102", "Ambulance"),
    ("🔥", "Fire", "101", "Fire emergency"),
    ("👮", "Police", "100", "Police / legacy number"),
]

# These are deliberately broad parent categories.
# Sub-components such as assault_women_above18/below18 are NOT added
# again when assault_women is present, avoiding double counting.
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

CATEGORY_EMOJI = {
    "Rape": "🔴",
    "Assault on women": "🟠",
    "Insult to modesty": "🟡",
    "Importation of girls": "🟣",
    "Immoral traffic": "🟣",
    "Procuration of minor girls": "🟣",
    "Human trafficking": "🟣",
    "Cyber explicit material": "💻",
    "Other women cyber crimes": "💻",
}

FACILITY_CONFIG = {
    "police": {
        "label": "👮 Police",
        "color": "red",
        "icon": "shield",
        "queries": ['nwr["amenity"="police"]'],
    },
    "hospital": {
        "label": "🏥 Hospital",
        "color": "green",
        "icon": "plus",
        "queries": ['nwr["amenity"="hospital"]'],
    },
    "clinic": {
        "label": "🏥 Clinic / Health Centre",
        "color": "cadetblue",
        "icon": "plus-sign",
        "queries": ['nwr["amenity"="clinic"]', 'nwr["healthcare"="centre"]'],
    },
    "pharmacy": {
        "label": "💊 Pharmacy",
        "color": "purple",
        "icon": "medkit",
        "queries": ['nwr["amenity"="pharmacy"]'],
    },
    "fire": {
        "label": "🔥 Fire Station",
        "color": "orange",
        "icon": "fire",
        "queries": ['nwr["amenity"="fire_station"]'],
    },
    "legal": {
        "label": "⚖️ Legal Support",
        "color": "darkblue",
        "icon": "balance-scale",
        "queries": ['nwr["office"="lawyer"]', 'nwr["amenity"="legal_services"]'],
    },
    "support": {
        "label": "🟣 Women / Child Support",
        "color": "pink",
        "icon": "home",
        "queries": [
            'nwr["name"~"one stop|sakhi|shakti sadan|women shelter|women support",i]'
        ],
    },
}

# ---------- Styling ----------
st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(180deg,#07111f 0%,#0b1220 55%,#0f172a 100%);
    color:#f8fafc;
}
.block-container {padding-top:1.1rem; max-width:1400px;}
.hero {
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(56,189,248,.28);
    border-radius: 22px;
    background: linear-gradient(135deg,rgba(14,165,233,.16),rgba(15,23,42,.72));
    margin-bottom:1rem;
}
.hero h1 {margin:0;color:#f8fafc;font-size:2.2rem;}
.hero p {margin:.35rem 0 0;color:#cbd5e1;font-size:1rem;}
.card {
    background:#111c2d;
    border:1px solid #24344d;
    border-radius:16px;
    padding:16px;
    margin-bottom:12px;
}
.pattern-low {color:#22c55e;font-weight:800;font-size:1.45rem;}
.pattern-mod {color:#f59e0b;font-weight:800;font-size:1.45rem;}
.pattern-high {color:#ef4444;font-weight:800;font-size:1.45rem;}
.small {color:#94a3b8;font-size:.86rem;}
.disclaimer {
    background:#241d0b;
    border:1px solid #7c5b12;
    border-radius:12px;
    padding:11px 14px;
    color:#fde68a;
}
[data-testid="stMetric"] {
    background:#111c2d;
    border:1px solid #263954;
    padding:10px;
    border-radius:14px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def normalize_text(value):
    if pd.isna(value):
        return ""
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
        return None, "Crime CSV was not found. Put it in data/raw/."

    df = pd.read_csv(path)
    # Normalize the actual supplied schema.
    required_base = {"state", "district", "year"}
    missing = required_base - set(df.columns)
    if missing:
        return None, f"CSV is missing required columns: {', '.join(sorted(missing))}"

    df["state"] = df["state"].fillna("UNKNOWN").astype(str).str.strip()
    df["district"] = df["district"].fillna("UNKNOWN").astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    for col in CRIME_CATEGORIES.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # Use only the first occurrence of duplicate-looking columns.
    return df, None


@st.cache_data(ttl=3600, show_spinner=False)
def build_geo_lookup(df):
    states = sorted(df["state"].dropna().unique().tolist())
    districts = sorted(df["district"].dropna().unique().tolist())
    district_norm = {normalize_text(d): d for d in districts}
    state_norm = {normalize_text(s): s for s in states}
    return states, districts, district_norm, state_norm


def fuzzy_best(value, choices, cutoff=0.78):
    n = normalize_text(value)
    if not n:
        return None, 0
    if n in choices:
        return choices[n], 1.0
    best_name, best_score = None, 0.0
    for c_norm, original in choices.items():
        score = SequenceMatcher(None, n, c_norm).ratio()
        if score > best_score:
            best_score, best_name = score, original
    return (best_name, best_score) if best_score >= cutoff else (None, best_score)


@st.cache_data(ttl=900, show_spinner=False)
def geocode_india(query):
    params = {
        "q": f"{query}, India",
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "in",
    }
    try:
        r = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        item = data[0]
        return {
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "display_name": item.get("display_name", query),
            "address": item.get("address", {}),
        }
    except Exception as exc:
        return {"error": str(exc)}


def extract_location_names(geo):
    addr = geo.get("address", {})
    state = addr.get("state") or addr.get("state_district")
    district_candidates = [
        addr.get("state_district"),
        addr.get("county"),
        addr.get("district"),
        addr.get("city_district"),
        addr.get("municipality"),
        addr.get("city"),
        addr.get("town"),
    ]
    district_candidates = [x for x in district_candidates if x]
    return state, district_candidates


def match_geo_to_dataset(geo, district_norm, state_norm):
    state_text, district_candidates = extract_location_names(geo)
    matched_state, state_score = fuzzy_best(state_text, state_norm, cutoff=0.72)

    matched_district, district_score = None, 0
    for candidate in district_candidates:
        m, score = fuzzy_best(candidate, district_norm, cutoff=0.72)
        if m and score > district_score:
            matched_district, district_score = m, score

    # If state is known, prefer a district from that state.
    if matched_state and matched_district:
        return matched_state, matched_district, state_score, district_score

    # Second pass: use the geocoder's full display name.
    display = geo.get("display_name", "")
    m, score = fuzzy_best(display, district_norm, cutoff=0.82)
    return matched_state, m, state_score, score


def level_from_percentile(value, reference):
    """Historical pattern level, not a safety guarantee."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return "🟡 MODERATE"
    percentile = float((ref <= float(value)).mean() * 100)
    if percentile <= 33:
        return "🟢 LOW"
    if percentile <= 66:
        return "🟡 MODERATE"
    return "🔴 HIGH"


def level_class(level):
    if "LOW" in level:
        return "pattern-low"
    if "HIGH" in level:
        return "pattern-high"
    return "pattern-mod"


def district_year_data(df, district):
    return df[df["district"].str.upper() == str(district).upper()].copy()


def category_reference_values(df, category_col):
    # District-year values across the supplied historical dataset.
    # This gives a transparent relative historical context.
    x = pd.to_numeric(df[category_col], errors="coerce").fillna(0)
    return x.values


def category_pattern(df, district_df, category_col):
    if category_col not in df.columns or district_df.empty:
        return None
    years = sorted(district_df["year"].unique())
    recent_years = years[-3:] if len(years) >= 3 else years
    value = float(district_df[district_df["year"].isin(recent_years)][category_col].mean())
    level = level_from_percentile(value, category_reference_values(df, category_col))
    return {
        "value": value,
        "level": level,
        "years_used": recent_years,
    }


def overall_pattern(df, district_df):
    available = [c for c in CRIME_CATEGORIES.values() if c in df.columns]
    if district_df.empty or not available:
        return None

    # Equal-weight category percentiles prevent rape/assault from dominating
    # simply because their counts are numerically larger.
    scores = []
    for c in available:
        ref = category_reference_values(df, c)
        years = sorted(district_df["year"].unique())
        recent = years[-3:] if len(years) >= 3 else years
        value = float(district_df[district_df["year"].isin(recent)][c].mean())
        ref = np.asarray(ref, dtype=float)
        percentile = float((ref <= value).mean() * 100) if len(ref) else 50
        scores.append(percentile)

    composite = float(np.mean(scores)) if scores else 50
    if composite <= 33:
        level = "🟢 LOW"
    elif composite <= 66:
        level = "🟡 MODERATE"
    else:
        level = "🔴 HIGH"

    return {
        "level": level,
        "composite_percentile": composite,
        "years_used": recent,
    }


def forecast_category(district_df, col, years_ahead=3):
    """Simple time-aware linear trend estimate. No random shuffling."""
    if col not in district_df.columns:
        return None

    ts = (
        district_df.groupby("year", as_index=False)[col]
        .sum()
        .sort_values("year")
    )
    ts[col] = pd.to_numeric(ts[col], errors="coerce").fillna(0)
    ts = ts[ts[col].notna()]
    if len(ts) < 5:
        return None

    X = ts[["year"]].values
    y = ts[col].values

    # Time-aware holdout: last 2 observations are validation.
    train = ts.iloc[:-2]
    test = ts.iloc[-2:]
    if len(train) >= 3:
        model_val = LinearRegression().fit(train[["year"]], train[col])
        pred_val = np.clip(model_val.predict(test[["year"]]), 0, None)
        mae = float(np.mean(np.abs(pred_val - test[col].values)))
    else:
        mae = None

    model = LinearRegression().fit(X, y)
    last_year = int(ts["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + years_ahead)
    future = np.clip(
        model.predict(future_years.reshape(-1, 1)),
        0,
        None,
    )

    out = pd.DataFrame({"year": future_years, "estimated_cases": future})
    return out, mae, ts


def address_for_element(tags):
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:district"),
    ]
    return ", ".join([str(x) for x in parts if x])


@st.cache_data(ttl=900, show_spinner=False)
def get_nearby_facilities(lat, lon, radius_km, selected_types):
    selected_types = list(selected_types)
    query_parts = []
    for key in selected_types:
        if key in FACILITY_CONFIG:
            query_parts.extend(
                f"{q}(around:{int(radius_km * 1000)},{lat},{lon});"
                for q in FACILITY_CONFIG[key]["queries"]
            )

    if not query_parts:
        return []

    query = f"""
    [out:json][timeout:25];
    (
      {"".join(query_parts)}
    );
    out center tags;
    """

    elements = []
    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=35,
            )
            r.raise_for_status()
            elements = r.json().get("elements", [])
            break
        except Exception as exc:
            last_error = exc

    if not elements:
        return []

    rows = []
    seen = set()

    def classify(tags):
        amenity = tags.get("amenity", "")
        healthcare = tags.get("healthcare", "")
        office = tags.get("office", "")
        name = (tags.get("name") or "").lower()

        if amenity == "police":
            return "police"
        if amenity == "hospital":
            return "hospital"
        if amenity == "clinic" or healthcare == "centre":
            return "clinic"
        if amenity == "pharmacy":
            return "pharmacy"
        if amenity == "fire_station":
            return "fire"
        if office == "lawyer" or amenity == "legal_services":
            return "legal"
        if any(x in name for x in ["one stop", "sakhi", "shakti sadan", "women shelter", "women support"]):
            return "support"
        return None

    for el in elements:
        tags = el.get("tags", {})
        key = classify(tags)
        if key not in selected_types:
            continue

        lat2 = el.get("lat") or el.get("center", {}).get("lat")
        lon2 = el.get("lon") or el.get("center", {}).get("lon")
        if lat2 is None or lon2 is None:
            continue

        osm_id = (el.get("type"), el.get("id"))
        if osm_id in seen:
            continue
        seen.add(osm_id)

        distance = haversine_km(lat, lon, lat2, lon2)
        cfg = FACILITY_CONFIG[key]
        rows.append(
            {
                "name": tags.get("name") or "Unnamed mapped facility",
                "type": key,
                "type_label": cfg["label"],
                "lat": float(lat2),
                "lon": float(lon2),
                "distance_km": round(distance, 2),
                "address": address_for_element(tags) or "Address not listed in OpenStreetMap",
                "phone": tags.get("phone") or tags.get("contact:phone") or "Not listed",
                "website": tags.get("website") or tags.get("contact:website") or "",
                "source": "OpenStreetMap / Overpass",
            }
        )

    rows.sort(key=lambda x: x["distance_km"])
    return rows


def make_map(lat, lon, zoom, facilities, searched_name=None):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    Fullscreen(position="topright").add_to(m)

    folium.Marker(
        [lat, lon],
        tooltip="📍 Selected / current location",
        popup=f"<b>📍 Selected location</b><br>{searched_name or 'Current location'}",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)

    for f in facilities:
        cfg = FACILITY_CONFIG.get(f["type"], {})
        popup = (
            f"<b>{f['name']}</b><br>"
            f"{f['type_label']}<br>"
            f"Distance: {f['distance_km']} km<br>"
            f"Address: {f['address']}<br>"
            f"Phone: {f['phone']}"
        )
        folium.Marker(
            [f["lat"], f["lon"]],
            tooltip=f"{f['name']} • {f['distance_km']} km",
            popup=popup,
            icon=folium.Icon(
                color=cfg.get("color", "blue"),
                icon=cfg.get("icon", "info-sign"),
                prefix="fa",
            ),
        ).add_to(m)

    return m


# ---------- Load dataset ----------
crime_df, data_error = load_crime_data()
if crime_df is None:
    st.error(data_error)
    st.stop()

states, districts, district_norm, state_norm = build_geo_lookup(crime_df)

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
<h1>🛡️ SWARAKSHA</h1>
<p><b>Know the Pattern. Find the Help.</b> — India-focused location-first community safety support.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">⚠️ <b>Important:</b> Historical crime patterns do not guarantee present or future safety. '
    'Swaraksha does not declare a place safe or dangerous and does not dispatch emergency services.</div>',
    unsafe_allow_html=True,
)

# ---------- Location controls ----------
st.subheader("📍 Check an area")
c1, c2 = st.columns([3, 1])
with c1:
    search = st.text_input(
        "Search any Indian city, district, locality, landmark or address",
        placeholder="Example: Thane West, Maharashtra",
        key="location_search",
    )
with c2:
    radius = st.slider("Help radius (km)", 1, 15, 5)

search_clicked = st.button("🔎 Check this location", type="primary", use_container_width=True)

st.caption("You can search a place manually or use your device's browser location. Current facilities come from live mapped geographic data when available.")

geo = None

# Browser geolocation component.
try:
    browser_location = streamlit_geolocation()
except Exception:
    browser_location = None

if browser_location and browser_location.get("latitude") is not None:
    if st.button("📍 Use my current location", use_container_width=True):
        st.session_state["selected_lat"] = float(browser_location["latitude"])
        st.session_state["selected_lon"] = float(browser_location["longitude"])
        st.session_state["selected_name"] = "My current location"
        st.session_state["selected_geo"] = {
            "lat": st.session_state["selected_lat"],
            "lon": st.session_state["selected_lon"],
            "display_name": "My current location",
            "address": {},
        }
        st.rerun()

if search_clicked:
    if not search.strip():
        st.warning("Enter an Indian location first.")
    else:
        with st.spinner("Finding the location..."):
            result = geocode_india(search.strip())
        if result and "error" not in result:
            st.session_state["selected_lat"] = result["lat"]
            st.session_state["selected_lon"] = result["lon"]
            st.session_state["selected_name"] = result["display_name"]
            st.session_state["selected_geo"] = result
            st.rerun()
        elif result and "error" in result:
            st.error("The location service is temporarily unavailable. Try again shortly.")
        else:
            st.warning("Location not found. Try a more specific Indian address/locality.")

if "selected_lat" not in st.session_state:
    # India-wide starting map.
    selected_lat, selected_lon = 22.9734, 78.6569
    selected_name = "India"
    selected_geo = None
    default_zoom = 5
else:
    selected_lat = st.session_state["selected_lat"]
    selected_lon = st.session_state["selected_lon"]
    selected_name = st.session_state.get("selected_name", "Selected location")
    selected_geo = st.session_state.get("selected_geo")
    default_zoom = 13

# ---------- Location summary ----------
if selected_geo:
    matched_state, matched_district, state_score, district_score = match_geo_to_dataset(
        selected_geo, district_norm, state_norm
    )
else:
    matched_state = matched_district = None
    state_score = district_score = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("📍 Location", "Selected" if selected_geo else "India map")
col2.metric("🗺️ State", matched_state or "Not matched")
col3.metric("🏙️ District", matched_district or "Not matched")
col4.metric("📅 Crime data", f"{crime_df['year'].min()}–{crime_df['year'].max()}")

if selected_geo:
    st.info(f"📍 {selected_name}")

# ---------- Historical pattern ----------
st.subheader("🛡️ Historical women-related crime pattern")

if matched_district:
    ddf = district_year_data(crime_df, matched_district)
    overall = overall_pattern(crime_df, ddf)

    if overall:
        st.markdown(
            f'<div class="card"><div class="{level_class(overall["level"])}">'
            f'{overall["level"]}</div>'
            f'<div class="small">Relative historical pattern for {matched_district}. '
            f'Computed from the latest available up to 3 years in the supplied district-level dataset '
            f'({overall["years_used"][0]}–{overall["years_used"][-1]}). '
            f'It is not a present-day safety guarantee.</div></div>',
            unsafe_allow_html=True,
        )

        # Category cards.
        available_results = []
        for label, col in CRIME_CATEGORIES.items():
            result = category_pattern(crime_df, ddf, col)
            if result:
                available_results.append((label, col, result))

        cols = st.columns(3)
        for i, (label, col, result) in enumerate(available_results):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="card"><b>{CATEGORY_EMOJI.get(label,"📌")} {label}</b><br>'
                    f'<span class="{level_class(result["level"])}">{result["level"]}</span><br>'
                    f'<span class="small">Recent-period average: {result["value"]:.1f} reported cases</span></div>',
                    unsafe_allow_html=True,
                )

        st.caption(
            "The LOW/MODERATE/HIGH label is a relative historical indicator. "
            "It compares the selected district's recent reported level with district-year values in the supplied dataset. "
            "It does not estimate the probability that an individual will be harmed."
        )

        # Historical trend chart.
        trend_cols = [c for c in CRIME_CATEGORIES.values() if c in ddf.columns]
        trend = ddf.groupby("year", as_index=False)[trend_cols].sum()
        long_trend = trend.melt(
            id_vars="year",
            var_name="crime",
            value_name="reported_cases",
        )
        name_map = {v: k for k, v in CRIME_CATEGORIES.items()}
        long_trend["crime"] = long_trend["crime"].map(name_map).fillna(long_trend["crime"])
        fig = px.line(
            long_trend,
            x="year",
            y="reported_cases",
            color="crime",
            markers=True,
            title=f"Historical reported women-related crime trends — {matched_district}",
        )
        fig.update_layout(template="plotly_dark", height=430, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        # Category comparison.
        latest_year = int(ddf["year"].max())
        latest = ddf[ddf["year"] == latest_year][trend_cols].sum().sort_values(ascending=False)
        latest_df = latest.reset_index()
        latest_df.columns = ["crime", "reported_cases"]
        latest_df["crime"] = latest_df["crime"].map(name_map).fillna(latest_df["crime"])
        fig2 = px.bar(
            latest_df,
            x="reported_cases",
            y="crime",
            orientation="h",
            title=f"Reported category comparison — {latest_year}",
        )
        fig2.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("The district was matched, but no usable historical crime records were found.")
else:
    st.info(
        "ℹ️ Search a location or use current location to get the district-level historical pattern. "
        "The map and current support search remain available even when historical district matching is unavailable."
    )

# ---------- Future estimates ----------
st.subheader("📈 Future crime trend estimates")

if matched_district:
    ddf = district_year_data(crime_df, matched_district)
    forecast_choices = [
        (label, col)
        for label, col in CRIME_CATEGORIES.items()
        if col in ddf.columns and ddf[col].sum() > 0
    ]
    if forecast_choices:
        selected_forecast = st.selectbox(
            "Choose a crime category for a historical-trend estimate",
            [x[0] for x in forecast_choices],
        )
        selected_col = dict(forecast_choices)[selected_forecast]
        forecast_result = forecast_category(ddf, selected_col, years_ahead=3)

        if forecast_result:
            forecast_df, mae, history_df = forecast_result
            st.caption(
                "These are model-based estimates from past district data, not predictions of individual risk. "
                "The model uses a time-aware holdout and does not randomly shuffle years."
            )

            combined = history_df[["year", selected_col]].copy()
            combined.columns = ["year", "reported_cases"]
            combined["series"] = "Historical reported cases"
            future_plot = forecast_df.rename(columns={"estimated_cases": "reported_cases"}).copy()
            future_plot["series"] = "Model estimate"
            plot_df = pd.concat([combined, future_plot], ignore_index=True)

            fig3 = px.line(
                plot_df,
                x="year",
                y="reported_cases",
                color="series",
                markers=True,
                title=f"Historical data + next 3-year estimate — {selected_forecast}",
            )
            fig3.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig3, use_container_width=True)

            if mae is not None:
                st.write(f"Time-aware validation MAE on the last 2 observed years: **{mae:.2f} cases**.")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough historical observations to produce a responsible trend estimate for this category.")
    else:
        st.info("No usable category is available for forecasting in this district.")
else:
    st.info("Future estimates appear after a location is matched to a district in the supplied dataset.")

# ---------- Current help ----------
st.subheader("🆘 Find current help near this location")

facility_options = st.multiselect(
    "Choose the support types to show on the live map",
    list(FACILITY_CONFIG.keys()),
    default=["police", "hospital", "clinic", "pharmacy", "fire", "legal", "support"],
    format_func=lambda x: FACILITY_CONFIG[x]["label"],
)

with st.spinner("Checking current mapped support around the selected location..."):
    facilities = get_nearby_facilities(
        selected_lat, selected_lon, radius, tuple(facility_options)
    )

m = make_map(
    selected_lat,
    selected_lon,
    default_zoom,
    facilities,
    selected_name,
)
map_result = st_folium(
    m,
    width="100%",
    height=600,
    returned_objects=["last_clicked"],
)

if facilities:
    st.success(f"Found {len(facilities)} currently mapped support locations within approximately {radius} km.")
    fdf = pd.DataFrame(facilities)
    display = fdf[["type_label", "name", "distance_km", "address", "phone", "website"]].copy()
    display.columns = ["Type", "Name", "Distance (km)", "Address", "Phone", "Website"]
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.warning(
        "No matching mapped facilities were returned for this search radius. "
        "This does NOT mean that no facility exists; it may simply not be mapped or the public geographic service may be temporarily unavailable."
    )

st.caption(
    "Current facility results are sourced from OpenStreetMap/Overpass. "
    "They are separate from the historical crime CSV and are not guaranteed to be complete."
)

# ---------- Helplines ----------
st.subheader("🚨 Emergency & support helplines")

cols = st.columns(3)
for i, (emoji, name, number, purpose) in enumerate(HELPLINES):
    with cols[i % 3]:
        st.markdown(
            f'<div class="card"><b>{emoji} {name}</b><br>'
            f'<span style="font-size:1.4rem;font-weight:800">{number}</span><br>'
            f'<span class="small">{purpose}</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    "If there is an immediate emergency, use **112** or the appropriate official emergency service. "
    "Swaraksha itself does not dispatch police, ambulance, fire or other responders.",
)

# ---------- Lightweight feedback ----------
st.subheader("👥 Community feedback")
st.caption(
    "Feedback is stored without names, phone numbers or other unnecessary personal identifiers. "
    "A single report does not change the historical indicator."
)

feedback_categories = [
    "Poor lighting",
    "Isolated area",
    "Low public activity",
    "Transport concern",
    "Harassment concern",
    "Poor access to support",
    "Visible police presence",
    "Good lighting",
    "Active/public area",
    "Other",
]

with st.form("feedback_form"):
    fb_category = st.selectbox("Observation", feedback_categories)
    fb_rating = st.slider("Optional rating", 1, 5, 3)
    fb_description = st.text_area("Optional description", max_chars=500)
    submitted = st.form_submit_button("Submit anonymous feedback")

if submitted:
    try:
        __import__("os").makedirs("database", exist_ok=True)
        conn = sqlite3.connect("database/swaraksha_feedback.db")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                category TEXT NOT NULL,
                rating INTEGER,
                description TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO feedback
            (timestamp_utc, latitude, longitude, category, rating, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                float(selected_lat) if selected_geo else None,
                float(selected_lon) if selected_geo else None,
                fb_category,
                int(fb_rating),
                fb_description.strip(),
            ),
        )
        conn.commit()
        conn.close()
        st.success("Thank you. Your anonymous observation was recorded.")
    except Exception:
        st.error("Feedback could not be saved on this deployment.")

# ---------- Methodology ----------
with st.expander("ℹ️ How Swaraksha calculates the historical indicator"):
    st.markdown(
        """
**What the indicator means**

- It is a **historical district-level pattern indicator**, not a live danger detector.
- The app uses the supplied crime CSV's actual women-related columns.
- For each category, it uses the average reported count across the latest available up to three years for the matched district.
- That value is compared with district-year values in the supplied dataset to obtain a relative percentile.
- LOW / MODERATE / HIGH are assigned from that relative historical position.
- The overall indicator gives equal weight to the available categories rather than allowing a high-count category such as assault to dominate every result.
- No street-level crime hotspot is fabricated because the crime data is district-level.

**Future estimates**

- A simple linear time-trend model is fitted separately to a selected category.
- The last two observed years are held out for a time-aware validation check.
- Future values are clipped at zero.
- Forecasts are labelled **estimates** because the supplied data is historical and does not contain every factor that affects crime.

**Current support**

- The map uses OpenStreetMap.
- Search uses Nominatim.
- Nearby facility lookup uses Overpass.
- These services are public geographic sources and may be incomplete or temporarily rate-limited.
        """
    )

st.caption(
    "Swaraksha • Data coverage in supplied CSV: "
    f"{crime_df['year'].min()}–{crime_df['year'].max()} • "
    f"{crime_df['state'].nunique()} state/UT labels • "
    f"{crime_df['district'].nunique()} district labels."
)
