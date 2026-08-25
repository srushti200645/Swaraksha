# 🛡️ Swaraksha — Know the Pattern. Find the Help.

India-focused Streamlit civic-safety dashboard combining:
- historical district-level crime pattern analysis
- LOW / MODERATE / HIGH category indicators
- India-wide OpenStreetMap/Leaflet map
- browser GPS with permission
- current nearby police, hospitals, clinics, pharmacies and fire stations
- mapped women/child/social support where available
- historical trend charts
- time-aware future trend estimates
- SQLite community feedback

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
Place the supplied CSV at:
`data/raw/cleaned_community_safety_crime_2010_2022.csv`

## Important
Historical crime patterns do not guarantee present or future safety. The LOW/MODERATE/HIGH labels are historical pattern indicators, not live danger alerts.

Current facility data is not taken from the crime CSV and is not hard-coded. It is requested from public OpenStreetMap/Overpass services and may be incomplete or temporarily unavailable.
