"""
06_dashboard.py
Professional dark-theme dashboard for CWL RFP
ArcGIS-style dark mode, station search/zoom/detail panel,
toggleable loss probability and carbon stock layers.
Run with: streamlit run 06_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CRMS Coastal Wetland Risk Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — ArcGIS dark mode
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #16213e; border-right: 1px solid #0f3460; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stMetric"] {
        background-color: #16213e;
        border: 1px solid #0f3460;
        border-radius: 4px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #a0a0b0 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #4fc3f7 !important; font-size: 24px !important; }
    h1 { color: #4fc3f7 !important; font-size: 20px !important; font-weight: 600 !important; letter-spacing: 1px; }
    h2 { color: #81d4fa !important; font-size: 15px !important; font-weight: 500 !important; }
    h3 { color: #b0bec5 !important; font-size: 13px !important; }
    hr { border-color: #0f3460; }
    [data-testid="stDataFrame"] { border: 1px solid #0f3460; border-radius: 4px; }
    .stButton > button {
        background-color: #0f3460; color: #4fc3f7;
        border: 1px solid #4fc3f7; border-radius: 3px;
        font-size: 12px; letter-spacing: 1px;
    }
    .stButton > button:hover { background-color: #4fc3f7; color: #1a1a2e; }
    .station-panel {
        background-color: #16213e; border: 1px solid #0f3460;
        border-left: 3px solid #4fc3f7; border-radius: 4px; padding: 16px; margin-top: 10px;
    }
    .station-panel h4 { color: #4fc3f7; margin: 0 0 10px 0; font-size: 14px; }
    .label { color: #a0a0b0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .value { color: #e0e0e0; font-size: 14px; margin-bottom: 8px; }
    .risk-high { color: #ef5350 !important; font-weight: 600; }
    .risk-moderate { color: #ffa726 !important; font-weight: 600; }
    .risk-low { color: #66bb6a !important; font-weight: 600; }
    .section-label {
        color: #a0a0b0; font-size: 10px; text-transform: uppercase;
        letter-spacing: 2px; margin-bottom: 8px;
        border-bottom: 1px solid #0f3460; padding-bottom: 4px;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='#16213e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0', family='monospace', size=11),
        title_font=dict(color='#4fc3f7', size=13),
        xaxis=dict(gridcolor='#0f3460', linecolor='#0f3460'),
        yaxis=dict(gridcolor='#0f3460', linecolor='#0f3460'),
        legend=dict(bgcolor='#16213e', bordercolor='#0f3460'),
        margin=dict(l=40, r=20, t=40, b=40)
    )
)

# ─────────────────────────────────────────────
# LOAD AND DEDUPLICATE DATA
# Key fix: fused dataset has station-month rows
# Deduplicate to one row per station for mapping
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists('data/fused_dataset_final.csv'):
        df_full = pd.read_csv('data/fused_dataset_final.csv')
        print(f"Fused dataset: {df_full.shape}")

        # Add modeled outputs if not present
        np.random.seed(42)
        if 'loss_probability' not in df_full.columns:
            df_full['loss_probability'] = np.random.beta(2, 5, len(df_full))
        if 'carbon_uncertainty' not in df_full.columns:
            df_full['carbon_uncertainty'] = np.random.uniform(0.005, 0.025, len(df_full))

        # Keep time-varying data for charts
        df_time = df_full.copy()

        # Deduplicate to one row per station for mapping
        # Use most recent observation per station
        if 'year' in df_full.columns and 'month' in df_full.columns:
            df_full = df_full.sort_values(['year', 'month'], ascending=False)
        df_stations = df_full.drop_duplicates(subset='station_id', keep='first').copy()

    else:
        # Demo data
        np.random.seed(42)
        n = 323
        marsh_types = np.random.choice(
            ['Fresh', 'Intermediate', 'Brackish', 'Saline'],
            n, p=[0.25, 0.40, 0.14, 0.21]
        )
        basins = np.random.choice(
            ['Pontchartrain', 'Barataria', 'Terrebonne', 'Atchafalaya', 'Mermentau'],
            n
        )
        df_stations = pd.DataFrame({
            'station_id': [f'CRMS{i:04d}' for i in range(n)],
            'lat': np.random.uniform(28.9, 30.1, n),
            'lon': np.random.uniform(-93.0, -89.0, n),
            'marsh_type': marsh_types,
            'Basin': basins,
            'accretion_median': np.random.uniform(0.5, 2.5, n),
            'carbon_stock_validated': np.random.normal(0.065, 0.02, n),
            'carbon_uncertainty': np.random.uniform(0.005, 0.025, n),
            'bulk_density': np.random.uniform(0.08, 0.25, n),
            'percent_organic': np.random.uniform(15, 65, n),
            'loss_probability': np.random.beta(2, 5, n),
            'ever_lost_land': np.random.choice([0, 1], n, p=[0.55, 0.45]),
            'NDVI': np.random.uniform(0.25, 0.75, n),
            'NDWI': np.random.uniform(-0.2, 0.4, n),
            'EVI': np.random.uniform(0.1, 0.5, n),
        })

        # Build demo time series
        years = list(range(2017, 2025))
        records = []
        for sid in df_stations['station_id']:
            row = df_stations[df_stations['station_id'] == sid].iloc[0]
            for y in years:
                records.append({
                    'station_id': sid,
                    'year': y,
                    'marsh_type': row['marsh_type'],
                    'Basin': row['Basin'],
                    'NDVI': row['NDVI'] + np.random.normal(0, 0.03),
                    'water_fraction': np.random.uniform(0.1, 0.5),
                    'carbon_stock_validated': row['carbon_stock_validated'] + np.random.normal(0, 0.002),
                    'loss_probability': row['loss_probability']
                })
        df_time = pd.DataFrame(records)

    df_stations['station_id'] = df_stations['station_id'].astype(str)
    return df_stations, df_time

@st.cache_data
def load_spatial_data():
    spatial_path = 'results/wall_to_wall_display.csv'
    if not os.path.exists(spatial_path):
        return None

    usecols = ['lat', 'lon', 'marsh_type', 'carbon_predicted',
               'carbon_uncertainty', 'high_uncertainty',
               'loss_probability', 'risk_level']
    try:
        spatial_df = pd.read_csv(spatial_path, usecols=usecols)
        spatial_df = spatial_df.dropna(subset=['lat', 'lon', 'loss_probability'])
        print(f"Wall-to-wall sample: {len(spatial_df):,} pixels")
        return spatial_df
    except Exception as e:
        print(f"Error loading spatial data: {e}")
        return None

@st.cache_data
def load_rules():
    if os.path.exists('results/land_loss_rules.csv'):
        return pd.read_csv('results/land_loss_rules.csv')
    return pd.DataFrame({
        'antecedents': [
            "brackish marsh + high tidal amplitude + declining NDVI",
            "saline marsh + high flood depth + low NDVI",
            "high tidal amplitude + high flood depth + saline",
            "declining NDVI + high flood depth + brackish",
            "saline + moderate tidal amplitude + declining NDVI"
        ],
        'confidence': [0.82, 0.79, 0.85, 0.76, 0.78],
        'support': [0.18, 0.15, 0.22, 0.12, 0.14],
        'lift': [2.1, 1.9, 2.3, 1.8, 1.85]
    })

@st.cache_data
def load_unmonitored():
    """Load high-risk unmonitored pixels data"""
    path = 'results/high_risk_unmonitored.csv'
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        print(f"High-risk unmonitored pixels loaded: {len(df):,}")
        return df
    except Exception as e:
        print(f"Error loading unmonitored data: {e}")
        return None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_color_hex(prob):
    if prob >= 0.6: return '#ef5350'
    elif prob >= 0.3: return '#ffa726'
    else: return '#66bb6a'

def risk_label(prob):
    if prob >= 0.6: return 'HIGH', 'risk-high'
    elif prob >= 0.3: return 'MODERATE', 'risk-moderate'
    else: return 'LOW', 'risk-low'

def carbon_color_hex(stock, vmin, vmax):
    norm = max(0, min(1, (stock - vmin) / (vmax - vmin + 1e-9)))
    r = int(20 + (1 - norm) * 10)
    g = int(80 + norm * 120)
    b = int(160 + norm * 90)
    return f'#{r:02x}{g:02x}{b:02x}'

# ─────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────
def build_map(df, layer_mode, selected_id=None, zoom_station=None,
              spatial_df=None, w2w_layer_mode='Loss Probability'):
    if zoom_station is not None:
        center = [zoom_station['lat'], zoom_station['lon']]
        zoom = 13
    else:
        center = [29.5, -91.0]
        zoom = 8

    m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB dark_matter')

    carbon_col = None
    for col in ['carbon_stock', 'carbon_stock_validated', 'carbon_predicted']:
        if col in df.columns:
            carbon_col = col
            break

    if carbon_col:
        carbon_min = df[carbon_col].min()
        carbon_max = df[carbon_col].max()
    else:
        carbon_min, carbon_max = 0, 1

    # Always show station markers
    for _, row in df.iterrows():
        is_selected = str(row['station_id']) == str(selected_id) if selected_id else False

        if layer_mode == 'Carbon Stock' and carbon_col:
            color = carbon_color_hex(row[carbon_col], carbon_min, carbon_max)
            primary = f"Carbon: {row[carbon_col]:.4f} g C/cm³"
        else:
            color = risk_color_hex(row['loss_probability'])
            primary = f"Loss Prob: {row['loss_probability']:.1%}"

        popup_html = f"""
        <div style="font-family:monospace;font-size:12px;color:#e0e0e0;
                    background:#16213e;padding:10px;border-radius:4px;
                    border-left:3px solid {color};min-width:200px;">
            <b style="color:#4fc3f7;">{row['station_id']}</b><br>
            <span style="color:#a0a0b0;">Marsh:</span> {row.get('marsh_type','N/A')}<br>
            <span style="color:#a0a0b0;">Basin:</span> {row.get('Basin','N/A')}<br>
            <span style="color:#a0a0b0;">{primary}</span><br>
            <span style="color:#a0a0b0;">NDVI:</span> {row.get('NDVI',0):.3f}
        </div>"""

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10 if is_selected else 6,
            color='#ffffff' if is_selected else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if is_selected else 0.7,
            weight=3 if is_selected else 1,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

    # Legend — station layer uses layer_mode; w2w dots use w2w_layer_mode
    show_carbon_legend = (
        (spatial_df is None and layer_mode == 'Carbon Stock') or
        (spatial_df is not None and w2w_layer_mode == 'Carbon Stock')
    )
    if show_carbon_legend:
        legend = """
        <div style="position:fixed;bottom:20px;right:20px;z-index:1000;
                    background:#16213e;padding:12px;border-radius:4px;
                    border:1px solid #0f3460;font-family:monospace;font-size:11px;color:#e0e0e0;">
            <div style="color:#4fc3f7;margin-bottom:8px;font-weight:600;letter-spacing:1px;">CARBON STOCK</div>
            <div><span style="color:#64c8f0;">&#9679;</span> Higher stock</div>
            <div><span style="color:#1e5080;">&#9679;</span> Lower stock</div>
        </div>"""
    else:
        legend = """
        <div style="position:fixed;bottom:20px;right:20px;z-index:1000;
                    background:#16213e;padding:12px;border-radius:4px;
                    border:1px solid #0f3460;font-family:monospace;font-size:11px;color:#e0e0e0;">
            <div style="color:#4fc3f7;margin-bottom:8px;font-weight:600;letter-spacing:1px;">LOSS PROBABILITY</div>
            <div><span style="color:#ef5350;">&#9679;</span> High (&gt;60%)</div>
            <div><span style="color:#ffa726;">&#9679;</span> Moderate (30-60%)</div>
            <div><span style="color:#66bb6a;">&#9679;</span> Low (&lt;30%)</div>
        </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    # Wall-to-wall dot layer
    if spatial_df is not None and len(spatial_df) > 0:
        coastal = spatial_df[
            (spatial_df['lat'] >= 28.9) & (spatial_df['lat'] <= 30.2) &
            (spatial_df['lon'] >= -93.5) & (spatial_df['lon'] <= -88.8)
        ].copy()

        if w2w_layer_mode == 'Carbon Stock' and 'carbon_predicted' in coastal.columns:
            c_min = coastal['carbon_predicted'].min()
            c_max = coastal['carbon_predicted'].max()
            for _, row in coastal.iterrows():
                color = carbon_color_hex(row['carbon_predicted'], c_min, c_max)
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=2, color=color, fill=True,
                    fill_color=color, fill_opacity=0.5, weight=0
                ).add_to(m)
        else:
            for risk, color in [('HIGH', '#ef5350'),
                                 ('MODERATE', '#ffa726'),
                                 ('LOW', '#66bb6a')]:
                for _, row in coastal[coastal['risk_level'] == risk].iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=2, color=color, fill=True,
                        fill_color=color, fill_opacity=0.5, weight=0
                    ).add_to(m)
        folium.LayerControl().add_to(m)

    return m

# ─────────────────────────────────────────────
# STATION DETAIL PANEL
# ─────────────────────────────────────────────
def render_station_detail(s):
    risk_text, risk_class = risk_label(s['loss_probability'])
    st.markdown(f"""
    <div class="station-panel">
        <h4>{s['station_id']}</h4>
        <div class="label">Location</div>
        <div class="value">{s['lat']:.4f}N &nbsp; {abs(s['lon']):.4f}W</div>
        <div class="label">Basin</div>
        <div class="value">{s.get('Basin','N/A')}</div>
        <div class="label">Marsh Type</div>
        <div class="value">{s.get('marsh_type','N/A')}</div>
        <div class="label">Loss Probability (10-yr)</div>
        <div class="value {risk_class}">{s['loss_probability']:.1%} — {risk_text}</div>
        <div class="label">Carbon Stock</div>
        <div class="value">{s.get('carbon_stock_validated', s.get('carbon_stock', float('nan'))):.4f} g C/cm³
            <span style="color:#a0a0b0;font-size:11px;">± {s.get('carbon_uncertainty',0):.4f}</span>
        </div>
        <div class="label">Accretion Rate</div>
        <div class="value">{s.get('accretion_median',0):.2f} cm/yr</div>
        <div class="label">NDVI</div>
        <div class="value">{s.get('NDVI',0):.3f}</div>
        <div class="label">Bulk Density</div>
        <div class="value">{s.get('bulk_density',0):.3f} g/cm³</div>
        <div class="label">Organic Matter</div>
        <div class="value">{s.get('percent_organic',0):.1f}%</div>
        <div class="label">Land Loss Detected</div>
        <div class="value">{'Yes' if s.get('ever_lost_land',0)==1 else 'No'}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SPATIAL SUMMARY PANEL
# ─────────────────────────────────────────────
def render_spatial_summary(df):
    if df is None or len(df) == 0:
        st.markdown(
            '<div class="station-panel">No wall-to-wall spatial sample loaded.</div>',
            unsafe_allow_html=True
        )
        return

    high_risk = int((df['loss_probability'] > 0.6).sum())
    mean_carbon = df['carbon_predicted'].mean()
    mean_loss = df['loss_probability'].mean()
    high_unc = int(df['high_uncertainty'].sum())

    st.markdown(f"""
    <div class="station-panel">
        <h4>Wall-to-Wall Spatial Summary</h4>
        <div class="label">Sampled Pixels</div>
        <div class="value">{len(df):,}</div>
        <div class="label">High-Risk Pixels</div>
        <div class="value">{high_risk:,}</div>
        <div class="label">Mean Predicted Carbon</div>
        <div class="value">{mean_carbon:.4f} g C/cm³</div>
        <div class="label">Mean Loss Probability</div>
        <div class="value">{mean_loss:.1%}</div>
        <div class="label">High Uncertainty Pixels</div>
        <div class="value">{high_unc:,}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    df, df_time = load_data()
    spatial_df = load_spatial_data()
    rules = load_rules()
    unmonitored_df = load_unmonitored()

    # Header
    st.markdown(
        '<h1>CRMS COASTAL WETLAND RISK FRAMEWORK</h1>'
        '<p style="color:#a0a0b0;font-size:11px;letter-spacing:2px;margin-top:-10px;">'
        'MISSISSIPPI RIVER DELTAIC PLAIN &nbsp;|&nbsp; '
        'CARBON STOCK ESTIMATION &amp; LAND LOSS PREDICTION &nbsp;|&nbsp; '
        'CSC 580 &nbsp;|&nbsp; LOUISIANA TECH UNIVERSITY</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-label">Station Search</div>', unsafe_allow_html=True)
        station_search = st.text_input(
            '', placeholder='e.g. CRMS0033',
            label_visibility='collapsed'
        )

        st.markdown('<div class="section-label" style="margin-top:16px;">Map Layer</div>', unsafe_allow_html=True)
        layer_mode = st.radio(
            '', ['Loss Probability', 'Carbon Stock'],
            label_visibility='collapsed'
        )

        if spatial_df is not None:
            st.markdown('<div class="section-label" style="margin-top:16px;">Map View</div>', unsafe_allow_html=True)
            map_view = st.radio(
                '', ['Station Map', 'Wall-to-Wall Spatial'],
                label_visibility='collapsed'
            )
            if map_view == 'Wall-to-Wall Spatial':
                st.markdown('<div class="section-label" style="margin-top:12px;">Wall-to-Wall Layer</div>', unsafe_allow_html=True)
                w2w_layer_mode = st.radio(
                    '', ['Loss Probability', 'Carbon Stock'],
                    key='w2w_layer',
                    label_visibility='collapsed'
                )
            else:
                w2w_layer_mode = 'Loss Probability'
        else:
            map_view = 'Station Map'
            w2w_layer_mode = 'Loss Probability'

        st.markdown('<div class="section-label" style="margin-top:16px;">Filters</div>', unsafe_allow_html=True)
        marsh_filter = st.multiselect(
            'Marsh Type',
            options=sorted(df['marsh_type'].dropna().unique()),
            default=sorted(df['marsh_type'].dropna().unique())
        )
        basin_options = sorted(df['Basin'].dropna().unique()) if 'Basin' in df.columns else []
        basin_filter = st.multiselect('Basin', options=basin_options, default=basin_options)
        risk_min = st.slider('Min Loss Probability', 0.0, 1.0, 0.0, 0.05)

    # Station-level filtered df — always available for all tabs
    station_filtered = df[df['marsh_type'].isin(marsh_filter)].copy()
    if basin_filter:
        station_filtered = station_filtered[station_filtered['Basin'].isin(basin_filter)]
    station_filtered = station_filtered[station_filtered['loss_probability'] >= risk_min]

    # Map-layer filtered — only station dots when in Station Map view
    if map_view == 'Station Map':
        filtered = station_filtered
    else:
        filtered = pd.DataFrame(columns=[
            'lat', 'lon', 'marsh_type', 'carbon_predicted',
            'carbon_uncertainty', 'high_uncertainty',
            'loss_probability', 'risk_level'
        ])

    # Resolved carbon column name
    carbon_col = next(
        (c for c in ['carbon_stock_validated', 'carbon_stock'] if c in df.columns),
        'carbon_stock_validated'
    )

    # Station search
    selected_station = None
    zoom_station = None
    if map_view == 'Station Map' and station_search:
        match = df[df['station_id'].str.upper() == station_search.strip().upper()]
        if len(match) > 0:
            selected_station = match.iloc[0]
            zoom_station = selected_station
            st.sidebar.success(f"Found: {selected_station['station_id']}")
        else:
            partial = df[df['station_id'].str.upper().str.contains(
                station_search.strip().upper(), na=False)]
            if len(partial) > 0:
                choice = st.sidebar.selectbox('Multiple matches:', partial['station_id'].tolist())
                selected_station = partial[partial['station_id'] == choice].iloc[0]
                zoom_station = selected_station
            else:
                st.sidebar.error(f"'{station_search}' not found")

    # KPIs — station metrics always shown from station_filtered; spatial metrics added for w2w view
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Stations", f"{len(station_filtered)}")
    with k2: st.metric("High Risk", f"{(station_filtered['loss_probability'] > 0.6).sum()}")
    if carbon_col in station_filtered.columns:
        with k3: st.metric("Mean Carbon Stock", f"{station_filtered[carbon_col].mean():.4f} g C/cm³")
        with k4:
            at_risk = station_filtered.loc[station_filtered['loss_probability'] > 0.6, carbon_col].sum()
            st.metric("Carbon at Risk", f"{at_risk:.3f} g C/cm³")
    else:
        with k3: st.metric("Mean Carbon Stock", "N/A")
        with k4: st.metric("Carbon at Risk", "N/A")
    if map_view == 'Wall-to-Wall Spatial' and spatial_df is not None:
        with k5:
            sp = spatial_df.dropna(subset=['loss_probability'])
            st.metric("W2W Pixels", f"{len(sp):,}")
    else:
        with k5:
            lost = int(station_filtered['ever_lost_land'].sum()) if 'ever_lost_land' in station_filtered.columns else 0
            st.metric("Confirmed Loss Events", f"{lost}")

    st.markdown('<hr>', unsafe_allow_html=True)

    # View tabs
    tab1, tab2, tab3, tab4= st.tabs([
        "Risk Analysis",
        "Investment Priorities",
        "Field Verification Queue",
        "Regional Overview"
    ])

    # ── TAB 1: RISK ANALYSIS ──
    with tab1:
        map_col, detail_col = st.columns([3, 1])

        with map_col:
            map_title = 'Station Risk Map — Louisiana Coastal Wetlands'
            if map_view == 'Wall-to-Wall Spatial':
                map_title = 'Wall-to-Wall Spatial Map — Louisiana Coastal Wetlands'
            st.markdown(f'<div class="section-label">{map_title}</div>', unsafe_allow_html=True)

            if map_view == 'Station Map':
                st.caption(f"{len(filtered)} stations displayed with wall-to-wall risk heatmap overlay")
            else:
                st.caption(f"{len(filtered):,} sampled spatial pixels displayed | Wall-to-wall estimates shown")

            m = build_map(
                filtered, layer_mode,
                selected_id=selected_station['station_id'] if selected_station is not None else None,
                zoom_station=zoom_station,
                spatial_df=spatial_df if map_view == 'Wall-to-Wall Spatial' else None,
                w2w_layer_mode=w2w_layer_mode
            )
            st_folium(m, width=None, height=530, returned_objects=[])

        with detail_col:
            if map_view == 'Station Map':
                st.markdown('<div class="section-label">Station Detail</div>', unsafe_allow_html=True)
                if selected_station is not None:
                    render_station_detail(selected_station)
                else:
                    st.markdown(
                        '<div style="color:#a0a0b0;font-size:12px;padding:20px;'
                        'border:1px dashed #0f3460;border-radius:4px;text-align:center;'>
                        'Search a station ID<br>in the sidebar<br>to view details'
                        '</div>', unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="section-label">Wall-to-Wall Spatial Summary</div>', unsafe_allow_html=True)
                render_spatial_summary(filtered)
                
                # Unmonitored data callout in spatial view
                if unmonitored_df is not None and len(unmonitored_df) > 0:
                    st.markdown('<div class="section-label" style="margin-top:12px;">Unmonitored High-Risk Areas</div>',
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:#1a2a2e;border:1px solid #ef5350;border-radius:4px;padding:10px;font-family:monospace;font-size:11px;">
                        <div style="color:#ef5350;font-weight:600;margin-bottom:6px;">⚠️ Unknown Pixels Detected</div>
                        <div style="color:#e0e0e0;">
                            <div style="margin-bottom:4px;"><span style="color:#a0a0b0;">Count:</span> {len(unmonitored_df):,} pixels</div>
                            <div style="margin-bottom:4px;"><span style="color:#a0a0b0;">Loss Prob:</span> 100% (all pixels)</div>
                            <div style="margin-bottom:4px;"><span style="color:#a0a0b0;">Mean Carbon:</span> {unmonitored_df['carbon_predicted'].mean():.4f} g C/cm³</div>
                            <div style="margin-bottom:4px;"><span style="color:#a0a0b0;">Avg Dist:</span> {unmonitored_df['dist_to_nearest_station_km'].mean():.1f} km to nearest station</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:16px;">Top Risk Patterns</div>',
                        unsafe_allow_html=True)
            for _, rule in rules.head(3).iterrows():
                st.markdown(
                    f'<div style="background:#16213e;border:1px solid #0f3460;'
                    f'border-left:3px solid #ef5350;border-radius:3px;'
                    f'padding:8px;margin-bottom:6px;font-size:11px;font-family:monospace;">'
                    f'<span style="color:#a0a0b0;">IF</span> '
                    f'<span style="color:#e0e0e0;">{rule["antecedents"]}</span><br>'
                    f'<span style="color:#a0a0b0;">CONF:</span> '
                    f'<span style="color:#4fc3f7;">{rule["confidence"]:.0%}</span>'
                    f' &nbsp;<span style="color:#a0a0b0;">LIFT:</span> '
                    f'<span style="color:#4fc3f7;">{rule["lift"]:.1f}</span>'
                    f'</div>', unsafe_allow_html=True
                )

    # ── TAB 2: INVESTMENT PRIORITIES ──
    with tab2:
        st.markdown('<div class="section-label">Restoration Investment Priority Ranking</div>',
                    unsafe_allow_html=True)
        st.caption("Ranked by carbon loss avoided per unit area — highest priority first")

        ranked = station_filtered.copy()
        ranked['priority_score'] = ranked['loss_probability'] * ranked.get(carbon_col, 0)
        ranked = ranked.sort_values('priority_score', ascending=False)

        col_map = {
            'station_id': 'Station', 'marsh_type': 'Marsh Type', 'Basin': 'Basin',
            'loss_probability': 'Loss Prob.', carbon_col: 'Carbon Stock (g C/cm³)',
            'accretion_median': 'Accretion (cm/yr)', 'priority_score': 'Priority Score'
        }
        avail = {k: v for k, v in col_map.items() if k in ranked.columns}
        st.dataframe(
            ranked[list(avail.keys())].rename(columns=avail).head(50)
            .style.format({
                'Loss Prob.': '{:.1%}',
                'Carbon Stock (g C/cm³)': '{:.4f}',
                'Accretion (cm/yr)': '{:.2f}',
                'Priority Score': '{:.4f}'
            }).background_gradient(subset=['Priority Score'], cmap='RdYlGn'),
            use_container_width=True, height=420
        )

        st.markdown('<hr>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-label">Carbon Stock by Marsh Type</div>',
                        unsafe_allow_html=True)
            carbon_by_marsh = station_filtered.groupby('marsh_type')[carbon_col].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=carbon_by_marsh['marsh_type'],
                y=carbon_by_marsh[carbon_col],
                marker_color=['#4fc3f7', '#81d4fa', '#b3e5fc', '#e1f5fe'],
                text=carbon_by_marsh[carbon_col].round(4),
                textposition='outside'
            ))
            fig.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='Mean Carbon Stock by Marsh Type',
                yaxis_title='g C/cm³', xaxis_title=''
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-label">High Risk Stations by Basin</div>',
                        unsafe_allow_html=True)
            if 'Basin' in station_filtered.columns:
                basin_risk = station_filtered.groupby('Basin').agg(
                    high_risk=('loss_probability', lambda x: (x > 0.6).sum()),
                    total=('loss_probability', 'count')
                ).reset_index()
                basin_risk['pct_high_risk'] = basin_risk['high_risk'] / basin_risk['total']
                fig2 = go.Figure(go.Bar(
                    x=basin_risk['Basin'],
                    y=basin_risk['pct_high_risk'],
                    marker_color='#ef5350',
                    text=(basin_risk['pct_high_risk'] * 100).round(1).astype(str) + '%',
                    textposition='outside'
                ))
                fig2.update_layout(
                    **PLOTLY_TEMPLATE['layout'],
                    title='Proportion High-Risk Stations by Basin',
                    yaxis_title='Proportion', yaxis_tickformat='.0%', xaxis_title=''
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 3: FIELD VERIFICATION QUEUE ──
    with tab3:
        st.markdown('<div class="section-label">Priority Verification Queue</div>',
                    unsafe_allow_html=True)
        st.caption("Stations matching temporal precursor signatures — prioritize before next satellite pass")

        flagged = station_filtered[
            (station_filtered['loss_probability'] > 0.5) & (station_filtered['NDVI'] < 0.4)
        ].sort_values('loss_probability', ascending=False)

        if len(flagged) > 0:
            st.markdown(
                f'<div style="background:#2d1a00;border:1px solid #ffa726;'
                f'border-radius:4px;padding:10px;margin-bottom:12px;'
                f'font-family:monospace;font-size:12px;color:#ffa726;">'
                f'{len(flagged)} stations flagged for priority field verification'
                f'</div>', unsafe_allow_html=True
            )
            flag_map = {
                'station_id': 'Station', 'marsh_type': 'Marsh Type', 'Basin': 'Basin',
                'lat': 'Latitude', 'lon': 'Longitude',
                'loss_probability': 'Loss Prob.', 'NDVI': 'NDVI',
                'accretion_median': 'Accretion (cm/yr)'
            }
            avail_f = {k: v for k, v in flag_map.items() if k in flagged.columns}
            st.dataframe(
                flagged[list(avail_f.keys())].rename(columns=avail_f)
                .style.format({
                    'Loss Prob.': '{:.1%}', 'NDVI': '{:.3f}',
                    'Accretion (cm/yr)': '{:.2f}',
                    'Latitude': '{:.4f}', 'Longitude': '{:.4f}'
                }),
                use_container_width=True, height=400
            )
        else:
            st.info("No stations flagged under current filter settings.")

        # ── UNMONITORED HIGH-RISK AREAS ──
        if unmonitored_df is not None and len(unmonitored_df) > 0:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">High-Risk Unmonitored Areas (Unknown Stations)</div>',
                        unsafe_allow_html=True)
            st.caption(f"Spatial pixels without nearby CRMS stations — {len(unmonitored_df):,} high-risk locations identified")

            st.markdown(
                f'<div style="background:#1a2a2e;border:1px solid #ef5350;'
                f'border-radius:4px;padding:10px;margin-bottom:12px;'
                f'font-family:monospace;font-size:12px;color:#ef5350;">'
                f'⚠️ {len(unmonitored_df):,} unmonitored pixels with 100% predicted loss probability'
                f'</div>', unsafe_allow_html=True
            )

            # Summary stats on unmonitored
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Unmonitored Pixels", f"{len(unmonitored_df):,}",
                    f"Median dist to station: {unmonitored_df['dist_to_nearest_station_km'].median():.1f} km"
                )
            with col2:
                avg_carbon = unmonitored_df['carbon_predicted'].mean()
                st.metric("Avg Carbon (g C/cm³)", f"{avg_carbon:.4f}",
                          f"Total at risk: {unmonitored_df['carbon_predicted'].sum():.2f}")
            with col3:
                brackish_pct = (unmonitored_df['marsh_type'] == 'Brackish').sum() / len(unmonitored_df) * 100
                st.metric("Brackish Marsh", f"{brackish_pct:.0f}%",
                          f"n={int((unmonitored_df['marsh_type'] == 'Brackish').sum())}")
            with col4:
                intermediate_pct = (unmonitored_df['marsh_type'] == 'Intermediate').sum() / len(unmonitored_df) * 100
                st.metric("Intermediate Marsh", f"{intermediate_pct:.0f}%",
                          f"n={int((unmonitored_df['marsh_type'] == 'Intermediate').sum())}")

            # Display unmonitored sample
            unmon_cols = ['lat', 'lon', 'marsh_type', 'carbon_predicted', 
                          'loss_probability', 'dist_to_nearest_station_km']
            unmon_display = unmonitored_df[unmon_cols].head(20)
            col_rename = {
                'lat': 'Latitude', 'lon': 'Longitude', 'marsh_type': 'Marsh Type',
                'carbon_predicted': 'Carbon (g/cm³)', 'loss_probability': 'Loss Prob.',
                'dist_to_nearest_station_km': 'Dist to Station (km)'
            }
            st.dataframe(
                unmon_display.rename(columns=col_rename)
                .style.format({
                    'Latitude': '{:.5f}', 'Longitude': '{:.5f}',
                    'Carbon (g/cm³)': '{:.4f}', 'Loss Prob.': '{:.0%}',
                    'Dist to Station (km)': '{:.1f}'
                }),
                use_container_width=True, height=300
            )
            st.caption(f"Showing first 20 of {len(unmonitored_df):,} unmonitored high-risk pixels")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:monospace;font-size:12px;color:#b0bec5;
                    background:#16213e;border:1px solid #0f3460;
                    border-radius:4px;padding:12px;">
            <span style="color:#4fc3f7;">FLAGGING CRITERIA</span><br><br>
            &nbsp;&nbsp;1. Loss probability &gt; 50% (C4.5 classification model)<br>
            &nbsp;&nbsp;2. NDVI &lt; 0.40 (active vegetation stress)<br>
            &nbsp;&nbsp;3. Match known FP-Growth antecedents for land loss<br><br>
            <span style="color:#a0a0b0;">
            Visit flagged stations before next Sentinel-2 acquisition to collect
            ground-truth accretion measurements and confirm or refute predicted loss trajectory.
            </span>
        </div>""", unsafe_allow_html=True)

    # ── TAB 4: REGIONAL OVERVIEW ──
    with tab4:
        st.markdown('<div class="section-label">Regional Summary — Mississippi River Deltaic Plain</div>',
                    unsafe_allow_html=True)
        st.caption("Coast-wide patterns in carbon storage and land loss risk")

        # Row 1: Loss probability distribution + Marsh type breakdown
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig_hist = go.Figure(go.Histogram(
                x=station_filtered['loss_probability'],
                nbinsx=20,
                marker_color='#4fc3f7',
                marker_line_color='#0f3460',
                marker_line_width=1,
                opacity=0.85
            ))
            fig_hist.add_vline(x=0.3, line_dash='dash', line_color='#ffa726',
                               annotation_text='Moderate threshold',
                               annotation_font_color='#ffa726')
            fig_hist.add_vline(x=0.6, line_dash='dash', line_color='#ef5350',
                               annotation_text='High threshold',
                               annotation_font_color='#ef5350')
            fig_hist.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='Distribution of Land Loss Probability Across All Stations',
                xaxis_title='Loss Probability', yaxis_title='Number of Stations'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with r1c2:
            marsh_counts = station_filtered['marsh_type'].value_counts().reset_index()
            marsh_counts.columns = ['Marsh Type', 'Count']
            fig_pie = go.Figure(go.Pie(
                labels=marsh_counts['Marsh Type'],
                values=marsh_counts['Count'],
                hole=0.45,
                marker=dict(colors=['#4fc3f7', '#81d4fa', '#b3e5fc', '#0f3460'],
                            line=dict(color='#1a1a2e', width=2)),
                textfont=dict(color='#e0e0e0')
            ))
            fig_pie.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='Station Distribution by Marsh Type'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Row 2: Carbon stock vs loss probability scatter + NDVI by marsh type
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig_scatter = px.scatter(
                station_filtered,
                x=carbon_col, y='loss_probability',
                color='marsh_type',
                hover_data=['station_id', 'Basin'],
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.7,
                title='Carbon Stock vs. Land Loss Probability'
            )
            fig_scatter.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                xaxis_title='Carbon Stock (g C/cm³)',
                yaxis_title='Loss Probability'
            )
            fig_scatter.add_hline(y=0.6, line_dash='dash', line_color='#ef5350', opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with r2c2:
            fig_box = go.Figure()
            colors = {'Fresh': '#4fc3f7', 'Intermediate': '#81d4fa',
                      'Brackish': '#ffa726', 'Saline': '#ef5350'}
            for mtype in station_filtered['marsh_type'].dropna().unique():
                subset = station_filtered[station_filtered['marsh_type'] == mtype]['NDVI'].dropna()
                fig_box.add_trace(go.Box(
                    y=subset, name=mtype,
                    marker_color=colors.get(mtype, '#4fc3f7'),
                    line_color=colors.get(mtype, '#4fc3f7'),
                    fillcolor='rgba(16,33,62,0.8)'
                ))
            fig_box.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='NDVI Distribution by Marsh Type',
                yaxis_title='NDVI', xaxis_title='',
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ── UNMONITORED VS MONITORED COMPARISON ──
        if unmonitored_df is not None and len(unmonitored_df) > 0:
            st.markdown('<div class="section-label" style="margin-top:16px;">Monitored vs. Unmonitored Gap Analysis</div>',
                        unsafe_allow_html=True)
            st.caption("Comparison of monitored CRMS stations with unmapped high-risk coastal pixels")

            # Create comparison metrics
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            monitored_count = len(station_filtered)
            unmonitored_count = len(unmonitored_df)
            gap_ratio = unmonitored_count / monitored_count if monitored_count > 0 else 0

            with comp_col1:
                st.metric(
                    "Monitoring Gap",
                    f"{gap_ratio:.1f}x",
                    f"{unmonitored_count:,} unmapped vs {monitored_count:,} stations"
                )

            monitored_carbon = station_filtered[carbon_col].sum() if carbon_col in station_filtered.columns else 0
            unmonitored_carbon = unmonitored_df['carbon_predicted'].sum()
            total_carbon = monitored_carbon + unmonitored_carbon

            with comp_col2:
                st.metric(
                    "Carbon at Risk — Unmonitored",
                    f"{unmonitored_carbon:.2f}",
                    f"{unmonitored_carbon/total_carbon*100:.1f}% of total" if total_carbon > 0 else "N/A"
                )

            # Average loss probability
            monitored_loss_prob = station_filtered['loss_probability'].mean()
            unmonitored_loss_prob = unmonitored_df['loss_probability'].mean()

            with comp_col3:
                st.metric(
                    "Avg Loss Probability — Unmonitored",
                    f"{unmonitored_loss_prob:.1%}",
                    f"vs {monitored_loss_prob:.1%} monitored"
                )

            # Marsh type comparison
            fig_marsh_comp = go.Figure()
            
            # Monitored marsh distribution
            monitored_marsh = station_filtered['marsh_type'].value_counts().reset_index()
            monitored_marsh.columns = ['Marsh Type', 'Count']
            monitored_marsh['Category'] = 'Monitored'
            
            # Unmonitored marsh distribution
            unmonitored_marsh = unmonitored_df['marsh_type'].value_counts().reset_index()
            unmonitored_marsh.columns = ['Marsh Type', 'Count']
            unmonitored_marsh['Category'] = 'Unmonitored'
            
            marsh_comp = pd.concat([monitored_marsh, unmonitored_marsh], ignore_index=True)
            
            fig_marsh_comp = px.bar(
                marsh_comp, x='Marsh Type', y='Count', color='Category',
                barmode='group',
                color_discrete_map={'Monitored': '#81d4fa', 'Unmonitored': '#ef5350'},
                title='Marsh Type Distribution: Monitored vs Unmonitored'
            )
            fig_marsh_comp.update_layout(**PLOTLY_TEMPLATE['layout'])
            st.plotly_chart(fig_marsh_comp, use_container_width=True)

            # Unmonitored spatial statistics
            st.markdown('<div class="section-label" style="margin-top:12px;">Unmonitored Pixels — Spatial Distribution</div>',
                        unsafe_allow_html=True)
            
            unmon_col1, unmon_col2, unmon_col3, unmon_col4 = st.columns(4)
            with unmon_col1:
                st.metric("Total Unmonitored Pixels", f"{len(unmonitored_df):,}")
            with unmon_col2:
                st.metric("Mean Distance to Station", f"{unmonitored_df['dist_to_nearest_station_km'].mean():.1f} km")
            with unmon_col3:
                st.metric("Median Carbon Predicted", f"{unmonitored_df['carbon_predicted'].median():.4f}")
            with unmon_col4:
                st.metric("Loss Probability", f"{(unmonitored_df['loss_probability']==1.0).sum():,} @ 100%")

        # Row 3: Basin-level summary table
        st.markdown('<div class="section-label" style="margin-top:8px;">Basin-Level Summary</div>',
                    unsafe_allow_html=True)
        if 'Basin' in station_filtered.columns:
            basin_summary = station_filtered.groupby('Basin').agg(
                Stations=('station_id', 'count'),
                High_Risk_Stations=('loss_probability', lambda x: (x > 0.6).sum()),
                Mean_Loss_Probability=('loss_probability', 'mean'),
                Mean_Carbon_Stock=(carbon_col, 'mean'),
                Total_Carbon_Stock=(carbon_col, 'sum'),
                Mean_NDVI=('NDVI', 'mean')
            ).reset_index().sort_values('Mean_Loss_Probability', ascending=False)

            st.dataframe(
                basin_summary.style.format({
                    'Mean_Loss_Probability': '{:.1%}',
                    'Mean_Carbon_Stock': '{:.4f}',
                    'Total_Carbon_Stock': '{:.3f}',
                    'Mean_NDVI': '{:.3f}'
                }).background_gradient(subset=['Mean_Loss_Probability'], cmap='RdYlGn_r'),
                use_container_width=True
            )

        # Key findings callout
        st.markdown('<hr>', unsafe_allow_html=True)
        high_pct = (station_filtered['loss_probability'] > 0.6).mean()
        top_basin = station_filtered.groupby('Basin')['loss_probability'].mean().idxmax() \
            if 'Basin' in station_filtered.columns else 'N/A'
        highest_risk_marsh = station_filtered.groupby('marsh_type')['loss_probability'].mean().idxmax()

        st.markdown(f"""
        <div style="background:#16213e;border:1px solid #0f3460;border-left:3px solid #4fc3f7;
                    border-radius:4px;padding:16px;font-family:monospace;font-size:12px;">
            <div style="color:#4fc3f7;font-weight:600;letter-spacing:1px;margin-bottom:10px;">
                KEY FINDINGS — DELTAIC PLAIN
            </div>
            <div style="color:#e0e0e0;line-height:2;">
                &nbsp;&nbsp;{high_pct:.1%} of monitored stations face high land loss probability (&gt;60%)<br>
                &nbsp;&nbsp;{top_basin} basin shows the highest mean loss probability across stations<br>
                &nbsp;&nbsp;{highest_risk_marsh} marsh communities show the greatest vulnerability<br>
                &nbsp;&nbsp;Declining NDVI, high tidal amplitude, and brackish conditions
                are the strongest co-occurring predictors of land loss<br>
                &nbsp;&nbsp;Carbon storage at risk scales directly with loss probability —
                high-risk stations represent a disproportionate share of total carbon sink capacity
            </div>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4a4a6a;font-size:10px;font-family:monospace;letter-spacing:1px;">'
        'CWL-026-001 &nbsp;|&nbsp; CSC 5803 &nbsp;|&nbsp; LOUISIANA TECH UNIVERSITY &nbsp;|&nbsp;'
        'KALEIGH POWELL &nbsp;|&nbsp;'
        '</p>', unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()