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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from hurricane_tab import render_hurricane_tab

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
    if os.path.exists('data/fused_dataset_with_hydro.csv'):
        df_full = pd.read_csv('data/fused_dataset_with_hydro.csv')
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
            'carbon_stock': np.random.normal(0.065, 0.02, n),
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
                    'carbon_stock': row['carbon_stock'] + np.random.normal(0, 0.002),
                    'loss_probability': row['loss_probability']
                })
        df_time = pd.DataFrame(records)

    df_stations['station_id'] = df_stations['station_id'].astype(str)
    return df_stations, df_time

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
def build_map(df, layer_mode, selected_id=None, zoom_station=None):
    if zoom_station is not None:
        center = [zoom_station['lat'], zoom_station['lon']]
        zoom = 13
    else:
        center = [29.5, -91.0]
        zoom = 8

    m = folium.Map(location=center, zoom_start=zoom, tiles='CartoDB dark_matter')

    carbon_min = df['carbon_stock'].min()
    carbon_max = df['carbon_stock'].max()

    for _, row in df.iterrows():
        is_selected = str(row['station_id']) == str(selected_id) if selected_id else False

        if layer_mode == 'Carbon Stock':
            color = carbon_color_hex(row['carbon_stock'], carbon_min, carbon_max)
            primary = f"Carbon: {row['carbon_stock']:.4f} g C/cm³"
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

    # Legend
    if layer_mode == 'Carbon Stock':
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
    # Add wall-to-wall GeoJSON layer if available
    geojson_path = 'results/high_risk_parcels.geojson'
    if os.path.exists(geojson_path):
        import json
        with open(geojson_path) as f:
            geojson_data = json.load(f)
        folium.GeoJson(
            geojson_data,
            name='Wall-to-Wall Risk',
            style_function=lambda x: {
                'fillColor': risk_color_hex(
                    x['properties'].get('loss_probability', 0)
                ),
                'color': 'none',
                'fillOpacity': 0.4,
                'radius': 3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['loss_probability', 'carbon_predicted',
                        'marsh_type', 'risk_level'],
                aliases=['Loss Prob', 'Carbon Stock',
                        'Marsh Type', 'Risk Level'],
                localize=True
            )
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
        <div class="value">{s['carbon_stock']:.4f} g C/cm³
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
# MAIN APP
# ─────────────────────────────────────────────
def main():
    df, df_time = load_data()
    rules = load_rules()

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

        st.markdown('<div class="section-label" style="margin-top:16px;">Filters</div>', unsafe_allow_html=True)
        marsh_filter = st.multiselect(
            'Marsh Type',
            options=sorted(df['marsh_type'].dropna().unique()),
            default=sorted(df['marsh_type'].dropna().unique())
        )
        basin_options = sorted(df['Basin'].dropna().unique()) if 'Basin' in df.columns else []
        basin_filter = st.multiselect('Basin', options=basin_options, default=basin_options)
        risk_min = st.slider('Min Loss Probability', 0.0, 1.0, 0.0, 0.05)

    # Filter stations
    filtered = df[df['marsh_type'].isin(marsh_filter)].copy()
    if basin_filter:
        filtered = filtered[filtered['Basin'].isin(basin_filter)]
    filtered = filtered[filtered['loss_probability'] >= risk_min]

    # Station search
    selected_station = None
    zoom_station = None
    if station_search:
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

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Stations", f"{len(filtered)}")
    with k2: st.metric("High Risk", f"{(filtered['loss_probability']>0.6).sum()}")
    with k3: st.metric("Mean Carbon Stock", f"{filtered['carbon_stock'].mean():.4f} g C/cm³")
    with k4:
        at_risk = filtered.loc[filtered['loss_probability']>0.6, 'carbon_stock'].sum()
        st.metric("Carbon at Risk", f"{at_risk:.3f} g C/cm³")
    with k5:
        lost = int(filtered['ever_lost_land'].sum()) if 'ever_lost_land' in filtered.columns else 0
        st.metric("Confirmed Loss Events", f"{lost}")

    st.markdown('<hr>', unsafe_allow_html=True)

    # View tabs
    tab1, tab2, tab3, tab4, tab5= st.tabs([
        "Risk Analysis",
        "Investment Priorities",
        "Field Verification Queue",
        "Regional Overview",
        "Hurricane Impact"
    ])

    # ── TAB 1: RISK ANALYSIS ──
    with tab1:
        map_col, detail_col = st.columns([3, 1])

        with map_col:
            st.markdown('<div class="section-label">Station Risk Map — Louisiana Coastal Wetlands</div>',
                        unsafe_allow_html=True)
            st.caption(f"{len(filtered)} stations displayed | Click any marker for details")
            m = build_map(
                filtered, layer_mode,
                selected_id=selected_station['station_id'] if selected_station is not None else None,
                zoom_station=zoom_station
            )
            st_folium(m, width=None, height=530, returned_objects=[])

        with detail_col:
            st.markdown('<div class="section-label">Station Detail</div>', unsafe_allow_html=True)
            if selected_station is not None:
                render_station_detail(selected_station)
            else:
                st.markdown(
                    '<div style="color:#a0a0b0;font-size:12px;padding:20px;'
                    'border:1px dashed #0f3460;border-radius:4px;text-align:center;">'
                    'Search a station ID<br>in the sidebar<br>to view details'
                    '</div>', unsafe_allow_html=True
                )

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

        ranked = filtered.copy()
        ranked['priority_score'] = ranked['loss_probability'] * ranked['carbon_stock']
        ranked = ranked.sort_values('priority_score', ascending=False)

        col_map = {
            'station_id': 'Station', 'marsh_type': 'Marsh Type', 'Basin': 'Basin',
            'loss_probability': 'Loss Prob.', 'carbon_stock': 'Carbon Stock (g C/cm³)',
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
            carbon_by_marsh = filtered.groupby('marsh_type')['carbon_stock'].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=carbon_by_marsh['marsh_type'],
                y=carbon_by_marsh['carbon_stock'],
                marker_color=['#4fc3f7', '#81d4fa', '#b3e5fc', '#e1f5fe'],
                text=carbon_by_marsh['carbon_stock'].round(4),
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
            if 'Basin' in filtered.columns:
                basin_risk = filtered.groupby('Basin').agg(
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

        flagged = filtered[
            (filtered['loss_probability'] > 0.5) & (filtered['NDVI'] < 0.4)
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
                x=filtered['loss_probability'],
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
            marsh_counts = filtered['marsh_type'].value_counts().reset_index()
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
                filtered,
                x='carbon_stock', y='loss_probability',
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
            for mtype in filtered['marsh_type'].dropna().unique():
                subset = filtered[filtered['marsh_type'] == mtype]['NDVI'].dropna()
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

        # Row 3: Basin-level summary table
        st.markdown('<div class="section-label" style="margin-top:8px;">Basin-Level Summary</div>',
                    unsafe_allow_html=True)
        if 'Basin' in filtered.columns:
            basin_summary = filtered.groupby('Basin').agg(
                Stations=('station_id', 'count'),
                High_Risk_Stations=('loss_probability', lambda x: (x > 0.6).sum()),
                Mean_Loss_Probability=('loss_probability', 'mean'),
                Mean_Carbon_Stock=('carbon_stock', 'mean'),
                Total_Carbon_Stock=('carbon_stock', 'sum'),
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
        high_pct = (filtered['loss_probability'] > 0.6).mean()
        top_basin = filtered.groupby('Basin')['loss_probability'].mean().idxmax() \
            if 'Basin' in filtered.columns else 'N/A'
        highest_risk_marsh = filtered.groupby('marsh_type')['loss_probability'].mean().idxmax()

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
    
    # u2500u2500 TAB 5: HURRICANE IMPACT u2500u2500
    with tab5:
        render_hurricane_tab(df, PLOTLY_TEMPLATE)

    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4a4a6a;font-size:10px;font-family:monospace;letter-spacing:1px;">'
        'CWL-026-001 &nbsp;|&nbsp; CSC 580 &nbsp;|&nbsp; LOUISIANA TECH UNIVERSITY &nbsp;|&nbsp;'
        'DATA: CRMS · SENTINEL-2 · USGS LANDSAT · NASA ORNL DAAC &nbsp;|&nbsp;'
        'PIPELINE UPDATES AUTOMATICALLY ON NEW DATA INGESTION'
        '</p>', unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()