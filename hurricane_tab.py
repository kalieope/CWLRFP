"""
hurricane_tab.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Hurricane Impact tab module for 06_dashboard.py.
    Imported and called by the main dashboard.
    Keeps dashboard modular and maintainable.

    Contains two views:
    1. Historical Storm Review — what past storms did to monitored stations
    2. Future Scenario Simulation — what would happen if a similar storm hit today

USAGE IN DASHBOARD:
    from hurricane_tab import render_hurricane_tab
    with tab5:
        render_hurricane_tab(df, PLOTLY_TEMPLATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_color_hex(prob):
    if prob >= 0.6: return '#ef5350'
    elif prob >= 0.3: return '#ffa726'
    else: return '#66bb6a'

# ─────────────────────────────────────────────
# HISTORICAL STORM REVIEW
# ─────────────────────────────────────────────
def render_historical_view(df, GULF_STORMS, STORM_SCENARIOS,
                            get_storm_affected_stations, PLOTLY_TEMPLATE):
    st.markdown(
        '<div class="section-label">Select a storm to review its '
        'impact on monitored stations</div>',
        unsafe_allow_html=True
    )

    storm_options = [
        f"{s['name']} {s['year']} (Cat {s['category']}, "
        f"{s['max_surge_ft']}ft surge)"
        for s in GULF_STORMS
    ]
    selected_idx = st.selectbox(
        "Storm Event", range(len(storm_options)),
        format_func=lambda i: storm_options[i],
        index=len(storm_options) - 1
    )
    selected_storm = GULF_STORMS[selected_idx]

    # Storm info card
    cat_label = ("Tropical Storm" if selected_storm['category'] == 0
                 else f"Category {selected_storm['category']}")
    st.markdown(f"""
    <div style="background:#16213e;border:1px solid #0f3460;
                border-left:3px solid #ef5350;border-radius:4px;
                padding:16px;margin:10px 0;font-family:monospace;font-size:12px;">
        <div style="color:#ef5350;font-weight:600;font-size:14px;margin-bottom:8px;">
            Hurricane {selected_storm['name']} ({selected_storm['year']})
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
            <div>
                <div class="label">Category</div>
                <div class="value">{cat_label}</div>
            </div>
            <div>
                <div class="label">Max Surge</div>
                <div class="value">{selected_storm['max_surge_ft']} ft</div>
            </div>
            <div>
                <div class="label">Impact Radius</div>
                <div class="value">{selected_storm['impact_radius_km']} km</div>
            </div>
        </div>
        <div style="margin-top:10px;">
            <div class="label">Primary Basins Affected</div>
            <div class="value">{", ".join(selected_storm['basins_affected'])}</div>
        </div>
        <div style="margin-top:10px;">
            <div class="label">Wetland Impact</div>
            <div style="color:#b0bec5;font-size:11px;line-height:1.6;">
                {selected_storm['notes']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Stations Within Impact Zone</div>',
                unsafe_allow_html=True)

    affected = get_storm_affected_stations(
        df, selected_storm['name'], selected_storm['year']
    )

    map_col, stats_col = st.columns([3, 1])

    with map_col:
        m = folium.Map(
            location=[selected_storm['landfall_lat'],
                      selected_storm['landfall_lon']],
            zoom_start=8, tiles='CartoDB dark_matter'
        )

        # Landfall marker
        folium.Marker(
            location=[selected_storm['landfall_lat'],
                      selected_storm['landfall_lon']],
            popup=f"Landfall: {selected_storm['name']} {selected_storm['year']}",
            icon=folium.Icon(color='red', icon='bolt', prefix='fa')
        ).add_to(m)

        # Impact radius circle
        folium.Circle(
            location=[selected_storm['landfall_lat'],
                      selected_storm['landfall_lon']],
            radius=selected_storm['impact_radius_km'] * 1000,
            color='#ef5350', fill=True, fill_opacity=0.08, weight=1
        ).add_to(m)

        # Plot all stations
        affected_ids = set(affected['station_id'].values) if len(affected) > 0 else set()
        for _, row in df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                continue
            is_affected = row['station_id'] in affected_ids
            dist_text = ''
            if is_affected and 'distance_from_landfall_km' in affected.columns:
                dist_row = affected[affected['station_id'] == row['station_id']]
                if len(dist_row) > 0:
                    dist_text = (f"Dist from landfall: "
                                 f"{dist_row['distance_from_landfall_km'].values[0]:.0f} km<br>")

            popup_html = (
                f'<div style="font-family:monospace;background:#16213e;'
                f'color:#e0e0e0;padding:8px;">'
                f'<b>{row["station_id"]}</b><br>'
                f'Marsh: {row.get("marsh_type", "N/A")}<br>'
                f'{dist_text}'
                f'Loss Prob: {row.get("loss_probability", 0):.1%}'
                f'</div>'
            )
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6 if is_affected else 3,
                color='#ef5350' if is_affected else '#4a4a6a',
                fill=True,
                fill_color='#ef5350' if is_affected else '#2a2a4a',
                fill_opacity=0.85 if is_affected else 0.3,
                weight=2 if is_affected else 1,
                popup=folium.Popup(popup_html, max_width=220)
            ).add_to(m)

        st_folium(m, width=None, height=480, returned_objects=[])

    with stats_col:
        st.markdown('<div class="section-label">Impact Statistics</div>',
                    unsafe_allow_html=True)
        st.metric("Stations in Impact Zone", f"{len(affected)}")

        if len(affected) > 0 and 'loss_probability' in affected.columns:
            st.metric("Mean Loss Probability",
                      f"{affected['loss_probability'].mean():.1%}")
            st.metric("High Risk Stations",
                      f"{(affected['loss_probability'] > 0.6).sum()}")

            if 'carbon_stock' in affected.columns:
                st.metric("Carbon at Risk",
                          f"{affected['carbon_stock'].sum():.3f} g C/cm³")

            st.markdown('<div class="section-label" style="margin-top:12px;">'
                        'Marsh Types Affected</div>',
                        unsafe_allow_html=True)
            for mtype, count in affected['marsh_type'].value_counts().items():
                st.markdown(
                    f'<div style="font-family:monospace;font-size:11px;'
                    f'color:#b0bec5;">{mtype}: {count} stations</div>',
                    unsafe_allow_html=True
                )

    # Timeline chart
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Historical Storm Timeline</div>',
                unsafe_allow_html=True)

    cat_colors = {0: '#81d4fa', 1: '#fff176', 2: '#ffa726',
                  3: '#ef5350', 4: '#b71c1c', 5: '#4a0000'}
    fig = go.Figure()

    for storm in GULF_STORMS:
        color = cat_colors.get(storm['category'], '#ef5350')
        label = ('Tropical Storm' if storm['category'] == 0
                 else f"Cat {storm['category']}")
        fig.add_trace(go.Scatter(
            x=[storm['year']],
            y=[storm['max_surge_ft']],
            mode='markers+text',
            marker=dict(
                size=storm['max_surge_ft'] * 1.2,
                color=color,
                line=dict(color='#1a1a2e', width=1),
                opacity=0.85
            ),
            text=[storm['name']],
            textposition='top center',
            textfont=dict(color='#e0e0e0', size=10),
            name=label,
            hovertemplate=(
                f"<b>{storm['name']} {storm['year']}</b><br>"
                f"Category: {label}<br>"
                f"Max Surge: {storm['max_surge_ft']} ft<br>"
                f"Basins: {', '.join(storm['basins_affected'])}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title='Major Gulf Coast Storms Affecting Louisiana Wetlands',
        xaxis_title='Year',
        yaxis_title='Maximum Storm Surge (ft)',
        showlegend=False, height=350
    )
    fig.add_hline(y=15, line_dash='dash', line_color='#ffa726',
                  annotation_text='Major damage threshold (~15 ft)',
                  annotation_font_color='#ffa726')
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# FUTURE SCENARIO SIMULATION
# ─────────────────────────────────────────────
def render_scenario_view(df, GULF_STORMS, STORM_SCENARIOS, simulate_storm_impact):
    st.markdown(
        '<div class="section-label">Configure a hypothetical storm to estimate '
        'wetland loss across the current coast</div>',
        unsafe_allow_html=True
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scenario_type = st.selectbox("Storm Intensity",
                                      list(STORM_SCENARIOS.keys()))
    with sc2:
        landfall_lat = st.slider("Landfall Latitude", 28.5, 30.5, 29.3, 0.05)
    with sc3:
        landfall_lon = st.slider("Landfall Longitude", -93.5, -88.5, -90.3, 0.05)

    scenario_info = STORM_SCENARIOS[scenario_type]
    st.markdown(f"""
    <div style="background:#2d1a00;border:1px solid #ffa726;border-radius:4px;
                padding:12px;margin:8px 0;font-family:monospace;font-size:12px;
                color:#ffa726;">
        <b>{scenario_type}</b><br>
        <span style="color:#b0bec5;">{scenario_info['description']}</span><br>
        Surge estimate: {scenario_info['surge_ft_range'][0]}–
        {scenario_info['surge_ft_range'][1]} ft &nbsp;|&nbsp;
        Impact radius: {scenario_info['impact_radius_km']} km
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Scenario Simulation", type="primary"):
        with st.spinner("Simulating storm impact..."):
            scenario_df = simulate_storm_impact(
                df, landfall_lat, landfall_lon, scenario_type
            )

        sc_map_col, sc_stats_col = st.columns([3, 1])

        with sc_map_col:
            m = folium.Map(
                location=[landfall_lat, landfall_lon],
                zoom_start=8, tiles='CartoDB dark_matter'
            )
            folium.Marker(
                location=[landfall_lat, landfall_lon],
                popup='Hypothetical Landfall',
                icon=folium.Icon(color='red', icon='bolt', prefix='fa')
            ).add_to(m)
            folium.Circle(
                location=[landfall_lat, landfall_lon],
                radius=scenario_info['impact_radius_km'] * 1000,
                color='#ffa726', fill=True, fill_opacity=0.06, weight=1
            ).add_to(m)

            for _, row in scenario_df.iterrows():
                if pd.isna(row['lat']) or pd.isna(row['lon']):
                    continue
                prob = row.get('scenario_loss_probability',
                               row.get('loss_probability', 0.3))
                color = risk_color_hex(prob)
                popup_html = (
                    f'<div style="font-family:monospace;background:#16213e;'
                    f'color:#e0e0e0;padding:8px;">'
                    f'<b>{row["station_id"]}</b><br>'
                    f'Marsh: {row.get("marsh_type", "N/A")}<br>'
                    f'Scenario Loss Prob: {prob:.1%}<br>'
                    f'Est. Surge: {row.get("estimated_surge_ft", 0):.1f} ft'
                    f'</div>'
                )
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=7 if row.get('in_storm_path') else 4,
                    color=color, fill=True, fill_color=color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=220)
                ).add_to(m)

            st_folium(m, width=None, height=480, returned_objects=[])

        with sc_stats_col:
            st.markdown('<div class="section-label">Scenario Summary</div>',
                        unsafe_allow_html=True)
            in_path = scenario_df['in_storm_path'].sum()
            st.metric("Stations in Path", f"{in_path}")

            in_path_df = scenario_df[scenario_df['in_storm_path']]
            if len(in_path_df) > 0 and 'scenario_loss_probability' in in_path_df.columns:
                high_post = (in_path_df['scenario_loss_probability'] > 0.6).sum()
                st.metric("High Risk Post-Storm", f"{high_post}")
                if 'carbon_stock' in in_path_df.columns:
                    threatened = in_path_df[
                        in_path_df['scenario_loss_probability'] > 0.6
                    ]['carbon_stock'].sum()
                    st.metric("Carbon Threatened",
                              f"{threatened:.3f} g C/cm³")

            st.markdown(
                '<div style="font-family:monospace;font-size:11px;'
                'color:#a0a0b0;margin-top:12px;line-height:1.8;">'
                '<b style="color:#ffa726;">Note:</b><br>'
                'Scenario applies surge intensity and distance decay '
                'to baseline loss probability.<br><br>'
                'Full hydrodynamic modeling (ADCIRC integration) '
                'is planned for Phase 2.'
                '</div>',
                unsafe_allow_html=True
            )

            # Historical analogs
            st.markdown('<div class="section-label" style="margin-top:12px;">'
                        'Historical Analogs</div>',
                        unsafe_allow_html=True)
            cat = scenario_info['category']
            analogs = [s for s in GULF_STORMS if abs(s['category'] - cat) <= 1]
            for a in analogs[:3]:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:11px;'
                    f'color:#b0bec5;margin-bottom:4px;">'
                    f'{a["name"]} {a["year"]} '
                    f'(Cat {a["category"]}, {a["max_surge_ft"]}ft)</div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────
# MAIN RENDER FUNCTION
# Called from 06_dashboard.py
# ─────────────────────────────────────────────
def render_hurricane_tab(df, PLOTLY_TEMPLATE):
    st.markdown('<div class="section-label">Hurricane Impact Analysis</div>',
                unsafe_allow_html=True)

    try:
        from storm_events import (GULF_STORMS, STORM_SCENARIOS,
                                   get_storm_affected_stations,
                                   simulate_storm_impact)
    except ImportError:
        st.error("storm_events.py not found — place it in the project directory")
        return

    h_mode = st.radio(
        '',
        ['Historical Storm Review', 'Future Scenario Simulation'],
        horizontal=True,
        label_visibility='collapsed'
    )

    if h_mode == 'Historical Storm Review':
        render_historical_view(
            df, GULF_STORMS, STORM_SCENARIOS,
            get_storm_affected_stations, PLOTLY_TEMPLATE
        )
    else:
        render_scenario_view(
            df, GULF_STORMS, STORM_SCENARIOS, simulate_storm_impact
        )