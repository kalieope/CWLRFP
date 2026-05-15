"""
storm_events.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Central registry of major Gulf Coast storm events affecting
    Louisiana's coastal wetlands. Used by:
    - 02_crms_preprocessing.py  (flag affected stations)
    - 04_fpgrowth_pattern_mining.py (separate storm/chronic rules)
    - 03_gaussian_process_regression.py (storm feature)
    - 05_c45_classification_temporal_roc.py (storm feature)
    - 06_dashboard.py (hurricane impact view)

    Each storm has:
    - Name, year, category, max surge (ft)
    - Approximate landfall location (lat/lon)
    - Impact radius (km) — stations within this radius are flagged
    - Primary basins affected
    - Notes on wetland impact

    Storm data sourced from NOAA National Hurricane Center
    Impact radii are conservative estimates based on NHC damage reports

ADDING NEW STORMS:
    Add a new dict entry to GULF_STORMS following the same format.
    Re-run 02_crms_preprocessing.py to update station flags.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# MAJOR GULF COAST STORMS AFFECTING LOUISIANA
# Storms that caused documented wetland loss in the Deltaic Plain
# ─────────────────────────────────────────────
GULF_STORMS = [
    {
        'name': 'Andrew',
        'year': 1992,
        'category': 5,
        'max_surge_ft': 16.9,
        'landfall_lat': 29.37,
        'landfall_lon': -90.56,
        'impact_radius_km': 150,
        'basins_affected': ['Barataria', 'Pontchartrain'],
        'notes': 'Category 5 at landfall near Morgan City. Severe wetland damage in Barataria basin.'
    },
    {
        'name': 'Isidore',
        'year': 2002,
        'category': 0,  # Tropical storm at landfall
        'max_surge_ft': 6.0,
        'landfall_lat': 29.68,
        'landfall_lon': -91.20,
        'impact_radius_km': 200,
        'basins_affected': ['Terrebonne', 'Barataria', 'Atchafalaya'],
        'notes': 'Tropical storm. Heavy rainfall and surge caused significant wetland flooding.'
    },
    {
        'name': 'Lili',
        'year': 2002,
        'category': 1,
        'max_surge_ft': 8.0,
        'landfall_lat': 29.77,
        'landfall_lon': -91.88,
        'impact_radius_km': 150,
        'basins_affected': ['Atchafalaya', 'Mermentau'],
        'notes': 'Category 1 near Intracoastal City. Wetland erosion in Atchafalaya basin.'
    },
    {
        'name': 'Katrina',
        'year': 2005,
        'category': 3,  # Category 3 at Louisiana landfall (was Cat 5 in Gulf)
        'max_surge_ft': 27.8,
        'landfall_lat': 29.28,
        'landfall_lon': -89.58,
        'impact_radius_km': 300,
        'basins_affected': ['Pontchartrain', 'Breton Sound', 'Barataria', 'Terrebonne'],
        'notes': 'Most destructive hurricane in Louisiana history. '
                 '100+ sq miles of wetland lost. Storm surge up to 27.8 ft. '
                 'Catastrophic land loss across eastern Deltaic Plain.'
    },
    {
        'name': 'Rita',
        'year': 2005,
        'category': 3,
        'max_surge_ft': 15.0,
        'landfall_lat': 29.77,
        'landfall_lon': -93.59,
        'impact_radius_km': 250,
        'basins_affected': ['Mermentau', 'Calcasieu', 'Sabine'],
        'notes': 'Category 3 near Sabine Pass. Severe wetland loss in southwestern Louisiana. '
                 'Struck just weeks after Katrina compounding recovery challenges.'
    },
    {
        'name': 'Gustav',
        'year': 2008,
        'category': 2,
        'max_surge_ft': 12.0,
        'landfall_lat': 29.37,
        'landfall_lon': -90.66,
        'impact_radius_km': 200,
        'basins_affected': ['Terrebonne', 'Barataria', 'Atchafalaya'],
        'notes': 'Category 2 near Cocodrie. Significant surge in Terrebonne and Barataria basins.'
    },
    {
        'name': 'Ike',
        'year': 2008,
        'category': 2,
        'max_surge_ft': 20.0,
        'landfall_lat': 29.31,
        'landfall_lon': -94.79,
        'impact_radius_km': 300,
        'basins_affected': ['Mermentau', 'Calcasieu'],
        'notes': 'Category 2 at Galveston. Ike surge extended far into western Louisiana wetlands. '
                 'Large fetch caused exceptional surge heights.'
    },
    {
        'name': 'Isaac',
        'year': 2012,
        'category': 1,
        'max_surge_ft': 11.0,
        'landfall_lat': 29.22,
        'landfall_lon': -90.06,
        'impact_radius_km': 200,
        'basins_affected': ['Barataria', 'Pontchartrain', 'Breton Sound'],
        'notes': 'Slow-moving Category 1. Extended surge duration caused disproportionate wetland damage. '
                 'Significant NDVI decline observed in Barataria basin post-storm.'
    },
    {
        'name': 'Delta',
        'year': 2020,
        'category': 2,
        'max_surge_ft': 9.0,
        'landfall_lat': 29.77,
        'landfall_lon': -92.93,
        'impact_radius_km': 150,
        'basins_affected': ['Mermentau', 'Atchafalaya'],
        'notes': 'Category 2 near Creole. Struck same area as Laura just weeks prior, '
                 'compounding wetland stress in southwestern Deltaic Plain.'
    },
    {
        'name': 'Laura',
        'year': 2020,
        'category': 4,
        'max_surge_ft': 15.0,
        'landfall_lat': 30.03,
        'landfall_lon': -93.22,
        'impact_radius_km': 200,
        'basins_affected': ['Mermentau', 'Calcasieu'],
        'notes': 'Category 4 near Cameron. Strongest hurricane to strike Louisiana since 1856 '
                 'in terms of wind speed. Catastrophic surge in Calcasieu and Mermentau basins.'
    },
    {
        'name': 'Ida',
        'year': 2021,
        'category': 4,
        'max_surge_ft': 16.0,
        'landfall_lat': 29.23,
        'landfall_lon': -90.29,
        'impact_radius_km': 250,
        'basins_affected': ['Terrebonne', 'Barataria', 'Pontchartrain', 'Atchafalaya'],
        'notes': 'Category 4 at Port Fourchon. Tied record for strongest Louisiana landfall. '
                 'Extensive wetland damage across central Deltaic Plain. '
                 'CRMS stations showed significant post-storm accretion and erosion signals.'
    }
]

# Convert to DataFrame for easy filtering
STORMS_DF = pd.DataFrame(GULF_STORMS)

# ─────────────────────────────────────────────
# INTENSITY CATEGORIES FOR SCENARIO SIMULATION
# Maps storm categories to expected surge ranges
# Used for future scenario simulation in dashboard
# ─────────────────────────────────────────────
STORM_SCENARIOS = {
    'Tropical Storm': {
        'category': 0,
        'surge_ft_range': (3, 8),
        'impact_radius_km': 150,
        'description': 'Tropical storm conditions. Moderate flooding, minimal permanent wetland loss.'
    },
    'Category 1-2': {
        'category': 1.5,
        'surge_ft_range': (8, 14),
        'impact_radius_km': 200,
        'description': 'Moderate hurricane. Significant surge, localized wetland loss likely.'
    },
    'Category 3 (e.g. Katrina-equivalent)': {
        'category': 3,
        'surge_ft_range': (12, 20),
        'impact_radius_km': 280,
        'description': 'Major hurricane. Large-scale surge, widespread wetland conversion likely.'
    },
    'Category 4-5 (e.g. Ida/Laura-equivalent)': {
        'category': 4.5,
        'surge_ft_range': (15, 28),
        'impact_radius_km': 350,
        'description': 'Catastrophic hurricane. Extreme surge, massive permanent wetland loss expected.'
    }
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points"""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def flag_storm_affected_stations(stations_df):
    """
    For each station, determine which storms affected it based on
    distance from landfall location vs impact radius.
    Adds columns: storm_year (bool), storms_hit (list), max_category_hit (int)
    """
    stations_df = stations_df.copy()
    stations_df['storm_year'] = False
    stations_df['storms_hit'] = [[] for _ in range(len(stations_df))]
    stations_df['max_category_hit'] = 0
    stations_df['max_surge_experienced_ft'] = 0.0
    stations_df['storm_count'] = 0

    for storm in GULF_STORMS:
        for idx, station in stations_df.iterrows():
            if pd.isna(station['lat']) or pd.isna(station['lon']):
                continue
            dist = haversine_km(
                storm['landfall_lat'], storm['landfall_lon'],
                station['lat'], station['lon']
            )
            if dist <= storm['impact_radius_km']:
                stations_df.at[idx, 'storm_year'] = True
                stations_df.at[idx, 'storms_hit'].append(
                    f"{storm['name']} {storm['year']}"
                )
                stations_df.at[idx, 'max_category_hit'] = max(
                    stations_df.at[idx, 'max_category_hit'],
                    storm['category']
                )
                stations_df.at[idx, 'max_surge_experienced_ft'] = max(
                    stations_df.at[idx, 'max_surge_experienced_ft'],
                    storm['max_surge_ft']
                )
                stations_df.at[idx, 'storm_count'] += 1

    # Convert list to string for CSV storage
    stations_df['storms_hit_str'] = stations_df['storms_hit'].apply(
        lambda x: '; '.join(x) if x else 'None'
    )

    n_affected = stations_df['storm_year'].sum()
    print(f"Storm flagging complete: {n_affected} stations affected by "
          f"≥1 major storm")
    print(f"Storm exposure by category:")
    for cat in [0, 1, 2, 3, 4, 5]:
        n = (stations_df['max_category_hit'] == cat).sum()
        if n > 0:
            label = 'Tropical Storm' if cat == 0 else f'Category {cat}'
            print(f"  {label}: {n} stations")
    return stations_df

def get_storm_affected_stations(station_df, storm_name, storm_year):
    """Get stations affected by a specific named storm"""
    storm = STORMS_DF[
        (STORMS_DF['name'] == storm_name) &
        (STORMS_DF['year'] == storm_year)
    ]
    if len(storm) == 0:
        print(f"Storm not found: {storm_name} {storm_year}")
        return pd.DataFrame()

    storm = storm.iloc[0]
    distances = station_df.apply(
        lambda row: haversine_km(
            storm['landfall_lat'], storm['landfall_lon'],
            row['lat'], row['lon']
        ) if pd.notna(row['lat']) else 999,
        axis=1
    )
    affected = station_df[distances <= storm['impact_radius_km']].copy()
    affected['distance_from_landfall_km'] = distances[affected.index]
    return affected

def simulate_storm_impact(station_df, landfall_lat, landfall_lon,
                          scenario_name, surge_multiplier=1.0):
    """
    Simulate impact of a storm scenario on current station conditions.
    Returns stations with estimated post-storm loss probability boost.
    Used for future scenario simulation in dashboard.
    """
    if scenario_name not in STORM_SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        return station_df

    scenario = STORM_SCENARIOS[scenario_name]
    radius = scenario['impact_radius_km']
    surge_mid = np.mean(scenario['surge_ft_range']) * surge_multiplier

    station_df = station_df.copy()
    distances = station_df.apply(
        lambda row: haversine_km(
            landfall_lat, landfall_lon,
            row['lat'], row['lon']
        ) if pd.notna(row['lat']) else 999,
        axis=1
    )

    # Distance decay: closer stations hit harder
    in_radius = distances <= radius
    decay = np.maximum(0, 1 - (distances / radius))

    # Surge impact on loss probability
    # Higher surge + closer distance = higher loss probability boost
    surge_factor = min(1.0, surge_mid / 20.0)
    # increase the decay as it goes inland, closer stations get hit much harder
    impact_boost = (decay**2) * surge_factor * 0.5

    station_df['scenario_loss_probability'] = np.minimum(
        1.0,
        station_df.get('loss_probability', 0.3) + impact_boost
    )
    station_df['in_storm_path'] = in_radius
    station_df['distance_from_scenario_km'] = distances
    station_df['scenario_name'] = scenario_name
    station_df['estimated_surge_ft'] = surge_mid * decay

    n_affected = in_radius.sum()
    print(f"Scenario '{scenario_name}': {n_affected} stations in impact zone")
    return station_df

if __name__ == '__main__':
    print("Gulf Coast Storm Registry")
    print("=" * 60)
    print(STORMS_DF[['name', 'year', 'category', 'max_surge_ft',
                      'basins_affected']].to_string(index=False))
    print(f"\nTotal storms: {len(GULF_STORMS)}")
    print(f"Year range: {STORMS_DF['year'].min()}–{STORMS_DF['year'].max()}")
    print(f"Categories: {sorted(STORMS_DF['category'].unique())}")