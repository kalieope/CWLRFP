"""
02_crms_preprocessing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Complete CRMS data preprocessing pipeline. Loads all ground-based
    monitoring datasets, merges them at the station level, fuses with
    Sentinel-2 spectral features, and writes the analysis-ready dataset
    consumed by every downstream model script.

INPUTS (place all in data/ folder):
    crms_all_stations_coords.csv / crms_stations_coords.csv
    crms_marsh_class.csv
    crms_accretion_rates.csv
    crms_bulk_density.csv
    crms_percent_organic.csv
    crms_land_water.csv
    crms_hydro_averages.csv
    crms_percent_flooded.csv
    crms_sentinel2_features.csv      (from 01_gee_sentinel2_pipeline.py)
    deltaic_plain_elevation_30m.tif  (optional — extracted from GEE)

OUTPUTS:
    data/crms_master.csv                  — station-level merged dataset
    data/fused_dataset_final.csv          — station-month fused dataset
                                            (primary input for 03/04/05/07)
    data/crms_freshwater_intermediate.csv — habitat split
    data/crms_brackish.csv                — habitat split
    data/crms_saline.csv                  — habitat split
    data/crms_marsh_timeseries.csv
    data/crms_loss_timeseries.csv
    data/crms_hydro_station_year.csv
    data/crms_flooded_station_year.csv
    data/carbon_training_labels.csv       — requires integrate_ornl_baustian.py
    data/carbon_validation_set.csv        — requires integrate_ornl_baustian.py

EXECUTION ORDER:
    python 02_crms_preprocessing.py
    python fix_loss_target.py
    python fix_integrate.py
    (then run 03, 04, 05, 07, 06)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# HABITAT TYPE MAPPING
# ─────────────────────────────────────────────
HABITAT_MAP = {1: 'Fresh', 2: 'Intermediate', 3: 'Brackish', 4: 'Saline',
               '1': 'Fresh', '2': 'Intermediate', '3': 'Brackish', '4': 'Saline'}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def strip_suffix(station_id):
    """Strip plot suffix: CRMS0033-H01 → CRMS0033, BA-01-04-V06 → BA-01-04"""
    sid = str(station_id).strip()
    parts = sid.split('-')
    if parts[0] in ['BA','TE','BS','TV','PO','ME','CS','DCPTE','BAFS','AT']:
        return '-'.join(parts[:3]) if len(parts) >= 3 else sid
    return parts[0]

def drop_unnamed(df):
    return df[[c for c in df.columns if 'Unnamed' not in str(c)]]

def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')

# ─────────────────────────────────────────────
# 1. LOAD COORDINATES
# ─────────────────────────────────────────────
def load_coordinates():
    print("\n--- Loading Coordinates ---")
    path = ('data/crms_all_stations_coords.csv'
            if os.path.exists('data/crms_all_stations_coords.csv')
            else 'data/crms_stations_coords.csv')
    df = drop_unnamed(pd.read_csv(path))
    df['station_id'] = df['Site_ID'].apply(strip_suffix)
    df = df.drop_duplicates('station_id')
    df = df.rename(columns={'Longitude': 'lon', 'Latitude': 'lat'})
    print(f"Loaded {len(df)} station coordinates")
    return df[['station_id', 'lat', 'lon']]

# ─────────────────────────────────────────────
# 2. LOAD MARSH CLASS
# ─────────────────────────────────────────────
def load_marsh_class():
    print("\n--- Loading Marsh Class ---")
    df = drop_unnamed(pd.read_csv('data/crms_marsh_class.csv'))
    df['station_id'] = df['Station_Id'].apply(strip_suffix)
    df = df[~df['Marsh_Class'].str.lower().str.contains('swamp', na=False)]
    agg = df.groupby(['station_id', 'veg_year']).agg(
        Marsh_Class=('Marsh_Class', lambda x: x.mode()[0]),
        Basin=('Basin', lambda x: x.mode()[0])
    ).reset_index()
    latest = agg.sort_values('veg_year').groupby('station_id').last().reset_index()
    latest = latest.rename(columns={'Marsh_Class': 'marsh_type',
                                     'veg_year': 'marsh_year'})
    print(f"Stations: {latest['station_id'].nunique()}")
    print(latest['marsh_type'].value_counts())
    return latest, agg

# ─────────────────────────────────────────────
# 3. LOAD ACCRETION RATES
# ─────────────────────────────────────────────
def load_accretion():
    print("\n--- Loading Accretion Rates ---")
    df = drop_unnamed(pd.read_csv('data/crms_accretion_rates.csv'))
    df['station_id'] = df['Site_ID'].apply(strip_suffix)
    rate_col = next((c for c in df.columns
                     if 'fullterm' in c.lower()), None)
    if not rate_col:
        rate_col = next((c for c in df.columns
                         if 'shortterm' in c.lower()), None)
    df['accretion_rate'] = to_numeric_safe(df[rate_col])
    agg = df.groupby('station_id').agg(
        accretion_median=('accretion_rate', 'median'),
        accretion_n=('accretion_rate', 'count')
    ).reset_index()
    print(f"Stations with accretion: {len(agg)}")
    return agg

# ─────────────────────────────────────────────
# 4. LOAD BULK DENSITY
# ─────────────────────────────────────────────
def load_bulk_density():
    print("\n--- Loading Bulk Density ---")
    df = drop_unnamed(pd.read_csv('data/crms_bulk_density.csv'))
    df = df.rename(columns={'Site ID': 'Site_ID'})
    df['station_id'] = df['Site_ID'].apply(strip_suffix)
    depth_col = next((c for c in df.columns if 'depth' in c.lower()), None)
    bulk_col = next((c for c in df.columns if 'bulk' in c.lower()), None)
    if depth_col:
        df['depth'] = to_numeric_safe(df[depth_col])
        df = df[df['depth'] <= 10]
    if bulk_col:
        df['bulk_density'] = to_numeric_safe(df[bulk_col])
    agg = df.groupby('station_id')['bulk_density'].mean().reset_index()
    print(f"Stations with bulk density: {len(agg)}")
    return agg

# ─────────────────────────────────────────────
# 5. LOAD PERCENT ORGANIC
# ─────────────────────────────────────────────
def load_percent_organic():
    print("\n--- Loading Percent Organic ---")
    df = drop_unnamed(pd.read_csv('data/crms_percent_organic.csv'))
    df = df.rename(columns={'Site ID': 'Site_ID'})
    df['station_id'] = df['Site_ID'].apply(strip_suffix)
    depth_col = next((c for c in df.columns if 'depth' in c.lower()), None)
    organic_col = next((c for c in df.columns if 'organic' in c.lower()), None)
    if depth_col:
        df['depth'] = to_numeric_safe(df[depth_col])
        df = df[df['depth'] <= 10]
    if organic_col:
        df['percent_organic'] = to_numeric_safe(df[organic_col])
    agg = df.groupby('station_id')['percent_organic'].mean().reset_index()
    print(f"Stations with percent organic: {len(agg)}")
    return agg

# ─────────────────────────────────────────────
# 6. LOAD LAND/WATER
# ─────────────────────────────────────────────
def load_land_water():
    print("\n--- Loading Land/Water ---")
    df = drop_unnamed(pd.read_csv('data/crms_land_water.csv'))
    df['station_id'] = df['crms_site'].apply(strip_suffix)
    df['map_year'] = to_numeric_safe(df['map_year'])
    df['land_acres'] = to_numeric_safe(df['land_acres'])
    df['water_acres'] = to_numeric_safe(df['water_acres'])
    df['total_acres'] = df['land_acres'] + df['water_acres']
    df['water_fraction'] = df['water_acres'] / df['total_acres']
    df = df.sort_values(['station_id', 'map_year'])
    df['water_fraction_prev'] = df.groupby('station_id')['water_fraction'].shift(1)
    df['land_loss'] = ((df['water_fraction'] - df['water_fraction_prev']) > 0.05
                       ).astype(int)
    station_loss = df.groupby('station_id').agg(
        ever_lost_land=('land_loss', 'max'),
        water_fraction_latest=('water_fraction', 'last'),
    ).reset_index()
    print(f"Stations with loss events: {station_loss['ever_lost_land'].sum()}")
    
    # Create recent loss metrics (2015+) — compute both binary flag and continuous rate
    recent_df = df[df['map_year'] >= 2015].copy()
    recent_loss = recent_df.groupby('station_id').agg(
        recent_land_loss_binary=('land_loss', 'max'),  # Any loss event 2015+
        recent_loss_count=('land_loss', 'sum'),        # Number of loss events
        recent_years_total=('land_loss', 'count')      # Total years observed
    ).reset_index()
    # Compute loss rate as proportion of years with loss
    recent_loss['recent_loss_rate'] = recent_loss['recent_loss_count'] / recent_loss['recent_years_total']
    
    station_loss = station_loss.merge(
        recent_loss[['station_id', 'recent_land_loss_binary', 'recent_loss_rate']], 
        on='station_id', how='left'
    )
    station_loss['recent_land_loss_binary'] = station_loss['recent_land_loss_binary'].fillna(0)
    station_loss['recent_loss_rate'] = station_loss['recent_loss_rate'].fillna(0)
    
    print(f"  Recent loss (2015+) stats:")
    print(f"    Binary flag: {station_loss['recent_land_loss_binary'].sum()} stations with ≥1 event")
    print(f"    Loss rate: mean={station_loss['recent_loss_rate'].mean():.3f}, "
          f"max={station_loss['recent_loss_rate'].max():.3f}")
    
    return station_loss, df

# ─────────────────────────────────────────────
# 7. LOAD HYDRO AVERAGES
# ─────────────────────────────────────────────
def load_hydro():
    print("\n--- Loading Hydro Averages ---")
    path = 'data/crms_hydro_averages.csv'
    if not os.path.exists(path):
        print("Hydro not found — skipping")
        return None, None

    df = drop_unnamed(pd.read_csv(path, low_memory=False))
    col_map = {
        'Station_id': 'raw_station_id',
        'year': 'year',
        'avg_salinity (ppt)': 'salinity',
        'avg_adj_water_elev_to_marsh(ft)': 'flood_depth',
        'min_adj_water_elev_datum(ft)': 'water_elev_min',
        'max_adj_water_elev_datum(ft)': 'water_elev_max',
        'avg_temperature (C degC)': 'water_temp',
        'percent_salinity_complete': 'pct_salinity_complete',
        'percent_water_level_complete': 'pct_water_level_complete',
        'GEOID': 'geoid_version',
        'mean_Water_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_navd88'
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    # H-type sensors only
    df['sensor_type'] = df['raw_station_id'].str.extract(r'-([A-Z])\d+')[0]
    df = df[df['sensor_type'] == 'H'].copy()
    df['station_id'] = df['raw_station_id'].str.split('-').str[0].str.strip()

    # Numeric conversion
    for col in ['salinity', 'flood_depth', 'water_elev_min',
                'water_elev_max', 'water_temp']:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])
    df['year'] = to_numeric_safe(df['year']).astype('Int64')

    # GEOID correction
    if 'geoid_version' in df.columns:
        df['geoid_version'] = df['geoid_version'].str.strip()
        df['geoid99_flag'] = df['geoid_version'].str.contains(
            'GEOID99', case=False, na=False)
    else:
        df['geoid99_flag'] = df['year'] < 2014

    if 'water_elev_navd88' in df.columns:
        df['water_elev_navd88'] = to_numeric_safe(df['water_elev_navd88'])
        df['water_elev_corrected'] = np.where(
            df['water_elev_navd88'].notna(),
            df['water_elev_navd88'],
            df.get('water_elev_mean', np.nan)
        )
    else:
        df['water_elev_corrected'] = df.get('flood_depth', np.nan)

    # Tidal amplitude
    if 'water_elev_max' in df.columns and 'water_elev_min' in df.columns:
        df['tidal_amplitude'] = (df['water_elev_max'] - df['water_elev_min']).abs()
        df.loc[df['tidal_amplitude'] > 10, 'tidal_amplitude'] = np.nan

    # Quality filter
    sal_ok = pd.to_numeric(df.get('pct_salinity_complete',
        pd.Series(100, index=df.index)), errors='coerce').fillna(0) >= 50
    wl_ok = pd.to_numeric(df.get('pct_water_level_complete',
        pd.Series(100, index=df.index)), errors='coerce').fillna(0) >= 50
    no_info = (
        pd.to_numeric(df.get('pct_salinity_complete',
            pd.Series(np.nan, index=df.index)), errors='coerce').isna() &
        pd.to_numeric(df.get('pct_water_level_complete',
            pd.Series(np.nan, index=df.index)), errors='coerce').isna()
        )
    df = df[sal_ok | wl_ok | no_info].copy()

    # Station-year aggregation
    agg_cols = {k: 'mean' for k in ['salinity', 'flood_depth',
                'tidal_amplitude', 'water_temp'] if k in df.columns}
    station_year = df.groupby(['station_id', 'year']).agg(agg_cols).reset_index()
    station_year = station_year.rename(columns={
        'salinity': 'salinity_annual_mean',
        'flood_depth': 'flood_depth_annual_mean',
        'tidal_amplitude': 'tidal_amplitude_annual_mean',
        'water_temp': 'water_temp_annual_mean'
    })

    # Station summary
    measure_cols = [c for c in station_year.columns
                    if c not in ['station_id', 'year']]
    summary = station_year.groupby('station_id')[measure_cols].median().reset_index()

    print(f"Stations with hydro: {len(summary)}")
    for col in ['salinity_annual_mean', 'flood_depth_annual_mean',
                'tidal_amplitude_annual_mean']:
        if col in summary.columns:
            n = summary[col].notna().sum()
            print(f"  {col}: {n}/{len(summary)} stations")

    return summary, station_year

# ─────────────────────────────────────────────
# 8. LOAD PERCENT FLOODED
# ─────────────────────────────────────────────
def load_percent_flooded():
    print("\n--- Loading Percent Flooded ---")
    path = 'data/crms_percent_flooded.csv'
    if not os.path.exists(path):
        print("Percent flooded not found — skipping")
        return None, None

    df = drop_unnamed(pd.read_csv(path, low_memory=False))
    print(f"Columns: {list(df.columns)}")

    # Standardize column names
    id_col = next((c for c in df.columns
                   if 'station' in c.lower() or 'site' in c.lower()), None)
    flooded_col = next((c for c in df.columns
                        if 'flood' in c.lower() or 'percent' in c.lower()
                        and 'water' not in c.lower()), None)
    year_col = next((c for c in df.columns
                     if 'year' in c.lower()), None)
    complete_col = next((c for c in df.columns
                         if 'complete' in c.lower()
                         or 'waterlevel' in c.lower()), None)

    if not id_col or not flooded_col:
        print(f"Could not detect columns — available: {list(df.columns)}")
        return None, None

    print(f"Using: id={id_col}, flooded={flooded_col}, year={year_col}")

    df['station_id'] = df[id_col].apply(strip_suffix)
    df['percent_flooded'] = to_numeric_safe(df[flooded_col])
    if year_col:
        df['year'] = to_numeric_safe(df[year_col]).astype('Int64')

    # Filter: remove zero values where water level completeness is also 0
    # (these are missing data, not true zero flooding)
    if complete_col:
        df['pct_complete'] = to_numeric_safe(df[complete_col])
        df = df[(df['percent_flooded'] > 0) | (df['pct_complete'] > 0)]

    # Remove obvious missing data (0% flooded with no date info)
    df = df[df['percent_flooded'].notna()]
    df = df[df['percent_flooded'] >= 0]
    df = df[df['percent_flooded'] <= 100]

    # Station-year level
    if year_col:
        station_year = df.groupby(['station_id', 'year']).agg(
            percent_flooded=('percent_flooded', 'mean')
        ).reset_index()
    else:
        station_year = df.groupby('station_id').agg(
            percent_flooded=('percent_flooded', 'mean')
        ).reset_index()

    # Station summary
    summary = station_year.groupby('station_id')['percent_flooded'].median().reset_index()

    print(f"Stations with percent flooded: {len(summary)}")
    print(f"Range: {summary['percent_flooded'].min():.1f}% – "
          f"{summary['percent_flooded'].max():.1f}%")
    print(f"Mean: {summary['percent_flooded'].mean():.1f}%")

    station_year.to_csv('data/crms_flooded_station_year.csv', index=False)
    summary.to_csv('data/crms_flooded_summary.csv', index=False)
    return summary, station_year

# ─────────────────────────────────────────────
# 9. LOAD ELEVATION FROM RASTER
# ─────────────────────────────────────────────
def load_elevation(coords_df):
    print("\n--- Loading Elevation ---")
    elev_path = 'data/deltaic_plain_elevation_30m.tif'
    elev_csv = 'data/crms_elevation.csv'

    # Use pre-extracted CSV if available
    if os.path.exists(elev_csv):
        elev = pd.read_csv(elev_csv)
        print(f"Loaded elevation from CSV: {len(elev)} stations")
        return elev

    if not os.path.exists(elev_path):
        print("Elevation raster not found — skipping")
        return None

    try:
        import rasterio
        from rasterio.sample import sample_gen

        with rasterio.open(elev_path) as src:
            xy = list(zip(coords_df['lon'], coords_df['lat']))
            elevations = [e[0] for e in sample_gen(src, xy)]

        coords_df = coords_df.copy()
        coords_df['elevation_m'] = elevations
        coords_df['elevation_m'] = to_numeric_safe(coords_df['elevation_m'])
        coords_df.loc[coords_df['elevation_m'] > 10, 'elevation_m'] = np.nan
        coords_df.loc[coords_df['elevation_m'] < -10, 'elevation_m'] = np.nan

        result = coords_df[['station_id', 'elevation_m']]
        result.to_csv(elev_csv, index=False)
        print(f"Elevation extracted: {result['elevation_m'].notna().sum()} stations")
        return result
    except Exception as e:
        print(f"Elevation extraction error: {e}")
        return None

# ─────────────────────────────────────────────
# 10. LOAD CARBON LABELS (Baustian + Delta-X)
# ─────────────────────────────────────────────
def load_carbon_labels():
    print("\n--- Loading Carbon Labels ---")

    # Check if already processed
    if os.path.exists('data/carbon_training_labels.csv'):
        training = pd.read_csv('data/carbon_training_labels.csv')
        validation = pd.read_csv('data/carbon_validation_set.csv')
        print(f"Loaded existing carbon labels: "
              f"{len(training)} training, {len(validation)} validation")
        return training, validation

    print("Carbon labels not found — run integrate_ornl_baustian.py first")
    return None, None

# ─────────────────────────────────────────────
# 11. OUTLIER REMOVAL
# Following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def remove_outliers(df, column='accretion_median'):
    if column not in df.columns:
        return df
    before = len(df)
    q75 = df[column].quantile(0.75)
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper = q75 + 1.5 * iqr
    df = df[df[column] <= upper]
    print(f"\nOutlier removal: {before - len(df)} stations removed, "
          f"{len(df)} remaining (upper bound: {upper:.3f})")
    return df

# ─────────────────────────────────────────────
# 12. STORM EVENT FLAGGING
# ─────────────────────────────────────────────
def flag_storms(df):
    print("\n--- Storm Event Flagging ---")
    print("⏸️  Storm processing deferred — adding placeholder fields")
    # DEFERRED: Storm data integration added back later after data format validation
    # For now, set defaults so models can train without storm features
    df['storm_year'] = False
    df['storm_count'] = 0
    df['max_category_hit'] = 0
    df['max_surge_experienced_ft'] = 0.0
    df['storms_hit_str'] = 'None'
    
    print(f"  Storm fields set to defaults (all False/0)")
    return df

# ─────────────────────────────────────────────
# 13. ESTIMATE CARBON STOCK
# ─────────────────────────────────────────────
def estimate_carbon_stock(df):
    if 'bulk_density' in df.columns and 'percent_organic' in df.columns:
        df['carbon_stock'] = (df['bulk_density'] *
                              df['percent_organic'] / 100)
        valid = df['carbon_stock'].notna().sum()
        print(f"\nCarbon stock: {valid} stations | "
              f"range {df['carbon_stock'].min():.4f}–"
              f"{df['carbon_stock'].max():.4f} g C/cm³")
    return df

# ─────────────────────────────────────────────
# 14. HABITAT SPLITS
# ─────────────────────────────────────────────
def split_by_habitat(df):
    print("\n--- Habitat Splits ---")
    splits = {
        'freshwater_intermediate': df[df['marsh_type'].isin(
            ['Fresh', 'Freshwater', 'Intermediate'])],
        'brackish': df[df['marsh_type'].isin(['Brackish'])],
        'saline': df[df['marsh_type'].isin(['Saline'])]
    }
    for name, subset in splits.items():
        print(f"  {name}: {len(subset)} stations")
    return splits

# ─────────────────────────────────────────────
# 15. MERGE SENTINEL-2 FEATURES
# ─────────────────────────────────────────────
def merge_sentinel2(master_df):
    print("\n--- Merging Sentinel-2 Features ---")
    gee_path = 'data/crms_sentinel2_features.csv'
    if not os.path.exists(gee_path):
        print("GEE export not found — run 01_gee_sentinel2_pipeline.py")
        return None

    s2 = drop_unnamed(pd.read_csv(gee_path))
    print(f"GEE export: {s2.shape}")

    # Ensure station_id exists
    if 'station_id' not in s2.columns:
        id_col = next((c for c in s2.columns
                       if 'station' in c.lower()), None)
        if id_col:
            s2['station_id'] = s2[id_col].apply(strip_suffix)

    fused = master_df.merge(s2, on='station_id', how='inner')
    print(f"Fused: {fused.shape[0]} station-month records")
    return fused

# ─────────────────────────────────────────────
# 16. BUILD MASTER DATASET
# ─────────────────────────────────────────────
def build_master_dataset():
    print("=" * 60)
    print("CRMS PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load all components
    coords = load_coordinates()
    marsh, marsh_ts = load_marsh_class()
    accretion = load_accretion()
    bulk = load_bulk_density()
    organic = load_percent_organic()
    loss_station, loss_ts = load_land_water()
    hydro_summary, hydro_ts = load_hydro()
    flooded_summary, flooded_ts = load_percent_flooded()
    elevation = load_elevation(coords)
    carbon_train, carbon_val = load_carbon_labels()

    # Merge all on station_id
    df = coords.copy()
    df = df.merge(accretion, on='station_id', how='left')
    df = df.merge(marsh[['station_id', 'marsh_type', 'Basin']],
                  on='station_id', how='left')
    df = df.merge(bulk, on='station_id', how='left')
    df = df.merge(organic, on='station_id', how='left')
    df = df.merge(loss_station, on='station_id', how='left')

    if hydro_summary is not None:
        df = df.merge(hydro_summary, on='station_id', how='left')

    if elevation is not None:
        df = df.merge(elevation, on='station_id', how='left')

    print(f"\nMerged dataset: {df.shape}")

    # Remove swamp
    before = len(df)
    df = df[~df['marsh_type'].isin(['Swamp', 'swamp'])]
    print(f"Swamp removal: {before - len(df)} stations removed")

    # Estimate carbon stock
    df = estimate_carbon_stock(df)

    # Add Baustian validated carbon where available
    if carbon_train is not None:
        baustian_cols = ['station_id', 'carbon_stock_baustian',
                         'marsh_type_baustian', 'n_habitat_changes']
        avail = [c for c in baustian_cols if c in carbon_train.columns]
        df = df.merge(carbon_train[avail], on='station_id', how='left')
        df['carbon_stock_validated'] = np.where(
            df.get('carbon_stock_baustian', pd.Series(np.nan,
                   index=df.index)).notna(),
            df['carbon_stock_baustian'],
            df['carbon_stock']
        )
        n_validated = df['carbon_stock_baustian'].notna().sum()
        print(f"Baustian validated carbon: {n_validated} stations")
    else:
        df['carbon_stock_validated'] = df['carbon_stock']

    # Remove outliers
    df = remove_outliers(df, 'accretion_median')

    # Flag storms
    df = flag_storms(df)

    print(f"\nFinal master dataset: {len(df)} stations")

    # Save master
    df.to_csv('data/crms_master.csv', index=False)
    print("Saved: data/crms_master.csv")

    # Save time-varying datasets
    marsh_ts.to_csv('data/crms_marsh_timeseries.csv', index=False)
    loss_ts.to_csv('data/crms_loss_timeseries.csv', index=False)
    if hydro_ts is not None:
        hydro_ts.to_csv('data/crms_hydro_station_year.csv', index=False)
    if flooded_ts is not None:
        flooded_ts.to_csv('data/crms_flooded_station_year.csv', index=False)

    # Save validation set
    if carbon_val is not None:
        carbon_val.to_csv('data/carbon_validation_set.csv', index=False)

    return df

# ─────────────────────────────────────────────
# 17. BUILD FUSED DATASET
# Merges master with Sentinel-2 and all time-varying features
# ─────────────────────────────────────────────
def build_fused_dataset(master_df):
    print("\n--- Building Fused Dataset ---")

    # Merge Sentinel-2
    fused = merge_sentinel2(master_df)
    if fused is None:
        print("Cannot build fused dataset without GEE export")
        return None

    # Merge time-varying hydro
    if os.path.exists('data/crms_hydro_station_year.csv'):
        hydro_ts = pd.read_csv('data/crms_hydro_station_year.csv')
        if 'year' in fused.columns and 'year' in hydro_ts.columns:
            fused['year'] = to_numeric_safe(fused['year'])
            hydro_ts['year'] = to_numeric_safe(hydro_ts['year'])
            fused = fused.merge(hydro_ts, on=['station_id', 'year'],
                                how='left', suffixes=('', '_hydro'))

    # Merge percent flooded directly
    flooded_summary_temp, _ = load_percent_flooded()
    if flooded_summary_temp is not None:
        fused = fused.merge(flooded_summary_temp, on='station_id', how='left')
        flood_cols = [c for c in fused.columns if 'percent_flood' in c.lower()]
        if len(flood_cols) == 1:
            fused = fused.rename(columns={flood_cols[0]: 'percent_flooded'})
        print(f"Percent flooded merged:"
              f"{fused['percent_flooded'].notna().sum():,} records")

    # Drop duplicate columns from merges
    dup_cols = [c for c in fused.columns if c.endswith('_hydro')]
    fused = fused.drop(columns=dup_cols, errors='ignore')

    # Drop system columns
    drop_cols = [c for c in fused.columns
                 if c.startswith('system:') or c == '.geo']
    fused = fused.drop(columns=drop_cols, errors='ignore')

    print(f"Fused dataset: {fused.shape}")
    print(f"Columns: {list(fused.columns)}")

    fused.to_csv('data/fused_dataset_final.csv', index=False)
    print("Saved: data/fused_dataset_final.csv")
    return fused

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Step 1: Build master station dataset
    master = build_master_dataset()

    # Step 2: Habitat splits
    splits = split_by_habitat(master)
    for name, subset in splits.items():
        subset.to_csv(f'data/crms_{name}.csv', index=False)
        print(f"Saved: data/crms_{name}.csv")

    # Step 3: Build fused dataset with all features
    fused = build_fused_dataset(master)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    if fused is not None:
        print(f"Final fused dataset: {fused.shape}")
        print(f"Features available: {list(fused.columns)}")
    print("\nNext steps:")
    print("1. python 03_gaussian_process_regression.py")
    print("2. python 04_fpgrowth_pattern_mining.py")
    print("3. python 05_c45_classification_temporal_roc.py")
    print("4. python 07_spatial_prediction.py")
    print("5. streamlit run 06_dashboard.py")
    print("=" * 60)