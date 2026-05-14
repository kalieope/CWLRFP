"""
02_crms_preprocessing.py
CRMS station data preprocessing following Chenevert & Edmonds (2024) exactly.
Merges all downloaded CRMS files into a single analysis-ready dataset.

Input files (place in data/ folder):
- crms_stations_coords.csv     : Site_ID, Longitude, Latitude
- crms_marsh_class.csv         : Station_Id, veg_year, Marsh_Class, Basin
- crms_accretion_rates.csv     : Site_ID, Acc_rate_fullterm, etc.
- crms_bulk_density.csv        : Site ID, Sample_Date, Sample Depth, Mean Bulk Density
- crms_percent_organic.csv     : Site ID, Sample_Date, Sample Depth, Mean Organic Matter
- crms_land_water.csv          : crms_site, map_year, land_acres, water_acres
- crms_hydro_averages.csv      : (large file — add when email arrives)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# HELPER: Strip plot suffix from Station_Id
# CRMS0033-V06 → CRMS0033
# CRMS0184-F02VC → CRMS0184
# Works for any suffix format by splitting on first hyphen
# ─────────────────────────────────────────────
def strip_suffix(station_id):
    """Extract base station ID by splitting on first hyphen"""
    return str(station_id).split('-')[0].strip()

# ─────────────────────────────────────────────
# HELPER: Drop unnamed columns
# ─────────────────────────────────────────────
def drop_unnamed(df):
    """Drop any Unnamed columns from CSV artifacts"""
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    return df.drop(columns=unnamed)

# ─────────────────────────────────────────────
# 1. LOAD COORDINATES
# Site_ID, Longitude, Latitude
# ─────────────────────────────────────────────
def load_coordinates(filepath='data/crms_stations_coords.csv'):
    print("\n--- Loading Coordinates ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)
    df['station_id'] = df['Site_ID'].apply(strip_suffix)
    df = df.drop_duplicates(subset='station_id')
    df = df.rename(columns={'Longitude': 'lon', 'Latitude': 'lat'})
    df = df[['station_id', 'lat', 'lon']]
    print(f"Loaded {len(df)} station coordinates")
    return df

# ─────────────────────────────────────────────
# 2. LOAD MARSH CLASS
# Station_Id (with plot suffix), veg_year, Marsh_Class, Basin
# Aggregate to station-year level by taking mode across plots
# Remove Swamp (no sedimentologic data)
# ─────────────────────────────────────────────
def load_marsh_class(filepath='data/crms_marsh_class.csv'):
    print("\n--- Loading Marsh Class ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)

    # Strip plot suffix to get base station ID
    df['station_id'] = df['Station_Id'].apply(strip_suffix)

    # Remove swamp communities (Chenevert & Edmonds 2024)
    before = len(df)
    df = df[~df['Marsh_Class'].str.lower().str.contains('swamp', na=False)]
    print(f"Removed {before - len(df)} swamp plot records")

    # Aggregate to station-year: mode marsh class, keep basin
    agg = df.groupby(['station_id', 'veg_year']).agg(
        Marsh_Class=('Marsh_Class', lambda x: x.mode()[0]),
        Basin=('Basin', lambda x: x.mode()[0])
    ).reset_index()

    # Also get the most recent marsh class per station (for station-level merge)
    latest = agg.sort_values('veg_year').groupby('station_id').last().reset_index()
    latest = latest.rename(columns={'Marsh_Class': 'marsh_type', 'veg_year': 'marsh_year'})

    print(f"Stations after aggregation: {latest['station_id'].nunique()}")
    print(f"Marsh class distribution:\n{latest['marsh_type'].value_counts()}")
    return latest, agg  # return both station-level and time-varying

# ─────────────────────────────────────────────
# 3. LOAD ACCRETION RATES
# Site_ID, Acc_rate_fullterm (cm/y) — use full term rate
# ─────────────────────────────────────────────
def load_accretion(filepath='data/crms_accretion_rates.csv'):
    print("\n--- Loading Accretion Rates ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)
    print(f"Columns: {list(df.columns)}")

    df['station_id'] = df['Site_ID'].apply(strip_suffix)

    # Use full-term accretion rate (Chenevert & Edmonds 2024)
    rate_col = [c for c in df.columns if 'fullterm' in c.lower() or 'full' in c.lower()]
    if not rate_col:
        # Fall back to short term if full term not available
        rate_col = [c for c in df.columns if 'shortterm' in c.lower() or 'short' in c.lower()]
    
    if rate_col:
        df['accretion_rate'] = pd.to_numeric(df[rate_col[0]], errors='coerce')
        print(f"Using accretion column: {rate_col[0]}")
    else:
        print(f"WARNING: Could not find accretion rate column")
        print(f"Available: {list(df.columns)}")
        df['accretion_rate'] = np.nan

    # Median accretion per station (some stations have multiple plot measurements)
    agg = df.groupby('station_id').agg(
        accretion_median=('accretion_rate', 'median'),
        accretion_n=('accretion_rate', 'count')
    ).reset_index()

    print(f"Stations with accretion data: {len(agg)}")
    print(f"Accretion rate range: {agg['accretion_median'].min():.2f} to "
          f"{agg['accretion_median'].max():.2f} cm/yr")
    return agg

# ─────────────────────────────────────────────
# 4. LOAD BULK DENSITY
# Site ID (space not underscore), Sample Depth, Mean Bulk Density
# Filter to surface samples 0-10cm (Baustian et al. 2021)
# ─────────────────────────────────────────────
def load_bulk_density(filepath='data/crms_bulk_density.csv'):
    print("\n--- Loading Bulk Density ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)

    # Normalize column name (has space not underscore)
    df = df.rename(columns={'Site ID': 'Site_ID'})
    df['station_id'] = df['Site_ID'].apply(strip_suffix)

    # Filter to surface samples (0-10cm) following Baustian et al. (2021)
    depth_col = [c for c in df.columns if 'depth' in c.lower() or 'Depth' in c]
    if depth_col:
        df['depth'] = pd.to_numeric(df[depth_col[0]], errors='coerce')
        df = df[df['depth'] <= 10]
        print(f"Filtered to surface samples (≤10cm): {len(df)} records")

    bulk_col = [c for c in df.columns if 'bulk' in c.lower() or 'Bulk' in c]
    if bulk_col:
        df['bulk_density'] = pd.to_numeric(df[bulk_col[0]], errors='coerce')
    
    agg = df.groupby('station_id').agg(
        bulk_density=('bulk_density', 'mean')
    ).reset_index()

    print(f"Stations with bulk density: {len(agg)}")
    return agg

# ─────────────────────────────────────────────
# 5. LOAD PERCENT ORGANIC
# Site ID, Sample Depth, Mean Organic Matter (%)
# Filter to surface samples 0-10cm
# ─────────────────────────────────────────────
def load_percent_organic(filepath='data/crms_percent_organic.csv'):
    print("\n--- Loading Percent Organic ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)

    df = df.rename(columns={'Site ID': 'Site_ID'})
    df['station_id'] = df['Site_ID'].apply(strip_suffix)

    # Filter to surface samples
    depth_col = [c for c in df.columns if 'depth' in c.lower()]
    if depth_col:
        df['depth'] = pd.to_numeric(df[depth_col[0]], errors='coerce')
        df = df[df['depth'] <= 10]

    organic_col = [c for c in df.columns if 'organic' in c.lower()]
    if organic_col:
        df['percent_organic'] = pd.to_numeric(df[organic_col[0]], errors='coerce')

    agg = df.groupby('station_id').agg(
        percent_organic=('percent_organic', 'mean')
    ).reset_index()

    print(f"Stations with percent organic: {len(agg)}")
    return agg

# ─────────────────────────────────────────────
# 6. ESTIMATE CARBON STOCK
# Carbon stock (g C/cm3) = bulk density × percent organic / 100
# Following Baustian et al. (2021)
# ─────────────────────────────────────────────
def estimate_carbon_stock(df):
    """Estimate carbon stock from bulk density and percent organic"""
    if 'bulk_density' in df.columns and 'percent_organic' in df.columns:
        df['carbon_stock'] = df['bulk_density'] * (df['percent_organic'] / 100)
        valid = df['carbon_stock'].notna().sum()
        print(f"\nCarbon stock estimated for {valid} stations")
        print(f"Range: {df['carbon_stock'].min():.4f} to "
              f"{df['carbon_stock'].max():.4f} g C/cm³")
    else:
        print("WARNING: Cannot estimate carbon stock — missing bulk density or organic matter")
        df['carbon_stock'] = np.nan
    return df

# ─────────────────────────────────────────────
# 7. LOAD LAND/WATER
# crms_site, map_year, land_acres, water_acres
# Derive binary loss label: 1 if water_acres increased year-over-year
# ─────────────────────────────────────────────
def load_land_water(filepath='data/crms_land_water.csv'):
    print("\n--- Loading Land/Water ---")
    df = pd.read_csv(filepath)
    df = drop_unnamed(df)
    print(f"Columns: {list(df.columns)}")

    df['station_id'] = df['crms_site'].apply(strip_suffix)
    df['map_year'] = pd.to_numeric(df['map_year'], errors='coerce')
    df['land_acres'] = pd.to_numeric(df['land_acres'], errors='coerce')
    df['water_acres'] = pd.to_numeric(df['water_acres'], errors='coerce')

    # Derive loss label: water fraction per station-year
    df['total_acres'] = df['land_acres'] + df['water_acres']
    df['water_fraction'] = df['water_acres'] / df['total_acres']

    # Binary loss: 1 if water fraction increased by >5% from previous year
    df = df.sort_values(['station_id', 'map_year'])
    df['water_fraction_prev'] = df.groupby('station_id')['water_fraction'].shift(1)
    df['land_loss'] = (
        (df['water_fraction'] - df['water_fraction_prev']) > 0.05
    ).astype(int)

    # Station-level: did it ever lose land?
    station_loss = df.groupby('station_id').agg(
        ever_lost_land=('land_loss', 'max'),
        water_fraction_latest=('water_fraction', 'last'),
        loss_year=('map_year', lambda x: x[df.loc[x.index, 'land_loss'] == 1].min()
                   if (df.loc[x.index, 'land_loss'] == 1).any() else np.nan)
    ).reset_index()

    print(f"Stations with land loss events: {station_loss['ever_lost_land'].sum()}")
    return station_loss, df  # station-level and time-varying

# ─────────────────────────────────────────────
# 8. LOAD HYDRO AVERAGES
# Large file — run after email arrives
# Expected columns will vary — script auto-detects key variables
# ─────────────────────────────────────────────
def load_hydro(filepath='data/crms_hydro_averages.csv'):
    print("\n--- Loading Hydro Averages ---")
    if not os.path.exists(filepath):
        print(f"Hydro file not yet available at {filepath}")
        print("Will merge when CRMS email arrives")
        return None

    df = pd.read_csv(filepath)
    df = drop_unnamed(df)
    print(f"Hydro columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Auto-detect station ID column
    id_cols = [c for c in df.columns if 'site' in c.lower() or 'station' in c.lower()]
    if id_cols:
        df['station_id'] = df[id_cols[0]].apply(strip_suffix)
    
    # Auto-detect key hydrological variables
    # Will print available columns so you can verify mapping
    flood_cols = [c for c in df.columns if 'flood' in c.lower()]
    salinity_cols = [c for c in df.columns if 'salin' in c.lower()]
    water_cols = [c for c in df.columns if 'water' in c.lower() and 'level' in c.lower()]
    tidal_cols = [c for c in df.columns if 'tidal' in c.lower() or 'tide' in c.lower()]

    print(f"Flood columns found: {flood_cols}")
    print(f"Salinity columns found: {salinity_cols}")
    print(f"Water level columns found: {water_cols}")
    print(f"Tidal columns found: {tidal_cols}")

    return df

# ─────────────────────────────────────────────
# 9. OUTLIER REMOVAL
# 75th percentile + 1.5x IQR on accretion rate
# Following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def remove_outliers(df, column='accretion_median'):
    """Remove outlier stations"""
    if column not in df.columns:
        print(f"Column {column} not found, skipping outlier removal")
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
# 10. LOG TRANSFORM DISTANCE VARIABLES
# Following Chenevert & Edmonds (2024)
# Distance variables added from Google Earth Engine export
# ─────────────────────────────────────────────
def log_transform_distances(df):
    """Log-transform distance variables for normality"""
    for col in ['dist_to_water', 'dist_to_river']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            print(f"Log-transformed: {col}")
    return df

# ─────────────────────────────────────────────
# 11. HABITAT-STRATIFIED SPLITS
# Three modeling subsets following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def split_by_habitat(df):
    """Split dataset into habitat-stratified modeling subsets"""
    print("\n--- Habitat Splits ---")
    splits = {
        'freshwater_intermediate': df[df['marsh_type'].isin(
            ['Fresh', 'Freshwater', 'Intermediate'])],
        'brackish': df[df['marsh_type'].isin(['Brackish'])],
        'saline': df[df['marsh_type'].isin(['Saline'])]
    }
    for name, subset in splits.items():
        print(f"{name}: {len(subset)} stations")
    return splits

# ─────────────────────────────────────────────
# 12. BACKWARD ELIMINATION FEATURE SELECTION
# p < 0.05 threshold, following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def backward_elimination(X, y, feature_names, significance=0.05):
    """Backward elimination with p < 0.05"""
    features = list(feature_names)
    print(f"\nBackward elimination starting with: {features}")

    while len(features) > 1:
        X_sm = sm.add_constant(X[features])
        model = sm.OLS(y, X_sm).fit()
        p_values = model.pvalues[1:]  # exclude intercept
        max_p = p_values.max()
        if max_p > significance:
            remove = p_values.idxmax()
            features.remove(remove)
            print(f"  Removed: {remove} (p={max_p:.4f})")
        else:
            break

    print(f"Selected features: {features}")
    return features

# ─────────────────────────────────────────────
# 13. FULL MERGE PIPELINE
# ─────────────────────────────────────────────
def build_master_dataset():
    """Merge all CRMS files into single analysis-ready dataset"""
    print("=" * 60)
    print("CRMS PREPROCESSING PIPELINE")
    print("Following Chenevert & Edmonds (2024)")
    print("=" * 60)

    # Load all files
    coords = load_coordinates()
    marsh, marsh_timeseries = load_marsh_class()
    accretion = load_accretion()
    bulk = load_bulk_density()
    organic = load_percent_organic()
    loss_station, loss_timeseries = load_land_water()

    # Merge on station_id
    df = coords.copy()
    df = df.merge(accretion, on='station_id', how='left')
    df = df.merge(marsh[['station_id', 'marsh_type', 'Basin']], 
                  on='station_id', how='left')
    df = df.merge(bulk, on='station_id', how='left')
    df = df.merge(organic, on='station_id', how='left')
    df = df.merge(loss_station, on='station_id', how='left')

    print(f"\nMerged dataset: {df.shape}")
    print(f"Stations: {df['station_id'].nunique()}")

    # Estimate carbon stock
    df = estimate_carbon_stock(df)

    # Remove swamp (already done in marsh class but double-check)
    before = len(df)
    df = df[~df['marsh_type'].isin(['Swamp', 'swamp'])]
    print(f"\nSwamp removal: {before - len(df)} stations removed")

    # Flag storm-affected stations
    print("\n--- Storm Event Flagging ---")
    try:
        from storm_events import flag_storm_affected_stations
        df = flag_storm_affected_stations(df)
        print("Storm flags added: storm_year, storms_hit_str, max_category_hit, max_surge_experienced_ft")
    except ImportError:
        print("storm_events.py not found -- skipping storm flagging")
        df["storm_year"] = False
        df["storm_count"] = 0
        df["max_category_hit"] = 0
        df["max_surge_experienced_ft"] = 0.0
        df["storms_hit_str"] = "None"

    # Remove outliers on accretion
    df = remove_outliers(df, 'accretion_median')

    # Log transform distances (will be populated after GEE merge)
    df = log_transform_distances(df)

    print(f"\nFinal dataset: {len(df)} stations")

    # Save master dataset
    df.to_csv('data/crms_master.csv', index=False)
    print("Saved: data/crms_master.csv")

    # Save time-varying datasets for temporal mining
    marsh_timeseries.to_csv('data/crms_marsh_timeseries.csv', index=False)
    loss_timeseries.to_csv('data/crms_loss_timeseries.csv', index=False)

    return df

# ─────────────────────────────────────────────
# 14. MERGE WITH HYDRO (run when email arrives)
# ─────────────────────────────────────────────
def merge_hydro(master_path='data/crms_master.csv',
                hydro_path='data/crms_hydro_averages.csv'):
    """Merge hydro averages into master dataset when available"""
    print("\n--- Merging Hydro Averages ---")
    master = pd.read_csv(master_path)
    hydro = load_hydro(hydro_path)

    if hydro is None:
        print("Hydro not yet available — skipping")
        return master

    # Will update column mapping here once we see actual hydro columns
    merged = master.merge(hydro, on='station_id', how='left')
    merged.to_csv('data/crms_master_with_hydro.csv', index=False)
    print(f"Merged with hydro: {merged.shape}")
    print("Saved: data/crms_master_with_hydro.csv")
    return merged

# ─────────────────────────────────────────────
# 15. MERGE WITH GEE SENTINEL-2 EXPORT
# Run after GEE export downloads from Google Drive
# ─────────────────────────────────────────────
def merge_sentinel2(master_path='data/crms_master_with_hydro.csv',
                    gee_path='data/crms_sentinel2_features.csv'):
    """Merge GEE Sentinel-2 features into master dataset"""
    print("\n--- Merging Sentinel-2 Features ---")
    if not os.path.exists(gee_path):
        print(f"GEE export not yet available at {gee_path}")
        print("Download from Google Drive folder CWL_RFP when task completes")
        return None

    master = pd.read_csv(master_path)
    s2 = pd.read_csv(gee_path)
    print(f"GEE export columns: {list(s2.columns)}")

    # Merge on station_id
    fused = master.merge(s2, on='station_id', how='inner')
    fused = log_transform_distances(fused)

    fused.to_csv('data/fused_dataset.csv', index=False)
    print(f"Fused dataset: {fused.shape}")
    print("Saved: data/fused_dataset.csv")
    return fused

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Step 1: Build master dataset from all available files
    master = build_master_dataset()

    # Step 2: Split by habitat for modeling
    splits = split_by_habitat(master)
    for name, subset in splits.items():
        subset.to_csv(f'data/crms_{name}.csv', index=False)
        print(f"Saved: data/crms_{name}.csv")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("Next steps:")
    print("1. When hydro email arrives: run merge_hydro()")
    print("2. When GEE export completes: run merge_sentinel2()")
    print("3. Then run: 03_gaussian_process_regression.py")
    print("=" * 60)