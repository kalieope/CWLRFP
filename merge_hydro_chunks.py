"""
merge_hydro_chunks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Cleans and merges crms_hydro_averages.csv into the master dataset.
    All hydro data pasted into one file at data/crms_hydro_averages.csv

INPUTS:
    data/crms_hydro_averages.csv    — all hydro data pasted together
    data/crms_master.csv            — existing master dataset
    data/fused_dataset.csv          — CRMS + Sentinel-2 fused dataset

OUTPUTS:
    data/crms_hydro_combined.csv        — cleaned hydro file
    data/crms_hydro_station_year.csv    — one row per station per year
    data/crms_hydro_summary.csv         — one row per station (median)
    data/crms_master_with_hydro.csv     — master + hydro merged
    data/fused_dataset_with_hydro.csv   — fused dataset + hydro merged

RUN:
    python merge_hydro_chunks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

HYDRO_PATH = 'data/crms_hydro_averages.csv'
MASTER_PATH = 'data/crms_master.csv'
FUSED_PATH = 'data/fused_dataset.csv'

# ─────────────────────────────────────────────
# COLUMN MAPPING
# ─────────────────────────────────────────────
COLUMN_MAP = {
    'Station_id': 'raw_station_id',
    'year': 'year',
    'avg_salinity (ppt)': 'salinity',
    'std_deviation_salinity (ppt)': 'salinity_std',
    'min_salinity (ppt)': 'salinity_min',
    'max_salinity (ppt)': 'salinity_max',
    'avg_adj_water_elev_datum(ft)': 'water_elev_mean',
    'std_deviation_adj_water_elev_datum(ft)': 'water_elev_std',
    'min_adj_water_elev_datum(ft)': 'water_elev_min',
    'max_adj_water_elev_datum(ft)': 'water_elev_max',
    'avg_adj_water_elev_to_marsh(ft)': 'flood_depth',
    'std_deviation_adj_water_elev_to_marsh(ft)': 'flood_depth_std',
    'min_adj_water_elev_to_marsh(ft)': 'flood_depth_min',
    'max_adj_water_elev_to_marsh(ft)': 'flood_depth_max',
    'avg_temperature (C degC)': 'water_temp',
    'min_temperature (C degC)': 'water_temp_min',
    'max_temperature (C degC)': 'water_temp_max',
    'percent_salinity_complete': 'pct_salinity_complete',
    'percent_water_level_complete': 'pct_water_level_complete',
    'GEOID': 'geoid_version',
    'mean_Water_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_navd88',
    'min_Water_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_navd88_min',
    'max_Water_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_navd88_max',
    '90th%Upper_Water_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_90th',
    '10%thLower_Elev_Datum(ft, NAVD88, G12b)': 'water_elev_10th'
}

def run_hydro_merge():
    print("=" * 60)
    print("HYDRO MERGE PIPELINE")
    print("=" * 60)

    # ── LOAD ──
    if not os.path.exists(HYDRO_PATH):
        print(f"File not found: {HYDRO_PATH}")
        return
    df = pd.read_csv(HYDRO_PATH, low_memory=False)
    df = df[[c for c in df.columns if 'Unnamed' not in str(c)]]
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # ── RENAME ──
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    print(f"\nRenamed {len(rename)} columns")

    # ── FILTER H-TYPE SENSORS ONLY ──
    df['sensor_type'] = df['raw_station_id'].str.extract(r'-([A-Z])\d+')[0]
    before = len(df)
    df = df[df['sensor_type'] == 'H'].copy()
    print(f"H-type sensor filter: {before:,} → {len(df):,} rows")

    # ── STRIP STATION SUFFIX ──
    df['station_id'] = df['raw_station_id'].str.split('-').str[0].str.strip()
    print(f"Unique stations: {df['station_id'].nunique()}")

    # ── CONVERT TO NUMERIC ──
    numeric_cols = [
        'salinity', 'salinity_std', 'salinity_min', 'salinity_max',
        'water_elev_mean', 'water_elev_std', 'water_elev_min', 'water_elev_max',
        'flood_depth', 'flood_depth_std', 'flood_depth_min', 'flood_depth_max',
        'water_temp', 'pct_salinity_complete', 'pct_water_level_complete',
        'water_elev_navd88', 'water_elev_navd88_min', 'water_elev_navd88_max',
        'water_elev_90th', 'water_elev_10th'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    print("Numeric conversion complete")

    # ── GEOID DATUM FLAG ──
    # GEOID99 pre-2013, GEOID12B post-2013
    # Use NAVD88 corrected elevation where available
    if 'geoid_version' in df.columns:
        df['geoid_version'] = df['geoid_version'].str.strip()
        df['geoid99_flag'] = df['geoid_version'].str.contains(
            'GEOID99', case=False, na=False)
    else:
        df['geoid99_flag'] = df['year'] < 2014

    # Use NAVD88 corrected where available, raw otherwise
    if 'water_elev_navd88' in df.columns:
        df['water_elev_corrected'] = np.where(
            df['water_elev_navd88'].notna(),
            df['water_elev_navd88'],
            df.get('water_elev_mean', np.nan)
        )
    else:
        df['water_elev_corrected'] = df.get('water_elev_mean', np.nan)

    n_geoid99 = df['geoid99_flag'].sum()
    print(f"\nGEOID99 (pre-2013): {n_geoid99:,} records flagged")
    print("water_elev_corrected uses NAVD88 where available")

    # ── DERIVE TIDAL AMPLITUDE ──
    # Key predictor from Chenevert & Edmonds (2024)
    if 'water_elev_max' in df.columns and 'water_elev_min' in df.columns:
        df['tidal_amplitude'] = (
            df['water_elev_max'] - df['water_elev_min']
        ).abs()
        df.loc[df['tidal_amplitude'] > 10, 'tidal_amplitude'] = np.nan
        valid = df['tidal_amplitude'].notna().sum()
        print(f"\nTidal amplitude: {valid:,} valid records")
        print(f"  Mean: {df['tidal_amplitude'].mean():.3f} ft | "
              f"Range: {df['tidal_amplitude'].min():.3f}–"
              f"{df['tidal_amplitude'].max():.3f} ft")
    else:
        df['tidal_amplitude'] = np.nan
        print("Cannot derive tidal amplitude — missing min/max water elev")

    # ── DATA QUALITY FILTER ──
    before = len(df)
    sal_ok = df.get('pct_salinity_complete', pd.Series(100, index=df.index)) >= 50
    wl_ok = df.get('pct_water_level_complete', pd.Series(100, index=df.index)) >= 50
    no_info = (
        df.get('pct_salinity_complete', pd.Series(
            np.nan, index=df.index)).isna() &
        df.get('pct_water_level_complete', pd.Series(
            np.nan, index=df.index)).isna()
    )
    df = df[sal_ok | wl_ok | no_info].copy()
    print(f"\nQuality filter: {before:,} → {len(df):,} rows (≥50% complete)")

    # ── SAVE CLEANED FILE ──
    df.to_csv('data/crms_hydro_combined.csv', index=False)
    print("Saved: data/crms_hydro_combined.csv")

    # ── AGGREGATE TO STATION-YEAR ──
    agg_dict = {}
    for col in ['salinity', 'flood_depth', 'tidal_amplitude',
                'water_elev_corrected', 'water_temp',
                'salinity_std', 'flood_depth_std']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    agg_dict['year'] = 'first'  # keep year

    station_year = df.groupby(['station_id', 'year']).agg(
        {k: v for k, v in agg_dict.items() if k != 'year'}
    ).reset_index()

    # Rename to make clear these are annual means
    station_year = station_year.rename(columns={
        'salinity': 'salinity_annual_mean',
        'flood_depth': 'flood_depth_annual_mean',
        'tidal_amplitude': 'tidal_amplitude_annual_mean',
        'water_elev_corrected': 'water_elev_annual_mean',
        'water_temp': 'water_temp_annual_mean'
    })

    station_year.to_csv('data/crms_hydro_station_year.csv', index=False)
    print(f"\nStation-year: {len(station_year):,} records | "
          f"{station_year['station_id'].nunique()} stations")
    print(f"Year range: {station_year['year'].min()}–{station_year['year'].max()}")
    print("Saved: data/crms_hydro_station_year.csv")

    # ── STATION SUMMARY (median across years) ──
    measure_cols = [c for c in station_year.columns
                    if c not in ['station_id', 'year']]
    summary = station_year.groupby('station_id')[measure_cols].median().reset_index()
    year_counts = station_year.groupby('station_id')['year'].count().reset_index()
    year_counts.columns = ['station_id', 'hydro_years_available']
    summary = summary.merge(year_counts, on='station_id')

    summary.to_csv('data/crms_hydro_summary.csv', index=False)
    print(f"\nStation summary: {len(summary)} stations")
    print("Saved: data/crms_hydro_summary.csv")

    # Feature availability report
    for col in ['salinity_annual_mean', 'flood_depth_annual_mean',
                'tidal_amplitude_annual_mean']:
        if col in summary.columns:
            n = summary[col].notna().sum()
            print(f"  {col}: {n}/{len(summary)} stations ({n/len(summary)*100:.0f}%)")

    # ── MERGE WITH CRMS MASTER ──
    if os.path.exists(MASTER_PATH):
        master = pd.read_csv(MASTER_PATH)
        merged_master = master.merge(summary, on='station_id', how='left')
        matched = merged_master['salinity_annual_mean'].notna().sum() \
            if 'salinity_annual_mean' in merged_master.columns else 0
        print(f"\nMaster merge: {matched}/{len(merged_master)} stations matched")
        merged_master.to_csv('data/crms_master_with_hydro.csv', index=False)
        print("Saved: data/crms_master_with_hydro.csv")
    else:
        print(f"\nMaster not found: {MASTER_PATH} — run 02_crms_preprocessing.py first")

    # ── MERGE WITH FUSED DATASET ──
    if os.path.exists(FUSED_PATH):
        fused = pd.read_csv(FUSED_PATH)
        if 'year' not in fused.columns and 'date' in fused.columns:
            fused['year'] = pd.to_datetime(fused['date']).dt.year
        if 'year' in fused.columns:
            fused['year'] = pd.to_numeric(fused['year'], errors='coerce')
            station_year['year'] = pd.to_numeric(
                station_year['year'], errors='coerce')
            fused_hydro = fused.merge(
                station_year, on=['station_id', 'year'], how='left'
            )
        else:
            fused_hydro = fused.merge(summary, on='station_id', how='left')

        matched_fused = fused_hydro['salinity_annual_mean'].notna().sum() \
            if 'salinity_annual_mean' in fused_hydro.columns else 0
        print(f"Fused merge: {matched_fused:,}/{len(fused_hydro):,} records matched")
        fused_hydro.to_csv('data/fused_dataset_with_hydro.csv', index=False)
        print("Saved: data/fused_dataset_with_hydro.csv")
    else:
        print(f"Fused not found: {FUSED_PATH} — run 02_crms_preprocessing.py first")

    print("\n" + "=" * 60)
    print("HYDRO MERGE COMPLETE")
    print("Next steps:")
    print("1. Uncomment hydro features in 03_gaussian_process_regression.py")
    print("2. Uncomment hydro features in 05_c45_classification_temporal_roc.py")
    print("3. Run: python 03_gaussian_process_regression.py")
    print("4. Run: python 04_fpgrowth_pattern_mining.py")
    print("5. Run: python 05_c45_classification_temporal_roc.py")
    print("6. Run: python 07_spatial_prediction.py")
    print("=" * 60)

if __name__ == '__main__':
    run_hydro_merge()