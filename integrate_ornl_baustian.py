"""
integrate_ornl_baustian.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Integrates three external carbon datasets into the pipeline:

    1. Baustian et al. (2021) — USGS ScienceBase
       Long-term soil carbon data from 24 sites in Terrebonne
       and Barataria basins. Directly cited in the RFP.
       Files: long_term_carbon.csv, radionuclide.csv, site_history.csv

    2. Delta-X Soil Properties (ORNL DAAC 2239)
       Soil carbon, bulk density, organic matter at 6 CRMS-adjacent
       sites in Atchafalaya and Terrebonne basins, 2021.
       File: deltax_soil_properties.csv

    3. Tidal Wetland Soil Carbon Stocks (ORNL DAAC 1612)
       Spatial carbon stock estimates (GeoTIFF) for US coastal wetlands
       2006-2010. Used for spatial validation of GPR predictions.

OUTPUTS:
    data/carbon_training_labels.csv   — enriched carbon training labels
                                        for GPR (CRMS stations only)
    data/carbon_validation_set.csv    — independent validation set
                                        (BA/TE non-CRMS sites + Delta-X)
    data/ornl_carbon_spatial.csv      — ORNL 1612 pixel values clipped
                                        to Louisiana for spatial validation

DESIGN:
    Training set  → CRMS stations with full hydro + soil + spectral data
    Validation set → BA/TE Baustian sites + Delta-X sites (never in training)
    This provides independent spatial validation of GPR carbon predictions
    not possible with CRMS-only data.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import os
import warnings

from rasterio import transform
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# HABITAT TYPE CODE MAPPING
# From Baustian et al. (2021) Table 1
# 1=Fresh, 2=Intermediate, 3=Brackish, 4=Saline
# ─────────────────────────────────────────────
HABITAT_MAP = {
    1: 'Fresh',
    2: 'Intermediate',
    3: 'Brackish',
    4: 'Saline',
    '1': 'Fresh',
    '2': 'Intermediate',
    '3': 'Brackish',
    '4': 'Saline'
}

# ─────────────────────────────────────────────
# HELPER: Standardize station IDs
# Converts numeric IDs like 175 → CRMS0175
# Leaves BA-01-04 style IDs unchanged
# ─────────────────────────────────────────────
def standardize_station_id(sid):
    """Convert numeric or mixed IDs to standard format"""
    sid = str(sid).strip()
    # Already has prefix
    if any(sid.startswith(p) for p in
           ['CRMS', 'BA-', 'TE-', 'BS-', 'TV-',
            'PO-', 'ME-', 'CS-', 'DCP', 'BAF']):
        return sid
    # Pure numeric — pad to 4 digits and add CRMS prefix
    try:
        num = int(sid)
        return f'CRMS{num:04d}'
    except ValueError:
        return sid

# ─────────────────────────────────────────────
# STEP 1: LOAD BAUSTIAN LONG-TERM CARBON
# Columns: Batch, Site, 2014_Habitat Type,
#          Most_Freq_Occ_Habitat, Core Increment,
#          Moisture, Bulk Density, Organic matter
# ─────────────────────────────────────────────
def load_baustian_carbon(filepath='data/baustian_longterm_carbon.csv'):
    """Load Baustian soil carbon data and aggregate to site level"""
    print("\n--- Loading Baustian Long-term Carbon ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Download from: https://doi.org/10.5066/P93U3B3E")
        return None

    df = pd.read_csv(filepath)
    print(f"Raw: {df.shape} | Columns: {list(df.columns)}")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Find key columns flexibly
    site_col = next((c for c in df.columns
                     if 'site' in c.lower()), None)
    habitat_col = next((c for c in df.columns
                        if 'habitat' in c.lower()
                        and '2014' in c.lower()), None)
    bulk_col = next((c for c in df.columns
                     if 'bulk' in c.lower()
                     or 'density' in c.lower()), None)
    organic_col = next((c for c in df.columns
                        if 'organic' in c.lower()), None)
    depth_col = next((c for c in df.columns
                      if 'increment' in c.lower()
                      or 'depth' in c.lower()), None)

    print(f"  Site col: {site_col}")
    print(f"  Habitat col: {habitat_col}")
    print(f"  Bulk density col: {bulk_col}")
    print(f"  Organic matter col: {organic_col}")
    print(f"  Depth col: {depth_col}")

    # Standardize station IDs
    df['station_id'] = df[site_col].apply(standardize_station_id)

    # Map habitat codes to names
    if habitat_col:
        df['marsh_type_baustian'] = df[habitat_col].map(HABITAT_MAP)

    # Convert to numeric
    for col in [bulk_col, organic_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to surface samples (0-10cm) for carbon stock comparison
    if depth_col:
        depth_parsed = df[depth_col].astype(str).str.extract(r'(\d+)-(\d+)')
        df['depth_top'] = pd.to_numeric(depth_parsed[0], errors='coerce')
        df['depth_bot'] = pd.to_numeric(depth_parsed[1], errors='coerce')
        surface = df[df['depth_top'] < 10].copy()
        print(f"  Surface samples (0-10cm): {len(surface)} rows")
        if len(surface) == 0:
            print(f"  Sample depths: {df[depth_col].head(5).tolist()}")
            print(f"  Using all samples as fallback")
            surface = df.copy()
    else:
        surface = df.copy()

    # Estimate carbon stock: bulk_density × (organic_matter/100)
    if bulk_col and organic_col:
        surface['carbon_stock_baustian'] = (
            pd.to_numeric(surface[bulk_col], errors='coerce') *
            pd.to_numeric(surface[organic_col], errors='coerce') / 100
        )

    # Aggregate to site level
    agg_cols = {}
    if bulk_col:
        agg_cols[bulk_col] = 'mean'
    if organic_col:
        agg_cols[organic_col] = 'mean'
    if 'carbon_stock_baustian' in surface.columns:
        agg_cols['carbon_stock_baustian'] = 'mean'
    if 'marsh_type_baustian' in surface.columns:
        agg_cols['marsh_type_baustian'] = lambda x: x.mode()[0] \
            if len(x) > 0 else np.nan

    site_level = surface.groupby('station_id').agg(agg_cols).reset_index()

    # Rename
    rename = {}
    if bulk_col:
        rename[bulk_col] = 'bulk_density_baustian'
    if organic_col:
        rename[organic_col] = 'organic_matter_baustian'
    site_level = site_level.rename(columns=rename)
    site_level['data_source'] = 'Baustian_2021'

    print(f"  Sites after aggregation: {len(site_level)}")
    print(f"  Station IDs: {site_level['station_id'].tolist()}")
    return site_level

# ─────────────────────────────────────────────
# STEP 2: LOAD BAUSTIAN RADIONUCLIDE DATA
# Columns: CRMS Site (numeric), Field Collection Date,
#          Core Increment, Pb-210, Cs-137
# Used for: accretion rate validation
# ─────────────────────────────────────────────
def load_baustian_radionuclide(
        filepath='data/baustian_radionuclide.csv'):
    """Load radionuclide dating data for accretion validation"""
    print("\n--- Loading Baustian Radionuclide Data ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    print(f"Raw: {df.shape} | Columns: {list(df.columns)}")

    # Find site column
    site_col = df.columns[0]
    df['station_id'] = df[site_col].apply(standardize_station_id)

    # Find Cs-137 column (marks 1963 — most reliable date marker)
    cs_col = next((c for c in df.columns if 'cs' in c.lower()
                   and '137' in c.lower()), None)
    pb_col = next((c for c in df.columns if 'excess' in c.lower()
                   and 'pb' in c.lower()), None)

    agg_cols = {'station_id': 'first'}
    if cs_col:
        df[cs_col] = pd.to_numeric(df[cs_col], errors='coerce')
        agg_cols[cs_col] = 'max'
    if pb_col:
        df[pb_col] = pd.to_numeric(df[pb_col], errors='coerce')
        agg_cols[pb_col] = 'mean'

    site_level = df.groupby('station_id').agg(
        {k: v for k, v in agg_cols.items() if k != 'station_id'}
    ).reset_index()

    rename = {}
    if cs_col:
        rename[cs_col] = 'cs137_max'
    if pb_col:
        rename[pb_col] = 'pb210_excess_mean'
    site_level = site_level.rename(columns=rename)
    site_level['data_source_radio'] = 'Baustian_2021_radionuclide'

    print(f"  Sites with radionuclide data: {len(site_level)}")
    return site_level

# ─────────────────────────────────────────────
# STEP 3: LOAD BAUSTIAN SITE HISTORY
# Columns: Site (numeric), 1949, 1968, ..., 2014
# Used for: marsh type temporal validation
# ─────────────────────────────────────────────
def load_baustian_site_history(
        filepath='data/baustian_site_history.csv'):
    """Load marsh type history for temporal instability validation"""
    print("\n--- Loading Baustian Site History ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    print(f"Raw: {df.shape} | Columns: {list(df.columns)}")

    site_col = df.columns[0]
    df['station_id'] = df[site_col].apply(standardize_station_id)

    # Year columns are everything except site
    year_cols = [c for c in df.columns
                 if c.isdigit() and 1940 <= int(c) <= 2030]

    # Map habitat codes to names for each year
    for col in year_cols:
        df[f'marsh_{col}'] = df[col].map(HABITAT_MAP)

    # Count how many times marsh type changed
    if year_cols:
        habitat_matrix = df[year_cols].apply(
            pd.to_numeric, errors='coerce'
        )
        df['n_habitat_changes'] = habitat_matrix.apply(
            lambda row: (row.diff().dropna() != 0).sum(), axis=1
        )
        df['marsh_type_2014'] = df[year_cols[-1]].map(HABITAT_MAP) \
            if year_cols else np.nan

    keep_cols = ['station_id', 'n_habitat_changes', 'marsh_type_2014'] + \
                [f'marsh_{y}' for y in year_cols]
    keep_cols = [c for c in keep_cols if c in df.columns]
    site_level = df[keep_cols].copy()
    site_level['data_source_history'] = 'Baustian_2021_sitehistory'

    print(f"  Sites with history: {len(site_level)}")
    changed = (site_level['n_habitat_changes'] > 0).sum() \
        if 'n_habitat_changes' in site_level.columns else 'N/A'
    print(f"  Sites with ≥1 habitat change: {changed}")
    return site_level

# ─────────────────────────────────────────────
# STEP 4: LOAD DELTA-X SOIL PROPERTIES
# Columns: basin, campaign, date, latitude, longitude,
#          site, hydrogeomorphic_zone, bulk_density,
#          organic_matter_content, total_carbon_density
# ─────────────────────────────────────────────
def load_deltax_soil(
        filepath='data/deltax_soil_properties.csv'):
    """Load Delta-X soil properties and link to nearest CRMS station"""
    print("\n--- Loading Delta-X Soil Properties ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(' ', '_')
                  for c in df.columns]
    print(f"Raw: {df.shape} | Columns: {list(df.columns)}")

    # Find key columns
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c), None)
    site_col = next((c for c in df.columns if 'site' in c), None)
    carbon_col = next((c for c in df.columns
                       if 'carbon' in c and 'density' in c), None)
    bulk_col = next((c for c in df.columns
                     if 'bulk' in c), None)
    organic_col = next((c for c in df.columns
                        if 'organic' in c), None)
    depth_col = next((c for c in df.columns
                      if 'depth' in c), None)

    # Filter to surface samples (0-10cm)
    if depth_col:
        df['depth_str'] = df[depth_col].astype(str)
        df['depth_top'] = df['depth_str'].str.extract(
            r'(\d+)'
        )[0].astype(float)
        surface = df[df['depth_top'] <= 10].copy()
        print(f"  Surface samples: {len(surface)}")
    else:
        surface = df.copy()

    # Convert to numeric
    for col in [carbon_col, bulk_col, organic_col]:
        if col and col in surface.columns:
            surface[col] = pd.to_numeric(surface[col], errors='coerce')

    # Aggregate to site level
    agg_dict = {}
    if carbon_col:
        agg_dict[carbon_col] = 'mean'
    if bulk_col:
        agg_dict[bulk_col] = 'mean'
    if organic_col:
        agg_dict[organic_col] = 'mean'
    if lat_col:
        agg_dict[lat_col] = 'first'
    if lon_col:
        agg_dict[lon_col] = 'first'

    if site_col:
        site_level = surface.groupby(site_col).agg(agg_dict).reset_index()
        site_level = site_level.rename(columns={site_col: 'deltax_site'})
    else:
        site_level = surface.agg(agg_dict).to_frame().T

    # Rename columns
    rename = {}
    if carbon_col:
        rename[carbon_col] = 'carbon_density_deltax'
    if bulk_col:
        rename[bulk_col] = 'bulk_density_deltax'
    if organic_col:
        rename[organic_col] = 'organic_matter_deltax'
    if lat_col:
        rename[lat_col] = 'lat'
    if lon_col:
        rename[lon_col] = 'lon'
    site_level = site_level.rename(columns=rename)

    # Link to nearest CRMS station using coordinates
    coords = pd.read_csv('data/crms_all_stations_coords.csv')
    site_level['station_id'] = site_level.apply(
        lambda row: find_nearest_station(
            row.get('lat', np.nan),
            row.get('lon', np.nan),
            coords
        ), axis=1
    )

    site_level['data_source'] = 'DeltaX_2021'
    print(f"  Delta-X sites: {len(site_level)}")
    if 'station_id' in site_level.columns:
        print(f"  Linked to CRMS stations: "
              f"{site_level['station_id'].dropna().nunique()}")
    return site_level

# ─────────────────────────────────────────────
# HELPER: Find nearest CRMS station by coordinates
# ─────────────────────────────────────────────
def find_nearest_station(lat, lon, coords_df,
                          max_dist_km=50):
    """Find nearest station within max_dist_km"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan

    R = 6371
    coords_df = coords_df.dropna(subset=['Latitude', 'Longitude'])
    dlat = np.radians(coords_df['Latitude'] - lat)
    dlon = np.radians(coords_df['Longitude'] - lon)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat)) *
         np.cos(np.radians(coords_df['Latitude'])) *
         np.sin(dlon/2)**2)
    distances = R * 2 * np.arcsin(np.sqrt(a))

    min_idx = distances.idxmin()
    min_dist = distances[min_idx]

    if min_dist <= max_dist_km:
        return coords_df.loc[min_idx, 'Site_ID']
    return np.nan

# ─────────────────────────────────────────────
# STEP 5: LOAD ORNL 1612 TIDAL WETLAND CARBON
# GeoTIFF — clip to Louisiana, extract pixel values
# Used for spatial validation of GPR wall-to-wall predictions
# ─────────────────────────────────────────────
def load_ornl_1612_spatial(
        tif_path='data/ornl_tidal_wetland_soil_carbon.tif',
        output_path='data/ornl_carbon_spatial.csv'):
    """
    Clip ORNL 1612 carbon stock GeoTIFF to Louisiana Deltaic Plain
    and extract pixel values for spatial validation.
    """
    print("\n--- Loading ORNL 1612 Tidal Wetland Carbon (GeoTIFF) ---")
    if not os.path.exists(tif_path):
        print(f"GeoTIFF not found: {tif_path}")
        print("Download from: https://doi.org/10.3334/ORNLDAAC/1612")
        print("File: tidal_wetland_soil_carbon_stocks.tif")
        return None

    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import box
        from shapely.ops import transform
        import pyproj
        import json

        # Louisiana Deltaic Plain bounding box
        louisiana_bbox_latlon = box(-93.5, 28.5, -88.5, 30.5)

        with rasterio.open(tif_path) as src:
            print(f"  CRS: {src.crs}")
            print(f"  Resolution: {src.res}")
            print(f"  Bands: {src.count}")

            # Reproject bbox to match GeoTIFF CRS
            project = pyproj.Transformer.from_crs(
                'EPSG:4326', src.crs, always_xy=True
            ).transform
            louisiana_bbox = transform(project, louisiana_bbox_latlon)
            geom = [louisiana_bbox.__geo_interface__]

            # Clip to Louisiana
            out_image, out_transform = mask(src, geom, crop=True)
            data = out_image[0]  # First band

        # Convert to dataframe
        rows, cols = np.where(data > 0)  # Valid pixels
        lons, lats = rasterio.transform.xy(
            out_transform, rows, cols
        )
        carbon_values = data[rows, cols]

        df_spatial = pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'ornl_carbon_stock_gCcm2': carbon_values.astype(float)
        })
        # Remove nodata
        df_spatial = df_spatial[df_spatial['ornl_carbon_stock_gCcm2'] > 0]

        df_spatial.to_csv(output_path, index=False)
        print(f"  Valid pixels in Louisiana: {len(df_spatial):,}")
        print(f"  Carbon range: "
              f"{df_spatial['ornl_carbon_stock_gCcm2'].min():.3f}–"
              f"{df_spatial['ornl_carbon_stock_gCcm2'].max():.3f} g C/cm²")
        print(f"  Saved: {output_path}")
        return df_spatial

    except ImportError as e:
        print(f"  rasterio/shapely/pyproj import failed — skipping GeoTIFF extraction: {e}")
        print("  Install with: pip install rasterio shapely pyproj")
        return None
    except Exception as e:
        print(f"  Error processing GeoTIFF: {e}")
        return None

# ─────────────────────────────────────────────
# STEP 6: BUILD TRAINING AND VALIDATION SETS
# Training: CRMS stations with full data
# Validation: BA/TE Baustian sites + Delta-X (never in training)
# ─────────────────────────────────────────────
def build_carbon_datasets(baustian_carbon, baustian_radio,
                           baustian_history, deltax_soil,
                           coords_df):
    """
    Split into training labels (CRMS) and validation set (non-CRMS).
    Merges all Baustian datasets on station_id.
    """
    print("\n--- Building Carbon Training and Validation Sets ---")

    # Merge all Baustian data
    all_baustian = None
    for df, name in [(baustian_carbon, 'carbon'),
                     (baustian_radio, 'radio'),
                     (baustian_history, 'history')]:
        if df is None:
            continue
        if all_baustian is None:
            all_baustian = df
        else:
            all_baustian = all_baustian.merge(
                df, on='station_id', how='outer'
            )
        print(f"  After merging {name}: {len(all_baustian)} sites")

    if all_baustian is None:
        print("No Baustian data loaded")
        return None, None

    # Add coordinates
    all_baustian = all_baustian.merge(
        coords_df[['Site_ID', 'Latitude', 'Longitude']].rename(
            columns={'Site_ID': 'station_id',
                     'Latitude': 'lat', 'Longitude': 'lon'}
        ),
        on='station_id', how='left'
    )

    # Split: CRMS stations → training, BA/TE etc → validation
    is_crms = all_baustian['station_id'].str.startswith('CRMS')
    training = all_baustian[is_crms].copy()
    validation = all_baustian[~is_crms].copy()

    # Add Delta-X to validation set
    if deltax_soil is not None:
        validation = pd.concat(
            [validation, deltax_soil], ignore_index=True
        )

    training['split'] = 'training'
    validation['split'] = 'validation'

    training.to_csv('data/carbon_training_labels.csv', index=False)
    validation.to_csv('data/carbon_validation_set.csv', index=False)

    print(f"\n  Training set (CRMS): {len(training)} stations")
    print(f"  Validation set (non-CRMS + Delta-X): {len(validation)} sites")

    # Summary of carbon stock values
    if 'carbon_stock_baustian' in training.columns:
        valid = training['carbon_stock_baustian'].dropna()
        print(f"\n  Training carbon stock range: "
              f"{valid.min():.4f}–{valid.max():.4f} g C/cm³")
    if 'carbon_stock_baustian' in validation.columns:
        valid_v = validation['carbon_stock_baustian'].dropna()
        if len(valid_v) > 0:
            print(f"  Validation carbon stock range: "
                  f"{valid_v.min():.4f}–{valid_v.max():.4f} g C/cm³")

    return training, validation

# ─────────────────────────────────────────────
# STEP 7: MERGE TRAINING LABELS INTO MASTER
# Enriches crms_master_with_hydro.csv with
# Baustian carbon measurements where available
# ─────────────────────────────────────────────
def enrich_master_with_carbon(
        training_labels,
        master_path='data/crms_master_with_hydro.csv',
        output_path='data/crms_master_enriched.csv'):
    """Add Baustian carbon labels to master dataset"""
    print("\n--- Enriching Master Dataset with Carbon Labels ---")

    if not os.path.exists(master_path):
        print(f"Master not found: {master_path}")
        print("Run 02_crms_preprocessing.py and merge_hydro_chunks.py first")
        return None

    master = pd.read_csv(master_path)
    print(f"Master: {master.shape}")

    if training_labels is None or len(training_labels) == 0:
        print("No training labels to merge")
        return master

    # Select carbon columns to merge
    carbon_cols = ['station_id', 'carbon_stock_baustian',
                   'bulk_density_baustian', 'organic_matter_baustian',
                   'marsh_type_baustian', 'n_habitat_changes',
                   'cs137_max', 'pb210_excess_mean']
    merge_cols = [c for c in carbon_cols if c in training_labels.columns]

    enriched = master.merge(
        training_labels[merge_cols],
        on='station_id', how='left'
    )

    # Where Baustian carbon exists, use it to supplement
    # existing carbon_stock estimate from bulk density × organic matter
    if 'carbon_stock_baustian' in enriched.columns:
        n_baustian = enriched['carbon_stock_baustian'].notna().sum()
        print(f"  Stations with Baustian carbon validation: {n_baustian}")

        # Use Baustian as ground truth where available
        enriched['carbon_stock_validated'] = np.where(
            enriched['carbon_stock_baustian'].notna(),
            enriched['carbon_stock_baustian'],
            enriched.get('carbon_stock', np.nan)
        )

    enriched.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return enriched

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_integration():
    print("=" * 60)
    print("ORNL DAAC + BAUSTIAN INTEGRATION PIPELINE")
    print("=" * 60)

    # Load master coordinates
    coords_path = 'data/crms_all_stations_coords.csv'
    if not os.path.exists(coords_path):
        print(f"Coords not found: {coords_path}")
        print("Run the coordinate consolidation script first")
        return
    coords = pd.read_csv(coords_path)
    print(f"Loaded {len(coords)} station coordinates")

    # Load all datasets
    # Update filenames below to match your downloaded files
    baustian_carbon = load_baustian_carbon(
        'data/baustian_longterm_carbon.csv'
    )
    baustian_radio = load_baustian_radionuclide(
        'data/baustian_radionuclide.csv'
    )
    baustian_history = load_baustian_site_history(
        'data/baustian_site_history.csv'
    )
    deltax_soil = load_deltax_soil(
        'data/deltax_soil_properties.csv'
    )
    ornl_spatial = load_ornl_1612_spatial(
        'data/ornl_tidal_wetland_soil_carbon.tif'
    )

    # Build training and validation sets
    training, validation = build_carbon_datasets(
        baustian_carbon, baustian_radio,
        baustian_history, deltax_soil, coords
    )

    # Enrich master dataset
    enriched = enrich_master_with_carbon(training)

    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print("Outputs:")
    print("  data/carbon_training_labels.csv  — enriched GPR training")
    print("  data/carbon_validation_set.csv   — independent validation")
    print("  data/crms_master_enriched.csv    — master + carbon labels")
    if ornl_spatial is not None:
        print("  data/ornl_carbon_spatial.csv     — spatial validation layer")
    print("\nNext steps:")
    print("1. Update filepath in 03_gaussian_process_regression.py:")
    print("   filepath = 'data/crms_master_enriched.csv'")
    print("   target_col = 'carbon_stock_validated'")
    print("2. Run: python 03_gaussian_process_regression.py")
    print("3. Run: python 07_spatial_prediction.py")
    print("   (validates wall-to-wall predictions against ornl_carbon_spatial.csv)")
    print("=" * 60)

if __name__ == '__main__':
    run_integration()