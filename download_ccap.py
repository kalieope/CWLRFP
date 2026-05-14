"""
download_ccap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Downloads and processes NOAA C-CAP (Coastal Change Analysis
    Program) land cover classification for Louisiana.
    Provides 30m resolution marsh type classification across the
    full Deltaic Plain — replacing NDVI-threshold approximation
    in wall-to-wall predictions.

    C-CAP classes relevant to this project:
    9  = Palustrine Forested Wetland
    10 = Palustrine Scrub/Shrub Wetland
    11 = Palustrine Emergent Wetland (fresh/intermediate marsh)
    12 = Estuarine Forested Wetland
    13 = Estuarine Scrub/Shrub Wetland
    14 = Estuarine Emergent Wetland (brackish/saline marsh)
    21 = Open Water

MANUAL DOWNLOAD:
    Go to: https://coast.noaa.gov/digitalcoast/data/ccapregional.html
    Select: Louisiana, most recent year available
    Download the GeoTIFF
    Save as: data/ccap_louisiana.tif

RUN AFTER DOWNLOAD:
    python download_ccap.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# C-CAP CLASS TO MARSH TYPE MAPPING
# Maps C-CAP land cover codes to CRMS habitat types
# ─────────────────────────────────────────────
CCAP_TO_MARSH = {
    9:  'Swamp',          # Palustrine Forested — exclude per Chenevert & Edmonds
    10: 'Swamp',          # Palustrine Scrub/Shrub — exclude
    11: 'Fresh',          # Palustrine Emergent — fresh/intermediate marsh
    12: 'Swamp',          # Estuarine Forested — exclude
    13: 'Brackish',       # Estuarine Scrub/Shrub
    14: 'Saline',         # Estuarine Emergent — brackish/saline marsh
    21: 'Water',          # Open Water — not wetland
}

# More detailed mapping if available
CCAP_TO_HABITAT_DETAIL = {
    11: 'Fresh',          # Palustrine Emergent = fresh marsh
    14: 'Saline',         # Estuarine Emergent = saline/brackish
    # Note: C-CAP doesn't distinguish fresh/intermediate/brackish/saline
    # within these classes — salinity gradient requires hydro data
}

def process_ccap(
        ccap_path='data/ccap_louisiana.tif',
        output_path='data/ccap_marsh_type.csv'):
    """Process C-CAP using windowed reading to avoid memory issues"""
    print("=" * 60)
    print("C-CAP MARSH TYPE PROCESSING")
    print("=" * 60)

    if not os.path.exists(ccap_path):
        print(f"File not found: {ccap_path}")
        return None

    try:
        import rasterio
        from rasterio.windows import from_bounds
        import pyproj
        from shapely.geometry import box
        from shapely.ops import transform

        # Louisiana Deltaic Plain bbox in lat/lon
        louisiana_bbox = box(-93.5, 28.5, -88.5, 30.5)

        with rasterio.open(ccap_path) as src:
            print(f"CRS: {src.crs}")
            print(f"Full shape: {src.shape}")

            # Reproject bbox to raster CRS (EPSG:5070)
            project = pyproj.Transformer.from_crs(
                'EPSG:4326', str(src.crs), always_xy=True
            ).transform
            bbox_proj = transform(project, louisiana_bbox)
            minx, miny, maxx, maxy = bbox_proj.bounds

            # Get window for Louisiana only
            window = from_bounds(
                minx, miny, maxx, maxy,
                src.transform
            )
            print(f"Reading Louisiana window only...")

            # Read just the Louisiana portion
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)
            nodata = src.nodata

        print(f"Louisiana shape: {data.shape}")
        print(f"Memory used: {data.nbytes / 1e6:.1f} MB")

        # Replace nodata
        if nodata is not None:
            data = data.astype(float)
            data[data == nodata] = np.nan

        # Extract wetland pixels only
        valid_classes = list(CCAP_TO_MARSH.keys())
        rows, cols = np.where(np.isin(data, valid_classes))

        if len(rows) == 0:
            print("No wetland pixels found in Louisiana window")
            return None

        # Get coordinates
        import rasterio.transform as rt
        lons, lats = rt.xy(win_transform, rows, cols)
        classes = data[rows, cols].astype(int)

        df = pd.DataFrame({
            'lat': lats,
            'lon': lons,
            'ccap_class': classes,
            'marsh_type_ccap': [CCAP_TO_MARSH.get(c, 'Unknown')
                                 for c in classes]
        })

        # Remove swamp and water
        df = df[~df['marsh_type_ccap'].isin(['Swamp', 'Water'])]
        print(f"\nValid marsh pixels: {len(df):,}")
        print(df['marsh_type_ccap'].value_counts())

        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return df

    except ImportError as e:
        print(f"Missing package: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def assign_ccap_to_pixels(
        pixel_df,
        ccap_path='data/ccap_marsh_type.csv',
        radius_deg=0.001):
    """
    Assign C-CAP marsh type to each prediction pixel by
    finding nearest C-CAP classified pixel within radius_deg degrees.
    Replaces NDVI-threshold marsh type assignment in script 07.
    """
    if not os.path.exists(ccap_path):
        print("C-CAP marsh type file not found — run process_ccap() first")
        return pixel_df

    print(f"\nAssigning C-CAP marsh type to {len(pixel_df):,} pixels...")
    ccap = pd.read_csv(ccap_path)

    # For each pixel find nearest C-CAP point
    # Use vectorized approach for speed
    from scipy.spatial import cKDTree

    ccap_coords = ccap[['lat', 'lon']].values
    pixel_coords = pixel_df[['lat', 'lon']].values

    tree = cKDTree(ccap_coords)
    distances, indices = tree.query(pixel_coords, k=1)

    # Assign marsh type where within radius
    mask = distances <= radius_deg
    pixel_df['marsh_type_ccap'] = np.nan
    pixel_df.loc[mask, 'marsh_type_ccap'] = ccap['marsh_type_ccap'].values[
        indices[mask]
    ]

    # Use C-CAP where available, fall back to NDVI threshold
    if 'marsh_type' in pixel_df.columns:
        pixel_df['marsh_type'] = np.where(
            pixel_df['marsh_type_ccap'].notna(),
            pixel_df['marsh_type_ccap'],
            pixel_df['marsh_type']
        )
    else:
        pixel_df['marsh_type'] = pixel_df['marsh_type_ccap']

    ccap_assigned = pixel_df['marsh_type_ccap'].notna().sum()
    print(f"  C-CAP assigned: {ccap_assigned:,} pixels "
          f"({ccap_assigned/len(pixel_df)*100:.0f}%)")
    print(f"  Marsh type distribution after C-CAP:")
    print(pixel_df['marsh_type'].value_counts())

    return pixel_df


def merge_ccap_with_fused(
        fused_path='data/fused_dataset_with_hydro.csv',
        ccap_path='data/ccap_marsh_type.csv',
        output_path='data/fused_dataset_with_ccap.csv'):
    """
    Add C-CAP marsh type to fused dataset at station locations.
    Provides additional validation of CRMS marsh type assignments.
    """
    if not os.path.exists(ccap_path):
        print("C-CAP file not found")
        return None

    from scipy.spatial import cKDTree

    fused = pd.read_csv(fused_path)
    ccap = pd.read_csv(ccap_path)

    print(f"Matching {len(fused):,} station records to C-CAP...")

    ccap_coords = ccap[['lat', 'lon']].values
    station_coords = fused[['lat', 'lon']].dropna().values

    tree = cKDTree(ccap_coords)
    distances, indices = tree.query(station_coords, k=1)

    fused_valid = fused.dropna(subset=['lat', 'lon'])
    fused_valid['marsh_type_ccap'] = ccap['marsh_type_ccap'].values[indices]
    fused_valid['ccap_distance_deg'] = distances

    # Agreement check
    if 'marsh_type' in fused_valid.columns:
        # Simplify both to Fresh/Intermediate/Brackish/Saline for comparison
        fused_valid['marsh_match'] = (
            fused_valid['marsh_type'] == fused_valid['marsh_type_ccap']
        )
        agreement = fused_valid['marsh_match'].mean()
        print(f"C-CAP vs CRMS marsh type agreement: {agreement:.1%}")

    fused_valid.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return fused_valid


if __name__ == '__main__':
    # Step 1: Process C-CAP raster
    ccap_df = process_ccap()

    if ccap_df is not None:
        # Step 2: Merge with fused dataset
        merge_ccap_with_fused()
        print("\nC-CAP processing complete")
        print("Next: re-run 07_spatial_prediction.py")
        print("      It will automatically use C-CAP marsh types")
    else:
        print("\nDownload C-CAP data first then re-run")