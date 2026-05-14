"""
merge_elevation_aux.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Extracts elevation and auxiliary variables at each CRMS
    station location and merges into fused dataset.

    Elevation is a critical missing feature for saline marsh
    carbon prediction — controls inundation duration.

INPUTS:
    data/deltaic_plain_elevation_30m.tif  — SRTM elevation raster
    data/crms_all_stations_coords.csv     — station coordinates
    data/fused_dataset_with_hydro.csv     — main fused dataset

OUTPUTS:
    data/crms_elevation.csv               — elevation per station
    data/fused_dataset_with_elevation.csv — fused + elevation

RUN:
    python merge_elevation_aux.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def extract_elevation_at_stations(
        elev_path='data/deltaic_plain_elevation_30m.tif',
        coords_path='data/crms_all_stations_coords.csv',
        output_path='data/crms_elevation.csv'):
    """Extract elevation values at each CRMS station"""
    print("=" * 60)
    print("ELEVATION EXTRACTION AT CRMS STATIONS")
    print("=" * 60)

    if not os.path.exists(elev_path):
        print(f"Elevation raster not found: {elev_path}")
        return None

    try:
        import rasterio
        from rasterio.sample import sample_gen

        # Load station coordinates
        coords = pd.read_csv(coords_path)
        unnamed = [c for c in coords.columns if 'Unnamed' in str(c)]
        coords = coords.drop(columns=unnamed)
        coords = coords.dropna(subset=['Latitude', 'Longitude'])
        print(f"Stations: {len(coords)}")

        # Extract elevation at each station
        with rasterio.open(elev_path) as src:
            print(f"Raster CRS: {src.crs}")
            print(f"Raster shape: {src.shape}")

            # Sample elevation at station coordinates
            xy = list(zip(coords['Longitude'], coords['Latitude']))
            elevations = list(sample_gen(src, xy))
            elev_values = [e[0] for e in elevations]

        coords['elevation_m'] = elev_values
        # Replace nodata with NaN
        coords['elevation_m'] = pd.to_numeric(
            coords['elevation_m'], errors='coerce'
        )
        # Flag unrealistic values (elevation > 10m is above coastal marsh)
        coords.loc[
            coords['elevation_m'] > 10, 'elevation_m'
        ] = np.nan
        coords.loc[
            coords['elevation_m'] < -10, 'elevation_m'
        ] = np.nan

        valid = coords['elevation_m'].notna().sum()
        print(f"\nStations with valid elevation: {valid}/{len(coords)}")
        print(f"Elevation range: "
              f"{coords['elevation_m'].min():.2f} to "
              f"{coords['elevation_m'].max():.2f} m")
        print(f"Mean elevation: {coords['elevation_m'].mean():.3f} m")

        # Rename for merge
        result = coords.rename(columns={'Site_ID': 'station_id'})
        result = result[['station_id', 'elevation_m']]

        result.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return result

    except ImportError:
        print("rasterio not installed")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_elevation_with_fused(
        elev_path='data/crms_elevation.csv',
        fused_path='data/fused_dataset_with_hydro.csv',
        output_path='data/fused_dataset_with_elevation.csv'):
    """Merge elevation into fused dataset"""
    print("\n--- Merging Elevation with Fused Dataset ---")

    if not os.path.exists(elev_path):
        print(f"Elevation file not found: {elev_path}")
        return None

    elev = pd.read_csv(elev_path)
    fused = pd.read_csv(fused_path)

    print(f"Fused dataset: {fused.shape}")
    print(f"Elevation data: {len(elev)} stations")

    merged = fused.merge(elev, on='station_id', how='left')
    matched = merged['elevation_m'].notna().sum()
    print(f"Records with elevation: {matched:,}/{len(merged):,} "
          f"({matched/len(merged)*100:.0f}%)")

    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return merged


def merge_percent_flooded(
        flooded_path='data/crms_percent_flooded.csv',
        fused_path='data/fused_dataset_with_elevation.csv',
        output_path='data/fused_dataset_final.csv'):
    """
    Merge percent flooded data into fused dataset.
    Run after crms_percent_flooded.csv is downloaded.
    Column names will be updated once file is available.
    """
    print("\n--- Merging Percent Flooded ---")

    if not os.path.exists(flooded_path):
        print(f"Percent flooded not yet available: {flooded_path}")
        print("Download from CRMS and re-run")
        # Just copy fused to final without percent flooded
        fused = pd.read_csv(fused_path)
        fused.to_csv(output_path, index=False)
        print(f"Saved without percent flooded: {output_path}")
        return fused

    flooded = pd.read_csv(flooded_path)
    print(f"Percent flooded columns: {list(flooded.columns)}")
    print(f"Sample:\n{flooded.head(2)}")

    # Column detection — will auto-detect station ID and flooded column
    id_col = next((c for c in flooded.columns
                   if 'site' in c.lower() or 'station' in c.lower()), None)
    flooded_col = next((c for c in flooded.columns
                        if 'flood' in c.lower() or 'percent' in c.lower()
                        or 'inundat' in c.lower()), None)
    year_col = next((c for c in flooded.columns
                     if 'year' in c.lower() or 'date' in c.lower()), None)

    if not id_col or not flooded_col:
        print("Could not detect columns — paste column names for manual fix")
        return None

    print(f"Using: id={id_col}, flooded={flooded_col}, year={year_col}")

    # Strip suffix from station ID
    flooded['station_id'] = flooded[id_col].astype(str).str.split(
        '-'
    ).str[0].str.strip()
    flooded[flooded_col] = pd.to_numeric(flooded[flooded_col], errors='coerce')

    # Aggregate to station level (median across years)
    if year_col:
        flooded['year'] = pd.to_numeric(flooded[year_col], errors='coerce')
        station_flooded = flooded.groupby(['station_id', 'year'])[
            flooded_col
        ].median().reset_index()
        station_flooded = station_flooded.rename(
            columns={flooded_col: 'percent_flooded'}
        )
    else:
        station_flooded = flooded.groupby('station_id')[
            flooded_col
        ].median().reset_index()
        station_flooded = station_flooded.rename(
            columns={flooded_col: 'percent_flooded'}
        )

    fused = pd.read_csv(fused_path)

    if year_col and 'year' in fused.columns:
        merged = fused.merge(
            station_flooded, on=['station_id', 'year'], how='left'
        )
    else:
        merged = fused.merge(station_flooded, on='station_id', how='left')

    matched = merged['percent_flooded'].notna().sum()
    print(f"Records with percent flooded: {matched:,}/{len(merged):,} "
          f"({matched/len(merged)*100:.0f}%)")
    print(f"Mean percent flooded: {merged['percent_flooded'].mean():.1f}%")

    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return merged


if __name__ == '__main__':
    # Step 1: Extract elevation at stations
    elev = extract_elevation_at_stations()

    # Step 2: Merge with fused dataset
    if elev is not None:
        merge_elevation_with_fused()

    # Step 3: Merge percent flooded (runs now if available, skips if not)
    merge_percent_flooded()

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print("Output: data/fused_dataset_final.csv")
    print("\nNext steps:")
    print("1. Update filepath in 03, 04, 05 to use fused_dataset_final.csv")
    print("2. Add 'elevation_m' and 'percent_flooded' to FEATURE_SETS in 03")
    print("3. Add 'elevation_m' and 'percent_flooded' to CLASSIFICATION_FEATURES in 05")
    print("4. Re-run 03 (focus on saline improvement)")
    print("5. Re-run 04, 05, 07")
    print("=" * 60)