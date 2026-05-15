"""
01_gee_sentinel2_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Extracts Sentinel-2 spectral features AND auxiliary spatial
    variables at all CRMS station locations, and exports a
    wall-to-wall spatial grid for the full Deltaic Plain.

    Auxiliary variables added for improved wall-to-wall prediction:
    - SRTM elevation (USGS/SRTMGL1_003) — 30m, global

OUTPUTS (Google Drive folder: CWL_RFP):
    crms_sentinel2_features.csv       — station-level spectral + aux features
    deltaic_plain_spectral_YYYY_MM.tif — wall-to-wall spectral grid
    deltaic_plain_elevation.tif        — SRTM elevation raster
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import ee
import geemap
import pandas as pd
import os
import glob

# ─────────────────────────────────────────────
# INITIALIZE GEE
# ─────────────────────────────────────────────
ee.Initialize(project='cwlrfp')

# ─────────────────────────────────────────────
# 1. STUDY AREA
# ─────────────────────────────────────────────
study_area = ee.Geometry.Rectangle([-93.5, 28.5, -88.5, 30.5])

# ─────────────────────────────────────────────
# 2. LOAD CRMS STATION COORDINATES
# ─────────────────────────────────────────────
def load_crms_coordinates():
    coords_path = 'data/crms_all_stations_coords.csv'
    if not os.path.exists(coords_path):
        coords_path = 'data/crms_stations_coords.csv'
        print(f"Using fallback coords: {coords_path}")
    df = pd.read_csv(coords_path)
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    df = df.drop(columns=unnamed)
    df = df.dropna(subset=['Longitude', 'Latitude'])
    print(f"Loaded {len(df)} station coordinates")
    return df

coords_df = load_crms_coordinates()

crms_points = ee.FeatureCollection([
    ee.Feature(
        ee.Geometry.Point([float(row['Longitude']), float(row['Latitude'])]),
        {'station_id': str(row['Site_ID'])}
    )
    for _, row in coords_df.iterrows()
])
print(f"GEE FeatureCollection: {len(coords_df)} stations")

# ─────────────────────────────────────────────
# 3. CLOUD MASKING (Sentinel-2 Level-2A)
# ─────────────────────────────────────────────
def mask_s2_clouds(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000)

# ─────────────────────────────────────────────
# 4. SPECTRAL INDICES
# ─────────────────────────────────────────────
def add_ndvi(image):
    return image.addBands(
        image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

def add_ndwi(image):
    return image.addBands(
        image.normalizedDifference(['B3', 'B8']).rename('NDWI'))

def add_evi(image):
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('B8'),
         'RED': image.select('B4'),
         'BLUE': image.select('B2')}
    ).rename('EVI')
    return image.addBands(evi)

def add_all_indices(image):
    return add_evi(add_ndwi(add_ndvi(image)))

# ─────────────────────────────────────────────
# 5. AUXILIARY SPATIAL LAYERS
# ─────────────────────────────────────────────

# SRTM Elevation — 30m global DEM
srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(study_area)
print("SRTM elevation layer loaded")

# ─────────────────────────────────────────────
# 6. MONTHLY COMPOSITE FUNCTION
# ─────────────────────────────────────────────
def get_monthly_composite(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    composite = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(study_area)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
        .map(add_all_indices)
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                 'B8', 'B8A', 'B11', 'B12',
                 'NDVI', 'NDWI', 'EVI'])
        .median()
        .clip(study_area)
    )
    return composite.set({
        'year': year,
        'month': month,
        'system:time_start': start.millis()
    })

# ─────────────────────────────────────────────
# 7. BUILD COMPOSITE TIME SERIES
# ─────────────────────────────────────────────
years = list(range(2017, 2025))
months = list(range(1, 13))

print(f"Building {len(years) * len(months)} monthly composites...")
composites = [
    get_monthly_composite(year, month)
    for year in years
    for month in months
]
composite_collection = ee.ImageCollection(composites)
print(f"Composite collection ready: {len(composites)} images")

# ─────────────────────────────────────────────
# 8. EXTRACT STATION FEATURES
# Sentinel-2 + elevation + dist_water + TSS at each station
# ─────────────────────────────────────────────
def extract_station_features(image):
    date = ee.Date(image.get('system:time_start'))

    samples = image.sampleRegions(
        collection=crms_points,
        scale=30,
        geometries=True
    )
    return samples.map(lambda f: f.set({
        'year': date.get('year'),
        'month': date.get('month')
    }))

print("Mapping spectral + auxiliary extraction...")
all_samples = composite_collection.map(extract_station_features).flatten()

# ─────────────────────────────────────────────
# 9. EXPORT STATION FEATURES
# ─────────────────────────────────────────────
export_stations = ee.batch.Export.table.toDrive(
    collection=all_samples,
    description='CRMS_Sentinel2_Features_v2',
    folder='CWL_RFP',
    fileNamePrefix='crms_sentinel2_features_v2',
    fileFormat='CSV'
)
export_stations.start()
print("Station export started: crms_sentinel2_features_v2.csv")

# ── Export 2: Auxiliary variables at station locations (one-time) ──
aux_image = (srtm.rename('elevation'))

export_aux = ee.batch.Export.table.toDrive(
    collection=aux_image.sampleRegions(
        collection=crms_points,
        scale=30,
        geometries=True
    ),
    description='CRMS_Auxiliary_Features',
    folder='CWL_RFP',
    fileNamePrefix='crms_auxiliary_features',
    fileFormat='CSV'
)
export_aux.start()
print("Auxiliary features export started: crms_auxiliary_features.csv")

# ─────────────────────────────────────────────
# 10. EXPORT WALL-TO-WALL SPECTRAL GRID
# Peak season composite with all auxiliary layers
# ─────────────────────────────────────────────
peak_composite = get_monthly_composite(2023, 7)

# Combine spectral + auxiliary into single export
wall_to_wall = (peak_composite
    .select(['NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12'])
    .addBands(srtm.rename('elevation'))
    .toFloat())

export_grid = ee.batch.Export.image.toDrive(
    image=wall_to_wall,
    description='Deltaic_Plain_Spectral_Grid_v2',
    folder='CWL_RFP',
    fileNamePrefix='deltaic_plain_spectral_v2_2023_07',
    region=study_area.getInfo()['coordinates'],
    scale=100,
    maxPixels=1e13,
    fileFormat='GeoTIFF',
    crs='EPSG:4326'
)
export_grid.start()
print("Wall-to-wall grid export started: deltaic_plain_spectral_v2_2023_07.tif")

# ─────────────────────────────────────────────
# 11. EXPORT ELEVATION SEPARATELY AT 30m
# Higher resolution for better spatial detail
# ─────────────────────────────────────────────
export_elevation = ee.batch.Export.image.toDrive(
    image=srtm.toFloat(),
    description='Deltaic_Plain_Elevation_30m',
    folder='CWL_RFP',
    fileNamePrefix='deltaic_plain_elevation_30m',
    region=study_area.getInfo()['coordinates'],
    scale=30,
    maxPixels=1e13,
    fileFormat='GeoTIFF',
    crs='EPSG:4326'
)
export_elevation.start()
print("Elevation export started: deltaic_plain_elevation_30m.tif")

print("\nAll exports started!")
print("Monitor at: https://code.earthengine.google.com/tasks")
print("Files will appear in Google Drive folder: CWL_RFP")
print("\nNew features added to exports:")
print("  elevation        — SRTM 30m elevation (NAVD88 approx)")