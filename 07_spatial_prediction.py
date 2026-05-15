"""
07_spatial_prediction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Wall-to-wall carbon stock and loss probability prediction across
    the entire Mississippi River Deltaic Plain. Reads a Sentinel-2
    spatial raster, converts it to a pixel dataframe, trains
    spectral-only GPR and C4.5 models from the fused station dataset,
    and applies them to every pixel.

    Spectral-only models are trained here fresh from the fused dataset —
    the saved models from 03/05 are NOT loaded. This is intentional:
    full-feature station models use hydro + soil + spectral; wall-to-wall
    models are constrained to satellite-available features only.

    Loss probability is clipped to [0.02, 0.97] to avoid the exact 0/1
    artifacts that uncalibrated decision trees produce on pure leaf nodes.

INPUTS:
    data/deltaic_plain_spectral_v2_2023_07.tif  — Sentinel-2 + aux bands (preferred)
    data/deltaic_plain_spectral_2023_07.tif     — Sentinel-2 only (fallback)
    data/ccap_marsh_type.csv                    — C-CAP marsh classification
                                                  (NDVI thresholds used if absent)
    data/fused_dataset_final.csv                — station training data
    data/crms_all_stations_coords.csv           — for marsh type spatial assignment

OUTPUTS:
    results/wall_to_wall_predictions.csv    — full pixel predictions
    results/high_risk_parcels.geojson       — high-risk pixels (loss_prob > 0.6)

    NOTE: Raster outputs (wall_to_wall_carbon.tif, wall_to_wall_loss_prob.tif)
    are currently disabled for speed. Re-enable save_raster_predictions() call
    in run_spatial_prediction() if GeoTIFFs are needed.

    After this script, run in order:
        python sample_wall.py     → stratified sample for display
        python display_wall.py    → filter to dashboard-sized display set
        python get_high_risk.py   → high-risk unmonitored parcels
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────
# SPATIAL FEATURE SET
# Features available everywhere via satellite
# Updated when new GEE export includes aux variables
# ─────────────────────────────────────────────
SPATIAL_FEATURES_V2 = [
    'NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12',
    'elevation', 'dist_to_water_proxy', 'tss_proxy'
]
SPATIAL_FEATURES_V1 = ['NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12']

# ─────────────────────────────────────────────
# RASTER READING
# ─────────────────────────────────────────────
def read_geotiff(filepath):
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
            profile = src.profile
            print(f"Raster: {data.shape} | CRS: {crs} | "
                  f"Resolution: {src.res[0]:.0f}m")
        return data, transform, crs, profile
    except ImportError:
        print("rasterio not installed")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading raster: {e}")
        return None, None, None, None

# ─────────────────────────────────────────────
# RASTER TO DATAFRAME
# ─────────────────────────────────────────────
def raster_to_dataframe(data, transform, band_names):
    try:
        import rasterio
        from rasterio.transform import xy
    except ImportError:
        print("rasterio required")
        return None

    n_bands, height, width = data.shape
    print(f"Converting raster to dataframe: "
          f"{height}x{width} = {height*width:,} pixels")

    rows_list = []
    for r in range(0, height, 1):
        for c in range(width):
            lon, lat = xy(transform, r, c)
            pixel_vals = [float(data[b, r, c]) for b in range(n_bands)]
            if all(v < -9000 or np.isnan(v) for v in pixel_vals):
                continue
            rows_list.append([lat, lon] + pixel_vals)

        if r % 500 == 0 and r > 0:
            pct = r / height * 100
            print(f"  Processing: {pct:.0f}% ({r*width:,}/{height*width:,} pixels)")

    cols = ['lat', 'lon'] + band_names
    df = pd.DataFrame(rows_list, columns=cols)
    print(f"Valid pixels: {len(df):,}")
    return df

# ─────────────────────────────────────────────
# ASSIGN MARSH TYPE
# Uses C-CAP if available, else NDVI thresholds
# ─────────────────────────────────────────────
def assign_marsh_type(pixel_df):
    ccap_path = 'data/ccap_marsh_type.csv'

    if os.path.exists(ccap_path):
        print(f"Using C-CAP marsh classification...")
        try:
            from scipy.spatial import cKDTree
            ccap = pd.read_csv(ccap_path)
            ccap_coords = ccap[['lat', 'lon']].values
            pixel_coords = pixel_df[['lat', 'lon']].values
            tree = cKDTree(ccap_coords)
            distances, indices = tree.query(pixel_coords, k=1)
            mask = distances <= 0.005  # ~500m threshold
            pixel_df['marsh_type'] = 'Unknown'
            pixel_df.loc[mask, 'marsh_type'] = (
                ccap['marsh_type_ccap'].values[indices[mask]]
            )
            # Fall back to NDVI for unmatched pixels
            ndvi_mask = pixel_df['marsh_type'] == 'Unknown'
            if ndvi_mask.sum() > 0 and 'NDVI' in pixel_df.columns:
                pixel_df.loc[ndvi_mask & (pixel_df['NDVI'] >= 0.55),
                             'marsh_type'] = 'Fresh'
                pixel_df.loc[ndvi_mask & (pixel_df['NDVI'] >= 0.35) &
                             (pixel_df['NDVI'] < 0.55),
                             'marsh_type'] = 'Intermediate'
                pixel_df.loc[ndvi_mask & (pixel_df['NDVI'] >= 0.20) &
                             (pixel_df['NDVI'] < 0.35),
                             'marsh_type'] = 'Brackish'
                pixel_df.loc[ndvi_mask & (pixel_df['NDVI'] < 0.20),
                             'marsh_type'] = 'Saline'
            ccap_pct = (~ndvi_mask).mean() * 100
            print(f"  C-CAP assigned: {ccap_pct:.0f}% of pixels")
            print(f"  NDVI fallback: {ndvi_mask.mean()*100:.0f}% of pixels")
        except Exception as e:
            print(f"C-CAP assignment failed: {e} — using NDVI thresholds")
            pixel_df = assign_marsh_ndvi(pixel_df)
    else:
        print("C-CAP not available — using NDVI thresholds")
        print("Download C-CAP and run download_ccap.py to improve accuracy")
        pixel_df = assign_marsh_ndvi(pixel_df)

    print(f"Marsh type distribution:")
    print(pixel_df['marsh_type'].value_counts())
    return pixel_df

def assign_marsh_ndvi(pixel_df):
    if 'NDVI' in pixel_df.columns:
        conditions = [
            pixel_df['NDVI'] >= 0.55,
            (pixel_df['NDVI'] >= 0.35) & (pixel_df['NDVI'] < 0.55),
            (pixel_df['NDVI'] >= 0.20) & (pixel_df['NDVI'] < 0.35),
            pixel_df['NDVI'] < 0.20
        ]
        choices = ['Fresh', 'Intermediate', 'Brackish', 'Saline']
        pixel_df['marsh_type'] = np.select(
            conditions, choices, default='Brackish'
        )
    else:
        pixel_df['marsh_type'] = 'Brackish'
    return pixel_df

# ─────────────────────────────────────────────
# TRAIN SPATIAL GPR
# Spectral-only model for wall-to-wall prediction
# ─────────────────────────────────────────────
def train_spatial_gpr(
        pixel_df,
        fused_path='data/fused_dataset_final.csv'):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, WhiteKernel, ConstantKernel
    )
    from sklearn.preprocessing import StandardScaler

    print("\n--- Training Spatial GPR (satellite features only) ---")

    # Use v2 features if available in pixel_df, else v1
    available_spatial = [f for f in SPATIAL_FEATURES_V2
                         if f in pixel_df.columns]
    if len(available_spatial) < 6:
        available_spatial = [f for f in SPATIAL_FEATURES_V1
                             if f in pixel_df.columns]
    print(f"Spatial features: {available_spatial}")

    TARGET = 'carbon_stock'
    TARGET_FALLBACK = 'accretion_median'

    fused = pd.read_csv(fused_path)
    if 'year' in fused.columns:
        fused = fused.sort_values('year', ascending=False)
    fused = fused.drop_duplicates('station_id', keep='first')

    habitat_map = {
        'freshwater_intermediate': ['Fresh', 'Freshwater', 'Intermediate'],
        'brackish': ['Brackish'],
        'saline': ['Saline']
    }

    spatial_models = {}

    for habitat_name, marsh_types in habitat_map.items():
        df = fused[fused['marsh_type'].isin(marsh_types)].copy()
        available = [f for f in available_spatial if f in df.columns]

        target_col = TARGET if TARGET in df.columns else TARGET_FALLBACK
        df_clean = df.dropna(subset=available + [target_col])

        if len(df_clean) < 5:
            print(f"  {habitat_name}: too few stations — skipping")
            continue

        X = df_clean[available].values
        y = df_clean[target_col].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            alpha=1e-6
        )
        gpr.fit(X_scaled, y)
        spatial_models[habitat_name] = (gpr, scaler, available)
        print(f"  {habitat_name}: {len(df_clean)} stations | "
              f"{len(available)} features | target={target_col}")

    return spatial_models

# ─────────────────────────────────────────────
# TRAIN SPATIAL C4.5
# Spectral-only classifier for wall-to-wall prediction
# ─────────────────────────────────────────────
def train_spatial_c45(
        pixel_df,
        fused_path='data/fused_dataset_final.csv'):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n--- Training Spatial C4.5 (satellite features only) ---")

    available_spatial = [f for f in SPATIAL_FEATURES_V2
                         if f in pixel_df.columns]
    if len(available_spatial) < 6:
        available_spatial = [f for f in SPATIAL_FEATURES_V1
                             if f in pixel_df.columns]
    print(f"Spatial features: {available_spatial}")

    TARGET = 'loss_severity'  # 3-class: LOW / MODERATE / HIGH

    fused = pd.read_csv(fused_path)
    if 'year' in fused.columns:
        fused = fused.sort_values('year', ascending=False)
    fused = fused.drop_duplicates('station_id', keep='first')

    # Fall back to ever_lost_land if fix scripts haven't been run yet
    if TARGET not in fused.columns:
        print(f"  '{TARGET}' not found — falling back to 'ever_lost_land'")
        TARGET = 'ever_lost_land'

    available = [f for f in available_spatial if f in fused.columns]
    fused_clean = fused.dropna(subset=available + [TARGET])

    X = fused_clean[available].values
    y = fused_clean[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_scaled, y)
    print(f"  Spatial C4.5: {len(fused_clean)} stations | "
          f"{len(available)} features")
    return clf, scaler, available

# ─────────────────────────────────────────────
# WALL-TO-WALL CARBON PREDICTION
# ─────────────────────────────────────────────
def predict_carbon_wall_to_wall(pixel_df, gpr_models):
    from sklearn.preprocessing import StandardScaler

    print("\n--- Wall-to-Wall Carbon Stock Prediction ---")
    pixel_df['carbon_predicted'] = np.nan
    pixel_df['carbon_uncertainty'] = np.nan
    pixel_df['high_uncertainty'] = False

    habitat_map = {
        'freshwater_intermediate': ['Fresh', 'Intermediate', 'Freshwater'],
        'brackish': ['Brackish'],
        'saline': ['Saline']
    }

    for habitat_name, marsh_types in habitat_map.items():
        if habitat_name not in gpr_models:
            continue

        gpr, scaler, features = gpr_models[habitat_name]
        mask = pixel_df['marsh_type'].isin(marsh_types)
        subset = pixel_df[mask].copy()

        if len(subset) == 0:
            continue

        available = [f for f in features if f in subset.columns]
        if len(available) < 2:
            print(f"  {habitat_name}: insufficient features")
            continue

        subset_clean = subset.dropna(subset=available)
        if len(subset_clean) == 0:
            continue

        X = subset_clean[available].values
        X_scaled = scaler.transform(X)

        chunk_size = 10000
        preds, stds = [], []
        for i in range(0, len(X_scaled), chunk_size):
            chunk = X_scaled[i:i+chunk_size]
            p, s = gpr.predict(chunk, return_std=True)
            preds.extend(p)
            stds.extend(s)

        preds = np.array(preds)
        stds = np.array(stds)

        pixel_df.loc[subset_clean.index, 'carbon_predicted'] = preds
        pixel_df.loc[subset_clean.index, 'carbon_uncertainty'] = stds

        high_unc = stds > (stds.mean() + stds.std())
        pixel_df.loc[subset_clean.index, 'high_uncertainty'] = high_unc

        print(f"  {habitat_name}: {len(subset_clean):,} pixels | "
              f"{high_unc.sum():,} high uncertainty")

    valid = pixel_df['carbon_predicted'].notna().sum()
    print(f"  Total pixels with carbon estimate: {valid:,}")
    return pixel_df

# ─────────────────────────────────────────────
# WALL-TO-WALL LOSS PREDICTION
# ─────────────────────────────────────────────
def predict_loss_wall_to_wall(pixel_df, clf, scaler, features):
    print("\n--- Wall-to-Wall Loss Probability Prediction ---")

    available = [f for f in features if f in pixel_df.columns]
    if len(available) < 2:
        print("  Not enough features — skipping")
        return pixel_df

    pixel_df_clean = pixel_df.dropna(subset=available)
    print(f"  Scoring {len(pixel_df_clean):,} pixels...")

    X = pixel_df_clean[available].values
    X_scaled = scaler.transform(X)

    # Weighted risk score: 0*P(LOW) + 0.5*P(MODERATE) + 1.0*P(HIGH)
    # Uses clf.classes_ ordering so it is robust to sklearn label sorting
    class_weights = {'LOW': 0.0, 'MODERATE': 0.5, 'HIGH': 1.0}
    weight_array = np.array([
        class_weights.get(c, 0.5) for c in clf.classes_
    ])

    chunk_size = 10000
    probs = []
    for i in range(0, len(X_scaled), chunk_size):
        chunk = X_scaled[i:i+chunk_size]
        proba = clf.predict_proba(chunk)
        risk_score = proba @ weight_array
        probs.extend(risk_score)

    # Clip to avoid exact endpoints — pure decision-tree leaves produce this artifact
    probs = np.clip(probs, 0.02, 0.97)

    pixel_df.loc[pixel_df_clean.index, 'loss_probability'] = probs
    pixel_df['risk_level'] = pd.cut(
        pixel_df['loss_probability'],
        bins=[0, 0.4, 0.75, 1.0],
        labels=['LOW', 'MODERATE', 'HIGH']
    )

    high_risk = (pixel_df['loss_probability'] > 0.6).sum()
    moderate = ((pixel_df['loss_probability'] > 0.3) &
                (pixel_df['loss_probability'] <= 0.6)).sum()
    print(f"  High risk (>60%): {high_risk:,}")
    print(f"  Moderate (30-60%): {moderate:,}")
    return pixel_df

# ─────────────────────────────────────────────
# EXPORT HIGH-RISK GEOJSON
# ─────────────────────────────────────────────
def export_high_risk_geojson(
        pixel_df,
        output_path='results/high_risk_parcels.geojson',
        threshold=0.6):
    high_risk = pixel_df[pixel_df['loss_probability'] > threshold].copy()
    print(f"\nExporting {len(high_risk):,} high-risk parcels...")

    # Sample to reduce file size if very large
    if len(high_risk) > 100000:
        high_risk = high_risk.sample(100000, random_state=42)
        print(f"  Sampled to 100,000 for GeoJSON file size")

    features = []
    for _, row in high_risk.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['lon']), float(row['lat'])]
            },
            "properties": {
                "loss_probability": float(row.get('loss_probability', 0)),
                "carbon_predicted": float(row['carbon_predicted'])
                    if pd.notna(row.get('carbon_predicted')) else None,
                "carbon_uncertainty": float(row['carbon_uncertainty'])
                    if pd.notna(row.get('carbon_uncertainty')) else None,
                "marsh_type": str(row.get('marsh_type', 'Unknown')),
                "risk_level": str(row.get('risk_level', 'HIGH')),
                "high_uncertainty": bool(row.get('high_uncertainty', False))
            }
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    print(f"GeoJSON saved: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# SAVE RASTER OUTPUTS
# ─────────────────────────────────────────────
def save_raster_predictions(pixel_df, transform, profile, height, width):
    try:
        import rasterio
        from rasterio.transform import rowcol

        for col_name, output_name in [
            ('carbon_predicted', 'results/wall_to_wall_carbon.tif'),
            ('carbon_uncertainty', 'results/wall_to_wall_uncertainty.tif'),
            ('loss_probability', 'results/wall_to_wall_loss_prob.tif')
        ]:
            if col_name not in pixel_df.columns:
                continue

            valid = pixel_df.dropna(subset=[col_name]).copy()
            if len(valid) == 0:
                continue

            print(f"  Writing {col_name} ({len(valid):,} pixels)...")
            raster = np.full((height, width), -9999, dtype='float32')

            # Vectorized — all pixels at once instead of one by one
            rows, cols = rowcol(
                transform,
                valid['lon'].values,
                valid['lat'].values
            )
            rows = np.array(rows)
            cols = np.array(cols)
            mask = ((rows >= 0) & (rows < height) &
                    (cols >= 0) & (cols < width))
            raster[rows[mask], cols[mask]] = valid[col_name].values[mask]

            profile.update(count=1, dtype='float32', nodata=-9999)
            with rasterio.open(output_name, 'w', **profile) as dst:
                dst.write(raster, 1)
            print(f"  Saved: {output_name}")

    except Exception as e:
        print(f"Raster save error: {e}")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_spatial_prediction():
    print("=" * 60)
    print("WALL-TO-WALL SPATIAL PREDICTION PIPELINE")
    print("Filling gaps between 266+ CRMS stations")
    print("=" * 60)

    # Try v2 raster first (with aux features), fall back to v1
    tif_path_v2 = 'data/deltaic_plain_spectral_v2_2023_07.tif'
    tif_path_v1 = 'data/deltaic_plain_spectral_2023_07.tif'

    if os.path.exists(tif_path_v2):
        tif_path = tif_path_v2
        band_names = ['NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12',
                      'elevation', 'dist_to_water_proxy', 'tss_proxy']
        print("Using v2 raster (spectral + auxiliary features)")
    elif os.path.exists(tif_path_v1):
        tif_path = tif_path_v1
        band_names = ['NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12']
        print("Using v1 raster (spectral only)")
        print("Run updated 01_gee_sentinel2_pipeline.py for v2")
    else:
        print(f"No raster found — run 01_gee_sentinel2_pipeline.py first")
        return

    # Read raster
    print("\nReading Sentinel-2 spatial grid...")
    data, transform, crs, profile = read_geotiff(tif_path)
    if data is None:
        return

    n_bands, height, width = data.shape
    if n_bands < len(band_names):
        band_names = band_names[:n_bands]

    # Convert to dataframe
    print("\nConverting raster to pixel dataframe...")
    pixel_df = raster_to_dataframe(data, transform, band_names)
    if pixel_df is None or len(pixel_df) == 0:
        print("Failed to convert raster")
        return

    # Assign marsh type
    coords_df = pd.read_csv('data/crms_all_stations_coords.csv') \
        if os.path.exists('data/crms_all_stations_coords.csv') \
        else pd.read_csv('data/crms_stations_coords.csv')
    print(f"Assigning marsh type from {len(coords_df)} stations "
          f"to {len(pixel_df):,} pixels...")
    pixel_df = assign_marsh_type(pixel_df)

    # Train spatial models
    print("\nTraining spatial models (satellite features only)...")
    gpr_models = train_spatial_gpr(pixel_df)
    clf, c45_scaler, c45_features = train_spatial_c45(pixel_df)

    # Carbon stock prediction
    pixel_df = predict_carbon_wall_to_wall(pixel_df, gpr_models)

    # Loss probability prediction
    pixel_df = predict_loss_wall_to_wall(
        pixel_df, clf, c45_scaler, c45_features
    )

    # Save outputs
    output_csv = 'results/wall_to_wall_predictions.csv'
    pixel_df.to_csv(output_csv, index=False)
    print(f"\nFull predictions saved: {output_csv}")

    if 'loss_probability' in pixel_df.columns:
        export_high_risk_geojson(pixel_df)


    #Raster saving skipped for speed for now-- CSV and GeoJSON is more important for dashboard.
    #print("\nSaving raster outputs...")
    #save_raster_predictions(pixel_df, transform, profile, height, width)

    # Summary
    print("\n" + "=" * 60)
    print("SPATIAL PREDICTION SUMMARY")
    print("=" * 60)
    if 'carbon_predicted' in pixel_df.columns:
        valid = pixel_df['carbon_predicted'].dropna()
        print(f"Carbon stock — Mean: {valid.mean():.4f} | "
              f"Range: {valid.min():.4f}–{valid.max():.4f} g C/cm³")
    if 'loss_probability' in pixel_df.columns:
        valid_loss = pixel_df['loss_probability'].dropna()
        high_pct = (valid_loss > 0.6).mean()
        print(f"High risk pixels: {high_pct:.1%} of coast")
    if 'high_uncertainty' in pixel_df.columns:
        high_unc_pct = pixel_df['high_uncertainty'].mean()
        print(f"High uncertainty parcels: {high_unc_pct:.1%} "
              f"(flagged for field re-measurement)")
    print("=" * 60)
    print("Dashboard ready: streamlit run 06_dashboard.py")

if __name__ == '__main__':
    run_spatial_prediction()