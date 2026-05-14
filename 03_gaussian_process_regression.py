"""
03_gaussian_process_regression.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Trains habitat-stratified Gaussian Process Regression (GPR) models
    to estimate carbon stock at every CRMS station location. Once trained,
    the saved models can score new data without full retraining.

INPUTS:
    data/fused_dataset.csv          — CRMS + Sentinel-2 merged features
    data/crms_freshwater_intermediate.csv
    data/crms_brackish.csv
    data/crms_saline.csv

OUTPUTS:
    results/gpr_results.csv         — Cross-validation R² and RMSE per habitat
    results/gpr_model_{habitat}.pkl — Saved trained model (for re-scoring)
    results/gpr_scaler_{habitat}.pkl— Saved feature scaler (for re-scoring)
    results/shap_{habitat}.csv      — Feature importance rankings
    figures/shap_{habitat}.png      — SHAP summary plots
    figures/gpr_predictions_{habitat}.png — Predicted vs observed plots

WHEN TO RE-RUN:
    Full retrain (this script): Annually, or after a major event like a storm
    Re-score only (rescore_new_data): When new CRMS/Sentinel-2 data arrives

FOLLOWS:
    Chenevert & Edmonds (2024) — habitat-stratified modeling,
    5-fold CV x 100 runs, backward elimination feature selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE SETS
# Initial set — refined by backward elimination per habitat
# Hydro features (tidal_amplitude, flood_depth, salinity) added
# after crms_hydro_averages.csv is merged
# Spectral features (NDVI, NDWI, EVI, B8, B11, B12) from GEE export
# ─────────────────────────────────────────────
FEATURE_SETS = {
      'freshwater_intermediate': [
        'NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12',
        'bulk_density', 'percent_organic',
        'tidal_amplitude_annual_mean',
        'flood_depth_annual_mean',
        'salinity_annual_mean',
        'elevation_m',
    ],
    'brackish': [
    'bulk_density', 'percent_organic',
    'tidal_amplitude_annual_mean', 'elevation_m',
    ],
    'saline': [
        'NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12',
        'bulk_density', 'percent_organic',
        'tidal_amplitude_annual_mean',
        'flood_depth_annual_mean',
        'salinity_annual_mean', 'elevation_m',
    ]
}

TARGET = 'carbon_stock'
TARGET_FALLBACK = 'accretion_median'

# ─────────────────────────────────────────────
# BACKWARD ELIMINATION FEATURE SELECTION
# p < 0.05, following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def backward_elimination(df, features, target, significance=0.05):
    """Remove features with p > 0.05 iteratively"""
    selected = [f for f in features if f in df.columns]
    print(f"  Starting backward elimination with: {selected}")
    while len(selected) > 1:
        X = sm.add_constant(df[selected])
        model = sm.OLS(df[target], X).fit()
        p_values = model.pvalues[1:]
        max_p = p_values.max()
        if max_p > significance:
            drop = p_values.idxmax()
            selected.remove(drop)
            print(f"  Removed: {drop} (p={max_p:.4f})")
        else:
            break
    print(f"  Selected features: {selected}")
    return selected

# ─────────────────────────────────────────────
# SPATIAL BLOCK ASSIGNMENT
# Prevents spatial autocorrelation from inflating CV scores
# ─────────────────────────────────────────────
def assign_spatial_blocks(df, n_blocks=5):
    df = df.copy()
    df['lat_block'] = pd.cut(df['lat'], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df['lon'], bins=n_blocks, labels=False)
    df['cv_block'] = (
        df['lat_block'].astype(float) * n_blocks +
        df['lon_block'].astype(float)
    ) % n_blocks
    return df

# ─────────────────────────────────────────────
# BUILD GPR MODEL
# RBF kernel + noise term following Chenevert & Edmonds (2024)
# ─────────────────────────────────────────────
def build_gpr():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=2,
        normalize_y=True,
        alpha=1e-6
    )

# ─────────────────────────────────────────────
# 5-FOLD SPATIALLY BLOCKED CV x 100 RUNS
# Following Chenevert & Edmonds (2024) exactly
# ─────────────────────────────────────────────
def cross_validate_gpr(df, features, target, habitat_name, n_runs=100):
    print(f"\n{'='*60}")
    print(f"GPR CROSS-VALIDATION: {habitat_name.upper()}")
    print(f"n={len(df)} stations | {n_runs} runs x 5-fold spatially blocked CV")
    print('='*60)

    df = assign_spatial_blocks(df)
    all_r2, all_rmse = [], []

    for run in range(n_runs):
        fold_r2, fold_rmse = [], []
        blocks = df['cv_block'].unique()
        np.random.shuffle(blocks)

        for fold_idx in range(min(5, len(blocks))):
            test_block = blocks[fold_idx % len(blocks)]
            train_mask = df['cv_block'] != test_block
            test_mask = df['cv_block'] == test_block

            X_train = df.loc[train_mask, features].values
            y_train = df.loc[train_mask, target].values
            X_test = df.loc[test_mask, features].values
            y_test = df.loc[test_mask, target].values

            if len(X_test) == 0 or len(X_train) < 5:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            gpr = build_gpr()
            gpr.fit(X_train_s, y_train)
            y_pred, _ = gpr.predict(X_test_s, return_std=True)

            if len(np.unique(y_test)) > 1:
                fold_r2.append(r2_score(y_test, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        if fold_r2:
            all_r2.append(np.mean(fold_r2))
            all_rmse.append(np.mean(fold_rmse))

        if (run + 1) % 20 == 0:
            print(f"  Run {run+1}/{n_runs}: "
                  f"R²={np.mean(all_r2):.3f} | "
                  f"RMSE={np.mean(all_rmse):.4f}")

    results = {
        'habitat': habitat_name,
        'mean_r2': np.mean(all_r2),
        'std_r2': np.std(all_r2),
        'mean_rmse': np.mean(all_rmse),
        'std_rmse': np.std(all_rmse),
        'n_stations': len(df),
        'features_used': str(features)
    }
    print(f"\nFINAL — {habitat_name}:")
    print(f"  R²   = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}")
    print(f"  RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    return results

# ─────────────────────────────────────────────
# TRAIN FINAL MODEL AND SAVE
# Saved to models/ folder for re-scoring without retraining
# ─────────────────────────────────────────────
def train_and_save_model(df, features, target, habitat_name):
    """Train on full dataset and save model + scaler for future scoring"""
    X = df[features].values
    y = df[target].values

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gpr = build_gpr()
    gpr.fit(X_scaled, y)

    # Save model and scaler
    model_path = f'models/gpr_model_{habitat_name}.pkl'
    scaler_path = f'models/gpr_scaler_{habitat_name}.pkl'
    features_path = f'models/gpr_features_{habitat_name}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(gpr, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"  Model saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")
    print(f"  Features saved: {features_path}")
    return gpr, scaler

# ─────────────────────────────────────────────
# RE-SCORE NEW DATA (no retraining needed)
# Call this when new CRMS or Sentinel-2 data arrives
# Loads saved model and scores new observations
# ─────────────────────────────────────────────
def rescore_new_data(new_data_path='data/fused_dataset_final.csv',
                     output_path='results/gpr_scores_latest.csv'):
    """
    Apply saved GPR models to new data without retraining.
    Called when new monthly CRMS or Sentinel-2 data is ingested.
    Produces updated carbon stock estimates with uncertainty intervals.
    """
    print("\n" + "="*60)
    print("GPR RE-SCORING — NEW DATA")
    print("Applying saved models to new observations")
    print("="*60)

    if not os.path.exists(new_data_path):
        print(f"New data not found at {new_data_path}")
        return None

    df_new = pd.read_csv(new_data_path)
    print(f"New data: {df_new.shape[0]} records")

    all_scores = []

    for habitat_name in ['freshwater_intermediate', 'brackish', 'saline']:
        model_path = f'models/gpr_model_{habitat_name}.pkl'
        scaler_path = f'models/gpr_scaler_{habitat_name}.pkl'
        features_path = f'models/gpr_features_{habitat_name}.pkl'

        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            print(f"  {habitat_name}: No saved model found — run full training first")
            continue

        # Load saved model
        with open(model_path, 'rb') as f:
            gpr = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        # Filter to habitat
        if habitat_name == 'freshwater_intermediate':
            mask = df_new['marsh_type'].isin(['Fresh', 'Freshwater', 'Intermediate'])
        elif habitat_name == 'brackish':
            mask = df_new['marsh_type'].isin(['Brackish'])
        else:
            mask = df_new['marsh_type'].isin(['Saline'])

        df_habitat = df_new[mask].copy()

        # Check features available
        available = [f for f in features if f in df_habitat.columns]
        missing = [f for f in features if f not in df_habitat.columns]
        if missing:
            print(f"  {habitat_name}: Missing features {missing} — skipping")
            continue

        df_habitat = df_habitat.dropna(subset=available)
        if len(df_habitat) == 0:
            print(f"  {habitat_name}: No valid rows after dropna")
            continue

        # Score
        X = df_habitat[available].values
        X_scaled = scaler.transform(X)
        y_pred, y_std = gpr.predict(X_scaled, return_std=True)

        df_habitat['carbon_stock_predicted'] = y_pred
        df_habitat['carbon_stock_uncertainty'] = y_std
        df_habitat['high_uncertainty_flag'] = y_std > (y_std.mean() + y_std.std())
        df_habitat['habitat_model'] = habitat_name
        df_habitat['scored_at'] = pd.Timestamp.now().isoformat()

        n_flagged = df_habitat['high_uncertainty_flag'].sum()
        print(f"  {habitat_name}: {len(df_habitat)} records scored | "
              f"{n_flagged} flagged high uncertainty")

        all_scores.append(df_habitat)

    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
        scores_df.to_csv(output_path, index=False)
        print(f"\nScores saved: {output_path}")
        print(f"Total records scored: {len(scores_df)}")
        return scores_df
    else:
        print("No records scored — check model files and data")
        return None

# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY
# Post-training feature importance for C6 Human Understandability
# ─────────────────────────────────────────────
def compute_shap_values(gpr, scaler, X, features, habitat_name):
    # Remove any NaN rows before SHAP computation
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if len(X) == 0:
        print(f"  No valid rows for SHAP — skipping")
        return None
    X_scaled = scaler.transform(X)
    background = X_scaled[:min(20, len(X_scaled))]
    sample = X_scaled[:min(50, len(X_scaled))]

    explainer = shap.KernelExplainer(gpr.predict, background)
    shap_values = explainer.shap_values(sample, nsamples=100)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, feature_names=features, show=False)
    plt.title(f'Feature Importance — {habitat_name} (SHAP)')
    plt.tight_layout()
    plt.savefig(f'figures/shap_{habitat_name}.png', dpi=150)
    plt.close()

    mean_shap = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    mean_shap.to_csv(f'results/shap_{habitat_name}.csv', index=False)
    print(f"  Top feature: {mean_shap.iloc[0]['feature']} "
          f"(SHAP={mean_shap.iloc[0]['mean_abs_shap']:.4f})")
    return shap_values

# ─────────────────────────────────────────────
# PREDICTION PLOT
# ─────────────────────────────────────────────
def plot_predictions(df, pred_col, std_col, target, habitat_name):
    if target not in df.columns or pred_col not in df.columns:
        return
    y_true = df[target].values
    y_pred = df[pred_col].values
    y_std = df[std_col].values

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.scatter(y_true, y_pred, alpha=0.5, s=30,
               color='#4fc3f7', label='Predictions')
    ax.errorbar(np.sort(y_true), y_pred[np.argsort(y_true)],
                yerr=1.96 * y_std[np.argsort(y_true)],
                fmt='none', alpha=0.15, color='#a0a0b0', label='95% CI')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, label='1:1 line')
    ax.set_xlabel('Observed', color='#e0e0e0')
    ax.set_ylabel('Predicted', color='#e0e0e0')
    ax.set_title(f'GPR Carbon Stock — {habitat_name}', color='#4fc3f7')
    ax.tick_params(colors='#a0a0b0')
    ax.legend(facecolor='#16213e', labelcolor='#e0e0e0')
    plt.tight_layout()
    plt.savefig(f'figures/gpr_predictions_{habitat_name}.png', dpi=150)
    plt.close()
    print(f"  Plot saved: figures/gpr_predictions_{habitat_name}.png")

# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_full_training():
    """Full model training — run annually or after major storm events"""
    print("\n" + "="*60)
    print("GPR FULL TRAINING PIPELINE")
    print("="*60)

    fused_path = 'data/fused_dataset_final.csv'
    if not os.path.exists(fused_path):
        print(f"Fused dataset not found at {fused_path}")
        print("Run 02_crms_preprocessing.py first")
        return

    df_all = pd.read_csv(fused_path)
    print(f"Loaded fused dataset: {df_all.shape}")

    # Deduplicate to one row per station (use most recent)
    if 'year' in df_all.columns:
        df_all = df_all.sort_values('year', ascending=False)
    df_stations = df_all.drop_duplicates(subset='station_id', keep='first').copy()
    print(f"Station-level dataset: {len(df_stations)} unique stations")

    all_results = []

    habitat_filters = {
    'freshwater_intermediate': ['Fresh', 'Freshwater', 'Intermediate'],
    'brackish': ['Brackish'],
    'saline': ['Saline']
    }

    for habitat_name, marsh_types in habitat_filters.items():
        df = df_stations[df_stations['marsh_type'].isin(marsh_types)].copy()
        features = FEATURE_SETS[habitat_name]

        print(f"\n--- {habitat_name.upper()} ({len(df)} stations) ---")

        # Filter to available features
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  Missing (add after hydro merge): {missing}")
        if len(available) < 2:
            print(f"  Not enough features — skipping")
            continue

        # Use carbon_stock if available, else accretion as proxy
        target_col = TARGET if TARGET in df.columns else TARGET_FALLBACK
        print(f"  Target: {target_col}")

        df_clean = df.dropna(subset=available + [target_col])
        # Skip CV for small habitat groups — insufficient stations for
# spatially blocked validation. Train on full dataset instead.
        if len(df_clean) < 50:
            print(f"  n={len(df_clean)} — too few for spatially blocked CV")
            print(f"  Training final model on full dataset")
            print(f"  Validation: Baustian independent sites only")
            gpr, scaler = train_and_save_model(
                df_clean, selected_features, target_col, habitat_name
            )
            X = df_clean[selected_features].values
            compute_shap_values(gpr, scaler, X, selected_features, habitat_name)
            all_results.append({
                'habitat': habitat_name,
                'mean_r2': 'N/A - insufficient n for CV',
                'std_r2': 'N/A',
                'mean_rmse': 'N/A',
                'n_stations': len(df_clean),
                'features_used': str(selected_features)
            })
            continue

        # Backward elimination
        # Skip backward elimination for GPR — OLS p-values don't reflect
        # GPR's nonlinear feature relationships. Use all available features.
        selected_features = available
        print(f"  Using all {len(selected_features)} available features: {selected_features}")

        # Cross-validate
        results = cross_validate_gpr(
            df_clean, selected_features, target_col, habitat_name, n_runs=100
        )
        all_results.append(results)

        # Train final model and save
        gpr, scaler = train_and_save_model(
            df_clean, selected_features, target_col, habitat_name
        )

        # SHAP values
        X = df_clean[selected_features].values
        compute_shap_values(gpr, scaler, X, selected_features, habitat_name)

        # Prediction plot
        X_s = scaler.transform(X)
        y_pred, y_std = gpr.predict(X_s, return_std=True)
        df_clean = df_clean.copy()
        df_clean[f'carbon_pred'] = y_pred
        df_clean[f'carbon_std'] = y_std
        plot_predictions(df_clean, 'carbon_pred', 'carbon_std', target_col, habitat_name)

    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('results/gpr_results.csv', index=False)
        print("\n" + "="*60)
        print("GPR TRAINING COMPLETE")
        print("="*60)
        print(results_df[['habitat', 'mean_r2', 'std_r2',
                           'mean_rmse', 'n_stations']].to_string(index=False))
        print("\nModels saved to models/ — use rescore_new_data() for updates")

# ─────────────────────────────────────────────
# ENTRY POINT
# Run full training by default
# Pass --rescore to score new data without retraining
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if '--rescore' in sys.argv:
        # Re-score new data using saved models
        rescore_new_data()
    else:
        # Full training pipeline
        run_full_training()