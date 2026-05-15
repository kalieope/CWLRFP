"""
05_c45_classification_temporal_roc.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Trains a C4.5 Decision Tree classifier to predict parcel-level
    wetland-to-open-water conversion. C4.5 is chosen because it
    produces human-readable if-then rules restoration planners can
    act on directly.

    Target variable: loss_severity — 3-class ordinal (LOW/MODERATE/HIGH)
    added by fix_loss_target.py + fix_integrate.py.
    Run those two scripts after 02 before running this one.

    Once trained, the saved model can score new data without retraining
    (see rescore_new_data()).

INPUTS:
    data/fused_dataset_final.csv    — fused dataset with recent_land_loss
                                      column (added by fix_integrate.py)

OUTPUTS:
    models/c45_model.pkl             — trained C4.5 classifier
    models/c45_scaler.pkl            — fitted feature scaler
    models/c45_features.pkl          — selected feature list
    results/c45_rules.txt            — plain-language if-then rules
    results/c45_cv_results.csv       — spatially blocked CV performance
    results/temporal_roc_results.csv — ROC-AUC per time horizon
    results/rfp_classifier_report.csv— precision/recall/F1 report
    figures/temporal_roc.png         — temporal ROC curves
    figures/c45_tree.png             — decision tree visualization

WHEN TO RE-RUN:
    Full retrain (this script): annually, or after major storm events
    Re-score only (rescore_new_data): when new CRMS/Sentinel-2 data arrives

VALIDATION DESIGN:
    Temporal ROC: ROC-AUC at 1-year, 3-year, 5-year look-ahead slices
    Spatially blocked CV: folds by geographic block to prevent leakage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (roc_auc_score, f1_score,
                              roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
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
# CLASSIFICATION FEATURES
# Spectral + soil features available now
# 
# ─────────────────────────────────────────────
CLASSIFICATION_FEATURES = [
    # Spectral (Thomas et al. 2019)
    'NDVI', 'EVI',
    # Hydrological (Chenevert & Edmonds 2024)
    'tidal_amplitude_annual_mean',
    'flood_depth_annual_mean',
    'salinity_annual_mean',
    'percent_flooded',
    # Soil (Baustian et al. 2021)
    'bulk_density',
    'percent_organic',
    # Elevation
    'elevation_m',
    # Storm (justified by Louisiana hurricane history)
    'storm_year',
    # Temporal lags (sequential mining)
    'NDVI_lag3', 'NDVI_lag12',
    # Marsh type
    'marsh_Fresh', 'marsh_Intermediate',
    'marsh_Brackish', 'marsh_Saline',
]

TARGET = 'loss_severity'  # 3-class ordinal: LOW / MODERATE / HIGH

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# Temporal lags and marsh type one-hot encoding
# ─────────────────────────────────────────────
def engineer_features(df):
    """Add temporal lag features and marsh type dummies"""
    df = df.copy()

    # Marsh type one-hot
    if 'marsh_type' in df.columns:
        dummies = pd.get_dummies(df['marsh_type'], prefix='marsh')
        for col in ['marsh_Fresh', 'marsh_Intermediate',
                    'marsh_Brackish', 'marsh_Saline']:
            if col not in df.columns:
                df[col] = dummies.get(col, 0)

    # Temporal lags (only if time series data available)
    if 'station_id' in df.columns and 'year' in df.columns:
        df = df.sort_values(['station_id', 'year'])
        for var in ['NDVI', 'flood_depth', 'salinity']:
            if var in df.columns:
                df[f'{var}_lag3'] = df.groupby('station_id')[var].shift(3)
                df[f'{var}_lag12'] = df.groupby('station_id')[var].shift(12)

    return df

# ─────────────────────────────────────────────
# SPATIAL BLOCK ASSIGNMENT
# ─────────────────────────────────────────────
def assign_spatial_blocks(df, n_blocks=5):
    df = df.copy()
    if 'lat' not in df.columns or 'lon' not in df.columns:
        df['cv_block'] = np.random.randint(0, n_blocks, len(df))
        return df
    df['lat_block'] = pd.cut(df['lat'], bins=n_blocks, labels=False)
    df['lon_block'] = pd.cut(df['lon'], bins=n_blocks, labels=False)
    df['cv_block'] = (
        df['lat_block'].astype(float) * n_blocks +
        df['lon_block'].astype(float)
    ) % n_blocks
    return df

# ─────────────────────────────────────────────
# C4.5 DECISION TREE
# entropy criterion approximates C4.5 information gain ratio
# class_weight='balanced' handles imbalanced loss events
# ─────────────────────────────────────────────
def build_c45(max_depth=6):
    return DecisionTreeClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )

# ─────────────────────────────────────────────
# EXPORT PLAIN-LANGUAGE RULES
# C6: Human-readable output for restoration planners
# ─────────────────────────────────────────────
def export_rules(clf, feature_names, output_path='results/c45_rules.txt'):
    rules_text = export_text(clf, feature_names=feature_names, max_depth=6)
    with open(output_path, 'w') as f:
        f.write("C4.5 WETLAND LOSS PREDICTION RULES\n")
        f.write("=" * 60 + "\n")
        f.write("Plain-language decision rules for coastal restoration planners\n")
        f.write("Generated by: 05_c45_classification_temporal_roc.py\n")
        f.write("=" * 60 + "\n\n")
        f.write(rules_text)
    print(f"  Plain-language rules saved: {output_path}")
    return rules_text

# ─────────────────────────────────────────────
# TEMPORAL ROC VALIDATION
# Train on early years, evaluate on 1/3/5-year future slices
# Shows how model accuracy degrades over prediction horizon (C8)
# ─────────────────────────────────────────────
def temporal_roc_validation(df, features, target):
    print("\n" + "="*60)
    print("TEMPORAL ROC VALIDATION")
    print("Evaluating accuracy degradation over prediction horizon")
    print("="*60)

    available = [f for f in features if f in df.columns]
    df = df.dropna(subset=available + [target])

    if 'year' not in df.columns:
        print("No year column — skipping temporal ROC")
        return None, None, None, None

    years = sorted(df['year'].unique())
    if len(years) < 4:
        print("Not enough years for temporal validation")
        return None, None, None, None

    cutoff_idx = int(len(years) * 0.6)
    train_years = years[:cutoff_idx]
    print(f"Training years: {train_years[0]}–{train_years[-1]}")
    print(f"Validation years: {years[cutoff_idx]}–{years[-1]}")

    test_slices = {
        '1yr_ahead': years[cutoff_idx:cutoff_idx+1],
        '3yr_ahead': years[cutoff_idx:min(cutoff_idx+3, len(years))],
        '5yr_ahead': years[cutoff_idx:min(cutoff_idx+5, len(years))]
    }

    train_mask = df['year'].isin(train_years)
    X_train = df.loc[train_mask, available].values
    y_train = df.loc[train_mask, target].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = build_c45()
    clf.fit(X_train_s, y_train)

    temporal_results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')

    for idx, (horizon, test_years) in enumerate(test_slices.items()):
        ax = axes[idx]
        ax.set_facecolor('#16213e')

        test_mask = df['year'].isin(test_years)
        if test_mask.sum() == 0:
            ax.text(0.5, 0.5, 'No test data',
                    transform=ax.transAxes, color='#a0a0b0', ha='center')
            continue

        X_test = df.loc[test_mask, available].values
        y_test = df.loc[test_mask, target].values
        X_test_s = scaler.transform(X_test)
        y_proba = clf.predict_proba(X_test_s)
        y_pred = clf.predict(X_test_s)

        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(
                y_test, y_proba,
                multi_class='ovr', average='macro',
                labels=clf.classes_
            )
            f1 = f1_score(
                y_test, y_pred, average='macro', zero_division=0
            )

            # Plot HIGH-class one-vs-rest ROC for interpretability
            high_idx = list(clf.classes_).index('HIGH') \
                if 'HIGH' in clf.classes_ else 0
            y_test_bin = (y_test == 'HIGH').astype(int)
            fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, high_idx])

            temporal_results[horizon] = {
                'auc_macro_ovr': auc, 'f1_macro': f1,
                'n_test': len(y_test),
                'n_high': int((y_test == 'HIGH').sum())
            }

            ax.plot(fpr, tpr, color='#4fc3f7', linewidth=2,
                    label=f'HIGH OvR AUC={auc:.3f}')
            ax.plot([0, 1], [0, 1], color='#a0a0b0',
                    linestyle='--', alpha=0.5)
            ax.set_xlabel('False Positive Rate', color='#a0a0b0')
            ax.set_ylabel('True Positive Rate', color='#a0a0b0')
            ax.set_title(f'{horizon.replace("_", " ").title()}\nF1={f1:.3f}',
                         color='#4fc3f7')
            ax.tick_params(colors='#a0a0b0')
            ax.legend(facecolor='#16213e', labelcolor='#e0e0e0')
            ax.spines['bottom'].set_color('#0f3460')
            ax.spines['left'].set_color('#0f3460')
            ax.spines['top'].set_color('#0f3460')
            ax.spines['right'].set_color('#0f3460')

            print(f"  {horizon}: AUC={auc:.3f} | F1={f1:.3f} | n={len(y_test)}")
        else:
            print(f"  {horizon}: Only one class in test set")

    plt.suptitle('Temporal ROC — Wetland Loss Prediction\n'
                 'Performance envelope across prediction horizons',
                 color='#4fc3f7', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/temporal_roc.png', dpi=150,
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print("  Temporal ROC plot saved: figures/temporal_roc.png")

    return temporal_results, clf, scaler, available

# ─────────────────────────────────────────────
# SPATIALLY BLOCKED CROSS-VALIDATION
# ─────────────────────────────────────────────
def spatially_blocked_cv(df, features, target, n_blocks=5):
    print("\n" + "="*60)
    print("SPATIALLY BLOCKED CROSS-VALIDATION")
    print("="*60)

    df = assign_spatial_blocks(df, n_blocks)
    available = [f for f in features if f in df.columns]
    df = df.dropna(subset=available + [target])

    aucs, f1s = [], []

    for block in range(n_blocks):
        train_mask = df['cv_block'] != block
        test_mask = df['cv_block'] == block

        if test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, available].values
        y_train = df.loc[train_mask, target].values
        X_test = df.loc[test_mask, available].values
        y_test = df.loc[test_mask, target].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = build_c45()
        clf.fit(X_train_s, y_train)
        y_proba = clf.predict_proba(X_test_s)
        y_pred = clf.predict(X_test_s)

        test_classes = set(np.unique(y_test))
        all_classes_present = test_classes >= set(clf.classes_)

        if len(test_classes) > 1 and all_classes_present:
            auc = roc_auc_score(
                y_test, y_proba,
                multi_class='ovr', average='macro',
                labels=clf.classes_
            )
            f1 = f1_score(
                y_test, y_pred, average='macro', zero_division=0
            )
            aucs.append(auc)
            f1s.append(f1)
            print(f"  Block {block}: AUC={auc:.3f} | F1={f1:.3f} | n={len(y_test)}")
        else:
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1s.append(f1)
            missing = set(clf.classes_) - test_classes
            print(f"  Block {block}: AUC=skipped (missing classes: {missing}) | "
                  f"F1={f1:.3f} | n={len(y_test)}")

    if f1s:
        cv_results = {
            'mean_auc': np.mean(aucs) if aucs else float('nan'),
            'std_auc': np.std(aucs) if aucs else float('nan'),
            'mean_f1': np.mean(f1s), 'std_f1': np.std(f1s),
            'n_blocks_with_auc': len(aucs)
        }
        if aucs:
            print(f"\n  Mean AUC ({len(aucs)} blocks): "
                  f"{cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}")
        else:
            print("\n  Mean AUC: n/a (no block had all 3 classes in test set)")
        print(f"  Mean F1:  {cv_results['mean_f1']:.3f} "
              f"± {cv_results['std_f1']:.3f}")
        pd.DataFrame([cv_results]).to_csv(
            'results/c45_cv_results.csv', index=False
        )
        return cv_results

# ─────────────────────────────────────────────
# TRAIN FINAL MODEL AND SAVE
# ─────────────────────────────────────────────
def train_and_save_model(df, features, target):
    available = [f for f in features if f in df.columns]
    df_clean = df.dropna(subset=available + [target])

    X = df_clean[available].values
    y = df_clean[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = build_c45()
    clf.fit(X_scaled, y)

    with open('models/c45_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('models/c45_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/c45_features.pkl', 'wb') as f:
        pickle.dump(available, f)

    print("  Model saved: models/c45_model.pkl")
    print("  Scaler saved: models/c45_scaler.pkl")
    print("  Features saved: models/c45_features.pkl")
    return clf, scaler, available

# ─────────────────────────────────────────────
# RE-SCORE NEW DATA (no retraining needed)
# Call when new CRMS or Sentinel-2 data arrives
# ─────────────────────────────────────────────
def rescore_new_data(new_data_path='data/fused_dataset_final.csv',
                     output_path='results/c45_scores_latest.csv'):
    """
    Apply saved C4.5 model to new data without retraining.
    Produces updated loss probability scores and high-risk flags.
    Called when new monthly CRMS or Sentinel-2 data is ingested.
    """
    print("\n" + "="*60)
    print("C4.5 RE-SCORING — NEW DATA")
    print("Applying saved model to new observations")
    print("="*60)

    model_files = ['models/c45_model.pkl',
                   'models/c45_scaler.pkl',
                   'models/c45_features.pkl']
    if not all(os.path.exists(f) for f in model_files):
        print("No saved model found — run full training first")
        return None

    with open('models/c45_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/c45_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/c45_features.pkl', 'rb') as f:
        features = pickle.load(f)

    if not os.path.exists(new_data_path):
        print(f"New data not found: {new_data_path}")
        return None

    df = pd.read_csv(new_data_path)
    df = engineer_features(df)
    print(f"New data: {df.shape[0]} records")

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Missing features: {missing}")

    df_clean = df.dropna(subset=available)
    if len(df_clean) == 0:
        print("No valid rows after dropna")
        return None

    X = df_clean[available].values
    X_scaled = scaler.transform(X)

    y_proba = clf.predict_proba(X_scaled)
    y_pred = clf.predict(X_scaled)

    # Weighted risk score: 0*P(LOW) + 0.5*P(MODERATE) + 1.0*P(HIGH)
    # Uses clf.classes_ ordering (alphabetical: HIGH, LOW, MODERATE)
    class_weights = {'LOW': 0.0, 'MODERATE': 0.5, 'HIGH': 1.0}
    weight_array = np.array([class_weights[c] for c in clf.classes_])
    risk_score = np.clip(y_proba @ weight_array, 0.02, 0.97)

    df_clean = df_clean.copy()
    df_clean['loss_probability'] = risk_score
    df_clean['loss_predicted'] = y_pred
    df_clean['risk_level'] = pd.cut(
        risk_score,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['LOW', 'MODERATE', 'HIGH']
    )
    df_clean['scored_at'] = pd.Timestamp.now().isoformat()

    df_clean.to_csv(output_path, index=False)
    print(f"Scores saved: {output_path}")
    print(f"Records scored: {len(df_clean)}")
    print(f"High risk: {(df_clean['risk_level']=='HIGH').sum()}")
    print(f"Moderate:  {(df_clean['risk_level']=='MODERATE').sum()}")
    print(f"Low risk:  {(df_clean['risk_level']=='LOW').sum()}")
    return df_clean

# ─────────────────────────────────────────────
# VISUALIZE DECISION TREE
# ─────────────────────────────────────────────
def visualize_tree(clf, feature_names, max_depth=4):
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    plot_tree(
        clf, feature_names=feature_names,
        class_names=['HIGH', 'LOW', 'MODERATE'],  # alphabetical — matches clf.classes_
        filled=True, rounded=True,
        max_depth=max_depth, ax=ax, fontsize=8
    )
    plt.title('C4.5 Wetland Loss Decision Tree',
              color='#4fc3f7', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/c45_tree.png', dpi=150,
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print("  Decision tree saved: figures/c45_tree.png")

# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_full_training():
    print("\n" + "="*60)
    print("C4.5 FULL TRAINING PIPELINE")
    print("="*60)

    fused_path = 'data/fused_dataset_final.csv'
    if not os.path.exists(fused_path):
        print(f"Fused dataset not found: {fused_path}")
        print("Run 02_crms_preprocessing.py first")
        return

    df = pd.read_csv(fused_path)
    df = engineer_features(df)
    print(f"Loaded and engineered features: {df.shape}")

    # Deduplicate to station level for spatial CV
    if 'year' in df.columns:
        df_stations = df.sort_values('year', ascending=False)\
                        .drop_duplicates('station_id', keep='first').copy()
    else:
        df_stations = df.drop_duplicates('station_id', keep='first').copy()
    print(f"Station-level dataset: {len(df_stations)} unique stations")

    if TARGET not in df_stations.columns:
        print(f"Target '{TARGET}' not found in dataset")
        print(f"Available: {list(df_stations.columns)}")
        return

    # Temporal ROC on time series data
    results = temporal_roc_validation(df, CLASSIFICATION_FEATURES, TARGET)
    if results[0]:
        temporal_results, clf_temporal, scaler_temporal, features_used = results
        pd.DataFrame(temporal_results).T.to_csv(
            'results/temporal_roc_results.csv'
        )

    # Spatially blocked CV on station-level data
    spatially_blocked_cv(df_stations, CLASSIFICATION_FEATURES, TARGET)

    # Train final model and save
    clf, scaler, features_used = train_and_save_model(
        df_stations, CLASSIFICATION_FEATURES, TARGET
    )

    # Export plain-language rules
    export_rules(clf, features_used)

    # Visualize tree
    visualize_tree(clf, features_used)

    print("\n" + "="*60)
    print("C4.5 TRAINING COMPLETE")
    print("Model saved to models/ — use --rescore flag for updates")
    print("="*60)

# ─────────────────────────────────────────────
# ENTRY POINT
# python 05_c45_classification_temporal_roc.py          → full training
# python 05_c45_classification_temporal_roc.py --rescore → score new data
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if '--rescore' in sys.argv:
        rescore_new_data()
    else:
        run_full_training()