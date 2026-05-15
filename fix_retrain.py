"""
fix_retrain.py
Merges recent loss flag into fused dataset and retrains C4.5
with recent_land_loss as target instead of ever_lost_land
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: Merge recent loss into fused dataset ──
print("Merging recent loss flag into fused dataset...")
fused = pd.read_csv('data/fused_dataset_final.csv')
recent = pd.read_csv('data/crms_recent_loss.csv')

# Drop if already exists from previous run
if 'recent_land_loss' in fused.columns:
    fused = fused.drop(columns=['recent_land_loss'])

merged = fused.merge(recent[['station_id','recent_land_loss']], 
                     on='station_id', how='left')
print(f"Columns after merge: {[c for c in merged.columns if 'loss' in c.lower()]}")
print(f"Recent loss matched: {merged['recent_land_loss'].notna().sum():,}")
print(f"Loss rate: {merged['recent_land_loss'].mean():.1%}")

merged.to_csv('data/fused_dataset_final.csv', index=False)
fused = merged
print("Saved updated fused_dataset_final.csv")

# ── Step 2: Retrain C4.5 with new target ──
print("\nRetraining C4.5 with recent_land_loss target...")

TARGET = 'recent_land_loss'

FEATURES = [
    'NDVI', 'NDWI', 'EVI', 'B8', 'B11', 'B12',
    'bulk_density', 'percent_organic',
    'tidal_amplitude_annual_mean',
    'flood_depth_annual_mean',
    'salinity_annual_mean',
    'elevation_m',
    'percent_flooded',
    'storm_year',
    'max_category_hit',
]

# Deduplicate to station level
if 'year' in fused.columns:
    df = fused.sort_values('year', ascending=False).drop_duplicates(
        'station_id', keep='first'
    ).copy()
else:
    df = fused.drop_duplicates('station_id', keep='first').copy()

# Add marsh type dummies
if 'marsh_type' in df.columns:
    dummies = pd.get_dummies(df['marsh_type'], prefix='marsh')
    df = pd.concat([df, dummies], axis=1)
    for col in ['marsh_Fresh', 'marsh_Intermediate',
                'marsh_Brackish', 'marsh_Saline']:
        if col not in df.columns:
            df[col] = 0
    FEATURES += ['marsh_Fresh', 'marsh_Intermediate',
                 'marsh_Brackish', 'marsh_Saline']

available = [f for f in FEATURES if f in df.columns]
df_clean = df.dropna(subset=available + [TARGET])
print(f"Training on {len(df_clean)} stations")
print(f"Loss rate: {df_clean[TARGET].mean():.1%}")

# ── Temporal ROC ──
print("\nTemporal ROC Validation...")
if 'year' in fused.columns:
    df_ts = fused.copy()
    df_ts['year'] = pd.to_numeric(df_ts['year'], errors='coerce')

    # Add marsh dummies first
    if 'marsh_type' in df_ts.columns:
        dummies_ts = pd.get_dummies(df_ts['marsh_type'], prefix='marsh')
        df_ts = pd.concat([df_ts, dummies_ts], axis=1)
        for col in ['marsh_Fresh', 'marsh_Intermediate',
                    'marsh_Brackish', 'marsh_Saline']:
            if col not in df_ts.columns:
                df_ts[col] = 0

    avail_ts = [f for f in available if f in df_ts.columns]
    df_ts = df_ts.dropna(subset=avail_ts + [TARGET])
    years = sorted(df_ts['year'].dropna().unique())
    cutoff = years[int(len(years) * 0.6)]
    train_mask = df_ts['year'] <= cutoff
    test_mask = df_ts['year'] > cutoff

    X_train = df_ts.loc[train_mask, avail_ts].values
    y_train = df_ts.loc[train_mask, TARGET].values
    X_test = df_ts.loc[test_mask, avail_ts].values
    y_test = df_ts.loc[test_mask, TARGET].values

    scaler_ts = StandardScaler()
    X_train_s = scaler_ts.fit_transform(X_train)
    X_test_s = scaler_ts.transform(X_test)

    clf_ts = DecisionTreeClassifier(
        criterion='entropy', max_depth=6,
        min_samples_leaf=5, class_weight='balanced',
        random_state=42
    )
    clf_ts.fit(X_train_s, y_train)
    y_prob = clf_ts.predict_proba(X_test_s)[:, 1]
    y_pred = clf_ts.predict(X_test_s)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"Temporal AUC: {auc:.3f} | F1: {f1:.3f} | n={len(y_test)}")

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color='#4fc3f7', linewidth=2,
                label=f'AUC={auc:.3f}')
        ax.plot([0,1],[0,1], color='#a0a0b0', linestyle='--')
        ax.set_xlabel('False Positive Rate', color='#a0a0b0')
        ax.set_ylabel('True Positive Rate', color='#a0a0b0')
        ax.set_title('Temporal ROC — Recent Land Loss (2015+)',
                     color='#4fc3f7')
        ax.legend(facecolor='#16213e', labelcolor='#e0e0e0')
        plt.tight_layout()
        plt.savefig('figures/temporal_roc_recent.png', dpi=150,
                    facecolor='#1a1a2e')
        plt.close()
        print("Saved: figures/temporal_roc_recent.png")

# ── Train final model ──
X = df_clean[available].values
y = df_clean[TARGET].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = DecisionTreeClassifier(
    criterion='entropy', max_depth=6,
    min_samples_leaf=5, class_weight='balanced',
    random_state=42
)
clf.fit(X_scaled, y)

# Save
with open('models/c45_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('models/c45_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/c45_features.pkl', 'wb') as f:
    pickle.dump(available, f)

# Export rules
rules = export_text(clf, feature_names=available, max_depth=6)
with open('results/c45_rules.txt', 'w') as f:
    f.write("C4.5 WETLAND LOSS PREDICTION RULES\n")
    f.write("Target: Recent land loss (2015-2021)\n")
    f.write("=" * 50 + "\n\n")
    f.write(rules)

print("\nModel saved to models/")
print("Rules saved to results/c45_rules.txt")
print("\nNow run: python 07_spatial_prediction.py")
print("Then run: streamlit run 06_dashboard.py")