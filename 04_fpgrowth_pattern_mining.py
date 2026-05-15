"""
04_fpgrowth_pattern_mining.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Discretizes continuous CRMS features into ecologically meaningful
    bins and runs FP-Growth frequent pattern mining to discover
    co-occurrence rules linking habitat, hydrology, spectral conditions,
    and elevation to wetland loss outcomes.

    Apriori is also run as a baseline for class comparison. Rules with
    'recent_loss_YES' in the consequent are extracted as land-loss rules
    for the dashboard.

INPUTS:
    data/fused_dataset_final.csv    — station-month fused dataset from 02

OUTPUTS:
    results/fpgrowth_itemsets.csv   — frequent itemsets (FP-Growth)
    results/apriori_itemsets.csv    — frequent itemsets (Apriori baseline)
    results/association_rules.csv   — all association rules mined
    results/land_loss_rules.csv     — rules whose consequent is loss_significant
                                      (MODERATE or HIGH severity — any detectable loss)
    figures/                        — rule scatter plots (support vs confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────
# 1. DISCRETIZATION BINS
# Ecologically meaningful thresholds from literature
# ─────────────────────────────────────────────
BINS = {
    'flood_depth_annual_mean': {
    'bins': [-np.inf, -0.18, 0.39, np.inf],
    'labels': ['flood_low', 'flood_moderate', 'flood_high']
    },
    'salinity_annual_mean': {
        'bins': [0, 0.5, 8.0, np.inf],
        'labels': ['sal_fresh', 'sal_brackish', 'sal_saline']
    },
    'tidal_amplitude_annual_mean': {
        'bins': [0, 2.5, 5.5, np.inf],
        'labels': ['tide_low', 'tide_moderate', 'tide_high']
    },
    'NDVI': {
        'bins': [-1, 0.2, 0.5, 1.0],
        'labels': ['ndvi_low', 'ndvi_moderate', 'ndvi_high']
    },
    'accretion_median': {
        'bins': [0, 2, 5, np.inf],
        'labels': ['accretion_low', 'accretion_moderate', 'accretion_high']
    },
    'elevation_m': {
    'bins': [-np.inf, -0.5, 0.5, np.inf],
    'labels': ['elev_below_sea', 'elev_near_sea', 'elev_above_sea']
    },
    'percent_flooded': {
    'bins': [0, 30, 70, 100],
    'labels': ['flooded_low', 'flooded_moderate', 'flooded_high']
    },
}

# ─────────────────────────────────────────────
# 2. DISCRETIZE CONTINUOUS FEATURES
# ─────────────────────────────────────────────
def discretize_features(df):
    """Discretize continuous features into ecological bins"""
    df_disc = df.copy()

    for col, config in BINS.items():
        if col in df_disc.columns:
            df_disc[f'{col}_bin'] = pd.cut(
                df_disc[col],
                bins=config['bins'],
                labels=config['labels'],
                include_lowest=True
            )
            print(f"Discretized {col}: {df_disc[f'{col}_bin'].value_counts().to_dict()}")

    # Marsh type as-is (already categorical)
    if 'marsh_type' in df_disc.columns:
        df_disc['marsh_bin'] = 'marsh_' + df_disc['marsh_type'].str.lower()

    # Loss severity: collapse to binary for pattern mining — MODERATE and HIGH
    # both represent detectable, ecologically meaningful loss. HIGH alone is too
    # rare to generate rules at any reasonable support threshold.
    if 'loss_severity' in df_disc.columns:
        df_disc['loss_bin'] = df_disc['loss_severity'].map({
            'LOW': 'loss_stable',
            'MODERATE': 'loss_significant',
            'HIGH': 'loss_significant'
        })
    return df_disc

# ─────────────────────────────────────────────
# 3. BUILD TRANSACTION DATABASE
# Each station-month = one transaction (set of discretized features)
# ─────────────────────────────────────────────
def build_transactions(df_disc):
    """Convert discretized features to transaction format"""
    bin_cols = [c for c in df_disc.columns if c.endswith('_bin')]
    transactions = []

    for _, row in df_disc.iterrows():
        transaction = []
        for col in bin_cols:
            if pd.notna(row[col]):
                transaction.append(str(row[col]))
        if transaction:
            transactions.append(transaction)

    print(f"Built {len(transactions)} transactions with avg {np.mean([len(t) for t in transactions]):.1f} items each")
    return transactions

# ─────────────────────────────────────────────
# 4. ONE-HOT ENCODE FOR MLXTEND
# ─────────────────────────────────────────────
def encode_transactions(transactions):
    """One-hot encode transactions for mlxtend"""
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    print(f"Encoded transaction matrix: {df_encoded.shape}")
    return df_encoded, te

# ─────────────────────────────────────────────
# 5. APRIORI BASELINE
# Run first for comparison with FP-Growth
# ─────────────────────────────────────────────
def run_apriori(df_encoded, min_support=0.10):
    """Run Apriori as baseline frequent itemset mining"""
    print(f"\nRunning Apriori (min_support={min_support})...")
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=4
    )
    print(f"Apriori found {len(frequent_itemsets)} frequent itemsets")
    return frequent_itemsets

# ─────────────────────────────────────────────
# 6. FP-GROWTH PRIMARY ENGINE
# More efficient than Apriori at this scale
# ─────────────────────────────────────────────
def run_fpgrowth(df_encoded, min_support=0.10):
    """Run FP-Growth as primary pattern mining engine"""
    print(f"\nRunning FP-Growth (min_support={min_support})...")
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=5
    )
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    print(f"FP-Growth found {len(frequent_itemsets)} frequent itemsets")
    print(f"Itemset length distribution:\n{frequent_itemsets['length'].value_counts().sort_index()}")
    return frequent_itemsets

# ─────────────────────────────────────────────
# 7. ASSOCIATION RULES
# Filter by confidence and lift thresholds
# ─────────────────────────────────────────────
def mine_association_rules(frequent_itemsets, min_confidence=0.75, min_lift=1.5):
    """Mine association rules from frequent itemsets"""
    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_confidence
    )
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values('lift', ascending=False)
    print(f"\nAssociation rules (conf≥{min_confidence}, lift≥{min_lift}): {len(rules)}")
    return rules

# ─────────────────────────────────────────────
# 8. FILTER LAND-LOSS RULES
# Extract rules with land_loss_YES as consequent
# These are the planner-facing rules
# ─────────────────────────────────────────────
def extract_loss_rules(rules):
    """Extract rules predicting land loss"""
    if len(rules) == 0:
        print("No rules found — returning empty DataFrame")
        return pd.DataFrame()

    loss_rules = rules[
        rules['consequents'].apply(lambda x: 'loss_significant' in str(x))
    ].copy()

    if len(loss_rules) == 0:
        print("No land loss rules found — check if loss_significant bin exists in transactions")
        print("All rules found:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        return pd.DataFrame()

    loss_rules = loss_rules.sort_values('confidence', ascending=False)
    print(f"\nLand loss prediction rules: {len(loss_rules)}")

    print("\nTop 5 land loss rules:")
    for _, row in loss_rules.head(5).iterrows():
        antecedent = ', '.join(sorted(row['antecedents']))
        print(f"  IF {{{antecedent}}} → land_loss")
        print(f"  support={row['support']:.3f}, "
              f"confidence={row['confidence']:.3f}, "
              f"lift={row['lift']:.2f}")

    return loss_rules

# ─────────────────────────────────────────────
# 9. TEMPORAL SEQUENTIAL MINING
# Identify ordered precursor sequences to land loss
# ─────────────────────────────────────────────
def temporal_sequential_mining(df, station_col='station_id',
                                 time_col='date', target_col='land_loss'):
    """
    Simple sequential pattern mining:
    For each station that eventually lost land, extract the
    sequence of monthly states in the 24 months prior to loss.
    Compare support vs non-loss stations.
    """
    print("\nTemporal Sequential Mining...")

    if target_col not in df.columns:
        print(f"'{target_col}' column not found — skipping temporal mining")
        return None

    loss_stations = df[df[target_col] == 1][station_col].unique()
    no_loss_stations = df[df[target_col] == 0][station_col].unique()

    print(f"Loss stations: {len(loss_stations)}")
    print(f"No-loss stations: {len(no_loss_stations)}")

    # For each loss station, get 24-month precursor sequence
    precursor_patterns = []
    for station in loss_stations:
        station_data = df[df[station_col] == station].sort_values(time_col)
        if len(station_data) >= 6:
            # Last 24 months before loss
            sequence = station_data.tail(24)
            # Check for declining NDVI trend
            if 'NDVI' in sequence.columns:
                ndvi_trend = sequence['NDVI'].diff().mean()
                if ndvi_trend < -0.02:
                    precursor_patterns.append('ndvi_declining_precursor')
            # Check for flood depth increase
            if 'flood_depth' in sequence.columns:
                flood_trend = sequence['flood_depth'].diff().mean()
                if flood_trend > 0.01:
                    precursor_patterns.append('flood_increasing_precursor')

    if precursor_patterns:
        from collections import Counter
        pattern_counts = Counter(precursor_patterns)
        total = len(loss_stations)
        print("\nPrecursor patterns in loss stations:")
        for pattern, count in pattern_counts.most_common():
            print(f"  {pattern}: {count}/{total} ({count/total:.1%})")

    return precursor_patterns

# ─────────────────────────────────────────────
# 10. VISUALIZE RULES
# ─────────────────────────────────────────────
def plot_rules(rules, habitat_name='all'):
    """Plot support vs confidence colored by lift"""
    if len(rules) == 0:
        return
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        rules['support'], rules['confidence'],
        c=rules['lift'], cmap='RdYlGn', alpha=0.7, s=50
    )
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title(f'Association Rules — {habitat_name}')
    plt.axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label='conf=0.75')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/rules_{habitat_name}.png', dpi=150)
    plt.close()
    print(f"Rules plot saved: figures/rules_{habitat_name}.png")

# ─────────────────────────────────────────────
# RUN FULL PIPELINE
# ─────────────────────────────────────────────
if __name__ == '__main__':
    fused_path = 'data/fused_dataset_final.csv'

    if not os.path.exists(fused_path):
        print(f"Fused dataset not found at {fused_path}")
        print("Run 02_crms_preprocessing.py and wait for GEE export first")
    else:
        df = pd.read_csv(fused_path)
        print(f"Loaded fused dataset: {df.shape}")

        # Discretize
        df_disc = discretize_features(df)

        # Build transactions
        transactions = build_transactions(df_disc)

        # Encode
        df_encoded, te = encode_transactions(transactions)

        # Apriori baseline
        apriori_itemsets = run_apriori(df_encoded, min_support=0.10)
        apriori_itemsets.to_csv('results/apriori_itemsets.csv', index=False)

        # FP-Growth primary
        fp_itemsets = run_fpgrowth(df_encoded, min_support=0.10)
        fp_itemsets.to_csv('results/fpgrowth_itemsets.csv', index=False)

        # Association rules
        rules = mine_association_rules(fp_itemsets, min_confidence=0.35, min_lift=1.0)
        rules.to_csv('results/association_rules.csv', index=False)

        # Land loss rules
        loss_rules = extract_loss_rules(rules)
        loss_rules.to_csv('results/land_loss_rules.csv', index=False)

        # Temporal mining — use MODERATE|HIGH as binary proxy (mirrors FP-Growth collapse)
        if 'loss_severity' in df.columns:
            df['_loss_flag'] = df['loss_severity'].isin(['MODERATE', 'HIGH']).astype(int)
            temporal_sequential_mining(df, target_col='_loss_flag', time_col='year')
        else:
            temporal_sequential_mining(df, target_col='ever_lost_land', time_col='year')

        # Plot
        plot_rules(rules, habitat_name='full_deltaic_plain')

        print("\nFP-Growth mining complete.")
        print(f"Results saved in results/")