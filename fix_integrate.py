"""
fix_integrate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Merges recent_land_loss (continuous 0–1 proportional loss) and
    loss_severity (3-class: LOW/MODERATE/HIGH) from fix_loss_target.py
    into fused_dataset_final.csv. Both columns are used downstream:
      - loss_severity → C4.5 target (05, 07)
      - recent_land_loss → FP-Growth discretization + GPR reference (04, 03)

    Run after fix_loss_target.py, before 03/04/05.

INPUTS:
    data/fused_dataset_final.csv    — from 02_crms_preprocessing.py
    data/crms_recent_loss.csv       — from fix_loss_target.py
                                      (contains recent_land_loss + loss_severity)

OUTPUTS:
    data/fused_dataset_final.csv    — updated in-place with both columns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
import numpy as np

# ── Merge recent loss columns into fused dataset ──
print("Merging recent_land_loss + loss_severity into fused dataset...")
fused = pd.read_csv('data/fused_dataset_final.csv')
recent = pd.read_csv('data/crms_recent_loss.csv')

# Drop if already exists from a previous run
for col in ['recent_land_loss', 'loss_severity']:
    if col in fused.columns:
        fused = fused.drop(columns=[col])

merged = fused.merge(
    recent[['station_id', 'recent_land_loss', 'loss_severity']],
    on='station_id', how='left'
)

print(f"Loss columns after merge: {[c for c in merged.columns if 'loss' in c.lower()]}")
print(f"recent_land_loss matched: {merged['recent_land_loss'].notna().sum():,}")
print(f"loss_severity distribution:")
print(merged['loss_severity'].value_counts())

merged.to_csv('data/fused_dataset_final.csv', index=False)
print("Saved updated fused_dataset_final.csv")