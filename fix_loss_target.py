"""
fix_loss_target.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Computes recent_land_loss (proportional land lost since 2015) for
    each CRMS station from the land/water time series. This is a more
    meaningful classification target than the cumulative ever_lost_land
    flag, since it captures current trajectory rather than historical
    occurrence.

    Run after 02_crms_preprocessing.py, before fix_integrate.py.

INPUT:
    data/crms_loss_timeseries.csv   — from 02_crms_preprocessing.py

OUTPUT:
    data/crms_recent_loss.csv       — station_id, recent_land_loss (0–1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
import numpy as np

# Load data
loss_ts = pd.read_csv('data/crms_loss_timeseries.csv')

# Keep recent years
recent = loss_ts[loss_ts['map_year'] >= 2015].copy()

# Aggregate sub-sites within each station/year
yearly = recent.groupby(['station_id', 'map_year']).agg(
    land_acres=('land_acres', 'mean')
).reset_index()

# Sort properly
yearly = yearly.sort_values(['station_id', 'map_year'])

# Count years available
counts = yearly.groupby('station_id').size().reset_index(name='n_years')

# Compute first/last land acreage
recent_loss = yearly.groupby('station_id').agg(
    first_land=('land_acres', 'first'),
    last_land=('land_acres', 'last'),
    n_years=('map_year', 'count')
).reset_index()

# Compute proportional loss only when >=2 years exist
recent_loss['recent_land_loss'] = np.where(
    recent_loss['n_years'] >= 2,
    (recent_loss['first_land'] - recent_loss['last_land']) / recent_loss['first_land'],
    0
)

# Prevent negatives if land increased
recent_loss['recent_land_loss'] = recent_loss['recent_land_loss'].clip(lower=0)

# 3-class ordinal severity based on ecologically meaningful thresholds:
#   LOW      < 5%  — negligible or no detectable loss
#   MODERATE 5–20% — measurable loss, monitoring recommended
#   HIGH     > 20% — significant loss, restoration priority
recent_loss['loss_severity'] = pd.cut(
    recent_loss['recent_land_loss'],
    bins=[-np.inf, 0.05, 0.20, np.inf],
    labels=['LOW', 'MODERATE', 'HIGH']
).astype(str)

recent_loss = recent_loss[['station_id', 'recent_land_loss', 'loss_severity']]

print(recent_loss.head())
print("\nloss_severity distribution:")
print(recent_loss['loss_severity'].value_counts())

recent_loss.to_csv('data/crms_recent_loss.csv', index=False)
print('\nSaved: data/crms_recent_loss.csv')