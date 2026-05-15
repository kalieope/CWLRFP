"""
sample_wall.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Takes the full wall_to_wall_predictions.csv (all coastal pixels)
    and creates a stratified sample of up to 15,000 points — 5,000
    each from HIGH / MODERATE / LOW risk bands — for use in downstream
    display filtering. Filters to valid coastal Louisiana wetland pixels
    before sampling.

    Run after 07_spatial_prediction.py, before display_wall.py.

INPUT:
    results/wall_to_wall_predictions.csv    — from 07_spatial_prediction.py

OUTPUT:
    results/wall_to_wall_sample.csv         — stratified sample (~15,000 rows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
import numpy as np

df = pd.read_csv('results/wall_to_wall_predictions.csv',
                  usecols=['lat','lon','loss_probability',
                           'carbon_predicted','carbon_uncertainty',
                           'marsh_type','risk_level','high_uncertainty'])
 
 # Filter to coastal Louisiana wetlands only
df = df[
     (df['lat'] >= 28.9) & (df['lat'] <= 30.2) &
     (df['lon'] >= -93.5) & (df['lon'] <= -88.8) &
     (df['marsh_type'].isin(['Fresh','Intermediate','Brackish','Saline'])) &
     (df['loss_probability'].notna()) &
     (df['carbon_predicted'].notna())
 ]
print(f'Filtered pixels: {len(df):,}')
print(df['risk_level'].value_counts())
 
 # Stratified sample
high = df[df['loss_probability']>0.6].sample(min(5000,len(df[df['loss_probability']>0.6])), random_state=42)
mod = df[(df['loss_probability']>0.3)&(df['loss_probability']<=0.6)].sample(min(5000,len(df[(df['loss_probability']>0.3)&(df['loss_probability']<=0.6)])), random_state=42)
low = df[df['loss_probability']<=0.3].sample(min(5000,len(df[df['loss_probability']<=0.3])), random_state=42)
sampled = pd.concat([high,mod,low]).reset_index(drop=True)
sampled.to_csv('results/wall_to_wall_sample.csv', index=False)
print(f'Saved {len(sampled)} points')
print(sampled['risk_level'].value_counts())
