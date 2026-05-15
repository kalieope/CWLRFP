"""
display_wall.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Further reduces the stratified wall-to-wall sample to a small
    dashboard-ready display set: up to 1,000 points per risk band
    (HIGH / MODERATE / LOW), filtered to tight coastal wetland bounds
    and requiring carbon_predicted > 0.01 to exclude non-wetland pixels.

    This output is what the dashboard map loads as its wall-to-wall layer.

    Run after sample_wall.py, before get_high_risk.py.

INPUT:
    results/wall_to_wall_sample.csv     — from sample_wall.py

OUTPUT:
    results/wall_to_wall_display.csv    — up to 3,000 display points for dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
import numpy as np
df = pd.read_csv('results/wall_to_wall_sample.csv')
 
# Tighter coastal wetland bounds — exclude open ocean and upland
df = df[
     (df['lat'] >= 28.9) & (df['lat'] <= 30.0) &
     (df['lon'] >= -93.8) & (df['lon'] <= -88.8) &
     (df['carbon_predicted'] > 0.01)  # exclude non-wetland pixels
 ]
print(f'After filtering: {len(df):,}')
print(df['risk_level'].value_counts())
 
high = df[df['risk_level']=='HIGH'].sample(min(1000,len(df[df['risk_level']=='HIGH'])), random_state=42)
mod = df[df['risk_level']=='MODERATE'].sample(min(1000,len(df[df['risk_level']=='MODERATE'])), random_state=42)
low = df[df['risk_level']=='LOW'].sample(min(1000,len(df[df['risk_level']=='LOW'])), random_state=42)
small = pd.concat([high,mod,low])
small.to_csv('results/wall_to_wall_display.csv', index=False)
print(f'Saved {len(small)} display points')
