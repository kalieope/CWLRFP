"""
get_high_risk.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE:
    Identifies high-risk wall-to-wall pixels that are more than 5 km
    from any CRMS monitoring station — these are the unmonitored parcels
    most in need of field verification or new station placement.

    Uses a KD-tree nearest-neighbor search against all CRMS station
    coordinates. Distance threshold is ~5 km (0.045 degrees × 111 km/deg).

    Run after display_wall.py.

INPUTS:
    results/wall_to_wall_display.csv        — from display_wall.py
    data/crms_all_stations_coords.csv       — station locations

OUTPUT:
    results/high_risk_unmonitored.csv       — high-risk pixels > 5 km from any station,
                                              sorted by loss_probability descending
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pandas as pd
import numpy as np

wt = pd.read_csv('results/wall_to_wall_display.csv')
stations = pd.read_csv('data/crms_all_stations_coords.csv')

# Find pixels far from any station (>5km = unmonitored)
from scipy.spatial import cKDTree
station_coords = stations[['Latitude','Longitude']].dropna().values
tree = cKDTree(station_coords)
pixel_coords = wt[['lat','lon']].values
distances, _ = tree.query(pixel_coords, k=1)
wt['dist_to_nearest_station_km'] = distances * 111  # rough km conversion
 
 # High risk unmonitored parcels
unmonitored = wt[
     (wt['dist_to_nearest_station_km'] > 5) &
     (wt['loss_probability'] > 0.6)
 ].sort_values('loss_probability', ascending=False)
 
print(f'High risk unmonitored pixels: {len(unmonitored)}')
print(unmonitored[['lat','lon','loss_probability',
                    'carbon_predicted','marsh_type',
                    'dist_to_nearest_station_km']].head(10))
unmonitored.to_csv('results/high_risk_unmonitored.csv', index=False)
