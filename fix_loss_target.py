import pandas as pd
import numpy as np

# Load timeseries
loss_ts = pd.read_csv('data/crms_loss_timeseries.csv')

# Create recent loss flag: lost land in 2015 or later
recent_loss = loss_ts[loss_ts['map_year'] >= 2015].groupby('station_id').agg(
    recent_land_loss=('land_loss', 'max')
).reset_index()

print('Recent loss distribution:')
print(recent_loss['recent_land_loss'].value_counts())
loss_rate = recent_loss['recent_land_loss'].mean()
print(f'Loss rate: {loss_rate:.1%}')

# Save for merging
recent_loss.to_csv('data/crms_recent_loss.csv', index=False)
print('Saved: data/crms_recent_loss.csv')