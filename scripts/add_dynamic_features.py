"""Add dynamic features to delivery feature parquet files.
Outputs saved next to inputs with suffix `_with_dynamic.parquet`.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
FILES = list(DATA_DIR.glob('delivery_features_*.parquet'))

for p in FILES:
    print(f'Processing: {p}')
    df = pd.read_parquet(p)

    # ensure correct sorting within batch
    df = df.sort_values(['batch_id', 'batch_rank_dispatch'])

    # Ensure numeric types
    df['batch_size'] = df['batch_size'].astype('Int64')
    df['batch_rank_dispatch'] = df['batch_rank_dispatch'].astype('Int64')

    # delivery index (assume batch_rank_dispatch is 0-based; convert to index)
    df['delivery_index'] = df['batch_rank_dispatch'].astype('Int64')

    # remaining orders and progress
    df['remaining_orders'] = df['batch_size'] - (df['delivery_index'] + 1)
    df['batch_progress'] = (df['delivery_index'] + 1) / df['batch_size']

    # parse times
    if 'sign_time' in df.columns:
        df['sign_time'] = pd.to_datetime(df['sign_time'])
    else:
        df['sign_time'] = pd.NaT

    # last_duration in minutes (difference between consecutive sign_time within batch)
    df['last_duration'] = df.groupby('batch_id')['sign_time'].diff().dt.total_seconds() / 60.0
    df['last_duration'] = df['last_duration'].fillna(0.0)

    # elapsed route time: cumulative sum of last_duration within batch
    df['elapsed_route_time'] = df.groupby('batch_id')['last_duration'].cumsum()

    # current hour: dispatch time + elapsed_route_time
    dispatch_col = 'datetime' if 'datetime' in df.columns else 'receipt_time'
    df[dispatch_col] = pd.to_datetime(df[dispatch_col])
    df['current_timestamp'] = df[dispatch_col] + pd.to_timedelta(df['elapsed_route_time'], unit='m')
    df['current_hour'] = df['current_timestamp'].dt.hour + df['current_timestamp'].dt.minute / 60.0

    # cumulative_distance: cumulative Euclidean distance between consecutive poi points
    for col in ['poi_lng', 'poi_lat', 'last_x', 'last_y', 'pickup_destination_distance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['prev_poi_lng'] = df.groupby('batch_id')['poi_lng'].shift(1)
    df['prev_poi_lat'] = df.groupby('batch_id')['poi_lat'].shift(1)
    df['leg_dist'] = ((df['poi_lng'] - df['prev_poi_lng'])**2 + (df['poi_lat'] - df['prev_poi_lat'])**2)**0.5
    df['leg_dist'] = df['leg_dist'].fillna(0.0)
    df['cumulative_distance'] = df.groupby('batch_id')['leg_dist'].cumsum()

    # dist_from_current: distance from last_x,last_y to poi
    if 'last_x' in df.columns and 'last_y' in df.columns:
        df['dist_from_current'] = ((df['poi_lng'] - df['last_x'])**2 + (df['poi_lat'] - df['last_y'])**2)**0.5
        # fallback to pickup_destination_distance if dist_from_current is NaN
        if 'pickup_destination_distance' in df.columns:
            df['dist_from_current'] = df['dist_from_current'].fillna(df['pickup_destination_distance'])
    else:
        df['dist_from_current'] = pd.NA

    # select and keep original columns plus new dynamic features
    dynamic_cols = ['dist_from_current', 'remaining_orders', 'batch_progress', 'elapsed_route_time', 'last_duration', 'current_hour', 'cumulative_distance']
    for c in dynamic_cols:
        if c not in df.columns:
            df[c] = pd.NA

    out_path = p.with_name(p.stem + '_with_dynamic.parquet')
    df.to_parquet(out_path, index=False)
    print(f'  Wrote: {out_path} | rows: {len(df)}')

print('Done.')
