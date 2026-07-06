We are building a reward oracle for a bandits simulator . What is the oracle predicting?

Features ↓ Incremental delivery duration

Specifically

y
t
	​

={
sign
1
	​

−dispatch,
sign
t
	​

−sign
t−1
	​

,
	​

t=1
t>1
	​


This is the reward model.

The simulator asks

"If I deliver this order next, how much additional time will it cost?"

That is exactly what the oracle predicts.

Steps : Create the following features from data first . In the simulator these values would be updated in each pass , but for training lightgbm we must compute them individually for dataset creation , 

DYNAMIC_FEATURES = [

    "dist_from_current",
    "remaining_orders",
    "batch_progress",
    "elapsed_route_time",
    "last_duration",
    "current_hour",
    "cumulative_distance"
]

dist_from_current can be computed from consecutive logged deliveries.
remaining_orders = batch size − delivery index.
batch_progress = delivery index / batch size.
elapsed_route_time = cumulative sum of incremental durations.
current_hour = dispatch time + elapsed route time.
cumulative_distance = cumulative Euclidean distance between consecutive logged deliveries.

Read the features from Check 01_master_feature_creator and compute these values 

STATIC_FEATURES = [

    # Structural
    "pickup_destination_distance",
    "batch_size",
    "batch_rank_dispatch",
    "same_aoi_share_in_batch",
    "isolated_delivery",
    "distance_to_batch_centroid",
    "typecode_cb",

    # Operational
    "courier_eta_ewm",
    "gps_points",
    "speed_mean_15m",
    "speed_std_15m",
    "distance_travelled_15m",
    "coverage_ratio",
    "gps_gap_min",
    "idle_fraction",
    "is_trajectory_available",

    # Environment
    "WSI",
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "spatial_congestion_daily",
    "spatial_congestion_norm",

    # Time
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve"
]

final feature vector is phi = STATIC_FEATURES + DYNAMIC_FEATURES

TARGET = "incremental_duration"
