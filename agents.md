# Role: Contextual Bandit Simulator Architect
You are building a scientifically accurate multi-armed contextual bandit simulator for a master's/PhD thesis.

## Simulator Requirements
1. Every feature engineering step must map exactly to a deterministic pipeline.
2. Arm reward distributions MUST respect the structural equations dictated by the provided Causal DAG.
3. No code placeholders or truncated snippets. Output production-ready, object-oriented Python.
4. Implement strict tracking of counterfactual data (unobserved potential outcomes) for accuracy verification.


Analyze the city-specific DAG structures located in the folder `data\causal_graphs\pc_delivery`. Extract the structural equations, parent-child relationships, and confounding factors.

Read the existing feature engineering pipeline inside `00_master_feature_creator.ipynb`. Locate the final feature transformation functions and convert that exact data-cleaning and transformation logic into a clean, reusable Python class named `CityFeatureTransformer`.

Create a production-grade, object-oriented multi-city contextual bandit simulator based on the following local data:

1. Feature Logic: Use the transformation steps you extracted from `/notebooks/feature_engineering.ipynb`.
2. Causal Constraints: Implement structural causal equations that match the directional paths parsed from the `docs/dags/` PNG files. If a city has a unique edge (e.g., an extra confounder in City B), create a subclass or configuration dictionary for that specific city.

The simulator must:
- Accept a `city_name` string during initialization to load the correct causal graph parameters.
- Provide a `.step(arm_id)` method returning (engineered_context, observed_reward, true_regret).
- Save the final simulator implementation directly to a new file named `src/simulator/causal_bandit_env.py`.



Simulator features that remains :

A. Static (never change) : 
[ batch_size , pickup_destination_distance , typecode, weather , Spatial congestion index , GPS statistics , POI category ]
B. Dynamic (recomputed every decision) [ current_position → candidate distance , remaining_orders , cumulative_distance
elapsed_route_time , current_hour , batch_progress ]


Simulator state : 

State

current_x
current_y

last_duration

elapsed_route_time

current_hour

remaining_orders

batch_progress

cumulative_distance

--- 
candidate specific dynamic features: 
dist_from_current , remaining_haul_distance (recomputed) , distance_to_remaining_centroid (optional)
---
Static Dataset features : 
batch_size , pickup_destination_distance , GPS features ,Weather , Congestion , POI , Courier history



Utilising batch features : 

Static batch descriptors

Created once.

batch_size , same_aoi_share , isolated_delivery , distance_to_batch_centroid

--- 

Dynamic batch descriptors

Created by simulator.

remaining_orders , remaining_centroid , batch_progress , remaining_compactness , remaining_AOI_diversity


