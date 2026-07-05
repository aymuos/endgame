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


