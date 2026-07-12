# Spatio-Temporal Feature Engineering & System Mapping

This document lists the features engineered in `01_master_feature_creator.ipynb` at both the **Delivery-Level** (Order-Level) and the **Batch-Level**. It outlines their mathematical formulations, creation logic, and roles in both causal discovery (PC, CDNOD, PCMCI+) and the online contextual bandit simulator.

---

## 1. Delivery-Level (Order-Level) Features

These features capture the characteristics of individual delivery tasks. In the bandit environment, they act either as **arm-level** features (varying per candidate order) or **context-level** features (shared across the batch step).

| Feature Name | Mathematical Notation & Creation Logic | Used for Causal? | Used for Bandits? |
| :--- | :--- | :--- | :--- |
| **`pickup_destination_distance`** | Straight-line Euclidean distance on affine planar space:<br>$$d_{\text{pickup\_dest}} = \sqrt{(x_{\text{receipt}} - x_{\text{poi}})^2 + (y_{\text{receipt}} - y_{\text{poi}})^2}$$ | **Yes**<br>Identified as a direct parent of delivery duration (`eta_mins`) across cities. | **Yes**<br>Key arm-level feature representing the base distance of the task. |
| **`batch_size`** | Count of orders grouped into a single dispatch batch $B$:<br>$$N_{\text{batch}} = \vert B \vert$$ | **Yes**<br>Acts as a structural confounder; influences routing choices and spatial complexity. | **Yes**<br>Static context feature indicating total batch size. |
| **`batch_rank_dispatch`** | 0-indexed rank of the order sorted by receipt time and order ID:<br>$$r_{\text{dispatch}} = \text{Rank}_{\text{receipt\_time}, \text{order\_id}}(o_i \in B)$$ | **Yes**<br>Identified as a causal parent of distance metrics in local DAGs. | **Yes**<br>Tracks decision steps (used to compute progress metrics). |
| **`distance_to_batch_centroid`** | Distance from POI to the cumulative dispatch centroid $(\bar{x}_r, \bar{y}_r)$:<br>$$\bar{x}_r = \frac{1}{r+1}\sum_{k=0}^{r} x_{\text{poi}, k}$$<br>$$d_{\text{centroid}} = \sqrt{(x_{\text{poi}} - \bar{x}_r)^2 + (y_{\text{poi}} - \bar{y}_r)^2}$$ | **Yes**<br>CDNOD displays high-stability edges linking this to spatial features. | **Yes**<br>Used as a static batch descriptor tracking dispersion at dispatch. |
| **`isolated_delivery`** | Binary indicator if an order is the sole delivery in its Area of Interest (AOI):<br>$$\mathbb{I}\left( \sum_{o \in B} \mathbb{I}(\text{aoi\_id}_o == \text{aoi\_id}_{\text{self}}) == 1 \right)$$ | **No**<br>(Auxiliary spatial constraint). | **Yes**<br>Static batch descriptor. |
| **`same_aoi_share_in_batch`** | Ratio of orders in the batch sharing the same AOI category:<br>$$\text{Share}_{\text{aoi}} = \frac{1}{N_{\text{batch}}} \sum_{o \in B} \mathbb{I}(\text{aoi\_id}_o == \text{aoi\_id}_{\text{self}})$$ | **No**<br>(Auxiliary spatial constraint). | **Yes**<br>Static batch descriptor. |
| **`last_delivery_duration_clipped`** | Delivery duration from previous sign-off within the batch, capped at the 99th percentile $q_{0.99}$:<br>$$\Delta t_i = t_{\text{sign}, i} - t_{\text{sign}, i-1}$$<br>$$\Delta t_i^{\text{clipped}} = \min(\Delta t_i, q_{0.99})$$ | **Yes**<br>Used in PCMCI+ to model sequence transition durations. | **No**<br>(Replaced in the simulator by the dynamic `last_duration` state). |
| **`is_first_in_batch`** | Binary flag identifying if the delivery is the first task of a batch sequence:<br>$$\mathbb{I}(\Delta t_i == 0.0)$$ | **No**<br>(Environment flag). | **Yes**<br>Tells the bandit if it is at the initial step of a batch. |
| **`courier_eta_ewm`** | Exponentially Weighted Moving Average of courier's historical durations (half-life $\lambda=10$):<br>$$s_t = \beta \cdot y_{t-1} + (1-\beta) \cdot s_{t-1}$$ | **Yes**<br>Direct causal parent of `eta_mins` ($1.00$ stability in CDNOD). | **Yes**<br>Represents the courier's efficiency history context. |
| **`remaining_haul_distance`** | Sum of remaining task distances along the dispatch order sequence:<br>$$D_{\text{remaining}, i} = \sum_{k=i}^{N_{\text{batch}}-1} d_{\text{pickup\_dest}, k}$$ | **No**<br>(Correlated helper feature). | **Yes**<br>Recomputed dynamically as a candidate decision metric. |
| **`speed_mean_15m`** | Courier mean velocity (filtered by outlier ceiling $s_{\text{cap}}$) in the 15m prior to batch receipt:<br>$$v_{\text{mean}} = \frac{1}{\vert S_{\text{valid}} \vert} \sum_{s \in S_{\text{valid}}} \frac{d_s}{\Delta t_s}$$ | **Yes**<br>Found to causally influence initial travel time in local DAGs. | **Yes**<br>Represents the courier's immediate baseline momentum. |
| **`is_trajectory_available`** | Binary flag representing if a courier has active GPS pings recorded:<br>$$\mathbb{I}(\text{speed\_mean\_15m} \neq \text{Null})$$ | **No**<br>(Data quality indicator). | **Yes**<br>Dynamic binary indicator context. |
| **`hour_sin` / `hour_cos`** | Trigonometric mapping of receipt hour $h \in [0, 23]$:<br>$$\sin\left(\frac{2\pi h}{24}\right), \quad \cos\left(\frac{2\pi h}{24}\right)$$ | **Yes**<br>Exogenous context nodes accounting for diurnal cycles. | **Yes**<br>Advances dynamically in simulation as the route clock moves. |
| **`is_weekend` / `is_holiday` / `is_holiday_eve`** | Binary flags mapping the receipt date to the calendar:<br>$$\mathbb{I}(\text{wkday} \geq 6), \; \mathbb{I}(\text{date} \in H), \; \mathbb{I}(\text{date} = E)$$ | **Yes**<br>Exogenous context variables. | **Yes**<br>Static context variables. |
| **`WSI`** | Weather Severity Index Min-Max normalized:<br>$$W = w_1 P + w_2 S_w + w_3 V$$<br>$$\text{WSI} = (W - W_{\min})/(W_{\max} - W_{\min})$$ | **Yes**<br>Exogenous confounding node affecting delivery speed and ETA. | **Yes**<br>Static environment context. |
| **`spatial_congestion_norm`** | standard normal scaling of order count in $500\text{m}$ grid cell $G$ during hour $T$:<br>$$\text{SCI}_{\text{norm}} = (C_i - \mu_C)/\sigma_C$$ | **Yes**<br>Direct local confounder representing traffic/demand density. | **Yes**<br>Static spatial context. |
| **`typecode_cb`** | Frequency encoding representing product/delivery category popularities:<br>$$\text{Freq}(\text{typecode})$$ | **Yes**<br>Causal parent in local graphs. | **Yes**<br>Static arm descriptor. |
| **`eta_mins`** | Elapsed delivery duration in minutes:<br>$$\text{ETA} = (t_{\text{sign}} - t_{\text{receipt}}) / 60$$ | **Yes**<br>Target outcome variable $Y$ for all graphs. | **Yes**<br>Defines the negative observed reward ($R = - \text{eta\_mins}$). |

---

## 2. Batch-Level Aggregated Features

These features are aggregated at the unique `batch_id` level. They are primarily used in **PCMCI+** and **CDNOD** to verify system characteristics (e.g., confirming that batch transitions behave as a memoryless Markov process).

| Feature Name | Mathematical Notation & Creation Logic | Used for Causal? | Used for Bandits? |
| :--- | :--- | :--- | :--- |
| **`batch_aoi_entropy`** | Shannon Entropy of Area of Interest (AOI) categories within batch $B$:<br>$$H_{\text{AOI}} = -\sum_{i=1}^{K} p_i \log_2(p_i)$$ where $p_i = \frac{n_i}{N_{\text{batch}}}$ | **Yes**<br>Used as a batch-level exogenous parent of dispatch complexity. | **Yes**<br>Used to check batch transition constraints. |
| **`batch_grid_cells_unique`** | Count of unique spatial grid cells spanned by orders in the batch:<br>$$U_{\text{grid}} = \left\vert \bigcup_{o \in B} \{ G_o(x, y) \} \right\vert$$ | **Yes**<br>Causal parent of average travel time in batch-level SCMs. | **Yes**<br>Recomputed during simulation to track remaining route compactness. |
| **`speed_mean_15m_mean`** | Average speed across the batch window prior to start:<br>$$\frac{1}{N_{\text{batch}}} \sum_{o \in B} v_{\text{mean}, o}$$ | **Yes**<br>Causal node in batch-level models. | **Yes**<br>Acts as baseline speed context. |
| **`eta_mean`** | Mean delivery time (receipt to sign) for all orders in the batch:<br>$$\text{Mean}_{o \in B}(\text{eta\_mins}_o)$$ | **Yes**<br>The primary target outcome node $Y$ in batch-level causal graphs. | **Yes**<br>Used to evaluate batch-level policies. |
| **`is_singleton_batch`** | Binary flag representing single-order dispatches:<br>$$\mathbb{I}(N_{\text{batch}} == 1)$$ | **Yes**<br>Controls for dispatch policy transitions. | **No**<br>(Single-order batches are trivial to sequence). |
