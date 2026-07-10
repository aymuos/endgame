# Simplified System Architecture: Causal-Informed Contextual Bandit Simulator

This document provides a simplified, reviewer-friendly system architecture block diagram and operational breakdown. It is designed to be easily incorporated into your thesis presentation or methodology overview.

---

### System Architecture Block Diagram

```mermaid
graph TD
    %% Step 1: Inputs
    A[("Observational Dataset<br/>(Courier GPS Logs & Weather API)")] --> B["1. Feature Engineering Pipeline<br/>(Static & Dynamic Feature Generation)"]
    
    %% Step 2: Decoupled Processing
    B -->|"Extract Causal Parents"| C["2. Causal Discovery (PC & CDNOD)<br/>(Isolates Invariant Causal Parents φ_bandit)"]
    B -->|"Generate State & Target Vectors"| D["3. Environment Surrogate (LightGBM)<br/>(Trains Predictor of Incremental Duration yt)"]
    
    %% Step 3: Interactive Simulation Loop
    subgraph Simulation_Loop [Causal Contextual Bandit Loop (Iterated until Batch is empty)]
        E["4. Contextual Bandit Policy<br/>(SquareCB / Thompson Sampling)"]
        F["5. Batch Simulation Environment<br/>(Tracks Courier State & Updates Coordinates)"]
        
        E -->|"a) Chooses Next Delivery Task (at)"| F
        F -->|"b) Returns Reward (Rt = -yt) & Updates Context"| E
    end
    
    C -->|"Restricted Context Mask"| E
    D -->|"Predicts Action Duration (yt)"| F
    
    %% Step 4: Output
    F --> G["6. Output: Optimal Permutation Sequence & Regret Curves"]

    %% Styling for review
    classDef input fill:#eceff1,stroke:#37474f,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef loop fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#fff8e1,stroke:#f57f17,stroke-width:2px;

    class A,B input;
    class C,D process;
    class E,F loop;
    class G output;
```

---

### Step-by-Step Walkthrough for Reviewers

1. **Step 1 & 2: Feature Engineering & Preprocessing**  
   Raw delivery logs and weather database records are parsed into clean static and dynamic feature buckets (Stage 0 to Stage 2).

2. **Step 2: Causal Parent Filtration**  
   Causal discovery algorithms (PC and CDNOD) are executed to identify the direct causal parents ($\phi_{\text{bandit}}$) of delivery duration. Non-causal features (confounders) are masked out to protect the policy from overfitting to spurious correlation patterns.

3. **Step 3: Environmental Surrogate Training**  
   A LightGBM regressor is trained offline using chronological splits of historical data. The regressor acts as a physics engine of the environment, predicting how many time ($y_t$) a delivery will take.

4. **Step 4 & 5: Sequential Decision Simulation Loop**  
   For every batch of orders, a simulation episode is run:
   - The **Contextual Bandit Policy** evaluates the remaining tasks using the *causally restricted context* and selects the next order ($a_t$).
   - The **Batch Environment** updates coordinates, queries the LightGBM oracle for the time penalty ($y_t$), and returns the reward ($R_t = -y_t$) to update the policy weights.
   - The loop runs until the batch is empty.

5. **Step 6: Optimal Permutation Output**  
   The simulator outputs the sequential routing permutation that minimizes travel time, alongside online regret curves comparing the policies (SquareCB, Thompson Sampling) to greedy and random baselines.
