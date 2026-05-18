#  Deep Reinforcement Learning for Supply Chain Inventory Optimization

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![RL Framework](https://img.shields.io/badge/Gymnasium-0.29%2B-green.svg?logo=gymnasium&logoColor=white)](https://gymnasium.farama.org/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Data Processing](https://img.shields.io/badge/pandas-2.0%2B-darkblue.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

Supply chain inventory management is a classic high-dimensional optimization challenge. Balancing holding costs, transaction/ordering expenses, and stockout penalties requires agents to make complex, multi-period decisions under highly uncertain demand. 

This repository presents a **unified Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) framework** designed to optimize inventory replenishment decisions. Utilizing real-world sales and delivery data from the **DataCo Smart Supply Chain Dataset**, we implement and evaluate a diverse set of agents ranging from tabular techniques (Q-Learning, TD(0)) to deep policy gradient methods (PPO, DQN) and sequence-modeling architectures (LSTM).

---

##  Key Highlights & Implementations

*   **Production & Exploratory Workspaces**: Contains both clean, production-ready models and detailed exploratory scripts incorporating theoretical convergence comparisons.
*   **Dual Environment Architectures**: Includes a standard immediate-delivery environment and an advanced 2-day lead time simulation with queue-based order tracking.
*   **Integrated Q-Learning & TD(0)**: Unique script combining action-value learning ($Q(s, a)$) with a Temporal Difference ($TD(0)$) state-value function ($V(s)$) estimator for deep validation.
*   **High-Fidelity Baselines**: Features a deterministic Heuristic Policy, Random Exploration Baselines, and a supervised Linear Regression model forecasting customer sales.
*   **Robust Deep Learning (DRL)**: State-of-the-art DRL implementations including custom Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN) designed for continuous observations.

---

##  Repository Structure

The project is structured into modular sections, keeping a clean division between core production scripts and exploratory work.

```directory
Supply_chain_management_DRL/
│
├── Code files/                          # Core Production Environment & Models
│   └── supply_chain_management/
│       ├── simple_supply_chain_qlearning.py   # Production Q-learning implementation
│       ├── supply_chain_with_leadtime.py      # Q-learning with 2-day lead-time model
│       ├── supply-chain-inventory-with-lstm.ipynb  # LSTM forecasting/policy notebook
│       ├── supply-chain-inventory-with-ppo.ipynb   # PPO-based deep RL notebook
│       ├── supply_chain_dqn_kaggle.ipynb      # DQN-based deep RL notebook
│       └── DataCoSupplyChainDataset.csv        # Real-world supply chain dataset (Git LFS)
│
├── Exploratory_Work(Code_files)/         # Exploratory & Diagnostics Environment
│   └── Exploratory_Work(Code_files)/Code/
│       ├── simple_supply_chain_qlearning.py   # Integrated Q-learning + TD(0) diagnostics
│       ├── train_ppo.py                       # Modular PPO benchmark training script
│       ├── train_model.py                     # Supervised Linear Regression sales model
│       └── project.ipynb                      # Exploratory data analysis and model prototyping
│
├── Report.pdf                           # Detailed technical report of RL/DRL implementations
├── Demo Screenshots.pdf                 # Performance visualization and convergence plots
├── RL_PPT.pptx                         # Technical presentation slides for the project
└── README.md                            # Repository documentation (This file)
```

---

##  Environment Formulation

The inventory management system is modeled as a Markov Decision Process (MDP) with the following components:

### 1. State / Observation Space
To maintain tabular efficiency, observations are discretized:
*   **Inventory Level (3 levels)**:
    *   `0: Low` ($< 200$ units) — Danger zone, prone to stockouts.
    *   `1: Medium` ($200$ to $500$ units) — Standard operating buffer.
    *   `2: High` ($\ge 500$ units) — Safe level, higher holding cost risk.
*   **Recent Demand Trend (3 levels)**: Average demand over the last 7 days:
    *   `0: Low` ($< 200$ units/day)
    *   `1: Medium` ($200$ to $350$ units/day)
    *   `2: High` ($\ge 350$ units/day)
*   **Day of the Week (7 values)**: `0` (Monday) through `6` (Sunday) to capture weekly demand periodicity.
*   **Pending Order Flag (2 values - Lead-Time Variant Only)**:
    *   `0`: No incoming order scheduled.
    *   `1`: Incoming order scheduled to arrive in the next 2 days.

**Total States**:
*   *Standard MDP*: $3 \times 3 \times 7 = 63$ discrete states.
*   *Lead-Time MDP*: $3 \times 3 \times 7 \times 2 = 126$ discrete states.

### 2. Action Space
A discrete action space of **4 replenishment decisions**:
*   `0`: Do not order ($0$ units).
*   `1`: Order $200$ units.
*   `2`: Order $400$ units.
*   `3`: Order $600$ units.

### 3. Transition Dynamics
*   **Standard Environment**: Orders arrive instantly at the start of the step.
*   **Lead-Time Environment**: Orders take a fixed lead-time ($\tau = 2$ days) to arrive. Pending orders are managed via a queue structure.

### 4. Reward Function
The daily reward (net profit) is modeled as:
$$R_t = \text{Revenue}_t - \text{Holding Cost}_t - \text{Stockout Penalty}_t - \text{Order Cost}_t$$

Where:
*   $$\text{Revenue}_t = \min(\text{Inventory}_t + \text{Arriving}_t, \text{Demand}_t) \times \bar{P}$$ (where $\bar{P}$ is the average product price)
*   $$\text{Holding Cost}_t = \max(0, \text{Inventory}_{t+1}) \times \$2.0$$ per unit/day
*   $$\text{Stockout Penalty}_t = \max(0, \text{Demand}_t - (\text{Inventory}_t + \text{Arriving}_t)) \times \$15.0$$ per unit short
*   $$\text{Order Cost}_t = \$100.0 \text{ (fixed setup)} + \text{Order Qty} \times (0.6 \times \bar{P}) \text{ (variable purchasing)}$$

---

##  Implemented Models & Mathematical Details

### 1. Tabular Q-Learning
Learns the action-value function $Q(s, a)$, representing the expected cumulative discounted reward of taking action $a$ in state $s$ and thereafter following the optimal policy:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 2. Temporal Difference learning - TD(0)
Learns the state-value function $V(s)$ directly under the exploration policy, providing baseline state valuation:

$$V(s) \leftarrow V(s) + \alpha \left[ R_{t} + \gamma V(s') - V(s) \right]$$

*   *Theoretical Convergence Check*: At optimality, the relation $V(s) \approx \max_a Q(s, a)$ holds under the greedy policy. The exploratory script validates this by training both algorithms concurrently.

### 3. Proximal Policy Optimization (PPO)
An actor-critic model learning a parameterized policy $\pi_\theta(a|s)$. It optimizes a clipped surrogate objective to avoid destructively large policy updates:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

Where the probability ratio is $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$, and $\hat{A}_t$ is the Generalized Advantage Estimator (GAE).

### 4. Deep Q-Networks (DQN)
Applies a deep neural network $Q(s, a; \theta)$ to approximate the action values over high-dimensional observations. Features an experience replay buffer and a target network $Q(s, a; \theta^-)$ to stabilize convergence.

### 5. Supervised Linear Regression
Trained in `train_model.py` to establish a classical ML baseline, predicting `Sales per customer` based on order demographics, category features, and shipping details:

$$Y = \beta_0 + \sum_{i=1}^{k} \beta_i X_i + \epsilon$$

---

##  Getting Started & Execution

### 1. Setup Environment
Clone the repository and install the dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn gymnasium torch stable-baselines3 joblib
```

### 2. Extract Data
Ensure `DataCoSupplyChainDataset.csv` is extracted and placed in both the `Code files/supply_chain_management/` and `Exploratory_Work(Code_files)/Exploratory_Work(Code_files)/Code/` folders.

### 3. Running Production Models
*   **Train Q-Learning (Immediate Delivery)**:
    ```bash
    python "Code files/supply_chain_management/simple_supply_chain_qlearning.py"
    ```
*   **Train Q-Learning (With 2-Day Lead Time)**:
    ```bash
    python "Code files/supply_chain_management/supply_chain_with_leadtime.py"
    ```

### 4. Running Exploratory & Diagnostics Code
*   **Train Q-Learning + TD(0) Diagnostic Runner**:
    ```bash
    python "Exploratory_Work(Code_files)/Exploratory_Work(Code_files)/Code/simple_supply_chain_qlearning.py"
    ```
    This script runs the joint tabular agent, outputs exploration/exploitation metrics, and saves state-value convergence plots.
    
*   **Train Exploratory PPO Agent**:
    ```bash
    python "Exploratory_Work(Code_files)/Exploratory_Work(Code_files)/Code/train_ppo.py"
    ```
    Benchmarks PPO against deterministic heuristics and random baselines, generating standard learning curves and bar chart comparisons.
    
*   **Train Supervised Sales Predictor**:
    ```bash
    python "Exploratory_Work(Code_files)/Exploratory_Work(Code_files)/Code/train_model.py"
    ```
    Outputs `model.joblib`, `encoders.joblib`, and `model_diagnostics.json` diagnostics including coefficients and residuals.

---

##  Performance Benchmarks & Convergence

### Q-Learning & TD(0) Training Progress
During training, epsilon decays from $1.0$ down to $0.01$ over $1000$ episodes. The agent rapidly transitions from exploration to exploitation, causing the average reward (profit) to climb and stabilize.

| Parameter | Standard Environment Value | Lead-Time Environment Value |
| :--- | :--- | :--- |
| **Observation States** | 63 | 126 |
| **Epsilon Start ($\epsilon$)** | 1.000 | 1.000 |
| **Epsilon Min ($\epsilon_{min}$)**| 0.010 | 0.010 |
| **Epsilon Decay Rate** | 0.995 | 0.995 |
| **Trained Policy Profit** | **Significant Improvement** | **Strategic Buffering** |
| **Improvement vs Random** | **> 120%** | **> 85%** |

### Key Insight: The Challenge of Lead Time
With immediate delivery, the agent behaves reactively: ordering only when inventory falls below demand. However, under a **2-day lead-time**, the agent is forced to *order ahead*. Including the `pending_orders` boolean flag in the state representation prevents the agent from placing redundant orders and over-accumulating inventory, demonstrating why proper MDP formulation is essential for supply chain dynamics.

---

##  References & Reports
*   For complete research details, see [Report.pdf](Report.pdf).
*   For slide decks and presentation materials, see [RL_PPT.pptx](RL_PPT.pptx).
*   For visual convergence progress, see [Demo Screenshots.pdf](Demo Screenshots.pdf).

---
*Developed as part of Deep Reinforcement Learning for Supply Chain Inventory Optimization.*
