# HalfCheetah Reinforcement Learning

Implementation and comparison of three reinforcement learning algorithms — **Q-Learning**, **Deep Q-Network (DQN)**, and **Double DQN (DDQN)** — trained to control the `HalfCheetah-v5` environment from Gymnasium.

Since HalfCheetah has a continuous 6-dimensional action space and all three algorithms here are value-based methods designed for discrete actions, an `ActionDiscretizer` maps a finite set of discrete action IDs onto continuous torque vectors.

---

## Overview

The `HalfCheetah-v5` environment simulates a 2D cheetah-like robot that must learn to run forward as fast as possible by applying torques to its joints. The observation space is continuous (17-D state vector), and the native action space is continuous (6-D torque vector).

Since Q-Learning, DQN, and DDQN are all designed for **discrete** action spaces, this project discretizes both:

- **Actions** — via `ActionDiscretizer`, which builds a catalogue of discrete torque combinations (no torque, uniform torque across all joints, single-joint torque at several magnitudes).
- **States** (Q-Learning only) — via `StateDiscretizer`, which bins selected continuous observation features (torso height, torso angle, forward velocity) into a discrete tuple, since tabular Q-Learning cannot use raw continuous states as table keys.

DQN and DDQN use a neural network (`QNetwork`) to approximate Q-values directly from the raw continuous state, so they don't require state discretization.

---

## Algorithms

### 1. Q-Learning

Tabular, model-free, off-policy algorithm. Maintains a Q-table mapping (discretized state, action) pairs to expected future reward.

**Update rule:**
```bash
Q(s, a) <-- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]


Where:
- `alpha` = learning rate
- `gamma` = discount factor
- `r` = reward received after taking action `a` in state `s`
- `s'` = next state
- `max_a' Q(s', a')` = best estimated Q-value achievable from the next state
```

Action selection uses an epsilon-greedy policy, with epsilon decaying geometrically from an initial value to a minimum value over the course of training.

---

### 2. Deep Q-Network (DQN)

Replaces the Q-table with a neural network `Q(s, a; theta)` to handle the high-dimensional continuous state space. Uses two key stabilization techniques:

- **Experience Replay** — transitions `(s, a, r, s', done)` are stored in a replay buffer and sampled in random mini-batches, breaking the correlation between consecutive training samples.
- **Target Network** — a separate, periodically-updated copy of the network (`theta⁻`) is used to compute the target, preventing the network from chasing a constantly-moving target.

**Target value:**
```bash
y = r + gamma * max_a' Q(s', a'; theta_target) * (1 - done)
```

**Loss (Mean Squared Error):**
```bash
L(theta) = ( Q(s, a; theta) - y )^2
```


The online network's weights (`theta`) are updated via gradient descent on this loss. The target network's weights (`theta_target`) are synced to the online network every `target_update_frequency` training steps.

---

### 3. Double DQN (DDQN)

Standard DQN uses the **same network** (the target network) to both *select* and *evaluate* the best next action, via a single `max` operation. This tends to systematically **overestimate** Q-values, since `max` picks up on the network's own noise.

DDQN fixes this by **decoupling action selection from evaluation**:

- The **online network** selects which action is best for the next state.
- The **target network** evaluates that specific action's Q-value.

**Target value:**
```bash
a* = argmax_a' Q(s', a'; theta) <--- action chosen by ONLINE network
```
```bash
y = r + gamma * Q(s', a*; theta_target) * (1 - done) <--- evaluated by TARGET network
```



Compare this to DQN, where both selection and evaluation come from the same `max` over the target network:
```bash
y = r + gamma * max_a' Q(s', a'; theta_target) * (1 - done) <--- DQN
```

Everything else (replay buffer, network architecture, loss function) is identical to DQN — only the target computation changes.

---

## Setup & Installation

**Requirements:** 

Python 3.10+ (tested on 3.13)

1. Clone the repository:

```bash
git clone <your-repo-url>
cd rl
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies: `gymnasium[mujoco]`, `torch`, `numpy`, `pandas`, `matplotlib`

---

## How to Run

All commands should be run from the project root.

### Train

**Q-Learning**:

```bash
python experiments/q_learning/train_q_learning.py
```

Outputs:
- `q_table.pkl` — trained Q-table
- `results/q_learning/training_results.csv`
- `results/q_learning/episode_rewards.png`

**DQN**:

```bash
python experiments/dqn/train_dqn.py
```

Outputs:
- `results/dqn/dqn_model.pt` — trained network weights
- `results/dqn/training_results.csv`

**DDQN**:

```bash
python experiments/ddqn/train_ddqn.py
```

Outputs:
- `results/ddqn/ddqn_model.pt` — trained network weights
- `results/ddqn/training_results.csv`

> Episode counts can be adjusted via the `number_of_episodes` variable at the top of each training script.

### Run a Trained Agent (Demo)

Renders the environment with `render_mode="human"` so you can watch the trained agent run.

```bash
python run_half_cheetah.py          # Q-Learning
python run_half_cheetah_dqn.py      # DQN
python run_half_cheetah_ddqn.py     # DDQN
```

Each script loads its respective saved model/table and runs one greedy episode (`epsilon = 0`), printing total reward and step count at the end.

### Compare Algorithms

Plots episode return curves for all three algorithms on one chart:

```bash
python experiments/compare_algorithms.py
```
