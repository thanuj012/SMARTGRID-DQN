# SMARTGRID-DQN

### DQN-Based Battery Energy Management for Renewable Microgrids

> A Deep Q-Network agent that learns to manage a battery in a solar microgrid — minimizing grid dependency over a 24-hour cycle — validated against a MATLAB Simulink physical model.

---

## Project Overview

SMARTGRID-DQN combines a **MATLAB Simulink physical model** with a **Python DQN reinforcement learning agent** to solve the battery dispatch problem in a PV-battery microgrid.

The agent learns when to:
- **Charge** the battery (store surplus solar energy)
- **Discharge** the battery (cover load when solar is unavailable)
- **Import** from the grid (last resort backup)

---

## Results

| Metric | Rule-Based Baseline | DQN Agent |
|---|---|---|
| Total Reward | -17.31 | **-9.31** |
| Grid Import | 4.65 kWh | **4.65 kWh** |
| Reward Improvement | — | **46% better** |

DQN achieves 46% better reward than the rule-based baseline by learning smarter SoC management — avoiding battery boundary violations that the reactive rule-based controller triggers.

---

## System Architecture

```
┌─────────────────────────────────────────┐
│           MATLAB / Simulink             │
│  PV Subsystem → Power Balance           │
│  Load Subsystem → Battery Subsystem     │
│  Grid Subsystem → Data Export (.mat)    │
└────────────────┬────────────────────────┘
                 │  PV & Load profiles
┌────────────────▼────────────────────────┐
│           Python DQN                    │
│  microgrid_env.py → train_rl.py         │
│  baseline.py → evaluate.py             │
└─────────────────────────────────────────┘
```

---

## Simulink Model

| Subsystem | Key Blocks | Output |
|---|---|---|
| PV | Sine Wave → Gain → Saturation | P_pv (0–9 kW) |
| Load | Sine Wave + Constant → Sum | P_load (0.5–3 kW) |
| Battery | MATLAB Fn + Unit Delay | SoC (0.2–0.9) |
| Grid | Saturation (import only) | P_grid (kW) |
| Power Balance | MATLAB Fn (KCL) | P_cmd, P_deficit |

**Simulation settings:** 86400s stop time, 3600s fixed step, ode3 solver

---

## Python Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one place |
| `microgrid_env.py` | Gym-style RL environment, loads Simulink data |
| `utils.py` | Plotting and logging helpers |
| `baseline.py` | Rule-based benchmark controller |
| `train_rl.py` | DQN neural network + training loop |
| `evaluate.py` | DQN vs baseline comparison |

---

## DQN Architecture

```
Input:  [SoC, P_pv, P_load, hour]  →  4 values
           ↓
        Linear(4 → 64) + ReLU
           ↓
        Linear(64 → 64) + ReLU
           ↓
        Linear(64 → 3)
           ↓
Output: [Q_idle, Q_charge, Q_discharge]  →  pick highest
```

**Key hyperparameters:**
- Episodes: 2000
- Epsilon decay: 1.0 → 0.01 (decay rate 0.9995)
- Replay buffer: 10,000
- Batch size: 64
- Gamma: 0.99
- Target network update: every 10 episodes

---

## Installation

```bash
pip install numpy scipy torch matplotlib
```

---

## Usage

**1. Export Simulink data** (MATLAB):
```matlab
save('microgrid_data.mat', 'PV_data', 'Load_data', 'SoC_data', 'Grid_data', 'time')
```

**2. Run baseline:**
```bash
python baseline.py
```

**3. Train DQN:**
```bash
python train_rl.py
```

**4. Evaluate:**
```bash
python evaluate.py
```

---

## Tech Stack

- **MATLAB Simulink** — Physical microgrid modeling
- **Python 3.14** — RL environment and agent
- **PyTorch** — DQN neural network
- **SciPy** — MATLAB .mat file loading
- **Matplotlib** — Visualization

---

## Author

**S. Thanuj**  
B.E. Electrical and Electronics Engineering  
SSN College of Engineering, Chennai  
[github.com/thanuj012](https://github.com/thanuj012)
