# Safe RL for Joint Lighting and Blind Control

This repository contains the code for the paper:

**"Safe Reinforcement Learning for Joint Lighting and Blind Control Using Predictive Control Barrier Functions"**

---

This repo contains:
1. All the material, object, and IES files to create the described room in Radiance
2. All the controllers mentioned in the paper

There are two step functions:
- `fast_step.py` — Uses matrix multiplication (faster)
- `RAD_STEP.py` — Uses Radiance directly (slower)

---

## Requirements

- Python 3.8+
- NumPy, PyTorch, Matplotlib
- Gurobi
- Radiance (for full simulation mode)

Tested on Ubuntu.

## Repository Structure
```
safe-rl-building-control/
├── Env/                          # Radiance simulation environment
│   ├── D_mats/                   # Daylight matrices (D1-4.dmx)
│   ├── V_mats/                   # View matrices - workplane (V1-4.vmx)
│   ├── V_ver/                    # View matrices - vertical/glare (V1-4.vmx)
│   ├── Elec_light/               # Electric light definitions (.rad, .ies)
│   ├── Material_and_geometry/    # Scene geometry, materials, sensor points
│   ├── bl_state/                 # Blind state BSDF files (.xml)
│   ├── glazings/                 # Window definitions (.rad)
│   └── Wea_files/                # Weather data generation
│
├── src/                          # Source code
│   ├── agents/
│   │   ├── networks/             # actor.py, critic.py, memory.py
│   │   └── ppo.py                # PPOAgent1_epch
│   ├── steps/
│   │   ├── fast_step.py          # RADstep_fast (lookup-based)
│   │   └── RAD_STEP.py           # RADstep (full Radiance)
│   ├── utils/
│   │   └── sun_pos_day_of_year.py
│   ├── mpc_filter.py             # MPC-MILP safety filter
│   ├── train_RL+MPC.py           # Main training script
│   ├── RL_ONLY.py                # RL baseline
│   ├── MPC_ONLY.py               # MPC baseline
│   ├── RB_MEAN.py                # Rule-based baseline (mean)
│   └── RB_SENSOR.py              # Rule-based baseline (per-sensor)
│
└── README.md
```
