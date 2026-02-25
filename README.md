# NLBST: Non-Local Bayesian Spatio-Temporal Model

Code for the paper submitted to UAI 2026:  
**"Nonlocal Bayesian Modeling of Continuous Spatio-Temporal Dynamics"**

## Requirements

- Python 3.9+
- PyTorch 2.5+ (CUDA 11.8+)
- Pyro-PPL 1.9+
- NumPy, SciPy, matplotlib

Tested with: Python 3.9.23, PyTorch 2.5.1, Pyro 1.9.1.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Package Structure

```
NLBST/
├── config.py        # Data structures and experiment configuration
├── components.py    # Spatial basis functions, ODE dynamics, Kalman filter
├── model.py         # BST-NODE model (main contribution)
├── train.py         # Pyro SVI training loop
├── data.py          # Synthetic data generation (Advection, NonlocalIDE)
├── real_data.py     # EPA PM2.5 data loader
├── baselines.py     # Baseline implementations (Linear-DSTM, GRU-D, Latent ODE, FNO, GraFITi, APN)
├── evaluate.py      # Evaluation protocol and metrics computation
├── experiment.py    # End-to-end experiment runner
├── metrics.py       # RMSE, CRPS, coverage metrics
├── viz.py           # Visualization utilities
└── __init__.py
```

## Datasets

**Synthetic (D1–D2):** Generated automatically via `data.py`.

| Dataset | Description |
|---------|-------------|
| D1: Advection | Advection-diffusion PDE |
| D2: NonlocalIDE | Nonlocal integro-differential equation |

**Real-world (D3):** EPA PM2.5 air quality data.  
Downloaded automatically via `real_data.py`, or place pre-downloaded CSVs in `data/pm25/`.
