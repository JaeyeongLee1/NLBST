
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .config import ExperimentConfig, ExperimentResults
from .experiment import SyntheticExperimentRunner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456, 789, 1024, 2025, 3141, 4269, 5555, 6789]
RESULTS_DIR = Path(PROJECT_ROOT) / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_MODELS = [
    'NLBST', 'Linear-DSTM',
    'GRU-D', 'Latent-ODE', 'FNO',
    'GraFITi', 'APN',
]

COMMON = dict(
    time_dropout_rate=0.0,
    block_missing=False,
    num_mc_samples=100,
    sensor_pattern="mixed",
    noise_std=0.1,
    train_ratio=0.75,
    stride=5,
)


D1_DEFAULTS = dict(
    generator="advection",
    T_final=150.0,
    save_every=1.0,
    n_sensors=50,
    num_basis=16,
    num_epochs=200,
    forecast_horizon=10,
    window_size=50,
    **COMMON,
)

D2_DEFAULTS = dict(
    generator="nonlocal_ide",
    T_final=20.0,
    save_every=0.1,
    spinup=0.0,
    n_sensors=36,
    sensor_pattern="grid",
    noise_std=0.05,
    num_basis=16,
    num_epochs=300,
    forecast_horizon=10,
    window_size=50,
    stride=5,
    num_mc_samples=100,
    train_ratio=0.75,
    missing_rate=0.0,
    time_dropout_rate=0.0,
    block_missing=False,
    alpha=1.0,
    kappa=0.01,
    kernel_rank=4,
    sim_dt=0.001,
    domain_size=1.0,
    nx=64,
    ny=64,
)




def save_results(results, name):
    pkl_path = RESULTS_DIR / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved: {pkl_path}")


def load_results(name):
    pkl_path = RESULTS_DIR / f"{name}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    print(f"  ✗ Not found: {pkl_path}")
    return None


def print_model_summary(runs, model_names=None):
    if model_names is None:
        model_names = ALL_MODELS

    print(f"  {'Model':<20} {'RMSE':>14} {'CRPS':>14} {'Cov90':>10} {'NLL':>14}")
    print(f"  {'-'*74}")

    for model in model_names:
        rmses, crpss, covs, nlls = [], [], [], []
        for r in runs:
            if not isinstance(r, dict) or model not in r:
                continue
            res = r[model]
            if not isinstance(res, ExperimentResults):
                continue
            if not np.isnan(res.rmse_measured): rmses.append(res.rmse_measured)
            if not np.isnan(res.crps_measured): crpss.append(res.crps_measured)
            if not np.isnan(res.coverage_90):   covs.append(res.coverage_90)
            if not np.isnan(res.nll_measured):   nlls.append(res.nll_measured)

        if not rmses:
            continue
        print(f"  {model:<20} "
              f"{np.mean(rmses):.4f}±{np.std(rmses):.4f} "
              f"{np.mean(crpss):.4f}±{np.std(crpss):.4f} "
              f"{np.mean(covs):>9.1%} "
              f"{np.mean(nlls):.3f}±{np.std(nlls):.3f}")


def print_sweep_table(sweep_results, sweep_param, models=None):
    if sweep_results is None:
        print("  No results available.")
        return
    if models is None:
        models = ALL_MODELS

    param_values = sorted(sweep_results.keys())

    print(f"\n  {'Model':<20}", end="")
    for pv in param_values:
        print(f"  {sweep_param}={pv:<8}", end="")
    print()
    print(f"  {'-'*20}" + f"  {'-'*14}" * len(param_values))

    for model in models:
        has_data = False
        row = f"  {model:<20}"
        for pv in param_values:
            rmses = []
            for entry in sweep_results[pv]:
                if isinstance(entry, dict) and model in entry:
                    res = entry[model]
                    if isinstance(res, ExperimentResults) and not np.isnan(res.rmse_measured):
                        rmses.append(res.rmse_measured)
            if rmses:
                has_data = True
                row += f"  {np.mean(rmses):.4f}±{np.std(rmses):.3f}"
            else:
                row += f"  {'--':>14}"
        if has_data:
            print(row)


def print_info():
    print(f"Device: {DEVICE}")
    print(f"CUDA: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'}")
    print(f"Seeds: {SEEDS}")
    print(f"Results dir: {RESULTS_DIR}")