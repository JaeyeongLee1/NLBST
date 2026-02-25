import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SpatialCoordinate:
    """Spatial coordinate with metadata."""
    coordinates: torch.Tensor
    location_id: str
    metadata: Optional[Dict] = None


@dataclass
class SpatiotemporalData:
    """Container for spatio-temporal observations."""
    observations: torch.Tensor          # [T, S] - all locations
    timestamps: torch.Tensor            # [T] - normalized to [0, 1]
    raw_timestamps: torch.Tensor        # [T] - original physical times
    spatial_locations: List[SpatialCoordinate]
    spatial_coords: torch.Tensor        # [S, 2] - normalized coordinates
    raw_coords: torch.Tensor            # [S, 2] - original coordinates
    measured_indices: torch.Tensor      # [S_m] - indices of measured locations
    unmeasured_indices: torch.Tensor    # [S_u] - indices of held-out locations
    missing_mask: Optional[torch.Tensor] = None  # [T, S] True = missing
    ground_truth: Optional[torch.Tensor] = None  # [T, S] noise-free values
    metadata: Optional[Dict] = None


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    generator: str  # 'advection', 'kolmogorov', or 'nonlocal_ide'
    
    # --- Simulation parameters (shared) ---
    T_final: float = 150.0
    save_every: float = 1.0
    spinup: float = 20.0       # for Kolmogorov / Nonlocal IDE
    
    
    # --- Nonlocal IDE-specific (D1, RQ4) ---
    alpha: float = 1.0         # Nonlocal coupling strength
    kappa: float = 0.01        # Local diffusion coefficient
    kernel_rank: int = 4       # R*: number of modes in ground truth kernel
    sim_dt: float = 0.001      # PDE integration time step
    domain_size: float = 1.0   # Spatial domain [0, L]^2
    nx: int = 64               # Grid resolution x
    ny: int = 64               # Grid resolution y
    
    # --- Sensor configuration ---
    n_sensors: int = 50
    sensor_pattern: str = 'mixed'
    noise_std: float = 0.1
    
    # --- Irregularity settings ---
    time_dropout_rate: float = 0.0
    missing_rate: float = 0.0
    block_missing: bool = False
    irregular_mode: str = 'sensor_wise'  # 'global_drop' or 'sensor_wise'
    
    # --- Spatial holdout ---
    holdout_ratio: float = 0.0
    holdout_mode: str = 'random'  # or 'region'
    holdout_region: Tuple[float, float, float, float] = (0.7, 1.0, 0.7, 1.0)
    
    # --- Model hyperparameters ---
    num_basis: int = 12
    basis_type: str = 'rbf'
    length_scale: float = 0.3
    process_noise_scale: float = 0.1
    initial_noise_scale: float = 0.5
    hidden_dim: int = 64
    ablation_mode: str = 'full'  # 'full', 'linear_only', 'neural_only', 'diagonal'
    
    # --- Training ---
    num_epochs: int = 200
    learning_rate: float = 0.005
    patience: int = 30
    train_ratio: float = 0.75
    
    # --- Evaluation ---
    forecast_horizon: int = 10
    window_size: int = 50
    stride: int = 5
    num_mc_samples: int = 100

    seed: int = 42


@dataclass  
class ExperimentResults:
    """Container for experiment results."""
    config: ExperimentConfig
    
    # --- Point metrics (measured locations) ---
    rmse_measured: float = np.nan
    mae_measured: float = np.nan
    
    # --- Point metrics (unmeasured locations) ---
    rmse_unmeasured: float = np.nan
    mae_unmeasured: float = np.nan
    rmse_unmeasured_noisefree: float = np.nan
    
    # --- Probabilistic metrics ---
    crps_measured: float = np.nan
    crps_unmeasured: float = np.nan
    nll_measured: float = np.nan
    nll_unmeasured: float = np.nan
    
    # --- Calibration metrics ---
    coverage_50: float = np.nan
    coverage_90: float = np.nan
    coverage_95: float = np.nan
    interval_score_90: float = np.nan
    calibration_error: float = np.nan
    sharpness: float = np.nan
    
    # --- Horizon-wise metrics ---
    rmse_by_horizon: Optional[Dict[int, float]] = None
    crps_by_horizon: Optional[Dict[int, float]] = None
    
    # --- Skill scores (vs persistence baseline) ---
    skill_rmse: float = np.nan
    skill_crps: float = np.nan
    
    # --- Timing ---
    train_time: float = np.nan
    eval_time: float = np.nan
    
    # --- Raw predictions for further analysis ---
    predictions: Optional[Dict] = None
    
    # --- Learned noise parameters (in RAW/physical units) ---
    sigma_obs: float = np.nan
    sigma_process: float = np.nan
    sigma_obs_normalized: float = np.nan
    sigma_process_normalized: float = np.nan
    