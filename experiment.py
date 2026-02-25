"""
nlbst/experiment.py

Experiment runner for synthetic datasets.
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from scipy.stats import norm

import pyro

from .config import ExperimentConfig, ExperimentResults, SpatialCoordinate, SpatiotemporalData
from .data import (
    AdvectionDiffusionSimulator, NonlocalIDESimulator,
    generate_sensor_locations, normalize_times, create_missing_mask,
    split_measured_unmeasured, apply_time_dropout, apply_sensorwise_irregular
)
from .model import BayesianSTNeuralODE_Kalman, VariationalKalmanGuide
from .baselines import (
    LinearDSTMBaseline, GRUDBaseline, LatentODEBaseline,
    FNOBaseline, GraFITiBaseline, APNBaseline
)
from .train import train_nlbst
from .evaluate import evaluate_nlbst
from .metrics import ProbabilisticMetrics


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pyro.set_rng_seed(seed)


def _get_eval_target(test_data: SpatiotemporalData, t: int, indices):
    if test_data.ground_truth is not None:
        return test_data.ground_truth[t, indices].cpu().numpy()
    else:
        return test_data.observations[t, indices].cpu().numpy()


class SyntheticExperimentRunner:

    
    def __init__(self, device: str = "cuda", results_dir: str = "./results"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment Runner initialized on {self.device}")
    
    def generate_dataset(self, config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sim = None
        
        if config.generator == 'advection':
            sim = AdvectionDiffusionSimulator(seed=config.seed)
            snapshots, times = sim.simulate(
                T_final=config.T_final,
                save_every=config.save_every,
                verbose=False
            )
        elif config.generator == 'nonlocal_ide':
            sim = NonlocalIDESimulator(
                nx=config.nx,
                ny=config.ny,
                domain_size=config.domain_size,
                alpha=config.alpha,
                kappa=config.kappa,
                kernel_rank=config.kernel_rank,
                dt=config.sim_dt,
                seed=config.seed
            )
            snapshots, times = sim.simulate(
                T_final=config.T_final,
                save_every=config.save_every,
                spinup=config.spinup,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown generator: {config.generator}")
        
        sensor_coords = generate_sensor_locations(
            config.n_sensors,
            pattern=config.sensor_pattern,
            seed=config.seed
        )
        
        observations, ground_truth = sim.sample_at_sensors(
            snapshots, sensor_coords,
            noise_std=config.noise_std,
            seed=config.seed
        )
        
        # Store simulator reference for later kernel extraction (RQ4)
        self._last_simulator = sim
        
        return observations, ground_truth, times, sensor_coords, snapshots
    
    def prepare_data(self, observations: np.ndarray,
                     ground_truth: np.ndarray,
                     times: np.ndarray,
                     sensor_coords: np.ndarray,
                     config: ExperimentConfig) -> Tuple[SpatiotemporalData, SpatiotemporalData]:
        """
        Prepare train/test SpatiotemporalData objects with all preprocessing.
        """
        T, S = observations.shape
        
        # Track sensor-wise irregular mask (if applicable)
        sensor_irregular_mask = None
        
        # Apply irregularity
        if config.time_dropout_rate > 0:
            if config.irregular_mode == 'global_drop':
                observations, ground_truth, times, _ = apply_time_dropout(
                    observations, ground_truth, times,
                    config.time_dropout_rate,
                    seed=config.seed
                )
            elif config.irregular_mode == 'sensor_wise':
                observations, ground_truth, times, sensor_irregular_mask = apply_sensorwise_irregular(
                    observations, ground_truth, times,
                    keep_rate=1.0 - config.time_dropout_rate,
                    seed=config.seed
                )
        
        T, S = observations.shape
        
        # Spatial holdout
        measured_idx, unmeasured_idx = split_measured_unmeasured(
            S,
            holdout_ratio=config.holdout_ratio,
            seed=config.seed,
            mode=config.holdout_mode,
            coords=sensor_coords,
            region=config.holdout_region
        )
        
        # Train/test split FIRST (before applying missing)
        T_train = int(T * config.train_ratio)
        
        # ===== TRAIN: Apply missing mask =====
        train_missing_mask = create_missing_mask(
            T_train, S,
            missing_rate=config.missing_rate,
            block_missing=config.block_missing,
            seed=config.seed
        )
        
        # Combine with sensor-wise irregular mask if applicable
        if sensor_irregular_mask is not None:
            train_missing_mask = np.logical_or(
                train_missing_mask, 
                sensor_irregular_mask[:T_train]
            )
        
        # Enforce spatial holdout in training (prevent data leakage)
        if len(unmeasured_idx) > 0:
            train_missing_mask[:, unmeasured_idx] = True
        
        # ===== TEST: No missing for clean ground truth evaluation =====
        test_missing_mask = np.zeros((T - T_train, S), dtype=bool)
        
        # Only spatial holdout for test (unmeasured locations still hidden)
        if len(unmeasured_idx) > 0:
            test_missing_mask[:, unmeasured_idx] = True
        
        # Apply sensor-wise irregular only to context portion of test
        if sensor_irregular_mask is not None and config.time_dropout_rate > 0:
            context_portion = min(config.window_size, T - T_train)
            test_missing_mask[:context_portion] = np.logical_or(
                test_missing_mask[:context_portion],
                sensor_irregular_mask[T_train:T_train + context_portion]
            )
        
        # Convert to tensors
        obs_t = torch.tensor(observations, dtype=torch.float32)
        gt_t = torch.tensor(ground_truth, dtype=torch.float32)
        times_norm = torch.tensor(normalize_times(times), dtype=torch.float32)
        times_raw = torch.tensor(times, dtype=torch.float32)
        coords_t = torch.tensor(sensor_coords, dtype=torch.float32)
        train_mask_t = torch.tensor(train_missing_mask, dtype=torch.bool)
        test_mask_t = torch.tensor(test_missing_mask, dtype=torch.bool)
        measured_t = torch.tensor(measured_idx, dtype=torch.long)
        unmeasured_t = torch.tensor(unmeasured_idx, dtype=torch.long)
        
        spatial_locations = [
            SpatialCoordinate(coords_t[i], f"sensor_{i}")
            for i in range(S)
        ]
        
        train_data = SpatiotemporalData(
            observations=obs_t[:T_train],
            timestamps=times_norm[:T_train],
            raw_timestamps=times_raw[:T_train],
            spatial_locations=spatial_locations,
            spatial_coords=coords_t,
            raw_coords=coords_t,
            measured_indices=measured_t,
            unmeasured_indices=unmeasured_t,
            missing_mask=train_mask_t,
            ground_truth=gt_t[:T_train]
        )
        
        test_data = SpatiotemporalData(
            observations=obs_t[T_train:],
            timestamps=times_norm[T_train:],
            raw_timestamps=times_raw[T_train:],
            spatial_locations=spatial_locations,
            spatial_coords=coords_t,
            raw_coords=coords_t,
            measured_indices=measured_t,
            unmeasured_indices=unmeasured_t,
            missing_mask=test_mask_t,
            ground_truth=gt_t[T_train:]
        )
        
        return train_data, test_data
    
    def _create_model(self, config: ExperimentConfig, 
                      sensor_coords: np.ndarray,
                      train_data: SpatiotemporalData,
                      ablation_mode: Optional[str] = None) -> BayesianSTNeuralODE_Kalman:
        """Create NLBST model with proper configuration."""
        if ablation_mode is None:
            ablation_mode = config.ablation_mode
        model = BayesianSTNeuralODE_Kalman(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            num_basis=config.num_basis,
            basis_type=config.basis_type,
            length_scale=config.length_scale,
            process_noise_scale=config.process_noise_scale,
            initial_noise_scale=config.initial_noise_scale,
            hidden_dim=config.hidden_dim,
            ablation_mode=ablation_mode
        ).to(self.device)
        
        return model
    
    
    def run_single_experiment(self, config: ExperimentConfig,
                              verbose: bool = True,
                              ablation_mode: str = 'full',
                              skip_baselines: bool = False,
                              ) -> Dict[str, Any]:
        """
        Run a single experiment with NLBST and baselines.
        """
        set_seed(config.seed)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Experiment: {config.name}")
            if ablation_mode != 'full':
                print(f"Ablation mode: {ablation_mode}")
            print(f"{'='*70}")
        
        # Generate data
        observations, ground_truth, times, sensor_coords, snapshots = self.generate_dataset(config)
        
        # Prepare train/test data
        train_data, test_data = self.prepare_data(
            observations, ground_truth, times, sensor_coords, config
        )
        
        # Move to device
        def to_device(data):
            data.observations = data.observations.to(self.device)
            data.timestamps = data.timestamps.to(self.device)
            data.spatial_coords = data.spatial_coords.to(self.device)
            data.measured_indices = data.measured_indices.to(self.device)
            data.unmeasured_indices = data.unmeasured_indices.to(self.device)
            if data.missing_mask is not None:
                data.missing_mask = data.missing_mask.to(self.device)
            if data.ground_truth is not None:
                data.ground_truth = data.ground_truth.to(self.device)
        
        to_device(train_data)
        to_device(test_data)
        
        if verbose and train_data.missing_mask is not None:
            train_missing_rate = train_data.missing_mask[:, train_data.measured_indices].float().mean().item()
            print(f"Train missing rate (measured sensors): {train_missing_rate:.1%}")
            print(f"Test: evaluate against ground truth (all sensors)")
        
        results = {}
        
        # ===== NLBST =====
        if verbose:
            mode_label = f"NLBST ({ablation_mode})" if ablation_mode != 'full' else "NLBST"
            print(f"\n[{mode_label}] Training...")
        
        t0 = time.time()
        model = self._create_model(config, sensor_coords, train_data, ablation_mode)
        svi, losses, guide = train_nlbst(model, train_data, config, verbose=verbose)
        train_time = time.time() - t0
        
        if verbose:
            print(f"[NLBST] Evaluating...")
        
        t0 = time.time()
        bst_results = evaluate_nlbst(model, guide, test_data, train_data, config)
        bst_results.train_time = train_time
        bst_results.eval_time = time.time() - t0
        
        result_key = f'NLBST-{ablation_mode}' if ablation_mode != 'full' else 'NLBST'
        results[result_key] = bst_results
        
        if verbose:
            print(f"[{result_key}] RMSE(m): {bst_results.rmse_measured:.4f}, "
                  f"CRPS(m): {bst_results.crps_measured:.4f}, "
                  f"Coverage(90%): {bst_results.coverage_90:.2%}")
            if not np.isnan(bst_results.rmse_unmeasured):
                print(f"[{result_key}] RMSE(u): {bst_results.rmse_unmeasured:.4f}")
            if not np.isnan(bst_results.sigma_obs):
                print(f"[{result_key}] σ_obs: {bst_results.sigma_obs:.4f} (raw), "
                      f"σ_proc: {bst_results.sigma_process:.4f} (raw)")
        
        # ===== Visualization =====
        if verbose and bst_results.predictions is not None:
            try:
                self._visualize_bst_predictions(
                    bst_results, test_data, train_data, config.name
                )
            except Exception as e:
                print(f"[Visualization] Error: {e}")
        
        # ===== Baselines =====
        if not skip_baselines:
            results.update(self._run_baselines(
                sensor_coords, train_data, test_data, config, verbose
            ))
        
        return results
    
    def _run_baselines(self, sensor_coords, train_data, test_data, 
                       config, verbose) -> Dict[str, ExperimentResults]:
        """Run all 5 baseline models and return results dict."""
        results = {}
        
        baseline_specs = [
            ('Linear-DSTM', self._create_linear_dstm, config.num_epochs, {}),
            ('GRU-D', self._create_grud, config.num_epochs, {}),
            ('Latent-ODE', self._create_latent_ode, config.num_epochs, {}),
            ('FNO', self._create_fno, config.num_epochs, {}),
            ('GraFITi', self._create_grafiti, config.num_epochs,
             {'ctx_len': config.window_size, 'forecast_h': config.forecast_horizon}),
            ('APN', self._create_apn, config.num_epochs,
             {'ctx_len': config.window_size, 'forecast_h': config.forecast_horizon}),
        ]
        
        for name, create_fn, n_epochs, fit_kwargs in baseline_specs:
            if verbose:
                print(f"\n[{name}] Training...")
            
            t0 = time.time()
            try:
                model = create_fn(sensor_coords, train_data, config)
                model.fit(train_data, num_epochs=n_epochs, **fit_kwargs)
                train_time = time.time() - t0
                
                if verbose:
                    print(f"[{name}] Evaluating (rolling-origin)...")
                
                t0 = time.time()
                res = self._evaluate_new_baseline_rolling(
                    model, test_data, train_data, config, name
                )
                res.train_time = train_time
                res.eval_time = time.time() - t0
                
                results[name] = res
                
                if verbose:
                    print(f"[{name}] RMSE(m): {res.rmse_measured:.4f}, "
                          f"CRPS(m): {res.crps_measured:.4f}, "
                          f"Coverage(90%): {res.coverage_90:.2%}")
                    if not np.isnan(res.rmse_unmeasured):
                        print(f"[{name}] RMSE(u): {res.rmse_unmeasured:.4f}")
                        
            except Exception as e:
                print(f"[{name}] Error: {e}")
                import traceback
                traceback.print_exc()
        
        if verbose:
            print(f"\n{'='*70}")
            print("Summary:")
            for name, res in results.items():
                print(f"  {name:20s}: RMSE={res.rmse_measured:.4f}")
        
        return results
    
    # ----- Baseline factory methods -----
    
    def _create_linear_dstm(self, sensor_coords, train_data, config):
        return LinearDSTMBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            num_basis=config.num_basis,
            length_scale=config.length_scale
        ).to(self.device)
    
    def _create_grud(self, sensor_coords, train_data, config):
        return GRUDBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            hidden_dim=config.hidden_dim
        ).to(self.device)
    
    def _create_latent_ode(self, sensor_coords, train_data, config):
        return LatentODEBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            latent_dim=config.num_basis,
            hidden_dim=config.hidden_dim
        ).to(self.device)
    
    def _create_fno(self, sensor_coords, train_data, config):
        return FNOBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            width=64,
            modes=min(16, len(train_data.timestamps) // 4),
            n_layers=4,
            dropout=0.1
        ).to(self.device)
    
    def _create_grafiti(self, sensor_coords, train_data, config):
        return GraFITiBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            d_model=128,
            n_layers=2,
            n_heads=4,
            dropout=0.1
        ).to(self.device)
    
    def _create_apn(self, sensor_coords, train_data, config):
        return APNBaseline(
            num_locations=len(sensor_coords),
            spatial_coords=train_data.spatial_coords,
            measured_indices=train_data.measured_indices,
            unmeasured_indices=train_data.unmeasured_indices,
            d_model=128,
            te_dim=16,
            num_patches=16,
            dropout=0.1
        ).to(self.device)
    
    # ----- Visualization -----
    
    def _visualize_bst_predictions(self, 
                                   results: ExperimentResults,
                                   test_data: SpatiotemporalData,
                                   train_data: SpatiotemporalData,
                                   exp_name: str):
        """Visualize NLBST predictions vs ground truth."""
        import matplotlib.pyplot as plt
        
        pred = results.predictions
        if pred is None:
            return
        
        pred_means_m = pred['pred_means_measured']
        pred_stds_m = pred['pred_stds_measured']
        targets_m = pred['targets_measured']
        
        T_pred, S_m = pred_means_m.shape
        
        n_plot = min(4, S_m)
        sensor_indices = np.linspace(0, S_m - 1, n_plot, dtype=int)
        
        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
        if n_plot == 1:
            axes = [axes]
        
        time_axis = np.arange(T_pred)
        z_90 = 1.645
        
        for i, (ax, s_idx) in enumerate(zip(axes, sensor_indices)):
            ax.plot(time_axis, targets_m[:, s_idx], 'k.-', label='Ground Truth', linewidth=1.5, markersize=4)
            
            mean = pred_means_m[:, s_idx]
            std = pred_stds_m[:, s_idx]
            lower = mean - z_90 * std
            upper = mean + z_90 * std
            
            ax.plot(time_axis, mean, 'b-', label='NLBST Prediction', linewidth=1.5)
            ax.fill_between(time_axis, lower, upper, alpha=0.3, color='blue', label='90% CI')
            
            ax.set_ylabel(f'Sensor {s_idx}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Time Step')
        fig.suptitle(f'NLBST Predictions: {exp_name}', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        if 'pred_means_unmeasured' in pred and pred['pred_means_unmeasured'] is not None:
            pred_means_u = pred['pred_means_unmeasured']
            pred_stds_u = pred['pred_stds_unmeasured']
            targets_u = pred['targets_unmeasured']
            
            S_u = pred_means_u.shape[1]
            if S_u > 0:
                n_plot_u = min(2, S_u)
                sensor_indices_u = np.linspace(0, S_u - 1, n_plot_u, dtype=int)
                
                fig2, axes2 = plt.subplots(n_plot_u, 1, figsize=(12, 3 * n_plot_u), sharex=True)
                if n_plot_u == 1:
                    axes2 = [axes2]
                
                for i, (ax, s_idx) in enumerate(zip(axes2, sensor_indices_u)):
                    ax.plot(time_axis[:len(targets_u)], targets_u[:, s_idx], 'k.-', 
                           label='Ground Truth (Unmeasured)', linewidth=1.5, markersize=4)
                    
                    mean = pred_means_u[:, s_idx]
                    std = pred_stds_u[:, s_idx]
                    lower = mean - z_90 * std
                    upper = mean + z_90 * std
                    
                    ax.plot(time_axis[:len(mean)], mean, 'g-', label='NLBST Prediction', linewidth=1.5)
                    ax.fill_between(time_axis[:len(mean)], lower, upper, alpha=0.3, color='green', label='90% CI')
                    
                    ax.set_ylabel(f'Unmeasured {s_idx}')
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.legend(loc='upper right', fontsize=8)
                
                axes2[-1].set_xlabel('Time Step')
                fig2.suptitle(f'NLBST Spatial Generalization: {exp_name}', fontsize=12)
                plt.tight_layout()
                plt.show()
    
    # ----- Rolling-origin evaluation for baselines -----
    
    def _evaluate_new_baseline_rolling(self,
                                       model,
                                       test_data: SpatiotemporalData,
                                       train_data: SpatiotemporalData,
                                       config: ExperimentConfig,
                                       name: str) -> ExperimentResults:
        """
        Rolling-origin evaluation for baseline models.
        
        IMPORTANT (evaluation fix):
        - Uses ground_truth as target for synthetic data (noise-free)
        - Evaluates on ALL measured sensors at ALL test timesteps
          (no test-time mask filtering → consistent evaluation set across missing rates)
        """
        results = ExperimentResults(config=config)
        
        device = test_data.observations.device
        
        try:
            T_train = len(train_data.timestamps)
            T_test = len(test_data.timestamps)
            
            combined_obs = torch.cat([train_data.observations, test_data.observations], dim=0)
            
            train_t_max = train_data.timestamps[-1].item()
            test_t_range = test_data.timestamps[-1].item() - test_data.timestamps[0].item()
            
            if test_t_range > 0:
                test_timestamps_shifted = train_t_max + (test_data.timestamps - test_data.timestamps[0]) * (1.0 - train_t_max) / test_t_range
            else:
                test_timestamps_shifted = train_t_max + torch.linspace(0, 1 - train_t_max, T_test, device=device)
            
            combined_timestamps = torch.cat([train_data.timestamps.to(device), test_timestamps_shifted.to(device)], dim=0)
            
            train_mask = train_data.missing_mask if train_data.missing_mask is not None else torch.zeros_like(train_data.observations, dtype=torch.bool)
            test_mask = test_data.missing_mask if test_data.missing_mask is not None else torch.zeros_like(test_data.observations, dtype=torch.bool)
            combined_mask = torch.cat([train_mask.to(device), test_mask.to(device)], dim=0)
            
            measured_idx = model.measured_indices
            S_m = len(measured_idx)
            S_u = model.S_u
            
            all_predictions_measured = defaultdict(list)
            all_predictions_unmeasured = defaultdict(list)
            
            window_size = config.window_size
            forecast_horizon = config.forecast_horizon
            stride = config.stride
            num_samples = config.num_mc_samples
            
            start_idx = max(0, T_train - window_size)
            
            while start_idx + window_size < T_train + T_test:
                window_end = start_idx + window_size
                forecast_end = min(window_end + forecast_horizon, T_train + T_test)
                actual_horizon = forecast_end - window_end
                
                if window_end >= T_train + T_test or actual_horizon == 0:
                    break
                
                window_obs = combined_obs[start_idx:window_end, measured_idx]
                window_timestamps = combined_timestamps[start_idx:window_end]
                window_mask = combined_mask[start_idx:window_end, measured_idx]
                
                forecast_times = combined_timestamps[window_end:forecast_end]
                
                with torch.no_grad():
                    predictions = model.forecast(
                        window_obs, window_mask, window_timestamps,
                        forecast_times, num_samples=num_samples
                    )
                    
                    ensemble_m = predictions['measured']
                    ensemble_u = predictions.get('unmeasured', None)
                
                for k in range(actual_horizon):
                    global_t = window_end + k
                    test_t = global_t - T_train
                    
                    if 0 <= test_t < T_test:
                        all_predictions_measured[test_t].append({
                            'samples': ensemble_m[:, k].cpu(),
                            'horizon': k + 1
                        })
                        
                        if ensemble_u is not None and S_u > 0:
                            all_predictions_unmeasured[test_t].append({
                                'samples': ensemble_u[:, k].cpu(),
                                'horizon': k + 1
                            })
                
                start_idx += stride
            
            if all_predictions_measured:
                pred_times = sorted(all_predictions_measured.keys())
                
                pred_means_m = []
                pred_stds_m = []
                targets_m = []
                horizons = []
                
                indices = measured_idx.cpu()
                
                for t in pred_times:
                    preds = all_predictions_measured[t][-1]
                    samples = preds['samples'].numpy()
                    
                    pred_means_m.append(samples.mean(axis=0))
                    pred_stds_m.append(samples.std(axis=0))
                    # Use ground truth for synthetic, observations for real
                    targets_m.append(_get_eval_target(test_data, t, indices))
                    horizons.append(preds['horizon'])
                
                pred_means_m = np.array(pred_means_m)
                pred_stds_m = np.array(pred_stds_m)
                targets_m = np.array(targets_m)
                horizons = np.array(horizons)
                
                # ===== NO mask filtering: evaluate on ALL points =====
                pred_flat = pred_means_m.flatten()
                std_flat = pred_stds_m.flatten()
                target_flat = targets_m.flatten()
                
                if len(pred_flat) > 0:
                    results.rmse_measured = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
                    results.mae_measured = np.mean(np.abs(pred_flat - target_flat))
                    
                    crps_flat = ProbabilisticMetrics.crps_gaussian(pred_flat, std_flat, target_flat)
                    results.crps_measured = np.mean(crps_flat)
                    
                    nll_flat = ProbabilisticMetrics.nll_gaussian(pred_flat, std_flat, target_flat)
                    results.nll_measured = np.mean(nll_flat)
                    
                    results.coverage_50 = ProbabilisticMetrics.compute_coverage(pred_flat, std_flat, target_flat, 0.5)
                    results.coverage_90 = ProbabilisticMetrics.compute_coverage(pred_flat, std_flat, target_flat, 0.9)
                    results.coverage_95 = ProbabilisticMetrics.compute_coverage(pred_flat, std_flat, target_flat, 0.95)
                    
                    z_90 = norm.ppf(0.95)
                    lower_90 = pred_flat - z_90 * std_flat
                    upper_90 = pred_flat + z_90 * std_flat
                    is_90 = ProbabilisticMetrics.interval_score(lower_90, upper_90, target_flat, alpha=0.1)
                    results.interval_score_90 = np.mean(is_90)
                    
                    ece, _, _ = ProbabilisticMetrics.calibration_error(pred_flat, std_flat, target_flat)
                    results.calibration_error = ece
                    results.sharpness = np.mean(std_flat)
                
                # Per-horizon metrics
                unique_horizons = np.unique(horizons)
                results.rmse_by_horizon = {}
                results.crps_by_horizon = {}
                
                for h in unique_horizons:
                    mask_h = horizons == h
                    if mask_h.sum() > 0:
                        h_pred = pred_means_m[mask_h].flatten()
                        h_target = targets_m[mask_h].flatten()
                        h_std = pred_stds_m[mask_h].flatten()
                        
                        if len(h_pred) > 0:
                            results.rmse_by_horizon[int(h)] = np.sqrt(np.mean((h_pred - h_target) ** 2))
                            crps_h = ProbabilisticMetrics.crps_gaussian(h_pred, h_std, h_target)
                            results.crps_by_horizon[int(h)] = np.mean(crps_h)
                
                # Unmeasured locations
                if all_predictions_unmeasured and S_u > 0:
                    pred_means_u = []
                    pred_stds_u = []
                    targets_u = []
                    unmeasured_idx = model.unmeasured_indices
                    
                    for t in pred_times:
                        if t in all_predictions_unmeasured:
                            preds = all_predictions_unmeasured[t][-1]
                            samples = preds['samples'].numpy()
                            
                            pred_means_u.append(samples.mean(axis=0))
                            pred_stds_u.append(samples.std(axis=0))
                            targets_u.append(_get_eval_target(test_data, t, unmeasured_idx.cpu()))
                    
                    if pred_means_u:
                        pred_means_u = np.array(pred_means_u)
                        pred_stds_u = np.array(pred_stds_u)
                        targets_u = np.array(targets_u)
                        
                        results.rmse_unmeasured = np.sqrt(np.mean((pred_means_u - targets_u) ** 2))
                        results.mae_unmeasured = np.mean(np.abs(pred_means_u - targets_u))
                        
                        crps_u = ProbabilisticMetrics.crps_gaussian(pred_means_u, pred_stds_u, targets_u)
                        results.crps_unmeasured = np.mean(crps_u)
        
        except Exception as e:
            print(f"Error in {name} evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    # ----- Sweep utilities -----
    
    def run_sweep(self, base_config: ExperimentConfig,
                  sweep_param: str,
                  sweep_values: List[float],
                  n_repeats: int = 3,
                  verbose: bool = True,
                  **kwargs) -> Dict[float, List[Dict[str, Any]]]:
        """Run a parameter sweep with multiple repeats."""
        all_results = {v: [] for v in sweep_values}
        
        for seed_offset in range(n_repeats):
            for value in sweep_values:
                config = ExperimentConfig(
                    **{k: v for k, v in base_config.__dict__.items()}
                )
                config.seed = base_config.seed + seed_offset
                config.name = f"{base_config.name}_{sweep_param}={value}_seed={config.seed}"
                
                setattr(config, sweep_param, value)
                
                try:
                    results = self.run_single_experiment(config, verbose=verbose, **kwargs)
                    all_results[value].append(results)
                except Exception as e:
                    print(f"Error in {config.name}: {e}")
                    continue
        
        return all_results
    
    def aggregate_sweep_results(self, sweep_results: Dict[float, List[Dict[str, Any]]],
                                metric: str = 'rmse_measured') -> Dict[str, Dict[float, Tuple[float, float]]]:
        """Aggregate sweep results across seeds. Returns {model: {param: (mean, std)}}."""
        aggregated = {}
        
        model_names = set()
        for results_list in sweep_results.values():
            for results_dict in results_list:
                for key in results_dict:
                    if isinstance(results_dict[key], ExperimentResults):
                        model_names.add(key)
        
        for model_name in model_names:
            aggregated[model_name] = {}
            
            for param_value, results_list in sweep_results.items():
                metrics = []
                for results_dict in results_list:
                    if model_name in results_dict:
                        val = getattr(results_dict[model_name], metric, np.nan)
                        if not np.isnan(val):
                            metrics.append(val)
                
                if metrics:
                    aggregated[model_name][param_value] = (np.mean(metrics), np.std(metrics))
                else:
                    aggregated[model_name][param_value] = (np.nan, np.nan)
        
        return aggregated
    



# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def run_nonlocal_ide_experiments(device: str = "cuda",
                                 n_repeats: int = 3,
                                 verbose: bool = True) -> Dict:
    """
    Run Nonlocal IDE experiments for RQ4 (Structure Recovery).
    
    1. α-sweep: α ∈ {0.0, 0.25, 0.5, 1.0}
    2. Ablation: full / linear_only / neural_only / diagonal at α=1.0
    3. Basis dimension sweep: K ∈ {8, 12, 16, 24} at α=1.0
    4. Full baseline comparison at α=1.0
    """
    runner = SyntheticExperimentRunner(device=device)
    all_results = {}
    
    base_config = ExperimentConfig(
        name="NonlocalIDE",
        generator="nonlocal_ide",
        T_final=20.0,
        save_every=0.1,
        spinup=0.0,
        n_sensors=36,
        sensor_pattern="grid",
        noise_std=0.05,
        num_basis=16,
        num_epochs=300,
        train_ratio=0.75,
        forecast_horizon=10,
        window_size=50,
        stride=5,
        num_mc_samples=100,
        missing_rate=0.0,
        time_dropout_rate=0.0,
        alpha=1.0,
        kappa=0.01,
        kernel_rank=4,
        sim_dt=0.001,
        domain_size=1.0,
        nx=64,
        ny=64,
    )
    
    # 1. Alpha sweep
    print("\n" + "="*70)
    print("RQ4-1: NONLOCAL STRENGTH (α) SWEEP")
    print("="*70)
    
    alpha_results = runner.run_sweep(
        base_config,
        sweep_param='alpha',
        sweep_values=[0.0, 0.25, 0.5, 1.0],
        n_repeats=n_repeats,
        verbose=verbose,
        skip_baselines=True
    )
    all_results['alpha_sweep'] = alpha_results
    
    
    # 2. Ablation study (α=1.0)
    print("\n" + "="*70)
    print("RQ4-2: ABLATION STUDY (α=1.0)")
    print("="*70)
    
    ablation_config = ExperimentConfig(**base_config.__dict__)
    ablation_config.alpha = 1.0
    
    ablation_results = {}
    for mode in ['full', 'linear_only', 'neural_only', 'diagonal']:
        print(f"\n--- Ablation: {mode} ---")
        mode_runs = []
        for seed_offset in range(n_repeats):
            config = ExperimentConfig(**ablation_config.__dict__)
            config.seed = 42 + seed_offset
            config.name = f"NL-IDE_ablation_{mode}_seed={config.seed}"
            
            try:
                results = runner.run_single_experiment(
                    config, verbose=verbose,
                    ablation_mode=mode,
                    skip_baselines=True
                )
                mode_runs.append(results)
            except Exception as e:
                print(f"Error in ablation {mode}, seed={config.seed}: {e}")
        
        ablation_results[mode] = mode_runs
    
    all_results['ablation'] = ablation_results
    
    if verbose:
        print("\n" + "="*70)
        print("Ablation Summary (α=1.0)")
        print("="*70)
        print(f"{'Mode':<16} {'RMSE':>8} {'CRPS':>8} {'Correlation':>12} {'Spec.Overlap':>13}")
        print("-" * 60)
        
        for mode in ['full', 'linear_only', 'neural_only', 'diagonal']:
            runs = ablation_results[mode]
            key = f'NLBST-{mode}' if mode != 'full' else 'NLBST'
            rmses = [r[key].rmse_measured for r in runs if key in r]
            crps = [r[key].crps_measured for r in runs if key in r]
            
            rmse_str = f"{np.mean(rmses):.4f}" if rmses else "N/A"
            crps_str = f"{np.mean(crps):.4f}" if crps else "N/A"
    
    # 3. Basis dimension sweep (α=1.0)
    print("\n" + "="*70)
    print("RQ4-3: BASIS DIMENSION (K) SWEEP (α=1.0)")
    print("="*70)
    
    basis_config = ExperimentConfig(**base_config.__dict__)
    basis_config.alpha = 1.0
    
    basis_results = runner.run_sweep(
        basis_config,
        sweep_param='num_basis',
        sweep_values=[8, 12, 16, 24],
        n_repeats=n_repeats,
        verbose=verbose,
        skip_baselines=True
    )
    all_results['basis_sweep'] = basis_results
    
    # 4. Full baseline comparison (α=1.0)
    print("\n" + "="*70)
    print("RQ4-4: FULL BASELINE COMPARISON (α=1.0)")
    print("="*70)
    
    comparison_runs = []
    for seed_offset in range(n_repeats):
        config = ExperimentConfig(**base_config.__dict__)
        config.alpha = 1.0
        config.seed = 42 + seed_offset
        config.name = f"NL-IDE_comparison_seed={config.seed}"
        
        try:
            results = runner.run_single_experiment(
                config, verbose=verbose,
                skip_baselines=False
            )
            comparison_runs.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    all_results['baseline_comparison'] = comparison_runs
    
    return all_results


def run_ablation_experiments(device: str = "cuda",
                             generator: str = "advection",
                             n_repeats: int = 3,
                             verbose: bool = True) -> Dict:
    """Run ablation study: full / linear_only / neural_only / diagonal."""
    runner = SyntheticExperimentRunner(device=device)
    
    if generator == 'advection':
        base_config = ExperimentConfig(
            name="Ablation_AD",
            generator="advection",
            T_final=150.0,
            save_every=1.0,
            n_sensors=50,
            sensor_pattern="mixed",
            noise_std=0.1,
            num_basis=16,
            num_epochs=200,
            train_ratio=0.75,
            forecast_horizon=10,
            window_size=50,
            stride=5,
            num_mc_samples=100,
            missing_rate=0.1,
            block_missing=True
        )
    else:
        raise ValueError(f"Unknown generator for ablation: {generator}")
    
    ablation_modes = ['full', 'linear_only', 'neural_only', 'diagonal']
    all_results = {}
    
    for mode in ablation_modes:
        print(f"\n{'='*70}")
        print(f"ABLATION: {mode}")
        print(f"{'='*70}")
        
        mode_runs = []
        for seed_offset in range(n_repeats):
            config = ExperimentConfig(**base_config.__dict__)
            config.seed = 42 + seed_offset
            config.name = f"{base_config.name}_{mode}_seed={config.seed}"
            
            try:
                results = runner.run_single_experiment(
                    config, verbose=verbose,
                    ablation_mode=mode,
                    skip_baselines=True
                )
                mode_runs.append(results)
            except Exception as e:
                print(f"Error: {e}")
        
        all_results[mode] = mode_runs
    
    if verbose:
        print("\n" + "="*70)
        print(f"ABLATION SUMMARY ({generator})")
        print("="*70)
        print(f"{'Mode':<16} {'RMSE(m)':>8} {'CRPS(m)':>8} {'Cov(90%)':>8}")
        print("-" * 50)
        
        for mode in ablation_modes:
            runs = all_results[mode]
            key = f'NLBST-{mode}' if mode != 'full' else 'NLBST'
            rmses = [r[key].rmse_measured for r in runs if key in r]
            crps = [r[key].crps_measured for r in runs if key in r]
            covs = [r[key].coverage_90 for r in runs if key in r]
            
            print(f"{mode:<16} "
                  f"{np.mean(rmses):>8.4f} "
                  f"{np.mean(crps):>8.4f} "
                  f"{np.mean(covs):>8.2%}")
    
    return all_results


def run_advection_experiments(device: str = "cuda",
                              n_repeats: int = 3,
                              verbose: bool = True) -> Dict:
    """Run full Advection-Diffusion experiment suite."""
    runner = SyntheticExperimentRunner(device=device)
    all_results = {}
    
    base_config = ExperimentConfig(
        name="Advection",
        generator="advection",
        T_final=150.0,
        save_every=1.0,
        n_sensors=50,
        sensor_pattern="mixed",
        noise_std=0.1,
        num_basis=16,
        num_epochs=200,
        train_ratio=0.75,
        forecast_horizon=10,
        window_size=50,
        stride=5,
        num_mc_samples=100
    )
    
    # 1. Baseline experiment
    print("\n" + "="*70)
    print("1. BASELINE EXPERIMENT (Realistic Setting)")
    print("="*70)
    
    baseline_config = ExperimentConfig(**base_config.__dict__)
    baseline_config.time_dropout_rate = 0.2
    baseline_config.missing_rate = 0.1
    baseline_config.block_missing = True
    
    baseline_results = []
    for seed in range(n_repeats):
        config = ExperimentConfig(**baseline_config.__dict__)
        config.seed = 42 + seed
        config.name = f"AD_baseline_seed={config.seed}"
        results = runner.run_single_experiment(config, verbose=verbose)
        baseline_results.append(results)
    
    all_results['baseline'] = baseline_results
    
    # 2. Irregular time sweep
    print("\n" + "="*70)
    print("2. IRREGULAR TIME SWEEP")
    print("="*70)
    
    time_dropout_results = runner.run_sweep(
        base_config,
        sweep_param='time_dropout_rate',
        sweep_values=[0.1, 0.2, 0.4, 0.6],
        n_repeats=n_repeats,
        verbose=verbose
    )
    all_results['time_dropout'] = time_dropout_results
    
    # 3. Missing rate sweep
    print("\n" + "="*70)
    print("3. MISSING RATE SWEEP")
    print("="*70)
    
    missing_config = ExperimentConfig(**base_config.__dict__)
    missing_config.block_missing = True
    
    missing_results = runner.run_sweep(
        missing_config,
        sweep_param='missing_rate',
        sweep_values=[0.1, 0.2, 0.3, 0.4],
        n_repeats=n_repeats,
        verbose=verbose
    )
    all_results['missing_rate'] = missing_results
    
    # 4. Spatial holdout sweep
    print("\n" + "="*70)
    print("4. SPATIAL HOLDOUT SWEEP")
    print("="*70)
    
    holdout_config = ExperimentConfig(**base_config.__dict__)
    holdout_config.holdout_mode = 'region'
    
    holdout_results = runner.run_sweep(
        holdout_config,
        sweep_param='holdout_ratio',
        sweep_values=[0.1, 0.2, 0.3, 0.4],
        n_repeats=n_repeats,
        verbose=verbose
    )
    all_results['spatial_holdout'] = holdout_results
    
    return all_results




def run_full_experiment_suite(device: str = "cuda",
                              n_repeats: int = 3,
                              verbose: bool = True) -> Dict:

    print("="*70)
    print("NLBST: Full Synthetic Experiment Suite")
    print("="*70)
    
    all_results = {}
    
    print("\n" + "="*70)
    print("D1: NONLOCAL IDE EXPERIMENTS (RQ4)")
    print("="*70)
    nlide_results = run_nonlocal_ide_experiments(device, n_repeats, verbose)
    all_results['nonlocal_ide'] = nlide_results
    
    print("\n" + "="*70)
    print("D2: ADVECTION-DIFFUSION EXPERIMENTS (RQ1-3)")
    print("="*70)
    ad_results = run_advection_experiments(device, n_repeats, verbose)
    all_results['advection_diffusion'] = ad_results
    
    
    return all_results