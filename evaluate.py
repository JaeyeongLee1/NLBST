import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from collections import defaultdict
from typing import Dict

from .config import SpatiotemporalData, ExperimentConfig, ExperimentResults
from .model import BayesianSTNeuralODE_Kalman, VariationalKalmanGuide
from .metrics import ProbabilisticMetrics


def _get_eval_target(data: SpatiotemporalData, t: int, sensor_indices):
    if data.ground_truth is not None:
        return data.ground_truth[t, sensor_indices.cpu()].cpu().numpy()
    else:
        return data.observations[t, sensor_indices.cpu()].cpu().numpy()


def evaluate_nlbst(model: BayesianSTNeuralODE_Kalman,
                      guide: VariationalKalmanGuide,
                      test_data: SpatiotemporalData,
                      train_data: SpatiotemporalData,
                      config: ExperimentConfig) -> ExperimentResults:
    
    model.eval()
    guide.eval()
    
    device = test_data.observations.device
    results = ExperimentResults(config=config)
    has_ground_truth = test_data.ground_truth is not None
    
    try:
        T_train = len(train_data.timestamps)
        T_test = len(test_data.timestamps)
        
        # Combine train and test for rolling evaluation
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
        
        if has_ground_truth:
            combined_gt = torch.cat([train_data.ground_truth, test_data.ground_truth], dim=0)
        
        # Get learned hyperparameters
        sigma_process_loc = guide.sigma_process_loc.detach()
        sigma_process_scale = (F.softplus(guide.sigma_process_scale_unc).detach() + 0.01)
        
        sigma_obs_loc = guide.sigma_obs_loc.detach()
        sigma_obs_scale = (F.softplus(guide.sigma_obs_scale_unc).detach() + 0.01)
        
        sigma_process_mean_norm = np.exp(sigma_process_loc.item() + 0.5 * sigma_process_scale.item()**2)
        sigma_obs_mean_norm = np.exp(sigma_obs_loc.item() + 0.5 * sigma_obs_scale.item()**2)
        
        data_range = (model.data_max - model.data_min).mean().item()
        sigma_obs_mean_raw = sigma_obs_mean_norm * data_range
        sigma_process_mean_raw = sigma_process_mean_norm * data_range
        
        results.sigma_obs = sigma_obs_mean_raw
        results.sigma_process = sigma_process_mean_raw
        results.sigma_obs_normalized = sigma_obs_mean_norm
        results.sigma_process_normalized = sigma_process_mean_norm

        Phi_m = model.get_basis_measured()
        Phi_u = model.get_basis_unmeasured()
        
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
            
            window_obs = combined_obs[start_idx:window_end, model.measured_indices]
            window_timestamps = combined_timestamps[start_idx:window_end]
            window_mask = combined_mask[start_idx:window_end, model.measured_indices]
            
            window_obs_norm = model.normalize(window_obs)
            t_normalized = (window_timestamps / model.global_t_max).float()
            
            with torch.no_grad():
                sample_predictions_m = []
                sample_predictions_u = []
                
                for _ in range(num_samples):
                    sigma_obs_sample = torch.exp(
                        sigma_obs_loc + sigma_obs_scale * torch.randn(1, device=device)
                    ).squeeze()
                    sigma_process_sample = torch.exp(
                        sigma_process_loc + sigma_process_scale * torch.randn(1, device=device)
                    ).squeeze()
                    
                    z0_mean, z0_std = guide.z0_encoder(window_obs_norm[0], Phi_m, sigma_obs_sample, window_mask[0])
                    m_curr = z0_mean + z0_std * torch.randn_like(z0_std)
                    P_curr = z0_std ** 2
                    
                    for t in range(1, len(window_timestamps)):
                        dt = (t_normalized[t] - t_normalized[t-1]).clamp_min(1e-6)
                        
                        m_prior, P_prior = model.kalman_filter.predict(
                            m_curr, P_curr, sigma_process_sample, dt,
                            model.ode_func,
                            t_normalized[t-1].item(),
                            t_normalized[t].item()
                        )
                        
                        m_post, P_post = model.kalman_filter.update(
                            m_prior, P_prior,
                            window_obs_norm[t], Phi_m, sigma_obs_sample,
                            window_mask[t]
                        )
                        
                        m_curr = m_post + P_post.sqrt() * torch.randn_like(P_post)
                        P_curr = P_post
                    
                    t_last = t_normalized[-1].item()
                    forecast_preds_m = []
                    forecast_preds_u = []
                    
                    future_indices = list(range(window_end, forecast_end))
                    future_ts = combined_timestamps[future_indices] / model.global_t_max
                    
                    for k in range(actual_horizon):
                        t_next = future_ts[k].item()
                        dt_f = torch.tensor(max(t_next - t_last, 1e-6), device=device)
                        
                        m_curr = model.ode_func.integrate_step(m_curr, t_last, t_next)
                        
                        y_pred_m_norm = model.decode(m_curr, Phi_m)
                        y_pred_m = model.denormalize(y_pred_m_norm)
                        forecast_preds_m.append(y_pred_m)
                        
                        if model.S_u > 0:
                            y_pred_u_norm = model.decode(m_curr, Phi_u)
                            avg_scale = (model.data_max - model.data_min).mean() ###
                            avg_offset = model.data_min.mean() ###
                            y_pred_u = y_pred_u_norm * avg_scale + avg_offset
                            forecast_preds_u.append(y_pred_u)
                        
                        t_last = t_next
                    
                    if forecast_preds_m:
                        sample_predictions_m.append(torch.stack(forecast_preds_m))
                        if forecast_preds_u:
                            sample_predictions_u.append(torch.stack(forecast_preds_u))
                
                if sample_predictions_m:
                    ensemble_m = torch.stack(sample_predictions_m)
                    if sample_predictions_u:
                        ensemble_u = torch.stack(sample_predictions_u)
            
            for k in range(actual_horizon):
                global_t = window_end + k
                test_t = global_t - T_train
                
                if 0 <= test_t < T_test:
                    all_predictions_measured[test_t].append({
                        'samples': ensemble_m[:, k].cpu(),
                        'horizon': k + 1
                    })
                    
                    if model.S_u > 0 and sample_predictions_u:
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
            
            for t in pred_times:
                preds = all_predictions_measured[t][-1]
                samples = preds['samples'].numpy()
                
                pred_means_m.append(samples.mean(axis=0))
                pred_stds_m.append(samples.std(axis=0))
                targets_m.append(_get_eval_target(test_data, t, model.measured_indices))
                horizons.append(preds['horizon'])
            
            pred_means_m = np.array(pred_means_m)
            pred_stds_m = np.array(pred_stds_m)
            targets_m = np.array(targets_m)
            horizons = np.array(horizons)
            
            pred_flat = pred_means_m.flatten()
            std_flat = pred_stds_m.flatten()
            target_flat = targets_m.flatten()
            
            if len(pred_flat) > 0:
                results.rmse_measured = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
                results.mae_measured = np.mean(np.abs(pred_flat - target_flat))
                
                crps_flat = ProbabilisticMetrics.crps_ensemble(pred_flat, std_flat, target_flat)
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
            
            unique_horizons = np.unique(horizons)
            results.rmse_by_horizon = {}
            results.crps_by_horizon = {}
            
            for h in unique_horizons:
                mask_h = horizons == h
                if mask_h.sum() > 0:
                    h_pred = pred_means_m[mask_h]
                    h_target = targets_m[mask_h]
                    h_std = pred_stds_m[mask_h]
                    
                    h_pred_flat = h_pred.flatten()
                    h_target_flat = h_target.flatten()
                    h_std_flat = h_std.flatten()
                    
                    if len(h_pred_flat) > 0:
                        results.rmse_by_horizon[int(h)] = np.sqrt(np.mean((h_pred_flat - h_target_flat) ** 2))
                        crps_h = ProbabilisticMetrics.crps_gaussian(h_pred_flat, h_std_flat, h_target_flat)
                        results.crps_by_horizon[int(h)] = np.mean(crps_h)
            
            persistence_preds = np.roll(targets_m, 1, axis=0)
            persistence_preds[0] = _get_eval_target(train_data, -1, model.measured_indices)
            rmse_persistence = np.sqrt(np.mean((persistence_preds - targets_m) ** 2))
            results.skill_rmse = 1 - (results.rmse_measured / rmse_persistence) if rmse_persistence > 0 else 0
            
            if all_predictions_unmeasured and model.S_u > 0:
                pred_means_u = []
                pred_stds_u = []
                targets_u = []
                
                for t in pred_times:
                    if t in all_predictions_unmeasured:
                        preds = all_predictions_unmeasured[t][-1]
                        samples = preds['samples'].numpy()
                        
                        pred_means_u.append(samples.mean(axis=0))
                        pred_stds_u.append(samples.std(axis=0))
                        targets_u.append(_get_eval_target(test_data, t, model.unmeasured_indices))
                
                if pred_means_u:
                    pred_means_u = np.array(pred_means_u)
                    pred_stds_u = np.array(pred_stds_u)
                    targets_u = np.array(targets_u)
                    
                    pred_u_flat = pred_means_u.flatten()
                    std_u_flat = pred_stds_u.flatten()
                    target_u_flat = targets_u.flatten()
                    
                    results.rmse_unmeasured = np.sqrt(np.mean((pred_u_flat - target_u_flat) ** 2))
                    results.mae_unmeasured = np.mean(np.abs(pred_u_flat - target_u_flat))
                    
                    crps_u = ProbabilisticMetrics.crps_gaussian(pred_u_flat, std_u_flat, target_u_flat)
                    results.crps_unmeasured = np.mean(crps_u)
                    
                    nll_u = ProbabilisticMetrics.nll_gaussian(pred_u_flat, std_u_flat, target_u_flat)
                    results.nll_unmeasured = np.mean(nll_u)
                    
                    if has_ground_truth:
                        results.rmse_unmeasured_noisefree = results.rmse_unmeasured
                        obs_u = np.array([
                            test_data.observations[t, model.unmeasured_indices.cpu()].cpu().numpy()
                            for t in pred_times if t in all_predictions_unmeasured
                        ])
                        if len(obs_u) > 0:
                            results.rmse_unmeasured_vs_obs = np.sqrt(np.mean((pred_means_u - obs_u) ** 2))
            
            results.predictions = {
                'pred_means_measured': pred_means_m,
                'pred_stds_measured': pred_stds_m,
                'targets_measured': targets_m,
                'horizons': horizons,
                'times': pred_times,
                'target_is_ground_truth': has_ground_truth,
            }
            
            if all_predictions_unmeasured and model.S_u > 0 and len(pred_means_u) > 0:
                results.predictions['pred_means_unmeasured'] = pred_means_u
                results.predictions['pred_stds_unmeasured'] = pred_stds_u
                results.predictions['targets_unmeasured'] = targets_u
    
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def evaluate_baseline_rolling(baseline_model,
                              test_data: SpatiotemporalData,
                              train_data: SpatiotemporalData,
                              config: ExperimentConfig,
                              model_name: str = "Baseline") -> ExperimentResults:
    results = ExperimentResults(config=config)
    
    device = test_data.observations.device
    has_ground_truth = test_data.ground_truth is not None
    
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
        
        measured_idx = train_data.measured_indices
        unmeasured_idx = train_data.unmeasured_indices
        S_m = len(measured_idx)
        S_u = len(unmeasured_idx)
        
        all_predictions_measured = defaultdict(list)
        all_predictions_unmeasured = defaultdict(list)
        
        window_size = config.window_size
        forecast_horizon = config.forecast_horizon
        stride = config.stride
        num_samples = config.num_mc_samples
        
        if hasattr(baseline_model, 'spatial_basis'):
            Phi_m = baseline_model.spatial_basis(baseline_model.measured_coords)
            Phi_u = baseline_model.spatial_basis(baseline_model.unmeasured_coords) if S_u > 0 else None
        else:
            Phi_m = None
            Phi_u = None
        
        from .baselines import LinearDSTMBaseline
        
        if hasattr(baseline_model, 'fit') and not isinstance(baseline_model, LinearDSTMBaseline):
            baseline_model.fit(train_data, num_epochs=config.num_epochs, lr=config.learning_rate)
        
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
            
            if hasattr(baseline_model, 'data_min'):
                obs_norm = (window_obs - baseline_model.data_min) / (baseline_model.data_max - baseline_model.data_min + 1e-6)
            else:
                obs_norm = window_obs
            
            with torch.no_grad():
                sample_predictions_m = []
                sample_predictions_u = []
                
                if isinstance(baseline_model, LinearDSTMBaseline):
                    sigma_obs = torch.exp(baseline_model.log_sigma_obs)
                    sigma_process = torch.exp(baseline_model.log_sigma_process)
                    
                    for _ in range(num_samples):
                        z_mean = torch.zeros(baseline_model.K, device=device)
                        z_cov = torch.eye(baseline_model.K, device=device)
                        
                        for t in range(len(window_timestamps)):
                            if t > 0:
                                z_mean = baseline_model.A @ z_mean
                                z_cov = baseline_model.A @ z_cov @ baseline_model.A.T + sigma_process**2 * torch.eye(baseline_model.K, device=device)
                            
                            valid_mask = ~window_mask[t]
                            n_valid = valid_mask.sum().item()
                            
                            if n_valid == 0:
                                continue
                            
                            y_valid = obs_norm[t][valid_mask]
                            Phi_valid = Phi_m[valid_mask]
                            
                            y_pred = Phi_valid @ z_mean * baseline_model.output_scale + baseline_model.output_bias
                            innov = y_valid - y_pred
                            S = Phi_valid @ z_cov @ Phi_valid.T + sigma_obs**2 * torch.eye(n_valid, device=device)
                            K_gain = z_cov @ Phi_valid.T @ torch.linalg.inv(S + 1e-4 * torch.eye(n_valid, device=device))
                            z_mean = z_mean + K_gain @ innov
                            z_cov = (torch.eye(baseline_model.K, device=device) - K_gain @ Phi_valid) @ z_cov
                        
                        try:
                            L = torch.linalg.cholesky(z_cov + 1e-4 * torch.eye(baseline_model.K, device=device))
                            z = z_mean + L @ torch.randn(baseline_model.K, device=device)
                        except:
                            z = z_mean + 0.1 * torch.randn(baseline_model.K, device=device)
                        
                        preds_m = []
                        preds_u = []
                        
                        for h in range(actual_horizon):
                            z = baseline_model.A @ z + sigma_process * torch.randn(baseline_model.K, device=device)
                            
                            y_m = Phi_m @ z * baseline_model.output_scale + baseline_model.output_bias
                            y_m = y_m * (baseline_model.data_max - baseline_model.data_min) + baseline_model.data_min
                            preds_m.append(y_m)
                            
                            if Phi_u is not None and S_u > 0:
                                y_u = Phi_u @ z * baseline_model.output_scale + baseline_model.output_bias
                                y_u = y_u * (baseline_model.global_max - baseline_model.global_min) + baseline_model.global_min
                                preds_u.append(y_u)
                        
                        sample_predictions_m.append(torch.stack(preds_m))
                        if preds_u:
                            sample_predictions_u.append(torch.stack(preds_u))
                
                else:
                    future_indices = list(range(window_end, forecast_end))
                    forecast_ts = combined_timestamps[future_indices]
                    
                    result = baseline_model.forecast(
                        context_obs=window_obs,
                        context_mask=window_mask,
                        context_times=window_timestamps,
                        forecast_times=forecast_ts,
                        num_samples=num_samples,
                    )
                    
                    if isinstance(result, dict):
                        samples_m = result.get('measured', None)
                        samples_u = result.get('unmeasured', None)
                    elif isinstance(result, tuple):
                        samples_m, samples_u = result
                    else:
                        samples_m = result
                        samples_u = None
                    
                    if samples_m is not None and len(samples_m) > 0:
                        if not isinstance(samples_m, torch.Tensor):
                            samples_m = torch.tensor(samples_m, dtype=torch.float32, device=device)
                        
                        for i in range(samples_m.shape[0]):
                            sample_predictions_m.append(samples_m[i])  
                        
                        if samples_u is not None and S_u > 0 and len(samples_u) > 0:
                            if not isinstance(samples_u, torch.Tensor):
                                samples_u = torch.tensor(samples_u, dtype=torch.float32, device=device)
                            for i in range(samples_u.shape[0]):
                                sample_predictions_u.append(samples_u[i]) 
                
                if sample_predictions_m:
                    ensemble_m = torch.stack(sample_predictions_m)
                    if sample_predictions_u:
                        ensemble_u = torch.stack(sample_predictions_u)
            
            for k in range(actual_horizon):
                global_t = window_end + k
                test_t = global_t - T_train
                
                if 0 <= test_t < T_test:
                    if sample_predictions_m:
                        all_predictions_measured[test_t].append({
                            'samples': ensemble_m[:, k].cpu(),
                            'horizon': k + 1
                        })
                        
                        if S_u > 0 and sample_predictions_u:
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
            
            for t in pred_times:
                preds = all_predictions_measured[t][-1]
                samples = preds['samples'].numpy()
                
                pred_means_m.append(samples.mean(axis=0))
                pred_stds_m.append(samples.std(axis=0))
                targets_m.append(_get_eval_target(test_data, t, measured_idx))
                horizons.append(preds['horizon'])
            
            pred_means_m = np.array(pred_means_m)
            pred_stds_m = np.array(pred_stds_m)
            targets_m = np.array(targets_m)
            horizons = np.array(horizons)
            
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
            
            unique_horizons = np.unique(horizons)
            results.rmse_by_horizon = {}
            results.crps_by_horizon = {}
            
            for h in unique_horizons:
                mask_h = horizons == h
                if mask_h.sum() > 0:
                    h_pred = pred_means_m[mask_h]
                    h_target = targets_m[mask_h]
                    h_std = pred_stds_m[mask_h]
                    
                    h_pred_flat = h_pred.flatten()
                    h_target_flat = h_target.flatten()
                    h_std_flat = h_std.flatten()
                    
                    if len(h_pred_flat) > 0:
                        results.rmse_by_horizon[int(h)] = np.sqrt(np.mean((h_pred_flat - h_target_flat) ** 2))
                        crps_h = ProbabilisticMetrics.crps_gaussian(h_pred_flat, h_std_flat, h_target_flat)
                        results.crps_by_horizon[int(h)] = np.mean(crps_h)
            
            if all_predictions_unmeasured and S_u > 0:
                pred_means_u = []
                pred_stds_u = []
                targets_u = []
                
                for t in pred_times:
                    if t in all_predictions_unmeasured:
                        preds = all_predictions_unmeasured[t][-1]
                        samples = preds['samples'].numpy()
                        
                        pred_means_u.append(samples.mean(axis=0))
                        pred_stds_u.append(samples.std(axis=0))
                        targets_u.append(_get_eval_target(test_data, t, unmeasured_idx))
                
                if pred_means_u:
                    pred_means_u = np.array(pred_means_u)
                    pred_stds_u = np.array(pred_stds_u)
                    targets_u = np.array(targets_u)
                    
                    pred_u_flat = pred_means_u.flatten()
                    std_u_flat = pred_stds_u.flatten()
                    target_u_flat = targets_u.flatten()
                    results.rmse_unmeasured = np.sqrt(np.mean((pred_u_flat - target_u_flat) ** 2))
                    results.mae_unmeasured = np.mean(np.abs(pred_u_flat - target_u_flat))
                    
                    crps_u = ProbabilisticMetrics.crps_gaussian(pred_u_flat, std_u_flat, target_u_flat)
                    results.crps_unmeasured = np.mean(crps_u)
            
            results.predictions = {
                'pred_means_measured': pred_means_m,
                'pred_stds_measured': pred_stds_m,
                'targets_measured': targets_m,
                'horizons': horizons,
                'target_is_ground_truth': has_ground_truth,
            }
            if all_predictions_unmeasured and S_u > 0 and len(pred_means_u) > 0:
                results.predictions['pred_means_unmeasured'] = pred_means_u
                results.predictions['pred_stds_unmeasured'] = pred_stds_u
                results.predictions['targets_unmeasured'] = targets_u
    
    except Exception as e:
        print(f"Evaluation error for {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return results