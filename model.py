import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from .config import SpatiotemporalData
from .components import (
    SpatialBasisFunctions, 
    TemporalODEFunc, 
    BasisKalmanFilter, 
    BasisInitialEncoder
)

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False




class CombinedODEFunc(nn.Module):

    
    def __init__(self, 
                 A_NL: nn.Parameter, 
                 neural_residual: TemporalODEFunc,
                 ablation_mode: str = 'full'):
        super().__init__()
        self.A_NL = A_NL
        self.neural_residual = neural_residual
        self.ablation_mode = ablation_mode
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.ablation_mode == 'neural_only':
            linear_term = 0.0
        elif self.ablation_mode == 'diagonal':
            linear_term = torch.diag(self.A_NL) * z
        else:
            if z.dim() == 1:
                linear_term = self.A_NL @ z
            else:
                linear_term = z @ self.A_NL.T
        
        if self.ablation_mode == 'linear_only':
            return linear_term
        
        nonlinear_term = self.neural_residual(t, z)
        return linear_term + nonlinear_term
    
    def integrate_step(self, z: torch.Tensor, 
                       t_start: float, t_end: float) -> torch.Tensor:
        if abs(t_end - t_start) < 1e-6:
            return z
        
        device = z.device
        dtype = z.dtype
        t_span = torch.tensor([t_start, t_end], device=device, dtype=dtype)
        
        if TORCHDIFFEQ_AVAILABLE:
            z_traj = odeint(self, z, t_span, method='rk4')
            return z_traj[-1]
        else:
            n_steps = 5
            dt = (t_end - t_start) / n_steps
            z_current = z
            for i in range(n_steps):
                t_i = t_start + i * dt
                t_tensor = torch.tensor(t_i, device=device, dtype=dtype)
                dz = self.forward(t_tensor, z_current)
                z_current = z_current + dz * dt
                z_current = torch.clamp(z_current, -10, 10)
            return z_current


# ============================================================================
# Main model
# ============================================================================

class BayesianSTNeuralODE_Kalman(PyroModule):

    
    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 num_basis: int = 16,
                 hidden_dim: int = 64,
                 basis_type: str = 'rbf',
                 length_scale: float = 1.0,
                 normalization: str = 'minmax',
                 process_noise_scale: float = 0.1,
                 initial_noise_scale: float = 0.5,
                 global_t_max: float = 1.0,
                 ablation_mode: str = 'full'):
        super().__init__()
        
        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.K = num_basis
        self.normalization = normalization
        self.global_t_max = global_t_max
        self.process_noise_scale = process_noise_scale
        self.initial_noise_scale = initial_noise_scale
        self.ablation_mode = ablation_mode
        
        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords', 
                             spatial_coords[unmeasured_indices] 
                             if len(unmeasured_indices) > 0 
                             else torch.zeros(0, spatial_coords.shape[1]))
        

        self.spatial_basis = SpatialBasisFunctions(
            num_basis=num_basis,
            spatial_coords=spatial_coords,       
            basis_type=basis_type,
            length_scale=length_scale,
            learn_basis=False
            )

        self.A_NL = nn.Parameter(torch.eye(num_basis) * 0.9 + torch.randn(num_basis, num_basis) * 0.01)
        
        self._neural_residual = TemporalODEFunc(num_basis, hidden_dim)
        
        self.ode_func = CombinedODEFunc(
            A_NL=self.A_NL,
            neural_residual=self._neural_residual,
            ablation_mode=ablation_mode
        )
        
        self.kalman_filter = BasisKalmanFilter(num_basis)
        
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.zeros(1))
        
        self.register_buffer('data_mean', torch.zeros(self.S_m))
        self.register_buffer('data_std', torch.ones(self.S_m))
        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))
        self.register_buffer('normalization_computed', torch.tensor(False))
    

    
    def set_ablation_mode(self, mode: str):
        assert mode in ('full', 'linear_only', 'neural_only', 'diagonal'), \
            f"Unknown ablation mode: {mode}"
        self.ablation_mode = mode
        self.ode_func.ablation_mode = mode

    
    def get_induced_kernel(self, 
                           eval_coords: Optional[torch.Tensor] = None,
                           ) -> torch.Tensor:
        with torch.no_grad():
            if eval_coords is None:
                eval_coords = self.spatial_coords
            
            Phi = self.spatial_basis(eval_coords)  # [N, K]
            N = Phi.shape[0]
            M_approx = Phi.T @ Phi / N  # [K, K]
            M_inv = torch.linalg.inv(
                M_approx + 1e-4 * torch.eye(self.K, device=Phi.device)
            )
            G = Phi @ self.A_NL @ M_inv @ Phi.T  # [N, N]
            return G
    
    def get_A_NL_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eig(self.A_NL)
            idx = torch.argsort(eigenvalues.abs(), descending=True)
            return eigenvalues[idx], eigenvectors[:, idx]
    

    def compute_normalization_params(self, observations: torch.Tensor, 
                                     missing_mask: Optional[torch.Tensor] = None):
        if self.normalization == 'none':
            self.normalization_computed.fill_(True)
            return
        
        obs_measured = observations[:, self.measured_indices]
        device = observations.device
        S = obs_measured.shape[1]
        
        if missing_mask is not None:
            mask_m = missing_mask[:, self.measured_indices]
            
            data_mean_list, data_std_list = [], []
            data_min_list, data_max_list = [], []
            
            for s in range(S):
                valid_mask_s = ~mask_m[:, s]
                if valid_mask_s.any():
                    valid_vals = obs_measured[:, s][valid_mask_s]
                    data_mean_list.append(valid_vals.mean())
                    data_std_list.append(valid_vals.std() + 1e-6)
                    data_min_list.append(valid_vals.min())
                    data_max_list.append(valid_vals.max())
                else:
                    data_mean_list.append(torch.tensor(0.0, device=device))
                    data_std_list.append(torch.tensor(1.0, device=device))
                    data_min_list.append(torch.tensor(0.0, device=device))
                    data_max_list.append(torch.tensor(1.0, device=device))
            
            self.data_mean.copy_(torch.stack(data_mean_list))
            self.data_std.copy_(torch.stack(data_std_list))
            self.data_min.copy_(torch.stack(data_min_list))
            self.data_max.copy_(torch.stack(data_max_list))
        else:
            self.data_mean.copy_(obs_measured.mean(dim=0))
            self.data_std.copy_(obs_measured.std(dim=0) + 1e-6)
            self.data_min.copy_(obs_measured.min(dim=0)[0])
            self.data_max.copy_(obs_measured.max(dim=0)[0])
        
        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())
        self.normalization_computed.fill_(True)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'none':
            return x
        range_val = self.data_max - self.data_min + 1e-6
        if x.dim() == 1:
            return (x - self.data_min) / range_val
        return (x - self.data_min.unsqueeze(0)) / range_val.unsqueeze(0)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'none':
            return x
        range_val = self.data_max - self.data_min
        if x.dim() == 1:
            return x * range_val + self.data_min
        return x * range_val.unsqueeze(0) + self.data_min.unsqueeze(0)
    
    def denormalize_global(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using global min/max (for unmeasured locations)."""
        if self.normalization == 'none':
            return x
        range_val = self.global_max - self.global_min
        return x * range_val + self.global_min
    
    
    def get_basis_measured(self) -> torch.Tensor:
        return self.spatial_basis(self.measured_coords)
    
    def get_basis_unmeasured(self) -> torch.Tensor:
        if self.S_u == 0:
            return torch.zeros(0, self.K, device=self.spatial_coords.device)
        return self.spatial_basis(self.unmeasured_coords)
    
    def decode(self, z: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """Decode latent coefficients to observation space: y = Φ z * scale + bias."""
        if z.dim() == 1:
            y = Phi @ z
        else:
            y = z @ Phi.T
        return y * self.output_scale + self.output_bias
    
    
    def model(self, data: SpatiotemporalData, prediction_mode: bool = False):
        device = data.observations.device
        T = len(data.timestamps)
        
        if not prediction_mode and not self.normalization_computed.item():
            self.compute_normalization_params(data.observations, data.missing_mask)
        
        obs_measured = data.observations[:, self.measured_indices]
        obs_norm = self.normalize(obs_measured)
        
        if data.missing_mask is not None:
            obs_mask = data.missing_mask[:, self.measured_indices].to(device)
        else:
            obs_mask = torch.zeros(T, self.S_m, dtype=torch.bool, device=device)
        
        t_normalized = data.timestamps.to(device).float() / self.global_t_max
        Phi_m = self.get_basis_measured()
        
        # --- Level 3: Parameter priors ---
        sigma_obs = pyro.sample(
            "sigma_obs", 
            dist.LogNormal(torch.tensor(-1.0, device=device), 
                          torch.tensor(0.5, device=device))
        )
        sigma_process = pyro.sample(
            "sigma_process", 
            dist.LogNormal(torch.tensor(np.log(self.process_noise_scale), device=device), 
                          torch.tensor(0.3, device=device))
        )
        z_0 = pyro.sample(
            "z_0", 
            dist.Normal(torch.zeros(self.K, device=device), 
                       torch.full((self.K,), self.initial_noise_scale, device=device)
            ).to_event(1)
        )
        
        # --- Level 2: Latent dynamics ---
        z_states = [z_0]
        z_prev = z_0
        
        for t in range(1, T):
            z_pred = self.ode_func.integrate_step(
                z_prev, t_normalized[t-1].item(), t_normalized[t].item()
            )
            
           
            noise_scale = sigma_process
            
            z_t = pyro.sample(
                f"z_{t}", 
                dist.Normal(z_pred, noise_scale).to_event(1)
            )
            z_states.append(z_t)
            z_prev = z_t
        
        # --- Level 1: Observation likelihood ---
        z_all = torch.stack(z_states, dim=0)
        y_pred_norm = self.decode(z_all, Phi_m)
        
        y_pred_flat = y_pred_norm.reshape(-1)
        obs_flat = obs_norm.reshape(-1)
        valid_mask = ~obs_mask.reshape(-1)
        
        if valid_mask.any():
            y_pred_valid = y_pred_flat[valid_mask]
            obs_valid = obs_flat[valid_mask]
            
            with pyro.plate("obs_plate", len(obs_valid)):
                pyro.sample("y_obs", dist.Normal(y_pred_valid, sigma_obs), obs=obs_valid)
        
        forecast_steps = getattr(self, '_forecast_steps', 3)
        n_forecast_origins = getattr(self, '_n_forecast_origins', 2)
        forecast_weight = getattr(self, '_forecast_weight', 0)
        
        if not prediction_mode and T > forecast_steps + 20:
            for fi in range(n_forecast_origins):
                origin = torch.randint(T // 2, T - forecast_steps - 1, (1,)).item()
                z_fc = z_states[origin].detach()
                
                fc_preds = []
                t_curr = t_normalized[origin].item()
                
                for k in range(1, forecast_steps + 1):
                    t_next = t_normalized[origin + k].item()
                    z_fc = self.ode_func.integrate_step(z_fc, t_curr, t_next)
                    y_fc = self.decode(z_fc, Phi_m)
                    fc_preds.append(y_fc)
                    t_curr = t_next
                
                fc_preds = torch.stack(fc_preds)
                fc_targets = obs_norm[origin+1:origin+1+forecast_steps]
                fc_mask = obs_mask[origin+1:origin+1+forecast_steps]
                
                fc_valid = ~fc_mask.reshape(-1)
                n_fc_valid = int(fc_valid.sum().item())
                
                if n_fc_valid > 0:
                    fc_pred_flat = fc_preds.reshape(-1)[fc_valid]
                    fc_target_flat = fc_targets.reshape(-1)[fc_valid]
                    fc_sigma = sigma_obs * 2.0
                    
                    pyro.factor(
                        f"forecast_loss_{fi}", 
                        forecast_weight * dist.Normal(fc_pred_flat, fc_sigma)
                            .log_prob(fc_target_flat).sum()
                    )
        
        return z_prev
    

    
    def get_parameter_groups(self, lr_A_NL: float = 0.001, 
                              lr_neural: float = 0.005,
                              lr_other: float = 0.005) -> list:
        a_nl_params = [self.A_NL]
        neural_params = list(self._neural_residual.parameters())
        
        a_nl_ids = {id(p) for p in a_nl_params}
        neural_ids = {id(p) for p in neural_params}
        other_params = [p for p in self.parameters() 
                       if id(p) not in a_nl_ids and id(p) not in neural_ids]
        
        return [
            {"params": a_nl_params, "lr": lr_A_NL},
            {"params": neural_params, "lr": lr_neural},
            {"params": other_params, "lr": lr_other},
        ]



class VariationalKalmanGuide(PyroModule):

    def __init__(self, model: BayesianSTNeuralODE_Kalman, hidden_dim: int = 64):
        super().__init__()
        
        self.model = model
        self.K = model.K
        self.S_m = model.S_m
        

        self.sigma_obs_loc = nn.Parameter(torch.tensor(-1.0))
        self.sigma_obs_scale_unc = nn.Parameter(torch.tensor(0.0))
        
        self.sigma_process_loc = nn.Parameter(
            torch.tensor(np.log(model.process_noise_scale))
        )
        self.sigma_process_scale_unc = nn.Parameter(torch.tensor(0.0))
        

        self.z0_encoder = BasisInitialEncoder(self.K, model.initial_noise_scale)
    
    def _kalman_predict(self, m: torch.Tensor, P: torch.Tensor,
                        sigma_process: torch.Tensor,
                        t_start: float, t_end: float) -> Tuple[torch.Tensor, torch.Tensor]:

        device = m.device
        K = self.K
        

        m_prior = self.model.ode_func.integrate_step(m, t_start, t_end)
        
        Q = sigma_process ** 2 * torch.eye(K, device=device)
        P_prior = P + Q
        
        return m_prior, P_prior
    
    def _kalman_update(self, m_prior: torch.Tensor, P_prior: torch.Tensor,
                       y_obs: torch.Tensor, Phi: torch.Tensor,
                       sigma_obs: torch.Tensor,
                       obs_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        device = m_prior.device
        K = self.K
        
        valid = ~obs_mask
        n_valid = valid.sum().item()
        
        if n_valid == 0:
            return m_prior, P_prior
        
        # Extract valid observations and basis rows
        y_valid = y_obs[valid]                                
        Phi_valid = Phi[valid]                                   
        
        # Effective observation matrix: H = scale * Phi
        scale = self.model.output_scale
        bias = self.model.output_bias.squeeze()
        H_valid = scale * Phi_valid                            
        
        # Innovation
        y_pred = H_valid @ m_prior + bias                       
        innov = y_valid - y_pred                                  
        
        R = sigma_obs ** 2 * torch.eye(n_valid, device=device)
        S = H_valid @ P_prior @ H_valid.T + R                     
        
        jitter = 1e-5 * torch.eye(n_valid, device=device)
        try:
            L_S = torch.linalg.cholesky(S + jitter)
            K_gain = P_prior @ H_valid.T @ torch.cholesky_solve(
                torch.eye(n_valid, device=device), L_S
            )                                                    
        except RuntimeError:
            S_reg = S + 1e-3 * torch.eye(n_valid, device=device)
            K_gain = P_prior @ H_valid.T @ torch.linalg.inv(S_reg)
        
        # Update
        m_post = m_prior + K_gain @ innov                       
        P_post = (torch.eye(K, device=device) - K_gain @ H_valid) @ P_prior 

        P_post = 0.5 * (P_post + P_post.T)

        P_post = P_post + 1e-6 * torch.eye(K, device=device)
        
        return m_post, P_post
    
    def forward(self, data: SpatiotemporalData, prediction_mode: bool = False):

        device = data.observations.device
        T = len(data.timestamps)
        K = self.K
        
        obs_measured = data.observations[:, self.model.measured_indices]
        obs_norm = self.model.normalize(obs_measured)
        
        if data.missing_mask is not None:
            obs_mask = data.missing_mask[:, self.model.measured_indices].to(device)
        else:
            obs_mask = torch.zeros(T, self.S_m, dtype=torch.bool, device=device)
        
        t_normalized = data.timestamps.to(device).float() / self.model.global_t_max
        Phi_m = self.model.get_basis_measured()
        
        # --- Sample global noise scales ---
        sigma_obs = pyro.sample(
            "sigma_obs",
            dist.LogNormal(self.sigma_obs_loc,
                          F.softplus(self.sigma_obs_scale_unc) + 0.01)
        )
        sigma_process = pyro.sample(
            "sigma_process",
            dist.LogNormal(self.sigma_process_loc,
                          F.softplus(self.sigma_process_scale_unc) + 0.01)
        )
        
        # --- Initialize from first observation ---
        z0_mean, z0_std = self.z0_encoder(
            obs_norm[0], Phi_m, sigma_obs, 
            obs_mask[0] if obs_mask is not None else None
        )
        z_0 = pyro.sample("z_0", dist.Normal(z0_mean, z0_std).to_event(1))
        
        m_curr = z_0
        P_curr = torch.diag(z0_std ** 2)              
        
        # --- Predict-update-sample recursion ---
        for t in range(1, T):
            # Predict: ODE propagation + process noise
            m_prior, P_prior = self._kalman_predict(
                m_curr, P_curr, sigma_process,
                t_normalized[t-1].item(), t_normalized[t].item()
            )
            
            # Update: Kalman correction with observations
            m_post, P_post = self._kalman_update(
                m_prior, P_prior,
                obs_norm[t], Phi_m, sigma_obs,
                obs_mask[t]
            )
            
            # Sample from posterior
            try:
                L_post = torch.linalg.cholesky(P_post)
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.MultivariateNormal(m_post, scale_tril=L_post)
                )
            except RuntimeError:
                P_diag = P_post.diag().clamp(min=1e-6)
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.Normal(m_post, P_diag.sqrt()).to_event(1)
                )
            
            m_curr = z_t
            P_curr = P_post