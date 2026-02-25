"""
Baseline models for NLBST comparison.

| Baseline       | Spatial basis | Continuous-time | Bayesian UQ  | Nonlocal     |
|----------------|---------------|-----------------|--------------|--------------|
| Linear-DSTM    | ✓ (same)      | ✗ (discrete)    | ✓ (Kalman)   | ✗ (linear A) |
| GRU-D          | ✗             | △ (decay)       | ✗ (dropout)  | ✗            |
| Latent-ODE     | ✗             | ✓ (Neural ODE)  | △ (VAE)      | ✗            |
| FNO            | ✗ (Fourier)   | ✗ (discrete)    | ✗ (dropout)  | △ (global)   |
| GraFITi        | ✗ (bipartite) | ✗ (attention)   | ✗ (dropout)  | △ (graph)    |
| APN            | ✗ (patches)   | ✗ (TAPA)        | ✗ (dropout)  | ✗            |

GraFITi: Cho et al., AAAI 2024.
APN:     Liu et al., arXiv 2025.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import SpatiotemporalData
from .components import SpatialBasisFunctions, SimpleODEFunc


# ===================================================================
#  Shared: GP kriging weights for spatial interpolation
# ===================================================================

def build_gp_weights(measured_coords: torch.Tensor,
                     unmeasured_coords: torch.Tensor,
                     length_scale: float = 0.3,
                     nugget: float = 1e-2) -> torch.Tensor:
    """Precompute RBF-GP kriging weights: W = K(X_u, X_m) @ K(X_m, X_m)^{-1}."""
    S_m = measured_coords.shape[0]
    S_u = unmeasured_coords.shape[0]
    if S_u == 0:
        return torch.zeros(0, S_m, device=measured_coords.device)

    dist_mm = torch.cdist(measured_coords, measured_coords)
    K_mm = torch.exp(-0.5 * dist_mm ** 2 / length_scale ** 2)
    K_mm = K_mm + nugget * torch.eye(S_m, device=K_mm.device)

    dist_um = torch.cdist(unmeasured_coords, measured_coords)
    K_um = torch.exp(-0.5 * dist_um ** 2 / length_scale ** 2)

    try:
        L = torch.linalg.cholesky(K_mm)
        W = torch.cholesky_solve(K_um.T, L).T
    except RuntimeError:
        W = torch.linalg.solve(K_mm, K_um.T).T

    return W


def gp_spatial_interpolate(measured_values: torch.Tensor,
                           gp_weights: torch.Tensor) -> torch.Tensor:
    if gp_weights.shape[0] == 0:
        return torch.zeros(measured_values.shape[0], 0,
                           device=measured_values.device)
    return measured_values @ gp_weights.T


# ===================================================================
#  1. Linear-DSTM
# ===================================================================

class LinearDSTMBaseline(nn.Module):
    """
    Linear Dynamic Spatio-Temporal Model.
    z_t = A z_{t-1} + ε_t,  y_t = Φ z_t + ν_t
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 num_basis: int = 16,
                 length_scale: float = 1.0):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.K = num_basis

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        self.spatial_basis = SpatialBasisFunctions(
            num_basis=num_basis,
            spatial_coords=spatial_coords[measured_indices],
            basis_type='rbf',
            length_scale=length_scale,
            learn_basis=False
        )

        self.A = nn.Parameter(
            torch.eye(num_basis) * 0.9 + torch.randn(num_basis, num_basis) * 0.05)

        self.log_sigma_obs = nn.Parameter(torch.tensor(-1.0))
        self.log_sigma_process = nn.Parameter(torch.tensor(-2.0))

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.zeros(1))

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

    def fit(self, train_data: SpatiotemporalData, num_epochs: int = 200,
            lr: float = 0.01):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
            data_min = torch.zeros(self.S_m, device=device)
            data_max = torch.zeros(self.S_m, device=device)
            for s in range(self.S_m):
                valid_vals = obs_m[~mask_m[:, s], s]
                if len(valid_vals) > 0:
                    data_min[s] = valid_vals.min()
                    data_max[s] = valid_vals.max()
                else:
                    data_min[s], data_max[s] = 0.0, 1.0
            self.data_min.copy_(data_min)
            self.data_max.copy_(data_max)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)
            self.data_min.copy_(obs_m.min(dim=0)[0])
            self.data_max.copy_(obs_m.max(dim=0)[0])

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            Phi_m = self.spatial_basis(self.measured_coords)
            obs_norm = (obs_m - self.data_min) / (self.data_max - self.data_min + 1e-6)

            sigma_obs = torch.exp(self.log_sigma_obs)
            sigma_process = torch.exp(self.log_sigma_process)

            T = len(train_data.timestamps)
            z_mean = torch.zeros(self.K, device=device)
            z_cov = torch.eye(self.K, device=device)

            nll = 0.0
            n_obs = 0

            for t in range(T):
                if t > 0:
                    z_mean = self.A @ z_mean
                    z_cov = (self.A @ z_cov @ self.A.T
                             + sigma_process**2 * torch.eye(self.K, device=device))

                valid_mask = ~mask_m[t]
                n_valid = valid_mask.sum().item()
                if n_valid == 0:
                    continue

                y_valid = obs_norm[t][valid_mask]
                Phi_valid = Phi_m[valid_mask]
                y_pred = Phi_valid @ z_mean * self.output_scale + self.output_bias
                innov = y_valid - y_pred

                S = (Phi_valid @ z_cov @ Phi_valid.T
                     + sigma_obs**2 * torch.eye(n_valid, device=device))
                try:
                    L = torch.linalg.cholesky(
                        S + 1e-4 * torch.eye(n_valid, device=device))
                    nll += 0.5 * (innov @ torch.cholesky_solve(
                        innov.unsqueeze(1), L).squeeze())
                    nll += torch.sum(torch.log(torch.diag(L)))
                except Exception:
                    nll += 0.5 * (innov ** 2 / sigma_obs**2).sum()

                n_obs += n_valid

                K_gain = z_cov @ Phi_valid.T @ torch.linalg.inv(
                    S + 1e-4 * torch.eye(n_valid, device=device))
                z_mean = z_mean + K_gain @ innov
                z_cov = (torch.eye(self.K, device=device)
                         - K_gain @ Phi_valid) @ z_cov

            if n_obs > 0:
                loss = nll / n_obs
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        Phi_m = self.spatial_basis(self.measured_coords)
        Phi_u = (self.spatial_basis(self.unmeasured_coords)
                 if self.S_u > 0 else None)

        obs_norm = (context_obs - self.data_min) / (self.data_max - self.data_min + 1e-6)
        sigma_obs = torch.exp(self.log_sigma_obs)
        sigma_process = torch.exp(self.log_sigma_process)

        z_mean = torch.zeros(self.K, device=device)
        z_cov = torch.eye(self.K, device=device)

        for t in range(len(context_times)):
            if t > 0:
                z_mean = self.A @ z_mean
                z_cov = (self.A @ z_cov @ self.A.T
                         + sigma_process**2 * torch.eye(self.K, device=device))

            valid_mask = ~context_mask[t]
            n_valid = valid_mask.sum().item()
            if n_valid == 0:
                continue

            y_valid = obs_norm[t][valid_mask]
            Phi_valid = Phi_m[valid_mask]
            y_pred = Phi_valid @ z_mean * self.output_scale + self.output_bias
            innov = y_valid - y_pred
            S = (Phi_valid @ z_cov @ Phi_valid.T
                 + sigma_obs**2 * torch.eye(n_valid, device=device))
            K_gain = z_cov @ Phi_valid.T @ torch.linalg.inv(
                S + 1e-4 * torch.eye(n_valid, device=device))
            z_mean = z_mean + K_gain @ innov
            z_cov = (torch.eye(self.K, device=device)
                     - K_gain @ Phi_valid) @ z_cov

        H = len(forecast_times)
        with torch.no_grad():
            samples_m = []
            samples_u = []

            for _ in range(num_samples):
                try:
                    L = torch.linalg.cholesky(
                        z_cov + 1e-4 * torch.eye(self.K, device=device))
                    z = z_mean + L @ torch.randn(self.K, device=device)
                except Exception:
                    z = z_mean + 0.1 * torch.randn(self.K, device=device)

                preds_m, preds_u = [], []
                for h in range(H):
                    z = (self.A @ z
                         + sigma_process * torch.randn(self.K, device=device))
                    y_m = Phi_m @ z * self.output_scale + self.output_bias
                    y_m = y_m * (self.data_max - self.data_min) + self.data_min
                    preds_m.append(y_m)

                    if Phi_u is not None and self.S_u > 0:
                        y_u = Phi_u @ z * self.output_scale + self.output_bias
                        y_u = (y_u * (self.global_max - self.global_min)
                               + self.global_min)
                        preds_u.append(y_u)

                samples_m.append(torch.stack(preds_m))
                if preds_u:
                    samples_u.append(torch.stack(preds_u))

            result = {'measured': torch.stack(samples_m)}
            if samples_u:
                result['unmeasured'] = torch.stack(samples_u)
            return result


# ===================================================================
#  2. GRU-D
# ===================================================================

class GRUDBaseline(nn.Module):
    """
    GRU-D: GRU with trainable Decays for irregular time series.
    (Che et al., Scientific Reports 2018)
    UQ via MC dropout (added post-GRU).
    Spatial prediction via GP kriging.
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.hidden_dim = hidden_dim

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        input_dim = self.S_m * 3
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.W_gamma_x = nn.Linear(self.S_m, self.S_m, bias=False)
        self.W_gamma_h = nn.Linear(self.S_m, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_measured = nn.Linear(hidden_dim, self.S_m)

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

        gp_w = build_gp_weights(spatial_coords[measured_indices],
                                spatial_coords[unmeasured_indices]
                                if len(unmeasured_indices) > 0
                                else torch.zeros(0, spatial_coords.shape[1]))
        self.register_buffer('_gp_weights', gp_w)

    def fit(self, train_data: SpatiotemporalData, num_epochs: int = 200,
            lr: float = 0.001):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]
        T = len(train_data.timestamps)

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)

        for s in range(self.S_m):
            valid_vals = obs_m[:, s][~mask_m[:, s]]
            if len(valid_vals) > 0:
                self.data_min[s] = valid_vals.min()
                self.data_max[s] = valid_vals.max()

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (obs_m - self.data_min) / data_range

        norm_mean = torch.zeros(self.S_m, device=device)
        for s in range(self.S_m):
            valid_vals = obs_norm[:, s][~mask_m[:, s]]
            norm_mean[s] = valid_vals.mean() if len(valid_vals) > 0 else 0.5

        times = train_data.timestamps
        delta = torch.zeros_like(obs_m)
        for t in range(1, T):
            dt = (times[t] - times[t-1]).item()
            delta[t] = delta[t-1] + dt
            delta[t][~mask_m[t-1]] = dt

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            h = torch.zeros(1, self.hidden_dim, device=device)
            x_last = norm_mean.clone()

            loss = 0.0
            n_obs = 0

            for t in range(T):
                gamma_x = torch.sigmoid(self.W_gamma_x(delta[t:t+1]))
                x_imputed = (mask_m[t].float() * norm_mean
                             + (1 - mask_m[t].float()) * obs_norm[t])
                x_imputed = (gamma_x.squeeze() * x_last
                             + (1 - gamma_x.squeeze()) * x_imputed)

                gamma_h = torch.sigmoid(self.W_gamma_h(delta[t:t+1]))
                h = gamma_h * h

                gru_input = torch.cat([
                    x_imputed.unsqueeze(0),
                    (~mask_m[t]).float().unsqueeze(0),
                    delta[t:t+1] / (delta.max() + 1e-6)
                ], dim=1)
                h = self.gru_cell(gru_input, h)
                h_dropped = self.dropout(h)

                y_pred = self.output_measured(h_dropped).squeeze()
                valid_mask = ~mask_m[t]
                if valid_mask.any():
                    loss += (F.mse_loss(y_pred[valid_mask],
                                        obs_norm[t][valid_mask])
                             * valid_mask.sum())
                    n_obs += valid_mask.sum().item()

                x_last = torch.where(mask_m[t], x_last, obs_norm[t])

            if n_obs > 0:
                (loss / n_obs).backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        T_ctx = len(context_times)
        H = len(forecast_times)

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (context_obs - self.data_min) / data_range

        norm_mean = torch.zeros(self.S_m, device=device)
        for s in range(self.S_m):
            valid_vals = obs_norm[:, s][~context_mask[:, s]]
            norm_mean[s] = valid_vals.mean() if len(valid_vals) > 0 else 0.5

        self.train()
        with torch.no_grad():
            samples_m = []
            for _ in range(num_samples):
                h = torch.zeros(1, self.hidden_dim, device=device)
                x_last = norm_mean.clone()
                delta_last = torch.zeros(self.S_m, device=device)
                delta_max = 1.0

                for t in range(T_ctx):
                    if t > 0:
                        dt = (context_times[t] - context_times[t-1]).item()
                        delta_last = delta_last + dt
                        delta_last[~context_mask[t-1]] = dt
                        delta_max = max(delta_max, delta_last.max().item())

                    gamma_x = torch.sigmoid(
                        self.W_gamma_x(delta_last.unsqueeze(0)))
                    x_imputed = (context_mask[t].float() * norm_mean
                                 + (1 - context_mask[t].float()) * obs_norm[t])
                    x_imputed = (gamma_x.squeeze() * x_last
                                 + (1 - gamma_x.squeeze()) * x_imputed)

                    gamma_h = torch.sigmoid(
                        self.W_gamma_h(delta_last.unsqueeze(0)))
                    h = gamma_h * h

                    gru_input = torch.cat([
                        x_imputed.unsqueeze(0),
                        (~context_mask[t]).float().unsqueeze(0),
                        delta_last.unsqueeze(0) / (delta_max + 1e-6)
                    ], dim=1)
                    h = self.gru_cell(gru_input, h)
                    x_last = torch.where(context_mask[t], x_last, obs_norm[t])

                preds_m = []
                t_last = context_times[-1]
                for t_fut in forecast_times:
                    dt = (t_fut - t_last).item()
                    delta_last = delta_last + dt

                    gamma_h = torch.sigmoid(
                        self.W_gamma_h(delta_last.unsqueeze(0)))
                    h = gamma_h * h
                    gamma_x = torch.sigmoid(
                        self.W_gamma_x(delta_last.unsqueeze(0)))
                    x_imputed = (gamma_x.squeeze() * x_last
                                 + (1 - gamma_x.squeeze()) * norm_mean)

                    gru_input = torch.cat([
                        x_imputed.unsqueeze(0),
                        torch.zeros(1, self.S_m, device=device),
                        delta_last.unsqueeze(0) / (delta_max + 1e-6)
                    ], dim=1)
                    h = self.gru_cell(gru_input, h)
                    h_dropped = self.dropout(h)

                    y_m_norm = self.output_measured(h_dropped).squeeze()
                    y_m = y_m_norm * data_range + self.data_min
                    preds_m.append(y_m)
                    x_last = y_m_norm
                    t_last = t_fut

                samples_m.append(torch.stack(preds_m))

            samples_m = torch.stack(samples_m)

        self.eval()

        result = {'measured': samples_m}
        if self.S_u > 0:
            samples_u = torch.stack([
                gp_spatial_interpolate(samples_m[i], self._gp_weights)
                for i in range(num_samples)
            ])
            result['unmeasured'] = samples_u
        return result


# ===================================================================
#  3. Latent ODE
# ===================================================================

class LatentODEBaseline(nn.Module):
    """
    Latent ODE (Rubanova et al., NeurIPS 2019).
    ODE-RNN encoder → latent z₀ → ODE decoder → output.
    UQ via VAE posterior sampling.
    Spatial prediction via GP kriging.
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 latent_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        self.encoder_gru = nn.GRUCell(self.S_m * 2 + 1, hidden_dim)
        self.encoder_ode = SimpleODEFunc(hidden_dim)

        self.z0_mean = nn.Linear(hidden_dim, latent_dim)
        self.z0_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_ode = SimpleODEFunc(latent_dim)
        self.output_measured = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.S_m)
        )

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

        gp_w = build_gp_weights(spatial_coords[measured_indices],
                                spatial_coords[unmeasured_indices]
                                if len(unmeasured_indices) > 0
                                else torch.zeros(0, spatial_coords.shape[1]))
        self.register_buffer('_gp_weights', gp_w)

    def fit(self, train_data: SpatiotemporalData, num_epochs: int = 200,
            lr: float = 0.001):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]
        T = len(train_data.timestamps)
        times = train_data.timestamps / train_data.timestamps[-1]

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)

        for s in range(self.S_m):
            valid_vals = obs_m[:, s][~mask_m[:, s]]
            if len(valid_vals) > 0:
                self.data_min[s] = valid_vals.min()
                self.data_max[s] = valid_vals.max()

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (obs_m - self.data_min) / data_range

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            h = torch.zeros(1, self.hidden_dim, device=device)
            for t in range(T):
                if t > 0:
                    h = self.encoder_ode.integrate_step(
                        h.squeeze(), times[t-1].item(), times[t].item()
                    ).unsqueeze(0)
                if mask_m[t].all():
                    continue
                x_input = obs_norm[t:t+1].clone()
                x_input[:, mask_m[t]] = 0
                mask_input = (~mask_m[t]).float().unsqueeze(0)
                dt_input = torch.tensor([[times[t].item()]], device=device)
                gru_input = torch.cat([x_input, mask_input, dt_input], dim=1)
                h = self.encoder_gru(gru_input, h)

            z_mean = self.z0_mean(h)
            z_logvar = self.z0_logvar(h).clamp(-5, 2)
            z_std = torch.exp(0.5 * z_logvar)
            z = (z_mean + z_std * torch.randn_like(z_std)).squeeze()

            recon_loss = 0.0
            n_obs = 0

            for t in range(T):
                if t > 0:
                    z = self.decoder_ode.integrate_step(
                        z, times[t-1].item(), times[t].item())
                y_pred = self.output_measured(z)
                valid_mask = ~mask_m[t]
                if valid_mask.any():
                    recon_loss += (F.mse_loss(y_pred[valid_mask],
                                              obs_norm[t][valid_mask])
                                  * valid_mask.sum())
                    n_obs += valid_mask.sum().item()

            kl_loss = -0.5 * torch.sum(
                1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

            if n_obs > 0:
                loss = recon_loss / n_obs + 0.001 * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        T_ctx = len(context_times)

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (context_obs - self.data_min) / data_range
        t_max = max(context_times[-1].item(), forecast_times[-1].item())
        times_norm = context_times / t_max
        forecast_times_norm = forecast_times / t_max

        with torch.no_grad():
            samples_m = []
            for _ in range(num_samples):
                h = torch.zeros(1, self.hidden_dim, device=device)
                for t in range(T_ctx):
                    if t > 0:
                        h = self.encoder_ode.integrate_step(
                            h.squeeze(), times_norm[t-1].item(),
                            times_norm[t].item()
                        ).unsqueeze(0)
                    if context_mask[t].all():
                        continue
                    x_input = obs_norm[t:t+1].clone()
                    x_input[:, context_mask[t]] = 0
                    mask_input = (~context_mask[t]).float().unsqueeze(0)
                    dt_input = torch.tensor(
                        [[times_norm[t].item()]], device=device)
                    gru_input = torch.cat(
                        [x_input, mask_input, dt_input], dim=1)
                    h = self.encoder_gru(gru_input, h)

                z_mean = self.z0_mean(h)
                z_logvar = self.z0_logvar(h).clamp(-5, 2)
                z = (z_mean + torch.exp(0.5 * z_logvar)
                     * torch.randn_like(z_mean)).squeeze()

                for t in range(T_ctx):
                    if t > 0:
                        z = self.decoder_ode.integrate_step(
                            z, times_norm[t-1].item(), times_norm[t].item())

                preds_m = []
                t_last = times_norm[-1].item()
                for t_fut in forecast_times_norm:
                    z = self.decoder_ode.integrate_step(
                        z, t_last, t_fut.item())
                    y_m_norm = self.output_measured(z)
                    y_m = y_m_norm * data_range + self.data_min
                    preds_m.append(y_m)
                    t_last = t_fut.item()
                samples_m.append(torch.stack(preds_m))

            samples_m = torch.stack(samples_m)

            result = {'measured': samples_m}
            if self.S_u > 0:
                samples_u = torch.stack([
                    gp_spatial_interpolate(samples_m[i], self._gp_weights)
                    for i in range(num_samples)
                ])
                result['unmeasured'] = samples_u
            return result


# ===================================================================
#  4. FNO
# ===================================================================

class _SpectralConv1d(nn.Module):
    """1-D Fourier layer: FFT → linear in frequency → iFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes,
                                dtype=torch.cfloat))

    def forward(self, x):
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = torch.einsum(
            'bct,cot->bot', x_ft[:, :, :modes], self.weights[:, :, :modes])
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class _FourierBlock(nn.Module):
    """One Fourier block: spectral conv + pointwise linear + residual."""

    def __init__(self, width: int, modes: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = _SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.InstanceNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(
            F.gelu(self.norm(self.spectral(x) + self.pointwise(x))))


class FNOBaseline(nn.Module):
    """
    Fourier Neural Operator for sensor time-series forecasting.
    (Li et al., ICLR 2021)
    UQ via MC dropout.
    Spatial prediction via GP kriging.
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 width: int = 64,
                 modes: int = 16,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.width = width

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        self.lift = nn.Linear(self.S_m + 1, width)
        self.blocks = nn.ModuleList(
            [_FourierBlock(width, modes, dropout) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, self.S_m),
        )

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

        gp_w = build_gp_weights(spatial_coords[measured_indices],
                                spatial_coords[unmeasured_indices]
                                if len(unmeasured_indices) > 0
                                else torch.zeros(0, spatial_coords.shape[1]))
        self.register_buffer('_gp_weights', gp_w)

    def _prepare_input(self, obs_norm, mask):
        x = obs_norm.clone()
        x[mask] = 0.0
        obs_frac = (~mask).float().mean(dim=1, keepdim=True)
        x_in = torch.cat([x, obs_frac], dim=1)
        x_lifted = self.lift(x_in)
        return x_lifted.unsqueeze(0).permute(0, 2, 1)

    def fit(self, train_data: SpatiotemporalData, num_epochs: int = 200,
            lr: float = 0.001):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]
        T = len(train_data.timestamps)

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)

        for s in range(self.S_m):
            valid_vals = obs_m[:, s][~mask_m[:, s]]
            if len(valid_vals) > 0:
                self.data_min[s] = valid_vals.min()
                self.data_max[s] = valid_vals.max()

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (obs_m - self.data_min) / data_range

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        win = min(64, T - 1)

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            t0 = np.random.randint(0, max(T - win - 1, 1))
            t1 = min(t0 + win, T - 1)

            x_in = self._prepare_input(obs_norm[t0:t1], mask_m[t0:t1])
            h = x_in
            for block in self.blocks:
                h = block(h)
            h = h.permute(0, 2, 1)
            y_pred = self.proj(h).squeeze(0)

            target = obs_norm[t0+1:t1+1]
            target_mask = mask_m[t0+1:t1+1]
            valid = ~target_mask

            if valid.any():
                loss = F.mse_loss(y_pred[valid], target[valid])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        H_len = len(forecast_times)

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (context_obs - self.data_min) / data_range

        self.train()
        with torch.no_grad():
            samples_m = []
            for _ in range(num_samples):
                current_seq = obs_norm.clone()
                current_mask = context_mask.clone()

                preds_m = []
                for k in range(H_len):
                    x_in = self._prepare_input(current_seq, current_mask)
                    h = x_in
                    for block in self.blocks:
                        h = block(h)
                    h = h.permute(0, 2, 1)
                    y_all = self.proj(h).squeeze(0)

                    y_next_norm = y_all[-1]
                    y_next_raw = y_next_norm * data_range + self.data_min
                    preds_m.append(y_next_raw)

                    current_seq = torch.cat(
                        [current_seq, y_next_norm.unsqueeze(0)], dim=0)
                    current_mask = torch.cat(
                        [current_mask,
                         torch.zeros(1, self.S_m, dtype=torch.bool,
                                     device=device)], dim=0)

                samples_m.append(torch.stack(preds_m))
            samples_m = torch.stack(samples_m)

        self.eval()

        result = {'measured': samples_m}
        if self.S_u > 0:
            with torch.no_grad():
                samples_u = torch.stack([
                    gp_spatial_interpolate(samples_m[i], self._gp_weights)
                    for i in range(num_samples)
                ])
            result['unmeasured'] = samples_u
        return result


# ===================================================================
#  5. GraFITi  (Cho et al., AAAI 2024)
# ===================================================================

class _MAB2(nn.Module):
    """Multi-head Attention Block (from official GraFITi)."""
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        Att_mat = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(
                mask.repeat(self.num_heads, 1, 1) == 0, -1e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class _GraFITiEncoder(nn.Module):
    """GraFITi Encoder: bipartite graph with observation edges ↔ time/channel nodes."""
    def __init__(self, dim, nkernel=128, n_layers=2, attn_head=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nheads = attn_head
        self.nkernel = nkernel
        self.n_layers = n_layers

        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)

        self.channel_time_attn = nn.ModuleList()
        self.time_channel_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        for _ in range(n_layers):
            self.channel_time_attn.append(
                _MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.time_channel_attn.append(
                _MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.edge_nn.append(nn.Linear(3 * nkernel, nkernel))

        self.output = nn.Sequential(
            nn.Linear(3 * nkernel, nkernel),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nkernel, 1),
        )
        self.relu = nn.ReLU()

    def gather(self, x, inds):
        return x.gather(
            dim=1, index=inds[:, :, None].repeat(1, 1, x.shape[-1]))

    def forward(self, x_y_mark, value, mask, target_value, target_mask):
        B, L, C = value.shape
        device = value.device

        time_indices = torch.arange(L, device=device).view(1, L, 1).expand(B, L, C)
        channel_indices = torch.arange(C, device=device).view(1, 1, C).expand(B, L, C)
        mask_bool = mask.to(torch.bool)

        N_obs_max = max(int(mask.sum((1, 2)).max().item()), 1)

        def pad(v, ml):
            n = v.shape[0]
            return v[:ml] if n >= ml else F.pad(v, (0, ml - n), value=0)

        val_fl, ti_fl, ci_fl, m_fl, tv_fl, tm_fl = [], [], [], [], [], []
        for b in range(B):
            m = mask_bool[b]
            val_fl.append(pad(value[b][m], N_obs_max))
            ti_fl.append(pad(time_indices[b][m], N_obs_max))
            ci_fl.append(pad(channel_indices[b][m], N_obs_max))
            m_fl.append(pad(mask[b][m], N_obs_max))
            tv_fl.append(pad(target_value[b][m], N_obs_max))
            tm_fl.append(pad(target_mask[b][m], N_obs_max))

        value_flat = torch.stack(val_fl)
        time_idx_flat = torch.stack(ti_fl).long()
        chan_idx_flat = torch.stack(ci_fl).long()
        mask_flat = torch.stack(m_fl)
        tgt_val_flat = torch.stack(tv_fl)
        tgt_mask_flat = torch.stack(tm_fl)

        lookback_flag = 1 - mask_flat + tgt_mask_flat
        edge_feat = torch.stack([value_flat, lookback_flag], dim=-1)

        chan_ids = torch.arange(C, device=device).view(1, C, 1).expand(B, C, N_obs_max)
        channel_mask = (chan_ids == chan_idx_flat.unsqueeze(1).expand(B, C, N_obs_max)).float()
        channel_mask = channel_mask * mask_flat.unsqueeze(1).expand(B, C, N_obs_max)

        time_ids = torch.arange(L, device=device).view(1, L, 1).expand(B, L, N_obs_max)
        time_mask = (time_idx_flat.unsqueeze(1).expand(B, L, N_obs_max) == time_ids).float()
        time_mask = time_mask * mask_flat.unsqueeze(1).expand(B, L, N_obs_max)

        edge_emb = (self.relu(self.edge_init(edge_feat))
                    * mask_flat.unsqueeze(-1).expand(B, N_obs_max, self.nkernel))
        time_emb = torch.sin(self.time_init(x_y_mark))
        chan_onehot = (F.one_hot(torch.arange(C, device=device), C)
                      .float().unsqueeze(0).expand(B, C, C))
        chan_emb = self.relu(self.chan_init(chan_onehot))

        for i in range(self.n_layers):
            q_c = chan_emb
            k_t = self.gather(time_emb, time_idx_flat)
            k = torch.cat([k_t, edge_emb], dim=-1)
            C__ = self.channel_time_attn[i](q_c, k, channel_mask)

            q_t = time_emb
            k_c = self.gather(chan_emb, chan_idx_flat)
            k = torch.cat([k_c, edge_emb], dim=-1)
            T__ = self.time_channel_attn[i](q_t, k, time_mask)

            edge_emb = (self.relu(
                edge_emb + self.edge_nn[i](
                    torch.cat([edge_emb, k_t, k_c], dim=-1)))
                * mask_flat.unsqueeze(-1).expand(B, N_obs_max, self.nkernel))

            chan_emb = C__
            time_emb = T__

        k_t = self.gather(time_emb, time_idx_flat)
        k_c = self.gather(chan_emb, chan_idx_flat)
        output = self.output(torch.cat([edge_emb, k_t, k_c], dim=-1))

        return output, tgt_val_flat, tgt_mask_flat


class GraFITiBaseline(nn.Module):
    """
    GraFITi baseline (Cho et al., AAAI 2024).
    UQ via MC dropout. Spatial prediction via GP kriging.
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 d_model: int = 128,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        self.encoder = _GraFITiEncoder(
            dim=self.S_m, nkernel=d_model, n_layers=n_layers,
            attn_head=n_heads, dropout=dropout,
        )

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

        gp_w = build_gp_weights(spatial_coords[measured_indices],
                                spatial_coords[unmeasured_indices]
                                if len(unmeasured_indices) > 0
                                else torch.zeros(0, spatial_coords.shape[1]))
        self.register_buffer('_gp_weights', gp_w)

    def _prepare_inputs(self, ctx_obs_norm, ctx_mask, ctx_times_norm,
                        tgt_times_norm, H):
        device = ctx_obs_norm.device
        T_ctx = ctx_obs_norm.shape[0]
        L = T_ctx + H

        value = torch.zeros(1, L, self.S_m, device=device)
        value[0, :T_ctx] = ctx_obs_norm * (~ctx_mask).float()

        mask = torch.zeros(1, L, self.S_m, device=device)
        mask[0, :T_ctx] = (~ctx_mask).float()
        mask[0, T_ctx:] = 1.0

        target_value = torch.zeros(1, L, self.S_m, device=device)
        target_mask = torch.zeros(1, L, self.S_m, device=device)
        target_mask[0, T_ctx:] = 1.0

        all_times = torch.cat([ctx_times_norm, tgt_times_norm])
        x_y_mark = all_times.view(1, L, 1)

        return x_y_mark, value, mask, target_value, target_mask

    def _unpad_and_reshape(self, output, mask, L, C):
        device = output.device
        result = torch.zeros(1, L, C, device=device)
        mask_bool = mask[0].bool()
        n_valid = mask_bool.sum().item()
        if n_valid > 0:
            result[0][mask_bool] = output[0, :n_valid, 0]
        return result

    def fit(self, train_data, num_epochs: int = 200, lr: float = 1e-3,
            ctx_len: int = None, forecast_h: int = None):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]
        T = len(train_data.timestamps)

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)

        for s in range(self.S_m):
            valid_vals = obs_m[:, s][~mask_m[:, s]]
            if len(valid_vals) > 0:
                self.data_min[s] = valid_vals.min()
                self.data_max[s] = valid_vals.max()
            else:
                self.data_min[s], self.data_max[s] = 0.0, 1.0

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (obs_m - self.data_min) / data_range
        times = train_data.timestamps.float()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs)

        self.train()
        if ctx_len is None:
            ctx_len = min(50, max(10, T // 3))
        if forecast_h is None:
            forecast_h = max(1, min(5, T // 10))
        ctx_len = min(ctx_len, T - forecast_h)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_windows = 0

            n_wins = min(8, max(1, T // forecast_h))
            max_start = max(0, T - ctx_len - forecast_h)
            starts = torch.randint(0, max_start + 1, (n_wins,))

            for start_val in starts:
                start = start_val.item()
                ctx_end = start + ctx_len
                H = min(forecast_h, T - ctx_end)
                if H < 1:
                    continue

                ctx_obs = obs_norm[start:ctx_end]
                ctx_mask = mask_m[start:ctx_end]
                tgt_obs = obs_norm[ctx_end:ctx_end + H]
                tgt_mask = mask_m[ctx_end:ctx_end + H]

                win_times = times[start:ctx_end + H]
                t_start, t_end = win_times[0], win_times[-1]
                ctx_times = ((times[start:ctx_end] - t_start)
                             / (t_end - t_start + 1e-8))
                tgt_times = ((times[ctx_end:ctx_end + H] - t_start)
                             / (t_end - t_start + 1e-8))

                x_y_mark, value, mask, target_value, target_mask = \
                    self._prepare_inputs(
                        ctx_obs, ctx_mask, ctx_times, tgt_times, H)

                output, _, _ = self.encoder(
                    x_y_mark, value, mask, target_value, target_mask)

                L_total = ctx_len + H
                result_grid = self._unpad_and_reshape(
                    output, mask, L_total, self.S_m)
                pred_tgt = result_grid[0, ctx_len:]

                observed_tgt = ~tgt_mask
                if observed_tgt.any():
                    loss = F.mse_loss(pred_tgt[observed_tgt],
                                      tgt_obs[observed_tgt])
                else:
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_windows += 1

            scheduler.step()

            if (epoch + 1) % 50 == 0 and n_windows > 0:
                print(f"  [GraFITi] Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {epoch_loss / n_windows:.6f}")

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        T_ctx = len(context_times)
        H = len(forecast_times)

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (context_obs - self.data_min) / data_range

        all_times = torch.cat([context_times, forecast_times]).float()
        t_min, t_max = all_times.min(), all_times.max()
        ctx_times_norm = ((context_times.float() - t_min)
                          / (t_max - t_min + 1e-8))
        tgt_times_norm = ((forecast_times.float() - t_min)
                          / (t_max - t_min + 1e-8))

        x_y_mark, value, mask, target_value, target_mask = \
            self._prepare_inputs(
                obs_norm, context_mask, ctx_times_norm, tgt_times_norm, H)

        self.train()
        samples_m = []

        with torch.no_grad():
            for _ in range(num_samples):
                output, _, _ = self.encoder(
                    x_y_mark, value, mask, target_value, target_mask)
                L = T_ctx + H
                result_grid = self._unpad_and_reshape(
                    output, mask, L, self.S_m)
                pred_norm = result_grid[0, T_ctx:]
                pred_raw = pred_norm * data_range + self.data_min
                samples_m.append(pred_raw)

        self.eval()
        samples_m = torch.stack(samples_m)

        result = {'measured': samples_m}
        if self.S_u > 0:
            samples_u = torch.stack([
                gp_spatial_interpolate(samples_m[i], self._gp_weights)
                for i in range(num_samples)
            ])
            result['unmeasured'] = samples_u
        return result


# ===================================================================
#  6. APN  (Liu et al., arXiv 2025)
# ===================================================================

class _LearnableTE(nn.Module):
    """Learnable Time Encoding: linear scale + periodic sin."""
    def __init__(self, te_dim):
        super().__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)

    def forward(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], dim=-1)


class _AttentionPatchAggregation(nn.Module):
    """TAPA: sigmoid-boundary soft patches with per-variable learnable boundaries."""
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__()
        self.N = N
        self.P = P
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.feature_dim = 1 + te_dim

        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(
            torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))

        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        device = t_stacked.device
        B_N, L, _ = t_stacked.shape
        B = B_N // self.N

        patch_centers = torch.linspace(
            self.S / 2, self.history - self.S / 2, self.P, device=device)
        base_left = (patch_centers - self.S / 2).unsqueeze(0)
        t_left = base_left + self.delta_left_params
        width = torch.exp(self.raw_log_width_params) + 1e-6
        t_right = t_left + width
        taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6

        t_left_bn = (t_left.unsqueeze(0).expand(B, -1, -1)
                     .reshape(B_N, self.P).unsqueeze(-1))
        t_right_bn = (t_right.unsqueeze(0).expand(B, -1, -1)
                      .reshape(B_N, self.P).unsqueeze(-1))
        taus_bn = (taus.unsqueeze(0).expand(B, -1, -1)
                   .reshape(B_N, 1).unsqueeze(-1))

        t_raw = t_stacked.transpose(-1, -2)
        weights_raw = (torch.sigmoid((t_right_bn - t_raw) / taus_bn)
                       * torch.sigmoid((t_raw - t_left_bn) / taus_bn))

        mask_t = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_t
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        weighted_sum = torch.bmm(temporal_weights, x_with_te)
        h_avg = weighted_sum / sum_weights

        h_proj = self.projection_layer(h_avg)
        h_patches = self.norm(h_proj + self.ffn(h_proj))
        return h_patches


class APNBaseline(nn.Module):
    """
    APN baseline (Liu et al., arXiv 2025).
    TAPA + attention aggregation + time-query decoder.
    UQ via MC dropout. Spatial prediction via GP kriging.
    """

    def __init__(self,
                 num_locations: int,
                 spatial_coords: torch.Tensor,
                 measured_indices: torch.Tensor,
                 unmeasured_indices: torch.Tensor,
                 d_model: int = 128,
                 te_dim: int = 16,
                 num_patches: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.S = num_locations
        self.S_m = len(measured_indices)
        self.S_u = len(unmeasured_indices)
        self.d_model = d_model
        self.te_dim = te_dim
        self.num_patches = num_patches

        self.register_buffer('measured_indices', measured_indices)
        self.register_buffer('unmeasured_indices', unmeasured_indices)
        self.register_buffer('spatial_coords', spatial_coords)
        self.register_buffer('measured_coords', spatial_coords[measured_indices])
        self.register_buffer('unmeasured_coords',
                             spatial_coords[unmeasured_indices] if len(unmeasured_indices) > 0
                             else torch.zeros(0, 2))

        self.te = _LearnableTE(te_dim)

        self.tapa = _AttentionPatchAggregation(
            N=self.S_m, P=num_patches, S=None, te_dim=te_dim,
            hid_dim=d_model, history=1.0, dropout_rate=dropout,
        )

        pe = torch.zeros(num_patches, d_model)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('patch_pe', pe.unsqueeze(0))

        self.var_queries = nn.Parameter(
            torch.randn(1, self.S_m, 1, d_model))
        self.aggregation_norm = nn.LayerNorm(d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model + te_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

        self.register_buffer('data_min', torch.zeros(self.S_m))
        self.register_buffer('data_max', torch.ones(self.S_m))
        self.register_buffer('global_min', torch.tensor(0.0))
        self.register_buffer('global_max', torch.tensor(1.0))

        gp_w = build_gp_weights(spatial_coords[measured_indices],
                                spatial_coords[unmeasured_indices]
                                if len(unmeasured_indices) > 0
                                else torch.zeros(0, spatial_coords.shape[1]))
        self.register_buffer('_gp_weights', gp_w)

    def _encode(self, x, x_times_norm, x_mask):
        B, L, N = x.shape
        X_stacked = x.permute(0, 2, 1).reshape(B * N, L, 1)
        mask_stacked = (~x_mask).float().permute(0, 2, 1).reshape(B * N, L, 1)
        t_stacked = (x_times_norm.unsqueeze(1).expand(B, N, L)
                     .reshape(B * N, L, 1))

        te_his = self.te(t_stacked)
        x_with_te = torch.cat([X_stacked, te_his], dim=-1)

        h_patches = self.tapa(t_stacked, x_with_te, mask_stacked)
        h_patches = h_patches + self.patch_pe
        h_patches = h_patches.view(B, N, self.num_patches, self.d_model)

        attn_scores = torch.matmul(
            self.var_queries, h_patches.transpose(-1, -2)
        ) * (self.d_model ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches).squeeze(-2)
        h_final = self.aggregation_norm(h_final)
        return h_final

    def _decode(self, h_final, pred_times_norm):
        B, N, _ = h_final.shape
        H = pred_times_norm.shape[1]

        h_expanded = h_final.unsqueeze(2).expand(B, N, H, self.d_model)
        t_pred = pred_times_norm.unsqueeze(-1)
        te_pred = self.te(t_pred)
        te_pred = te_pred.unsqueeze(1).expand(B, N, H, self.te_dim)

        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)
        outputs = self.decoder(decoder_input).squeeze(-1)
        return outputs.permute(0, 2, 1)

    def fit(self, train_data, num_epochs: int = 200, lr: float = 1e-3,
            ctx_len: int = None, forecast_h: int = None):
        device = train_data.observations.device
        self.to(device)

        obs_m = train_data.observations[:, self.measured_indices]
        T = len(train_data.timestamps)

        if train_data.missing_mask is not None:
            mask_m = train_data.missing_mask[:, self.measured_indices].to(device)
        else:
            mask_m = torch.zeros_like(obs_m, dtype=torch.bool)

        for s in range(self.S_m):
            valid_vals = obs_m[:, s][~mask_m[:, s]]
            if len(valid_vals) > 0:
                self.data_min[s] = valid_vals.min()
                self.data_max[s] = valid_vals.max()
            else:
                self.data_min[s], self.data_max[s] = 0.0, 1.0

        self.global_min.fill_(self.data_min.min().item())
        self.global_max.fill_(self.data_max.max().item())

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (obs_m - self.data_min) / data_range
        times = train_data.timestamps.float()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs)

        self.train()
        if forecast_h is None:
            forecast_h = max(1, min(5, T // 10))
        if ctx_len is None:
            ctx_len = min(50, max(10, T // 3))
        ctx_len = min(ctx_len, T - forecast_h)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_windows = 0

            n_wins = min(8, max(1, T // forecast_h))
            max_start = max(0, T - ctx_len - forecast_h)
            starts = torch.randint(0, max_start + 1, (n_wins,))

            for start_val in starts:
                start = start_val.item()
                ctx_end = start + ctx_len
                H = min(forecast_h, T - ctx_end)
                if H < 1:
                    continue

                win_times = times[start:ctx_end + H]
                t_start, t_end = win_times[0], win_times[-1]
                ctx_t_local = ((times[start:ctx_end] - t_start)
                               / (t_end - t_start + 1e-8))
                tgt_t_local = ((times[ctx_end:ctx_end + H] - t_start)
                               / (t_end - t_start + 1e-8))

                ctx_obs = obs_norm[start:ctx_end].unsqueeze(0)
                ctx_mask = mask_m[start:ctx_end].unsqueeze(0)
                ctx_times_b = ctx_t_local.unsqueeze(0)
                tgt_obs = obs_norm[ctx_end:ctx_end + H]
                tgt_mask = mask_m[ctx_end:ctx_end + H]
                tgt_times_b = tgt_t_local.unsqueeze(0)

                h_final = self._encode(ctx_obs, ctx_times_b, ctx_mask)
                pred_norm = self._decode(h_final, tgt_times_b).squeeze(0)

                valid = ~tgt_mask
                if valid.any():
                    loss = F.mse_loss(pred_norm[valid], tgt_obs[valid])
                else:
                    loss = F.mse_loss(pred_norm, tgt_obs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_windows += 1

            scheduler.step()

            if (epoch + 1) % 50 == 0 and n_windows > 0:
                print(f"  [APN] Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {epoch_loss / n_windows:.6f}")

        return self

    def forecast(self, context_obs, context_mask, context_times,
                 forecast_times, num_samples: int = 50):
        device = context_obs.device
        H = len(forecast_times)

        data_range = self.data_max - self.data_min + 1e-6
        obs_norm = (context_obs - self.data_min) / data_range

        all_times = torch.cat([context_times, forecast_times]).float()
        t_min, t_max = all_times.min(), all_times.max()
        ctx_t_norm = ((context_times.float() - t_min)
                      / (t_max - t_min + 1e-8)).unsqueeze(0)
        tgt_t_norm = ((forecast_times.float() - t_min)
                      / (t_max - t_min + 1e-8)).unsqueeze(0)

        ctx_obs = obs_norm.unsqueeze(0)
        ctx_mask = context_mask.unsqueeze(0)

        self.train()
        samples_m = []

        with torch.no_grad():
            for _ in range(num_samples):
                h_final = self._encode(ctx_obs, ctx_t_norm, ctx_mask)
                pred_norm = self._decode(h_final, tgt_t_norm).squeeze(0)
                pred_raw = pred_norm * data_range + self.data_min
                samples_m.append(pred_raw)

        self.eval()
        samples_m = torch.stack(samples_m)

        result = {'measured': samples_m}
        if self.S_u > 0:
            samples_u = torch.stack([
                gp_spatial_interpolate(samples_m[i], self._gp_weights)
                for i in range(num_samples)
            ])
            result['unmeasured'] = samples_u
        return result