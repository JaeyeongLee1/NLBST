import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Protocol, runtime_checkable

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False



@runtime_checkable
class ODEFuncProtocol(Protocol):
    def integrate_step(self, z: torch.Tensor, 
                       t_start: float, t_end: float) -> torch.Tensor: ...





class SpatialBasisFunctions(nn.Module):
    
    def __init__(self,
                 num_basis: int,
                 spatial_coords: torch.Tensor,
                 basis_type: str = 'rbf',
                 length_scale: float = 1.0,
                 learn_basis: bool = True):
        super().__init__()
        
        self.num_basis = num_basis
        self.basis_type = basis_type.lower()
        self.coord_dim = spatial_coords.shape[1]
        
        self.register_buffer('reference_coords', spatial_coords)
        
        if self.basis_type == 'rbf':
            self._init_rbf_basis(spatial_coords, length_scale, learn_basis)
        elif self.basis_type == 'fourier':
            self._init_fourier_basis(length_scale)
        elif self.basis_type == 'wendland':
            self._init_wendland_basis(spatial_coords, length_scale, learn_basis)
        elif self.basis_type == 'chebyshev':
            self._init_chebyshev_basis()
        elif self.basis_type == 'multiscale_rbf':
            self._init_multiscale_rbf_basis(spatial_coords, length_scale, learn_basis)
        else:
            raise ValueError(
                f"Unknown basis type: {basis_type}. "
                f"Choose from: rbf, fourier, wendland, chebyshev, multiscale_rbf"
            )
    

    
    def _init_rbf_basis(self, spatial_coords, length_scale, learn_basis):
        num_locations = spatial_coords.shape[0]
    
        if learn_basis:
            # 기존 그대로
            if self.num_basis <= num_locations:
                indices = torch.linspace(0, num_locations - 1, self.num_basis).long()
                centers = spatial_coords[indices].clone()
            else:
                coord_min = spatial_coords.min(dim=0)[0]
                coord_max = spatial_coords.max(dim=0)[0]
                centers = torch.rand(self.num_basis, self.coord_dim,
                                 device=spatial_coords.device) * (coord_max - coord_min) + coord_min
            self.centers = nn.Parameter(centers)
            self.log_length_scale = nn.Parameter(torch.tensor(np.log(length_scale)))
        else:

            coord_min = spatial_coords.min(dim=0)[0]
            coord_max = spatial_coords.max(dim=0)[0]
        
            if self.coord_dim == 2:
                n_per_dim = int(np.ceil(np.sqrt(self.num_basis)))
                g1 = torch.linspace(coord_min[0].item(), coord_max[0].item(), n_per_dim)
                g2 = torch.linspace(coord_min[1].item(), coord_max[1].item(), n_per_dim)
                grid = torch.stack(torch.meshgrid(g1, g2, indexing='ij'), dim=-1)
                centers = grid.reshape(-1, 2)[:self.num_basis]
            elif self.coord_dim == 1:
                centers = torch.linspace(coord_min[0].item(), coord_max[0].item(),
                                     self.num_basis).unsqueeze(1)
            else:
                centers = torch.rand(self.num_basis, self.coord_dim) * (coord_max - coord_min) + coord_min
        
            self.register_buffer('centers', centers)
            self.register_buffer('log_length_scale', torch.tensor(np.log(length_scale)))
    

    
    def _init_fourier_basis(self, length_scale: float):
        omega = torch.randn(self.num_basis, self.coord_dim) / length_scale
        bias = torch.rand(self.num_basis) * 2 * np.pi
        self.register_buffer('omega', omega)
        self.register_buffer('bias', bias)
    
    
    def _init_wendland_basis(self, spatial_coords: torch.Tensor,
                             length_scale: float, learn_basis: bool):
        
        num_locations = spatial_coords.shape[0]
        
        if self.num_basis <= num_locations:
            indices = torch.linspace(0, num_locations - 1, self.num_basis).long()
            centers = spatial_coords[indices].clone()
        else:
            coord_min = spatial_coords.min(dim=0)[0]
            coord_max = spatial_coords.max(dim=0)[0]
            centers = torch.rand(self.num_basis, self.coord_dim,
                                 device=spatial_coords.device) * (coord_max - coord_min) + coord_min
        
        if learn_basis:
            self.centers = nn.Parameter(centers)
            self.log_radius = nn.Parameter(torch.tensor(np.log(length_scale)))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('log_radius', torch.tensor(np.log(length_scale)))
    

    def _init_chebyshev_basis(self):

        if self.coord_dim == 1:
            self.n_per_dim = self.num_basis
            self.register_buffer('degrees', torch.arange(self.num_basis))
        elif self.coord_dim == 2:
            n_per_dim = int(np.ceil(np.sqrt(self.num_basis)))
            self.n_per_dim = n_per_dim
            di, dj = torch.meshgrid(
                torch.arange(n_per_dim), torch.arange(n_per_dim), indexing='ij'
            )
            # Sort by total degree (i+j) for truncation
            pairs = torch.stack([di.flatten(), dj.flatten()], dim=-1) 
            total_deg = pairs.sum(dim=-1)
            sorted_idx = torch.argsort(total_deg)
            pairs = pairs[sorted_idx[:self.num_basis]]  
            self.register_buffer('degree_pairs', pairs)  
        else:
            raise NotImplementedError(
                f"Chebyshev basis for d={self.coord_dim} not implemented (d=1,2 supported)"
            )
    

    
    def _init_multiscale_rbf_basis(self, spatial_coords: torch.Tensor,
                                    length_scale: float, learn_basis: bool):
        num_locations = spatial_coords.shape[0]
        

        n_fine = self.num_basis // 3
        n_medium = self.num_basis // 3
        n_coarse = self.num_basis - n_fine - n_medium
        counts = [n_fine, n_medium, n_coarse]
        scale_multipliers = [0.3, 1.0, 3.0]
        
        all_centers = []
        all_log_ls = []
        
        for n_k, scale_mult in zip(counts, scale_multipliers):
            if n_k == 0:
                continue

            if n_k <= num_locations:
                idx = torch.linspace(0, num_locations - 1, n_k).long()
                centers_k = spatial_coords[idx].clone()
            else:
                coord_min = spatial_coords.min(dim=0)[0]
                coord_max = spatial_coords.max(dim=0)[0]
                centers_k = torch.rand(n_k, self.coord_dim,
                                       device=spatial_coords.device) * (coord_max - coord_min) + coord_min
            
            ls_k = length_scale * scale_mult
            log_ls_k = torch.full((n_k,), np.log(ls_k))
            
            all_centers.append(centers_k)
            all_log_ls.append(log_ls_k)
        
        centers = torch.cat(all_centers, dim=0)      
        log_ls = torch.cat(all_log_ls, dim=0)        
        
        if learn_basis:
            self.centers = nn.Parameter(centers)
            self.log_length_scales = nn.Parameter(log_ls) 
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('log_length_scales', log_ls)
    

    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.basis_type == 'rbf':
            return self._compute_rbf(coords)
        elif self.basis_type == 'fourier':
            return self._compute_fourier(coords)
        elif self.basis_type == 'wendland':
            return self._compute_wendland(coords)
        elif self.basis_type == 'chebyshev':
            return self._compute_chebyshev(coords)
        elif self.basis_type == 'multiscale_rbf':
            return self._compute_multiscale_rbf(coords)
    
    
    def _compute_rbf(self, coords: torch.Tensor) -> torch.Tensor:
        length_scale = torch.exp(self.log_length_scale)
        diff = coords.unsqueeze(1) - self.centers.unsqueeze(0)  # [S, K, d]
        dist_sq = (diff ** 2).sum(dim=-1)                       # [S, K]
        return torch.exp(-dist_sq / (2 * length_scale ** 2))
    
    def _compute_fourier(self, coords: torch.Tensor) -> torch.Tensor:
        proj = coords @ self.omega.T + self.bias                # [S, K]
        return np.sqrt(2.0 / self.num_basis) * torch.cos(proj)
    
    def _compute_wendland(self, coords: torch.Tensor) -> torch.Tensor:
        """Wendland C2: φ(r) = (1-r)^4_+ · (4r+1), r = ||x-c|| / radius."""
        radius = torch.exp(self.log_radius)
        diff = coords.unsqueeze(1) - self.centers.unsqueeze(0)  # [S, K, d]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)       # [S, K]
        r = dist / radius                                        # normalized distance
        
        r_clamped = r.clamp(max=1.0)
        phi = ((1.0 - r_clamped) ** 4) * (4.0 * r_clamped + 1.0)  # [S, K]
        
        # Hard zero outside support
        phi = phi * (r < 1.0).float()
        return phi
    
    def _compute_chebyshev(self, coords: torch.Tensor) -> torch.Tensor:
        """Chebyshev tensor product: T_i(2x-1) * T_j(2y-1) on [0,1]^d."""
        x_mapped = 2.0 * coords - 1.0  # [S, d]
        x_mapped = x_mapped.clamp(-1.0, 1.0)
        
        if self.coord_dim == 1:
            # T_k(x) via recurrence: T_0=1, T_1=x, T_{k+1}=2x·T_k - T_{k-1}
            S = coords.shape[0]
            max_deg = self.degrees.max().item()
            
            T = torch.zeros(S, max_deg + 1, device=coords.device, dtype=coords.dtype)
            T[:, 0] = 1.0
            if max_deg >= 1:
                T[:, 1] = x_mapped[:, 0]
            for k in range(2, max_deg + 1):
                T[:, k] = 2.0 * x_mapped[:, 0] * T[:, k-1] - T[:, k-2]
            
            return T[:, self.degrees]  # [S, K]
        
        else:  # coord_dim == 2
            S = coords.shape[0]
            max_deg = self.degree_pairs.max().item()
            
            # Compute 1D Chebyshev values for each dimension
            Tx = torch.zeros(S, max_deg + 1, device=coords.device, dtype=coords.dtype)
            Ty = torch.zeros(S, max_deg + 1, device=coords.device, dtype=coords.dtype)
            Tx[:, 0] = 1.0
            Ty[:, 0] = 1.0
            if max_deg >= 1:
                Tx[:, 1] = x_mapped[:, 0]
                Ty[:, 1] = x_mapped[:, 1]
            for k in range(2, max_deg + 1):
                Tx[:, k] = 2.0 * x_mapped[:, 0] * Tx[:, k-1] - Tx[:, k-2]
                Ty[:, k] = 2.0 * x_mapped[:, 1] * Ty[:, k-1] - Ty[:, k-2]
            
            # Tensor product: φ_{i,j}(x,y) = T_i(x) · T_j(y)
            deg_i = self.degree_pairs[:, 0]  # [K]
            deg_j = self.degree_pairs[:, 1]  # [K]
            
            Phi = Tx[:, deg_i] * Ty[:, deg_j]  # [S, K]
            return Phi
    
    def _compute_multiscale_rbf(self, coords: torch.Tensor) -> torch.Tensor:
        """Multi-scale RBF: per-basis learnable length scale."""
        length_scales = torch.exp(self.log_length_scales)        # [K]
        diff = coords.unsqueeze(1) - self.centers.unsqueeze(0)   # [S, K, d]
        dist_sq = (diff ** 2).sum(dim=-1)                        # [S, K]
        return torch.exp(-dist_sq / (2 * length_scales.unsqueeze(0) ** 2))




class TemporalODEFunc(nn.Module):

    def __init__(self, num_basis: int, hidden_dim: int = 64):
        super().__init__()
        self.num_basis = num_basis

        
        self.net = nn.Sequential(
            nn.Linear(num_basis + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_basis),
            nn.Tanh()
        )
        
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        t_emb = self.time_embed(t.view(1, 1)).squeeze(0)  
        
        if z.dim() == 1:
            z_t = torch.cat([z, t], dim=-1) 
        else:
            t_emb_expanded = t.unsqueeze(0).expand(z.shape[0], -1)
            z_t = torch.cat([z, t_emb_expanded], dim=-1)  
        
        return self.net(z_t)


class SimpleODEFunc(nn.Module):

    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()
        )
        
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) * self.scale
    
    def integrate_step(self, z: torch.Tensor, t_start: float, t_end: float) -> torch.Tensor:
        if abs(t_end - t_start) < 1e-6:
            return z
        
        device = z.device
        
        if TORCHDIFFEQ_AVAILABLE:
            t_span = torch.tensor([t_start, t_end], device=device, dtype=z.dtype)
            z_traj = odeint(self, z, t_span, method='rk4')
            return z_traj[-1]
        else:
            n_steps = 5
            dt = (t_end - t_start) / n_steps
            z_current = z
            t_current = t_start
            for _ in range(n_steps):
                dz = self(torch.tensor(t_current, device=device), z_current)
                z_current = z_current + dz * dt
                z_current = torch.clamp(z_current, -10, 10)
                t_current += dt
            return z_current



class BasisKalmanFilter(nn.Module):
    
    def __init__(self, num_basis: int):
        super().__init__()
        self.K = num_basis
    
    def predict(self,
                m: torch.Tensor,
                P_diag: torch.Tensor,
                sigma_process: torch.Tensor,
                dt: torch.Tensor,
                ode_func, 
                t_start: float,
                t_end: float) -> Tuple[torch.Tensor, torch.Tensor]:

        m_prior = ode_func.integrate_step(m, t_start, t_end)
        Q_diag = (sigma_process ** 2) * dt
        P_prior_diag = P_diag + Q_diag
        return m_prior, P_prior_diag
    
    def update(self,
               m_prior: torch.Tensor,
               P_prior_diag: torch.Tensor,
               y: torch.Tensor,
               Phi: torch.Tensor,
               sigma_obs: torch.Tensor,
               obs_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = m_prior.device
        dtype = m_prior.dtype
        K = self.K
        
        if obs_mask is not None:
            valid_mask = ~obs_mask
            if valid_mask.sum() == 0:
                return m_prior, P_prior_diag
            y_valid = y[valid_mask]
            Phi_valid = Phi[valid_mask]
        else:
            y_valid = y
            Phi_valid = Phi
        
        S_valid = y_valid.shape[0]
        
        Phi_scaled = Phi_valid * P_prior_diag.unsqueeze(0) 
        
        R_val = sigma_obs ** 2
        S_cov = Phi_scaled @ Phi_valid.T + R_val * torch.eye(S_valid, device=device, dtype=dtype)
        
        P_Phi_T = Phi_valid.T * P_prior_diag.unsqueeze(1)
        
        try:
            K_gain = torch.linalg.solve(S_cov, P_Phi_T.T).T 
        except:
            S_cov_reg = S_cov + 1e-4 * torch.eye(S_valid, device=device, dtype=dtype)
            K_gain = torch.linalg.solve(S_cov_reg, P_Phi_T.T).T
        
        innovation = y_valid - Phi_valid @ m_prior
        m_post = m_prior + K_gain @ innovation
        
        IKPhi = torch.eye(K, device=device, dtype=dtype) - K_gain @ Phi_valid
        P_post_full = IKPhi @ torch.diag(P_prior_diag) @ IKPhi.T + R_val * (K_gain @ K_gain.T)
        P_post_diag = torch.diag(P_post_full).clamp(min=1e-6)
        
        return m_post, P_post_diag



class BasisInitialEncoder(nn.Module):
    
    def __init__(self, num_basis: int, initial_std: float = 0.5):
        super().__init__()
        self.K = num_basis
        self.initial_std = initial_std
    
    def forward(self, y0: torch.Tensor, Phi: torch.Tensor,
                sigma_obs: torch.Tensor,
                obs_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
        device = y0.device
        dtype = y0.dtype
        K = self.K
        
        if obs_mask is not None and obs_mask.any():
            valid_mask = ~obs_mask
            n_valid = valid_mask.sum().item()
            
            if n_valid == 0:
                z0_mean = torch.zeros(K, device=device, dtype=dtype)
                z0_std = torch.full((K,), self.initial_std, device=device, dtype=dtype)
                return z0_mean, z0_std
            
            y0_valid = y0[valid_mask]
            Phi_valid = Phi[valid_mask]
        else:
            y0_valid = y0
            Phi_valid = Phi
        
        prior_var = self.initial_std ** 2
        prior_precision = 1.0 / prior_var
        obs_precision = 1.0 / (sigma_obs ** 2 + 1e-6)

        Phi_T_Phi = Phi_valid.T @ Phi_valid
        post_precision = obs_precision * Phi_T_Phi + prior_precision * torch.eye(K, device=device, dtype=dtype)
        Phi_T_y = Phi_valid.T @ y0_valid
        
        try:
            z0_mean = torch.linalg.solve(post_precision, obs_precision * Phi_T_y)
        except:
            post_precision_reg = post_precision + 1e-4 * torch.eye(K, device=device, dtype=dtype)
            z0_mean = torch.linalg.solve(post_precision_reg, obs_precision * Phi_T_y)
        
        try:
            post_cov = torch.linalg.inv(post_precision)
        except:
            post_cov = torch.linalg.inv(post_precision + 1e-4 * torch.eye(K, device=device, dtype=dtype))
        
        z0_std = torch.diag(post_cov).clamp(min=1e-6).sqrt()
        return z0_mean, z0_std