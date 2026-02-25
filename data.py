import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

from .config import SpatialCoordinate, SpatiotemporalData, ExperimentConfig


# ============================================================================
# PHYSICAL SIMULATORS
# ============================================================================

class NonlocalIDESimulator:
    
    def __init__(self,
                 nx: int = 64,
                 ny: int = 64,
                 domain_size: float = 1.0,
                 alpha: float = 1.0,
                 kappa: float = 0.01,
                 kernel_rank: int = 4,
                 kernel_eigenvalues: Optional[np.ndarray] = None,
                 dt: float = 0.001,
                 seed: int = 42):
        self.nx = nx
        self.ny = ny
        self.N = nx * ny  # total grid points
        self.domain_size = domain_size
        self.alpha = alpha
        self.kappa = kappa
        self.kernel_rank = kernel_rank
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        # Spatial grid
        self.dx = domain_size / nx
        self.dy = domain_size / ny
        self.x = np.linspace(0, domain_size, nx, endpoint=False)
        self.y = np.linspace(0, domain_size, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Normalized coordinates for kernel construction [0, 1]^2
        self.X_norm = self.X / domain_size
        self.Y_norm = self.Y / domain_size
        
        # Eigenvalues: default geometric decay
        if kernel_eigenvalues is not None:
            assert len(kernel_eigenvalues) == kernel_rank
            self.lambdas = kernel_eigenvalues.copy()
        else:
            self.lambdas = np.array([1.0 / (r + 1) for r in range(kernel_rank)])
        
        # Build kernel modes and precompute kernel matrix
        self.psi_modes = self._build_kernel_modes()   # [R, ny, nx]
        self.G_star = self._build_kernel_matrix()      # [N, N] ground truth
    
    # ------------------------------------------------------------------
    # Kernel construction
    # ------------------------------------------------------------------
    
    def _build_kernel_modes(self) -> np.ndarray:
        
        R = self.kernel_rank
        psi = np.zeros((R, self.ny, self.nx))
        
        Xn, Yn = self.X_norm, self.Y_norm  # [0, 1]^2
        
        for r in range(R):
            if r == 0:
                # Mode 1: North-South dipole teleconnection
                # Positive in north, negative in south → creates remote coupling
                psi[r] = np.sin(np.pi * Yn)  # zero at y=0,1, max at y=0.5
                # Modulate: positive top half, negative bottom half
                psi[r] = np.sin(2 * np.pi * Yn)
            
            elif r == 1:
                # Mode 2: East-West dipole teleconnection
                psi[r] = np.sin(2 * np.pi * Xn)
            
            elif r == 2:
                # Mode 3: Quadrupole — four-corner interaction
                # Positive at (NW, SE), negative at (NE, SW)
                psi[r] = np.sin(2 * np.pi * Xn) * np.sin(2 * np.pi * Yn)
            
            else:
                # Higher modes: multi-scale wave coupling
                # Alternating sin/cos with increasing frequency
                kx = (r // 2) + 1
                ky = ((r + 1) // 2) + 1
                if r % 2 == 1:
                    psi[r] = np.sin(2 * np.pi * kx * Xn) * np.cos(2 * np.pi * ky * Yn)
                else:
                    psi[r] = np.cos(2 * np.pi * kx * Xn) * np.sin(2 * np.pi * ky * Yn)
            
            # L2 normalize on discrete grid (∫ ψ² dx ≈ (1/N) Σ ψ²)
            norm = np.sqrt(np.mean(psi[r] ** 2))
            if norm > 1e-10:
                psi[r] /= norm
        
        return psi
    
    def _build_kernel_matrix(self) -> np.ndarray:

        N = self.N
        R = self.kernel_rank
        
        # Flatten modes: [R, N]
        Psi = self.psi_modes.reshape(R, N)  # [R, N]
        
        # G* = Ψ^T diag(λ) Ψ = Σ_r λ_r ψ_r ψ_r^T
        G = np.zeros((N, N))
        for r in range(R):
            G += self.lambdas[r] * np.outer(Psi[r], Psi[r])
        
        return G
    
    def get_kernel_at_sensors(self, sensor_coords: np.ndarray) -> np.ndarray:
        
        S = sensor_coords.shape[0]
        R = self.kernel_rank
        
        # Evaluate modes at sensor locations via bilinear interpolation
        psi_sensors = np.zeros((R, S))
        for r in range(R):
            interp = RectBivariateSpline(self.y, self.x, self.psi_modes[r])
            sx = sensor_coords[:, 0] * self.domain_size
            sy = sensor_coords[:, 1] * self.domain_size
            psi_sensors[r] = interp(sy, sx, grid=False)
        
        # G*_sensors = Σ_r λ_r ψ_r(s_i) ψ_r(s_j)
        G_sensors = np.zeros((S, S))
        for r in range(R):
            G_sensors += self.lambdas[r] * np.outer(psi_sensors[r], psi_sensors[r])
        
        return G_sensors
    
    def get_kernel_metadata(self) -> Dict:

        return {
            'eigenvalues': self.lambdas.copy(),
            'modes_grid': self.psi_modes.copy(),
            'kernel_matrix': self.G_star.copy(),
            'kernel_rank': self.kernel_rank,
            'alpha': self.alpha,
        }
    
    # ------------------------------------------------------------------
    # PDE integration
    # ------------------------------------------------------------------
    
    def _laplacian(self, u: np.ndarray) -> np.ndarray:
       
        d2u_dx2 = (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2 * u) / (self.dx ** 2)
        d2u_dy2 = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2 * u) / (self.dy ** 2)
        return d2u_dx2 + d2u_dy2
    
    def _nonlocal_term(self, u: np.ndarray) -> np.ndarray:
        
        u_flat = u.ravel()
        # Quadrature weight = dx * dy (area element)
        integral = self.G_star @ u_flat * (self.dx * self.dy)
        return self.alpha * integral.reshape(self.ny, self.nx)
    
    def _source_term(self, t: float) -> np.ndarray:
        sigma = 0.08 * self.domain_size  # localized source width
        
        # Source 1: center-left, oscillating
        cx1 = 0.25 * self.domain_size
        cy1 = 0.5 * self.domain_size
        amp1 = 2.0 * (1 + 0.5 * np.sin(2 * np.pi * t / 5.0))
        
        # Source 2: center-right, different frequency
        cx2 = 0.75 * self.domain_size
        cy2 = 0.5 * self.domain_size
        amp2 = 1.5 * (1 + 0.5 * np.cos(2 * np.pi * t / 3.0))
        
        f = (amp1 * np.exp(-((self.X - cx1)**2 + (self.Y - cy1)**2) / (2 * sigma**2))
           + amp2 * np.exp(-((self.X - cx2)**2 + (self.Y - cy2)**2) / (2 * sigma**2)))
        
        return f
    
    def _rhs(self, u: np.ndarray, t: float) -> np.ndarray:

        return self._nonlocal_term(u) + self.kappa * self._laplacian(u) + self._source_term(t)
    
    def simulate(self, T_final: float = 20.0,
                 save_every: float = 0.1,
                 spinup: float = 0.0,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        total_time = spinup + T_final
        n_steps = int(total_time / self.dt)
        save_interval = max(1, int(save_every / self.dt))
        
        # Initial condition: smooth random field
        u = self.rng.normal(size=(self.ny, self.nx)) * 0.1
        u = gaussian_filter(u, sigma=3)
        
        snapshots, times = [], []
        
        iterator = range(n_steps + 1)
        if verbose:
            iterator = tqdm(iterator, desc=f"Nonlocal IDE (α={self.alpha})")
        
        for step in iterator:
            t = step * self.dt
            
            # RK4 integration
            k1 = self._rhs(u, t)
            k2 = self._rhs(u + 0.5 * self.dt * k1, t + 0.5 * self.dt)
            k3 = self._rhs(u + 0.5 * self.dt * k2, t + 0.5 * self.dt)
            k4 = self._rhs(u + self.dt * k3, t + self.dt)
            u = u + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            u = np.clip(u, -50, 50)
            
            if t >= spinup and (step % save_interval == 0):
                snapshots.append(u.copy())
                times.append(t - spinup)
        
        return np.array(snapshots), np.array(times)
    
    def sample_at_sensors(self, snapshots: np.ndarray,
                          sensor_coords: np.ndarray,
                          noise_std: float = 0.1,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        T = snapshots.shape[0]
        S = sensor_coords.shape[0]
        
        sx = sensor_coords[:, 0] * self.domain_size
        sy = sensor_coords[:, 1] * self.domain_size
        
        ground_truth = np.zeros((T, S), dtype=np.float32)
        
        for t in range(T):
            interp = RectBivariateSpline(self.y, self.x, snapshots[t])
            ground_truth[t] = interp(sy, sx, grid=False).astype(np.float32)
        
        noise = rng.normal(size=ground_truth.shape).astype(np.float32) * noise_std
        observations = ground_truth + noise
        
        return observations, ground_truth


class AdvectionDiffusionSimulator:
    
    def __init__(self,
                 nx: int = 64,
                 ny: int = 64,
                 domain_size: float = 10.0,
                 diffusion_coeff: float = 0.1,
                 velocity: Tuple[float, float] = (0.5, 0.3),
                 dt: float = 0.01,
                 seed: int = 42):
        self.nx = nx
        self.ny = ny
        self.domain_size = domain_size
        self.D = diffusion_coeff
        self.velocity = velocity
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        self.dx = domain_size / nx
        self.dy = domain_size / ny
        self.x = np.linspace(0, domain_size, nx, endpoint=False)
        self.y = np.linspace(0, domain_size, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def _laplacian(self, u: np.ndarray) -> np.ndarray:
        d2u_dx2 = (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2 * u) / (self.dx ** 2)
        d2u_dy2 = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2 * u) / (self.dy ** 2)
        return d2u_dx2 + d2u_dy2
    
    def _advection(self, u: np.ndarray) -> np.ndarray:
        vx, vy = self.velocity
        
        if vx > 0:
            dudx = (u - np.roll(u, 1, axis=1)) / self.dx
        else:
            dudx = (np.roll(u, -1, axis=1) - u) / self.dx
        
        if vy > 0:
            dudy = (u - np.roll(u, 1, axis=0)) / self.dy
        else:
            dudy = (np.roll(u, -1, axis=0) - u) / self.dy
        
        return vx * dudx + vy * dudy
    
    def _source_term(self, t: float) -> np.ndarray:
        source = np.zeros((self.ny, self.nx))
        
        cx1, cy1 = 0.3 * self.domain_size, 0.3 * self.domain_size
        cx2, cy2 = 0.7 * self.domain_size, 0.6 * self.domain_size
        sigma = 0.5
        
        amp1 = 2.0 * (1 + 0.5 * np.sin(2 * np.pi * t / 20))
        amp2 = 1.5 * (1 + 0.5 * np.cos(2 * np.pi * t / 15))
        
        source += amp1 * np.exp(-((self.X - cx1)**2 + (self.Y - cy1)**2) / (2 * sigma**2))
        source += amp2 * np.exp(-((self.X - cx2)**2 + (self.Y - cy2)**2) / (2 * sigma**2))
        
        return source
    
    def simulate(self, T_final: float = 150.0, 
                 save_every: float = 1.0,
                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(T_final / self.dt)
        save_interval = max(1, int(save_every / self.dt))
        
        u = self.rng.normal(size=(self.ny, self.nx)) * 0.1
        u = gaussian_filter(u, sigma=3)
        
        snapshots, times = [], []
        
        iterator = range(n_steps + 1)
        if verbose:
            iterator = tqdm(iterator, desc="Advection-Diffusion")
        
        for step in iterator:
            t = step * self.dt
            
            k1 = -self._advection(u) + self.D * self._laplacian(u) + self._source_term(t)
            u_temp = u + 0.5 * self.dt * k1
            k2 = -self._advection(u_temp) + self.D * self._laplacian(u_temp) + self._source_term(t + 0.5 * self.dt)
            u = u + self.dt * k2
            u = np.clip(u, -10, 10)
            
            if step % save_interval == 0:
                snapshots.append(u.copy())
                times.append(t)
        
        return np.array(snapshots), np.array(times)
    
    def sample_at_sensors(self, snapshots: np.ndarray,
                          sensor_coords: np.ndarray,
                          noise_std: float = 0.1,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        T = snapshots.shape[0]
        S = sensor_coords.shape[0]
        
        sx = sensor_coords[:, 0] * self.domain_size
        sy = sensor_coords[:, 1] * self.domain_size
        
        ground_truth = np.zeros((T, S), dtype=np.float32)
        
        for t in range(T):
            interp = RectBivariateSpline(self.y, self.x, snapshots[t])
            ground_truth[t] = interp(sy, sx, grid=False).astype(np.float32)
        
        noise = rng.normal(size=ground_truth.shape).astype(np.float32) * noise_std
        observations = ground_truth + noise
        
        return observations, ground_truth



# ============================================================================
# DATA GENERATION UTILITIES
# ============================================================================

def generate_sensor_locations(n_sensors: int,
                              pattern: str = "mixed",
                              seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    
    if pattern == "random":
        coords = rng.random((n_sensors, 2))
    
    elif pattern == "mixed":
        n_center = n_sensors // 2
        n_outer = n_sensors - n_center
        
        center = rng.normal(size=(n_center, 2)) * 0.15 + 0.5
        center = np.clip(center, 0.05, 0.95)
        
        outer = rng.random((n_outer, 2)) * 0.9 + 0.05
        
        coords = np.vstack([center, outer])
    
    elif pattern == "grid":
        n_side = int(np.ceil(np.sqrt(n_sensors)))
        xs = np.linspace(0.1, 0.9, n_side)
        ys = np.linspace(0.1, 0.9, n_side)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.column_stack([xx.ravel(), yy.ravel()])[:n_sensors]
    
    elif pattern == "clustered":
        n_clusters = 4
        sensors_per_cluster = n_sensors // n_clusters
        remainder = n_sensors % n_clusters
        
        cluster_centers = rng.random((n_clusters, 2)) * 0.6 + 0.2
        coords_list = []
        
        for i, center in enumerate(cluster_centers):
            n = sensors_per_cluster + (1 if i < remainder else 0)
            cluster_coords = rng.normal(size=(n, 2)) * 0.1 + center
            cluster_coords = np.clip(cluster_coords, 0.05, 0.95)
            coords_list.append(cluster_coords)
        
        coords = np.vstack(coords_list)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return coords.astype(np.float32)


def normalize_times(times: np.ndarray) -> np.ndarray:
    """Normalize times to [0, 1]."""
    times = np.asarray(times, dtype=np.float32)
    t_min, t_max = times[0], times[-1]
    if t_max - t_min < 1e-8:
        return np.zeros_like(times)
    return (times - t_min) / (t_max - t_min)


def create_missing_mask(T: int, S: int,
                        missing_rate: float = 0.0,
                        block_missing: bool = False,
                        seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((T, S), dtype=bool)
    
    if missing_rate <= 0:
        return mask
    
    if block_missing:
        # Generate block outages, then calibrate to match target rate.
        # Strategy: generate excess blocks, then subsample to hit target.
        avg_duration = 8
        target_missing = int(T * S * missing_rate)
        
        # Over-generate by 2x, then trim
        n_outages_max = max(1, int(2.0 * target_missing / avg_duration))
        
        # Collect candidate outage blocks
        blocks = []
        for _ in range(n_outages_max):
            s = rng.integers(0, S)
            start = rng.integers(0, max(1, T - 3))
            duration = rng.integers(3, min(15, T - start) + 1)
            blocks.append((s, start, duration))
        
        # Apply blocks one-by-one until we reach target
        rng.shuffle(blocks)
        current_missing = 0
        
        for s, start, duration in blocks:
            # Count how many NEW missing entries this block adds
            new_entries = np.sum(~mask[start:start+duration, s])
            mask[start:start+duration, s] = True
            current_missing += new_entries
            
            if current_missing >= target_missing:
                break
        
        # Fine-tune: if still under target, add random point missing
        actual_missing = mask.sum()
        if actual_missing < target_missing:
            n_remaining = target_missing - actual_missing
            available = np.argwhere(~mask)
            if len(available) > 0:
                idx = rng.choice(len(available), size=min(n_remaining, len(available)), replace=False)
                for i in idx:
                    mask[available[i][0], available[i][1]] = True
        
        # Fine-tune: if over target, remove random missing entries
        actual_missing = mask.sum()
        if actual_missing > target_missing * 1.05:  # 5% tolerance
            excess = actual_missing - target_missing
            missing_entries = np.argwhere(mask)
            idx = rng.choice(len(missing_entries), size=min(excess, len(missing_entries)), replace=False)
            for i in idx:
                mask[missing_entries[i][0], missing_entries[i][1]] = False
    else:
        mask = rng.random((T, S)) < missing_rate
    
    return mask


def split_measured_unmeasured(n_sensors: int,
                              holdout_ratio: float = 0.0,
                              seed: int = 42,
                              mode: str = "random",
                              coords: Optional[np.ndarray] = None,
                              region: Tuple[float, float, float, float] = (0.7, 1.0, 0.7, 1.0)
                              ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_sensors)
    
    if holdout_ratio <= 0:
        return idx, np.array([], dtype=int)
    
    n_holdout = max(1, int(np.floor(n_sensors * holdout_ratio)))
    
    if mode == "random":
        perm = rng.permutation(n_sensors)
        unmeasured = np.sort(perm[:n_holdout])
        measured = np.sort(perm[n_holdout:])
    
    elif mode == "region":
        if coords is None:
            raise ValueError("coords required for region mode")
        
        x0, x1, y0, y1 = region
        in_region = np.where(
            (coords[:, 0] >= x0) & (coords[:, 0] <= x1) &
            (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
        )[0]
        
        if len(in_region) < n_holdout // 2:
            return split_measured_unmeasured(n_sensors, holdout_ratio, seed, "random")
        
        if len(in_region) > n_holdout:
            unmeasured = np.sort(rng.choice(in_region, size=n_holdout, replace=False))
        else:
            unmeasured = np.sort(in_region)
        
        measured = np.setdiff1d(idx, unmeasured)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return measured, unmeasured


def apply_time_dropout(observations: np.ndarray,
                       ground_truth: np.ndarray,
                       times: np.ndarray,
                       dropout_rate: float,
                       seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    T = len(times)
    
    keep = rng.random(T) > dropout_rate
    keep[0] = True
    keep[-1] = True
    
    idx = np.where(keep)[0]
    return observations[idx], ground_truth[idx], times[idx], idx


def apply_sensorwise_irregular(observations: np.ndarray,
                               ground_truth: np.ndarray,
                               times: np.ndarray,
                               keep_rate: float,
                               seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T, S = observations.shape
    keep_rate = float(np.clip(keep_rate, 0.1, 1.0))
    
    keep_sets = []
    for s in range(S):
        keep = rng.random(T) < keep_rate
        keep[0] = True
        keep[-1] = True
        keep_sets.append(np.where(keep)[0])
    
    union_idx = np.unique(np.concatenate(keep_sets))
    
    times_union = times[union_idx]
    obs_union = observations[union_idx].copy()
    gt_union = ground_truth[union_idx].copy()
    
    missing = np.zeros((len(union_idx), S), dtype=bool)
    
    idx_to_pos = {idx: i for i, idx in enumerate(union_idx)}
    
    for s in range(S):
        kept = set(keep_sets[s].tolist())
        for j, idx in enumerate(union_idx):
            if idx not in kept:
                missing[j, s] = True
    
    return obs_union, gt_union, times_union, missing


def enforce_spatial_holdout_mask(mask: np.ndarray,
                                 unmeasured_idx: np.ndarray) -> np.ndarray:

    if len(unmeasured_idx) > 0:
        mask[:, unmeasured_idx] = True
    return mask