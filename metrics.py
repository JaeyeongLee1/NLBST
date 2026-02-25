import numpy as np
from scipy.stats import norm
from typing import Tuple


class ProbabilisticMetrics:
    
    @staticmethod
    def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-6)
        z = (y - mu) / sigma
        
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        return crps
    
    @staticmethod
    def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = samples.shape[0]
        
        # E|X - y|
        term1 = np.mean(np.abs(samples - y[np.newaxis, ...]), axis=0)
        
        # E|X - X'| via sorting
        xs = np.sort(samples, axis=0)
        
        w = (2 * np.arange(1, n + 1) - n - 1).astype(np.float64)
        w = w.reshape((n,) + (1,) * (xs.ndim - 1))
        
        term2 = (2.0 / (n ** 2)) * np.sum(w * xs, axis=0)
        
        return term1 - 0.5 * term2
    
    @staticmethod
    def nll_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-6)
        return 0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + ((y - mu) / sigma) ** 2)
    
    @staticmethod
    def interval_score(lower: np.ndarray, upper: np.ndarray, 
                       y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        width = upper - lower
        
        below = y < lower
        above = y > upper
        
        penalty_below = (2 / alpha) * (lower - y) * below
        penalty_above = (2 / alpha) * (y - upper) * above
        
        return width + penalty_below + penalty_above
    
    @staticmethod
    def compute_coverage(mu: np.ndarray, sigma: np.ndarray, 
                        y: np.ndarray, level: float = 0.9) -> float:
    
        z = norm.ppf((1 + level) / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        
        covered = (y >= lower) & (y <= upper)
        return np.mean(covered)
    
    @staticmethod
    def calibration_error(mu: np.ndarray, sigma: np.ndarray, 
                         y: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:

        prob_levels = np.linspace(0.1, 0.99, n_bins)
        expected_coverage = prob_levels
        actual_coverage = []
        
        for p in prob_levels:
            cov = ProbabilisticMetrics.compute_coverage(mu, sigma, y, level=p)
            actual_coverage.append(cov)
        
        actual_coverage = np.array(actual_coverage)
        ece = np.mean(np.abs(expected_coverage - actual_coverage))
        
        return ece, expected_coverage, actual_coverage
    
    @staticmethod
    def pit_histogram(mu: np.ndarray, sigma: np.ndarray, 
                     y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sigma = np.maximum(sigma, 1e-6)
        pit_values = norm.cdf((y - mu) / sigma)
        
        hist, bin_edges = np.histogram(pit_values.flatten(), bins=n_bins, range=(0, 1))
        hist = hist / hist.sum()
        
        return hist, bin_edges