import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from .config import ExperimentResults
from .metrics import ProbabilisticMetrics


MODEL_COLORS = {
    'NLBST': 'tab:blue',
    'NLBST-full': 'tab:blue',
    'NLBST-linear_only': 'tab:cyan',
    'NLBST-neural_only': 'tab:orange',
    'NLBST-diagonal': 'tab:purple',
    'Linear-DSTM': 'tab:orange',
    'GRU-D': 'tab:green',
    'Latent-ODE': 'tab:red',
}

MODEL_MARKERS = {
    'NLBST': 'o',
    'NLBST-full': 'o',
    'NLBST-linear_only': 'D',
    'NLBST-neural_only': '^',
    'NLBST-diagonal': 'v',
    'Linear-DSTM': 's',
    'GRU-D': 'p',
    'Latent-ODE': '*',
}

MODEL_DISPLAY_NAMES = {
    'NLBST': 'NLBST (Proposed)',
    'NLBST-full': r'$\mathbf{A}_{\mathrm{NL}} \mathbf{z} + g_\theta$ (Full)',
    'NLBST-linear_only': r'$\mathbf{A}_{\mathrm{NL}} \mathbf{z}$ only',
    'NLBST-neural_only': r'$g_\theta$ only',
    'NLBST-diagonal': r'$\mathrm{diag}(\mathbf{A}_{\mathrm{NL}}) \mathbf{z} + g_\theta$',
    'Linear-DSTM': 'Linear DSTM',
    'GRU-D': 'GRU-D',
    'Latent-ODE': 'Latent ODE',
}

MODEL_COLORS.update({
    'FNO': 'tab:brown',
    'GraFITi': 'tab:pink',
    'APN': 'tab:olive',
})
MODEL_MARKERS.update({
    'FNO': 'd',
    'GraFITi': 'P',
    'APN': 'X',
})
MODEL_DISPLAY_NAMES.update({
    'FNO': 'FNO',
    'GraFITi': 'GraFITi',
    'APN': 'APN',
})

def _get_color(name: str) -> str:
    return MODEL_COLORS.get(name, 'tab:pink')

def _get_marker(name: str) -> str:
    return MODEL_MARKERS.get(name, 'D')

def _get_display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


class ExperimentVisualizer:
    """Publication-quality visualization for experiment results."""
    
    def __init__(self, figsize_scale: float = 1.0):
        self.figsize_scale = figsize_scale
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    
    def plot_sweep_comparison(self, aggregated_results: Dict[str, Dict[float, Tuple[float, float]]],
                              sweep_param: str,
                              metric: str = 'RMSE',
                              title: str = None,
                              save_path: str = None):
        fig, ax = plt.subplots(figsize=(6 * self.figsize_scale, 4 * self.figsize_scale))
        
        for model_name, results in aggregated_results.items():
            x_vals = sorted(results.keys())
            means = [results[x][0] for x in x_vals]
            stds = [results[x][1] for x in x_vals]
            
            ax.errorbar(x_vals, means, yerr=stds,
                       label=_get_display_name(model_name),
                       color=_get_color(model_name),
                       marker=_get_marker(model_name),
                       capsize=3,
                       linewidth=2,
                       markersize=6)
        
        ax.set_xlabel(sweep_param.replace('_', ' ').title())
        ax.set_ylabel(metric)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    
    def plot_calibration_diagram(self, results: ExperimentResults,
                                 save_path: str = None,
                                 model_name: str = "NLBST"):
        if results.predictions is None:
            print(f"No predictions available for calibration plot ({model_name})")
            return None
        
        pred_means = results.predictions['pred_means_measured']
        pred_stds = results.predictions['pred_stds_measured']
        targets = results.predictions['targets_measured']
        
        ece, expected, actual = ProbabilisticMetrics.calibration_error(
            pred_means, pred_stds, targets, n_bins=10
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(10 * self.figsize_scale, 4 * self.figsize_scale))
        
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=2)
        ax.plot(expected, actual, 'bo-', label=f'{model_name} (ECE={ece:.3f})', linewidth=2, markersize=6)
        ax.fill_between(expected, expected - 0.1, expected + 0.1, alpha=0.2, color='gray')
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Calibration Diagram')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        ax = axes[1]
        pit_hist, bin_edges = ProbabilisticMetrics.pit_histogram(pred_means, pred_stds, targets)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, pit_hist, width=0.08, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axhline(0.1, color='red', linestyle='--', label='Uniform')
        ax.set_xlabel('PIT Value')
        ax.set_ylabel('Density')
        ax.set_title(f'PIT Histogram ({model_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_calibration_comparison(self, all_results: Dict[str, ExperimentResults],
                                    save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(12 * self.figsize_scale, 5 * self.figsize_scale))
        
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=2, zorder=0)
        
        for model_name, results in all_results.items():
            if results.predictions is None:
                continue
            if 'pred_means_measured' not in results.predictions:
                continue
                
            pred_means = results.predictions['pred_means_measured']
            pred_stds = results.predictions['pred_stds_measured']
            targets = results.predictions['targets_measured']
            
            ece, expected, actual = ProbabilisticMetrics.calibration_error(
                pred_means, pred_stds, targets, n_bins=10
            )
            
            ax.plot(expected, actual, 
                   marker=_get_marker(model_name), 
                   color=_get_color(model_name),
                   label=f'{_get_display_name(model_name)} (ECE={ece:.3f})', 
                   linewidth=2, markersize=6)
        
        ax.fill_between([0, 1], [0, 1], [0.1, 1.1], alpha=0.1, color='gray')
        ax.fill_between([0, 1], [-0.1, 0.9], [0, 1], alpha=0.1, color='gray')
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Calibration Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        ax = axes[1]
        for model_name, results in all_results.items():
            if np.isnan(results.sharpness) or np.isnan(results.calibration_error):
                continue
            
            ax.scatter(results.sharpness, results.calibration_error, 
                      color=_get_color(model_name), 
                      marker=_get_marker(model_name), 
                      s=150, label=_get_display_name(model_name), zorder=3)
        
        ax.set_xlabel('Sharpness (avg. predictive std)')
        ax.set_ylabel('Calibration Error (ECE)')
        ax.set_title('Sharpness vs Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    
    def plot_horizon_analysis(self, results: ExperimentResults,
                              save_path: str = None):
        if results.rmse_by_horizon is None:
            print("No horizon-wise metrics available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(10 * self.figsize_scale, 4 * self.figsize_scale))
        
        horizons = sorted(results.rmse_by_horizon.keys())
        
        ax = axes[0]
        rmse_vals = [results.rmse_by_horizon[h] for h in horizons]
        ax.plot(horizons, rmse_vals, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE by Horizon')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        if results.crps_by_horizon:
            crps_vals = [results.crps_by_horizon[h] for h in horizons]
            ax.plot(horizons, crps_vals, 's-', color='coral', linewidth=2, markersize=6)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('CRPS')
        ax.set_title('CRPS by Horizon')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    
    def plot_spatial_predictions(self, results: ExperimentResults,
                                 sensor_coords: np.ndarray,
                                 measured_indices: np.ndarray,
                                 unmeasured_indices: np.ndarray,
                                 time_idx: int = 0,
                                 save_path: str = None):
        if results.predictions is None:
            print("No predictions available")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(14 * self.figsize_scale, 4 * self.figsize_scale))
        
        pred_m = results.predictions['pred_means_measured'][time_idx]
        target_m = results.predictions['targets_measured'][time_idx]
        std_m = results.predictions['pred_stds_measured'][time_idx]
        
        coords_m = sensor_coords[measured_indices]
        
        ax = axes[0]
        scatter = ax.scatter(coords_m[:, 0], coords_m[:, 1], c=target_m,
                            cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
        if len(unmeasured_indices) > 0:
            coords_u = sensor_coords[unmeasured_indices]
            ax.scatter(coords_u[:, 0], coords_u[:, 1], c='gray', s=30, marker='x', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Value')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('True Values (measured)')
        
        ax = axes[1]
        scatter = ax.scatter(coords_m[:, 0], coords_m[:, 1], c=pred_m,
                            cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Predicted')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Predictions')
        
        ax = axes[2]
        errors = np.abs(pred_m - target_m)
        scatter = ax.scatter(coords_m[:, 0], coords_m[:, 1], c=errors,
                            cmap='Reds', s=60, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='|Error|')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Absolute Errors')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    

    def plot_ablation_comparison(self,
                                 ablation_results: Dict[str, List[Dict]],
                                 metrics: List[str] = None,
                                 save_path: str = None):
        """
        Grouped bar chart comparing ablation modes.
        
        Args:
            ablation_results: {mode: [run_results_dict, ...]}
            metrics: List of metric names to plot (default: RMSE, CRPS)
            save_path: Optional path to save figure
        """
        if metrics is None:
            metrics = ['rmse_measured', 'crps_measured']
        
        modes = ['full', 'linear_only', 'neural_only', 'diagonal']
        mode_labels = [
            r'$\mathbf{A}_{\mathrm{NL}}\mathbf{z} + g_\theta$' + '\n(Full)',
            r'$\mathbf{A}_{\mathrm{NL}}\mathbf{z}$' + '\n(Linear only)',
            r'$g_\theta$' + '\n(Neural only)',
            r'$\mathrm{diag}(\mathbf{A}_{\mathrm{NL}})\mathbf{z} + g_\theta$' + '\n(Diagonal)',
        ]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, 
                                 figsize=(5 * n_metrics * self.figsize_scale, 
                                          4 * self.figsize_scale))
        if n_metrics == 1:
            axes = [axes]
        
        bar_colors = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:purple']
        
        for ax, metric_name in zip(axes, metrics):
            means = []
            stds = []
            
            for mode in modes:
                if mode not in ablation_results:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                
                vals = []
                key = f'NLBST-{mode}' if mode != 'full' else 'NLBST'
                for result_dict in ablation_results[mode]:
                    if key in result_dict:
                        v = getattr(result_dict[key], metric_name, np.nan)
                        if not np.isnan(v):
                            vals.append(v)
                
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)
            
            x = np.arange(len(modes))
            bars = ax.bar(x, means, yerr=stds, 
                         color=bar_colors[:len(modes)],
                         capsize=4, edgecolor='black', linewidth=0.5,
                         alpha=0.85)
            
            # Highlight best
            valid_means = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
            if valid_means:
                best_idx = min(valid_means, key=lambda x: x[1])[0]
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(2.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(mode_labels, fontsize=8)
            ax.set_ylabel(metric_name.replace('_', ' ').upper())
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Ablation Study: Component Contribution', fontsize=13, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    
    
    def create_summary_table(self, all_results: Dict[str, Dict[str, ExperimentResults]]) -> str:
        """Generate LaTeX table for paper."""
        models = list(set(
            model for exp_results in all_results.values() 
            for model in exp_results 
            if isinstance(exp_results.get(model), ExperimentResults)
        ))
        models.sort()
        
        n_models = len(models)
        
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Experiment Results Summary}",
            "\\label{tab:results}",
            "\\begin{tabular}{l" + "cccc" * n_models + "}",
            "\\toprule",
        ]
        
        # Header
        header_parts = [""]
        for model in models:
            display = _get_display_name(model).replace('$', '').replace(r'\mathbf', '')
            header_parts.append(f"\\multicolumn{{4}}{{c}}{{{display}}}")
        lines.append(" & ".join(header_parts) + " \\\\")
        
        # Cmidrules
        cmidrule_parts = []
        for i, model in enumerate(models):
            start = 2 + i * 4
            end = start + 3
            cmidrule_parts.append(f"\\cmidrule(lr){{{start}-{end}}}")
        lines.append(" ".join(cmidrule_parts))
        
        # Subheader
        subheader = ["Experiment"]
        for _ in models:
            subheader.extend(["RMSE", "CRPS", "Cov90", "NLL"])
        lines.append(" & ".join(subheader) + " \\\\")
        lines.append("\\midrule")
        
        def fmt(val, is_coverage=False):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "-"
            if is_coverage:
                return f"{val:.0%}"
            return f"{val:.3f}"
        
        for exp_name, exp_results in all_results.items():
            row_parts = [exp_name.replace('_', '\\_')]
            
            for model in models:
                if model in exp_results and isinstance(exp_results[model], ExperimentResults):
                    r = exp_results[model]
                    row_parts.extend([
                        fmt(r.rmse_measured),
                        fmt(r.crps_measured),
                        fmt(r.coverage_90, is_coverage=True),
                        fmt(r.nll_measured)
                    ])
                else:
                    row_parts.extend(["-", "-", "-", "-"])
            
            lines.append(" & ".join(row_parts) + " \\\\")
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)
    
   