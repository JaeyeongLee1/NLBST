import numpy as np
import torch
from typing import Tuple, List, Any

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from .config import SpatiotemporalData, ExperimentConfig
from .model import BayesianSTNeuralODE_Kalman, VariationalKalmanGuide


def train_nlbst(model: BayesianSTNeuralODE_Kalman,
                   train_data: SpatiotemporalData,
                   config: ExperimentConfig,
                   verbose: bool = True) -> Tuple[Any, List[float], VariationalKalmanGuide]:

    device = next(model.parameters()).device
    
    # Move data to device
    train_data.observations = train_data.observations.to(device)
    train_data.timestamps = train_data.timestamps.to(device)
    if train_data.missing_mask is not None:
        train_data.missing_mask = train_data.missing_mask.to(device)
    train_data.measured_indices = train_data.measured_indices.to(device)
    train_data.unmeasured_indices = train_data.unmeasured_indices.to(device)
    if train_data.spatial_coords is not None:
        train_data.spatial_coords = train_data.spatial_coords.to(device)
    
    model.compute_normalization_params(train_data.observations, train_data.missing_mask)
    
    pyro.clear_param_store()
    
    guide = VariationalKalmanGuide(model, hidden_dim=config.hidden_dim).to(device)
    

    
    base_lr = config.learning_rate
    lr_A_NL = base_lr  
    
    
    def per_param_optim_args(module_name, param_name):
        if 'A_NL' in param_name:
            return {"lr": lr_A_NL, "clip_norm": 10.0}
        return {"lr": base_lr, "clip_norm": 10.0}
    
    optimizer = ClippedAdam(per_param_optim_args)
    svi = SVI(model.model, guide, optimizer, loss=Trace_ELBO())
    
    losses = []
    best_loss = float('inf')
    best_params = None
    best_torch = None
    patience_counter = 0
    
    log_interval = max(1, config.num_epochs // 10)
    
    for epoch in range(config.num_epochs):
        try:
            loss = svi.step(train_data, prediction_mode=False)
            losses.append(loss)
            
            if verbose and (epoch % log_interval == 0 or epoch == config.num_epochs - 1):
                avg_loss = np.mean(losses[-log_interval:]) if len(losses) >= log_interval else np.mean(losses)
                
                a_nl_norm = torch.norm(model.A_NL).item()
                print(f"  Epoch {epoch+1:4d}/{config.num_epochs}: "
                      f"loss={loss:.2f}, avg={avg_loss:.2f}, best={best_loss:.2f}, "
                      f"||A_NL||={a_nl_norm:.4f}")
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                best_params = pyro.get_param_store().get_state()
                best_torch = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "guide": {k: v.detach().cpu().clone() for k, v in guide.state_dict().items()},
                }
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (patience={config.patience})")
                if best_params:
                    pyro.get_param_store().set_state(best_params)
                if best_torch:
                    model.load_state_dict({k: v.to(device) for k, v in best_torch["model"].items()}, strict=False)
                    guide.load_state_dict({k: v.to(device) for k, v in best_torch["guide"].items()}, strict=False)
                break
            
            if np.isnan(loss) or np.isinf(loss):
                if verbose:
                    print(f"  NaN/Inf detected at epoch {epoch+1}, restoring best state")
                if best_params:
                    pyro.get_param_store().set_state(best_params)
                if best_torch:
                    model.load_state_dict({k: v.to(device) for k, v in best_torch["model"].items()}, strict=False)
                    guide.load_state_dict({k: v.to(device) for k, v in best_torch["guide"].items()}, strict=False)
                break
                
        except Exception as e:
            if verbose:
                print(f"  Training error at epoch {epoch+1}: {e}")
            if best_params:
                pyro.get_param_store().set_state(best_params)
            if best_torch:
                model.load_state_dict({k: v.to(device) for k, v in best_torch["model"].items()}, strict=False)
                guide.load_state_dict({k: v.to(device) for k, v in best_torch["guide"].items()}, strict=False)
            break
    
    if best_params:
        pyro.get_param_store().set_state(best_params)
    if best_torch:
        model.load_state_dict({k: v.to(device) for k, v in best_torch["model"].items()}, strict=False)
        guide.load_state_dict({k: v.to(device) for k, v in best_torch["guide"].items()}, strict=False)
    
    if verbose:
        final_a_nl_norm = torch.norm(model.A_NL).item()
        print(f"  Training complete. Final ||A_NL||={final_a_nl_norm:.4f}")
    
    return svi, losses, guide