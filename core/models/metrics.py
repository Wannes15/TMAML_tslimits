"""
Custom metrics for time series forecasting.

Implements Weighted Quantile Loss (WQL) from Salinas et al. (2020) /
Gneiting & Raftery (2007), the accuracy metric used throughout evaluation.
"""

import numpy as np
import torch
from typing import List


def compute_wql(
    y_pred_quantiles: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: List[float] = None
) -> float:
    """
    Compute Weighted Quantile Loss (WQL).
    
    WQL_α = 2 * Σ_{i,t} QL_α(q^(α)_{i,t}, x_{i,t}) / Σ_{i,t} |x_{i,t}|
    WQL = (1/K) * Σ_j WQL_{α_j}
    
    At level α, the quantile loss is:
        QL_α(q, x) = α * (x - q)  if x > q
                    = (1-α) * (q - x)  otherwise
    
    Reference: Koenker & Hallock (2001), Gneiting & Raftery (2007)
    
    Args:
        y_pred_quantiles: Predicted quantiles [batch, time, num_quantiles]
        y_true: True values [batch, time]
        quantiles: List of quantile levels matching the last dim of y_pred_quantiles.
                   If None, inferred from output size (e.g. 3 → [0.1, 0.5, 0.9]).
    
    Returns:
        WQL value (float). Lower is better.
    """
    n_quantiles = y_pred_quantiles.shape[-1]
    
    if quantiles is None:
        if n_quantiles == 3:
            quantiles = [0.1, 0.5, 0.9]
        elif n_quantiles == 7:
            quantiles = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
        elif n_quantiles == 9:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            # Evenly spaced quantiles
            quantiles = [round((i + 1) / (n_quantiles + 1), 4) for i in range(n_quantiles)]
    
    assert len(quantiles) == n_quantiles, (
        f"Number of quantiles ({len(quantiles)}) must match last dim of predictions ({n_quantiles})"
    )
    
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)  # [1, time]

    # Denominator: Σ_{i,t} |x_{i,t}|
    abs_sum = torch.abs(y_true).sum()
    if abs_sum < 1e-8:
        return float('nan')
    
    # Compute WQL per quantile level, then average
    wql_per_quantile = []
    for q_idx, alpha in enumerate(quantiles):
        q_pred = y_pred_quantiles[:, :, q_idx]  # [batch, time]
        error = y_true - q_pred  # [batch, time]
        
        # QL_α(q, x) = α*(x-q) if x>q, else (1-α)*(q-x)
        ql = torch.where(error >= 0, alpha * error, (alpha - 1) * error)
        
        # WQL_α = 2 * Σ QL / Σ|x|
        wql_alpha = 2.0 * ql.sum() / abs_sum
        wql_per_quantile.append(wql_alpha.item())
    
    # WQL = (1/K) * Σ WQL_α
    wql = float(np.mean(wql_per_quantile))
    return wql