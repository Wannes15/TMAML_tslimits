"""
Meta-test analysis utilities for k-shot forecasting results.

Provides functions for loading results, computing bootstrap CIs, and plotting
step-wise performance metrics.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

METRICS = ['WQL']


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                  confidence: float = 0.95, seed: int = 42):
    """Return (mean, ci_lower, ci_upper) via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.array([np.mean(rng.choice(values, size=n, replace=True))
                      for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    return np.mean(values), np.percentile(means, alpha / 2 * 100), np.percentile(means, (1 - alpha / 2) * 100)


def load_results_for_model(
    results_dir: Path,
    file_pattern: str,
    k_values: Optional[List[int]] = None
) -> Dict[int, dict]:
    """
    Load k-shot result pickles matching a file pattern.
    
    Scans `results_dir` for files matching `file_pattern` (will automatically 
    append '*.pkl' if no wildcard or extension is present at the end).
    Extracts the K value from the dict stored inside.
    
    Args:
        results_dir: Directory containing .pkl result files.
        file_pattern: Glob pattern or prefix, e.g. 'EL_base_k*' or '*_old_metasets'.
        k_values: If given, only load these K values.
    
    Returns:
        Dict mapping K -> loaded result dict.
    """
    results_by_k = {}
    
    # If the user provides a direct glob pattern with '*' or it already ends with .pkl
    if '*' in file_pattern or file_pattern.endswith('.pkl'):
        pattern = file_pattern
        if not pattern.endswith('.pkl'):
            pattern += '*.pkl'
    else:
        pattern = f'{file_pattern}*.pkl'

    for pkl_path in sorted(results_dir.glob(pattern)):
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)
        k = res['K']
        if k_values is not None and k not in k_values:
            continue
        results_by_k[k] = res
    print(f"Loaded {len(results_by_k)} K-values for pattern '{pattern}': "
          f"K={sorted(results_by_k.keys())}")
    return results_by_k


def summarize_model(
    results_by_k: Dict[int, dict],
    model_name: str,
    metrics: List[str] = METRICS,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    step_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Build a summary DataFrame for one model across all loaded K values.
    
    Columns: model, K, metric, mean, ci_lower, ci_upper, n_series
    
    Args:
        results_by_k: Dict mapping K -> result dict
        model_name: Name for the model in output
        metrics: List of metric names to extract
        n_bootstrap: Bootstrap samples for CI
        confidence: Confidence level (0-1)
        seed: Random seed for bootstrap
        step_idx: If specified, extract metrics from step_histories at this step index
                  instead of from final losses. For example, step_idx=2 gets metrics after 
                  3rd adaptation step (0-indexed). If None, uses final losses (default).
    
    Returns:
        Summary DataFrame with columns: model, K, metric, mean, ci_lower, ci_upper, n_series
    """
    rows = []
    for k in sorted(results_by_k.keys()):
        res = results_by_k[k]
        
        for metric in metrics:
            # Choose data source: step_histories or final losses
            if step_idx is not None and 'step_histories' in res:
                # Extract from step_histories at specified step
                step_histories = res['step_histories']
                vals = []
                for s_id, hist in step_histories.items():
                    if step_idx < len(hist):
                        query_metrics = hist[step_idx].get('query_metrics')
                        if query_metrics is not None:
                            val = query_metrics.get(metric)
                            if val is not None and not np.isnan(val):
                                vals.append(val)
                vals = np.array(vals)
            else:
                # Extract from final losses (original behavior)
                losses = res['losses']
                vals = np.array([
                    losses[s][metric] for s in losses
                    if not np.isnan(losses[s].get(metric, float('nan')))
                ])
            
            if len(vals) == 0:
                rows.append({
                    'model': model_name, 'K': k, 'metric': metric,
                    'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                    'n_series': 0
                })
                continue
            
            mean, ci_lo, ci_hi = bootstrap_ci(vals, n_bootstrap, confidence, seed)
            rows.append({
                'model': model_name, 'K': k, 'metric': metric,
                'mean': mean, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'n_series': len(vals)
            })
    return pd.DataFrame(rows)


def display_results_table(summary_df: pd.DataFrame, model_name: Optional[str] = None, metrics: List[str] = METRICS):
    """
    Pivot the long-form summary into a readable table: rows = K, columns = metrics.
    Each cell shows  mean [ci_lower, ci_upper].
    If model_name is given, filter to that model; otherwise show all models.
    """
    df = summary_df.copy()
    if model_name is not None:
        df = df[df['model'] == model_name]

    # Build formatted string column
    def _fmt(row):
        if np.isnan(row['mean']):
            return 'N/A'
        return f"{row['mean']:.4f}  [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"

    df['display'] = df.apply(_fmt, axis=1)

    for model, mdf in df.groupby('model'):
        pivot = mdf.pivot(index='K', columns='metric', values='display')
        # Reorder columns to metrics order (where available)
        cols = [m for m in metrics if m in pivot.columns]
        pivot = pivot[cols]
        print(f"\n=== {model} ===")
        print(pivot.to_string())


def plot_wql_per_step(results_dir: Path, file_pattern: str, model_name: str, target_k: int, ylim: Optional[tuple] = None):
    """
    Reads in results for a specific run pattern, filters for a target K, 
    and plots WQL per inner loop step.
    """
    results_by_k = load_results_for_model(results_dir, file_pattern)
    
    if target_k not in results_by_k:
        print(f"K={target_k} not found in results. Available K: {list(results_by_k.keys())}")
        return
    
    res = results_by_k[target_k]
    if 'step_histories' not in res:
        print("No 'step_histories' found in the result dictionary.")
        return
        
    step_histories = res['step_histories']
    
    # Process step_histories to group WQL per step
    max_steps = max([len(hist) for hist in step_histories.values()])
    
    step_means = []
    step_lower = []
    step_upper = []
    steps = list(range(max_steps))
    
    for step_idx in steps:
        step_wqls = []
        for s_id, hist in step_histories.items():
            if step_idx < len(hist):
                step_entry = hist[step_idx]
                # Some result pickles store metrics flat on the step dict,
                # others nest them under 'query_metrics'.
                query_metrics = step_entry.get('query_metrics')
                val = query_metrics.get('WQL') if query_metrics is not None else step_entry.get('WQL')
                if val is not None and not np.isnan(val):
                    step_wqls.append(val)

        if len(step_wqls) > 0:
            mean, ci_lo, ci_hi = bootstrap_ci(np.array(step_wqls))
        else:
            mean, ci_lo, ci_hi = np.nan, np.nan, np.nan

        step_means.append(mean)
        step_lower.append(ci_lo)
        step_upper.append(ci_hi)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, step_means, marker='o', label=f'{model_name} (K={target_k})')
    ax.fill_between(steps, step_lower, step_upper, alpha=0.2)

    ax.set_xlabel('Inner Loop Step')
    ax.set_ylabel('WQL')
    ax.set_title(f'WQL per Inner Loop Step: {model_name} (K={target_k})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0.15, top=0.5)
    plt.show()


def plot_support_vs_query_per_step(results_dir: Path, file_pattern: str, model_name: str, target_k: int, ylim: Optional[tuple] = None):
    """
    Reads in results for a specific run pattern, filters for a target K,
    and plots both support loss (train mode) and query WQL (eval mode) per inner loop step.
    This directly shows the support → query gap evolution during adaptation.
    """
    results_by_k = load_results_for_model(results_dir, file_pattern)
    
    if target_k not in results_by_k:
        print(f"K={target_k} not found in results. Available K: {list(results_by_k.keys())}")
        return
    
    res = results_by_k[target_k]
    if 'step_histories' not in res:
        print("No 'step_histories' found in the result dictionary.")
        return
        
    step_histories = res['step_histories']
    
    # Process step_histories to group both metrics per step
    max_steps = max([len(hist) for hist in step_histories.values()])
    
    support_means = []
    support_lower = []
    support_upper = []
    query_means = []
    query_lower = []
    query_upper = []
    steps = list(range(max_steps))
    
    for step_idx in steps:
        # Support losses
        step_support_losses = []
        for s_id, hist in step_histories.items():
            if step_idx < len(hist):
                val = hist[step_idx].get('support_loss')
                if val is not None and not np.isnan(val):
                    step_support_losses.append(val)
        
        if len(step_support_losses) > 0:
            mean, ci_lo, ci_hi = bootstrap_ci(np.array(step_support_losses))
        else:
            mean, ci_lo, ci_hi = np.nan, np.nan, np.nan
        support_means.append(mean)
        support_lower.append(ci_lo)
        support_upper.append(ci_hi)
        
        # Query WQL
        step_wqls = []
        for s_id, hist in step_histories.items():
            if step_idx < len(hist):
                query_metrics = hist[step_idx].get('query_metrics')
                if query_metrics is not None:
                    val = query_metrics.get('WQL')
                    if val is not None and not np.isnan(val):
                        step_wqls.append(val)
        
        if len(step_wqls) > 0:
            mean, ci_lo, ci_hi = bootstrap_ci(np.array(step_wqls))
        else:
            mean, ci_lo, ci_hi = np.nan, np.nan, np.nan
        query_means.append(mean)
        query_lower.append(ci_lo)
        query_upper.append(ci_hi)
        
    # Plot both on same axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, support_means, marker='o', label=f'Support Loss (Train Mode)', linewidth=2, color='#FF6B6B')
    ax.fill_between(steps, support_lower, support_upper, alpha=0.2, color='#FF6B6B')
    
    ax.plot(steps, query_means, marker='s', label=f'Query WQL (Eval Mode)', linewidth=2, color='#4ECDC4')
    ax.fill_between(steps, query_lower, query_upper, alpha=0.2, color='#4ECDC4')
    
    ax.set_xlabel('Inner Loop Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Support vs Query Losses per Inner Loop Step: {model_name} (K={target_k})', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.show()


def plot_support_loss_per_step(results_dir: Path, file_pattern: str, model_name: str, target_k: int, ylim: Optional[tuple] = None):
    """
    Reads in results for a specific run pattern, filters for a target K, 
    and plots support loss per inner loop step (in train mode during adaptation).
    """
    results_by_k = load_results_for_model(results_dir, file_pattern)
    
    if target_k not in results_by_k:
        print(f"K={target_k} not found in results. Available K: {list(results_by_k.keys())}")
        return
    
    res = results_by_k[target_k]
    if 'step_histories' not in res:
        print("No 'step_histories' found in the result dictionary.")
        return
        
    step_histories = res['step_histories']
    
    # Process step_histories to group support loss per step
    max_steps = max([len(hist) for hist in step_histories.values()])
    
    step_means = []
    step_lower = []
    step_upper = []
    steps = list(range(max_steps))
    
    for step_idx in steps:
        step_support_losses = []
        for s_id, hist in step_histories.items():
            if step_idx < len(hist):
                val = hist[step_idx].get('support_loss')
                if val is not None and not np.isnan(val):
                    step_support_losses.append(val)
        
        if len(step_support_losses) > 0:
            mean, ci_lo, ci_hi = bootstrap_ci(np.array(step_support_losses))
        else:
            mean, ci_lo, ci_hi = np.nan, np.nan, np.nan
            
        step_means.append(mean)
        step_lower.append(ci_lo)
        step_upper.append(ci_hi)
        
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, step_means, marker='o', label=f'{model_name} (K={target_k})')
    ax.fill_between(steps, step_lower, step_upper, alpha=0.2)
    
    ax.set_xlabel('Inner Loop Step')
    ax.set_ylabel('Support Loss (Train Mode)')
    ax.set_title(f'Support Loss per Inner Loop Step: {model_name} (K={target_k})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.show()
