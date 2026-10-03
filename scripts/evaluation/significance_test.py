"""
Significance Testing for K-Shot Results

Bootstrap-based hypothesis testing comparing TMAML vs baseline,
following the methodology from the TFT paper (Appendix C).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def bootstrap_losses(
    losses: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate bootstrap distribution of mean losses.
    
    Args:
        losses: Array of per-series losses (one value per series)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed
    
    Returns:
        Array of bootstrap mean losses (shape: [n_bootstrap])
    """
    rng = np.random.RandomState(seed)
    n = len(losses)
    bootstrap_means = np.empty(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = rng.choice(losses, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
    
    return bootstrap_means


def significance_test(
    tmaml_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    metrics: List[str] = ['WQL'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Bootstrap significance test comparing TMAML vs baseline.

    Performs two tests per metric:
    1. Paper method: Bootstrap CI on TMAML, check if baseline point estimate falls outside
    2. Paired bootstrap: CI on (TMAML - baseline) difference per series

    Args:
        tmaml_results: Output from k_shot_test() for TMAML model
        baseline_results: Output from k_shot_test() for baseline model
        metrics: List of metrics to compare
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for intervals (default: 0.95)
        seed: Random seed
        verbose: Print results

    Returns:
        Dictionary with test results per metric
    """
    tmaml_losses = tmaml_results['losses']
    baseline_losses = baseline_results['losses']
    
    # Find common series
    common_series = sorted(set(tmaml_losses.keys()) & set(baseline_losses.keys()))
    if len(common_series) == 0:
        raise ValueError("No common series between TMAML and baseline results")
    
    alpha = 1 - confidence_level
    results = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIGNIFICANCE TEST: TMAML vs Baseline ({len(common_series)} series)")
        print(f"Bootstrap: {n_bootstrap} resamples, {confidence_level*100:.0f}% CI")
        print(f"{'='*70}")
    
    for metric in metrics:
        # Filter out series with NaN values for this metric
        valid_series = [
            s for s in common_series
            if not np.isnan(tmaml_losses[s].get(metric, float('nan')))
            and not np.isnan(baseline_losses[s].get(metric, float('nan')))
        ]
        
        if len(valid_series) == 0:
            if verbose:
                print(f"\n  {metric}: SKIPPED (no valid series)")
            results[metric] = {'skipped': True, 'reason': 'no valid series'}
            continue
        
        if verbose and len(valid_series) < len(common_series):
            print(f"\n  {metric}: Using {len(valid_series)}/{len(common_series)} series (rest have NaN)")
        
        tmaml_vals = np.array([tmaml_losses[s][metric] for s in valid_series])
        baseline_vals = np.array([baseline_losses[s][metric] for s in valid_series])
        
        tmaml_mean = np.mean(tmaml_vals)
        baseline_mean = np.mean(baseline_vals)
        
        # --- Paper method: Bootstrap CI on TMAML ---
        tmaml_bootstrap = bootstrap_losses(tmaml_vals, n_bootstrap, seed)
        ci_lower = np.percentile(tmaml_bootstrap, alpha / 2 * 100)
        ci_upper = np.percentile(tmaml_bootstrap, (1 - alpha / 2) * 100)
        
        # One-tailed: TMAML < baseline?
        # Significant if baseline falls above TMAML's upper CI bound
        paper_significant = baseline_mean > ci_upper
        
        # --- Paired bootstrap: CI on (TMAML - baseline) ---
        diffs = tmaml_vals - baseline_vals
        diff_bootstrap = bootstrap_losses(diffs, n_bootstrap, seed)
        diff_ci_lower = np.percentile(diff_bootstrap, alpha / 2 * 100)
        diff_ci_upper = np.percentile(diff_bootstrap, (1 - alpha / 2) * 100)
        diff_mean = np.mean(diffs)
        
        # Significant improvement if entire CI is below 0
        paired_significant = diff_ci_upper < 0
        
        # p-value: proportion of bootstrap diffs >= 0
        p_value = np.mean(diff_bootstrap >= 0)
        
        results[metric] = {
            'tmaml_mean': tmaml_mean,
            'baseline_mean': baseline_mean,
            'improvement': baseline_mean - tmaml_mean,
            'improvement_pct': (baseline_mean - tmaml_mean) / baseline_mean * 100,
            'tmaml_ci': (ci_lower, ci_upper),
            'paper_significant': paper_significant,
            'diff_mean': diff_mean,
            'diff_ci': (diff_ci_lower, diff_ci_upper),
            'paired_significant': paired_significant,
            'p_value': p_value
        }
        
        if verbose:
            sig_paper = "YES" if paper_significant else "NO"
            sig_paired = "YES" if paired_significant else "NO"
            
            print(f"\n  {metric}:")
            print(f"    TMAML:    {tmaml_mean:.4f}  CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"    Baseline: {baseline_mean:.4f}")
            print(f"    Improvement: {results[metric]['improvement']:.4f} ({results[metric]['improvement_pct']:.1f}%)")
            print(f"    Paper method significant: {sig_paper}")
            print(f"    Paired diff: {diff_mean:.4f}  CI: [{diff_ci_lower:.4f}, {diff_ci_upper:.4f}]")
            print(f"    Paired significant: {sig_paired}  (p={p_value:.4f})")
    
    if verbose:
        print(f"\n{'='*70}\n")

    return results