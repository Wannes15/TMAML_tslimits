"""K-shot adaptation testing: build support/query samples from a held-out
series, adapt on the support windows, and score the query horizon on WQL."""

from pathlib import Path

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
import pickle
from datetime import datetime

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import DistributionLoss
import wandb

from core.evaluation.model_loaders import get_model_loader
from core.models.metrics import compute_wql

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')


def create_k_shot_samples(
    series_df: pd.DataFrame,
    base_dataset: TimeSeriesDataSet,
    K: int,
    forecast_horizon: int,
    max_encoder_length: int,
    group_id: str = 'traj_id',
    unified_support: bool = False,
    support_window_size: Optional[int] = None,
    take_last_as_query: bool = True,
    anonymize_series_id: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Create K support datasets + 1 query dataset for a single series.
    Uses the LAST meta-window of the series (like validation mode in MetaWindowTSDataset).

    Args:
        unified_support: If True, create ONE support dataset spanning all K*S timesteps
                         instead of K separate datasets (one per window).
        support_window_size: Size of each support window (S). Defaults to forecast_horizon.
                             Can be smaller (e.g. 6 or 12) to test adaptation with less data.

    Returns None if series is too short.
    """
    S = support_window_size if support_window_size is not None else forecast_horizon
    time_idx = base_dataset.time_idx
    target = base_dataset.target

    # Keep support/query construction aligned with TMAML MetaWindowTSDataset.
    support_ds_kwargs = {}
    if S != forecast_horizon:
        support_ds_kwargs['max_prediction_length'] = min(K * S, forecast_horizon)
        support_ds_kwargs['min_prediction_length'] = base_dataset.min_prediction_length

    valid_data = series_df.dropna(subset=[target])
    if len(valid_data) == 0:
        return None

    min_time = valid_data[time_idx].min()
    max_time = valid_data[time_idx].max()

    meta_window_length = K * S + forecast_horizon
    if max_time - min_time + 1 < meta_window_length:
        return None

    if take_last_as_query:
        window_end = max_time
        window_start = window_end - meta_window_length + 1
    else:
        window_start = min_time
        window_end = window_start + meta_window_length - 1

    full_data = series_df[
        (series_df[time_idx] >= window_start) &
        (series_df[time_idx] <= window_end)
    ].copy()

    # Keep evaluation consistent with TMAML runs that anonymize traj_id.
    if anonymize_series_id:
        full_data[group_id] = '__dummy__'

    if len(full_data) < meta_window_length:
        return None

    support_datasets = []

    if unified_support and K > 0:
        support_start = window_start
        support_end = window_start + K * S - 1

        support_data = full_data[
            (full_data[time_idx] >= support_start) &
            (full_data[time_idx] <= support_end)
        ].copy()

        if len(support_data) == 0:
            return None

        try:
            support_dataset = TimeSeriesDataSet.from_dataset(
                base_dataset,
                support_data,
                predict=True,
                stop_randomization=True,
                **support_ds_kwargs
            )
            if len(support_dataset) == 0:
                return None
            support_datasets = [support_dataset]
        except Exception:
            return None
    else:
        for k in range(1, K + 1):
            support_start = window_start + (k - 1) * S
            support_end = support_start + S - 1

            if k == 1:
                encoder_start = support_start
            else:
                encoder_length = min((k - 1) * S, max_encoder_length)
                encoder_start = support_start - encoder_length

            support_data = full_data[
                (full_data[time_idx] >= encoder_start) &
                (full_data[time_idx] <= support_end)
            ].copy()

            if len(support_data) == 0:
                return None

            try:
                support_dataset = TimeSeriesDataSet.from_dataset(
                    base_dataset,
                    support_data,
                    predict=False,
                    stop_randomization=True,
                    **support_ds_kwargs
                )
                if len(support_dataset) == 0:
                    return None
                support_datasets.append(support_dataset)
            except Exception:
                return None

    query_start = window_start + K * S
    query_end = query_start + forecast_horizon - 1

    if K > 0:
        encoder_length = min(K * S, max_encoder_length)
        encoder_start = query_start - encoder_length
    else:
        encoder_start = query_start

    query_data = full_data[
        (full_data[time_idx] >= encoder_start) &
        (full_data[time_idx] <= query_end)
    ].copy()

    if len(query_data) == 0:
        return None

    try:
        query_dataset = TimeSeriesDataSet.from_dataset(
            base_dataset,
            query_data,
            predict=True,
            stop_randomization=True
        )
        if len(query_dataset) == 0:
            return None
    except Exception:
        return None

    return {
        'support_datasets': support_datasets,
        'query_dataset': query_dataset,
        'K': K,
    }


# Default quantile grid for materializing distribution-based models.
# Matches the 9-quantile output of the QuantileLoss models so WQL is comparable
# across model families.
_DEFAULT_QUANTILE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _predicted_quantiles(model: Any, raw: torch.Tensor,
                         quantiles: Optional[List[float]] = None):
    """Turn a model's raw output['prediction'] into a quantile tensor.

    QuantileLoss models (e.g. TFT) already emit [batch, horizon, n_quantiles].
    DistributionLoss models instead emit distribution parameters, mapped to
    quantiles via loss.to_quantiles.
    """
    loss = getattr(model, "loss", None)
    if isinstance(loss, DistributionLoss):
        levels = list(quantiles) if quantiles is not None else _DEFAULT_QUANTILE_GRID
        return loss.to_quantiles(raw, quantiles=levels), levels
    levels = list(quantiles) if quantiles is not None else getattr(loss, "quantiles", None)
    return raw, levels


def evaluate_query(
    model: Any,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    quantiles: Optional[List[float]] = None,
    return_predictions: bool = False
) -> Dict[str, float]:
    """Evaluate model on the query dataset: WQL."""
    model.eval()
    all_preds_quantiles, all_targets = [], []
    quantile_levels = None

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            for key in x:
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to(device)
            y_true = y[0].to(device)

            output = model(x)
            y_pred_quantiles, quantile_levels = _predicted_quantiles(
                model, output['prediction'], quantiles
            )

            all_preds_quantiles.append(y_pred_quantiles)
            all_targets.append(y_true)

    y_pred_quantiles = torch.cat(all_preds_quantiles, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    wql = compute_wql(y_pred_quantiles, y_true, quantiles=quantile_levels)

    result = {'WQL': wql}
    if return_predictions:
        result['pred_quantiles'] = y_pred_quantiles.cpu().numpy()
        result['quantile_levels'] = quantile_levels
        result['query_actuals'] = y_true.cpu().numpy()
    return result


def adapt_on_support(
    model: Any,
    support_dataset: TimeSeriesDataSet,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: str,
    query_loader: Optional[torch.utils.data.DataLoader] = None,
    quantiles: Optional[List[float]] = None,
    grad_clip: Optional[float] = 1.0,
    support_batch_size: int = 32
) -> List[Dict[str, Any]]:
    """Run adaptation steps on a support dataset, tracking support & query metrics."""
    step_metrics = []

    support_sampler = torch.utils.data.RandomSampler(
        support_dataset,
        replacement=True,
        num_samples=support_batch_size
    )
    dataloader = support_dataset.to_dataloader(
        train=True,
        batch_size=support_batch_size,
        num_workers=0,
        shuffle=False,
        sampler=support_sampler
    )

    for step_idx in range(num_steps):
        model.train()
        support_loss = None
        for batch in dataloader:
            x, y = batch
            for key in x:
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to(device)
            y = tuple(yi.to(device) if isinstance(yi, torch.Tensor) else yi for yi in y)

            optimizer.zero_grad()
            output = model(x)
            loss = model.loss(output['prediction'], y)
            support_loss = loss.item()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        query_metrics = None
        if query_loader is not None:
            query_metrics = evaluate_query(
                model=model,
                dataloader=query_loader,
                device=device,
                quantiles=quantiles
            )

        step_metrics.append({
            'support_loss': support_loss,
            'query_metrics': query_metrics
        })

    return step_metrics


def k_shot_test(
    test_df: pd.DataFrame,
    model_checkpoint_path: str,
    base_dataset: TimeSeriesDataSet,
    K: int = 2,
    architecture: str = 'tft',
    model_config: Optional[Dict[str, Any]] = None,
    forecast_horizon: Optional[int] = None,
    adaptation_steps: int = 5,
    adaptation_lr: float = 1e-4,
    group_id: str = 'traj_id',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    is_maml: Optional[bool] = None,
    quantiles: Optional[List[float]] = None,
    unified_support: bool = True,
    support_window_size: Optional[int] = None,
    anonymize_series_id: bool = False,
    inner_grad_clip: Optional[float] = 1.0,
    use_wandb: bool = False,
    verbose: bool = True,
    take_last_as_query: bool = True,
    save_predictions: bool = False
) -> Dict[str, Any]:
    """
    K-shot adaptation testing on held-out time series. For each series:
    K=0 evaluates the query with no adaptation, K>0 adapts on K support
    windows first.

    save_predictions: if True, additionally return a 'predictions' dict
    mapping series_id to the final predicted quantiles/levels/actuals.

    Returns a dict with 'losses' mapping series_id to its loss dict.
    """
    if forecast_horizon is None:
        forecast_horizon = base_dataset.max_prediction_length
    if model_config is None:
        model_config = {}

    max_encoder_length = base_dataset.max_encoder_length
    test_series_ids = test_df[group_id].unique()

    S = support_window_size if support_window_size is not None else forecast_horizon
    support_mode = "unified" if unified_support else "per-window"

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-SHOT TEST: K={K}, {len(test_series_ids)} series")
        print(f"Support window size: S={S} (forecast horizon F={forecast_horizon})")
        print(f"Support mode: {support_mode}")
        print(f"Adaptation: {adaptation_steps} steps/window @ LR={adaptation_lr}")
        print(f"{'='*60}\n")

    model_loader = get_model_loader(architecture, **model_config)

    def _finite_values(values: List[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    def _finite_mean(values: List[float]) -> float:
        finite = _finite_values(values)
        return float(np.mean(finite)) if finite.size > 0 else float('nan')

    all_losses = {}
    all_step_histories = {}
    all_predictions = {} if save_predictions else None

    for idx, series_id in enumerate(test_series_ids):
        if verbose:
            print(f"[{idx+1}/{len(test_series_ids)}] {series_id}", end=" ")

        series_df = test_df[test_df[group_id] == series_id].copy()
        series_df = series_df.sort_values(base_dataset.time_idx).reset_index(drop=True)

        samples = create_k_shot_samples(
            series_df=series_df,
            base_dataset=base_dataset,
            K=K,
            forecast_horizon=forecast_horizon,
            max_encoder_length=max_encoder_length,
            group_id=group_id,
            unified_support=unified_support,
            support_window_size=support_window_size,
            take_last_as_query=take_last_as_query,
            anonymize_series_id=anonymize_series_id
        )

        if samples is None:
            if verbose:
                print("skipped (series too short)")
            continue

        model = model_loader.load(
            checkpoint_path=model_checkpoint_path,
            base_dataset=base_dataset,
            device=device,
            is_maml=is_maml
        )
        model.to(device)

        query_loader = samples['query_dataset'].to_dataloader(
            train=False, batch_size=1, num_workers=0
        )

        initial_losses = evaluate_query(
            model, query_loader, device,
            quantiles=quantiles
        )
        step_history = [{'support_loss': None, 'query_metrics': initial_losses}]

        if K > 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=adaptation_lr)
            for k_idx, support_dataset in enumerate(samples['support_datasets']):
                window_step_metrics = adapt_on_support(
                    model=model,
                    support_dataset=support_dataset,
                    optimizer=optimizer,
                    num_steps=adaptation_steps,
                    device=device,
                    query_loader=query_loader,
                    quantiles=quantiles,
                    grad_clip=inner_grad_clip,
                    support_batch_size=64
                )
                step_history.extend(window_step_metrics)

        losses = step_history[-1]['query_metrics'] if step_history[-1]['query_metrics'] is not None else step_history[-1]

        all_losses[series_id] = losses
        all_step_histories[series_id] = step_history

        if save_predictions:
            pred_out = evaluate_query(
                model, query_loader, device,
                quantiles=quantiles,
                return_predictions=True
            )
            all_predictions[series_id] = {
                'pred_quantiles': pred_out['pred_quantiles'],
                'quantile_levels': pred_out['quantile_levels'],
                'query_actuals': pred_out['query_actuals'],
            }

        if verbose:
            support_loss_str = f"{step_history[-1]['support_loss']:.4f}" if step_history[-1]['support_loss'] is not None else "N/A"
            print(f"Support={support_loss_str}  WQL={losses['WQL']:.4f}")

        del model
        torch.cuda.empty_cache()

    if verbose and len(all_losses) > 0:
        wql_values = _finite_values([l['WQL'] for l in all_losses.values()])
        avg_wql = float(np.mean(wql_values)) if wql_values.size > 0 else float('nan')
        print(f"\n{'='*60}")
        print(f"RESULTS: {len(all_losses)} series tested, Avg WQL: {avg_wql:.4f}")
        print(f"{'='*60}\n")

    if use_wandb:
        avg_wql = _finite_mean([l['WQL'] for l in all_losses.values()])
        wandb.log({f'test/K{K}_WQL': avg_wql})

    result = {'losses': all_losses, 'K': K, 'num_series': len(all_losses), 'step_histories': all_step_histories}
    if save_predictions:
        result['predictions'] = all_predictions
    return result


def save_results(results: Dict[str, Any], save_dir: Path, run_name: str) -> Path:
    """Save test results to a pickle file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = save_dir / f'{run_name}_K{results["K"]}_{timestamp}.pkl'

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {results_path}")
    return results_path


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
