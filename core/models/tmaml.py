## Standard libraries
import os
import numpy as np
import random
from copy import deepcopy

## tqdm for loading bars
from tqdm.auto import tqdm

## PyTorch
import torch
import torch.utils.data as data
import torch.optim as optim

# PyTorch Lightning
import lightning.pytorch as pl

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # allow non-deterministic ops for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore")


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch


class MetaWindowTSDataset(data.Dataset):
    """
    Meta-window dataset for time series forecasting.
    
    Creates tasks as meta-windows that slide across each series.
    Each meta-window = one meta-learning task with fixed K support windows + 1 query window.
    
    Meta-window structure:
    - Length: (K + 1) * F where K = num support windows, F = forecast horizon
    - First K * F timesteps: K support windows (each of length F)
    - Last F timesteps: 1 query window
    
    Support windows are created as full TSDataset inputs with proper encoder/decoder splits.
    """
    
    def __init__(self, train_df, base_dataset, forecast_horizon=24, K_support=3, 
                validation_mode=False, unified_support=False, anonymize_series_id=True,
                support_window_size=None):
        """
        Args:
            train_df: DataFrame with training data
            base_dataset: TimeSeriesDataSet object for accessing encoder/decoder configs
            forecast_horizon: Number of time steps per window (F) — used for query window
            K_support: Number of support windows (K >= 0)
            max_series: Limit number of series in dataset (None = use all)
            validation_mode: If True, only return LAST meta-window per series
            unified_support: If True, create ONE support dataset spanning all K*S timesteps
                            instead of K separate support datasets (one per window).
                            The query dataset remains unchanged.
            anonymize_series_id: If True, replace traj_id with a constant dummy value
                                 in all support/query DataFrames to prevent the TFT from
                                 memorizing series-specific embeddings. Set to False to
                                 keep original series identities (default behavior).
            support_window_size: Size of each support window (S). If None, defaults to
                                 forecast_horizon (F) for backward-compatible behavior.
                                 Set to e.g. 6 or 12 for smaller support windows.
                                 Meta-window length becomes K*S + F.
        """
        if K_support < 0:
            raise ValueError(f"K_support must be >= 0, got {K_support}")
        
        self.train_df = train_df
        self.base_dataset = base_dataset
        self.forecast_horizon = forecast_horizon
        self.K_support = K_support
        self.support_window_size = support_window_size if support_window_size is not None else forecast_horizon
        self.validation_mode = validation_mode
        self.unified_support = unified_support
        self.anonymize_series_id = anonymize_series_id
        
        # Extract important parameters from base dataset
        self.max_encoder_length = base_dataset.max_encoder_length
        self.max_prediction_length = base_dataset.max_prediction_length
        self.time_idx = base_dataset.time_idx
        self.target = base_dataset.target
        self.min_prediction_length = base_dataset.min_prediction_length
        
        # Meta-window length: K * S + F timesteps (S = support_window_size, F = forecast_horizon)
        self.meta_window_length = K_support * self.support_window_size + forecast_horizon
        
        # Get valid series
        all_series = train_df['traj_id'].unique().tolist()
        
        # Build list of valid meta-windows (tasks)
        self.tasks = []
        
        for series_id in all_series:
            series_data = train_df[train_df['traj_id'] == series_id].copy()
            series_data = series_data.sort_values(self.time_idx).reset_index(drop=True)
            
            # Filter valid data
            valid_data = series_data.dropna(subset=[self.target])
            if len(valid_data) == 0:
                continue
            
            min_time = valid_data[self.time_idx].min()
            max_time = valid_data[self.time_idx].max()
            series_length = max_time - min_time + 1
            
            # Check minimum length requirement
            # Meta-window is self-contained - no pre-window encoder history needed
            min_required_length = self.meta_window_length
            
            if series_length < min_required_length:
                continue
            
            if validation_mode:
                # For validation: only take the LAST meta-window
                window_end = max_time
                window_start = window_end - self.meta_window_length + 1
                
                if window_start >= min_time:
                    self.tasks.append({
                        'series_id': series_id,
                        'window_start': window_start,
                        'window_end': window_end
                    })
            else:
                # For training: slide meta-window across series
                # No pre-window encoder history needed - meta-window is self-contained
                earliest_start = min_time
                latest_start = max_time - self.meta_window_length + 1
                
                window_idx = 0
                for window_start in range(earliest_start, latest_start + 1): #, forecast_horizon)
                    window_end = window_start + self.meta_window_length - 1
                    
                    if window_end > max_time:
                        break
                    
                    self.tasks.append({
                        'series_id': series_id,
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_id': window_idx
                    })
                    window_idx += 1
        
        mode_str = "validation" if validation_mode else "training"
        support_mode = "unified" if unified_support else "per-window"
        sws_str = f", S={self.support_window_size}" if self.support_window_size != forecast_horizon else ""
        print(f"MetaWindowTSDataset ({mode_str}) initialized with {len(self.tasks)} meta-window tasks (K={K_support}{sws_str}, support={support_mode})")
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        """
        Returns one meta-window task with K support windows + 1 query window.
        
        Support windows are numbered 1 to K:
        - Support window 1: decoder only, NO encoder history (for K >= 1)
        - Support window k (k > 1): decoder + encoder from previous (k-1) windows
        - Query window: decoder + encoder from all K support windows
        
        For K=0 (0-shot learning), only the query window is created with no encoder history.
        
        Returns:
            dict with:
                - 'support_datasets': list of K TimeSeriesDataSet objects
                - 'query_dataset': TimeSeriesDataSet object for query window
                - 'series_id': series identifier
                - 'K_support': number of support windows
        """
        task_info = self.tasks[idx]
        series_id = task_info['series_id']
        window_start = task_info['window_start']
        window_end = task_info['window_end']
        
        # Get series data
        series_data = self.train_df[self.train_df['traj_id'] == series_id].copy()
        series_data = series_data.sort_values(self.time_idx).reset_index(drop=True)
        
        # Extract data for entire meta-window (no pre-window encoder history)
        full_data = series_data[
            (series_data[self.time_idx] >= window_start) &
            (series_data[self.time_idx] <= window_end)
        ].copy()
        
        if len(full_data) == 0:
            return None
        
        # Anonymize series identity to prevent overfitting on traj_id embedding.
        # Comment out / set anonymize_series_id=False to restore original behavior.
        if self.anonymize_series_id:
            full_data['traj_id'] = '__dummy__'
        
        # Keep original time_idx values for from_dataset() compatibility
        # Don't reset to 0-based indexing

        # Extra kwargs for from_dataset when support window size differs from forecast horizon
        support_ds_kwargs = {}
        if self.support_window_size != self.forecast_horizon:
            support_ds_kwargs['max_prediction_length'] = min(self.K_support * self.support_window_size, self.forecast_horizon)
            support_ds_kwargs['min_prediction_length'] = self.min_prediction_length   #self.support_window_size
        
        if self.unified_support and self.K_support > 0:
            # UNIFIED SUPPORT MODE: create ONE support dataset spanning all K*S timesteps
            support_start = window_start
            support_end = window_start + self.K_support * self.support_window_size - 1
            
            support_data = full_data[
                (full_data[self.time_idx] >= support_start) &
                (full_data[self.time_idx] <= support_end)
            ].copy()
            
            if len(support_data) == 0:
                return None
            
            support_data = support_data.sort_values(self.time_idx).reset_index(drop=True)
            
            try:
                unified_dataset = TimeSeriesDataSet.from_dataset(
                    self.base_dataset,
                    support_data,
                    predict=True,
                    stop_randomization=True,
                    **support_ds_kwargs
                )
                if len(unified_dataset) == 0:
                    print(f"WARNING: Unified support dataset for series {series_id} is empty")
                    return None
                support_datasets = [unified_dataset]  # Single-element list for consistency
            except Exception as e:
                print(f"ERROR creating unified support dataset for series {series_id}: {type(e).__name__}: {str(e)}")
                return None
        else:
            # PER-WINDOW SUPPORT MODE: create K separate support datasets (numbered 1 to K)
            support_datasets = []
            for k in range(1, self.K_support + 1):
                # Each support window is S timesteps (using original time_idx values)
                # Window k is at position (k-1) * S from window_start
                support_start = window_start + (k - 1) * self.support_window_size
                support_end = support_start + self.support_window_size - 1
                
                # For k=1: No encoder history (only decoder window)
                # For k>1: Encoder history includes previous support windows (k-1 windows)
                if k == 1:
                    encoder_length = 0
                    encoder_for_support = support_start  # No encoder, start at decoder
                else:
                    available_history = (k - 1) * self.support_window_size
                    encoder_length = min(available_history, self.max_encoder_length)
                    encoder_for_support = support_start - encoder_length
                
                # Extract data for this support window (encoder + decoder)
                support_data = full_data[
                    (full_data[self.time_idx] >= encoder_for_support) &
                    (full_data[self.time_idx] <= support_end)
                ].copy()
                
                if len(support_data) == 0:
                    return None
                
                # Sort by time_idx but keep original values (don't reset)
                support_data = support_data.sort_values(self.time_idx).reset_index(drop=True)

                
                # Create TimeSeriesDataSet for this support window
                try:
                    support_dataset = TimeSeriesDataSet.from_dataset(
                        self.base_dataset,
                        support_data,
                        predict=False,
                        stop_randomization=False,
                        **support_ds_kwargs
                    )
                    if len(support_dataset) == 0:
                        print(f"WARNING: Support dataset {k} for series {series_id} is empty")
                        return None
                    support_datasets.append(support_dataset)
                except Exception as e:
                    print(f"ERROR creating support dataset {k} for series {series_id}: {type(e).__name__}: {str(e)}")
                    return None
        
        # Create query dataset (last F timesteps of meta-window, using original time_idx)
        query_start = window_start + self.K_support * self.support_window_size
        query_end = query_start + self.forecast_horizon - 1
        
        # Query encoder history: ONLY from within meta-window, no pre-window history
        # - If K > 0: Use support windows as encoder (from start of first support window)
        # - If K = 0: No encoder history (query starts at query window itself)
        if self.K_support > 0:
            # Query encoder spans from first support window to query (capped at max_encoder_length)
            available_history = self.K_support * self.support_window_size
            encoder_length = min(available_history, self.max_encoder_length)
            encoder_for_query = max(window_start, query_start - encoder_length)
        else:
            # K=0 case: no encoder history (query window only)
            encoder_for_query = query_start
        
        query_data = full_data[
            (full_data[self.time_idx] >= encoder_for_query) &
            (full_data[self.time_idx] <= query_end)
        ].copy()
        
        if len(query_data) == 0:
            return None
        
        # Sort by time_idx but keep original values (don't reset)
        query_data = query_data.sort_values(self.time_idx).reset_index(drop=True)
        
        # Create TimeSeriesDataSet for query. predict=True yields a single
        # sample per series; stop_randomization keeps the full encoder length.
        try:
            query_dataset = TimeSeriesDataSet.from_dataset(
                self.base_dataset,
                query_data,
                predict=True,
                stop_randomization=True
            )
            if len(query_dataset) == 0:
                print(f"WARNING: Query dataset for series {series_id} is empty")
                return None
        except Exception as e:
            print(f"ERROR creating query dataset for series {series_id}: {type(e).__name__}: {str(e)}")
            return None
        
        return {
            'support_datasets': support_datasets,
            'query_dataset': query_dataset,
            'series_id': series_id,
            'K_support': self.K_support,
            'window_id': task_info.get('window_id', 0)
        }


class EntityDiverseBatchSampler(torch.utils.data.Sampler):
    """
    Custom batch sampler that ensures each batch contains meta-windows from
    unique entities (traj_id). Each entity appears at most once per batch.
    
    Works with any MetaWindowTSDataset (does not require class information).
    
    For each batch of size N, samples N tasks from N different entities.
    If batch_size > num_entities, some entities will repeat within a batch.
    """
    
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
        """
        Args:
            dataset: MetaWindowTSDataset (or subclass) instance
            batch_size: Number of tasks per batch
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle entity order and tasks within entities
        """
        if not isinstance(dataset, MetaWindowTSDataset):
            raise ValueError("EntityDiverseBatchSampler requires a MetaWindowTSDataset (or subclass)")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Build mapping from entity (series_id) to task indices
        self.tasks_by_entity = {}
        for task_idx, task in enumerate(dataset.tasks):
            entity = task['series_id']
            if entity not in self.tasks_by_entity:
                self.tasks_by_entity[entity] = []
            self.tasks_by_entity[entity].append(task_idx)
        
        self.entity_list = list(self.tasks_by_entity.keys())
        num_entities = len(self.entity_list)
        
        if num_entities < batch_size:
            print(f"Warning: Only {num_entities} unique entities available but batch_size={batch_size}")
            print(f"         Some batches will have repeated entities")
        
        print(f"EntityDiverseBatchSampler: {num_entities} entities, batch_size={batch_size}")
    
    def __iter__(self):
        """Generate batches with entity diversity."""
        entity_list = list(self.entity_list)
        num_entities = len(entity_list)
        
        if self.shuffle:
            random.shuffle(entity_list)
        
        # Prepare task pools for each entity (shuffled if requested)
        entity_task_pools = {}
        for entity in entity_list:
            task_indices = self.tasks_by_entity[entity].copy()
            if self.shuffle:
                random.shuffle(task_indices)
            entity_task_pools[entity] = task_indices
        
        # Generate batches
        batches = []
        
        while True:
            batch = []
            entity_idx = 0
            
            while len(batch) < self.batch_size:
                entity = entity_list[entity_idx % num_entities]
                
                if len(entity_task_pools[entity]) > 0:
                    task_idx = entity_task_pools[entity].pop(0)
                    batch.append(task_idx)
                    entity_idx += 1
                else:
                    entity_idx += 1
                    if entity_idx >= num_entities:
                        any_tasks_left = any(len(pool) > 0 for pool in entity_task_pools.values())
                        if not any_tasks_left:
                            break
            
            if len(batch) < self.batch_size:
                if not self.drop_last and len(batch) > 0:
                    batches.append(batch)
                break
            
            batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        """Estimate number of batches."""
        total_tasks = sum(len(tasks) for tasks in self.tasks_by_entity.values())
        
        if self.drop_last:
            return total_tasks // self.batch_size
        else:
            return (total_tasks + self.batch_size - 1) // self.batch_size


def meta_window_collate_fn(batch):
    """
    Custom collate function for MetaWindowTSDataset.
    Returns batch of meta-window tasks.
    
    Args:
        batch: List of items from MetaWindowTSDataset
        
    Returns:
        List of tasks, where each task is a dict with:
            - 'support_datasets': list of K TimeSeriesDataSet objects
            - 'query_dataset': TimeSeriesDataSet object
            - 'series_id': identifier
            - 'K_support': number of support windows
    """
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return []
    
    return batch 

class TMAML(pl.LightningModule):
    """
    Temporal Model-Agnostic Meta-Learning (T-MAML) for time series forecasting.
    Implements First-Order MAML (FOMAML) with temporal inner-loop updates over expanding support windows.
    
    Note: Uses FOMAML (first-order approximation) for stability and memory efficiency.
    Gradients do not flow through inner-loop updates, only through the adapted model's query predictions.
    """
    
    def __init__(
        self,
        tft_model,
        inner_lr=1e-3,
        inner_steps=5,
        outer_lr=1e-4,
        forecast_horizon=30,
        max_support_samples=128,
        max_query_samples=32,
        average_query_loss_across_shots_per_task=True,
        average_query_loss_across_tasks=True,
        K_support=3,
        inner_grad_clip=0.01,  # the main knob to tune during inner-loop adaptation
        outer_grad_clip=None,
    ):
        """
        Args:
            tft_model: TemporalFusionTransformer model instance
            inner_lr: Learning rate for inner loop (task adaptation)
            inner_steps: Number of gradient steps per support window
            outer_lr: Learning rate for outer loop (meta-update)
            forecast_horizon: Forecast window size (query window size)
            max_support_samples: Max samples to use from support set
            max_query_samples: Max samples to use from query set
            average_query_loss_across_shots_per_task: Average query losses over k-shots within each task
            average_query_loss_across_tasks: Average meta-loss across tasks
            K_support: Number of support windows (K >= 0)
            inner_grad_clip: Max gradient norm for inner-loop clipping (None to disable)
            outer_grad_clip: Max gradient norm for outer-loop clipping (None to disable)
        """
        super().__init__()
        self.save_hyperparameters(ignore=['tft_model'])

        self.model = tft_model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.K_support = K_support
        self.forecast_horizon = forecast_horizon
        self.max_support_samples = max_support_samples
        self.max_query_samples = max_query_samples
        self.average_query_loss_across_shots_per_task = average_query_loss_across_shots_per_task
        self.average_query_loss_across_tasks = average_query_loss_across_tasks
        self.inner_grad_clip = inner_grad_clip
        self.outer_grad_clip = outer_grad_clip

        # Track metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def _compute_mae(self, predictions, target):
        """Compute Mean Absolute Error between predictions and target.
        
        Handles shape mismatches by flattening both tensors if necessary.
        """
        try:
            # Try direct computation if shapes match
            if predictions.shape == target.shape:
                return torch.mean(torch.abs(predictions - target)).item()
            else:
                # Shape mismatch: flatten both and compute MAE
                pred_flat = predictions.flatten()
                target_flat = target.flatten()
                
                # Use minimum length to avoid broadcasting errors
                min_len = min(pred_flat.shape[0], target_flat.shape[0])
                pred_flat = pred_flat[:min_len]
                target_flat = target_flat[:min_len]
                
                return torch.mean(torch.abs(pred_flat - target_flat)).item()
        except Exception as e:
            # If all else fails, return 0.0 and log a warning
            print(f"Warning: MAE computation failed with error {e}. Predictions shape: {predictions.shape}, Target shape: {target.shape}")
            return 0.0
        
    def configure_optimizers(self):
        """Configure outer loop optimizer."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/meta_loss',
            }
        }
    
    def inner_loop_meta_window(self, task_data, mode='train'):
        """
        Meta-window inner loop: sequentially adapt on K support windows, then evaluate query.
        Uses First-Order MAML (FOMAML).
        
        When unified_support is used in the dataset, support_datasets contains a single
        dataset spanning all K*F timesteps instead of K separate datasets.
        
        Args:
            task_data: Dict with 'support_datasets' (list of K datasets, or 1 if unified), 
                       'query_dataset', 'K_support'
            mode: 'train' or 'val'
            
        Returns:
            tuple: (query_loss, support_losses)
        """
        support_datasets = task_data['support_datasets']
        query_dataset = task_data['query_dataset']
        K_support = task_data['K_support']
        
        # Clone model for task-specific adaptation
        task_model = deepcopy(self.model).to(self.device)
        task_model.train() if mode == 'train' else task_model.eval()

        inner_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)

        # Track support losses
        support_losses = []
        
        # Sequentially adapt on each support window (numbered 1 to K)
        # In unified mode, support_datasets has 1 element spanning all K*F timesteps
        num_support_datasets = len(support_datasets)
        with torch.set_grad_enabled(False):  # Don't track gradients through inner loop (FOMAML)
            for k_idx, support_dataset in enumerate(support_datasets, start=1):
                # Sample with replacement so max_support_samples can exceed the
                # window size; one batch covers the whole support dataloader.
                support_sampler = torch.utils.data.RandomSampler(
                    support_dataset,
                    replacement=True,
                    num_samples=self.max_support_samples,
                )
                support_loader = support_dataset.to_dataloader(
                    train=True,
                    shuffle=False,
                    batch_size=self.max_support_samples,
                    num_workers=0,
                    sampler=support_sampler
                )
                
                # Perform multiple gradient steps on this support window
                window_losses = []
                
                # Set model to train mode for inner updates (even during validation)
                task_model.train()
                
                # Create progress bar for inner steps
                if num_support_datasets == 1 and K_support > 1:
                    desc = f"Unified Support (K={K_support})"
                else:
                    desc = f"Support Window {k_idx}/{K_support}"
                pbar = tqdm(range(self.inner_steps), desc=desc, leave=False)
                
                for step in pbar:
                    step_losses = []
                    for batch_data in support_loader:
                        inner_optimizer.zero_grad()
                        
                        # Temporarily enable gradients for this forward/backward
                        with torch.set_grad_enabled(True):
                            # Move to device
                            if isinstance(batch_data, tuple):
                                batch, y = batch_data
                                batch = move_to_device(batch, self.device)
                                target = y[0].to(self.device) if isinstance(y, tuple) else y.to(self.device)
                            else:
                                batch = move_to_device(batch_data, self.device)
                                target = batch[support_dataset.target]
                            
                            # Forward pass
                            output = task_model(batch)
                            
                            # Compute loss
                            if hasattr(output, 'prediction'):
                                loss = task_model.loss(output.prediction, target)
                            else:
                                loss = task_model.loss(output, target)
                            
                            # Backward pass (gradients won't be tracked beyond this)
                            loss.backward()
                            if self.inner_grad_clip is not None:
                                torch.nn.utils.clip_grad_norm_(task_model.parameters(), self.inner_grad_clip)
                            inner_optimizer.step()
                        
                        step_losses.append(loss.item())
                    
                    if len(step_losses) > 0:
                        avg_step_loss = np.mean(step_losses)
                        window_losses.append(avg_step_loss)
                        pbar.set_postfix({'loss': f'{avg_step_loss:.4f}'})
                
                # Store final loss for this support window
                final_window_loss = window_losses[-1] if window_losses else 0.0
                support_losses.append(final_window_loss)

        # Set model back to eval mode for query evaluation (if in validation mode)
        if mode != 'train':
            task_model.eval()
        
        # Evaluate on query set after all support updates
        query_sampler = torch.utils.data.RandomSampler(
            query_dataset,
            replacement=False,
            num_samples=min(self.max_query_samples, len(query_dataset))
        )
        query_loader = query_dataset.to_dataloader(
            train=False,
            shuffle=False,
            batch_size=len(query_dataset), # Process all query samples in one batch for stability
            num_workers=0,
            sampler=query_sampler
        )
        
        query_losses = []
        query_maes = []
        support_maes = []
        
        if mode == 'train':
            # Enable gradients for query evaluation and backprop for meta-update
            task_model.zero_grad()
            
            for batch_data in query_loader:
                # Move to device
                if isinstance(batch_data, tuple):
                    batch, y = batch_data
                    batch = move_to_device(batch, self.device)
                    target = y[0].to(self.device) if isinstance(y, tuple) else y.to(self.device)
                else:
                    batch = move_to_device(batch_data, self.device)
                    target = batch[query_dataset.target]
                
                # Enable gradients for this forward pass
                with torch.set_grad_enabled(True):
                    # Forward pass
                    output = task_model(batch)
                    predictions = output.prediction if hasattr(output, 'prediction') else output
                    
                    # Compute loss
                    if hasattr(output, 'prediction'):
                        loss = task_model.loss(output.prediction, target)
                    else:
                        loss = task_model.loss(output, target)
                    
                    # Compute MAE
                    mae = self._compute_mae(predictions, target)
                    
                    # Store loss value (detached for later aggregation)
                    query_losses.append(loss.detach())
                    query_maes.append(mae)
                    
                    # FOMAML: Backprop immediately and accumulate gradients in task_model
                    loss.backward()
            
            avg_query_loss = torch.stack(query_losses).mean()
            avg_query_mae = np.mean(query_maes) if query_maes else 0.0
        else:
            # Validation mode: just evaluate, no gradients
            with torch.no_grad():
                for batch_data in query_loader:
                    # Move to device
                    if isinstance(batch_data, tuple):
                        batch, y = batch_data
                        batch = move_to_device(batch, self.device)
                        target = y[0].to(self.device) if isinstance(y, tuple) else y.to(self.device)
                    else:
                        batch = move_to_device(batch_data, self.device)
                        target = batch[query_dataset.target]
                    
                    # Forward pass
                    output = task_model(batch)
                    predictions = output.prediction if hasattr(output, 'prediction') else output
                    
                    # Compute loss
                    if hasattr(output, 'prediction'):
                        loss = task_model.loss(output.prediction, target)
                    else:
                        loss = task_model.loss(output, target)
                    
                    # Compute MAE
                    mae = self._compute_mae(predictions, target)
                    
                    query_losses.append(loss)
                    query_maes.append(mae)
                
                avg_query_loss = torch.stack(query_losses).mean()
                avg_query_mae = np.mean(query_maes) if query_maes else 0.0
        
        # Compute support MAEs (from the last step of each support window)
        for support_dataset in support_datasets:
            support_loader = support_dataset.to_dataloader(
                train=False,
                shuffle=False,
                batch_size=len(support_dataset),
                num_workers=0,
            )
            with torch.no_grad():
                for batch_data in support_loader:
                    if isinstance(batch_data, tuple):
                        batch, y = batch_data
                        batch = move_to_device(batch, self.device)
                        target = y[0].to(self.device) if isinstance(y, tuple) else y.to(self.device)
                    else:
                        batch = move_to_device(batch_data, self.device)
                        target = batch[support_dataset.target]
                    
                    output = task_model(batch)
                    predictions = output.prediction if hasattr(output, 'prediction') else output
                    mae = self._compute_mae(predictions, target)
                    support_maes.append(mae)
        
        avg_support_mae = np.mean(support_maes) if support_maes else 0.0
        
        return avg_query_loss, support_losses, task_model, avg_query_mae, avg_support_mae
    
    def outer_loop_meta_window(self, meta_batch, mode='train'):
        """
        Outer loop for meta-window mode: aggregate query losses across tasks and perform meta-update.
        Uses First-Order MAML (FOMAML) for stability.
        
        Args:
            meta_batch: List of meta-window tasks from meta_window_collate_fn
            mode: 'train' or 'val'
            
        Returns:
            dict with aggregated metrics
        """
        if len(meta_batch) == 0:
            return None
        
        # Zero gradients for meta-model
        if mode == 'train':
            self.model.zero_grad()
        
        all_query_losses = []
        all_query_maes = []
        all_support_losses = []
        all_support_maes = []

        # Process each task (meta-window)
        mode_str = "Training" if mode == 'train' else "Validation"
        for task_data in tqdm(meta_batch, desc=f"{mode_str} meta-batch", leave=False):
            query_loss, support_losses, adapted_model, query_mae, support_mae = \
                self.inner_loop_meta_window(task_data, mode=mode)

            all_query_losses.append(query_loss.item() if hasattr(query_loss, 'item') else query_loss)
            all_query_maes.append(query_mae)
            all_support_losses.append(support_losses)
            all_support_maes.append(support_mae)
            
            # FOMAML: Transfer gradients from adapted_model to meta-model
            if mode == 'train':
                for p_meta, p_adapted in zip(self.model.parameters(), adapted_model.parameters()):
                    if p_adapted.grad is not None:
                        if p_meta.grad is None:
                            p_meta.grad = p_adapted.grad.clone()
                        else:
                            p_meta.grad += p_adapted.grad
            
            # Clean up
            del adapted_model
        
        # Aggregate across tasks
        if self.average_query_loss_across_tasks:
            meta_loss = np.mean(all_query_losses)
        else:
            meta_loss = np.sum(all_query_losses)
        
        # Average gradients across tasks
        if mode == 'train' and len(meta_batch) > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad /= len(meta_batch)
        
        # Compute support statistics (average across tasks and windows)
        K = meta_batch[0]['K_support'] if len(meta_batch) > 0 else 0
        avg_support_losses = []
        for k in range(K):
            k_losses = [task_support[k] for task_support in all_support_losses if k < len(task_support)]
            if k_losses:
                avg_support_losses.append(np.mean(k_losses))
        
        # Compute MAE statistics
        avg_query_mae = np.mean(all_query_maes) if all_query_maes else 0.0
        avg_support_mae = np.mean(all_support_maes) if all_support_maes else 0.0

        return {
            'meta_loss': meta_loss,
            'query_losses': all_query_losses,
            'query_mae': avg_query_mae,
            'avg_support_losses': avg_support_losses,
            'avg_support_mae': avg_support_mae,
            'num_tasks': len(meta_batch),
            'K_support': K
        }
    
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step."""
        if batch is None or len(batch) == 0:
            return None

        metrics = self.outer_loop_meta_window(batch, mode='train')

        if metrics is None:
            return None
        
        # Check gradient statistics
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Clip outer-loop gradients
        if self.outer_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.outer_grad_clip)
        
        # Step optimizer
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        
        # Log metrics
        batch_size = metrics['num_tasks']

        # Meta-level metrics
        self.log('train/meta_loss', metrics['meta_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/grad_norm', total_grad_norm, on_step=True, on_epoch=True, batch_size=batch_size)

        self.log('train/query_loss_std', np.std(metrics['query_losses']), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('train/query_mae', metrics['query_mae'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/support_mae', metrics['avg_support_mae'], on_step=True, on_epoch=True, batch_size=batch_size)

        # Log support losses per window (numbered 1 to K)
        for k_idx, support_loss in enumerate(metrics['avg_support_losses'], start=1):
            self.log(f'train/support_window_{k_idx}_loss', support_loss, on_step=True, on_epoch=True, batch_size=batch_size)

        self.training_step_outputs.append(metrics['meta_loss'])

        return None

    def validation_step(self, batch, batch_idx):
        """PyTorch Lightning validation step: evaluate on the last meta-window
        of each validation series."""
        if batch is None or len(batch) == 0:
            return None

        metrics = self.outer_loop_meta_window(batch, mode='val')

        if metrics is None:
            return None

        batch_size = len(batch)

        self.log('val/meta_loss', metrics['meta_loss'], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/query_loss_std', np.std(metrics['query_losses']), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/query_mae', metrics['query_mae'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/support_mae', metrics['avg_support_mae'], on_step=False, on_epoch=True, batch_size=batch_size)

        for k_idx, support_loss in enumerate(metrics['avg_support_losses'], start=1):
            self.log(f'val/support_window_{k_idx}_loss', support_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.validation_step_outputs.append(metrics['meta_loss'])

        return metrics['meta_loss']

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if len(self.training_step_outputs) > 0:
            avg_train_loss = np.mean(self.training_step_outputs)
            print(f"\nEpoch {self.current_epoch} - Avg Train Meta-Loss: {avg_train_loss:.4f}")
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if len(self.validation_step_outputs) > 0:
            avg_val_loss = np.mean(self.validation_step_outputs)
            print(f"Epoch {self.current_epoch} - Avg Val Meta-Loss: {avg_val_loss:.4f}")
            self.validation_step_outputs.clear()
