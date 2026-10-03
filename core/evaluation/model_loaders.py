"""
Model Loading Utilities for Meta-Testing

Provides abstract interface and implementations for loading different model architectures.
Supports both TMAML meta-learned models and baseline pre-trained models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from core.models.builders import _anonymized_embedding_sizes


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class ModelLoader(ABC):
    """Abstract base class for loading models from checkpoints."""
    
    @abstractmethod
    def load(
        self,
        checkpoint_path: str,
        base_dataset: TimeSeriesDataSet,
        device: str,
        is_maml: Optional[bool] = None
    ) -> Any:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            base_dataset: Base dataset for model configuration
            device: Device to load model on
            is_maml: Whether checkpoint is TMAML (auto-detect if None)
        
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def auto_detect_maml(self, checkpoint_path: str, device: str) -> bool:
        """
        Auto-detect if checkpoint is TMAML or baseline.
        
        Args:
            checkpoint_path: Path to checkpoint
            device: Device for loading
        
        Returns:
            True if TMAML, False if baseline
        """
        pass


# ============================================================================
# TFT MODEL LOADER
# ============================================================================

class TFTModelLoader(ModelLoader):
    """Loader for Temporal Fusion Transformer models."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        hidden_size: int = 240,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        output_size: int = 7,
        optimizer: str = "ranger",
        anonymize_series_id: bool = True,
    ):
        """
        Initialize TFT loader with model hyperparameters.

        Args:
            learning_rate: Learning rate for optimizer
            hidden_size: Hidden size for LSTM and attention
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
            output_size: Number of quantiles to predict
            optimizer: Optimizer type
            anonymize_series_id: Whether the checkpoint was trained with the
                traj_id embedding shrunk to dim 1 (core.models.builders.build_tft's
                anonymize_series_id trick). Needed when rebuilding the architecture
                from scratch (the TMAML path, and the baseline manual-fallback path)
                — load_from_checkpoint doesn't need this since it restores the
                checkpoint's own saved hyperparameters, embedding sizes included.
        """
        self.model_config = {
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'attention_head_size': attention_head_size,
            'dropout': dropout,
            'hidden_continuous_size': hidden_continuous_size,
            'output_size': output_size,
            'log_interval': 1,
            'reduce_on_plateau_patience': 4,
            'optimizer': optimizer,
        }
        self.anonymize_series_id = anonymize_series_id
        # Avoid printing the same mismatch tensor list for every per-series reload.
        self._detailed_mismatch_logs_by_checkpoint = {}
    
    def auto_detect_maml(self, checkpoint_path: str, device: str) -> bool:
        """Auto-detect if checkpoint is TMAML by checking for 'model.' prefix."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        is_maml = any(key.startswith('model.') for key in checkpoint['state_dict'].keys())
        return is_maml
    
    def load(
        self,
        checkpoint_path: str,
        base_dataset: TimeSeriesDataSet,
        device: str,
        is_maml: Optional[bool] = None
    ) -> TemporalFusionTransformer:
        """Load TFT model from checkpoint (TMAML or baseline)."""
        if is_maml is None:
            is_maml = self.auto_detect_maml(checkpoint_path, device)
            model_type = 'TMAML' if is_maml else 'Baseline TFT'
            print(f"Auto-detected checkpoint type: {model_type}")
        
        if is_maml:
            return self._load_maml_checkpoint(checkpoint_path, base_dataset, device)
        else:
            return self._load_baseline_checkpoint(checkpoint_path, base_dataset, device)
    
    def _load_maml_checkpoint(
        self,
        checkpoint_path: str,
        base_dataset: TimeSeriesDataSet,
        device: str
    ) -> TemporalFusionTransformer:
        """Extract TFT model from TMAML checkpoint."""
        # Infer quantiles from output_size to ensure loss function matches model output
        output_size = self.model_config.get('output_size', 7)
        if output_size == 3:
            loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        elif output_size == 9:
            loss = QuantileLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        else:
            loss = QuantileLoss()  # default 7 quantiles

        rebuild_kwargs = {}
        if self.anonymize_series_id:
            embedding_sizes = _anonymized_embedding_sizes(base_dataset)
            if embedding_sizes is not None:
                rebuild_kwargs["embedding_sizes"] = embedding_sizes

        model = TemporalFusionTransformer.from_dataset(
            base_dataset,
            loss=loss,
            **self.model_config,
            **rebuild_kwargs,
        )

        # Load MAML checkpoint and extract TFT parameters
        maml_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tft_state_dict = {
            key[6:]: value  # Remove 'model.' prefix
            for key, value in maml_checkpoint['state_dict'].items()
            if key.startswith('model.')
        }
        
        # Filter out keys with shape mismatches (e.g. embedding sizes differ between
        # training and test sets). These layers keep their randomly initialized values.
        current_state = model.state_dict()
        compatible_state_dict = {}
        skipped_keys = []
        for key, value in tft_state_dict.items():
            if key in current_state and current_state[key].shape != value.shape:
                skipped_keys.append((key, value.shape, current_state[key].shape))
            else:
                compatible_state_dict[key] = value
        
        model.load_state_dict(compatible_state_dict, strict=False)
        total_keys = len(tft_state_dict)
        loaded_keys = len(compatible_state_dict)

        ckpt_key = str(Path(checkpoint_path).resolve())
        already_logged = self._detailed_mismatch_logs_by_checkpoint.get(ckpt_key, False)

        if skipped_keys and not already_logged:
            for key, ckpt_shape, model_shape in skipped_keys:
                print(f"  ⚠️  Skipping '{key}': checkpoint {ckpt_shape} vs model {model_shape}")
            self._detailed_mismatch_logs_by_checkpoint[ckpt_key] = True
            print(
                f"  ℹ️  Loaded {loaded_keys}/{total_keys} model tensors "
                f"({len(skipped_keys)} skipped due to shape mismatch)"
            )
        elif skipped_keys:
            print(
                f"  ℹ️  Loaded {loaded_keys}/{total_keys} model tensors "
                f"({len(skipped_keys)} skipped due to shape mismatch; details shown earlier)"
            )
        else:
            print(f"  ℹ️  Loaded {loaded_keys}/{total_keys} model tensors (0 skipped due to shape mismatch)")

        print(f"✅ Loaded TMAML model from {checkpoint_path}")
        
        return model.to(device)
    
    def _load_baseline_checkpoint(
        self,
        checkpoint_path: str,
        base_dataset: TimeSeriesDataSet,
        device: str
    ) -> TemporalFusionTransformer:
        """Load baseline TFT model from PyTorch Lightning checkpoint."""
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(
                checkpoint_path,
                map_location=device
            )
            print(f"✅ Loaded baseline TFT model from {checkpoint_path}")
            return model.to(device)
        except Exception as e:
            print(f"⚠️  Failed direct load: {e}")
            print("Attempting manual state_dict loading...")
            
            # Match loss to output_size
            output_size = self.model_config.get('output_size', 7)
            if output_size == 3:
                loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
            elif output_size == 9:
                loss = QuantileLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            else:
                loss = QuantileLoss()

            rebuild_kwargs = {}
            if self.anonymize_series_id:
                embedding_sizes = _anonymized_embedding_sizes(base_dataset)
                if embedding_sizes is not None:
                    rebuild_kwargs["embedding_sizes"] = embedding_sizes

            model = TemporalFusionTransformer.from_dataset(
                base_dataset,
                loss=loss,
                **self.model_config,
                **rebuild_kwargs,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"✅ Loaded baseline TFT model (manual) from {checkpoint_path}")
            
            return model.to(device)


# ============================================================================
# MODEL LOADER REGISTRY
# ============================================================================

_MODEL_LOADERS = {
    'tft': TFTModelLoader,
    # Future architectures can be registered here:
}


def get_model_loader(architecture: str = 'tft', **kwargs) -> ModelLoader:
    """
    Factory function to get model loader by architecture name.
    
    Args:
        architecture: Model architecture name (currently 'tft')
        **kwargs: Architecture-specific configuration parameters
    
    Returns:
        ModelLoader instance for the specified architecture
    
    Raises:
        ValueError: If architecture is not registered
    
    Examples:
        >>> loader = get_model_loader('tft', hidden_size=240)
        >>> model = loader.load(checkpoint_path, base_dataset, device)
    """
    architecture = architecture.lower()
    
    if architecture not in _MODEL_LOADERS:
        available = ', '.join(_MODEL_LOADERS.keys())
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {available}"
        )
    
    loader_class = _MODEL_LOADERS[architecture]
    return loader_class(**kwargs)


def register_model_loader(architecture: str, loader_class: type):
    """
    Register a new model loader.
    
    Args:
        architecture: Name to register under
        loader_class: ModelLoader subclass
    
    Example:
        >>> register_model_loader('custom_model', CustomModelLoader)
    """
    if not issubclass(loader_class, ModelLoader):
        raise TypeError(f"{loader_class} must inherit from ModelLoader")
    
    _MODEL_LOADERS[architecture.lower()] = loader_class
    print(f"✅ Registered model loader: {architecture}")
