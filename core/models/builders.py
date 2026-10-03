"""
Model builders: construct an *untrained* forecasting model from a base dataset.

Each builder takes a base `TimeSeriesDataSet` and a plain config dict (the `model`
block of an experiment config) and returns a model ready to be trained — either
directly (ERM) or wrapped inside the TMAML LightningModule.

This is the single place that knows how to call `<Model>.from_dataset(...)` for
each architecture.
"""

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.nn.embeddings import get_embedding_size


def _quantile_loss(output_size: int) -> QuantileLoss:
    """Build a QuantileLoss whose number of quantiles matches `output_size`.

    Keeping the loss in sync with `output_size` avoids the silent shape
    mismatches we used to hit when loading checkpoints (see model loaders).
    """
    if output_size == 3:
        return QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    if output_size == 9:
        return QuantileLoss(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return QuantileLoss()  # pytorch-forecasting default: 7 quantiles


def _anonymized_embedding_sizes(dataset: TimeSeriesDataSet):
    """Return embedding sizes with the `traj_id` embedding shrunk to dimension 1.

    We don't touch the DataFrame (that would break group_ids when several
    series share a time_idx). Instead we build the default embedding sizes for
    every categorical and only collapse `traj_id` to dim=1, making the model
    effectively series-agnostic.

    Returns None when `traj_id` is not a categorical model input (nothing to do).
    """
    embedding_sizes = {
        cat: (
            len(dataset._categorical_encoders[cat].classes_),
            get_embedding_size(len(dataset._categorical_encoders[cat].classes_)),
        )
        for cat in dataset.categoricals
        if cat in dataset._categorical_encoders
    }
    if "traj_id" not in embedding_sizes:
        return None
    num_traj_classes = embedding_sizes["traj_id"][0]
    embedding_sizes["traj_id"] = (num_traj_classes, 1)
    return embedding_sizes


def build_tft(dataset: TimeSeriesDataSet, cfg: dict) -> TemporalFusionTransformer:
    """Build a Temporal Fusion Transformer from a base dataset.

    `cfg` is the `model` block of an experiment config. Recognised keys:
        hidden_size, attention_head_size, dropout, output_size,
        optimizer, learning_rate, anonymize_series_id
    """
    output_size = cfg.get("output_size", 9)

    extra_kwargs = {}
    if cfg.get("anonymize_series_id", False):
        embedding_sizes = _anonymized_embedding_sizes(dataset)
        if embedding_sizes is not None:
            extra_kwargs["embedding_sizes"] = embedding_sizes

    # Attention-interpretation logging (log_interval=1) divides by
    # attention_occurances.max() over the encoder, undefined when the encoder
    # is empty (K=0 configs). Disable it in that case.
    default_log_interval = -1 if dataset.max_encoder_length == 0 else 1

    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=cfg.get("learning_rate", 0.001),
        hidden_size=cfg.get("hidden_size", 160),
        attention_head_size=cfg.get("attention_head_size", 4),
        dropout=cfg.get("dropout", 0.1),
        output_size=output_size,
        loss=_quantile_loss(output_size),
        log_interval=cfg.get("log_interval", default_log_interval),
        reduce_on_plateau_patience=4,
        optimizer=cfg.get("optimizer", "ranger"),
        **extra_kwargs,
    )
