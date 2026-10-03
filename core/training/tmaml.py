"""
TMAML training procedure: meta-learning a forecaster with the T-MAML algorithm.

The base model (e.g. a TFT) is built by the caller (build_model) and passed
in; this runner wraps it in the TMAML LightningModule, builds the meta-window
datasets + loaders, and fits.

Configurable knobs come from cfg["tmaml"]. A handful of values that were never
varied across experiments (forecast horizon source, loss-averaging flags,
inner-step schedule) are kept as in-function defaults, rather than cluttering
the config.
"""

import lightning.pytorch as pl
import torch.utils.data as data
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from core.models.tmaml import (
    TMAML,
    MetaWindowTSDataset,
    EntityDiverseBatchSampler,
    meta_window_collate_fn,
)
from core.training.common import (
    build_wandb_logger,
    checkpoint_dir,
    make_run_name,
    save_resolved_config,
)

# Values that were fixed across the final experiments.
_NUM_WORKERS = 0
_AVERAGE_QUERY_LOSS_ACROSS_SHOTS = True
_AVERAGE_QUERY_LOSS_ACROSS_TASKS = True


def _build_meta_datasets(train_df, val_df, base_dataset, tcfg, dcfg):
    """Build (meta_train_dataset, meta_val_dataset)."""
    forecast_horizon = dcfg.get("max_prediction_length", 24)

    common = dict(
        base_dataset=base_dataset,
        forecast_horizon=forecast_horizon,
        K_support=tcfg["K_support"],
        unified_support=tcfg.get("unified_support", True),
        anonymize_series_id=tcfg.get("anonymize_series_id", True),
        support_window_size=tcfg.get("support_window_size", 24),
    )

    meta_train_dataset = MetaWindowTSDataset(train_df=train_df, validation_mode=False, **common)
    meta_val_dataset = MetaWindowTSDataset(train_df=val_df, validation_mode=True, **common)
    return meta_train_dataset, meta_val_dataset


def _meta_batch_size(meta_train_dataset, train_df, tcfg):
    """Pick the meta-batch size for the chosen sampling mode."""
    sampling = tcfg.get("sampling", "entity_diverse")

    if sampling == "entity_diverse":
        return max(1, tcfg.get("meta_batch_size", 20))
    # plain random sampling
    train_set_size = train_df["traj_id"].nunique()
    return min(train_set_size, 32)


def _build_loaders(meta_train_dataset, meta_val_dataset, meta_batch_size, sampling):
    """Build the meta-train/val DataLoaders for the chosen sampling mode."""
    collate_fn = meta_window_collate_fn
    val_loader = data.DataLoader(
        meta_val_dataset,
        batch_size=max(1, len(meta_val_dataset)),  # validate on all tasks at once
        shuffle=False,
        num_workers=_NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if sampling == "entity_diverse":
        sampler = EntityDiverseBatchSampler(meta_train_dataset, batch_size=meta_batch_size, drop_last=True, shuffle=True)
        train_loader = data.DataLoader(meta_train_dataset, batch_sampler=sampler, num_workers=_NUM_WORKERS, collate_fn=collate_fn)
    else:
        train_loader = data.DataLoader(meta_train_dataset, batch_size=meta_batch_size, shuffle=True, num_workers=_NUM_WORKERS, collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader


def run_tmaml(model, train_df, val_df, base_dataset, cfg: dict) -> str:
    """Meta-train `model` with T-MAML. Returns best checkpoint path.

    Args:
        model:        untrained base model from build_model(...) (e.g. a TFT)
        train_df:     full meta-train DataFrame
        val_df:       meta-val DataFrame (REQUIRED for TMAML)
        base_dataset: TimeSeriesDataSet built on the training split (schema carrier)
        cfg:          full resolved config (reads cfg["tmaml"] for hyperparameters)
    """
    if val_df is None:
        raise ValueError("TMAML requires a meta-validation set: set dataset.meta_val_path in the config.")

    tcfg = cfg["tmaml"]
    dcfg = cfg["dataset"]
    run_name = make_run_name(cfg)
    ckpt_dir = checkpoint_dir(cfg)
    sampling = tcfg.get("sampling", "entity_diverse")

    meta_train_dataset, meta_val_dataset = _build_meta_datasets(train_df, val_df, base_dataset, tcfg, dcfg)
    meta_batch_size = _meta_batch_size(meta_train_dataset, train_df, tcfg)
    train_loader, val_loader = _build_loaders(meta_train_dataset, meta_val_dataset, meta_batch_size, sampling)

    print(f"[TMAML] meta-train windows: {len(meta_train_dataset)}, meta-val windows: {len(meta_val_dataset)}")
    print(f"[TMAML] sampling={sampling}, meta_batch_size={meta_batch_size}")

    tmaml_model = TMAML(
        tft_model=model,
        inner_lr=tcfg["inner_lr"],
        inner_steps=tcfg["inner_steps"],
        outer_lr=tcfg["outer_lr"],
        forecast_horizon=dcfg.get("max_prediction_length", 24),
        max_support_samples=tcfg.get("max_support_samples", 128),
        max_query_samples=tcfg.get("max_query_samples", 128),
        average_query_loss_across_shots_per_task=_AVERAGE_QUERY_LOSS_ACROSS_SHOTS,
        average_query_loss_across_tasks=_AVERAGE_QUERY_LOSS_ACROSS_TASKS,
        K_support=tcfg["K_support"],
        inner_grad_clip=tcfg.get("inner_grad_clip", 0.01),
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val/meta_loss",
        dirpath=str(ckpt_dir),
        filename=f"{run_name}_{{epoch:02d}}-{{val/meta_loss:.4f}}",
        save_top_k=1,
        mode="min",
        save_last=False,
        every_n_train_steps=5,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_cb,
        EarlyStopping(monitor="val/meta_loss", min_delta=0.001, patience=30, mode="min", verbose=True, strict=True, check_finite=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=tcfg.get("max_epochs", 150),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=build_wandb_logger(cfg, run_name),
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        val_check_interval=tcfg.get("val_check_interval", 5),
    )

    save_resolved_config(cfg, ckpt_dir, run_name)
    trainer.fit(tmaml_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path
    print(f"[TMAML] best checkpoint: {best_path}")
    trainer.logger.experiment.finish()
    return best_path
