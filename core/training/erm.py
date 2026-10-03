"""
ERM training procedure: plain supervised training of a forecasting model.

This is the "train a base model the normal way" path — no meta-learning.
Hyperparameters come from the `erm:` config block.
"""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet

from core.training.common import (
    build_wandb_logger,
    checkpoint_dir,
    make_run_name,
    save_resolved_config,
)


def run_erm(model, train_df, val_df, base_dataset, cfg: dict) -> str:
    """Train `model` with standard supervised learning. Returns best checkpoint path.

    Args:
        model:        untrained model from build_model(...)
        train_df:     full meta-train DataFrame
        val_df:       meta-val DataFrame, or None
        base_dataset: TimeSeriesDataSet built on the training split (schema carrier)
        cfg:          the full resolved config (reads cfg["erm"] for hyperparameters)
    """
    ecfg = cfg["erm"]
    run_name = make_run_name(cfg)
    ckpt_dir = checkpoint_dir(cfg)

    # Validation set: use a dedicated meta-val df if provided, otherwise build
    # one from the training df in predict mode.
    val_source = val_df if val_df is not None else train_df
    val_dataset = TimeSeriesDataSet.from_dataset(
        base_dataset, val_source, predict=True, stop_randomization=True
    )

    # Dynamic batch size, as before (never larger than the dataset). num_workers=0
    # avoids multiprocessing issues with the manual normalizer state.
    batch_size = min(len(base_dataset), ecfg.get("batch_size", 64))
    train_dl = base_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_name}_{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        LearningRateMonitor(),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=ecfg.get("early_stopping_patience", 100),
            mode="min",
            verbose=True,
        ),
        checkpoint_cb,
    ]

    trainer = pl.Trainer(
        max_epochs=ecfg.get("max_epochs", 500),
        default_root_dir=str(ckpt_dir),
        gradient_clip_val=ecfg.get("gradient_clip_val", 0.01),
        callbacks=callbacks,
        logger=build_wandb_logger(cfg, run_name),
        accelerator="auto",
        val_check_interval=ecfg.get("val_check_interval", 5),
        log_every_n_steps=1,
    )

    save_resolved_config(cfg, ckpt_dir, run_name)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = checkpoint_cb.best_model_path
    print(f"[ERM] best checkpoint: {best_path}")
    trainer.logger.experiment.finish()
    return best_path
