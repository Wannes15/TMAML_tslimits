"""
Shared helpers for the training procedures (ERM, TMAML).

Run names and checkpoint paths are derived from the resolved config, and the
full config is saved next to each checkpoint so a run's settings are always
recoverable.
"""

from pathlib import Path

import yaml
from lightning.pytorch.loggers import WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def make_run_name(cfg: dict) -> str:
    """A readable, config-derived run name, e.g. 'electricity_tft_tmaml_K3'.

    A free-text `tag` in the config (top level) is appended verbatim, so you can
    distinguish otherwise-identical runs (e.g. tag: 'ablation1' ->
    'electricity_tft_tmaml_K3_ablation1'). Applies to both ERM and TMAML.
    """
    name = f"{cfg['dataset']['name']}_{cfg['model']['name']}_{cfg['method']}"
    if cfg["method"] == "tmaml":
        name += f"_K{cfg['tmaml']['K_support']}"
    tag = cfg.get("tag")
    if tag:
        name += f"_{tag}"
    return name


def checkpoint_dir(cfg: dict) -> Path:
    """data/checkpoints/<dataset>/<model>_<method>/ — created if missing."""
    d = (
        PROJECT_ROOT
        / "data" / "checkpoints"
        / cfg["dataset"]["name"]
        / f"{cfg['model']['name']}_{cfg['method']}"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_wandb_logger(cfg: dict, run_name: str) -> WandbLogger:
    wandb_cfg = cfg.get("wandb", {})
    return WandbLogger(
        project=wandb_cfg.get("project", "cold_start"),
        name=run_name,
        log_model=False,
    )


def save_resolved_config(cfg: dict, out_dir: Path, run_name: str) -> Path:
    """Dump the fully-merged config next to the checkpoint for provenance."""
    path = Path(out_dir) / f"{run_name}.config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
