"""
Single training entrypoint.

    python scripts/train.py --config configs/<experiment>.yaml

Ties the three axes together: load config -> build dataset -> build model ->
run the chosen training procedure (ERM or TMAML). Replaces the per-method
train_*.py scripts as the thing you actually run.
"""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.data.registry import build_dataset
from core.models.registry import build_model
from core.training.erm import run_erm
from core.training.tmaml import run_tmaml

RUNNERS = {
    "erm": run_erm,
    "tmaml": run_tmaml,
}


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(cfg: dict):
    """Run one full experiment from a resolved config dict.

    Sets the seed, builds dataset + model, and dispatches to the ERM/TMAML runner.
    Returns the best-checkpoint path. This is the unit that scripts/run_grid.py
    calls once per swept config.
    """
    set_seed(cfg.get("seed", 42))

    method = cfg["method"]
    if method not in RUNNERS:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(RUNNERS)}")

    print("=" * 80)
    print(f"RUN: dataset={cfg['dataset']['name']}  model={cfg['model']['name']}  method={method}")
    print("=" * 80)

    train_df, val_df, base_dataset = build_dataset(cfg)
    model = build_model(cfg["model"], base_dataset)

    best_ckpt = RUNNERS[method](model, train_df, val_df, base_dataset, cfg)
    print(f"\nDONE. Best checkpoint: {best_ckpt}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train a forecasting model (ERM or TMAML).")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML (merged onto configs/base.yaml).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
