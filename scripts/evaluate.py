"""
Single evaluation entrypoint.

    python scripts/evaluate.py --config configs/eval_<experiment>.yaml

The eval-side twin of scripts/train.py. The config points at a trained checkpoint
and inherits that run's dataset+model blocks (see core.config.load_eval_config),
so you only specify what's eval-specific (test data, K values, adaptation knobs).
Flow: load_eval_config -> run_k_shot (loops K, runs the k-shot adaptation test,
saves one result pickle per K).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_eval_config
from core.evaluation.k_shot import run_k_shot
from core.evaluation.k_shot_test import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run k-shot evaluation of a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML (inherits the checkpoint's saved config).")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    set_seed(cfg.get("seed", 42))

    saved_paths = run_k_shot(cfg)
    print("\nDONE. Saved results:")
    for p in saved_paths:
        print(f"  {p}")
    return saved_paths


if __name__ == "__main__":
    main()
