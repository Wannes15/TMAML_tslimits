"""
K-shot evaluation procedure (config-driven wiring).

This is the eval-side twin of core/training/{erm,tmaml}.py: it takes a resolved
config and runs the k-shot adaptation test. It does NOT reimplement the test math
— `k_shot_test(...)` and friends in core/evaluation/k_shot_test.py are imported
unchanged. This module only replaces the hardcoded `main()` wiring with values
pulled from the config (which inherits the checkpoint's dataset+model blocks; see
core.config.load_eval_config).

Flow, per K in cfg["eval"]["K"]:
    build_eval_base_dataset(cfg, K)  ->  k_shot_test(...)  ->  save_results(...)
"""

from pathlib import Path

import pandas as pd

from core.data.registry import build_eval_base_dataset
from core.preprocessing.M5_prep import fix_m5_scale_for_modeling
from core.training.common import PROJECT_ROOT, make_run_name
from core.evaluation.k_shot_test import k_shot_test, save_results

# Per-architecture: which model-block keys the loader's __init__ accepts.
# The model block is the full deep-merged config (every default from base.yaml),
# so we filter to just the keys each loader knows about.
_LOADER_KEYS = {
    "tft": (
        "learning_rate",
        "hidden_size",
        "attention_head_size",
        "dropout",
        "hidden_continuous_size",
        "output_size",
        "optimizer",
        "anonymize_series_id",
    ),
}


def _model_config(cfg: dict) -> dict:
    """The architecture hyperparameters to reconstruct the model for loading."""
    mcfg = cfg["model"]
    arch = mcfg["name"].lower()
    keys = _LOADER_KEYS.get(arch, _LOADER_KEYS["tft"])
    return {k: mcfg[k] for k in keys if k in mcfg}


def _resolve_is_maml(cfg: dict):
    """True/False from an explicit flag or the inherited training method;
    None means "let the loader auto-detect from the checkpoint"."""
    explicit = cfg["eval"].get("is_maml")
    if explicit is not None:
        return explicit
    method = cfg.get("method")
    if method is not None:
        return method == "tmaml"
    return None


def _results_dir(cfg: dict) -> Path:
    """Explicit eval.results_dir (absolute, or relative to project root), else
    derived as data/meta_test_results/<dataset>/k_shot."""
    rd = cfg["eval"].get("results_dir")
    if rd:
        rd = Path(rd)
        return rd if rd.is_absolute() else PROJECT_ROOT / rd
    return PROJECT_ROOT / "data" / "meta_test_results" / cfg["dataset"]["name"] / "k_shot"


def run_k_shot(cfg: dict):
    """Run k-shot evaluation for every K in cfg["eval"]["K"]. Returns saved paths."""
    ecfg = cfg["eval"]

    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Eval config must set 'checkpoint'.")
    test_path = ecfg.get("test_data")
    if not test_path:
        raise ValueError("Set eval.test_data to the held-out meta-test pickle.")

    test_df = pd.read_pickle(test_path)
    if cfg["dataset"]["name"].lower() == "m5":
        # base_dataset is fit on the corrected target scale; test_df needs it too.
        test_df = fix_m5_scale_for_modeling(test_df)
    if "time_idx" in test_df.columns and test_df["time_idx"].dtype.kind != "i":
        # TimeSeriesDataSet requires an integer time index.
        test_df["time_idx"] = test_df["time_idx"].astype(int)
    is_maml = _resolve_is_maml(cfg)
    model_config = _model_config(cfg)
    architecture = cfg["model"]["name"]
    results_dir = _results_dir(cfg)
    run_name = ecfg.get("run_name") or make_run_name(cfg)
    inner_grad_clip = ecfg.get("inner_grad_clip")  # null in YAML -> None disables clipping

    print("=" * 80)
    print(f"EVAL: checkpoint={checkpoint}")
    print(f"      dataset={cfg['dataset']['name']}  arch={architecture}  is_maml={is_maml}")
    print(f"      K={ecfg['K']}  test_series={test_df['traj_id'].nunique()}  run_name={run_name}")
    print("=" * 80)

    use_wandb = ecfg.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get("wandb", {}).get("project", "cold_start"), name=run_name)

    saved_paths = []
    for K in ecfg["K"]:
        base_dataset = build_eval_base_dataset(cfg, K)
        results = k_shot_test(
            test_df=test_df,
            model_checkpoint_path=str(checkpoint),
            base_dataset=base_dataset,
            K=K,
            architecture=architecture,
            model_config=model_config,
            adaptation_steps=ecfg["adaptation_steps"],
            adaptation_lr=ecfg["adaptation_lr"],
            is_maml=is_maml,
            unified_support=ecfg["unified_support"],
            support_window_size=ecfg["support_window_size"],
            anonymize_series_id=ecfg["anonymize_series_id"],
            inner_grad_clip=inner_grad_clip,
            use_wandb=use_wandb,
            take_last_as_query=ecfg["take_last_as_query"],
            save_predictions=ecfg.get("save_predictions", False),
        )
        saved_paths.append(save_results(results, results_dir, run_name))

    if use_wandb:
        import wandb
        wandb.finish()

    return saved_paths
