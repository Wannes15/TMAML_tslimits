"""
Config loading: merge an experiment YAML on top of configs/base.yaml.

The result is a plain nested dict. base.yaml holds every default; an experiment
file lists only what differs. There is no schema and no magic — just a recursive
dict merge plus a cheap typo guard that warns when an experiment sets a key that
doesn't exist in base.yaml (the one downside of plain dicts).
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "configs" / "base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (mutates and returns `base`)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _warn_unknown_keys(base: dict, override: dict, path: str = "", _ignore: tuple = ()) -> None:
    """Warn about keys present in the experiment but absent from base.yaml.

    These are almost always typos (e.g. `hiden_size`). We only warn — we don't
    raise — so genuinely new keys are still allowed. `_ignore` lists top-level
    keys that are legitimately not in base.yaml (e.g. an eval config's
    `checkpoint`), so they don't trip the guard.
    """
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if path == "" and key in _ignore:
            continue
        if key not in base:
            print(f"[config] WARNING: '{where}' is not in base.yaml — typo?")
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _warn_unknown_keys(base[key], value, where)


def load_config(experiment_path: str, base_path: Path = DEFAULT_BASE) -> dict:
    """Load base.yaml, deep-merge the experiment file on top, return the dict."""
    base = yaml.safe_load(Path(base_path).read_text()) or {}
    experiment = yaml.safe_load(Path(experiment_path).read_text()) or {}
    _warn_unknown_keys(base, experiment)
    return deep_merge(base, experiment)


# Suffix used by save_resolved_config(); a checkpoint's sibling config is
# "<run_name>.config.yaml" in the same directory as "<run_name>_<epoch>-....ckpt".
_RUN_CONFIG_SUFFIX = ".config.yaml"


def _find_run_config(checkpoint_path: Path) -> Path | None:
    """Find the training run's saved config that produced `checkpoint_path`.

    Both runners name checkpoints "<run_name>_<epoch>-<metric>.ckpt" and dump
    "<run_name>.config.yaml" beside them. Several runs can share a checkpoint
    dir (e.g. K2 and K3), so we match the config whose run_name is the longest
    prefix of the checkpoint's filename (boundary-aware, so "K2" can't match a
    "K20" checkpoint). Returns None if no sibling config matches.
    """
    ckpt = Path(checkpoint_path)
    stem = ckpt.name[:-len(".ckpt")] if ckpt.name.endswith(".ckpt") else ckpt.stem

    matches = []
    for cfg_file in ckpt.parent.glob(f"*{_RUN_CONFIG_SUFFIX}"):
        run_name = cfg_file.name[: -len(_RUN_CONFIG_SUFFIX)]
        if stem == run_name or stem.startswith(run_name + "_"):
            matches.append((len(run_name), cfg_file))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


def load_eval_config(eval_path: str, base_path: Path = DEFAULT_BASE) -> dict:
    """Resolve an evaluation config (hybrid inherit-then-override).

    Layering, low precedence to high:
        base.yaml  <  the checkpoint's saved run config  <  the eval file.

    The eval file must set `checkpoint`. We then load the training run's
    resolved config (auto-discovered beside the checkpoint, or pointed at
    explicitly via `run_config`) so the dataset schema + model hyperparameters
    match the checkpoint without re-typing them. Anything the eval file sets
    wins — so you can still override a field when needed.
    """
    base = yaml.safe_load(Path(base_path).read_text()) or {}
    eval_cfg = yaml.safe_load(Path(eval_path).read_text()) or {}

    checkpoint = eval_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"Eval config '{eval_path}' must set a top-level 'checkpoint'.")

    explicit = eval_cfg.get("run_config")
    run_cfg_path = Path(explicit) if explicit else _find_run_config(Path(checkpoint))

    if run_cfg_path and Path(run_cfg_path).exists():
        run_cfg = yaml.safe_load(Path(run_cfg_path).read_text()) or {}
        print(f"[eval] inheriting run config: {run_cfg_path}")
    else:
        run_cfg = {}
        print(
            f"[eval] no sibling '{_RUN_CONFIG_SUFFIX}' found for checkpoint "
            f"'{checkpoint}' — dataset/model must be declared in the eval config."
        )

    # base + run config is the set of "known" keys; warn on eval-file typos
    # against it, then layer the eval file on top.
    merged = deep_merge(base, run_cfg)
    _warn_unknown_keys(merged, eval_cfg, _ignore=("checkpoint", "run_config"))
    return deep_merge(merged, eval_cfg)
