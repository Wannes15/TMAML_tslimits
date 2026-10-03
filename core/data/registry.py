"""
Dataset registry: map a dataset name (from config) to a builder function.

This is the "dataset" axis of the experiment cross-product. A builder takes the
`dataset` block of an experiment config and returns the three things every
training procedure needs:

    (train_df, val_df, base_dataset)

  - train_df:      the full meta-train DataFrame (loaded from meta_train_path)
  - val_df:        the meta-val DataFrame (from meta_val_path), or None
  - base_dataset:  a TimeSeriesDataSet built on the *training split* of train_df.
                   It carries the feature schema + encoders + normalizers, and is
                   what `build_model(...)` and the val/meta datasets are derived from.

Each dataset's own split rule lives in its builder, because it's dataset-specific
knowledge (electricity drops the final day; Favorita uses a fixed time_idx cutoff).
These builders wrap the existing `create_base_*` functions in core/preprocessing
unchanged — we'll fold preprocessing into this package in a later step.
"""

import pandas as pd

from core.preprocessing.electricity_prep import create_base_electricity_dataset
from core.preprocessing.favorita_prep import create_base_favorita_dataset
from core.preprocessing.M5_prep import create_base_M5_dataset, fix_m5_scale_for_modeling


# TimeSeriesDataSet length args. These are NOT shared across datasets, so we don't impose any defaults here. We only
# forward the keys a config actually sets; anything omitted falls through to the
# dataset-correct default in the relevant create_base_* function signature.
_LENGTH_KEYS = (
    "max_encoder_length",
    "max_prediction_length",
    "min_encoder_length",
    "min_prediction_length",
    "randomize_length",
)


def _length_kwargs(cfg: dict) -> dict:
    """Pass through only the length args that are explicitly set (non-null) in the config."""
    return {k: cfg[k] for k in _LENGTH_KEYS if cfg.get(k) is not None}


def build_electricity(cfg: dict):
    """Build the electricity base dataset.

    The training split is method-dependent:
      - ERM:   drop the final day so it can serve as ERM's validation horizon.
      - TMAML: keep all days (it validates on a separate meta-val set).
    """
    dcfg = cfg["dataset"]
    train_df = pd.read_pickle(dcfg["meta_train_path"])
    # TimeSeriesDataSet requires an integer time index.
    if train_df["time_idx"].dtype.kind != "i":
        train_df["time_idx"] = train_df["time_idx"].astype(int)

    max_day = train_df["days_from_start"].max()
    if cfg["method"] == "erm":
        train_split = train_df[train_df["days_from_start"] < max_day]
    else:
        train_split = train_df[train_df["days_from_start"] <= max_day]
    base_dataset = create_base_electricity_dataset(
        train_split, target=dcfg.get("target", "power_usage"), **_length_kwargs(dcfg)
    )

    val_path = dcfg.get("meta_val_path")
    val_df = pd.read_pickle(val_path) if val_path else None
    if val_df is not None and val_df["time_idx"].dtype.kind != "i":
        val_df["time_idx"] = val_df["time_idx"].astype(int)
    return train_df, val_df, base_dataset


def build_favorita(cfg: dict):
    """Build the Favorita base dataset. Training split = time_idx <= split_threshold.

    Favorita uses a fixed train/test boundary rather than a drop-last-day rule, so
    the split is the same for both methods.
    """
    dcfg = cfg["dataset"]
    train_df = pd.read_pickle(dcfg["meta_train_path"])

    threshold = dcfg.get("split_threshold", 304)
    train_split = train_df[train_df["time_idx"] <= threshold]
    base_dataset = create_base_favorita_dataset(train_split, **_length_kwargs(dcfg))

    val_path = dcfg.get("meta_val_path")
    val_df = pd.read_pickle(val_path) if val_path else None
    return train_df, val_df, base_dataset


def build_m5(cfg: dict):
    """Build the M5 base dataset. Training split = time_idx <= split_threshold,
    mirroring build_favorita's fixed-cutoff design.

    fix_m5_scale_for_modeling is applied to train_df and val_df right after
    loading (not left to create_base_M5_dataset), so both carry the corrected
    target scale wherever they're reused directly, e.g. run_erm's validation
    fallback to train_df.
    """
    dcfg = cfg["dataset"]
    train_df = fix_m5_scale_for_modeling(pd.read_pickle(dcfg["meta_train_path"]))

    threshold = dcfg.get("split_threshold", 1829)
    train_split = train_df[train_df["time_idx"] <= threshold]
    base_dataset = create_base_M5_dataset(train_split, **_length_kwargs(dcfg))

    val_path = dcfg.get("meta_val_path")
    val_df = fix_m5_scale_for_modeling(pd.read_pickle(val_path)) if val_path else None
    return train_df, val_df, base_dataset


# name (as written in config: dataset.name) -> builder function
DATASET_BUILDERS = {
    "electricity": build_electricity,
    "favorita": build_favorita,
    "m5": build_m5,
}


def build_dataset(cfg: dict):
    """Build (train_df, val_df, base_dataset) from the full resolved config.

    Takes the whole config (not just the dataset block) because the training split
    is method-dependent for some datasets (see build_electricity).
    """
    name = cfg["dataset"]["name"].lower()
    if name not in DATASET_BUILDERS:
        available = ", ".join(DATASET_BUILDERS)
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_BUILDERS[name](cfg)


def register_dataset_builder(name: str, builder):
    """Register a new dataset builder at runtime."""
    DATASET_BUILDERS[name.lower()] = builder


# Evaluation-side dataset construction (used by core/evaluation/k_shot.py).
# Differs from the training builders above: the base dataset is rebuilt per K
# (trimming the held-out horizon out of the training data first), uses
# eval-specific lengths (min_encoder_length=0, min_prediction_length=support_window_size),
# and returns only the base_dataset — test data lives in cfg["eval"].


def build_eval_electricity(cfg: dict, K: int):
    """Electricity eval base dataset, trimmed for K-shot (old k_shot_test rule)."""
    dcfg, ecfg = cfg["dataset"], cfg["eval"]
    train_df = pd.read_pickle(dcfg["meta_train_path"])

    base_df = train_df
    if K > 0 and "days_from_start" in train_df.columns:
        cutoff_day = train_df["days_from_start"].max() - K
        trimmed = train_df[train_df["days_from_start"] <= cutoff_day]
        if len(trimmed) > 0:
            base_df = trimmed
            print(f"[eval] base dataset cutoff: days_from_start <= {cutoff_day} (K={K})")

    return create_base_electricity_dataset(
        base_df,
        target=dcfg.get("target", "power_usage"),
        max_encoder_length=dcfg.get("max_encoder_length", 168),
        max_prediction_length=dcfg.get("max_prediction_length", 24),
        min_encoder_length=0,
        min_prediction_length=ecfg.get("support_window_size", 24),
        randomize_length=False,
    )


def build_eval_favorita(cfg: dict, K: int):
    """Favorita eval base dataset.

    Mirrors build_eval_electricity: min_encoder_length is forced to 0 (allow
    short support) and min_prediction_length comes from eval.support_window_size,
    not the inherited training dataset config.

    Trims train_df to time_idx <= split_threshold, the same cutoff build_favorita
    uses at training time, rather than a K-dependent heuristic — this must match
    exactly, since the embedding vocabularies fit here need to line up with what
    the checkpoint was actually trained on.
    """
    dcfg, ecfg = cfg["dataset"], cfg["eval"]
    train_df = pd.read_pickle(dcfg["meta_train_path"])
    max_prediction_length = dcfg.get("max_prediction_length", 30)

    threshold = dcfg.get("split_threshold", 304)
    base_df = train_df[train_df["time_idx"] <= threshold] if "time_idx" in train_df.columns else train_df
    print(f"[eval] base dataset cutoff: time_idx <= {threshold} (matches training split)")

    return create_base_favorita_dataset(
        base_df,
        max_encoder_length=dcfg.get("max_encoder_length", 84),
        max_prediction_length=max_prediction_length,
        min_encoder_length=0,
        anonymize_series_id=True,
        min_prediction_length=ecfg.get("support_window_size", 28),
        randomize_length=False,
    )


def build_eval_m5(cfg: dict, K: int):
    """M5 eval base dataset. Mirrors build_eval_favorita, plus fix_m5_scale_for_modeling
    applied to train_df before anything else so the base dataset's encoders and
    normalizers are fit on the same target scale the checkpoint was trained on.
    """
    dcfg, ecfg = cfg["dataset"], cfg["eval"]
    train_df = fix_m5_scale_for_modeling(pd.read_pickle(dcfg["meta_train_path"]))
    max_prediction_length = dcfg.get("max_prediction_length", 28)

    threshold = dcfg.get("split_threshold", 1829)
    base_df = train_df[train_df["time_idx"] <= threshold] if "time_idx" in train_df.columns else train_df
    print(f"[eval] base dataset cutoff: time_idx <= {threshold} (matches training split)")

    return create_base_M5_dataset(
        base_df,
        max_encoder_length=dcfg.get("max_encoder_length", 84),
        max_prediction_length=max_prediction_length,
        min_encoder_length=0,
        anonymize_series_id=True,
        min_prediction_length=ecfg.get("support_window_size", 28),
        randomize_length=False,
    )


EVAL_DATASET_BUILDERS = {
    "electricity": build_eval_electricity,
    "favorita": build_eval_favorita,
    "m5": build_eval_m5,
}


def build_eval_base_dataset(cfg: dict, K: int):
    """Build the eval base TimeSeriesDataSet for a given K (rebuilt per K)."""
    name = cfg["dataset"]["name"].lower()
    if name not in EVAL_DATASET_BUILDERS:
        available = ", ".join(EVAL_DATASET_BUILDERS)
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return EVAL_DATASET_BUILDERS[name](cfg, K)
