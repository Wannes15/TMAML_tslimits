"""
Model registry: map an architecture name (from config) to its builder function.

This is the "model" axis of the experiment cross-product. To add a new
architecture you write a `build_<arch>` in builders.py and register it here —
nothing else in the codebase needs to change.
"""

from core.models.builders import build_tft

# name (as written in config: model.name) -> builder function
MODEL_BUILDERS = {
    "tft": build_tft,
}


def build_model(model_cfg: dict, dataset):
    """Build an untrained model from the `model` config block and a base dataset.

    `model_cfg["name"]` selects the architecture; the rest of the dict is passed
    through to that architecture's builder.
    """
    name = model_cfg["name"].lower()
    if name not in MODEL_BUILDERS:
        available = ", ".join(MODEL_BUILDERS)
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_BUILDERS[name](dataset, model_cfg)


def register_model_builder(name: str, builder):
    """Register a new architecture builder at runtime (mirrors loaders.py)."""
    MODEL_BUILDERS[name.lower()] = builder
