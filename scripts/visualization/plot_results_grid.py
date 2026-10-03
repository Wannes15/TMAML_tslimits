"""Combined cross-dataset WQL + MACE vs K figure for the paper (Figure 2).

Loads the per-dataset summary pickles exported by the calibration sections of
notebooks/{FAV,M5,EL}_meta_test_analysis.ipynb (docs/figures/_data/*.pkl) and
builds a single 2-row (WQL, MACE) x 3-column (Favorita, M5, Electricity)
figure. Run from the project root:

    conda activate cold_start_env
    python scripts/visualization/plot_results_grid.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "docs" / "figures" / "_data"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

DATASETS = ["favorita", "m5", "electricity"]
DATASET_LABELS = {"favorita": "Favorita", "electricity": "Electricity", "m5": "M5"}
METRICS = ["wql", "mace"]
METRIC_LABELS = {"wql": "WQL", "mace": "MACE"}

STYLE = {
    "ERM": dict(color="#FF6B6B", linestyle="-", marker="o", label="ERM"),
    "TMAML": dict(color="#4ECDC4", linestyle="-", marker="o", label="TMAML"),
    "Naive": dict(color="#888888", linestyle="--", marker="s", label="Naive"),
}
MODEL_ORDER = ["Naive", "ERM", "TMAML"]


def load_summary(name):
    with open(DATA_DIR / f"{name}_wql_mace_summary.pkl", "rb") as f:
        return pickle.load(f)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel(ax, summary, metric, show_xlabel, show_ylabel):
    ks = summary["K"]
    for model in MODEL_ORDER:
        kw = STYLE[model]
        m = summary[metric][model]
        ax.plot(ks, m["mean"], linewidth=2, **kw)
        ax.fill_between(ks, m["ci_lower"], m["ci_upper"], alpha=0.15, color=kw["color"])
    ax.set_xticks(ks)
    ax.grid(True, linestyle="--", alpha=0.5)
    despine(ax)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    if show_xlabel:
        ax.set_xlabel("K (support windows)", fontsize=11)


def main():
    summaries = {name: load_summary(name) for name in DATASETS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

    for col, name in enumerate(DATASETS):
        summary = summaries[name]
        for row, metric in enumerate(METRICS):
            is_last_row = row == len(METRICS) - 1
            is_first_col = col == 0
            plot_panel(axes[row, col], summary, metric, is_last_row, is_first_col)

    for col, name in enumerate(DATASETS):
        axes[0, col].set_title(DATASET_LABELS[name], fontsize=13)

    handles = [plt.Line2D([0], [0], **STYLE[m]) for m in MODEL_ORDER]
    labels = [STYLE[m]["label"] for m in MODEL_ORDER]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
        ncol=3, frameon=False, fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = OUT_DIR / "results_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
