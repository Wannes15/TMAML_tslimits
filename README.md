# TMAML
Code for reproducing the results in the paper **"TMAML: Temporal Model-Agnostic
Meta-Learning for Cold-Start Time Series Forecasting"**. T-MAML (`core/models/tmaml.py`) is
a first-order MAML wrapper around a Temporal Fusion Transformer (TFT, from
[pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)) that
meta-learns to adapt to a new series from a handful of observed windows
("shots"). T-MAML is implemented as a PyTorch Lightning module around the TFT.
We compare it against a standard TFT trained jointly and fine-tuned per series
at test time, across three datasets (Electricity, Favorita, M5), for K = 0 to 3
shots (Favorita, M5) or K = 0 to 7 (Electricity).

## Setup

```
conda create -n tmaml python=3.11
conda activate tmaml
pip install -r requirements.txt
```

`requirements.txt` installs CPU-only `torch` by default. For GPU training, install the CUDA build for your setup first (e.g. `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126`), then run the command above.

Run everything from the repo root, so `core` imports resolve.

## Repo layout

```
core/               config, dataset/model/training/evaluation code
scripts/            train.py, evaluate.py, visualization/plot_results_grid.py
configs/            one YAML per (dataset, model, K) training/eval run
notebooks/          meta-set construction + result analysis (paper tables/figures)
data/checkpoints/   trained model weights, one per config
data/meta_test_results/  k-shot eval result pickles (WQL/MACE per series)
docs/figures/       Figure 2 (results_grid.png) and its per-dataset panels
```

## Reproducing the paper's numbers and figures

The result pickles are already in the repo, and the three analysis notebooks
already have their outputs baked in
(`notebooks/{FAV,M5,EL}_meta_test_analysis.ipynb`) — open them to see the exact
WQL/MACE tables and significance tests cited in the paper. To re-run them,
just run the notebooks top to bottom (each needs its raw meta-set pickles
regenerated first, see Data below). Each writes a `docs/figures/_data/*.pkl`
summary; Figure 2 is regenerated from those three summaries with:

```
python scripts/visualization/plot_results_grid.py
```

## Training and evaluation from scratch

```
python scripts/train.py --config configs/fav_tft_tmaml_K1_50items.yaml
python scripts/evaluate.py --config configs/eval_fav_tft_tmaml_K1_50items.yaml
```

`train.py` saves a checkpoint plus a sibling `<run_name>.config.yaml` (full
resolved config) under `data/checkpoints/<dataset>/<model>_<method>/`.
`evaluate.py` takes an eval config pointing at a `checkpoint:`, inherits that
run's dataset/model config automatically, and writes one result pickle per K
under `data/meta_test_results/<dataset>/k_shot/`. `configs/base.yaml` holds
defaults; each experiment config only lists deltas.

## Data

None of the three datasets' meta-set pickles are shipped in this repo. To
regenerate them:

**Electricity** (UCI, CC BY 4.0):

1. `core/preprocessing/electricity_prep.py` downloads and cleans the raw series.
2. Run `notebooks/electricity_meta_set_creation.ipynb` to build the
   meta-train/val/test pickles under `data/electricity/meta_sets/`.

**Favorita / M5** (Kaggle competition data):

1. Download from Kaggle: [Corporación Favorita Grocery Sales
   Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) /
   [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy).
2. Run `core/preprocessing/favorita_prep.py` / `core/preprocessing/M5_prep.py`
   to clean and aggregate the raw files.
3. Run `notebooks/meta_set_creation.ipynb` / `notebooks/M5_meta_set_creation.ipynb`
   to build the meta-train/val/test pickles under `data/favorita/meta_sets/` /
   `data/M5/meta_sets/`, matching the filenames referenced in `configs/`.

Checkpoints for all 32 (dataset, model, K) runs are not tracked in this repo
due to size (753M); get in touch for access, or check back for hosting
(Git LFS / Zenodo) once available.

## Notes

- Electricity TMAML uses `outer_lr=0.001` at K=0 and `outer_lr=0.0001` at
  K=1-7 (see each config's `tag`).
- Only the TFT architecture is used in the paper; `core/models/registry.py`
  keeps the model-builder registry so a new architecture is one `build_*`
  function + one registry entry away, but nothing else is wired in.

<!-- ## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex

``` -->

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### Key Points:
- You may run, study, modify, and distribute the software freely.
- If you modify this code and deploy it over a network (e.g., web app, API), you must make your modified source code available to users under the same AGPL v3 license.

See the [`LICENSE`](LICENSE) file for the full license text, or visit [https://www.gnu.org/licenses/agpl-3.0.en.html](https://www.gnu.org/licenses/agpl-3.0.en.html).

**Copyright © 2025 Wannes Janssens, Matthias Bogaert, Dirk Van den Poel**

## Contact

For questions or collaboration opportunities, please contact:
- **Wannes Janssens**: wanjanss.Janssens@UGent.be
- **Matthias Bogaert**: Matthias.Bogaert@UGent.be
- **Dirk Van den Poel**: Dirk.VandenPoel@UGent.be
