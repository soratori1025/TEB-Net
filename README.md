# TEB-Net — Thermodynamics-Embedded Baseline Network

Research code for the manuscript *"Thermodynamics-Embedded Baseline Network (TEB-Net) Extrapolates
Molecular Thermochemistry to Fuel-Sized Chains and Flame Temperatures"*.

TEB-Net predicts the gas-phase **ΔfH°298**, **S°298** and **Cp(T)** of C/H/O species from a SMILES
string and returns them as two-range **NASA-7** polynomials. The prediction is the sum of two parts:

* a **frozen additive baseline** built from molecular bond, ring and radical-centre counts
  (explicitly defined descriptors, no learned parameters), and
* a **D-MPNN residual** that corrects the baseline.

An analytic output layer converts the resulting anchors into NASA-7 coefficients, and a high-temperature
completion keeps the exported heat capacity monotonic and below its classical limit up to 3000 K.

## What is in this repository

| path | contents |
|---|---|
| `models/tebnet-v3/tebnet/` | the model package (prior, featuriser, D-MPNN, NASA-7 layer, completion) |
| `models/tebnet-v3/scripts/` | training (`train.py`), evaluation (`deeptherm_eval.py`), descriptor cache (`compute_physics_features.py`), SMILES prediction (`predict_smiles.py`) |
| `models/tebnet-v3/check/` | diagnostic and analysis scripts (ensemble reports, homologous-series controls, NASA-7 and high-temperature checks, prior audits) |
| `models/tebnet-v3/benchmark_results_*/` | stored per-run `results.json` and predictions used for the paper's tables |
| `models-baseline/traditional_models/` | conventional-ML and GNN baselines under the same protocol, plus the CH2 extrapolation benchmark |
| `models-baseline/ch2_extrapolation/` | benchmark outputs: per-model predictions, per-series increments and summary tables |
| `models-baseline/sota_patches/` | local patches applied to the upstream SOTA repositories during the study |
| `data/` | datasets and splits used by the study (DeepTherm, Ureel, WUDILY-CHO, RMG references) |
| `models/tebnet-v2/` | archived earlier model generation (kept for provenance) |

## Environment

Python 3.12 with `torch`, `torch_geometric`, `rdkit`, `lightning`, `numpy`, `pandas`,
`scikit-learn`, `xgboost` and `chemprop` (v2).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric rdkit lightning numpy pandas scikit-learn xgboost chemprop
```

## Quick start

```bash
export PYTHONPATH=$PWD/models/tebnet-v3          # `tebnet` package lives here

# 1) physics-descriptor cache (built once next to the CSV)
python models/tebnet-v3/scripts/compute_physics_features.py --csv models-baseline/deeptherm_data.csv

# 2) train and evaluate on a DeepTherm random split
python models/tebnet-v3/scripts/train.py --split_type random --split_seed 0
python models/tebnet-v3/scripts/deeptherm_eval.py --runs models/tebnet-v3/benchmark_results_deeptherm

# 3) CH2 extrapolation benchmark (Figure 2 and Table 1 of the manuscript)
python models-baseline/traditional_models/ch2_extrapolation.py --list
python models-baseline/traditional_models/ch2_extrapolation.py --only ml-ridge_prior ml-rf_physics
python models-baseline/traditional_models/ch2_extrapolation_dmpnn.py
python models-baseline/traditional_models/ch2_extrapolation_summary.py
```

The benchmark rebuilds every baseline on the same random split 0 and scores it on the five homologous
series up to C16; details and the reproduction check are described in
`models-baseline/ch2_extrapolation/README.md`.

## Data

The 4,997-species DeepTherm set is included as `models-baseline/deeptherm_data.csv` (same row order as
the published Supplementary Data, verified by a SHA-1 fingerprint in `models-baseline/traditional_models/baseline_common.py`).
Further datasets used by the study are under `data/`. Upstream repositories (chemprop, DeepTherm,
HC-Thermochemistry, RMG-database) are **not** redistributed here; install them from their own sources,
the local modifications applied during the study are kept in `models-baseline/sota_patches/`.

## Notes

* Trained checkpoints (~2.5 GB) are not stored in the repository; a training run takes 6–29 min on one
  NVIDIA A100 and the scripts write their own checkpoints and `results.json` files.
* Scripts are launched from the repository root with `PYTHONPATH=models/tebnet-v3` (or from inside
  `models/tebnet-v3/`, where the package is importable directly).
