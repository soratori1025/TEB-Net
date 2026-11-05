# TEB-Net — Thermodynamics-Embedded Baseline Network

Research code for the manuscript *"Thermodynamics-Embedded Baseline Network (TEB-Net) Extrapolates Molecular Thermochemistry to Fuel-Sized Chains and Flame Temperatures"*.

TEB-Net predicts gas-phase **ΔfH°₂₉₈**, **S°₂₉₈** and **Cp(T)** of C/H/O species from a SMILES string and returns them as two-range **NASA-7** polynomials. The prediction is the sum of:

- A **frozen additive prior** built from molecular bond, ring and radical-centre counts plus fixed statistical-mechanical entropy terms (explicitly defined descriptors, no learned parameters in this component).
- A **D-MPNN residual** that learns property-specific corrections to the prior.

An analytic output layer converts the resulting anchors into NASA-7 coefficients, and a high-temperature completion keeps the exported heat capacity monotonic and below its classical limit up to 3000 K.

## Repository structure

```
TEB-Net/
├── models/tebnet-v3/          # Current model (manuscript version)
│   ├── tebnet/                # Core package: prior, featuriser, D-MPNN, NASA-7, completion
│   ├── scripts/               # Training, evaluation, inference, diagnostics
│   ├── check/                 # Analysis and validation scripts
│   └── benchmark_results_*/   # Stored per-run results.json and predictions
├── benchmarks/                # Baseline models and reproduction scripts
│   ├── traditional/           # Ridge, RF, XGBoost on structural descriptors
│   ├── gnn/                   # GIN, GCN, GAT, AttentiveFP baselines
│   ├── ch2_extrapolation/     # CH₂ increment benchmark (Figure 2, Table 1)
│   ├── sota_patches/          # Local patches applied to upstream SOTA repos
│   ├── baseline_common.py     # Shared utilities for all baselines
│   └── deeptherm_eval.py      # Unified evaluation across splits
├── data/                      # Datasets and splits (DeepTherm, Ureel, WUDILY-CHO, RMG)
├── archive/tebnet-v2/         # Archived earlier model generation (provenance only)
└── README.md                  # This file
```

## Environment

Python 3.12 with the following dependencies:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch torch_geometric rdkit lightning numpy pandas scikit-learn xgboost chemprop
```

All scripts assume `PYTHONPATH` includes `models/tebnet-v3` so that the `tebnet` package is importable:

```bash
export PYTHONPATH=$PWD/models/tebnet-v3              # Linux/macOS
set PYTHONPATH=%CD%\models\tebnet-v3                 # Windows CMD
$env:PYTHONPATH="$PWD\models\tebnet-v3"              # PowerShell
```

## Quick start

### 1. Compute physics-descriptor cache (one-time)

The additive prior requires precomputed structural and statistical-mechanical descriptors. Build the cache once next to the dataset CSV:

```bash
python models/tebnet-v3/scripts/compute_physics_features.py \
    --csv models-baseline/deeptherm_data.csv
```

This produces `deeptherm_data.physics_v15.csv` (~1.8 MB) containing 69 descriptor columns per species.

### 2. Train and evaluate on DeepTherm

```bash
# Train on random split 0 (default hyperparameters from the manuscript)
python models/tebnet-v3/scripts/train.py --split_type random --split_seed 0

# Evaluate all runs in a benchmark directory
python models/tebnet-v3/scripts/deeptherm_eval.py \
    --runs models/tebnet-v3/benchmark_results_deeptherm
```

Training takes 6–29 min on one NVIDIA A100 depending on the split. Checkpoints and `results.json` are written automatically; they are not stored in this repository.

### 3. Predict from SMILES

```bash
python models/tebnet-v3/scripts/predict_smiles.py \
    --checkpoint models/tebnet-v3/benchmark_results_deeptherm/random_0/member_0/checkpoint.pt \
    --smiles "CCCCCCCC" \
    --output n_octane_nasa7.json
```

The output contains NASA-7 coefficients for both temperature ranges, reference-state ΔfH°₂₉₈ and S°₂₉₈, and validity flags.

### 4. Reproduce the CH₂ extrapolation benchmark

This benchmark evaluates how well each model preserves per-CH₂ enthalpy increments along five homologous series extended to C₁₆ (Figure 2 and Table 1 of the manuscript):

```bash
# List available baselines
python benchmarks/ch2_extrapolation/ch2_extrapolation.py --list

# Run specific baselines
python benchmarks/ch2_extrapolation/ch2_extrapolation.py --only ml-ridge_prior ml-rf_physics

# Run Chemprop D-MPNN baseline
python benchmarks/ch2_extrapolation/ch2_extrapolation_dmpnn.py

# Aggregate all results into summary tables
python benchmarks/ch2_extrapolation/ch2_extrapolation_summary.py
```

Precomputed predictions and verification checksums are in `benchmarks/ch2_extrapolation/`. Details are in `benchmarks/ch2_extrapolation/README.md`.

## Using TEB-Net as a baseline

If you want to compare your own model against TEB-Net:

1. **Install** this repository and set `PYTHONPATH` as above.
2. **Load the package** in your code:
   ```python
   from tebnet.model import TEBNetV20
   from tebnet.data.featurizer import ThermoAtomFeaturizerV3
   ```
3. **Use the same data splits.** Split functions are in `models/tebnet-v3/tebnet/data/splits.py`. The DeepTherm random/complexity/scaffold/functional-group splits are deterministic given the seed.
4. **Match the training protocol.** Default hyperparameters are defined via argparse in `models/tebnet-v3/scripts/train.py`. Key settings: Huber loss, AdamW (lr 8e-4, weight decay 5e-5), 100 epochs, batch size 32, separate enthalpy and entropy/Cp models.
5. **Evaluate identically.** Use `benchmarks/deeptherm_eval.py` or replicate its bootstrap MAE estimation (5000 molecule-level resamples).
6. **Report the exact version.** The current model is `TEBNetV20` with `FEATURES_VERSION = "v15.3"` (defined in `models/tebnet-v3/tebnet/data/featurizer.py`). State this in your paper to ensure reproducibility.

## Data

| Dataset | Species | Location | Notes |
|---------|---------|----------|-------|
| DeepTherm | 4,997 C/H/O | `models-baseline/deeptherm_data.csv` | Same row order as published Supplementary Data; SHA-1 verified in `benchmarks/baseline_common.py` |
| Ureel hydrocarbons | 330 | `data/ureel/` | CBS-QB3 + 1D hindered rotors |
| Ureel radicals | 426 | `data/ureel/` | Same level of theory |
| Ureel carbenium ions | 162 | `data/ureel/` | Same level of theory |
| WUDILY-CHO | 6,054 | `data/wudily/` | After removal of 20 InChIKey duplicates |
| RMG references | 666 | `data/rmg/` | 47 evaluated + 619 quantum-chemical Cp references up to 3000 K |

Upstream repositories (chemprop, DeepTherm, HC-Thermochemistry, RMG-database) are **not** redistributed here. Install them from their own sources; local modifications applied during the study are kept in `benchmarks/sota_patches/`.

## Terminology note

The additive prior uses **structural descriptors** (bond/ring/radical counts) as input features and adds **fixed analytical statistical-mechanical terms** (Sackur-Tetrode translational entropy, rigid-rotor rotational entropy, symmetry and spin contributions) for the entropy target. These are physically motivated input features, not learned physical laws embedded in the network weights. The NASA-7 output layer and high-temperature completion enforce **thermodynamic constraints** (continuity of Cp, H, S and the classical Cp limit) by construction. When citing this work, please use these precise terms rather than "physics embedding."

## Archive

`archive/tebnet-v2/` contains the earlier model generation kept solely for provenance. It is not maintained, not compatible with the current featuriser version, and should not be used for new experiments or comparisons.

## License and citation

If you use TEB-Net or the benchmark infrastructure in your own work, please cite the manuscript (reference forthcoming). The code is released for research use; see individual upstream repositories for their respective licenses.