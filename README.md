# TEB-Net — Thermodynamics-Embedded Baseline Network

Research code for the manuscript *"Thermodynamics-Embedded Baseline Network (TEB-Net) Extrapolates
Molecular Thermochemistry to Fuel-Sized Chains and Flame Temperatures"*.

TEB-Net predicts the gas-phase **ΔfH°298**, **S°298** and **Cp(T)** of C/H/O species from a SMILES
string and returns them as two-range **NASA-7** polynomials. The prediction is the sum of two parts:

* a **frozen additive prior** built from molecular bond, ring and radical-centre counts, plus fixed
  analytical statistical-mechanical entropy terms (explicitly defined descriptors, no learned parameters
  in this component), and
* a **D-MPNN residual** that learns property-specific corrections to the prior.

An analytic output layer converts the predicted anchors into NASA-7 coefficients, and a high-temperature
completion keeps the exported heat capacity monotonic and below its classical limit up to 3000 K.

## What is in this repository

| path | contents |
|---|---|
| `models/tebnet-v3/tebnet/` | the model package (prior, featuriser, D-MPNN, NASA-7 layer, completion) |
| `models/tebnet-v3/scripts/` | training (`train.py`), cross-validation (`train_cv.py`, `train_wudily_cv.py`), evaluation (`deeptherm_eval.py`), descriptor cache (`compute_physics_features.py`), SMILES prediction (`predict_smiles.py`) |
| `models/tebnet-v3/*.csv` | the DeepTherm dataset and its precomputed caches (see *Data* below) |
| `data/` | other datasets and splits used by the study (Ureel, WUDILY-CHO, RMG references) |
| `archive/tebnet-v2/` | archived earlier model generation (kept for provenance, not maintained) |

## Environment

Python 3.12 with `torch`, `torch_geometric`, `rdkit`, `lightning`, `numpy`, `pandas`,
`scikit-learn`, `xgboost` and `chemprop` (v2).

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install torch torch_geometric rdkit lightning numpy pandas scikit-learn xgboost chemprop
```

## Quick start

All commands below are run **from the repository root**. The `tebnet` package lives in
`models/tebnet-v3/`; the training scripts add it to `sys.path` themselves, so no `PYTHONPATH`
export is required.

```bash
# 1) train one model on DeepTherm random split 0
python models/tebnet-v3/scripts/train.py --seed 0 --split_seed 0

# 2) summarise the run (single-model and ensemble MAE)
python models/tebnet-v3/scripts/deeptherm_eval.py --runs models/tebnet-v3/benchmark_results_deeptherm

# 3) predict NASA-7 coefficients for new SMILES
python models/tebnet-v3/scripts/predict_smiles.py \
    --smiles_csv my_species.csv \
    --h_runs   models/tebnet-v3/benchmark_results_deeptherm/*/random0/member0_seed0 \
    --out nasa7_predictions.csv
```

`train.py` writes `results.json` and predictions into `models/tebnet-v3/benchmark_results_deeptherm/`
(override with `--out_dir`). Checkpoints are not distributed with the repository; every run writes
its own.

### Tuning and variants

| flag | meaning |
|---|---|
| `--split_type {random,complexity,scaffold,functional_group}` | DeepTherm split protocol |
| `--split_seed N` | seed of the split (split 0 is the manuscript default) |
| `--member M --seed M` | deep-ensemble member sharing one test set |
| `--csv PATH` | dataset; defaults to `models/tebnet-v3/deeptherm_data.csv` |
| `--physics_cache PATH` | V15 descriptor cache; `auto` (default) resolves to `<csv stem>.physics_v15.csv` |
| `--max_epochs`, `--batch_size`, `--gpu` | standard training controls |

## Reproducing the manuscript

| manuscript item | command |
|---|---|
| Table 1 / initial benchmarks | `train.py --seed 0 --split_seed {0..4}` then `deeptherm_eval.py` |
| Complexity split | `train.py --split_type complexity` |
| Functional-group splits | `train.py --split_type functional_group --test_group "Ether group"` |
| Homologous-series extrapolation | `predict_smiles.py` with a `series, n_carbon, smiles` CSV |
| High-temperature validation | requires the trained ensemble above; see the SI for the protocol |
| Ureel and WUDILY-CHO cross-validation | `train_cv.py`, `train_wudily_cv.py` |

Baseline models (ridge, random forest, XGBoost, GIN/GCN/GAT/AttentiveFP, chemprop D-MPNN) and the
CH₂ extrapolation benchmark are **not part of this public repository**. They are kept in a private
results tree; contact the authors if you need them for a direct comparison.

## Using TEB-Net as a baseline

1. **Load the package.** Add `models/tebnet-v3` to `sys.path`, then import from `tebnet`:
   ```python
   import sys; sys.path.insert(0, "models/tebnet-v3")
   from tebnet.models.tebnet import TEBNetV20
   from tebnet.data.featurizer import ThermoAtomFeaturizerV3, FEATURES_VERSION
   ```
2. **Use the published splits.** `tebnet.data.splits.deeptherm_split` reproduces the DeepTherm
   protocol exactly (81/9/10 random, complexity, scaffold and functional-group splits). The same CSV
   row order and seed give the same test molecules as the DeepTherm code.
3. **Match the training protocol.** Defaults in `scripts/train.py`: Huber loss, AdamW
   (lr 8e-4, weight decay 5e-5), 100 epochs, batch size 32, separate enthalpy and S/Cp models, no
   early stopping.
4. **Report the version.** Current model is `TEBNetV20` with `FEATURES_VERSION = "v15.3"`
   (`models/tebnet-v3/tebnet/data/featurizer.py`). State it in your paper so numbers are comparable.
5. **Reuse the unit conventions.** All internal predictions are SI (J/mol, J mol⁻¹ K⁻¹); DeepTherm
   units are kcal/mol and cal mol⁻¹ K⁻¹.

## Data

| dataset | species | location | notes |
|---|---|---|---|
| DeepTherm | 4,997 C/H/O | `models/tebnet-v3/deeptherm_data.csv` | same row order as the published Supplementary Data |
| DeepTherm descriptors | — | `models/tebnet-v3/deeptherm_data.physics_v15.csv` | V15 cache, 71 columns (4,997 rows) |
| DeepTherm entropy cache | — | `models/tebnet-v3/deeptherm_data.entropy_v2.csv` | conformer-derived entropy terms |
| RMG Cp references | 666 used (619 qm + 47 evaluated) | `models/tebnet-v3/rmg_cp_reference.csv` | 1,110 raw rows; the manuscript subset keeps rank-0 rows whose SMILES round-trips and whose multiplicity is ≤ 2 |
| Ureel (hydrocarbons, radicals, carbenium) | 330 / 426 / 162 | `data/splits/` | CBS-QB3 |
| WUDILY-CHO | 6,074 rows in `wudily-cho.xlsx` | `data/WUDILY-CHO/` | InChIKey de-duplication is not reproduced by a script in this repo |

The descriptor cache is rebuilt with:

```bash
python models/tebnet-v3/scripts/compute_physics_features.py --csv models/tebnet-v3/deeptherm_data.csv
```

Upstream repositories (chemprop, DeepTherm, HC-Thermochemistry, RMG-database) are **not** redistributed
here; install them from their own sources.

## Terminology note

The additive prior uses **structural descriptors** (bond, ring and radical-centre counts) as input
features and adds **fixed analytical statistical-mechanical terms** (Sackur-Tetrode translational
entropy, rigid-rotor rotational entropy, symmetry and spin contributions) for the entropy target.
These are physically motivated input features, not learned physical laws embedded in the network
weights. The NASA-7 output layer and high-temperature completion enforce **thermodynamic constraints**
(continuity of Cp, H and S, and the classical Cp limit) by construction. Please use these terms rather
than "physics embedding" when describing the method.

## Notes

* Trained checkpoints (~2.5 GB) are not stored in the repository.
* `archive/tebnet-v2/` holds the earlier model generation for provenance only. It is not maintained,
  is not compatible with the current featuriser, and should not be used for new experiments.

## Citation

If you use TEB-Net in your work, please cite the manuscript (reference forthcoming).
