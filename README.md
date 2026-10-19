# TEB-Net — Thermodynamics-Enhanced Bond Network

Research code, benchmark harness and evaluation artefacts for **TEB-Net**, a
thermochemistry model built on a physics-constrained bond-network backbone with
Chebyshev polynomial expansions of temperature-dependent thermodynamic
quantities.

## Repository layout

```
teb-net-model/
├── framework/                 # git submodule -> soratori1025/Framework-Chemistry
│                              #   general chemistry-ML framework: data loaders,
│                              #   descriptor cache, benchmark suite, evidence routing
├── models/
│   ├── tebnet-v2/             # archived model generation (manuscript numbers)
│   └── tebnet-v3/             # current generation: training, benchmarks, check/
│       ├── tebnet/            # model + training package
│       ├── check/             # evaluation & diagnostic scripts (homolog tables,
│       │                      #   ablations, archive builder)
│       └── benchmark_results_deeptherm/   # per-run results.json + predictions
├── models-baseline/
│   ├── results/               # ML / GNN baseline runs (AttentiveFP, RF, ...)
│   ├── benchmark_results_baselines/
│   ├── traditional_models/
│   └── sota_patches/          # OUR local modifications to upstream SOTA repos
│                              #   (chemprop, DeepTherm, HC-Thermochemistry)
├── data/                      # datasets used by the study
├── figures/                   # manuscript figures
└── ARCHIVE_README.md          # what the 2.3 GB snapshot archive contains
```

## Not tracked here

| Excluded | Size | How to obtain |
|---|---|---|
| `.venv/` | 6.8 GB | `python -m venv .venv && pip install -e framework && pip install -e models/tebnet-v3` |
| `*.ckpt`, `*.pt`, `*.npz` | ~2.5 GB | full snapshot: `tebnet_research_2026-10-19.zip` (see `ARCHIVE_README.md`) |
| `models-baseline/sota/{chemprop_repo,DeepTherm,HC-Thermochemistry}/` | 1.2 GB | clone upstream; our diffs live in `models-baseline/sota_patches/` |
| `models/tebnet-v3/RMG-database/` | 43 MB | `git clone https://github.com/ReactionMechanismGenerator/RMG-database` |

`framework/` is a submodule, not a copy: clone with
`git clone --recurse-submodules <this-repo>` (or `git submodule update --init`).

## Reproducing the main results

```bash
# environment
python -m venv .venv && source .venv/bin/activate
pip install -e framework
pip install -e models/tebnet-v3    # if no pyproject.toml: add models/tebnet-v3 to PYTHONPATH

# DeepTherm benchmark (thermodynamic targets, grouped split)
cd models/tebnet-v3
python scripts/train.py --seed 0 --member 0 --split_seed 0 \
    --csv deeptherm_data.physics_v15.csv --s_prior v2_exact
# full multi-seed sweep: python orchestrate_v3.py

# homologous-series extrapolation table
python check/homologous_controls.py --series <predictions>.csv

# regression tests (from models/tebnet-v3)
pytest ../../framework/tests tests -q
```

> **First-run note.** `framework/tests/test_datasets.py` needs the MoleculeNet
> files, which are not tracked (they are large and re-downloadable). Run
> `python framework/benchmarks/prepare_moleculenet_data.py` once, or skip that
> module: `pytest framework/tests --ignore=framework/tests/test_datasets.py`
> (79 tests pass without it).

## Key result: CH2-homolog regulariser (lambda sweep)

Member 0, seed 0, split seed 0, 100 epochs, no early stop:

| run | best epoch | val DeepTherm MAE (9 targets) | vs baseline |
|---|---|---|---|
| baseline `sp-v2_exact` | 84 | **0.9216** | — |
| `hom` lambda = 1e-3 | 92 | 0.9991 | +8.4 % worse |
| `hom` lambda = 3e-3 | 92 | 1.0325 | +12.0 % worse |

The CH2-homolog consistency penalty **degrades** aggregate validation accuracy:
it buys consistency between homologs at the cost of absolute error. Full
discussion, including the timestamp caveat on the manuscript's homologous table,
is in `models/tebnet-v3/check/HOMOLOG_RETRAIN_NOTES.md`.

## Citation

Manuscript in preparation.
