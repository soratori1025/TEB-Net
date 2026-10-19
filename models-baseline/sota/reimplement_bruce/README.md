# Re-implementation: Bruce et al. (2025) Thermochemical Property Predictions

This folder contains a complete, 100% self-contained re-implementation of the paper:
> **Machine learning predictions of thermochemical properties for aliphatic carbon and oxygen species**  
> *Bruce et al. (2025)*  
> Benchmark dataset: **WUDILY-CHO** (6,074 aliphatic C, H, O molecules).

---

## 📁 Directory Structure

```
reimplement_bruce/
├── dataset.py                         # WUDILY-CHO Excel dataset loader & 10-Fold CV generator
├── featurizers/                       # Molecular representation engines
│   ├── cds.py                         # Custom Descriptor Set (104 bits: 34 domain-knowledge + 70 E-State/topological)
│   ├── ecfp.py                        # Extended Connectivity Fingerprints (ECFP4 - 1024 bits)
│   ├── mordred_feat.py                # Mordred 2D descriptors (error filtering & corr > 0.95 removal)
│   ├── cm.py                          # Coulomb Matrix (electrostatic 2D/3D matrix representation)
│   ├── soap_mbtr.py                   # SOAP / MBTR & RDKit 2D descriptor extractors
│   └── graph_builder.py               # PyTorch Geometric molecular graph builder (GNF representations)
├── models/                            # Machine learning and Deep Learning models
│   ├── traditional.py                 # Random Forest (RF), XGBoost (XGB), SVR, Decision Tree (DT), Extra Trees (EXT)
│   ├── ann.py                         # CDS-ANN-KR (MLP: 104 -> 256 -> 128 -> 1 with ReLU)
│   └── gnn.py                         # PyTorch Geometric GNF-MPNN (depth=3, hidden=256, edge_dim=10), GNF-GCN, GNF-GAT
├── utils/                             # Metrics and feature/target scaling utilities
│   ├── metrics.py                     # Evaluates MAE, RMSE, R2 (Train & Test)
│   └── scaler.py                      # MinMax feature scaler [0, 1] & target standardizer/normalizer
├── run_single_model.py                # CLI script to evaluate any model on 10-Fold Cross Validation
├── benchmark_table_3.py               # Master benchmark runner for all 16+ model combinations
└── output/                            # Output directory storing CSV and Markdown benchmark comparison tables
```

---

## 🚀 Quick Start Guide

### 1. Run 10-Fold Cross-Validation for a Single Model
You can run any specific model (e.g. `CDS-RF`, `MRD-RF`, `GNF-MPNN`, `CM-SVR`) using `run_single_model.py`:

```bash
# Evaluate CDS-RF with 10-Fold CV
python run_single_model.py --model CDS-RF --folds 10

# Evaluate GNF-MPNN with 10-Fold CV (400 epochs)
python run_single_model.py --model GNF-MPNN --folds 10 --epochs 400

# Evaluate CM-SVR with 10-Fold CV
python run_single_model.py --model CM-SVR --folds 10
```

### 2. Run the Complete Benchmark (Table 3 Re-implementation)
To evaluate all models in parallel/sequence and export the final comparative table matching Table 3 of Bruce et al. (2025):

```bash
python benchmark_table_3.py
```

The output will be automatically saved to:
- `output/benchmark_table_3.csv`
- `output/benchmark_table_3.md`

---

## 📊 Standard Protocol & Target Specifications

- **Dataset**: WUDILY-CHO (6,074 aliphatic C, H, O species).
- **Target Property**: Standard Enthalpy of Formation ($\Delta H^\circ_{f,298}$) in `kcal/mol`.
- **Evaluation Protocol**: 10-Fold Cross-Validation (`n_splits = 10`, `seed = 42`).
- **Metrics Reported**: `Train_MAE`, `Test_MAE`, `Train_RMSE`, `Test_RMSE`, `Train_R2`, `Test_R2`.
- **Feature Scaling**: Min-Max Scaling to $[0, 1]$.
- **Target Scaling**: Standardized prior to MSE loss computation for neural models (ANN, MPNN, GCN, GAT) and inverse-transformed for evaluation metrics.
