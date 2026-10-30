# CH2 extrapolation benchmark (manuscript Figure 2, Table 1, SI Table S13)

Every baseline of the manuscript's Table 1 was rebuilt on **DeepTherm random split 0** (4047/449/501
species) and evaluated on the **five homologous series up to C16** (78 species, 45 of them outside the
DeepTherm set), exactly like TEB-Net. Nothing is retuned: the models are recreated from their stored
runs so that the comparison isolates extrapolation behaviour.

## Models (13 rows of Figure 2 / Table 1 / Table S13)

| family | models | source |
|---|---|---|
| this work | TEB-Net, ten-member trimmed mean; TEB-Net, member 0 | `models/tebnet-v3/` checkpoints (random0) |
| published | DeepTherm (pretrained) | published Supplementary Data 4 |
| retrained GNNs | GIN, GCN, GAT, AttentiveFP | `models-baseline/benchmark_results_baselines/<tag>/random0/member0_seed0/best.pt` |
| retrained GNN | D-MPNN (chemprop v2, norm aggregation, no pretraining) | `models-baseline/sota/chemprop_repo/chemprop_dmpnn_random/replicate_0/model_0/best.pt` |
| retrained ML | random forest (physics), random forest (ECFP), XGBoost (physics), XGBoost (physics+ECFP), ridge (prior descriptors) | `best_params` of the same random0 runs, refit |

## How it was run

```bash
# 1) predictions on the 78 homologous species, one CSV per model
python models-baseline/traditional_models/ch2_extrapolation.py --list
python models-baseline/traditional_models/ch2_extrapolation.py --only ml-ridge_prior ml-rf_physics ml-rf_ecfp-r2b1024
python models-baseline/traditional_models/ch2_extrapolation.py --only ml-xgb_physics ml-xgb_physics+ecfp-r2b1024 --n_jobs 16
python models-baseline/traditional_models/ch2_extrapolation.py --only gnn-gin_all_L4_d300_sum gnn-gcn_all_L4_d300_sum gnn-gat_all_L4_d300_sum gnn-attentivefp_all_L4_d300_attn
python models-baseline/traditional_models/ch2_extrapolation_dmpnn.py         # chemprop CLI

# 2) score every model: increment per CH2, MAE from the extrapolated reference line, C16 error
python models-baseline/traditional_models/ch2_extrapolation_summary.py

# 3) manuscript figure and SI table
python tebnet_manuscript/01_Manuscript/figures/plot_ch2_benchmark.py
python tebnet_manuscript/03_SI_source/make_table_S13_ch2.py
python tebnet_manuscript/01_Manuscript/figures/plot_physical_evidence.py    # Figures 3, 4 and SI Fig. S2
```

## Verification

`verify_<tag>.json` records the maximum absolute difference between the re-predicted values and the
`predictions.npz` stored by the original runs on the common 501-species test set:

* ML refits (ridge / random forest / XGBoost): `<= 6e-10` J/mol (exact bit-wise reproduction besides float noise);
* GNN checkpoints (CPU inference of GPU-trained float32 weights): `<= 0.27` J/mol (~7e-5 kcal/mol);
* D-MPNN: the chemprop CLI reproduces the repo's own test MAEs to three decimals (H 4.417, S 2.210, Cp300 1.113 kcal).

## Outputs (this directory)

* `preds_<tag>.csv` - per-model predictions on the 78 species (kcal/mol, cal/mol/K).
* `ch2_extrapolation_series.csv` - per model x property x series: reference slope, predicted increment,
  recovered fraction, MAE from the line, end-of-series error.
* `ch2_extrapolation_models.csv` - per model: increment ranges, mean MAE from the line, in-distribution test MAE.
* `ch2_extrapolation_stats.json` - Spearman rank correlation between in-distribution MAE and extrapolation error.
