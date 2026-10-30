#!/usr/bin/env python3
"""
Residual-vs-Baseline Diagnostic for TEB-Net v14.4
==================================================

Computes three headline numbers on the TEST set to pinpoint where the accuracy
leverage lies, plus a correlation analysis of residual errors vs structural
descriptors.

Outputs (per seed, then averaged):
  1. MAE_baseline_only  = MAE(baseline(x), truth)
     → How good is the OLS atom-count baseline alone?
  2. MAE_final          = MAE(baseline + residual_net, truth)
     → The full model's error.
  3. MAE_residual_target = MAE(residual_net, truth − baseline)
     → Identical to #2 by construction (sanity check).
  4. Residual correlation with structural features:
     - Spearman ρ of (truth − baseline) vs {n_rings, unsaturation, heavy count,
       n_rotatable, aromaticity fraction, ...}
     Strong ρ = easy signal the residual net (or richer features) could exploit.

Key change vs draft version:
  - Baseline weights are READ FROM THE CHECKPOINT (frozen model buffers), NOT
    re-fit.  This guarantees the diagnostic exactly matches what the model saw.

Interpretation:
  • If MAE_final ≪ MAE_baseline_only → network is already doing heavy lifting.
    Next move: enrich the NETWORK (features, capacity, data).
  • If MAE_final ≈ MAE_baseline_only → network barely contributes. Bug or
    capacity problem on the residual side.
  • Correlation analysis tells you which structural features still leak into the
    residual: high |ρ| = low-hanging fruit for a richer baseline (group additivity)
    or better global descriptors.

Usage:
  python diagnose_residual.py                           # all 5 seeds
  python diagnose_residual.py --seeds 42                # single seed
  python diagnose_residual.py --seeds 42 123 --gpu 1    # specific seeds
  python diagnose_residual.py --plot                    # also save scatter PNGs
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
SCRIPT_DIR = Path(__file__).resolve().parent
TEBNET_V144_ROOT = SCRIPT_DIR.parent / "tebnet-v14.4"
TEBNET_MODEL_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(TEBNET_V144_ROOT))
sys.path.insert(0, str(TEBNET_MODEL_ROOT))
sys.path.insert(0, str(TEBNET_MODEL_ROOT / "data"))
from tebnet.data.dataset import ThermochemDataset
from tebnet.data.collate import ThermoCollater
from tebnet.models.tebnet import TEBNetV144 as TEBNet
from tebnet.data.featurizer import (
    CleanBondFeaturizer,
    COMP_DIM,
    AUX_DIM_MINIMAL,
    compute_composition_vector,
)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose")
KCAL_TO_J = 4184.0
CAL_TO_J = 4.184
J_TO_KCAL = 1.0 / KCAL_TO_J
J_TO_CAL = 1.0 / CAL_TO_J
T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
def load_wudily_dataset(csv_path: str, t_grid: np.ndarray) -> list[dict]:
    """Load WUDILY-CHO CSV (kcal/cal units) → TEB-Net records (SI: J)."""
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        dh_f_298 = float(row["Hf_298_kcal_mol"]) * KCAL_TO_J
        s_298 = float(row["S_298_cal_molK"]) * CAL_TO_J
        cp_values = []
        for T in t_grid:
            col = f"Cp_{T:.0f}_cal_molK"
            cp_values.append(float(row[col]) * CAL_TO_J)
        records.append({
            "smiles": smiles, "Cp": np.array(cp_values, dtype=np.float32),
            "dH_f_298": dh_f_298, "S_298": s_298, "weight": 1.0, "name": smiles,
        })
    return records
def compute_structural_descriptors(smiles: str) -> dict:
    """Cheap 2D descriptors that might correlate with the residual."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mol_h = Chem.AddHs(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    n_atoms = mol_h.GetNumAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_C = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_H_total = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    dou = max(0, (2 * n_C + 2 - n_H_total)) / 2.0
    n_sp2 = sum(1 for a in mol.GetAtoms()
                if a.GetHybridization() == Chem.rdchem.HybridizationType.SP2)
    frac_sp2 = n_sp2 / max(mol.GetNumAtoms(), 1)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    return {
        "n_heavy": n_heavy,
        "n_atoms": n_atoms,
        "n_rings": n_rings,
        "n_aromatic_rings": n_arom_rings,
        "n_rotatable": n_rot,
        "degree_of_unsaturation": dou,
        "frac_sp2": frac_sp2,
        "n_radical_electrons": n_radical,
        "mol_weight": mw,
    }
def diagnose_one_seed(
    seed: int, csv_path: str, split_dir: str, results_dir: str,
    t_grid: np.ndarray, device: torch.device,
) -> dict:
    """Run the 3-number diagnostic + correlation on one seed.

    Baseline weights are extracted FROM the checkpoint, not re-fit.
    """
    all_records = load_wudily_dataset(csv_path, t_grid)
    split_file = Path(split_dir) / f"split_seed_{seed}.json"
    with open(split_file) as f:
        split = json.load(f)
    test_idx = split["test_indices"]
    test_records = [all_records[i] for i in test_idx]
    ds_kwargs = dict(T_grid=t_grid, aux_feature_set="routed", compute_comp=True)
    test_ds = ThermochemDataset(test_records, **ds_kwargs)
    ckpt_path = Path(results_dir) / f"seed{seed}" / f"init{seed}" / "best.ckpt"
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return {}
    model = TEBNet.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device).eval()
    comp_weight = model.comp_weight.cpu().numpy().astype(np.float64)
    comp_bias = float(model.comp_bias.cpu().item())
    s_comp_weight = model.s_comp_weight.cpu().numpy().astype(np.float64)
    s_comp_bias = float(model.s_comp_bias.cpu().item())
    logger.info(
        f"  Loaded baseline from checkpoint: "
        f"H bias={comp_bias*J_TO_KCAL:.2f} kcal, "
        f"S bias={s_comp_bias*J_TO_CAL:.2f} cal"
    )
    test_comp = np.stack([dp.comp for dp in test_ds.datapoints]).astype(np.float64)
    test_h_true = np.array([dp.dh_f_298 for dp in test_ds.datapoints], dtype=np.float64)
    test_s_true = np.array([dp.s_298 for dp in test_ds.datapoints], dtype=np.float64)
    test_smiles = [dp.smiles for dp in test_ds.datapoints]
    h_baseline_pred = test_comp @ comp_weight + comp_bias
    s_baseline_pred = test_comp @ s_comp_weight + s_comp_bias
    mae_h_baseline = float(np.mean(np.abs(h_baseline_pred - test_h_true))) * J_TO_KCAL
    mae_s_baseline = float(np.mean(np.abs(s_baseline_pred - test_s_true))) * J_TO_CAL
    collater = ThermoCollater()
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collater, num_workers=0, pin_memory=False,
    )
    all_h_pred, all_s_pred, all_cp_pred, all_cp_true = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            all_h_pred.append(preds["dh_f_298"].cpu().numpy())
            all_s_pred.append(preds["s_298"].cpu().numpy())
            all_cp_pred.append(preds["cp"].cpu().numpy())
            all_cp_true.append(batch.cp_grid.cpu().numpy())
    h_model_pred = np.concatenate(all_h_pred)
    s_model_pred = np.concatenate(all_s_pred)
    cp_model_pred = np.concatenate(all_cp_pred)
    cp_true = np.concatenate(all_cp_true)
    mae_h_final = float(np.mean(np.abs(h_model_pred - test_h_true))) * J_TO_KCAL
    mae_s_final = float(np.mean(np.abs(s_model_pred - test_s_true))) * J_TO_CAL
    cp_mae_per_T = []
    for i, T in enumerate(t_grid):
        mask = np.isfinite(cp_true[:, i])
        if mask.any():
            cp_mae_per_T.append(
                float(np.mean(np.abs(cp_model_pred[mask, i] - cp_true[mask, i]))) * J_TO_CAL
            )
    mae_cp_avg = float(np.mean(cp_mae_per_T)) if cp_mae_per_T else float("nan")
    h_residual_target = (test_h_true - h_baseline_pred) * J_TO_KCAL
    s_residual_target = (test_s_true - s_baseline_pred) * J_TO_CAL
    h_residual_pred = (h_model_pred - h_baseline_pred) * J_TO_KCAL
    s_residual_pred = (s_model_pred - s_baseline_pred) * J_TO_CAL
    mae_h_residual = float(np.mean(np.abs(h_residual_pred - h_residual_target)))
    mae_s_residual = float(np.mean(np.abs(s_residual_pred - s_residual_target)))
    if abs(mae_h_residual - mae_h_final) > 0.01:
        logger.warning(
            f"  Sanity FAILED: MAE_residual({mae_h_residual:.4f}) ≠ "
            f"MAE_final({mae_h_final:.4f}) — check unit conversion"
        )
    h_errors = np.abs(h_model_pred - test_h_true) * J_TO_KCAL
    worst_h_idx = np.argsort(h_errors)[-10:][::-1]
    outliers_h = [
        {
            "smiles": test_smiles[i],
            "error_kcal": round(float(h_errors[i]), 3),
            "true_kcal": round(float(test_h_true[i]) * J_TO_KCAL, 2),
            "pred_kcal": round(float(h_model_pred[i]) * J_TO_KCAL, 2),
            "baseline_kcal": round(float(h_baseline_pred[i]) * J_TO_KCAL, 2),
            "residual_target_kcal": round(float(h_residual_target[i]), 3),
        }
        for i in worst_h_idx
    ]
    from scipy.stats import spearmanr
    h_final_error = test_h_true * J_TO_KCAL - h_model_pred * J_TO_KCAL
    s_final_error = test_s_true * J_TO_CAL - s_model_pred * J_TO_CAL
    desc_list = [compute_structural_descriptors(s) for s in test_smiles]
    valid_mask = [bool(d) for d in desc_list]
    if not all(valid_mask):
        logger.warning(f"  {sum(not v for v in valid_mask)} molecules had invalid SMILES for descriptors")
    desc_keys = list(desc_list[0].keys()) if desc_list and desc_list[0] else []
    desc_df = pd.DataFrame(desc_list)
    correlations_h = {}
    correlations_s = {}
    for key in desc_keys:
        vals = desc_df[key].values.astype(float)
        if np.std(vals) < 1e-12:
            continue
        rho_h, p_h = spearmanr(h_final_error, vals)
        rho_s, p_s = spearmanr(s_final_error, vals)
        correlations_h[key] = {"rho": float(rho_h), "p": float(p_h)}
        correlations_s[key] = {"rho": float(rho_s), "p": float(p_s)}
    result = {
        "seed": seed,
        "n_test": len(test_idx),
        "Hf298_kcal_mol": {
            "MAE_baseline_only": round(mae_h_baseline, 4),
            "MAE_final": round(mae_h_final, 4),
            "MAE_residual_sanity": round(mae_h_residual, 4),
            "baseline_reduction_pct": round(
                (1 - mae_h_final / max(mae_h_baseline, 1e-9)) * 100, 1
            ),
            "residual_std_kcal": round(float(np.std(h_residual_target)), 4),
            "residual_mean_kcal": round(float(np.mean(h_residual_target)), 4),
        },
        "S298_cal_molK": {
            "MAE_baseline_only": round(mae_s_baseline, 4),
            "MAE_final": round(mae_s_final, 4),
            "MAE_residual_sanity": round(mae_s_residual, 4),
            "baseline_reduction_pct": round(
                (1 - mae_s_final / max(mae_s_baseline, 1e-9)) * 100, 1
            ),
            "residual_std_cal": round(float(np.std(s_residual_target)), 4),
            "residual_mean_cal": round(float(np.mean(s_residual_target)), 4),
        },
        "Cp_avg_cal_molK": {
            "MAE_final": round(mae_cp_avg, 4),
        },
        "outliers_h_top10": outliers_h,
        "correlations_h_residual": {
            k: {"rho": round(v["rho"], 4), "p_value": f"{v['p']:.2e}"}
            for k, v in sorted(correlations_h.items(), key=lambda x: -abs(x[1]["rho"]))
        },
        "correlations_s_residual": {
            k: {"rho": round(v["rho"], 4), "p_value": f"{v['p']:.2e}"}
            for k, v in sorted(correlations_s.items(), key=lambda x: -abs(x[1]["rho"]))
        },
        "_h_residual_target": h_residual_target.tolist(),
        "_h_errors": h_errors.tolist(),
        "_smiles": test_smiles,
    }
    return result
def save_scatter_plots(all_results: list[dict], out_dir: Path):
    """Save diagnostic scatter plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping scatter plots")
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Residual Diagnostic — across seeds", fontsize=14, fontweight="bold")
    seeds = [r["seed"] for r in all_results]
    h_base = [r["Hf298_kcal_mol"]["MAE_baseline_only"] for r in all_results]
    h_final = [r["Hf298_kcal_mol"]["MAE_final"] for r in all_results]
    s_base = [r["S298_cal_molK"]["MAE_baseline_only"] for r in all_results]
    s_final = [r["S298_cal_molK"]["MAE_final"] for r in all_results]
    ax = axes[0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, h_base, w, label="Baseline only", alpha=0.8, color="#e74c3c")
    ax.bar(x + w/2, h_final, w, label="Full model", alpha=0.8, color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("ΔHf(298) — Baseline vs Full")
    ax.legend(fontsize=8)
    for i, (b, f) in enumerate(zip(h_base, h_final)):
        pct = (1 - f/max(b, 1e-9)) * 100
        ax.text(i, max(b, f) + 0.1, f"-{pct:.0f}%", ha="center", fontsize=8)
    ax = axes[1]
    r0 = all_results[0]
    h_resid = np.array(r0["_h_residual_target"])
    ax.hist(h_resid, bins=50, alpha=0.75, color="#3498db", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual target: truth − baseline (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title(f"ΔHf residual distribution (seed {r0['seed']})")
    ax.text(0.02, 0.95, f"μ = {np.mean(h_resid):.2f}\nσ = {np.std(h_resid):.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax = axes[2]
    desc_list = [compute_structural_descriptors(s) for s in r0["_smiles"]]
    n_heavy = np.array([d.get("n_heavy", 0) for d in desc_list])
    errors = np.array(r0["_h_errors"])
    ax.scatter(n_heavy, errors, alpha=0.3, s=8, color="#9b59b6")
    ax.set_xlabel("n_heavy atoms")
    ax.set_ylabel("|ΔHf error| (kcal/mol)")
    ax.set_title("Error vs molecule size")
    if len(n_heavy) > 10:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(n_heavy, errors)
        z = np.polyfit(n_heavy, errors, 1)
        x_line = np.linspace(n_heavy.min(), n_heavy.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7,
                label=f"ρ = {rho:.3f}")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = out_dir / "residual_diagnostic_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Scatter plots saved to: {plot_path}")
def main():
    parser = argparse.ArgumentParser(
        description="Residual-vs-Baseline diagnostic for TEB-Net v14.4"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 2024, 8888, 9999],
        help="Split seeds to diagnose"
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(SCRIPT_DIR / "data" / "wudily_cho_processed.csv"),
    )
    parser.add_argument(
        "--split_dir", type=str,
        default=str(SCRIPT_DIR / "data" / "splits"),
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(SCRIPT_DIR / "results_v144"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="Save scatter plots")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    all_results = []
    for seed in args.seeds:
        logger.info(f"─── Diagnosing seed {seed} ───")
        res = diagnose_one_seed(
            seed, args.csv, args.split_dir, args.results_dir, T_GRID, device,
        )
        if res:
            all_results.append(res)
    if not all_results:
        logger.error("No results produced. Check checkpoint paths.")
        return
    agg = {
        "H_baseline": np.mean([r["Hf298_kcal_mol"]["MAE_baseline_only"] for r in all_results]),
        "H_final": np.mean([r["Hf298_kcal_mol"]["MAE_final"] for r in all_results]),
        "H_reduction": np.mean([r["Hf298_kcal_mol"]["baseline_reduction_pct"] for r in all_results]),
        "S_baseline": np.mean([r["S298_cal_molK"]["MAE_baseline_only"] for r in all_results]),
        "S_final": np.mean([r["S298_cal_molK"]["MAE_final"] for r in all_results]),
        "S_reduction": np.mean([r["S298_cal_molK"]["baseline_reduction_pct"] for r in all_results]),
        "Cp_final": np.mean([r["Cp_avg_cal_molK"]["MAE_final"] for r in all_results]),
    }
    h_corr_agg = defaultdict(list)
    s_corr_agg = defaultdict(list)
    for r in all_results:
        for k, v in r["correlations_h_residual"].items():
            h_corr_agg[k].append(abs(v["rho"]))
        for k, v in r["correlations_s_residual"].items():
            s_corr_agg[k].append(abs(v["rho"]))
    sep = "═" * 78
    print(f"\n{sep}")
    print("  TEB-Net v14.4 — RESIDUAL vs BASELINE DIAGNOSTIC")
    print(f"  Baseline weights: FROM CHECKPOINT (frozen model buffers)")
    print(f"{sep}")
    print(f"\n  {'':8s}  {'─── ΔHf(298) kcal/mol ───':>42s}  │  {'─── S°298 cal/(mol·K) ───':>38s}")
    print(f"  {'Seed':>8s}  {'Baseline':>10s}  {'Final':>8s}  {'Reduction':>10s}  {'ResidStd':>9s}  │  {'Baseline':>10s}  {'Final':>8s}  {'Reduction':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*9}  │  {'─'*10}  {'─'*8}  {'─'*10}")
    for r in all_results:
        h = r["Hf298_kcal_mol"]
        s = r["S298_cal_molK"]
        print(
            f"  {r['seed']:>8d}  {h['MAE_baseline_only']:>10.3f}  {h['MAE_final']:>8.3f}"
            f"  {h['baseline_reduction_pct']:>9.1f}%  {h['residual_std_kcal']:>9.3f}"
            f"  │  {s['MAE_baseline_only']:>10.3f}  {s['MAE_final']:>8.3f}"
            f"  {s['baseline_reduction_pct']:>9.1f}%"
        )
    print(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*9}  │  {'─'*10}  {'─'*8}  {'─'*10}")
    print(
        f"  {'MEAN':>8s}  {agg['H_baseline']:>10.3f}  {agg['H_final']:>8.3f}"
        f"  {agg['H_reduction']:>9.1f}%  {'':>9s}"
        f"  │  {agg['S_baseline']:>10.3f}  {agg['S_final']:>8.3f}"
        f"  {agg['S_reduction']:>9.1f}%"
    )
    print(f"\n  Cp avg MAE (model, all seeds): {agg['Cp_final']:.3f} cal/(mol·K)")
    print(f"\n{'─'*78}")
    print("  INTERPRETATION:")
    h_ratio = agg["H_final"] / max(agg["H_baseline"], 1e-9)
    s_ratio = agg["S_final"] / max(agg["S_baseline"], 1e-9)
    if h_ratio < 0.5:
        print(f"  ΔHf: Network reduces baseline error by >{(1-h_ratio)*100:.0f}%.")
        print(f"       → The D-MPNN is ALREADY doing heavy lifting.")
        print(f"       → Biggest leverage: RICHER FEATURES / CAPACITY for the residual net.")
    elif h_ratio < 0.8:
        print(f"  ΔHf: Network reduces baseline error by ~{(1-h_ratio)*100:.0f}%.")
        print(f"       → Moderate improvement. Both baseline & features upgrades worthwhile.")
    else:
        print(f"  ΔHf: Network barely reduces baseline (ratio = {h_ratio:.2f}).")
        print(f"       → Problem is on the NETWORK side — capacity, features, or training issue.")
    if s_ratio < 0.5:
        print(f"  S°:  Network reduces baseline error by >{(1-s_ratio)*100:.0f}%.")
        print(f"       → Leverage: improve features/capacity for S-branch.")
    elif s_ratio < 0.8:
        print(f"  S°:  Network reduces baseline error by ~{(1-s_ratio)*100:.0f}%.")
        print(f"       → Both baseline & feature enrichment would help.")
    else:
        print(f"  S°:  Baseline-to-final ratio = {s_ratio:.2f}.")
        print(f"       → Network barely helps — capacity/feature issue on S-branch.")
    print(f"\n{'─'*78}")
    print("  RESIDUAL CORRELATION with structural features (mean |Spearman ρ| across seeds):")
    print(f"\n  {'Feature':>25s}  {'|ρ| vs ΔHf residual':>22s}  {'|ρ| vs S° residual':>22s}  {'Signal?'}")
    print(f"  {'─'*25}  {'─'*22}  {'─'*22}  {'─'*10}")
    all_keys = sorted(
        set(list(h_corr_agg.keys()) + list(s_corr_agg.keys())),
        key=lambda k: -(np.mean(h_corr_agg.get(k, [0])) + np.mean(s_corr_agg.get(k, [0]))),
    )
    for key in all_keys:
        rho_h = np.mean(h_corr_agg.get(key, [0]))
        rho_s = np.mean(s_corr_agg.get(key, [0]))
        signal = "★★★" if max(rho_h, rho_s) > 0.4 else "★★" if max(rho_h, rho_s) > 0.2 else "★" if max(rho_h, rho_s) > 0.1 else "−"
        print(f"  {key:>25s}  {rho_h:>22.3f}  {rho_s:>22.3f}  {signal}")
    print(f"\n  ★★★ = strong (>0.4)  ★★ = moderate (>0.2)  ★ = weak (>0.1)  − = negligible")
    print(f"\n{'─'*78}")
    print(f"  TOP-10 ΔHf OUTLIERS (seed {all_results[0]['seed']}):")
    print(f"  {'SMILES':>50s}  {'|Error|':>8s}  {'True':>8s}  {'Pred':>8s}  {'Baseline':>8s}  {'Resid':>8s}")
    print(f"  {'─'*50}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for o in all_results[0]["outliers_h_top10"]:
        smi = o["smiles"][:50]
        print(
            f"  {smi:>50s}  {o['error_kcal']:>8.2f}  {o['true_kcal']:>8.1f}"
            f"  {o['pred_kcal']:>8.1f}  {o['baseline_kcal']:>8.1f}"
            f"  {o['residual_target_kcal']:>8.2f}"
        )
    print(f"\n{'─'*78}")
    print("  High |ρ| ⇒ that descriptor still 'leaks' into the residual.")
    print("  → Can be captured by enriching the BASELINE (group additivity)")
    print("     or by feeding richer descriptors into the residual branches.")
    print(f"{sep}\n")
    out_path = SCRIPT_DIR / "residual_diagnostic.json"
    clean_results = []
    for r in all_results:
        cr = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(cr)
    with open(out_path, "w") as f:
        json.dump({
            "per_seed": clean_results,
            "aggregate": {k: round(v, 4) for k, v in agg.items()},
            "correlation_mean_abs_rho": {
                "h_residual": {k: round(np.mean(v), 4) for k, v in h_corr_agg.items()},
                "s_residual": {k: round(np.mean(v), 4) for k, v in s_corr_agg.items()},
            },
        }, f, indent=2)
    print(f"  Results saved to: {out_path}")
    if args.plot:
        save_scatter_plots(all_results, SCRIPT_DIR)
    print()
if __name__ == "__main__":
    main()
