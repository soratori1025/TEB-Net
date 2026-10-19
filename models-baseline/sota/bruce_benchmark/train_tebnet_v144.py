"""
Train TEB-Net v14.2 on the WUDILY-CHO dataset following the Bruce et al. (2025)
evaluation protocol.

Protocol:
  - Dataset: WUDILY-CHO (6,074 C/H/O species)
  - Split: 80/20 random train/test (matching paper)
  - Validation: 10% of train pool carved for early stopping
  - Seeds: [42, 123, 2024, 8888, 9999]
  - Targets: ΔHf(298K), S(298K), Cp at T = {300,400,500,600,800,1000,1500} K
  - Metrics: MAE, RMSE, R² per property
  - Units: kcal/mol (ΔHf), cal/(mol·K) (S, Cp)

Data flow:
  CSV (kcal, cal units) → ThermochemDataset (converts to SI: J/mol, J/(mol·K)) → 
  TEBNet (trains in SI) → predictions (SI) → convert back to paper units for metrics
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

# ── Project paths ──
BRUCE_ROOT = Path(__file__).resolve().parent
TEBNET_V144_ROOT = BRUCE_ROOT.parent / "tebnet-v14.4"
TEBNET_MODEL_ROOT = BRUCE_ROOT.parent.parent.parent
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
)
from tebnet.training.callbacks import (
    ThermodynamicConsistencyCallback,
    PrintValidationMetricsCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bruce_tebnet")

# ── Unit Conversions ──
KCAL_TO_J = 4184.0
CAL_TO_J = 4.184
J_TO_KCAL = 1.0 / KCAL_TO_J
J_TO_CAL = 1.0 / CAL_TO_J
T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])


def load_wudily_dataset(csv_path: str, t_grid: np.ndarray) -> list[dict]:
    """Load processed WUDILY-CHO CSV and convert to TEB-Net record format.

    CSV units: kcal/mol (Hf), cal/(mol·K) (S, Cp)
    Record units: J/mol (Hf), J/(mol·K) (S, Cp) — TEB-Net's internal convention
    """
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        dh_f_298 = float(row["Hf_298_kcal_mol"]) * KCAL_TO_J  # kcal → J
        s_298 = float(row["S_298_cal_molK"]) * CAL_TO_J  # cal → J

        cp_values = []
        for T in t_grid:
            col = f"Cp_{T:.0f}_cal_molK"
            val = float(row[col]) * CAL_TO_J  # cal → J
            cp_values.append(val)

        record = {
            "smiles": smiles,
            "Cp": np.array(cp_values, dtype=np.float32),
            "dH_f_298": dh_f_298,
            "S_298": s_298,
            "weight": 1.0,
            "name": smiles,
        }
        records.append(record)

    logger.info(f"Loaded {len(records)} molecules from {csv_path}")
    return records


def load_split(split_path: str) -> dict:
    """Load pre-generated split indices."""
    with open(split_path) as f:
        return json.load(f)


def fit_composition_baseline(datapoints, target="dh_f_298"):
    """OLS fit target ≈ comp · w + b on the training set (SI units)."""
    X = np.stack([dp.comp for dp in datapoints]).astype(np.float64)
    y = np.array([getattr(dp, target) for dp in datapoints], dtype=np.float64)
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    w, b = sol[:-1], float(sol[-1])
    resid = y - (X @ w + b)
    return w.astype(np.float32), b, float(resid.std()), float(y.std())


def compute_metrics(model, loader, t_grid, device):
    """Compute per-property MAE, RMSE, R² in original units (kcal, cal)."""
    model.eval()
    all_h_pred, all_h_true = [], []
    all_s_pred, all_s_true = [], []
    all_cp_pred, all_cp_true = [], []
    all_smiles = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)

            # ΔHf298 in kcal/mol
            h_pred = preds["dh_f_298"].cpu().numpy() * J_TO_KCAL
            h_true = batch.dh_f_298.cpu().numpy() * J_TO_KCAL
            all_h_pred.append(h_pred)
            all_h_true.append(h_true)

            # S298 in cal/(mol·K)
            s_pred = preds["s_298"].cpu().numpy() * J_TO_CAL
            s_true = batch.s_298.cpu().numpy() * J_TO_CAL
            all_s_pred.append(s_pred)
            all_s_true.append(s_true)

            # Cp at all temps in cal/(mol·K)
            cp_pred = preds["cp"].cpu().numpy() * J_TO_CAL
            cp_true = batch.cp_grid.cpu().numpy() * J_TO_CAL
            all_cp_pred.append(cp_pred)
            all_cp_true.append(cp_true)
            all_smiles.extend(batch.smiles)

    h_pred = np.concatenate(all_h_pred)
    h_true = np.concatenate(all_h_true)
    s_pred = np.concatenate(all_s_pred)
    s_true = np.concatenate(all_s_true)
    cp_pred = np.concatenate(all_cp_pred)
    cp_true = np.concatenate(all_cp_true)

    def _metrics(pred, true):
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    results = {
        "Hf298": _metrics(h_pred, h_true),
        "S298": _metrics(s_pred, s_true),
    }

    # Per-temperature Cp metrics
    cp_all_mae, cp_all_rmse = [], []
    for i, T in enumerate(t_grid):
        mask = np.isfinite(cp_true[:, i])
        if mask.any():
            m = _metrics(cp_pred[mask, i], cp_true[mask, i])
            results[f"Cp_{T:.0f}K"] = m
            cp_all_mae.append(m["MAE"])
            cp_all_rmse.append(m["RMSE"])

    # Average Cp metrics across all temperatures
    if cp_all_mae:
        results["Cp_avg"] = {
            "MAE": float(np.mean(cp_all_mae)),
            "RMSE": float(np.mean(cp_all_rmse)),
            "R2": float(np.mean([results[f"Cp_{T:.0f}K"]["R2"] for T in t_grid if f"Cp_{T:.0f}K" in results])),
        }

    return results, {
        "h_pred": h_pred, "h_true": h_true,
        "s_pred": s_pred, "s_true": s_true,
        "cp_pred": cp_pred, "cp_true": cp_true,
        "smiles": all_smiles,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TEBNet v14.2 on WUDILY-CHO (Bruce protocol)")
    parser.add_argument("--seed", type=int, required=True, help="Model initialization seed")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Data split seed (default: same as --seed)")
    parser.add_argument("--csv", type=str,
                        default=str(BRUCE_ROOT / "data" / "wudily_cho_processed.csv"))
    parser.add_argument("--split_dir", type=str,
                        default=str(BRUCE_ROOT / "data" / "splits"))
    parser.add_argument("--out_dir", type=str,
                        default=str(BRUCE_ROOT / "results"))
    parser.add_argument("--max_epochs", type=int, default=200,
                        help="Max training epochs (paper trains until convergence)")
    parser.add_argument("--patience", type=int, default=40,
                        help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.split_seed is None:
        args.split_seed = args.seed

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    t_grid = T_GRID
    n_temps = len(t_grid)
    aux_dim = AUX_DIM_MINIMAL

    logger.info(
        f"[Seed {args.seed}] TEBNet v14.2 on WUDILY-CHO (Bruce protocol) | "
        f"split_seed={args.split_seed} | GPU {args.gpu}"
    )

    # ── Load dataset ──
    all_records = load_wudily_dataset(args.csv, t_grid)

    # ── Load pre-generated split ──
    split_file = os.path.join(args.split_dir, f"split_seed_{args.split_seed}.json")
    split = load_split(split_file)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    test_idx = split["test_indices"]

    logger.info(
        f"[split_seed={args.split_seed}] "
        f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
    )

    train_records = [all_records[i] for i in train_idx]
    val_records = [all_records[i] for i in val_idx]
    test_records = [all_records[i] for i in test_idx]

    # ── Create Datasets ──
    ds_kwargs = dict(T_grid=t_grid, aux_feature_set="routed", compute_comp=True)
    train_ds = ThermochemDataset(train_records, **ds_kwargs)
    val_ds = ThermochemDataset(val_records, **ds_kwargs)
    test_ds = ThermochemDataset(test_records, **ds_kwargs)

    collater = ThermoCollater()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collater, num_workers=4, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collater, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collater, num_workers=4, pin_memory=True,
        persistent_workers=True,
    )

    # ── Normalization Stats ──
    h_list, s_list, cp_list = [], [], []
    for batch in train_loader:
        h_list.append(batch.dh_f_298.reshape(-1))
        s_list.append(batch.s_298.reshape(-1))
        cp_list.append(batch.cp_grid.reshape(-1))

    h_valid = torch.cat(h_list); h_valid = h_valid[torch.isfinite(h_valid)]
    s_valid = torch.cat(s_list); s_valid = s_valid[torch.isfinite(s_valid)]
    cp_valid = torch.cat(cp_list); cp_valid = cp_valid[torch.isfinite(cp_valid)]

    mean_h, std_h = h_valid.mean().item(), max(h_valid.std().item(), 1e-6)
    mean_s, std_s = s_valid.mean().item(), max(s_valid.std().item(), 1e-6)
    std_cp = max(cp_valid.std().item(), 1e-6)

    logger.info(
        f"[Seed {args.seed}] Norm: H298 mean={mean_h*J_TO_KCAL:.1f} kcal std={std_h*J_TO_KCAL:.1f} | "
        f"S298 mean={mean_s*J_TO_CAL:.1f} cal std={std_s*J_TO_CAL:.1f} | Cp std={std_cp*J_TO_CAL:.1f}"
    )

    # ── Composition baselines ──
    w_comp, b_comp, resid_std, y_std = fit_composition_baseline(train_ds.datapoints, target="dh_f_298")
    pred_std_h = max(resid_std, 1e-6)
    logger.info(
        f"[Seed {args.seed}] H298 baseline: target_std={y_std*J_TO_KCAL:.2f} → "
        f"residual_std={resid_std*J_TO_KCAL:.2f} kcal/mol "
        f"({y_std/max(resid_std,1e-6):.1f}x reduction)"
    )

    w_s_comp, b_s_comp, resid_std_s, y_std_s = fit_composition_baseline(train_ds.datapoints, target="s_298")
    pred_std_s = max(resid_std_s, 1e-6)
    logger.info(
        f"[Seed {args.seed}] S298 baseline: target_std={y_std_s*J_TO_CAL:.2f} → "
        f"residual_std={resid_std_s*J_TO_CAL:.2f} cal/(mol·K) "
        f"({y_std_s/max(resid_std_s,1e-6):.1f}x reduction)"
    )

    # Setup v14.4 specific routing
    from tebnet.data import featurizer
    aux_dim = 16
    n_temps = len(T_GRID)
    
    import tebnet.models.tebnet as tm
    tm.ROUTE_H_IDX = featurizer.ROUTE_H_IDX
    tm.ROUTE_S_IDX = featurizer.ROUTE_S_IDX
    tm.ROUTE_CP_IDX = featurizer.ROUTE_CP_IDX
    tm.AUX_DIM_ROUTED = featurizer.AUX_DIM_ROUTED

    # ── Model ──
    model = TEBNet(
        d_e=len(CleanBondFeaturizer()),
        mp_hidden_dim=300, mp_depth=3, mp_dropout=0.1,
        agg_type="sum",
        aux_dim=aux_dim,
        branch_hidden=128, branch_layers=2, branch_dropout=0.3,
        n_temps=n_temps, batch_norm=True,
        loss_type="huber",
        loss_weights={"cp": 1.0, "h298": 2.0, "s298": 1.5},
        weight_strategy="fixed",
        norm_mean_h=mean_h, norm_std_h=std_h,
        norm_mean_s=mean_s, norm_std_s=std_s, norm_std_cp=std_cp,
        warmup_epochs=3, init_lr=1e-4, max_lr=8e-4, final_lr=1e-5,
        weight_decay=5e-5,
        t_mid=1000.0, t_ref=298.15,
        use_composition_baseline=True,
        comp_dim=COMP_DIM,
        pred_std_h=pred_std_h,
        use_s_composition_baseline=True,
        pred_std_s=pred_std_s,
    )

    # Install frozen composition baseline weights
    model.comp_weight.copy_(torch.tensor(w_comp, dtype=torch.float32))
    model.comp_bias.copy_(torch.tensor([b_comp], dtype=torch.float32))
    model.s_comp_weight.copy_(torch.tensor(w_s_comp, dtype=torch.float32))
    model.s_comp_bias.copy_(torch.tensor([b_s_comp], dtype=torch.float32))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[Seed {args.seed}] Model params: {n_params:,}")

    # Set aux normalization
    aux_list = []
    for batch in train_loader:
        if batch.x_physic is not None:
            aux_list.append(batch.x_physic)
    if aux_list:
        aux_all = torch.cat(aux_list, dim=0)
        model.aux_mean.copy_(aux_all.mean(0))
        model.aux_std.copy_(aux_all.std(0).clamp_min(1e-6))

    # ── Train ──
    save_dir = os.path.join(args.out_dir, f"seed{args.split_seed}", f"init{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir, filename="best", monitor="val/loss_total",
            mode="min", save_top_k=1,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/loss_total", patience=args.patience, mode="min", verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ThermodynamicConsistencyCallback(),
        PrintValidationMetricsCallback(),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=1,
        precision="32-true", callbacks=callbacks, log_every_n_steps=10,
        default_root_dir=save_dir, gradient_clip_val=1.0,
        enable_progress_bar=False,
    )

    t_start = time.time()
    trainer.fit(model, train_loader, val_loader)
    t_train = time.time() - t_start
    logger.info(f"[Seed {args.seed}] Training done in {t_train:.0f}s ({t_train/60:.1f} min)")

    # ── Load best & evaluate ──
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        model = TEBNet.load_from_checkpoint(best_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_metrics, _ = compute_metrics(model, DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collater, num_workers=4, pin_memory=True,
    ), t_grid, device)
    test_metrics, test_preds = compute_metrics(model, test_loader, t_grid, device)
    val_metrics, val_preds = compute_metrics(model, val_loader, t_grid, device)

    # ── Save ──
    results = {
        "model": "TEBNet-v14.2",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "protocol": "Bruce_et_al_2025",
        "dataset": "WUDILY-CHO",
        "split_ratio": "80/20",
        "n_total": len(all_records),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_params": n_params,
        "epochs_trained": trainer.current_epoch,
        "training_time_sec": round(t_train, 1),
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score)
            if trainer.checkpoint_callback.best_model_score else None,
        "train_metrics": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in train_metrics.items()},
        "val_metrics": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in val_metrics.items()},
        "test_metrics": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in test_metrics.items()},
        "checkpoint": best_ckpt,
    }

    with open(Path(save_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    np.savez(
        Path(save_dir) / "predictions.npz",
        **{f"test_{k}": v for k, v in test_preds.items() if isinstance(v, np.ndarray)},
        **{f"val_{k}": v for k, v in val_preds.items() if isinstance(v, np.ndarray)},
        test_smiles=np.array(test_preds["smiles"], dtype=object),
    )

    # ── Summary ──
    print("\n" + "=" * 80)
    print(f" TEBNet v14.2 — Bruce Protocol (split_seed={args.split_seed}, init_seed={args.seed})")
    print("=" * 80)
    print(f"  Dataset: WUDILY-CHO ({len(all_records)} species)")
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    print(f"  Params: {n_params:,} | Epochs: {trainer.current_epoch} | Time: {t_train:.0f}s")
    print()
    print(f"  {'Property':<25s}  {'Train MAE':>12s}  {'Test MAE':>12s}  {'Test RMSE':>12s}  {'Test R²':>8s}  {'Unit':<15s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*15}")

    for prop, unit in [("Hf298", "kcal/mol"), ("S298", "cal/(mol·K)"), ("Cp_avg", "cal/(mol·K)")]:
        if prop in test_metrics:
            tm = train_metrics[prop]
            em = test_metrics[prop]
            print(f"  {prop:<25s}  {tm['MAE']:>12.4f}  {em['MAE']:>12.4f}  {em['RMSE']:>12.4f}  {em['R2']:>8.4f}  {unit:<15s}")

    # Individual Cp temperatures
    print()
    for T in T_GRID:
        key = f"Cp_{T:.0f}K"
        if key in test_metrics:
            tm = train_metrics[key]
            em = test_metrics[key]
            print(f"  {key:<25s}  {tm['MAE']:>12.4f}  {em['MAE']:>12.4f}  {em['RMSE']:>12.4f}  {em['R2']:>8.4f}  {'cal/(mol·K)':<15s}")

    print("=" * 80)
    print(f"  Results: {save_dir}/results.json")
    print()


if __name__ == "__main__":
    main()
