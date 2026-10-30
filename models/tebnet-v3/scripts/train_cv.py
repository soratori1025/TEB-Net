"""TEB-Net on small, hard datasets under the nested cross-validation protocol of Ureel et al.
("Beyond group additivity: Transfer learning for molecular thermochemistry prediction",
Chem. Eng. J. 472 (2023) 144874; data: github.com/yureel/HC-Thermochemistry, MIT licence).

Protocol (matched to Ureel et al. §2.2.2)
  * outer loop: K_out = 10 folds; every molecule is in the test fold exactly once
  * inner loop: the remaining 9 folds are split into K_in = 9 folds; each inner model uses one of them as
    validation (best checkpoint) and the other 8 for training
  * the test prediction of a molecule = mean of the K_in inner models of its outer fold
  * metrics over the whole dataset: MAE (RMSE) of ΔH_f,298 [kJ/mol], S298 and Cp(T) [J/(mol·K)]
    -> directly comparable with their Tables 1–3 (GAV, non-pretrained D-MPNN, transfer learning)

Arms (one tag each; run H and S/Cp models separately and join them with --combine, as in deeptherm_eval.py)
  scratch   : TEB-Net trained only on the small dataset (no pretraining)            [default]
  finetune  : TEB-Net initialised from a checkpoint trained on DeepTherm (--init_ckpt), baselines and feature
              normalisation refitted on the small dataset; all weights trainable (as Ureel's transfer step)
  prior     : the frozen ridge prior alone (no network) = TEB-Net's analogue of group additivity (--prior_only)

Typical use (ΔH-only and S/Cp-only models, outer folds in parallel on one GPU)
  CSV=data/ab_initio_database_hc.csv
  for k in 0 1 2 3 4 5 6 7 8 9; do
    python scripts/train_cv.py --csv $CSV --outer_folds $k --loss_weights 1 0 0 --monitor h &
    python scripts/train_cv.py --csv $CSV --outer_folds $k --loss_weights 0 1.5 1 --monitor scp &
  done; wait
  python scripts/train_cv.py --csv $CSV --prior_only
  python scripts/train_cv.py --csv $CSV --summarize                        # every tag of this dataset
  python scripts/train_cv.py --csv $CSV --summarize --combine <H_TAG> <SCP_TAG>

  # pretraining set without the molecules of the small dataset (no leakage), then pretrain with train.py
  python scripts/train_cv.py --csv $CSV --write_filtered_pretrain ../../models-baseline/deeptherm_data.csv \
         data/deeptherm_minus_hc.csv

Units: Ureel's CSVs are in kJ/mol and J/(mol·K), so --h_unit kj --s_unit j --cp_unit j are the defaults here
(the magnitude-based guess of the loader would read hydrocarbon ΔH in kJ/mol as kcal/mol). Check the column
mapping printed at start-up; pass --h_col / --s_col / --cp_cols if a column is not found.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    make_atom_featurizer, DeepThermValMAE, collect_predictions, COL, DEEPTHERM_T_GRID,
)
from tebnet.data.dataset import ThermochemDataset
from tebnet.data.collate import ThermoCollater
from tebnet.data.baselines import fit_scalar_baseline, fit_cp_cheb_baseline, cp_from_cheb
from tebnet.models.tebnet import TEBNetV20
from tebnet.data.featurizer import (
    CleanBondFeaturizer, V15_NAMES, FEATURES_VERSION, v15_route_names, v15_base_names, S_PRIOR_VARIANTS,
)
logger = logging.getLogger("tebnet_cv")
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
TRANSFER_PREFIXES = ("message_passing.", "bn.", "h_branch.", "s_branch.", "cp_branch.")
def outer_folds(n: int, k_out: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k_out)]
def inner_folds(pool: np.ndarray, k_in: int, seed: int, outer_k: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1000 * (seed + 1) + outer_k)
    perm = pool[rng.permutation(len(pool))]
    return [np.sort(f) for f in np.array_split(perm, k_in)]
def arrays(dps, idx):
    X = np.stack([dps[i].x_physic for i in idx]).astype(np.float64)
    H = np.array([dps[i].dh_f_298 for i in idx], dtype=np.float64)
    S = np.array([dps[i].s_298 for i in idx], dtype=np.float64)
    CP = np.stack([dps[i].cp_grid for i in idx]).astype(np.float64)
    return X, H, S, CP
def _feature_names(args) -> list[str]:
    """v15 names, plus the v3 group / ring-correction columns when --prior_groups is on."""
    return list(getattr(args, "feature_names", None) or V15_NAMES)
def fit_priors(args, X, H, S, CP, t_grid, t_mid, t_low_dom, use_3d):
    base = {k: v15_base_names(k, "composition" if args.baseline == "none" else args.baseline, use_3d)
            for k in ("h", "cp")}
    from tebnet.data.featurizer import s_prior_columns
    s_base, s_fixed_dict = s_prior_columns(getattr(args, "s_prior", "v1"))
    base["s"] = s_base
    extra = list(getattr(args, "extra_base_names", None) or [])
    if extra and args.baseline == "physics":
        base = {k: v + extra for k, v in base.items()}
    col = {n: i for i, n in enumerate(_feature_names(args))}
    cols = {k: [col[n] for n in v] for k, v in base.items()}
    Xb = np.zeros_like(X) if args.baseline == "none" else X.copy()
    if extra and args.baseline == "physics":
        mc = int(getattr(args, "group_min_count", 0) or 0)
        ex = [col[n] for n in extra]
        rare = [j for j in ex if (Xb[:, j] != 0).sum() < mc]
        Xb[:, rare] = 0.0
    bh = fit_scalar_baseline(Xb[:, cols["h"]], H, args.ridge_lambda)
    if s_fixed_dict:
        from tebnet.data.baselines import fit_scalar_baseline_fixed
        fixed_arr = np.full(len(cols["s"]), np.nan)
        for i, c_name in enumerate(base["s"]):
            if c_name in s_fixed_dict:
                fixed_arr[i] = s_fixed_dict[c_name]
        bs = fit_scalar_baseline_fixed(Xb[:, cols["s"]], S, fixed_arr, args.ridge_lambda)
    else:
        bs = fit_scalar_baseline(Xb[:, cols["s"]], S, args.ridge_lambda)
    bc = fit_cp_cheb_baseline(Xb[:, cols["cp"]], CP, t_grid, t_mid, t_low_dom, args.t_high_dom,
                              lam_rel=args.ridge_lambda)
    return base, cols, bh, bs, bc
def prior_predict(X, cols, bh, bs, bc, t_grid, t_mid, t_low_dom, t_high_dom):
    h = X[:, cols["h"]] @ bh["w"] + bh["b"]
    s = X[:, cols["s"]] @ bs["w"] + bs["b"]
    C = X[:, cols["cp"]] @ bc["W"] + bc["b"]
    cp = cp_from_cheb(C, t_grid, t_mid, t_low_dom, t_high_dom)
    return h, s, cp
def load_transfer_weights(model: TEBNetV20, ckpt: str) -> None:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    own = model.state_dict()
    take, skipped = {}, []
    for k, v in sd.items():
        if not k.startswith(TRANSFER_PREFIXES):
            continue
        if k in own and own[k].shape == v.shape:
            take[k] = v
        else:
            skipped.append(k)
    if not take:
        raise ValueError(f"no transferable weights in {ckpt} (architecture / featurizer mismatch?)")
    if skipped:
        raise ValueError(f"{len(skipped)} weights of {ckpt} do not match this model (e.g. {skipped[:3]}): "
                         f"use the same --atom_featurizer / --mp_* / --branch_* / --route_mode / --no_3d "
                         f"as the pretraining run")
    model.load_state_dict(take, strict=False)
    logger.info(f"transferred {len(take)} tensors from {ckpt}")
def _beta(delta, to_si: float, scale: float) -> float:
    """Huber transition point in loss units (target / scale); None -> 1.0 (= v1)."""
    return 1.0 if delta is None else float(delta) * to_si / float(scale)
def train_inner(args, full_ds, tr, va, seed, save_dir, t_grid, t_mid, t_low_dom, use_3d, collater, d_v, device):
    pl.seed_everything(seed, workers=True)
    dps = full_ds.datapoints
    X, H, S, CP = arrays(dps, tr)
    base, cols, bh, bs, bc = fit_priors(args, X, H, S, CP, t_grid, t_mid, t_low_dom, use_3d)
    route = {k: v15_route_names(k, args.route_mode, use_3d) for k in ("h", "s", "cp")}
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-3] = 1.0
    if args.loss_norm == "residual":
        ls_h, ls_s, ls_cp = bh["resid_std"], bs["resid_std"], bc["cp_resid_std"]
    else:
        ls_h, ls_s, ls_cp = bh["target_std"], bs["target_std"], bc["cp_target_std"]
    model = TEBNetV20(
        feature_names=_feature_names(args),
        route_h=route["h"], route_s=route["s"], route_cp=route["cp"],
        base_h=base["h"], base_s=base["s"], base_cp=base["cp"],
        mp_hidden_dim=args.mp_hidden, mp_depth=args.mp_depth, mp_dropout=0.1,
        agg_type="sum", batch_norm=True, d_v=d_v, d_e=len(CleanBondFeaturizer()),
        branch_hidden=args.branch_hidden, branch_layers=args.branch_layers, branch_dropout=args.branch_dropout,
        t_ref=298.15, t_mid=t_mid, t_low_dom=t_low_dom, t_high_dom=args.t_high_dom,
        cp_param="chebyshev", loss_type="huber",
        loss_weights={"h298": args.loss_weights[0], "s298": args.loss_weights[1], "cp": args.loss_weights[2]},
        weight_strategy="fixed",
        loss_scale_cp=ls_cp, loss_scale_h=ls_h, loss_scale_s=ls_s,
        huber_beta_h=_beta(getattr(args, "huber_delta_h", None), 4184.0, ls_h),
        huber_beta_s=_beta(getattr(args, "huber_delta_s", None), 4.184, ls_s),
        huber_beta_cp=_beta(getattr(args, "huber_delta_cp", None), 4.184, ls_cp),
        warmup_epochs=args.warmup_epochs, init_lr=args.init_lr, max_lr=args.max_lr, final_lr=args.final_lr,
        weight_decay=5e-5,
    )
    if args.init_ckpt:
        load_transfer_weights(model, args.init_ckpt)
    model.set_baselines(bh, bs, bc)
    model.set_aux_norm(mu, sd)
    dl = dict(batch_size=args.batch_size, collate_fn=collater, num_workers=args.num_workers)
    train_loader = DataLoader(Subset(full_ds, tr), shuffle=True, drop_last=len(tr) > args.batch_size, **dl)
    val_loader = DataLoader(Subset(full_ds, va), shuffle=False, **dl)
    monitor = {"deeptherm": "val/deeptherm_mae", "h": "val/mae_H_kcal", "s": "val/mae_S_cal",
               "cp300": "val/mae_Cp300_cal", "scp": "val/mae_SCp"}[args.monitor]
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="best", monitor=monitor, mode="min",
                                           save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu" if device.type == "cuda" else "cpu", devices=1,
        precision="32-true", callbacks=[DeepThermValMAE(n_temps=len(t_grid)), ckpt_cb],
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        gradient_clip_val=1.0, default_root_dir=save_dir, log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, val_loader)
    best_val = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    if ckpt_cb.best_model_path and Path(ckpt_cb.best_model_path).exists():
        model = TEBNetV20.load_from_checkpoint(ckpt_cb.best_model_path)
    return model.to(device), best_val, bh, bs, bc
def _mae_rmse(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    e = pred[m] - true[m]
    return (float(np.abs(e).mean()), float(np.sqrt(np.square(e).mean())), int(m.sum())) if m.any() \
        else (float("nan"), float("nan"), 0)
def report(p: dict, t_grid: np.ndarray, title: str) -> dict:
    """p in SI units (J/mol, J/(mol·K)). Prints Ureel units (kJ/mol, J/(mol·K)) and DeepTherm units."""
    out = {"H_kJ": _mae_rmse(p["h_pred"] / 1000.0, p["h_true"] / 1000.0),
           "S_J": _mae_rmse(p["s_pred"], p["s_true"])}
    for j, T in enumerate(t_grid):
        out[f"Cp{T:.0f}_J"] = _mae_rmse(p["cp_pred"][:, j], p["cp_true"][:, j])
    cp_mae = [out[f"Cp{T:.0f}_J"][0] for T in t_grid]
    out["Cp_mean_MAE_J"] = float(np.nanmean(cp_mae))
    i300 = int(np.argmin(np.abs(t_grid - 300.0)))
    print(f"\n  {title}   n = {len(p['h_true'])}")
    print(f"  {'':<14s} {'ΔH298 kJ/mol':>16s} {'S298 J/(mol K)':>16s} {'Cp300 J/(mol K)':>16s} {'Cp mean':>9s}")
    h, s, c = out["H_kJ"], out["S_J"], out[f"Cp{t_grid[i300]:.0f}_J"]
    print(f"  {'MAE (RMSE)':<14s} {h[0]:>7.2f} ({h[1]:>6.2f}) {s[0]:>7.2f} ({s[1]:>6.2f}) "
          f"{c[0]:>7.2f} ({c[1]:>6.2f}) {out['Cp_mean_MAE_J']:>9.2f}")
    print(f"  {'DeepTherm units':<14s} ΔH {h[0] * 1000 * J_TO_KCAL:.3f} kcal/mol | S {s[0] * J_TO_CAL:.3f} | "
          f"Cp300 {c[0] * J_TO_CAL:.3f} | Cp mean {out['Cp_mean_MAE_J'] * J_TO_CAL:.3f} cal/(mol K)")
    print("  per-T Cp MAE [J/(mol K)]: " + "  ".join(f"{T:.0f}K {out[f'Cp{T:.0f}_J'][0]:.2f}" for T in t_grid))
    return out
def load_tag(run_dir: Path):
    files = sorted(run_dir.glob("outer*.npz"))
    if not files:
        return None
    parts = [np.load(f, allow_pickle=True) for f in files]
    idx = np.concatenate([z["test_idx"] for z in parts])
    order = np.argsort(idx)
    p = {k: np.concatenate([z[k] for z in parts])[order]
         for k in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true")}
    meta = json.loads(str(parts[0]["meta"]))
    return p, idx[order], meta, len(files)
def summarize(args, root: Path) -> None:
    tags = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not tags:
        sys.exit(f"no runs under {root}")
    loaded = {}
    for d in tags:
        r = load_tag(d)
        if r is None:
            continue
        p, idx, meta, n_files = r
        loaded[d.name] = (p, idx, meta)
        t_grid = np.asarray(meta["t_grid"])
        note = "" if n_files == meta["n_outer"] else f"  [only {n_files}/{meta['n_outer']} outer folds done]"
        print("\n" + "=" * 92 + f"\n {d.name}{note}\n" + "=" * 92)
        report(p, t_grid, f"{meta['mode']} | weights {meta['loss_weights']} | monitor {meta['monitor']}")
    if args.combine:
        th, ts = args.combine
        if th not in loaded or ts not in loaded:
            sys.exit(f"--combine: tag not found. Available: {sorted(loaded)}")
        (ph, ih, mh), (ps, is_, ms) = loaded[th], loaded[ts]
        if not np.array_equal(ih, is_):
            sys.exit("--combine: the two tags do not cover the same molecules (different folds?)")
        p = dict(ps)
        p["h_pred"] = ph["h_pred"]
        print("\n" + "#" * 92 + f"\n DECOUPLED  ΔH <- {th}\n            S, Cp <- {ts}\n" + "#" * 92)
        report(p, np.asarray(ms["t_grid"]), "decoupled TEB-Net (nested CV, inner ensembles)")
def write_filtered_pretrain(target_csv: str, src_csv: str, out_csv: str, smiles_col: str) -> None:
    """Copies src_csv without the molecules of target_csv (matched by InChIKey, else canonical SMILES)."""
    import pandas as pd
    from rdkit import Chem
    def key(smi):
        m = Chem.MolFromSmiles(str(smi).strip())
        if m is None:
            return None
        try:
            k = Chem.MolToInchiKey(m)
            if k:
                return k
        except Exception:
            pass
        return Chem.MolToSmiles(m)
    tgt = pd.read_csv(target_csv)
    src = pd.read_csv(src_csv)
    t_keys = {key(s) for s in tgt[smiles_col]} - {None}
    s_keys = [key(s) for s in src[smiles_col]]
    drop = np.array([k in t_keys for k in s_keys])
    src.loc[~drop].to_csv(out_csv, index=False)
    print(f"{src_csv}: {len(src)} rows, {int(drop.sum())} shared with {target_csv} removed -> {out_csv} "
          f"({int((~drop).sum())} rows). Pretrain on it with scripts/train.py (row order now differs from the "
          f"DeepTherm SI, which is expected).")
def main():
    ap = argparse.ArgumentParser(description="TEB-Net, nested CV protocol of Ureel et al. (2023)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--h_col", default=None)
    ap.add_argument("--s_col", default=None)
    ap.add_argument("--cp_cols", default=None, help="comma-separated, one per T in 300,400,500,600,800,1000,1500 K")
    ap.add_argument("--h_unit", choices=["kcal", "kj", "j"], default="kj")
    ap.add_argument("--s_unit", choices=["cal", "j"], default="j")
    ap.add_argument("--cp_unit", choices=["cal", "j"], default="j")
    ap.add_argument("--physics_cache", default="auto")
    ap.add_argument("--out_dir", default="benchmark_results_cv")
    ap.add_argument("--tag_suffix", default="")
    ap.add_argument("--n_outer", type=int, default=10)
    ap.add_argument("--n_inner", type=int, default=9)
    ap.add_argument("--outer_folds", type=int, nargs="*", default=None, help="subset of outer folds to run")
    ap.add_argument("--cv_seed", type=int, default=0, help="fold assignment (keep fixed across arms)")
    ap.add_argument("--seed", type=int, default=0, help="model initialisation")
    ap.add_argument("--ensemble", choices=["mean", "inv_val"], default="mean",
                    help="mean = Ureel et al.; inv_val = weights 1/val-MAE as DeepTherm")
    ap.add_argument("--prior_only", action="store_true", help="frozen ridge prior alone, no network")
    ap.add_argument("--init_ckpt", default=None, help="fine-tune from this TEB-Net checkpoint (e.g. trained on DeepTherm)")
    ap.add_argument("--write_filtered_pretrain", nargs=2, metavar=("SRC_CSV", "OUT_CSV"), default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--combine", nargs=2, metavar=("H_TAG", "SCP_TAG"), default=None)
    ap.add_argument("--loss_weights", type=float, nargs=3, default=[2.0, 1.5, 1.0], metavar=("H", "S", "CP"))
    ap.add_argument("--monitor", choices=["deeptherm", "h", "s", "cp300", "scp"], default="deeptherm")
    ap.add_argument("--atom_featurizer", choices=["chemprop", "radical", "thermo"], default="thermo")
    ap.add_argument("--baseline", choices=["none", "composition", "physics"], default="physics")
    ap.add_argument("--s_prior", choices=list(S_PRIOR_VARIANTS), default="v1", help="Entropy prior variant")
    ap.add_argument("--n_conf", type=int, default=10, help="ETKDG conformers per species for the v2 entropy terms")
    ap.add_argument("--conf_workers", type=int, default=8, help="processes for the v2 conformer search")
    ap.add_argument("--no_3d", action="store_true")
    ap.add_argument("--route_mode", choices=["grouped", "full"], default="grouped")
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--loss_norm", choices=["global", "residual"], default="global")
    ap.add_argument("--mp_depth", type=int, default=5)
    ap.add_argument("--mp_hidden", type=int, default=300)
    ap.add_argument("--branch_hidden", type=int, default=256)
    ap.add_argument("--branch_layers", type=int, default=2)
    ap.add_argument("--branch_dropout", type=float, default=0.1)
    ap.add_argument("--t_mid", type=float, default=1000.0)
    ap.add_argument("--t_high_dom", type=float, default=3000.0)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--init_lr", type=float, default=1e-4)
    ap.add_argument("--max_lr", type=float, default=8e-4, help="use ~2e-4 with --init_ckpt")
    ap.add_argument("--final_lr", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.write_filtered_pretrain:
        write_filtered_pretrain(args.csv, *args.write_filtered_pretrain, smiles_col=args.smiles_col)
        return
    ds_name = Path(args.csv).stem
    root = Path(args.out_dir) / ds_name
    if args.summarize:
        summarize(args, root)
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_3d = not args.no_3d
    t_grid = DEEPTHERM_T_GRID
    t_mid = args.t_mid
    t_low_dom = float(t_grid[t_grid <= t_mid].min())
    mode = "prior" if args.prior_only else ("finetune" if args.init_ckpt else "scratch")
    tag = mode
    if mode != "prior":
        tag += ("" if args.loss_weights == [2.0, 1.5, 1.0] else "_w" + "-".join(f"{w:g}" for w in args.loss_weights))
        tag += ("" if args.monitor == "deeptherm" else f"_mon-{args.monitor}")
        tag += f"_{args.atom_featurizer}_d{args.mp_depth}_bh{args.branch_hidden}_ep{args.max_epochs}"
    tag += f"_{args.baseline}" + ("" if use_3d else "_2d") + f"_cv{args.cv_seed}_s{args.seed}" + args.tag_suffix
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    full_ds = ThermochemDataset.from_csv(
        args.csv, T_grid=t_grid, smiles_col=args.smiles_col, physics_cache=args.physics_cache,
        h_col=args.h_col, s_col=args.s_col,
        cp_cols=[c.strip() for c in args.cp_cols.split(",")] if args.cp_cols else None,
        h_unit=args.h_unit, s_unit=args.s_unit, cp_unit=args.cp_unit, strict=True,
        compute_aux=True, aux_feature_set="v15", use_3d=use_3d, compute_comp=False,
    )
    if args.s_prior != "v1":
        from tebnet.data.featurizer import load_or_compute_entropy_v2, entropy_v2_cache_path, ENTROPY_V2_NAMES
        cache_path = entropy_v2_cache_path(args.csv)
        ev2 = load_or_compute_entropy_v2([dp.smiles for dp in full_ds.datapoints], cache_path=cache_path, n_conf=args.n_conf, workers=args.conf_workers)
        for dp, g in zip(full_ds.datapoints, ev2):
            dp.x_physic = np.concatenate([dp.x_physic, g]).astype(np.float32)
        args.feature_names = list(getattr(args, "feature_names", V15_NAMES)) + ENTROPY_V2_NAMES
    dps = full_ds.datapoints
    n = len(dps)
    smiles = [dp.smiles for dp in dps]
    folds = outer_folds(n, args.n_outer, args.cv_seed)
    logger.info(f"{ds_name}: {n} species | {args.n_outer} outer x {args.n_inner} inner folds | arm '{mode}' | "
                f"run dir {run_dir}")
    if use_3d and max(dp.x_physic[COL["moi_valid"]] for dp in dps) < 0.5:
        logger.warning("use_3d=True but no valid 3-D features")
    atom_feat = make_atom_featurizer(args.atom_featurizer)
    collater = ThermoCollater(atom_featurizer=atom_feat)
    d_v = len(atom_feat) if atom_feat is not None else None
    todo = args.outer_folds if args.outer_folds else list(range(args.n_outer))
    for k in todo:
        out_file = run_dir / f"outer{k}.npz"
        if out_file.exists() and not args.overwrite:
            logger.info(f"outer fold {k} already done ({out_file}); skipping")
            continue
        te = folds[k]
        pool = np.sort(np.concatenate([f for j, f in enumerate(folds) if j != k]))
        _, Ht, St, CPt = arrays(dps, te)
        t0 = time.time()
        if args.prior_only:
            X, H, S, CP = arrays(dps, pool)
            _, cols, bh, bs, bc = fit_priors(args, X, H, S, CP, t_grid, t_mid, t_low_dom, use_3d)
            Xt = arrays(dps, te)[0]
            h, s, cp = prior_predict(Xt, cols, bh, bs, bc, t_grid, t_mid, t_low_dom, args.t_high_dom)
            members = {"h": h[None], "s": s[None], "cp": cp[None]}
            vals = np.array([np.nan])
        else:
            members = {"h": [], "s": [], "cp": []}
            vals = []
            test_loader = DataLoader(Subset(full_ds, te), batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collater, num_workers=args.num_workers)
            for j, va in enumerate(inner_folds(pool, args.n_inner, args.cv_seed, k)):
                tr = np.setdiff1d(pool, va)
                save_dir = run_dir / f"outer{k}" / f"inner{j}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model, best_val, *_ = train_inner(
                    args, full_ds, tr, va, args.seed + 100 * k + j, str(save_dir),
                    t_grid, t_mid, t_low_dom, use_3d, collater, d_v, device)
                p = collect_predictions(model, test_loader, device)
                members["h"].append(p["h_pred"])
                members["s"].append(p["s_pred"])
                members["cp"].append(p["cp_pred"])
                vals.append(best_val)
                logger.info(f"outer {k} inner {j}: {len(tr)} train / {len(va)} val | best val {best_val:.3f}")
                if not args.keep_ckpt:
                    shutil.rmtree(save_dir, ignore_errors=True)
                del model
                torch.cuda.empty_cache()
            members = {key: np.stack(v) for key, v in members.items()}
            vals = np.asarray(vals, dtype=float)
        if args.ensemble == "inv_val" and np.isfinite(vals).all():
            w = 1.0 / np.maximum(vals, 1e-12)
            w /= w.sum()
        else:
            w = np.full(len(vals), 1.0 / len(vals))
        ens = {key: np.tensordot(w, v, axes=(0, 0)) for key, v in members.items()}
        meta = {"mode": mode, "tag": tag, "n_outer": args.n_outer, "n_inner": args.n_inner,
                "cv_seed": args.cv_seed, "loss_weights": args.loss_weights, "monitor": args.monitor,
                "t_grid": t_grid.tolist(), "features_version": FEATURES_VERSION, "args": vars(args)}
        np.savez(out_file, test_idx=te, smiles=np.array([smiles[i] for i in te], dtype=object),
                 h_pred=ens["h"], s_pred=ens["s"], cp_pred=ens["cp"], h_true=Ht, s_true=St, cp_true=CPt,
                 member_h=members["h"], member_s=members["s"], member_cp=members["cp"],
                 member_val=vals, weights=w, meta=json.dumps(meta))
        report({"h_pred": ens["h"], "s_pred": ens["s"], "cp_pred": ens["cp"],
                "h_true": Ht, "s_true": St, "cp_true": CPt}, t_grid,
               f"outer fold {k} ({len(te)} test) | {time.time() - t0:.0f}s")
    if all((run_dir / f"outer{k}.npz").exists() for k in range(args.n_outer)):
        r = load_tag(run_dir)
        print("\n" + "=" * 92 + f"\n {ds_name} | {tag} | all {args.n_outer} outer folds\n" + "=" * 92)
        report(r[0], t_grid, "nested CV, whole dataset")
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
