#!/usr/bin/env python3
"""Measure the *gradient* each objective contributes, not just its value.

Why this exists
---------------
The CH2-homolog penalty and the per-molecule loss use different yardsticks:

* per-molecule loss divides ΔH by the train target std  (``loss_scale_h`` ~ 2.2e5 J/mol)
* homolog penalty   divides ΔH by the increment tolerance (``tol_h``      ~ 5.1e3 J/mol)

For a quadratic term the gradient carries a factor ``1/yardstick``, so for the same
prediction error the penalty's gradient is ``loss_scale_h / tol_h`` (~43x) steeper than
the main loss's.  A weight that makes the two *values* comparable therefore still lets
the penalty dominate the *update*.  This script reports both ratios on real data so the
weight can be chosen against the quantity that actually steers training.

Usage
-----
    python check/probe_homolog_gradients.py --lambda 0.1 [--member 0] [--split_seed 0]
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from tebnet.data.collate import ThermoCollater, HomologPairCollater
from tebnet.data.dataset import ThermochemDataset
from tebnet.data.homologs import build_homolog_pairs, HomologPairDataset
from tebnet.data.splits import deeptherm_split
from tebnet.models.tebnet import TEBNetV20
DEEPTHERM_T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
def grad_norm(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = sum(float(g.pow(2).sum()) for g in grads if g is not None)
    return sq ** 0.5
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=3)
    args = ap.parse_args()
    T_grid = DEEPTHERM_T_GRID
    full = ThermochemDataset.from_csv(
        "deeptherm_data.csv", T_grid=T_grid, strict=True,
        compute_aux=False, compute_comp=False,
    )
    records = full.records
    smiles = [r["smiles"] for r in records]
    tr_idx, _, _ = deeptherm_split(smiles, "random", seed=args.split_seed)
    train_ds = ThermochemDataset([records[i] for i in tr_idx], T_grid=T_grid,
                                 aux_feature_set="v15", compute_aux=True)
    collater = ThermoCollater()
    table = build_homolog_pairs(train_ds)
    pair_ds = HomologPairDataset(train_ds, table)
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=HomologPairCollater(base=collater, table=table))
    mol_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collater)
    model = TEBNetV20(homolog_weight=args.lam)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    main_batches = list(mol_loader)[: args.n_batches]
    pair_batches = list(pair_loader)[: args.n_batches]
    rows = []
    for mb, pb in zip(main_batches, pair_batches):
        losses = model.criterion(
            pred_cp=model(mb)["cp"], target_cp=mb.cp_grid,
            pred_dh_f=model(mb)["dh_f_298"], target_dh_f=mb.dh_f_298,
            pred_s298=model(mb)["s_298"], target_s298=mb.s_298,
            sample_weights=mb.weights,
        )
        diag = model.homolog_loss(pb)
        g_main = grad_norm(losses["total"], params)
        g_pen = grad_norm(diag["total"], params)
        rows.append((float(losses["total"]), float(diag["total"]),
                     float(diag["h"]), g_main, g_pen))
    print(f"lambda = {args.lam}   (batches: {len(rows)}, batch_size={args.batch_size})")
    print(f"{'main_loss':>10} {'pen':>10} {'pen_h':>10} {'|g_main|':>12} {'|g_pen|':>12} "
          f"{'value r':>9} {'grad r':>9} {'lam*grad r':>11}")
    for m, p, ph, gm, gp in rows:
        print(f"{m:>10.4f} {p:>10.4f} {ph:>10.4f} {gm:>12.3e} {gp:>12.3e} "
              f"{p / m:>9.2f} {gp / gm:>9.2f} {args.lam * gp / gm:>11.2f}")
    import statistics as st
    gr = st.mean(gp / gm for *_, gm, gp in rows)
    print(f"\nmean |g_pen| / |g_main| = {gr:.2f}")
    print(f"-> with lambda={args.lam} the penalty supplies {args.lam * gr:.2f}x the main "
          f"gradient.")
    target = 0.1
    print(f"-> for the penalty to supply {target:.0%} of the gradient, "
          f"lambda ~ {target / gr:.3e}")
if __name__ == "__main__":
    main()
