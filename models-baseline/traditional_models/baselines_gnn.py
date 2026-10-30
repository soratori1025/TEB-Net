"""Graph-neural-network baselines for Table 1 (PyTorch Geometric) under the DeepTherm protocol.

Same rows, units and splits as TEB-Net (scripts/baseline_common.py); output readable by
scripts/deeptherm_eval.py (including --combine and --exclude_file) and scripts/run_info.py.
Nothing here imports TEB-Net's model code; the graph featurisation below is written for the baselines.

Architectures (--arch):
  gin          GINE convolutions (edge features), residual + batch norm, sum readout
  gcn          GCN convolutions (no edge features), residual + batch norm, sum readout
  gat          GATv2 convolutions (edge features, 4 heads), residual + batch norm, sum readout
  attentivefp  PyG AttentiveFP (graph attention + GRU readout)
Sum readout: ΔHf, S and Cp are size-extensive, a mean readout would hide molecular size.

Targets (--targets):
  all   one multitask network for the 9 DeepTherm targets (DeepTherm / chemprop-multitask style)
  h     ΔHf only; s, cp: S or Cp(T) only; scp: S and Cp(T)
  Decoupled GNN = one "h" run + one "scp" run, combined with deeptherm_eval.py --combine H_TAG SCP_TAG.

Fair-comparison settings (no tuning, same budget as TEB-Net): atom features include the number of
radical electrons (half of the dataset are radicals); targets standardised on train; masked MSE;
AdamW, 3-epoch warm-up to 8e-4 then cosine to 1e-5 over 100 epochs, batch 32, gradient clip 1.0;
the checkpoint with the lowest validation MAE (DeepTherm units, averaged over the trained targets) is
kept — no early stopping, as in train.py.

  python scripts/baselines_gnn.py --arch gin --targets all --split_seed 0
  python scripts/baselines_gnn.py --arch attentivefp --targets h   --split_type complexity
"""
from __future__ import annotations
import argparse
import copy
import math
import random
import time
import numpy as np
from baseline_common import (
    TARGET_NAMES, add_common_args, dt_to_si, get_split, load_dataset, pred_dict, prepare_run_dir, targets_dt, targets_si,
    write_run,
)
TARGET_COLS = {"all": list(range(9)), "h": [0], "s": [1], "cp": list(range(2, 9)), "scp": list(range(1, 9))}
def _one_hot(v, choices):
    return [float(v == c) for c in choices] + [float(v not in choices)]
def atom_features(a) -> list[float]:
    from rdkit import Chem
    hyb = Chem.rdchem.HybridizationType
    f = _one_hot(a.GetAtomicNum(), [1, 6, 7, 8, 9, 16, 17])
    f += _one_hot(a.GetTotalDegree(), [0, 1, 2, 3, 4])
    f += _one_hot(a.GetFormalCharge(), [-1, 0, 1])
    f += _one_hot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
    f += _one_hot(a.GetNumRadicalElectrons(), [0, 1, 2])
    f += _one_hot(a.GetHybridization(), [hyb.S, hyb.SP, hyb.SP2, hyb.SP3])
    f += [float(a.GetIsAromatic()), float(a.IsInRing())]
    f += [float(a.IsInRingSize(k)) for k in range(3, 9)]
    f += [a.GetMass() / 100.0]
    return f
def bond_features(b) -> list[float]:
    from rdkit import Chem
    bt = b.GetBondType()
    f = [float(bt == t) for t in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
                                  Chem.BondType.AROMATIC)]
    f += [float(b.GetIsConjugated()), float(b.IsInRing())]
    f += [float(b.IsInRingSize(k)) for k in (3, 4, 5, 6)]
    return f
def mol_to_graph(mol, y_row: np.ndarray, rid: int):
    import torch
    from torch_geometric.data import Data
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    n_bond_feat = 10
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = (torch.tensor(ea, dtype=torch.float32) if ea
                 else torch.zeros((0, n_bond_feat), dtype=torch.float32))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(y_row, dtype=torch.float32).view(1, -1), rid=int(rid))
def build_model(args, in_dim: int, edge_dim: int, n_out: int):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, global_add_pool, global_mean_pool
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            h = args.hidden
            self.arch = args.arch
            if args.arch == "attentivefp":
                from torch_geometric.nn.models import AttentiveFP
                self.enc = AttentiveFP(in_channels=in_dim, hidden_channels=h, out_channels=h, edge_dim=edge_dim,
                                       num_layers=args.layers, num_timesteps=args.timesteps, dropout=args.dropout)
            else:
                self.inp = nn.Linear(in_dim, h)
                self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
                for _ in range(args.layers):
                    if args.arch == "gin":
                        mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
                        conv = GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
                    elif args.arch == "gcn":
                        conv = GCNConv(h, h)
                    else:
                        conv = GATv2Conv(h, h // args.heads, heads=args.heads, edge_dim=edge_dim)
                    self.convs.append(conv)
                    self.norms.append(nn.BatchNorm1d(h))
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(h, n_out))
        def forward(self, d):
            if self.arch == "attentivefp":
                g = self.enc(d.x, d.edge_index, d.edge_attr, d.batch)
            else:
                x = F.relu(self.inp(d.x))
                for conv, norm in zip(self.convs, self.norms):
                    m = conv(x, d.edge_index) if self.arch == "gcn" else conv(x, d.edge_index, d.edge_attr)
                    x = x + F.dropout(F.relu(norm(m)), p=args.dropout, training=self.training)
                g = global_add_pool(x, d.batch) if args.pool == "sum" else global_mean_pool(x, d.batch)
            return self.head(g)
    return Net()
def predict(model, loader, device, n_rows: int, n_out: int) -> np.ndarray:
    import torch
    out = np.full((n_rows, n_out), np.nan)
    model.eval()
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            out[d.rid.cpu().numpy()] = model(d).cpu().numpy()
    return out
def main():
    ap = argparse.ArgumentParser(description="PyG GNN baselines under the DeepTherm protocol")
    add_common_args(ap)
    ap.add_argument("--arch", choices=["gin", "gcn", "gat", "attentivefp"], required=True)
    ap.add_argument("--targets", choices=sorted(TARGET_COLS), default="all")
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4, help="GAT heads (hidden must be divisible by heads)")
    ap.add_argument("--timesteps", type=int, default=2, help="AttentiveFP readout steps")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pool", choices=["sum", "mean"], default="sum")
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--init_lr", type=float, default=1e-4)
    ap.add_argument("--max_lr", type=float, default=8e-4)
    ap.add_argument("--final_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    args = ap.parse_args()
    if args.arch == "gat" and args.hidden % args.heads:
        raise SystemExit("--hidden must be divisible by --heads for GAT")
    import torch
    from torch_geometric.loader import DataLoader
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    tag = f"gnn-{args.arch}_{args.targets}_L{args.layers}_d{args.hidden}"
    tag += "_attn" if args.arch == "attentivefp" else f"_{args.pool}"
    tag += args.tag_suffix
    data = load_dataset(args)
    tr, va, te, split_name, test_smarts = get_split(args, data["smiles"])
    save_dir = prepare_run_dir(args, tag, split_name)
    print(f"[{tag} | {split_name} | member {args.member} | seed {args.seed}] "
          f"{len(tr)} train / {len(va)} val / {len(te)} test | device {device}")
    cols = TARGET_COLS[args.targets]
    Y = targets_dt(data)
    mu = np.nanmean(Y[tr][:, cols], axis=0)
    sd = np.nanstd(Y[tr][:, cols], axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    Z = (Y[:, cols] - mu) / sd
    graphs = [mol_to_graph(m, Z[i], i) for i, m in enumerate(data["mols"])]
    in_dim, edge_dim = graphs[0].x.shape[1], graphs[0].edge_attr.shape[1]
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader([graphs[i] for i in tr], batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=gen)
    eval_loaders = {k: DataLoader([graphs[i] for i in idx], batch_size=256, shuffle=False)
                    for k, idx in (("train", tr), ("val", va), ("test", te))}
    model = build_model(args, in_dim, edge_dim, len(cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total, warm = args.max_epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    start, floor = args.init_lr / args.max_lr, args.final_lr / args.max_lr
    def lr_lambda(step):
        if step < warm:
            return start + (1.0 - start) * step / max(1, warm)
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    def val_score() -> float:
        p = predict(model, eval_loaders["val"], device, len(Y), len(cols))[va] * sd + mu
        t = Y[va][:, cols]
        maes = [np.nanmean(np.abs(p[:, j] - t[:, j])) for j in range(len(cols))]
        return float(np.nanmean(maes))
    best, best_state, best_epoch, t0 = float("inf"), None, 0, time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for d in train_loader:
            d = d.to(device)
            out = model(d)
            mask = torch.isfinite(d.y)
            if not bool(mask.any()):
                continue
            loss = torch.square(out[mask] - d.y[mask]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        score = val_score()
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == args.max_epochs:
            print(f"  epoch {epoch:>3d}  train loss {np.mean(losses):.4f}  val MAE {score:.3f}  "
                  f"(best {best:.3f} @ {best_epoch})")
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, save_dir / "best.pt")
    P = np.full_like(Y, np.nan)
    for k, idx in (("train", tr), ("val", va), ("test", te)):
        P[idx[:, None], np.asarray(cols)[None, :]] = (
            predict(model, eval_loaders[k], device, len(Y), len(cols))[idx] * sd + mu)
    Ysi, Psi = targets_si(data), dt_to_si(P)
    preds = {k: pred_dict(Psi[i], Ysi[i], data["X"][i], [data["smiles"][j] for j in i])
             for k, i in (("train", tr), ("val", va), ("test", te))}
    write_run(save_dir, args=args, tag=tag, model=f"GNN-{args.arch}", split_name=split_name,
              test_smarts=test_smarts, data=data, idx={"train": tr, "val": va, "test": te}, preds=preds,
              extra={"n_params": int(n_params), "epochs_trained": args.max_epochs, "best_epoch": best_epoch,
                     "early_stopped": False, "best_val_monitor": best,
                     "training_time_sec": round(time.time() - t0, 1),
                     "trained_targets": [TARGET_NAMES[c] for c in cols]})
if __name__ == "__main__":
    main()
