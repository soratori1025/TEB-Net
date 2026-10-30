"""Plain chemprop D-MPNN with mean vs sum aggregation on DeepTherm random split 0 (Reviewer #2, point 2).

Question: is the CH2 increment that TEB-Net keeps on long homologous series a property of the frozen prior, or
would any sum-aggregated message-passing network keep it? This trains the backbone of TEB-Net WITHOUT prior,
descriptors or NASA layer (chemprop v2: BondMessagePassing, depth 5, hidden 300, batch norm, MSE loss,
100 epochs, batch 32), once per aggregation, on the same split, and predicts the homologous series.

  --graph tebnet   (default) TEB-Net's atom featurizer (make_atom_featurizer('thermo')) and CleanBondFeaturizer,
                   so the only differences from TEB-Net are the prior, the descriptors and the output layer
  --graph chemprop chemprop's default featurizers

Outputs per run: <out_dir>/agg_<agg>_<targets>_s<seed>/{best checkpoint, test_metrics.json,
preds_homologous.csv (smiles, H_kcal / S_cal / Cp300_cal)}; feed the CSVs to check/homologous_controls.py.
The test MAE on the 501 test molecules is printed as a sanity check (DeepTherm's D-MPNN baseline: ≈1.38 kcal/mol).

Usage (from the repository root; ~15 min per run on one A100)
  python check/chemprop_agg_control.py --csv models-baseline/deeptherm_data.csv \
      --series check/homologous_series.csv --agg mean sum --targets H --out_dir runs_agg
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
TARGET_UNITS = {"H": "kcal", "S": "cal", "Cp300": "cal"}
def setup_paths():
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / "scripts"):
        if (p / "train.py").exists():
            sys.path.insert(0, str(p))
            return
    raise SystemExit("train.py not found next to this script or in ../scripts")
def import_split():
    for mod in ("tebnet.data.splits", "tebnet.splits", "train"):
        try:
            return __import__(mod, fromlist=["deeptherm_split"]).deeptherm_split
        except (ImportError, AttributeError):
            continue
    raise SystemExit("deeptherm_split not found (tebnet.data.splits / train.py)")
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--series", required=True, help="homologous_series.csv")
    ap.add_argument("--agg", nargs="+", default=["mean", "sum"], choices=["mean", "sum", "norm"])
    ap.add_argument("--targets", nargs="+", default=["H"], choices=list(TARGET_UNITS))
    ap.add_argument("--graph", choices=["tebnet", "chemprop"], default="tebnet")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--ffn_hidden", type=int, default=300)
    ap.add_argument("--ffn_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_agg")
    args = ap.parse_args()
    setup_paths()
    from train import DEEPTHERM_T_GRID, make_atom_featurizer
    import torch
    import lightning.pytorch as pl
    from rdkit import Chem
    from chemprop import data, featurizers, models, nn
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.featurizer import CleanBondFeaturizer
    deeptherm_split = import_split()
    t_grid = np.asarray(DEEPTHERM_T_GRID, dtype=np.float64)
    ds = ThermochemDataset.from_csv(args.csv, T_grid=t_grid, smiles_col=args.smiles_col, physics_cache=None,
                                    strict=True, compute_aux=False, compute_comp=False)
    dps = ds.datapoints
    smiles = [dp.smiles for dp in dps]
    j300 = int(np.argmin(np.abs(t_grid - 300.0)))
    cols = {"H": np.array([dp.dh_f_298 for dp in dps]) * J_TO_KCAL,
            "S": np.array([dp.s_298 for dp in dps]) * J_TO_CAL,
            "Cp300": np.array([dp.cp_grid[j300] for dp in dps]) * J_TO_CAL}
    Y = np.stack([cols[t] for t in args.targets], axis=1)
    tr, va, te = (np.asarray(a) for a in deeptherm_split(smiles, "random", seed=args.split_seed))
    print(f"{Path(args.csv).name}: {len(dps)} species | split {args.split_seed}: "
          f"{len(tr)} / {len(va)} / {len(te)} | targets {args.targets}")
    if args.graph == "tebnet":
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=make_atom_featurizer("thermo"),
                                                            bond_featurizer=CleanBondFeaturizer())
    else:
        feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    def dataset(idx, with_y=True):
        pts = [data.MoleculeDatapoint(mol=dps[i].mol, y=Y[i] if with_y else None) for i in idx]
        return data.MoleculeDataset(pts, featurizer=feat)
    ser = pd.read_csv(args.series)
    ser_smi = ser["smiles"].astype(str).str.strip().tolist()
    ser_mols = [Chem.MolFromSmiles(s) for s in ser_smi]
    ser_ds = data.MoleculeDataset([data.MoleculeDatapoint(mol=m) for m in ser_mols], featurizer=feat)
    out_root = Path(args.out_dir)
    tname = "-".join(args.targets)
    for agg_name in args.agg:
        run = out_root / (f"agg_{agg_name}_{tname}_s{args.seed}" + ("" if args.graph == "tebnet" else "_cpfeat"))
        run.mkdir(parents=True, exist_ok=True)
        pl.seed_everything(args.seed, workers=True)
        train_ds, val_ds, test_ds = dataset(tr), dataset(va), dataset(te)
        scaler = train_ds.normalize_targets()
        val_ds.normalize_targets(scaler)
        mp = nn.BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=args.hidden, depth=args.depth)
        agg = {"mean": nn.MeanAggregation, "sum": nn.SumAggregation, "norm": nn.NormAggregation}[agg_name]()
        ffn = nn.RegressionFFN(n_tasks=len(args.targets), input_dim=mp.output_dim, hidden_dim=args.ffn_hidden,
                               n_layers=args.ffn_layers,
                               output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
        model = models.MPNN(mp, agg, ffn, batch_norm=True)
        ckpt = pl.callbacks.ModelCheckpoint(dirpath=run, filename="best", monitor="val_loss", mode="min",
                                            save_top_k=1)
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices=1, logger=False,
                             enable_progress_bar=False, enable_model_summary=False, callbacks=[ckpt])
        dl = dict(batch_size=args.batch_size, num_workers=args.num_workers)
        trainer.fit(model, data.build_dataloader(train_ds, seed=args.seed, **dl),
                    data.build_dataloader(val_ds, shuffle=False, **dl))
        def predict(dset):
            out = trainer.predict(model, data.build_dataloader(dset, shuffle=False, **dl),
                                  ckpt_path=ckpt.best_model_path)
            return torch.cat(out).double().cpu().numpy().reshape(len(dset), len(args.targets))
        p_te = predict(test_ds)
        metrics = {t: {"test_mae": float(np.abs(p_te[:, k] - Y[te, k]).mean()),
                       "test_rmse": float(np.sqrt(np.mean((p_te[:, k] - Y[te, k]) ** 2))),
                       "unit": TARGET_UNITS[t]} for k, t in enumerate(args.targets)}
        metrics["args"] = vars(args) | {"agg": agg_name}
        (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        p_s = predict(ser_ds)
        out = pd.DataFrame({"smiles": ser_smi})
        for k, t in enumerate(args.targets):
            out[f"{t}_{TARGET_UNITS[t]}"] = p_s[:, k]
        out.to_csv(run / "preds_homologous.csv", index=False, float_format="%.3f")
        print(f"[{agg_name}] test " + ", ".join(f"{t} MAE {metrics[t]['test_mae']:.2f}" for t in args.targets)
              + f"  ->  {run / 'preds_homologous.csv'}")
if __name__ == "__main__":
    main()
