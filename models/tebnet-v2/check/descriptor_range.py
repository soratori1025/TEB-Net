"""Are the descriptors of long homologues outside the training range? (Reviewer #2, round 2, point 3)

The branches of TEB-Net receive standardised descriptors (z = (x - mean) / sd over the training split, clipped at
±10). For the homologous-series members outside the DeepTherm set this script reports, per branch (h, s, cp):
  * the largest |z| and the descriptor it belongs to,
  * how many routed descriptors lie above the training maximum (or below the minimum),
  * how many reach the clip (|z| > 10), where the network sees a constant instead of the true value.
Also prints the same numbers for the frozen-prior columns, which enter unstandardised and unclipped (linear).
No training; RDKit only (same featurizer as train.py).

Usage (repository root)
  python check/descriptor_range.py --csv models-baseline/deeptherm_data.csv --series check/homologous_series.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
def setup_paths():
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / "scripts"):
        if (p / "train.py").exists():
            sys.path.insert(0, str(p))
            return
    raise SystemExit("train.py not found next to this script or in ../scripts")
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--series", required=True)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--out_csv", default="descriptor_range.csv")
    args = ap.parse_args()
    setup_paths()
    from train import DEEPTHERM_T_GRID
    from rdkit import Chem
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.splits import deeptherm_split
    from tebnet.data.featurizer import V15_NAMES, v15_route_names, v15_base_names, compute_v15_vector
    ds = ThermochemDataset.from_csv(args.csv, T_grid=DEEPTHERM_T_GRID, smiles_col=args.smiles_col,
                                    physics_cache="auto", strict=True, compute_aux=True, aux_feature_set="v15",
                                    use_3d=True, compute_comp=False)
    smiles = [dp.smiles for dp in ds.datapoints]
    tr, _, _ = deeptherm_split(smiles, "random", seed=args.split_seed)
    X = np.stack([ds.datapoints[i].x_physic for i in tr]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-3] = 1.0
    lo, hi = X.min(0), X.max(0)
    col = {n: i for i, n in enumerate(V15_NAMES)}
    ser = pd.read_csv(args.series)
    out = ser[ser.in_deeptherm_set == 0].copy()
    Xs = np.stack([compute_v15_vector(Chem.MolFromSmiles(s), use_3d=True) for s in out.smiles]).astype(np.float64)
    rows = []
    groups = {f"route {r}": v15_route_names(r, "grouped", True) for r in ("h", "s", "cp")}
    groups.update({f"prior {r}": v15_base_names(r, "physics", True) for r in ("h", "s", "cp")})
    for k, (_, r) in enumerate(out.iterrows()):
        x = Xs[k]
        for gname, names in groups.items():
            idx = np.array([col[n] for n in names])
            z = (x[idx] - mu[idx]) / sd[idx]
            j = int(np.argmax(np.abs(z)))
            above = [names[i] for i in range(len(idx)) if x[idx[i]] > hi[idx[i]] + 1e-9]
            below = [names[i] for i in range(len(idx)) if x[idx[i]] < lo[idx[i]] - 1e-9]
            rows.append({"series": r.series, "n_carbon": int(r.n_carbon), "group": gname,
                         "max_abs_z": float(abs(z[j])), "max_feature": names[j],
                         "n_above_train_max": len(above), "n_below_train_min": len(below),
                         "n_clipped": int((np.abs(z) > args.clip).sum()),
                         "above": ";".join(above)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, float_format="%.2f")
    print(f"training split: random {args.split_seed}, {len(tr)} species\n")
    print("### Routed (network) descriptors of the members outside the set\n")
    print("| Series | n | branch | max |z| (feature) | above train max | clipped (|z| > "
          f"{args.clip:g}) |")
    print("|---|---|---|---|---|---|")
    for (s, n), g in df[df.group.str.startswith("route")].groupby(["series", "n_carbon"], sort=False):
        if n not in (out[out.series == s].n_carbon.min(), 12, 16):
            continue
        for _, r in g.iterrows():
            print(f"| {s} | {n} | {r.group[6:]} | {r.max_abs_z:.1f} ({r.max_feature}) | {r.n_above_train_max} | "
                  f"{r.n_clipped} |")
    print("\n### Descriptors above the training maximum (any member outside the set)\n")
    allabove = sorted({a for v in df.above for a in v.split(";") if a})
    for a in allabove:
        first = df[df.above.str.contains(rf"(^|;){a}(;|$)", regex=True)].n_carbon.min()
        print(f"  {a:<14s} train max {hi[col[a]]:.0f}   first exceeded at C{first}")
    print(f"\nwrote {args.out_csv}")
if __name__ == "__main__":
    main()
