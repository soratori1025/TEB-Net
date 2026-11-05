"""Export the DeepTherm splits (same indices as scripts/train.py) as a chemprop v2 --splits-file,
so the plain chemprop D-MPNN baseline can be re-run on exactly the same train/val/test molecules.

  python scripts/export_splits.py --csv deeptherm_data.csv --split_type random --split_seeds 0 1 2 3 4
  -> splits_random.json  (one replicate per split seed) + the chemprop command to run
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet")
from tebnet.data.splits import deeptherm_split, FUNCTIONAL_GROUP_SMARTS, DEEPTHERM_FRACTIONS  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--split_type", choices=["random", "complexity", "scaffold", "functional_group"], default="random")
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split_fractions", type=float, nargs=3, default=list(DEEPTHERM_FRACTIONS))
    ap.add_argument("--test_smarts", default=None)
    ap.add_argument("--test_group", default=None, choices=sorted(FUNCTIONAL_GROUP_SMARTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    col = next((c for c in df.columns if str(c).strip().lower() == args.smiles_col.lower()), None)
    if col is None:
        raise SystemExit(f"no SMILES column '{args.smiles_col}' in {list(df.columns)}")
    if df[col].isna().any():
        raise SystemExit("CSV has empty SMILES rows: indices would not match chemprop's row numbering; drop them first")
    smiles = df[col].astype(str).str.strip().tolist()
    smarts = args.test_smarts or (FUNCTIONAL_GROUP_SMARTS[args.test_group] if args.test_group else None)

    splits = []
    for s in args.split_seeds:
        tr, va, te = deeptherm_split(smiles, args.split_type, tuple(args.split_fractions), s, smarts)
        splits.append({"train": [int(i) for i in tr], "val": [int(i) for i in va], "test": [int(i) for i in te]})
        print(f"split seed {s}: {len(tr)} / {len(va)} / {len(te)}")
    out = Path(args.out or f"splits_{args.split_type}.json")
    out.write_text(json.dumps(splits))
    targets = [c for c in df.columns if c != col and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\nwrote {out} ({len(splits)} replicate(s)). Plain D-MPNN baseline on the same splits:\n")
    print("chemprop train --data-path " + args.csv + " --task-type regression --smiles-columns " + col
          + " \\\n  --target-columns " + " ".join(f"'{t}'" for t in targets)
          + f" \\\n  --splits-file {out} --epochs 100 --save-dir chemprop_dmpnn_{args.split_type}")


if __name__ == "__main__":
    main()