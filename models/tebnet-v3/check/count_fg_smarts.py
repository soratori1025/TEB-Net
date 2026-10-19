"""C7: how often do the held-out functional groups occur in QM9 (DeepTherm's pretraining set)?

If ether and epoxide are common in QM9 but enol is rare, a QM9-pretrained model has already seen the
"unseen" groups of the ether / epoxide hold-outs, which would explain why DeepTherm beats TEB-Net there
and not on enol. Matching is done exactly as the functional-group split does it
(Chem.MolFromSmiles(s).HasSubstructMatch(Chem.MolFromSmarts(smarts)), no explicit hydrogens).

QM9 SMILES: any CSV with a SMILES column, e.g. MoleculeNet's qm9.csv (133,885 rows, column "smiles").
Download it on a machine with internet access and pass its path.

Usage
  python scripts/count_fg_smarts.py --csv data/qm9.csv --ref_csv models-baseline/deeptherm_data.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# same SMARTS as tebnet.data.splits.FUNCTIONAL_GROUP_SMARTS (DeepTherm's data/functional_group.py)
GROUPS = {
    "Ether group": "[CX4][OX2][CX4]",
    "Epoxide group": "[C]1[O][C]1",
    "Enol group": "[CX3](=C[OX2H])",
}


def smiles_column(df: pd.DataFrame, name: str | None) -> str:
    if name:
        return name
    for c in df.columns:
        if str(c).strip().lower() in ("smiles", "smile", "canonical_smiles"):
            return c
    raise SystemExit(f"no SMILES column found; pass --smiles_col. Columns: {list(df.columns)}")


def count(path: str, col: str | None) -> tuple[int, int, dict[str, int]]:
    df = pd.read_csv(path)
    col = smiles_column(df, col)
    queries = {g: Chem.MolFromSmarts(s) for g, s in GROUPS.items()}
    hits = {g: 0 for g in GROUPS}
    n_ok = 0
    for s in df[col].astype(str):
        m = Chem.MolFromSmiles(s.strip())
        if m is None:
            continue
        n_ok += 1
        for g, q in queries.items():
            if m.HasSubstructMatch(q):
                hits[g] += 1
    return len(df), n_ok, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="QM9 (or any pretraining set) CSV with SMILES")
    ap.add_argument("--smiles_col", default=None)
    ap.add_argument("--ref_csv", default=None, help="optional: DeepTherm CSV, to print the hold-out sizes beside it")
    ap.add_argument("--ref_smiles_col", default=None)
    ap.add_argument("--all", action="store_true",
                    help="count all 27 SMARTS of tebnet.data.splits.FUNCTIONAL_GROUP_SMARTS (needs PYTHONPATH), "
                         "e.g. to check the peroxide group, whose DeepTherm hold-out error is low")
    args = ap.parse_args()
    if args.all:
        from tebnet.data.splits import FUNCTIONAL_GROUP_SMARTS
        GROUPS.clear()
        GROUPS.update(FUNCTIONAL_GROUP_SMARTS)

    n, n_ok, hits = count(args.csv, args.smiles_col)
    ref = count(args.ref_csv, args.ref_smiles_col) if args.ref_csv else None
    print(f"{args.csv}: {n} rows, {n_ok} parsed by RDKit" + (f" | reference {args.ref_csv}: {ref[1]} parsed" if ref else ""))
    head = "| Group | SMARTS | Pretraining set: count | share |" + (" Reference: count | share |" if ref else "")
    print(head)
    print("|---|---|---|---|" + ("---|---|" if ref else ""))
    for g, s in GROUPS.items():
        line = f"| {g} | `{s}` | {hits[g]:,} | {100 * hits[g] / max(n_ok, 1):.1f} % |"
        if ref:
            line += f" {ref[2][g]:,} | {100 * ref[2][g] / max(ref[1], 1):.1f} % |"
        print(line)


if __name__ == "__main__":
    main()