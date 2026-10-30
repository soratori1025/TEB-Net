"""C6 follow-up: which radical centres does the prior's `rad_aryl` coefficient actually average over?

The fitted rad_aryl − rad_alkyl_p (+22 kcal/mol) is twice the phenyl ΔBDE (+11.8). One explanation is that
many "aryl" centres are not phenyl-type σ-radicals. This script classifies every atom that the featurizer
labels rad_aryl (radical_center_type: aromatic carbon carrying radical electrons), AT THE ATOM, by the ring
it sits in. Counting molecules with a SMARTS such as [c*] does not answer this: inside brackets `*` is a
wildcard, so [c*] matches any aromatic carbon, radical or not.

Categories (first match wins)
  not σ-type      the radical carbon carries an H or a third heavy neighbour, or >1 radical electron
                  (a π-delocalised or carbene-like centre that RDKit happens to perceive as aromatic)
  ring + exo C=O  the aromatic ring carries an exocyclic C=O (2-pyranone-, tropone-type rings, which RDKit
                  treats as aromatic)
  phenyl-type     6-membered all-carbon aromatic ring
  furanyl-type    5-membered aromatic ring containing O
  other           anything else (size and O count printed)
Each centre is also flagged if its ring is fused and if the molecule has more than one radical electron.

Usage
  PYTHONPATH=<tebnet root> python scripts/aryl_centres.py --csv models-baseline/deeptherm_data.csv [--list 5]
  PYTHONPATH=<tebnet root> python scripts/aryl_centres.py --csv models-baseline/deeptherm_data.csv --all_types
  (--all_types: centres per radical type for every rad_* coefficient; types with < 20 centres, like rad_aryl
   with 9, are left out of the C6 BDE correlation)
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
try:
    from tebnet.data.featurizer import radical_center_type
    def is_aryl_centre(a) -> bool:
        return a.GetNumRadicalElectrons() > 0 and radical_center_type(a) == "rad_aryl"
    SOURCE = "tebnet.data.featurizer.radical_center_type"
except ImportError:
    radical_center_type = None
    def is_aryl_centre(a) -> bool:
        return a.GetNumRadicalElectrons() > 0 and a.GetAtomicNum() == 6 and a.GetIsAromatic()
    SOURCE = "built-in copy of the rad_aryl rule (tebnet not on PYTHONPATH)"
MIN_CENTRES = 20
def count_all_types(smiles: pd.Series) -> None:
    """Centres and molecules per radical_center_type over the whole CSV: how many data points support each
    rad_* coefficient of the ΔH prior (the prior is fitted on the 81 % train split, so ~0.81 × these)."""
    if radical_center_type is None:
        raise SystemExit("--all_types needs tebnet on PYTHONPATH (tebnet.data.featurizer.radical_center_type)")
    centres = defaultdict(int)
    mols = defaultdict(int)
    for s in smiles.astype(str):
        m = Chem.MolFromSmiles(s.strip())
        if m is None:
            continue
        seen = set()
        for a in m.GetAtoms():
            if a.GetNumRadicalElectrons() == 0:
                continue
            key = radical_center_type(a) or "unclassified (not C or O)"
            centres[key] += 1
            seen.add(key)
        for key in seen:
            mols[key] += 1
    print(f"radical centres per type ({SOURCE}); types with < {MIN_CENTRES} centres are flagged\n")
    print("| Radical type | Centres | Molecules | ≈ in an 81 % train split | Used in the C6 correlation |")
    print("|---|---|---|---|---|")
    for key in sorted(centres, key=lambda k: -centres[k]):
        n = centres[key]
        print(f"| {key} | {n} | {mols[key]} | {round(0.81 * n)} | {'yes' if n >= MIN_CENTRES else 'no (too few)'} |")
def smiles_column(df: pd.DataFrame, name: str | None) -> str:
    if name:
        return name
    for c in df.columns:
        if str(c).strip().lower() in ("smiles", "smile", "canonical_smiles"):
            return c
    raise SystemExit(f"no SMILES column found; pass --smiles_col. Columns: {list(df.columns)}")
def classify(mol: Chem.Mol, a: Chem.Atom) -> tuple[str, bool]:
    """(category, fused) for one rad_aryl atom."""
    rings = [r for r in mol.GetRingInfo().AtomRings()
             if a.GetIdx() in r and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    fused = mol.GetRingInfo().NumAtomRings(a.GetIdx()) > 1
    if a.GetTotalNumHs() > 0 or a.GetDegree() != 2 or a.GetNumRadicalElectrons() != 1:
        return "not σ-type (H or substituent on the radical C, or >1 radical e)", fused
    if not rings:
        return "other (no all-aromatic ring)", fused
    ring = min(rings, key=len)
    atoms = [mol.GetAtomWithIdx(i) for i in ring]
    n_o = sum(x.GetAtomicNum() == 8 for x in atoms)
    exo_co = any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(x).GetAtomicNum() == 8
                 and b.GetOtherAtomIdx(x.GetIdx()) not in ring
                 for x in atoms for b in x.GetBonds())
    if exo_co:
        return "ring with exocyclic C=O (pyranone/tropone-type)", fused
    if len(ring) == 6 and n_o == 0:
        return "phenyl-type (6-ring, all C)", fused
    if len(ring) == 5 and n_o >= 1:
        return "furanyl-type (5-ring with O)", fused
    return f"other ({len(ring)}-ring, {n_o} O)", fused
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset the prior was fitted on (DeepTherm CSV)")
    ap.add_argument("--smiles_col", default=None)
    ap.add_argument("--list", type=int, default=3, help="example SMILES printed per category")
    ap.add_argument("--all_types", action="store_true",
                    help="instead: count centres of EVERY radical type, to put n beside each coefficient in C6")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    col = smiles_column(df, args.smiles_col)
    if args.all_types:
        count_all_types(df[col])
        return
    centres = defaultdict(int)
    mols = defaultdict(set)
    fused = defaultdict(int)
    polyrad = defaultdict(int)
    examples = defaultdict(list)
    n_mol_any = 0
    for i, s in enumerate(df[col].astype(str)):
        m = Chem.MolFromSmiles(s.strip())
        if m is None:
            continue
        hits = [a for a in m.GetAtoms() if is_aryl_centre(a)]
        if not hits:
            continue
        n_mol_any += 1
        n_rad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
        for a in hits:
            cat, fz = classify(m, a)
            centres[cat] += 1
            mols[cat].add(i)
            fused[cat] += fz
            polyrad[cat] += n_rad > 1
            if len(examples[cat]) < args.list and s not in examples[cat]:
                examples[cat].append(s)
    total = sum(centres.values())
    print(f"{args.csv}: rad_aryl centres classified with {SOURCE}")
    print(f"{n_mol_any} molecules carry at least one rad_aryl centre; {total} centres in total\n")
    print("| Category | Centres | Share | Molecules | In a fused ring | In a polyradical | Examples |")
    print("|---|---|---|---|---|---|---|")
    for cat in sorted(centres, key=lambda c: -centres[c]):
        n = centres[cat]
        print(f"| {cat} | {n} | {100 * n / max(total, 1):.1f} % | {len(mols[cat])} | {fused[cat]} | "
              f"{polyrad[cat]} | {', '.join(f'`{e}`' for e in examples[cat])} |")
if __name__ == "__main__":
    main()
