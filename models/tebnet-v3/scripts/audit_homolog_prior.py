#!/usr/bin/env python
"""Audit the CH2-homolog prior: how many pairs and site classes does it actually use?

A draft of the Methods section quoted "12 insertion-site classes / 1,234 pairs" for the
DeepTherm random split, whereas ``scripts/train.py`` reports 4 classes / 967 pairs.  This
script reproduces both numbers from the data so the discrepancy can be attributed rather
than guessed at.  Result: the *pair* count is fully explained (1,248 raw insertions minus
the 14 ``O4|O4`` peroxide insertions the filter rejects = 1,234), but the *class* count is
not -- no labelling scheme tried here yields 12 (see section 3).  The 1,234 figure is
therefore the unfiltered pair count with peroxides removed, not the count the shipped
builder uses, and the "12 classes" figure is not reproducible.  Quote section 4.

It enumerates CH2 insertions on the seed-0 training split under the definitions that
separate the two figures:

  * insertion filter on/off (:func:`tebnet.data.homologs.insertion_allowed`).  The filter
    removes insertions that would land *inside* a functional group (peroxide, ester,
    carbonyl, ...) rather than extend the chain.
  * site-class granularity (coarse element+pi label, RDKit hybridization, element +
    hydrogen count, ...).
  * scope (training split, train+val, the full data set).

Usage
-----
    PYTHONPATH=. python scripts/audit_homolog_prior.py [--csv deeptherm_data.csv]
                                                       [--split_seed 0]

The printed table is what the "Methods" paragraph should quote; nothing here trains.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tebnet.data import homologs as H            # noqa: E402
from tebnet.data.dataset import ThermochemDataset  # noqa: E402
from tebnet.data.splits import deeptherm_split, DEEPTHERM_FRACTIONS  # noqa: E402

# RDKit chatters about every SMILES it cannot kekulise; irrelevant to the counts.
RDLogger.DisableLog("rdApp.*")


# ────────────────────────────── labelling schemes ──────────────────────────────
def lab_pi(atom: Chem.Atom) -> str:
    """Shipped scheme: element + pi state (``C4``, ``C3``, ``C3b``, ``O4``)."""
    return H.site_class(atom)


def lab_hybridization(atom: Chem.Atom) -> str:
    """RDKit hybridization, e.g. ``C3`` sp3, ``C2`` sp2, ``CP`` sp."""
    return f"{atom.GetSymbol()}{atom.GetHybridization().name[-1]}"


def lab_hcount(atom: Chem.Atom) -> str:
    """Element + number of attached hydrogens (finer than the pi label)."""
    return f"{atom.GetSymbol()}{atom.GetTotalNumHs()}"


def lab_pi_ring(atom: Chem.Atom) -> str:
    """Pi state split by ring membership."""
    return f"{lab_pi(atom)}{'R' if atom.IsInRing() else 'A'}"


SCHEMES = {
    "element+pi (shipped)": lab_pi,
    "hybridization": lab_hybridization,
    "element+Hcount": lab_hcount,
    "element+pi+ring": lab_pi_ring,
}


# ────────────────────────────── pair enumeration ──────────────────────────────
def enumerate_pairs(smiles: list[str], label_fn, use_filter: bool):
    """All CH2 insertion pairs inside ``smiles``, as ``{(i, j): class_label}``.

    ``i`` is the anchor (shorter molecule), ``j`` the product.  Mirrors
    :func:`tebnet.data.homologs.build_homolog_pairs` but keeps the flanking atoms so that
    alternative labelling schemes can be applied without re-running RDKit.
    """
    key_to_idx: dict[str, int] = {}
    mols: list[Chem.Mol | None] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        mols.append(mol)
        if mol is not None:
            key_to_idx.setdefault(Chem.MolToSmiles(mol, isomericSmiles=False), i)

    pairs: dict[tuple[int, int], tuple[Chem.Atom, Chem.Atom]] = {}
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        seen: set[str] = set()
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
                continue
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if use_filter and not H.insertion_allowed(a1, a2):
                continue
            rw = Chem.RWMol(mol)
            new_idx = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(a1.GetIdx(), new_idx, Chem.BondType.SINGLE)
            rw.AddBond(new_idx, a2.GetIdx(), Chem.BondType.SINGLE)
            rw.RemoveBond(a1.GetIdx(), a2.GetIdx())
            try:
                prod = rw.GetMol()
                Chem.SanitizeMol(prod)
            except Exception:                       # valence / kekulisation failure
                continue
            prod_smi = Chem.MolToSmiles(prod, isomericSmiles=False)
            if prod_smi in seen:                    # symmetry-equivalent sites
                continue
            seen.add(prod_smi)
            j = key_to_idx.get(prod_smi)
            if j is None or j == i:                 # product absent from the split
                continue
            pairs.setdefault((i, j), (a1, a2))

    return {key: "|".join(sorted([label_fn(a1), label_fn(a2)]))
            for key, (a1, a2) in pairs.items()}


def show(name: str, pairs: dict, target: tuple[int, int] | None = None) -> None:
    c = Counter(pairs.values())
    tag = ""
    if target is not None and (len(pairs), len(c)) == target:
        tag = "   <== matches the draft figure"
    print(f"  {name:22s} pairs={len(pairs):5d}  classes={len(c):3d}{tag}")
    if 0 < len(c) <= 12:
        print("      " + ", ".join(f"{k}:{v}" for k, v in
                                   sorted(c.items(), key=lambda kv: -kv[1])))


# ────────────────────────────── main ──────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="deeptherm_data.csv", help="DeepTherm data CSV")
    ap.add_argument("--split_seed", type=int, default=0, help="DeepTherm split seed")
    ap.add_argument("--split_fractions", type=float, nargs=3,
                    default=list(DEEPTHERM_FRACTIONS))
    args = ap.parse_args()

    print(f"data set: {args.csv}")
    full = ThermochemDataset.from_csv(
        args.csv, strict=True, compute_aux=False, compute_comp=False,
    )
    records = full.records
    smiles_all = [r["smiles"] for r in records]
    print(f"  {len(records)} species")

    tr, va, te = deeptherm_split(smiles_all, "random", tuple(args.split_fractions),
                                 args.split_seed)
    train_key = f"train (seed {args.split_seed})"
    scopes = {
        train_key: [smiles_all[i] for i in tr],
        "train+val": [smiles_all[i] for i in list(tr) + list(va)],
        "full data set": smiles_all,
    }
    print(f"  train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    print("\n1. Insertion filter (shipped element+pi labels)")
    for scope, smis in scopes.items():
        for use_filter in (False, True):
            kind = "filtered" if use_filter else "raw     "
            show(f"{scope} [{kind}]", enumerate_pairs(smis, lab_pi, use_filter),
                 target=(1234, 12))

    # The locked figure is a *pair* count, so test the decomposition directly: raw
    # insertions minus the classes the filter rejects inside a functional group.
    print("\n2. Decomposing the 1,234-pair figure (train split, raw insertions)")
    raw = enumerate_pairs(scopes[train_key], lab_pi, False)
    print(f"  raw insertions                     pairs={len(raw):5d}")
    raw_counts = Counter(raw.values())
    for cls in sorted(raw_counts, key=lambda k: -raw_counts[k]):
        after = {k: v for k, v in raw.items() if v != cls}
        print(f"  minus {cls:12s} (n={raw_counts[cls]:4d})          pairs={len(after):5d}"
              f"  classes={len(Counter(after.values())):3d}")

    print("\n3. Site-class granularity on the filtered pair set")
    for name, fn in SCHEMES.items():
        show(name, enumerate_pairs(scopes[train_key], fn, True), target=(1234, 12))

    print("\n4. What the shipped builder reports (authoritative)")
    table = H.build_homolog_pairs(ThermochemDataset([records[i] for i in tr]))
    print(f"  {table.summary()}")
    print(f"  pairs with complete targets: {len(table)}")
    inc, tol = table.increments, table.tolerances
    per_class = table.class_counts()
    for k, cls in enumerate(table.classes):
        print(f"    {cls:8s} n={per_class[cls]:4d}  "
              f"dHf={inc[k, 0] / 4184.0:+7.2f} kcal/mol  "
              f"dS={inc[k, 1] / 4.184:+6.2f} cal/mol/K  "
              f"dCp={inc[k, 2] / 4.184:+6.2f} cal/mol/K  "
              f"tol={tol[k, 0] / 4184.0:.2f}/{tol[k, 1] / 4.184:.2f}/{tol[k, 2] / 4.184:.2f}")

    print("\nNote: scripts/train.py logs exactly the section-4 numbers, so any figure "
          "quoted in the manuscript must match them unless the filter or the labels change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

