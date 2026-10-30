"""TEB-Net v3: Benson-type group counts and ring-correction keys for the frozen additive prior.

Why: on WUDILY-CHO (45 % radicals, 75 % ring species) the v15 prior (bond-pair counts + ring sizes +
radical-centre types) leaves 5.3 kcal/mol for ΔfH and 2.4 cal/(mol K) for S. Group additivity (Benson)
resolves the same additive information one level finer: every heavy atom is described together with its
first neighbours, e.g.  Cs(Cs)(Cs)(H)2 ,  Cd(Cb)(H) ,  Os(CO)(Cs) ,  Cs.(Cd)(H)2  (radical centre).
Ring-strain corrections are keyed by ring size, unsaturation and ring oxygens (cyclopentene, oxetane,
furan, …) and by the way two rings are joined (ortho-fused, bridged, spiro), as in Benson's polycyclic
corrections. The prior stays linear in counts, so additivity and size extrapolation are unchanged.

The vocabulary is built from structures only (no targets), so building it on the whole dataset leaks no
label information; columns that are rare in a training fold are dropped by the prior fit (min_count).
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem
GROUPS_VERSION = "g1.0"
_BT = Chem.BondType
def atom_type(a: Chem.Atom) -> str:
    """Benson-like atom type (no radical flag)."""
    z = a.GetAtomicNum()
    if z == 6:
        if a.GetIsAromatic():
            n_ar = sum(1 for n in a.GetNeighbors() if n.GetIsAromatic())
            return "Cbf" if n_ar >= 3 else "Cb"
        n_d = n_t = 0
        d_to_o = False
        for b in a.GetBonds():
            bt = b.GetBondType()
            if bt == _BT.DOUBLE:
                n_d += 1
                d_to_o |= b.GetOtherAtom(a).GetAtomicNum() == 8
            elif bt == _BT.TRIPLE:
                n_t += 1
        if n_t:
            return "Ct"
        if n_d >= 2:
            return "Ca"
        if n_d == 1:
            return "CO" if d_to_o else "Cd"
        return "Cs"
    if z == 8:
        if a.GetIsAromatic():
            return "Ob"
        if any(b.GetBondType() == _BT.DOUBLE for b in a.GetBonds()):
            return "Od"
        return "Os"
    if z == 1:
        return "H"
    return a.GetSymbol()
def group_keys(mol: Chem.Mol) -> Counter:
    """Counter of group / ring-correction keys of one molecule."""
    c: Counter = Counter()
    types = {a.GetIdx(): atom_type(a) for a in mol.GetAtoms()}
    for a in mol.GetAtoms():
        t = types[a.GetIdx()]
        if t in ("Od", "H"):
            continue
        rad = a.GetNumRadicalElectrons()
        chg = a.GetFormalCharge()
        centre = t + ("." * rad) + ("+" if chg > 0 else "-" if chg < 0 else "")
        nbs = sorted(types[n.GetIdx()] for n in a.GetNeighbors() if n.GetAtomicNum() > 1)
        n_h = a.GetTotalNumHs(includeNeighbors=True)
        key = "g:" + centre + "".join(f"({x})" for x in nbs) + (f"(H){n_h}" if n_h else "")
        c[key] += 1
    from tebnet.data.featurizer import sssr_bond_rings
    rings = sssr_bond_rings(mol)
    ring_atoms = []
    for r in rings:
        atoms = set()
        n_d = n_ar = 0
        for bi in r:
            b = mol.GetBondWithIdx(bi)
            atoms.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            if b.GetIsAromatic():
                n_ar += 1
            elif b.GetBondType() == _BT.DOUBLE:
                n_d += 1
        ring_atoms.append(atoms)
        n_o = sum(1 for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
        n_exo = sum(1 for i in atoms for b in mol.GetAtomWithIdx(i).GetBonds()
                    if b.GetBondType() == _BT.DOUBLE and not b.IsInRing())
        n_rad = sum(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)
        size = len(r) if len(r) <= 8 else 9
        kind = "ar" if n_ar == len(r) else f"d{n_d}"
        c[f"r:{size}_{kind}_o{n_o}"] += 1
        if n_exo:
            c[f"r:{size}_exo{min(n_exo, 2)}"] += 1
        if n_rad:
            c[f"r:{size}_rad"] += 1
    bond_sets = [set(r) for r in rings]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared_b = len(bond_sets[i] & bond_sets[j])
            shared_a = len(ring_atoms[i] & ring_atoms[j])
            if shared_a == 0:
                continue
            a_, b_ = sorted((min(len(rings[i]), 9), min(len(rings[j]), 9)))
            if shared_b == 0:
                c[f"r:spiro_{a_}_{b_}"] += 1
            elif shared_b == 1:
                c[f"r:fused_{a_}_{b_}"] += 1
            else:
                c[f"r:bridged_{a_}_{b_}"] += 1
    return c
def build_vocab(counters: list[Counter], min_df: int = 3) -> list[str]:
    """Keys present in at least `min_df` molecules + one 'other' bucket per centre type."""
    df: Counter = Counter()
    for c in counters:
        df.update(set(c))
    keep = sorted(k for k, v in df.items() if v >= min_df)
    others = sorted({_other_key(k) for k in df if df[k] < min_df})
    return keep + others
def _other_key(k: str) -> str:
    if k.startswith("g:"):
        centre = k[2:].split("(")[0]
        return f"g:other_{centre}"
    return "r:other"
def counts_matrix(counters: list[Counter], vocab: list[str]) -> np.ndarray:
    idx = {k: i for i, k in enumerate(vocab)}
    X = np.zeros((len(counters), len(vocab)), dtype=np.float32)
    for r, c in enumerate(counters):
        for k, v in c.items():
            j = idx.get(k)
            if j is None:
                j = idx.get(_other_key(k))
            if j is not None:
                X[r, j] += v
    return X
def groups_cache_path(csv_path) -> Path:
    p = Path(csv_path)
    return p.with_name(f"{p.stem}.groups_{GROUPS_VERSION}.npz")
def load_or_compute_groups(csv_path, smiles: list[str], min_df: int = 3) -> tuple[np.ndarray, list[str]]:
    """(counts [n, n_vocab], vocab) for `smiles`, cached next to the CSV."""
    path = groups_cache_path(csv_path)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["smiles"]) == list(smiles) and int(z["min_df"]) == min_df:
            return z["X"], list(z["vocab"])
    counters = [group_keys(Chem.MolFromSmiles(s)) for s in smiles]
    vocab = build_vocab(counters, min_df)
    X = counts_matrix(counters, vocab)
    np.savez(path, X=X, vocab=np.array(vocab, dtype=object), smiles=np.array(smiles, dtype=object),
             min_df=min_df, version=GROUPS_VERSION)
    return X, vocab
