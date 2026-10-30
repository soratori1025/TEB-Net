"""Data splits.

(1) deeptherm_split(...)  — EXACT port of `split_indices` in the public DeepTherm code
    (github.com/tairan-w/DeepTherm, data/dataset.py; Wang et al., Energy and AI 24 (2026) 100677).
    Default fractions 0.81 / 0.09 / 0.10 (train / validation / test), as in that repo:
      random           : np.random.default_rng(seed).shuffle -> first 81 % train, next 9 % val, last 10 % test
      complexity       : stable sort by heavy-atom count -> smallest 81 % train, next 9 % val, largest 10 % test
      scaffold         : Murcko scaffolds (acyclic species keyed by their own SMILES), greedy group
                         assignment to the least-filled bucket, then each bucket shuffled
      functional-group : every species matching `test_smarts` is test; the rest shuffled into train/val
    Same CSV row order + same seed  =>  identical indices to the DeepTherm code, so the same test
    molecules as anyone running their repository (row order = Supplementary Data 1).

(2) ensemble_member_split(...) — deep-ensemble members that SHARE the test set: member 0 is the
    canonical split, member m > 0 re-shuffles train ∪ val (same sizes). (The public DeepTherm code
    gives every member its own random split, i.e. a different test set; evaluating that ensemble on
    one test set would leak, so we keep the test set fixed.)

(3) make_split(...) — the previous TEB-Net protocol (kept for backward compatibility).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
DEEPTHERM_FRACTIONS = (0.81, 0.09, 0.10)
FUNCTIONAL_GROUP_SMARTS = {
    "Vinyl carbon": "[CX3]=[CX2]",
    "Allenic group": "[C]=[C]=[C]",
    "Acetylenic carbon": "[CX2]#C",
    "Aromatic ring": "c1ccccc1",
    "Non-aromatic cyclic group": "[R2]",
    "Polycyclic aromatic rings": "c1ccc2ccccc2c1",
    "Methyl Radical": "[CH3*]",
    "Alkenyl Radical": "[C]=[C*]",
    "Alkynyl Radical": "[C]#[C*]",
    "Aryl Radical": "[c*]",
    "Hydroxyl group": "[OX2H]",
    "Ether group": "[CX4][OX2][CX4]",
    "Peroxide group": "[OX2][OX2]",
    "Epoxide group": "[C]1[O][C]1",
    "Carbonyl group": "[CX3]=O",
    "Ester group": "[CX3](=O)[OX2][CX4]",
    "Anhydride group": "[CX3](=O)[OX2][CX3](=O)",
    "Phenol group": "[cH][OX2H]",
    "Carboxyl group": "[CX3](=O)[OX2H]",
    "Peroxycarboxylic acid group": "[CX3](=O)[OX2][OX2H]",
    "Enol group": "[CX3](=C[OX2H])",
    "Aldehyde group": "[CX3H1](=O)",
    "Ketone group": "[CX3](=O)[#6]",
    "Hydroxyl radical": "[OH*]",
    "Peroxy radical": "[O][O*]",
    "Alkoxy radical": "[O*][CX4]",
    "Carbonyl radical": "[C](=O)*",
}
def deeptherm_split(smiles, strategy: str = "random", fractions=DEEPTHERM_FRACTIONS, seed: int = 0, test_smarts: str | None = None):
    smiles = list(smiles)
    strategy = {"size": "complexity", "functional_group": "functional-group"}.get(strategy, strategy)
    if len(smiles) < 3:
        raise ValueError("At least three molecules are required for train/validation/test splitting")
    if not np.isclose(sum(fractions), 1.0) or any(v <= 0 for v in fractions):
        raise ValueError("Split fractions must be positive and sum to one")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(smiles))
    def partition(ordered):
        n_train = min(len(ordered) - 2, max(1, int(len(ordered) * fractions[0])))
        n_valid = min(len(ordered) - n_train - 1, max(1, int(len(ordered) * fractions[1])))
        return ordered[:n_train], ordered[n_train:n_train + n_valid], ordered[n_train + n_valid:]
    if strategy == "random":
        rng.shuffle(indices)
        return partition(indices)
    if strategy == "complexity":
        ordered = np.asarray(sorted(indices, key=lambda i: Chem.MolFromSmiles(smiles[i]).GetNumHeavyAtoms()))
        return partition(ordered)
    if strategy == "functional-group":
        if not test_smarts:
            raise ValueError("--test_smarts (or --test_group) is required for a functional-group split")
        query = Chem.MolFromSmarts(test_smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS: {test_smarts!r}")
        test = np.asarray([i for i, s in enumerate(smiles) if Chem.MolFromSmiles(s).HasSubstructMatch(query)])
        test_set = set(test.tolist())
        remainder = np.asarray([i for i in indices if i not in test_set])
        if len(test) == 0 or len(remainder) < 2:
            raise ValueError("Functional-group split produced an empty test set or insufficient training data")
        rng.shuffle(remainder)
        valid_size = max(1, round(len(remainder) * fractions[1] / (fractions[0] + fractions[1])))
        return remainder[valid_size:], remainder[:valid_size], test
    if strategy == "scaffold":
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(smiles):
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=True)
            key = scaffold or f"acyclic:{Chem.MolToSmiles(Chem.MolFromSmiles(value))}"
            groups.setdefault(key, []).append(index)
        ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
        buckets: list[list[int]] = [[], [], []]
        limits = np.asarray(fractions) * len(smiles)
        for group in ordered:
            ratios = [len(buckets[i]) / max(limits[i], 1.0) for i in range(3)]
            buckets[int(np.argmin(ratios))].extend(group)
        for bucket in buckets:
            rng.shuffle(bucket)
        if any(not b for b in buckets):
            raise ValueError("Scaffold split produced an empty partition")
        return tuple(np.asarray(b, dtype=np.int64) for b in buckets)
    raise ValueError(f"Unknown split strategy: {strategy}")
def ensemble_member_split(train_idx, val_idx, test_idx, member: int, seed: int):
    train_idx, val_idx, test_idx = (np.asarray(a, dtype=np.int64) for a in (train_idx, val_idx, test_idx))
    if member == 0:
        return train_idx, val_idx, test_idx
    pool = np.concatenate([train_idx, val_idx])
    rng = np.random.default_rng(10_000 * (seed + 1) + member)
    rng.shuffle(pool)
    n_val = len(val_idx)
    return pool[n_val:], pool[:n_val], test_idx
def _num_heavy_atoms(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m is not None else 0
def _murcko(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""
def _carve_val(pool, seed, frac_val):
    rng = np.random.default_rng(seed + 10007)
    pool = list(pool)
    rng.shuffle(pool)
    n_val = int(frac_val * len(pool))
    return pool[n_val:], pool[:n_val]
def _test_random(n, seed, frac_test):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(frac_test * n)
    return idx[n_test:].tolist(), idx[:n_test].tolist()
def _test_size(smiles_list, frac_test):
    sizes = np.array([_num_heavy_atoms(s) for s in smiles_list])
    order = np.argsort(sizes, kind="stable")
    n_test = int(frac_test * len(smiles_list))
    return order[:-n_test].tolist(), order[-n_test:].tolist()
def _test_scaffold(smiles_list, seed, frac_test):
    groups = defaultdict(list)
    for i, s in enumerate(smiles_list):
        groups[_murcko(s)].append(i)
    gl = list(groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(gl)
    gl.sort(key=len, reverse=True)
    n = len(smiles_list)
    pool_target = n - int(frac_test * n)
    pool, test = [], []
    for g in gl:
        if len(pool) < pool_target:
            pool += g
        else:
            test += g
    return pool, test
def make_split(smiles_list, split_type="random", seed=0, frac_test=0.1, frac_val=0.1):
    if split_type == "random":
        pool, test = _test_random(len(smiles_list), seed, frac_test)
    elif split_type in ("size", "complexity"):
        pool, test = _test_size(smiles_list, frac_test)
    elif split_type == "scaffold":
        pool, test = _test_scaffold(smiles_list, seed, frac_test)
    else:
        raise ValueError(f"unknown split_type: {split_type!r} (use random | scaffold | size)")
    train, val = _carve_val(pool, seed, frac_val)
    return train, val, test
