"""CH2-homolog consistency constraints (physics-informed training prior).

Two species are CH2 homologs when one is obtained from the other by inserting a single
methylene group into an acyclic single bond.  Only the chain length changes, so the
*difference* of their thermochemistry is close to a transferable increment.  On the
DeepTherm training split (4,047 species, seed 0: 967 distinct pairs, 603 anchors,
4 insertion-site classes) the measured increments are

    dHf(298)  = -5.09 kcal/mol    class medians  -5.63 (C3|C4) ... -3.30 (C4|O4)
    S(298)    = +7.70 cal/mol/K   class medians  +6.83 ... +8.25
    Cp(300)   = +5.40 cal/mol/K   class medians  +5.16 ... +5.44

(values after shrinkage towards the global increment, ``n_shrink=5``; the pooled
tolerances are 1.23 kcal/mol, 3.19 and 1.47 cal/mol/K).  The sign and magnitude are
universal, the insertion-site class sets the value (the spread of the class medians,
~1.4 cal/mol/K for S, is a sizeable fraction of the within-class scatter,
2.9 cal/mol/K), which is why the constraint below is class-conditioned rather than a
single constant.  The 4 surviving classes are the abundant ones: C4|C4 (chain
extension at sp3 carbon, n=537), C4|O4 (extension onto an sp3 oxygen, n=240),
C3|C4 (extension adjacent to a ring, n=189) and C3b|C4 (n=1).  The rarer classes of the
first pass (O3|C3, O3|O3, ...) had a handful of pairs each and were merged away by the
insertion filter, which is the intended behaviour: a class with three examples cannot
support its own increment.

Why a regulariser at all: per-molecule regression trains the two members of a pair
independently, so nothing in the objective ties them together.  The failure modes this
targets are exactly the ones a per-molecule loss cannot see — entropy that drifts with
chain length on extrapolation, and ensemble members whose dHf slope with carbon number
is unphysical.

This module only *builds* the constraints (RDKit + the training targets).  The tensor
plumbing lives in :class:`tebnet.data.collate.HomologPairCollater`, the penalty in
:class:`tebnet.nn.loss.HomologConsistencyLoss`.

Design notes
------------
* Pairs are built from the TRAINING split only, so the prior never sees a validation or
  test molecule.  ``build_homolog_pairs`` therefore expects the training dataset.
* Increments are estimated from the training targets with shrinkage towards the global
  increment (``n0`` pseudo-pairs), so a class with one or two examples cannot inject
  noise into the prior.
* Each increment carries a tolerance: the pooled within-class robust scale, floored at
  ``tol_floor * s_global``.  The penalty is the deviation measured in tolerances, so it
  is a soft, self-normalising constraint instead of a hard equality.
* Call this AFTER any per-datapoint feature augmentation (entropy-v2 columns, ECFP bits)
  has been applied to ``dp.x_physic``: the partner items are snapshotted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from rdkit import Chem
import torch

__all__ = [
    "HomologPairTable",
    "HomologPairDataset",
    "insert_ch2",
    "canonical_smiles",
    "site_class",
    "build_homolog_pairs",
]

# Increment is defined in the direction  partner = anchor + CH2, i.e. longer - shorter.
# All quantities are SI, matching ThermochemDataset (J/mol and J/(mol K)).
PROPERTIES: tuple[str, ...] = ("h", "s", "cp")

# Absolute floor on a tolerance, in SI, in the order of PROPERTIES: 0.05 kcal/mol and
# 0.1 cal/(mol K).  The pooled scale is a *measured* spread, so it is exactly zero when a
# split's pairs happen to agree perfectly (a tiny or synthetic split, or a class whose
# members are duplicated).  A zero tolerance would turn a soft constraint into a hard
# equality and the normalised residual into ~1e9, so the tolerance is floored at the
# precision of the tabulated data instead.  Never binding on the full DeepTherm split,
# where the pooled scales are 1.23 kcal/mol, 3.19 and 1.47 cal/(mol K).
TOL_MIN: np.ndarray = np.array([0.05 * 4184.0, 0.1 * 4.184, 0.1 * 4.184])


# ────────────────────────────── site classification ──────────────────────────────
def _in_pi_system(atom: Chem.Atom) -> bool:
    """True if the atom carries a multiple bond or is aromatic."""
    if atom.GetIsAromatic():
        return True
    return any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
               for b in atom.GetBonds())


def _conjugated_heteroatom(atom: Chem.Atom) -> bool:
    """True for a heteroatom that is part of, or conjugated to, a pi system.

    ``GetHybridization`` is not usable as a test here: RDKit reports the alkoxy oxygen of
    an ester as SP2 because it is conjugated with the carbonyl, and the same oxygen must
    also be excluded.  Testing for a multiple bond on the atom or on a neighbour catches
    both the carbonyl oxygen and the alkoxy oxygen, i.e. every bond *inside* an ester,
    amide, acid or anhydride group.
    """
    if atom.GetAtomicNum() == 6:
        return False
    if _in_pi_system(atom):
        return True
    return any(_in_pi_system(nb) for nb in atom.GetNeighbors())


def insertion_allowed(a1: Chem.Atom, a2: Chem.Atom) -> bool:
    """Whether CH2 may be inserted into the bond ``a1-a2`` and still be a homologation.

    The test is whether the inserted methylene ends up *in the chain* rather than *inside
    a functional group*.  Measured on DeepTherm, the excluded cases are exactly the ones
    whose increment is not a chain-length increment:

    * ``C(=O)-O`` / ``C(=O)-N`` (ester, amide, acid, anhydride): the group is destroyed,
      dHf increment collapses to ~0 instead of ~-5 kcal/mol,
    * ``C=C-O`` / ``C=C-N`` (enol ether, enamine): conjugation is relocated,
    * two unsaturated atoms (diene, diketone, biphenyl): conjugation is broken,
    * ``O-O``: the product is a peroxide, dHf increment ~-59 kcal/mol.

    Allowed, by contrast, are a saturated C-C or C-heteroatom bond, and a bond between a
    saturated atom and one unsaturated carbon: ``C(=O)-CH2R -> C(=O)-CH2-CH2R`` and
    ``Ar-CH3 -> Ar-CH2-CH3`` are both genuine chain extensions with a normal increment.
    """
    if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
        return False
    if a1.GetAtomicNum() == 8 and a2.GetAtomicNum() == 8:
        return False
    if a1.GetNumRadicalElectrons() or a2.GetNumRadicalElectrons():
        return False
    if _conjugated_heteroatom(a1) or _conjugated_heteroatom(a2):
        return False
    if _in_pi_system(a1) and _in_pi_system(a2):
        return False
    return True


def site_class(atom: Chem.Atom) -> str:
    """Coarse local environment of an atom: element + pi state.

    "C4" sp3 carbon, "C3" sp2/sp carbon, "C3b" aromatic carbon, "O4" sp3 oxygen, ...
    Derived from :func:`_in_pi_system` rather than ``GetHybridization`` so that the label
    agrees with the insertion filter: an ester oxygen is reported as SP2 by RDKit but is
    saturated for our purposes, and its bonds are never inserted into.

    Deliberately coarse: a fine-grained SMARTS environment would fragment the ~1,000
    pairs into classes too small to estimate an increment from.
    """
    if atom.GetIsAromatic():
        return f"{atom.GetSymbol()}3b"
    return f"{atom.GetSymbol()}{'3' if _in_pi_system(atom) else '4'}"


def canonical_smiles(smiles: str) -> str | None:
    """Non-isomeric canonical SMILES, used as the molecule key."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


# ────────────────────────────── pair enumeration ──────────────────────────────
def insert_ch2(mol: Chem.Mol) -> list[tuple[str, str]]:
    """All single CH2 insertions into acyclic single bonds of ``mol``.

    Returns a list of ``(product_smiles, class_label)`` with the product in canonical
    non-isomeric SMILES and ``class_label`` the sorted pair of flanking site classes
    (e.g. ``"C4|O4"``), which is the key the increment table is indexed by.

    A bond is skipped when the inserted methylene would land inside a functional group
    rather than in the chain -- see :func:`insertion_allowed` for the rule and the
    measured increments that justify it.  Saturated chain bonds, including those to a
    terminal methyl or to a ring carbon, are always eligible, which is what makes chain
    extension work.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if not insertion_allowed(a1, a2):
            continue
        rw = Chem.RWMol(mol)
        new_idx = rw.AddAtom(Chem.Atom(6))          # CH2: two implicit H by default
        rw.AddBond(a1.GetIdx(), new_idx, Chem.BondType.SINGLE)
        rw.AddBond(new_idx, a2.GetIdx(), Chem.BondType.SINGLE)
        rw.RemoveBond(a1.GetIdx(), a2.GetIdx())
        try:
            prod = rw.GetMol()
            Chem.SanitizeMol(prod)
        except Exception:                            # valence / kekulisation failure
            continue
        smi = Chem.MolToSmiles(prod, isomericSmiles=False)
        if smi in seen:                              # symmetry-equivalent sites
            continue
        seen.add(smi)
        label = "|".join(sorted([site_class(a1), site_class(a2)]))
        out.append((smi, label))
    return out


# ────────────────────────────── increment table ──────────────────────────────
@dataclass
class HomologPairTable:
    """CH2-homolog constraints for one split, in SI units (J/mol, J/(mol K)).

    Attributes
    ----------
    anchor_idx, partner_idx
        Indices into the dataset the table was built from. ``partner`` is always the
        longer molecule, so the increment is ``prop(partner) - prop(anchor)``.
    cls_idx
        Insertion-site class of each pair, indexing the rows of ``increments``.
    increments
        ``(n_classes, 3)`` target increments, in the order of :data:`PROPERTIES`.
    tolerances
        ``(n_classes, 3)`` admissible deviation (same units), used to normalise the
        penalty.
    classes
        Human-readable class labels, in row order of ``increments``.
    """

    anchor_idx: np.ndarray
    partner_idx: np.ndarray
    cls_idx: np.ndarray
    increments: np.ndarray
    tolerances: np.ndarray
    classes: list[str]
    n_pairs_total: int = 0
    n_pairs_skipped: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.anchor_idx.size)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_counts(self) -> dict[str, int]:
        """Number of pairs per insertion-site class."""
        counts = np.bincount(self.cls_idx, minlength=max(self.n_classes, 1))
        return {c: int(n) for c, n in zip(self.classes, counts)}

    def tensors(self) -> dict[str, torch.Tensor]:
        """Torch view of the table (long / float32), for collation."""
        return {
            "anchor_idx": torch.as_tensor(self.anchor_idx, dtype=torch.long),
            "partner_idx": torch.as_tensor(self.partner_idx, dtype=torch.long),
            "cls_idx": torch.as_tensor(self.cls_idx, dtype=torch.long),
            "increments": torch.as_tensor(self.increments, dtype=torch.float32),
            "tolerances": torch.as_tensor(self.tolerances, dtype=torch.float32),
        }

    def summary(self) -> str:
        """One-line description, for the training log."""
        c = self.class_counts()
        top = sorted(c.items(), key=lambda kv: -kv[1])[:4]
        top_s = ", ".join(f"{k}:{v}" for k, v in top)
        return (f"{len(self)} CH2-homolog pairs / {self.n_classes} site classes "
                f"({top_s}{', ...' if len(c) > len(top) else ''}) "
                f"over {int(np.unique(self.anchor_idx).size)} anchors"
                + (f"; {self.n_pairs_skipped} skipped (missing target)"
                   if self.n_pairs_skipped else ""))

    def to_json(self) -> dict[str, Any]:
        """Plain-python dump for results.json (kcal / cal for readability)."""
        return {
            "n_pairs": len(self),
            "n_anchors": int(np.unique(self.anchor_idx).size),
            "n_classes": self.n_classes,
            "n_pairs_total": self.n_pairs_total,
            "n_pairs_skipped": self.n_pairs_skipped,
            "classes": list(self.classes),
            "class_counts": self.class_counts(),
            "increments": {
                "h_kcal_per_mol": (self.increments[:, 0] / 4184.0).tolist(),
                "s_cal_per_molK": (self.increments[:, 1] / 4.184).tolist(),
                "cp_cal_per_molK": (self.increments[:, 2] / 4.184).tolist(),
            },
            "tolerances": {
                "h_kcal_per_mol": (self.tolerances[:, 0] / 4184.0).tolist(),
                "s_cal_per_molK": (self.tolerances[:, 1] / 4.184).tolist(),
                "cp_cal_per_molK": (self.tolerances[:, 2] / 4.184).tolist(),
            },
            "stats": dict(self.stats),
        }


def _robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD, falling back to the std for degenerate samples."""
    if x.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    return mad if mad > 0 else float(np.std(x))


def _extract_targets(dataset: Sequence[Any], cp_index: int | None):
    """Pull (smiles, dHf, S, Cp(T)) out of the dataset datapoints as float arrays."""
    # ThermochemDataset.__getitem__ returns a collated *dict*, not the datapoint, so unwrap
    # the datapoint list when the dataset exposes one.
    items = getattr(dataset, "datapoints", dataset)
    n = len(items)
    smiles = [str(getattr(dp, "smiles", "") or "") for dp in items]
    h = np.array([float(getattr(dp, "dh_f_298", np.nan)) for dp in items], dtype=float)
    s = np.array([float(getattr(dp, "s_298", np.nan)) for dp in items], dtype=float)
    cp = np.full(n, np.nan)
    t_grid = np.zeros(0)
    if n:
        t_grid = np.asarray(getattr(items[0], "T_grid", []), dtype=float).ravel()
    if cp_index is None and t_grid.size:
        cp_index = int(np.argmin(np.abs(t_grid - 300.0)))
    if cp_index is not None:
        for i, dp in enumerate(items):
            grid = np.asarray(getattr(dp, "cp_grid", []), dtype=float).ravel()
            if grid.size > cp_index:
                cp[i] = grid[cp_index]
    return smiles, h, s, cp, cp_index, t_grid


def build_homolog_pairs(
    dataset: Sequence[Any],
    n_shrink: float = 5.0,
    tol_floor: float = 0.5,
    min_class_pairs: int = 1,
    cp_index: int | None = None,
    max_pairs_per_anchor: int | None = None,
    log: Callable[[str], None] | None = None,
) -> HomologPairTable:
    """Find CH2-homolog pairs inside ``dataset`` and estimate their increments.

    Parameters
    ----------
    dataset
        A :class:`tebnet.data.dataset.ThermochemDataset` (or any sequence of objects with
        ``smiles``, ``dh_f_298``, ``s_298``, ``cp_grid`` and ``T_grid``), normally the
        TRAINING split so that the prior never touches validation or test molecules.
    n_shrink
        Shrinkage strength: a class with ``n`` pairs contributes with weight
        ``n / (n + n_shrink)`` and the global increment with the remainder.
    tol_floor
        Lower bound on a tolerance, as a fraction of the global increment scale, so that a
        class whose members happen to agree perfectly does not become a hard constraint.
    min_class_pairs
        Classes with fewer pairs fall back to the global increment (their pairs are still
        used, just without a class-specific target).
    cp_index
        Which temperature-grid entry supplies the Cp increment. ``None`` picks the entry
        closest to 300 K, matching the Cp@300K column of the benchmarks.
    max_pairs_per_anchor
        Optional cap on how many partners one anchor contributes, so that a flexible
        molecule cannot dominate the term.
    log
        Optional callable receiving a human-readable summary.

    Returns
    -------
    HomologPairTable
        Empty (length 0) if the dataset contains no homologs; the caller then trains
        without the regulariser rather than failing.
    """
    smiles, h, s, cp, cp_index, t_grid = _extract_targets(dataset, cp_index)

    key_to_idx: dict[str, int] = {}
    mols: list[Chem.Mol | None] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        mols.append(mol)
        if mol is not None:
            # first occurrence wins; DeepTherm contains no duplicate structures
            key_to_idx.setdefault(Chem.MolToSmiles(mol, isomericSmiles=False), i)

    # ── enumerate pairs (deduplicated; anchor -> longer partner) ──
    pair_of: dict[tuple[int, int], str] = {}
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        n_here = 0
        for prod_smi, label in insert_ch2(mol):
            j = key_to_idx.get(prod_smi)
            if j is None or j == i:
                continue
            pair_of.setdefault((i, j), label)
            n_here += 1
            if max_pairs_per_anchor is not None and n_here >= max_pairs_per_anchor:
                break

    if not pair_of:
        if log:
            log("CH2-homolog prior: no pairs found in this split -> regulariser disabled")
        empty = np.zeros(0, dtype=np.int64)
        return HomologPairTable(
            anchor_idx=empty, partner_idx=empty.copy(), cls_idx=empty.copy(),
            increments=np.zeros((0, 3)), tolerances=np.zeros((0, 3)), classes=[],
        )

    anchors = np.fromiter((a for a, _ in pair_of), dtype=np.int64, count=len(pair_of))
    partners = np.fromiter((p for _, p in pair_of), dtype=np.int64, count=len(pair_of))
    labels = list(pair_of.values())
    n_pairs_total = len(labels)

    # ── increments; drop pairs whose targets are missing ──
    d = np.stack([h[partners] - h[anchors],
                  s[partners] - s[anchors],
                  cp[partners] - cp[anchors]], axis=1)
    ok = np.isfinite(d).all(axis=1)
    n_skipped = int((~ok).sum())
    anchors, partners, d = anchors[ok], partners[ok], d[ok]
    labels = [lab for lab, keep in zip(labels, ok) if keep]

    classes = sorted(set(labels))
    cls_of = {c: k for k, c in enumerate(classes)}
    cls_idx = np.array([cls_of[lab] for lab in labels], dtype=np.int64)
    counts = np.bincount(cls_idx, minlength=len(classes))

    glob = np.median(d, axis=0)
    increments = np.empty((len(classes), 3))
    cls_scale = np.full((len(classes), 3), np.nan)
    for k in range(len(classes)):
        m = cls_idx == k
        if counts[k] >= min_class_pairs:
            lam = counts[k] / (counts[k] + float(n_shrink))
            increments[k] = lam * np.median(d[m], axis=0) + (1.0 - lam) * glob
            cls_scale[k] = [_robust_scale(d[m, q]) for q in range(3)]
        else:
            increments[k] = glob

    # Tolerance: pooled within-class spread (a class with a handful of pairs cannot set
    # its own tolerance), floored at tol_floor * the global spread, and finally floored at
    # the absolute TOL_MIN so a degenerate split cannot produce a hard constraint.
    resid = d - increments[cls_idx]
    glob_scale = np.array([_robust_scale(resid[:, q]) for q in range(3)])
    glob_scale = np.where(np.isfinite(glob_scale), glob_scale, 0.0)
    pooled = np.array([
        np.nanmedian(cls_scale[:, q]) if np.isfinite(cls_scale[:, q]).any() else glob_scale[q]
        for q in range(3)
    ])
    pooled = np.where(np.isfinite(pooled) & (pooled > 0), pooled, glob_scale)
    tolerances = np.tile(np.maximum(pooled, float(tol_floor) * glob_scale), (len(classes), 1))
    tolerances[counts < min_class_pairs] = pooled
    tolerances = np.maximum(np.where(np.isfinite(tolerances), tolerances, 0.0), TOL_MIN)

    if cp_index is not None and t_grid.size > cp_index:
        t_at = float(t_grid[cp_index])
    else:
        t_at = 300.0

    table = HomologPairTable(
        anchor_idx=anchors, partner_idx=partners, cls_idx=cls_idx,
        increments=increments, tolerances=tolerances, classes=classes,
        n_pairs_total=n_pairs_total, n_pairs_skipped=n_skipped,
        stats={
            "global_increment": {
                "h_kcal_per_mol": float(glob[0] / 4184.0),
                "s_cal_per_molK": float(glob[1] / 4.184),
                "cp_cal_per_molK": float(glob[2] / 4.184),
            },
            "global_spread": {
                "h_kcal_per_mol": float(glob_scale[0] / 4184.0),
                "s_cal_per_molK": float(glob_scale[1] / 4.184),
                "cp_cal_per_molK": float(glob_scale[2] / 4.184),
            },
            "tolerance": {
                "h_kcal_per_mol": float(tolerances[0, 0] / 4184.0),
                "s_cal_per_molK": float(tolerances[0, 1] / 4.184),
                "cp_cal_per_molK": float(tolerances[0, 2] / 4.184),
            },
            "cp_temperature_K": t_at,
            "n_shrink": float(n_shrink),
            "tol_floor": float(tol_floor),
            "min_class_pairs": int(min_class_pairs),
            "max_pairs_per_anchor": max_pairs_per_anchor,
        },
    )
    if log:
        log("CH2-homolog prior: " + table.summary())
    return table


# ────────────────────────────── paired dataset ──────────────────────────────
class HomologPairDataset(torch.utils.data.Dataset):
    """Iterate a :class:`HomologPairTable` as ``{anchor, partner, cls_idx}`` items.

    The pair items are pulled through the *same* ``__getitem__`` the main loader uses, so
    the graphs, the physics features and the weights are byte-identical to what the
    per-molecule objective sees for those species -- no second featurisation path, and the
    collater's graph cache is shared.  This matters for a regulariser: if the pair branch
    saw a differently-featurised molecule, part of the "consistency error" would be a
    featurisation artefact rather than a modelling error.

    ``dataset`` is the dataset the table was built from, i.e. the training split.  Using a
    different dataset is only meaningful if its indices line up, which is why this is
    documented rather than enforced.

    Parameters
    ----------
    dataset
        The training-split dataset the table indexes into.
    table
        Pairs to yield.  A table of length 0 yields an empty dataset (``len == 0``), and
        the caller is expected to skip the paired loader entirely.
    repeat
        Emit the pairs ``repeat`` times per epoch.  With ``mode="max_size_cycle"`` the
        paired loader is cycled up to the length of the main loader, so a *smaller* pair
        set already gets full coverage; ``repeat`` is for the opposite case (a pair table
        larger than the molecule loader) and for up-weighting the term.
    """

    def __init__(self, dataset: Sequence[Any], table: HomologPairTable, repeat: int = 1):
        self.dataset = dataset
        self.table = table
        self.repeat = max(1, int(repeat))
        self.anchor_idx = np.asarray(table.anchor_idx, dtype=np.int64)
        self.partner_idx = np.asarray(table.partner_idx, dtype=np.int64)
        self.cls_idx = np.asarray(table.cls_idx, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.anchor_idx.size) * self.repeat

    def __getitem__(self, i: int) -> dict[str, Any]:
        k = i % self.anchor_idx.size if self.anchor_idx.size else 0
        return {
            "anchor": self.dataset[int(self.anchor_idx[k])],
            "partner": self.dataset[int(self.partner_idx[k])],
            "cls_idx": int(self.cls_idx[k]),
        }



