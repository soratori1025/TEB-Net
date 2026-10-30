from __future__ import annotations
from typing import Sequence
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, HybridizationType
from rdkit.Chem import rdMolDescriptors
import sys
from pathlib import Path
from chemprop.featurizers.base import VectorFeaturizer
from rdkit.Chem.rdchem import Bond, BondType
_root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_root_dir) not in sys.path:
    sys.path.append(str(_root_dir))
if str(_root_dir / "data") not in sys.path:
    sys.path.append(str(_root_dir / "data"))
try:
    from symmetry import compute_symmetry_number
    from molecular_geometry import compute_is_linear
except ImportError:
    from data.symmetry import compute_symmetry_number
    from data.molecular_geometry import compute_is_linear
ELEMENT_VOCAB = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
COMP_DIM = len(ELEMENT_VOCAB) + 1
AUX_DIM_MINIMAL = 4
AUX_DIM_EXTENDED = 8
AUX_DIM_ROUTED = 16
ROUTE_H_IDX = [3, 5, 6, 8, 10]
ROUTE_S_IDX = [0, 1, 2, 4, 5, 7, 9, 12, 13, 14, 15]
ROUTE_CP_IDX = [1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]
FEATURES_VERSION = "v15.3"
COMP_NAMES = ["n_H", "n_C", "n_N", "n_O", "n_F", "n_P", "n_S", "n_Cl", "n_Br", "n_I", "n_other"]
assert len(COMP_NAMES) == COMP_DIM
PHYS_V15_NAMES = [
    "log_sigma", "is_linear", "is_monatomic", "n_atoms_H", "n_heavy",
    "n_rad_e", "ln_mult", "is_chiral", "net_charge",
    "n_rotors", "n_methyl",
    "n_rings", "n_arom_rings", "ring3", "ring4", "ring5", "ring6", "ring7p",
    "n_fused_bonds", "n_bridgehead", "n_spiro",
    "frac_sp3", "frac_sp2", "frac_sp",
    "n_single", "n_double", "n_triple", "n_arom_bond",
    "n_CC1", "n_CC2", "n_CC3", "n_CCa", "n_CO1", "n_CO2", "n_COa", "n_OO1", "n_CH", "n_OH",
    "ring3_unsat", "ring4_unsat", "ring_allene",
    "rad_alkyl_p", "rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_allylic", "rad_alpha_CO",
    "rad_vinyl", "rad_aryl", "rad_acyl", "rad_alkoxy", "rad_peroxy", "rad_ethynyl", "rad_vinoxy", "rad_captodative",
    "log_I_rot", "log_r_gyr", "moi_valid",
]
V15_NAMES = COMP_NAMES + PHYS_V15_NAMES
V15_DIM = len(V15_NAMES)
FEATS_3D = {"log_I_rot", "log_r_gyr", "moi_valid"}
_ROUTE_V15 = {
    "h": ["n_rad_e", "net_charge", "n_rings", "n_arom_rings", "ring3", "ring4", "ring5", "ring6",
          "ring7p", "n_fused_bonds", "n_bridgehead", "n_spiro", "frac_sp3", "frac_sp2", "frac_sp",
          "n_double", "n_triple", "n_arom_bond", "ring3_unsat", "ring4_unsat", "ring_allene",
          "rad_alkyl_p", "rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_allylic", "rad_alpha_CO", "rad_vinyl", "rad_aryl", "rad_acyl", "rad_alkoxy", "rad_peroxy", "rad_ethynyl", "rad_vinoxy", "rad_captodative"],
    "s": ["log_sigma", "is_linear", "is_monatomic", "n_atoms_H", "n_heavy", "n_rad_e", "ln_mult",
          "is_chiral", "n_rotors", "n_methyl", "n_rings", "n_arom_rings", "ring3", "ring4", "ring5",
          "ring6", "ring7p", "n_fused_bonds", "n_bridgehead", "n_spiro", "frac_sp3",
          "log_I_rot", "log_r_gyr", "moi_valid"],
    "cp": ["is_linear", "is_monatomic", "n_atoms_H", "n_heavy", "n_rad_e", "n_rotors", "n_methyl",
           "n_rings", "n_arom_rings", "ring3", "ring4", "ring5", "ring6", "ring7p", "n_fused_bonds",
           "frac_sp3", "frac_sp2", "frac_sp", "n_double", "n_triple", "n_arom_bond",
           "log_I_rot", "log_r_gyr", "moi_valid"],
}
_BASE_V15_PHYS = {
    "h": ["n_single", "n_double", "n_triple", "n_arom_bond",
          "n_CC1", "n_CC2", "n_CC3", "n_CCa", "n_CO1", "n_CO2", "n_COa", "n_OO1", "n_CH", "n_OH",
          "ring3", "ring4", "ring5", "ring6", "ring7p", "ring3_unsat", "ring4_unsat", "ring_allene",
          "n_arom_rings", "n_bridgehead", "n_spiro", "n_rad_e",
          "rad_alkyl_p", "rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_allylic", "rad_alpha_CO", "rad_vinyl", "rad_aryl", "rad_acyl", "rad_alkoxy", "rad_peroxy", "rad_ethynyl", "rad_vinoxy", "rad_captodative"],
    "s": ["log_sigma", "ln_mult", "is_chiral", "is_linear", "is_monatomic", "n_rotors", "n_methyl",
          "ring3", "ring4", "ring5", "ring6", "ring7p", "n_arom_rings", "n_double", "n_triple",
          "n_arom_bond", "log_I_rot", "moi_valid"],
    "cp": ["is_linear", "is_monatomic", "n_rotors", "n_methyl", "n_rings", "n_arom_rings", "n_rad_e",
           "n_double", "n_triple", "n_arom_bond"],
}
class ThermoAtomFeaturizer:
    """Multi-hot atom featurizer with radical electron support."""
    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
        num_radical_electrons: Sequence[int] = (0, 1, 2, 3),
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}
        self.num_radical_electrons_map = {i: i for i in num_radical_electrons}
        self._subfeats: list[dict] = [
            self.atomic_nums,
            self.degrees,
            self.formal_charges,
            self.chiral_tags,
            self.num_Hs,
            self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums),
            1 + len(self.degrees),
            1 + len(self.formal_charges),
            1 + len(self.chiral_tags),
            1 + len(self.num_Hs),
            1 + len(self.hybridizations),
            1,
            1,
            1 + len(self.num_radical_electrons_map),
        ]
        self.__size = sum(subfeat_sizes)
    def __len__(self) -> int:
        return self.__size
    def __call__(self, a: Atom | None) -> np.ndarray:
        x = np.zeros(self.__size)
        if a is None:
            return x
        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        i += 2
        n_rad = a.GetNumRadicalElectrons()
        j = self.num_radical_electrons_map.get(n_rad, len(self.num_radical_electrons_map))
        x[i + j] = 1
        return x
    @classmethod
    def v2_radical(cls) -> "ThermoAtomFeaturizer":
        return cls(
            atomic_nums=list(range(1, 37)) + [53],
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
            num_radical_electrons=[0, 1, 2, 3],
        )
def compute_composition_vector(mol: Chem.Mol) -> np.ndarray:
    """Element-count vector over ELEMENT_VOCAB, taken over the H-explicit molecule.

    This feeds the composition / reference-energy baseline for ΔH_f,298:
    formation enthalpy is dominated by molecular composition, so subtracting a
    linear composition fit lets the network learn only the small residual
    (ring strain, conjugation, radical stabilization, ...). Same idea as
    per-element reference-energy subtraction in ML interatomic potentials.
    """
    mol_h = Chem.AddHs(mol)
    idx = {z: i for i, z in enumerate(ELEMENT_VOCAB)}
    vec = np.zeros(COMP_DIM, dtype=np.float32)
    for a in mol_h.GetAtoms():
        z = a.GetAtomicNum()
        vec[idx.get(z, COMP_DIM - 1)] += 1.0
    return vec
def compute_physic_features(mol: Chem.Mol) -> list[float]:
    """v14 4-D core physics vector."""
    mol_h = Chem.AddHs(mol)
    N = float(mol_h.GetNumAtoms())
    try:
        sigma = compute_symmetry_number(mol)
    except Exception:
        sigma = 1
    try:
        is_linear = float(compute_is_linear(mol))
    except Exception:
        is_linear = 0.0
    radical_e = float(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    return [
        float(np.log(max(sigma, 1))),
        is_linear,
        N,
        radical_e,
    ]
def compute_physic_features_extended(mol: Chem.Mol) -> list[float]:
    """v14.1 8-D physics/topology vector = core-4 + 4 cheap graph descriptors.

    The added descriptors target the properties that still overfit in v14:
      5. n_rotatable_bonds — internal (hindered) rotors → S°298 and low-T Cp
      6. n_rings           — ring rigidity / strain → constrains S and shifts H_f
      7. n_aromatic_rings  — conjugation / rigidity
      8. n_heavy_atoms     — heavy-atom size (with core-4's N this exposes H-count)

    A sum-pooled D-MPNN can in principle recover some of these, but making them
    explicit global scalars gives the read-out a stronger, size-aware signal for
    entropy and heat capacity at negligible parameter cost.
    """
    base = compute_physic_features(mol)
    n_rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    n_ring = float(rdMolDescriptors.CalcNumRings(mol))
    n_arom = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    n_heavy = float(mol.GetNumHeavyAtoms())
    return base + [n_rot, n_ring, n_arom, n_heavy]
def compute_physic_features_routed(mol: Chem.Mol) -> list[float]:
    """v14.4 routed 12-D feature vector (see ROUTE_*_IDX for per-property slices).

    Element counts are NOT included: the H/S composition baselines already model
    the compositional part, so this vector carries only the residual structural
    determinants. All values come from the 2D graph — no QM.
    """
    mol_h = Chem.AddHs(mol)
    N = float(mol_h.GetNumAtoms())
    n_heavy = float(mol.GetNumHeavyAtoms())
    try:
        sigma = compute_symmetry_number(mol)
    except Exception:
        sigma = 1
    try:
        is_linear = float(compute_is_linear(mol))
    except Exception:
        is_linear = 0.0
    radical_e = float(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    net_charge = float(Chem.GetFormalCharge(mol))
    n_rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    n_ring = float(rdMolDescriptors.CalcNumRings(mol))
    n_arom = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    def _get_moi(m):
        try:
            from rdkit.Chem import AllChem
            mh = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mh, params) == -1:
                return [0.0, 0.0, 0.0, 0.0]
            conf = mh.GetConformer()
            pts = conf.GetPositions()
            masses = np.array([a.GetMass() for a in mh.GetAtoms()])
            com = np.average(pts, axis=0, weights=masses)
            pts_c = pts - com
            I_xx = np.sum(masses * (pts_c[:, 1]**2 + pts_c[:, 2]**2))
            I_yy = np.sum(masses * (pts_c[:, 0]**2 + pts_c[:, 2]**2))
            I_zz = np.sum(masses * (pts_c[:, 0]**2 + pts_c[:, 1]**2))
            I_xy = -np.sum(masses * pts_c[:, 0] * pts_c[:, 1])
            I_xz = -np.sum(masses * pts_c[:, 0] * pts_c[:, 2])
            I_yz = -np.sum(masses * pts_c[:, 1] * pts_c[:, 2])
            I_tensor = np.array([
                [I_xx, I_xy, I_xz],
                [I_xy, I_yy, I_yz],
                [I_xz, I_yz, I_zz]
            ])
            eigenvalues = np.linalg.eigvals(I_tensor)
            I_A, I_B, I_C = sorted([max(0.0, e.real) for e in eigenvalues])
            r_gyr = np.sqrt(np.sum(masses * np.sum(pts_c**2, axis=1)) / np.sum(masses))
            return [
                float(np.log(I_A + 1e-4)), 
                float(np.log(I_B + 1e-4)), 
                float(np.log(I_C + 1e-4)), 
                float(np.log(r_gyr + 1e-4))
            ]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    moi = _get_moi(mol)
    n_atoms = max(mol.GetNumAtoms(), 1)
    n_sp3 = n_sp2 = n_sp = 0
    for a in mol.GetAtoms():
        hyb = a.GetHybridization()
        if hyb == HybridizationType.SP3:
            n_sp3 += 1
        elif hyb == HybridizationType.SP2:
            n_sp2 += 1
        elif hyb == HybridizationType.SP:
            n_sp += 1
    return [
        float(np.log(max(sigma, 1))),
        is_linear,
        N,
        radical_e,
        n_rot,
        n_ring,
        n_arom,
        n_heavy,
        net_charge,
        float(n_sp3 / n_atoms),
        float(n_sp2 / n_atoms),
        float(n_sp / n_atoms),
        moi[0],
        moi[1],
        moi[2],
        moi[3],
    ]
def compute_auxiliary_features(mol: Chem.Mol, precomputed_physic: np.ndarray | list[float] | str | None = None, feature_set: str = "extended",) -> np.ndarray:
    """Compute the auxiliary feature vector.
    feature_set="minimal"  -> 4-D  (reproduces v14)
    feature_set="extended" -> 8-D  (v14.1 default)
    feature_set="routed"   -> 16-D (v14.4; sliced per-property by the model)
    feature_set="v15"      -> len(V15_NAMES) named vector (comp + physics; see v15 section)
    """
    if feature_set == "v15":
        return compute_v15_vector(mol, use_3d=True)
    if feature_set == "routed" and precomputed_physic is None:
        return np.array(compute_physic_features_routed(mol), dtype=np.float32)
    if precomputed_physic is not None:
        if isinstance(precomputed_physic, str):
            try:
                import json
                precomputed_physic = json.loads(precomputed_physic)
            except Exception:
                import ast
                precomputed_physic = ast.literal_eval(precomputed_physic)
        arr = np.array(precomputed_physic, dtype=np.float32)
        if len(arr) > AUX_DIM_EXTENDED:
            arr = np.array([arr[0], arr[1], arr[2], arr[8]], dtype=np.float32)
        return arr
    if feature_set == "minimal":
        return np.array(compute_physic_features(mol), dtype=np.float32)
    return np.array(compute_physic_features_extended(mol), dtype=np.float32)
class CleanBondFeaturizer(VectorFeaturizer[Bond]):
    """Bond featurizer without 3D conformer dependency.

    Features: bond type (one-hot), conjugated, aromatic, in_ring,
    ring sizes, stereo, bond order float.
    No bond_length — removes noisy 3D conformer dependency.
    """
    def __init__(
        self,
        bond_types: Sequence[BondType] | None = None,
        stereos: Sequence[int] | None = None,
    ):
        self.bond_types = bond_types or [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        self.stereo = stereos or range(6)
        self.ring_sizes = [3, 4, 5, 6, 7, 8]
    def __len__(self):
        return 1 + len(self.bond_types) + 1 + 1 + 1 + len(self.ring_sizes) + (len(self.stereo) + 1) + 1
    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), dtype=np.float32)
        if b is None:
            x[0] = 1
            return x
        i = 1
        bond_type = b.GetBondType()
        bt_bit, size = self.one_hot_index(bond_type, self.bond_types)
        if bt_bit != size:
            x[i + bt_bit] = 1
        i += size - 1
        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.GetIsAromatic())
        x[i + 2] = int(b.IsInRing())
        i += 3
        for rs_idx, rs in enumerate(self.ring_sizes):
            x[i + rs_idx] = int(b.IsInRingSize(rs))
        i += len(self.ring_sizes)
        stereo_bit, _ = self.one_hot_index(int(b.GetStereo()), self.stereo)
        x[i + stereo_bit] = 1
        i += len(self.stereo) + 1
        try:
            x[i] = b.GetBondTypeAsDouble()
        except Exception:
            x[i] = 1.0
        return x
    @classmethod
    def one_hot_index(cls, x, xs: Sequence) -> tuple[int, int]:
        n = len(xs)
        return (xs.index(x), n + 1) if x in xs else (n, n + 1)
def _drop_3d(names: list[str], use_3d: bool) -> list[str]:
    return list(names) if use_3d else [n for n in names if n not in FEATS_3D]
def v15_route_names(role: str, route_mode: str = "grouped", use_3d: bool = True) -> list[str]:
    """Columns fed (normalised) to branch `role` in {"h","s","cp"}."""
    if route_mode == "full":
        return _drop_3d(PHYS_V15_NAMES, use_3d)
    if route_mode != "grouped":
        raise ValueError(f"route_mode must be 'grouped' or 'full', got {route_mode!r}")
    return _drop_3d(_ROUTE_V15[role], use_3d)
def v15_base_names(role: str, baseline: str = "physics", use_3d: bool = True) -> list[str]:
    """Columns of the frozen linear baseline for `role`.

    baseline="composition" -> element counts only (= v14.2 behaviour for H/S)
    baseline="physics"     -> element counts + physics columns
    """
    if baseline == "composition":
        return list(COMP_NAMES)
    if baseline != "physics":
        raise ValueError(f"baseline must be 'composition' or 'physics', got {baseline!r}")
    return COMP_NAMES + _drop_3d(_BASE_V15_PHYS[role], use_3d)
class ThermoAtomFeaturizerV3(ThermoAtomFeaturizer):
    """v2_radical features + [in_ring, in_ring_size(3..8), #rings (0..3+), is_carbenium, radical_type].

    D-MPNN with depth 3-5 cannot see a whole 6-ring or a fused/bridged system from
    one atom; putting ring membership on the atom itself fixes that locally.
    We also add carbenium ion and detailed radical center types so the network
    doesn't have to infer them from scratch.
    """
    RING_SIZES = (3, 4, 5, 6, 7, 8)
    N_RING_CHOICES = (0, 1, 2)
    RADICAL_TYPES = [
        "rad_alkyl_p", "rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_allylic", "rad_alpha_CO",
        "rad_vinyl", "rad_aryl", "rad_acyl", "rad_alkoxy", "rad_peroxy", "rad_ethynyl", "rad_vinoxy", "rad_captodative"
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_extra = (
            1 + len(self.RING_SIZES) + len(self.N_RING_CHOICES) + 1
            + 1
            + len(self.RADICAL_TYPES) + 1
        )
    def __len__(self) -> int:
        return super().__len__() + self._n_extra
    def __call__(self, a: Atom | None) -> np.ndarray:
        base = super().__call__(a)
        extra = np.zeros(self._n_extra)
        if a is None:
            return np.concatenate([base, extra])
        extra[0] = float(a.IsInRing())
        for i, k in enumerate(self.RING_SIZES):
            extra[1 + i] = float(a.IsInRingSize(k))
        try:
            n_r = a.GetOwningMol().GetRingInfo().NumAtomRings(a.GetIdx())
        except Exception:
            n_r = 0
        off = 1 + len(self.RING_SIZES)
        j = self.N_RING_CHOICES.index(n_r) if n_r in self.N_RING_CHOICES else len(self.N_RING_CHOICES)
        extra[off + j] = 1.0
        off += len(self.N_RING_CHOICES) + 1
        is_carbenium = (a.GetAtomicNum() == 6 and a.GetFormalCharge() == 1)
        extra[off] = float(is_carbenium)
        off += 1
        if a.GetNumRadicalElectrons() > 0:
            rtype = radical_center_type(a)
            idx = self.RADICAL_TYPES.index(rtype) if rtype in self.RADICAL_TYPES else len(self.RADICAL_TYPES)
            extra[off + idx] = 1.0
        else:
            extra[off + len(self.RADICAL_TYPES)] = 1.0
        return np.concatenate([base, extra])
    @classmethod
    def thermo(cls) -> "ThermoAtomFeaturizerV3":
        return cls.v2_radical()
def is_linear_fast(mol: Chem.Mol) -> float:
    """1.0 if the molecule is linear. Monatomic species return 0.0 (see is_monatomic).

    Any atom with >= 3 neighbours (incl. H) or any ring => non-linear, exactly.
    Only chains where every atom has <= 2 neighbours (HCN, CO2, H2O, C2H, HO2, ...)
    fall through to the 3-D check.
    """
    try:
        n_tot = Chem.AddHs(mol).GetNumAtoms()
        if n_tot <= 1:
            return 0.0
        if n_tot == 2:
            return 1.0
        if mol.GetRingInfo().NumRings() > 0:
            return 0.0
        if any(a.GetTotalDegree() > 2 for a in mol.GetAtoms()):
            return 0.0
        return float(compute_is_linear(mol))
    except Exception:
        return 0.0
def _moi_features(mol: Chem.Mol, n_atoms_h: int, is_linear: float) -> tuple[float, float, float]:
    """(log_I_rot, log_r_gyr, valid).

    log_I_rot is the quantity that enters S_rot with coefficient R:
        non-linear : 0.5 * ln(I_A I_B I_C)      linear : ln(I)      monatomic : 0
    Moments in amu·Å² from one ETKDG conformer. valid=0 flags a failed embedding
    (values are then 0 and must not be read as a real geometry).
    """
    if n_atoms_h <= 1:
        return 0.0, 0.0, 1.0
    try:
        from rdkit.Chem import AllChem
        mh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mh, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mh, params)
        if cid < 0:
            return 0.0, 0.0, 0.0
        pts = np.asarray(mh.GetConformer(cid).GetPositions(), dtype=np.float64)
        m = np.array([a.GetMass() for a in mh.GetAtoms()], dtype=np.float64)
        com = (pts * m[:, None]).sum(0) / m.sum()
        x = pts - com
        r2 = (x ** 2).sum(1)
        inertia = np.eye(3) * float((m * r2).sum()) - (x.T * m) @ x
        ev = np.clip(np.linalg.eigvalsh(inertia), 1e-3, None)
        log_i = float(np.log(ev[2])) if is_linear >= 0.5 else float(0.5 * np.log(ev).sum())
        r_gyr = float(np.sqrt((m * r2).sum() / m.sum()))
        return log_i, float(np.log(max(r_gyr, 1e-3))), 1.0
    except Exception:
        return 0.0, 0.0, 0.0
def sssr_bond_rings(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """Smallest set of smallest rings as bond-index tuples (size = cyclomatic number).

    Built from RDKit's symmetrized SSSR (a superset of an SSSR): rings taken smallest first, kept
    only if linearly independent (GF(2)) of those already kept. Avoids depending on the
    return type of Chem.GetSSSR, which changed between RDKit versions."""
    ri = mol.GetRingInfo()
    candidates = sorted((tuple(r) for r in ri.BondRings()), key=len)
    n_frag = len(Chem.GetMolFrags(mol))
    target = mol.GetNumBonds() - mol.GetNumAtoms() + n_frag
    basis: dict[int, int] = {}
    chosen: list[tuple[int, ...]] = []
    for ring in candidates:
        if len(chosen) >= target:
            break
        x = 0
        for bi in ring:
            x |= 1 << bi
        while x:
            top = x.bit_length() - 1
            if top in basis:
                x ^= basis[top]
            else:
                basis[top] = x
                chosen.append(ring)
                break
    return chosen
def radical_center_type(a: Atom) -> str | None:
    """HBI-like class of a radical centre (priority: σ-radicals > resonance > α-heteroatom > alkyl)."""
    z = a.GetAtomicNum()
    heavy_nb = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
    def _unsat(n):
        return n.GetIsAromatic() or any(
            bb.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) for bb in n.GetBonds())
    if z == 8:
        if any(n.GetAtomicNum() == 8 for n in heavy_nb):
            return "rad_peroxy"
        if any(n.GetAtomicNum() == 6 and _unsat(n) for n in heavy_nb):
            return "rad_vinoxy"
        return "rad_alkoxy"
    if z != 6:
        return None
    if a.GetIsAromatic():
        return "rad_aryl"
    for b in a.GetBonds():
        if b.GetBondType() == Chem.BondType.TRIPLE:
            return "rad_ethynyl"
    for b in a.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            return "rad_acyl" if b.GetOtherAtom(a).GetAtomicNum() == 8 else "rad_vinyl"
    def _is_carbonyl_c(n):
        return n.GetAtomicNum() == 6 and any(
            bb.GetBondType() == Chem.BondType.DOUBLE and bb.GetOtherAtom(n).GetAtomicNum() == 8
            for bb in n.GetBonds())
    if any(_is_carbonyl_c(n) for n in heavy_nb) and any(n.GetAtomicNum() == 8 for n in heavy_nb):
        return "rad_captodative"
    def _has_pi(n, exclude):
        return any(bb.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
                   and bb.GetOtherAtomIdx(n.GetIdx()) != exclude for bb in n.GetBonds())
    for n in heavy_nb:
        if n.GetAtomicNum() == 6 and _has_pi(n, a.GetIdx()):
            pi_to_O = any(bb.GetBondType() == Chem.BondType.DOUBLE and bb.GetOtherAtom(n).GetAtomicNum() == 8
                          for bb in n.GetBonds())
            if not pi_to_O:
                return "rad_allylic"
    for n in heavy_nb:
        if n.GetAtomicNum() == 6 and any(bb.GetBondType() == Chem.BondType.DOUBLE and bb.GetOtherAtom(n).GetAtomicNum() == 8
                                         for bb in n.GetBonds()):
            return "rad_alpha_CO"
    if any(n.GetAtomicNum() == 8 for n in heavy_nb):
        return "rad_alpha_O"
    k = len(heavy_nb)
    return "rad_alkyl_p" if k <= 1 else ("rad_alkyl_s" if k == 2 else "rad_alkyl_t")
def compute_v15_features(mol: Chem.Mol, use_3d: bool = True) -> dict[str, float]:
    """Named v15 features. Never raises for a valid RDKit mol (falls back to 0)."""
    f: dict[str, float] = {n: 0.0 for n in V15_NAMES}
    comp = compute_composition_vector(mol)
    for n, v in zip(COMP_NAMES, comp):
        f[n] = float(v)
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    n_atoms_h = int(Chem.AddHs(mol).GetNumAtoms())
    f["n_atoms_H"] = float(n_atoms_h)
    f["n_heavy"] = float(mol.GetNumHeavyAtoms())
    f["is_monatomic"] = float(n_atoms_h == 1)
    try:
        sigma = float(compute_symmetry_number(mol))
    except Exception:
        sigma = 1.0
    f["log_sigma"] = float(np.log(max(sigma, 1.0)))
    f["is_linear"] = is_linear_fast(mol)
    n_rad = float(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    f["n_rad_e"] = n_rad
    f["ln_mult"] = float(np.log(n_rad + 1.0))
    f["net_charge"] = float(Chem.GetFormalCharge(mol))
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        f["is_chiral"] = float(len(centers) > 0)
    except Exception:
        f["is_chiral"] = 0.0
    n_rot = 0
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        if a1.GetTotalDegree() < 2 or a2.GetTotalDegree() < 2:
            continue
        if HybridizationType.SP in (a1.GetHybridization(), a2.GetHybridization()):
            continue
        n_rot += 1
    f["n_rotors"] = float(n_rot)
    f["n_methyl"] = float(sum(
        1 for a in heavy
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs(includeNeighbors=True) == 3
        and sum(nb.GetAtomicNum() > 1 for nb in a.GetNeighbors()) == 1
        and a.GetNumRadicalElectrons() == 0 and a.GetFormalCharge() == 0
    ))
    try:
        rings = sssr_bond_rings(mol)
        sizes = [len(r) for r in rings]
        f["n_rings"] = float(len(sizes))
        for k in (3, 4, 5, 6):
            f[f"ring{k}"] = float(sum(1 for s in sizes if s == k))
        f["ring7p"] = float(sum(1 for s in sizes if s >= 7))
        bond_count: dict[int, int] = {}
        for r in rings:
            for bi in r:
                bond_count[bi] = bond_count.get(bi, 0) + 1
        f["n_fused_bonds"] = float(sum(1 for c in bond_count.values() if c > 1))
        for r in rings:
            if len(r) in (3, 4) and any(mol.GetBondWithIdx(bi).GetBondType() == Chem.BondType.DOUBLE for bi in r):
                f[f"ring{len(r)}_unsat"] += 1
        ring_atoms = {a for r in rings for bi in r
                      for a in (mol.GetBondWithIdx(bi).GetBeginAtomIdx(), mol.GetBondWithIdx(bi).GetEndAtomIdx())}
        f["ring_allene"] = float(sum(
            1 for ai in ring_atoms
            if sum(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetAtomWithIdx(ai).GetBonds()) >= 2
        ))
        f["n_arom_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        f["n_bridgehead"] = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        f["n_spiro"] = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    except Exception:
        pass
    n_hv = max(len(heavy), 1)
    hyb = [a.GetHybridization() for a in heavy]
    f["frac_sp3"] = sum(h == HybridizationType.SP3 for h in hyb) / n_hv
    f["frac_sp2"] = sum(h == HybridizationType.SP2 for h in hyb) / n_hv
    f["frac_sp"] = sum(h == HybridizationType.SP for h in hyb) / n_hv
    for b in mol.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() == 1 or b.GetEndAtom().GetAtomicNum() == 1:
            continue
        bt = b.GetBondType()
        if bt == Chem.BondType.SINGLE:
            f["n_single"] += 1
        elif bt == Chem.BondType.DOUBLE:
            f["n_double"] += 1
        elif bt == Chem.BondType.TRIPLE:
            f["n_triple"] += 1
        elif bt == Chem.BondType.AROMATIC:
            f["n_arom_bond"] += 1
    pair_name = {(6, 6, "S"): "n_CC1", (6, 6, "D"): "n_CC2", (6, 6, "T"): "n_CC3", (6, 6, "A"): "n_CCa",
                 (6, 8, "S"): "n_CO1", (6, 8, "D"): "n_CO2", (6, 8, "A"): "n_COa", (8, 8, "S"): "n_OO1"}
    bt_code = {Chem.BondType.SINGLE: "S", Chem.BondType.DOUBLE: "D", Chem.BondType.TRIPLE: "T",
               Chem.BondType.AROMATIC: "A"}
    for b in mol.GetBonds():
        z = tuple(sorted((b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())))
        name = pair_name.get((z[0], z[1], bt_code.get(b.GetBondType(), "?")))
        if name:
            f[name] += 1
    f["n_CH"] = float(sum(a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms() if a.GetAtomicNum() == 6))
    f["n_OH"] = float(sum(a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms() if a.GetAtomicNum() == 8))
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() == 0:
            continue
        key = radical_center_type(a)
        if key:
            f[key] += 1
    if use_3d:
        log_i, log_r, ok = _moi_features(mol, n_atoms_h, f["is_linear"])
        f["log_I_rot"], f["log_r_gyr"], f["moi_valid"] = log_i, log_r, ok
    return f
def compute_v15_vector(mol: Chem.Mol, use_3d: bool = True) -> np.ndarray:
    f = compute_v15_features(mol, use_3d=use_3d)
    return np.array([f[n] for n in V15_NAMES], dtype=np.float32)
def compute_v15_from_smiles(smiles: str, use_3d: bool = True) -> list[float] | None:
    """For the multiprocessing precompute script. None if SMILES is invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        return compute_v15_vector(mol, use_3d=use_3d).tolist()
    except Exception:
        return None
def physics_cache_path(csv_path) -> Path:
    """<dir>/<stem>.physics_v15.csv next to the dataset CSV."""
    p = Path(csv_path)
    return p.with_name(f"{p.stem}.physics_v15.csv")
ENTROPY_V2_VERSION = "e2.0"
ENTROPY_V2_NAMES = ["S_trans_R", "S_rot_R", "log_I_rot_min", "moi2_invalid"]
_KB, _HP, _AMU, _ANG2 = 1.380649e-23, 6.62607015e-34, 1.66053906660e-27, 1e-20
_T_REF, _P_REF = 298.15, 1.0e5
_RAD_TYPES = ["rad_alkyl_p", "rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_allylic", "rad_alpha_CO",
              "rad_vinyl", "rad_aryl", "rad_acyl", "rad_alkoxy", "rad_peroxy", "rad_ethynyl", "rad_vinoxy",
              "rad_captodative"]
_BOND_PAIRS = ["n_CC1", "n_CC2", "n_CC3", "n_CCa", "n_CO1", "n_CO2", "n_COa", "n_OO1", "n_CH", "n_OH"]
_S_V2_FREE = ["is_chiral", "is_linear", "is_monatomic", "n_rotors", "n_methyl",
              "ring3", "ring4", "ring5", "ring6", "ring7p", "n_arom_rings",
              "n_double", "n_triple", "n_arom_bond", *_BOND_PAIRS, "n_rad_e", *_RAD_TYPES, "moi2_invalid"]
_S_V2_FIXED = {
    "v2_rot":   {"S_trans_R": 1.0, "S_rot_R": 1.0},
    "v2_exact": {"S_trans_R": 1.0, "S_rot_R": 1.0, "log_sigma": -1.0, "ln_mult": 1.0},
    "v2_free":  {},
}
S_PRIOR_VARIANTS = ("v1", "v1_minconf", "v2_free", "v2_rot", "v2_exact")
def s_prior_columns(variant: str) -> tuple[list[str], dict[str, float]]:
    """(column names, fixed coefficients in units of R) of the S prior.

    v1          the published v15 prior (v15_base_names("s", "physics"))
    v1_minconf  v1 with ln I from the lowest-energy conformer instead of one random conformer
    v2_free     exact-term columns + additive counts, everything fitted (checks the data conventions)
    v2_rot      S_trans and S_rot fixed, σ and spin fitted
    v2_exact    S_trans, S_rot, −ln σ and ln(2S+1) fixed
    Names refer to V15_NAMES + ENTROPY_V2_NAMES.
    """
    if variant == "v1":
        return v15_base_names("s", "physics", True), {}
    if variant == "v1_minconf":
        cols = [("log_I_rot_min" if n == "log_I_rot" else "moi2_invalid" if n == "moi_valid" else n)
                for n in v15_base_names("s", "physics", True)]
        return cols, {}
    if variant not in _S_V2_FIXED:
        raise ValueError(f"unknown S-prior variant {variant!r}; choose from {S_PRIOR_VARIANTS}")
    fixed = _S_V2_FIXED[variant]
    exact = ["S_trans_R", "S_rot_R", "log_sigma", "ln_mult"]
    return COMP_NAMES + exact + _S_V2_FREE, dict(fixed)
def s_trans_over_R(mass_amu: float, T: float = _T_REF, p: float = _P_REF) -> float:
    """Sackur–Tetrode translational entropy / R of an ideal gas."""
    m = mass_amu * _AMU
    return float(2.5 + 1.5 * np.log(2.0 * np.pi * m * _KB * T / _HP ** 2) + np.log(_KB * T / p))
def s_rot_over_R(moments_amu_A2, linear: bool, T: float = _T_REF) -> float:
    """Classical rigid-rotor entropy / R for σ = 1 (subtract ln σ separately). Moments in amu·Å²."""
    c = 8.0 * np.pi ** 2 * _KB * T / _HP ** 2
    ev = np.sort(np.asarray(moments_amu_A2, dtype=np.float64)) * _AMU * _ANG2
    if linear:
        return float(1.0 + np.log(c * ev[2]))
    return float(1.5 + 0.5 * np.log(np.pi) + 1.5 * np.log(c) + 0.5 * np.log(ev).sum())
def _all_anti_conformer(mh: Chem.Mol, conf) -> "Chem.Conformer":
    """Copy of `conf` with every heavy-atom torsion about an acyclic single bond set to 180° (the extended
    chain is the gas-phase minimum of n-alkanes and a good start for other chains)."""
    from rdkit.Chem import rdMolTransforms
    c = Chem.Conformer(conf)
    for b in mh.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        if i.GetAtomicNum() == 1 or j.GetAtomicNum() == 1:
            continue
        if HybridizationType.SP in (i.GetHybridization(), j.GetHybridization()):
            continue
        ni = [n.GetIdx() for n in i.GetNeighbors() if n.GetIdx() != j.GetIdx() and n.GetAtomicNum() > 1]
        nj = [n.GetIdx() for n in j.GetNeighbors() if n.GetIdx() != i.GetIdx() and n.GetAtomicNum() > 1]
        if not ni or not nj:
            continue
        try:
            rdMolTransforms.SetDihedralDeg(c, ni[0], i.GetIdx(), j.GetIdx(), nj[0], 180.0)
        except Exception:
            pass
    return c
def lowest_conformer_moments(mol: Chem.Mol, n_conf: int = 10, seed: int = 42, max_iters: int = 1000):
    """Principal moments (amu·Å², ascending) of the lowest force-field energy conformer among `n_conf` ETKDG
    conformers plus one all-anti conformer; MMFF94 if parametrised, else UFF (radicals), else unoptimised.
    Returns (moments or None, force field used)."""
    from rdkit.Chem import AllChem
    mh = Chem.AddHs(mol)
    if mh.GetNumAtoms() <= 1:
        return np.zeros(3), "none"
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.maxIterations = 10
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=params))
    if not cids:
        return None, "none"
    try:
        mh.AddConformer(_all_anti_conformer(mh, mh.GetConformer(cids[0])), assignId=True)
    except Exception:
        pass
    ff = "none"
    energies = None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mh, numThreads=1, maxIters=max_iters)]
            ff = "MMFF"
        elif AllChem.UFFHasAllMoleculeParams(mh):
            energies = [e for _, e in AllChem.UFFOptimizeMoleculeConfs(mh, numThreads=1, maxIters=max_iters)]
            ff = "UFF"
    except Exception:
        energies, ff = None, "none"
    confs = list(mh.GetConformers())
    if energies is None or len(energies) != len(confs) or not np.isfinite(energies).any():
        best = confs[0]
    else:
        e = np.where(np.isfinite(energies), energies, np.inf)
        best = confs[int(np.argmin(e))]
    pts = np.asarray(best.GetPositions(), dtype=np.float64)
    m = np.array([a.GetMass() for a in mh.GetAtoms()], dtype=np.float64)
    x = pts - (pts * m[:, None]).sum(0) / m.sum()
    inertia = np.eye(3) * float((m * (x ** 2).sum(1)).sum()) - (x.T * m) @ x
    return np.clip(np.linalg.eigvalsh(inertia), 1e-3, None), ff
def compute_entropy_v2_terms(mol: Chem.Mol, n_conf: int = 10, seed: int = 42) -> dict[str, float | str]:
    """Values of ENTROPY_V2_NAMES for one molecule (+ 'ff', the force field used). Never raises."""
    out: dict[str, float | str] = {"S_trans_R": 0.0, "S_rot_R": 0.0, "log_I_rot_min": 0.0, "moi2_invalid": 1.0,
                                   "ff": "none"}
    try:
        mh = Chem.AddHs(mol)
        n_at = mh.GetNumAtoms()
        out["S_trans_R"] = s_trans_over_R(sum(a.GetMass() for a in mh.GetAtoms()))
        if n_at <= 1:
            out["moi2_invalid"] = 0.0
            return out
        linear = is_linear_fast(mol) >= 0.5
        ev, ff = lowest_conformer_moments(mol, n_conf=n_conf, seed=seed)
        out["ff"] = ff
        if ev is None:
            return out
        out["S_rot_R"] = s_rot_over_R(ev, linear)
        out["log_I_rot_min"] = float(np.log(ev[2])) if linear else float(0.5 * np.log(ev).sum())
        out["moi2_invalid"] = 0.0
    except Exception:
        pass
    return out
def entropy_v2_from_smiles(smiles: str, n_conf: int = 10, seed: int = 42) -> dict | None:
    """Multiprocessing helper. None for an invalid SMILES."""
    mol = Chem.MolFromSmiles(str(smiles).strip()) if isinstance(smiles, str) else None
    if mol is None:
        return None
    d = compute_entropy_v2_terms(mol, n_conf=n_conf, seed=seed)
    d["smiles"] = str(smiles).strip()
    return d
def entropy_v2_cache_path(csv_path) -> Path:
    """<dir>/<stem>.entropy_v2.csv next to the dataset CSV."""
    p = Path(csv_path)
    return p.with_name(f"{p.stem}.entropy_v2.csv")
def load_or_compute_entropy_v2(smiles: list[str], cache_path=None, n_conf: int = 10, seed: int = 42,
                               workers: int = 8) -> np.ndarray:
    """ENTROPY_V2_NAMES for each SMILES as a float32 array (n, 4), in the order of `smiles`.

    Values are read from / added to `cache_path` (a CSV, see entropy_v2_cache_path; None = no cache) and
    recomputed when the cache was built with another ENTROPY_V2_VERSION or n_conf. A SMILES that cannot be
    processed gets zeros with moi2_invalid = 1 (the prior then treats it like a failed embedding)."""
    import pandas as pd
    from functools import partial
    smiles = [str(s).strip() for s in smiles]
    old = pd.DataFrame()
    if cache_path is not None and Path(cache_path).exists():
        old = pd.read_csv(cache_path)
        stale = ((old.get("version", pd.Series(dtype=str)).astype(str) != ENTROPY_V2_VERSION).any()
                 or (old.get("n_conf", pd.Series(dtype=float)) != n_conf).any())
        if stale:
            old = pd.DataFrame()
    have = set(old["smiles"].astype(str)) if not old.empty else set()
    todo = sorted({s for s in smiles if s not in have})
    if todo:
        fn = partial(entropy_v2_from_smiles, n_conf=n_conf, seed=seed)
        if workers > 1 and len(todo) > 20:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                res = pool.map(fn, todo, chunksize=8)
        else:
            res = [fn(s) for s in todo]
        new = pd.DataFrame([r for r in res if r is not None])
        if not new.empty:
            new["version"], new["n_conf"] = ENTROPY_V2_VERSION, n_conf
            old = pd.concat([old, new], ignore_index=True) if not old.empty else new
            if cache_path is not None:
                old.to_csv(cache_path, index=False, float_format="%.6f")
    table = old.drop_duplicates("smiles").set_index("smiles") if not old.empty else pd.DataFrame(
        columns=ENTROPY_V2_NAMES)
    out = table.reindex(smiles)[ENTROPY_V2_NAMES].astype(float)
    bad = out.isna().any(axis=1)
    out.loc[bad, :] = 0.0
    out.loc[bad, "moi2_invalid"] = 1.0
    return out.to_numpy(np.float32)
