from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

def compute_principal_moments(mol: Chem.Mol, *, optimize: bool = True,) -> tuple[float, float, float]:
    if mol is None:
        raise ValueError("Molecule is None")
    mol = Chem.Mol(mol)
    # Add explicit hydrogens because inertia depends on molecular mass.
    mol = Chem.AddHs(mol)
    # Generate a 3D conformer from SMILES.
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("RDKit 3D embedding failed")
    # Geometry optimization.
    if optimize:
        try:
            result = AllChem.MMFFOptimizeMolecule(mol)
            # MMFF may fail for unusual species.
            if result < 0:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            # Fallback to UFF.
            AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    _, moments = rdMolTransforms.ComputePrincipalAxesAndMoments(
        conf,
        ignoreHs=False,
    )
    moments = sorted(float(x) for x in moments)
    return moments[0], moments[1], moments[2]

def compute_is_linear(mol: Chem.Mol, *, relative_tolerance: float = 1e-3,) -> bool:
    I1, I2, I3 = compute_principal_moments(mol)
    if I3 <= 0:
        raise ValueError("Invalid principal moments")
    condition_1 = I1 / I3 < relative_tolerance
    condition_2 = abs(I2 - I3) / I3 < relative_tolerance
    return condition_1 and condition_2