from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def canonicalize_smiles(smiles: str, remove_stereo: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=not remove_stereo)

def get_default_temperature_grid() -> np.ndarray:
    return np.array([
        200.0, 250.0, 298.15, 300.0, 400.0, 500.0, 600.0, 700.0,
        800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0,
        2000.0, 2500.0, 3000.0,
    ], dtype=np.float64)

def smiles_to_mol_with_radicals(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def compute_molecular_formula(mol: Chem.Mol) -> dict[str, int]:
    mol_h = Chem.AddHs(mol)
    formula = {}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        formula[sym] = formula.get(sym, 0) + 1
    return formula
