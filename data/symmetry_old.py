from __future__ import annotations
from rdkit import Chem
from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph
from stereomolgraph.algorithms.symmetry import symmetry_number

def compute_symmetry_number(mol: Chem.Mol) -> int:
    if mol is None:
        raise ValueError("Molecule is None")
    mol = Chem.Mol(mol)
    if mol.GetNumHeavyAtoms() <= 4:
        mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
    )
    converter = RDMol2StereoMolGraph(
        stereo_complete=False,
        use_atom_map_number=False,
        lone_pair_stereo=True,
        resonance=False,
    )
    graph = converter(mol)
    sigma = symmetry_number(graph)
    if sigma < 1:
        raise ValueError(
            f"Invalid symmetry number returned: {sigma}"
        )
    return int(sigma)