from __future__ import annotations
from rdkit import Chem
from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph
from stereomolgraph.algorithms.symmetry import symmetry_number

# v15: explicit hydrogens for EVERY molecule.
# The old rule (Chem.AddHs only when <= 4 heavy atoms) switched convention at
# 5 heavy atoms: isobutane (4 heavy) got σ = 81 including the 3^3 methyl-rotor
# factor, while n-pentane (5 heavy) only got the heavy-skeleton automorphisms
# (≈ 2 instead of 18). The entropy term −R ln σ then follows two different
# conventions depending on molecular size, which hurts exactly the mid-size
# species. Set ADD_HS_MAX_HEAVY = 4 to reproduce the old behaviour.
ADD_HS_MAX_HEAVY: int | None = None


def compute_symmetry_number(mol: Chem.Mol) -> int:
    if mol is None:
        raise ValueError("Molecule is None")
    mol = Chem.Mol(mol)
    if ADD_HS_MAX_HEAVY is None or mol.GetNumHeavyAtoms() <= ADD_HS_MAX_HEAVY:
        mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
    )
    # v15.3: stereo_complete=True gives every non-stereogenic sp3 centre / double bond a default
    # parity, so only PROPER permutations are counted (ethane 18, not the 72 graph automorphisms).
    # In StereoMolGraph those default parities are applied inside the resonance loop, i.e. they are
    # silently skipped when resonance=False — which is why the previous version over-counted.
    # Same settings as the library's own consistency tests.
    converter = RDMol2StereoMolGraph(
        stereo_complete=True,
        use_atom_map_number=False,
        lone_pair_stereo=False,
        resonance=True,
    )
    graph = converter(mol)
    sigma = symmetry_number(graph)
    if sigma < 1:
        raise ValueError(
            f"Invalid symmetry number returned: {sigma}"
        )
    return int(sigma)


if __name__ == "__main__":
    # Quick self-check: total symmetry numbers (external × internal rotors),
    # the convention used by RMG / Benson group additivity.
    ref = {
        "C": 12, "CC": 18, "CCC": 18, "CC(C)C": 81, "CCCCC": 18, "CC(C)(C)C": 972,
        "c1ccccc1": 12, "O": 2, "N": 3, "O=C=O": 2, "C#C": 2, "[CH3]": 6,
        "C1CCCCC1": 6,   # chair D3d; a purely graph-based count may give 12 (planar D6h)
    }
    import time
    for smi, expected in ref.items():
        t0 = time.time()
        try:
            got = compute_symmetry_number(Chem.MolFromSmiles(smi))
        except Exception as e:
            got = f"ERROR {e}"
        flag = "" if got == expected else "   <-- check"
        print(f"{smi:12s} expected {expected:>4}  got {got!s:>6}  ({(time.time() - t0) * 1000:.0f} ms){flag}")