"""Tests for the CH2-homolog pair enumeration and increment table.

The module encodes a chemistry claim (a single methylene insertion is a homologation, and
its thermochemical increment is transferable), so the tests pin the two things that can
silently go wrong: the insertion filter accepting a bond that is *not* a chain extension,
and the increment table producing a tolerance of zero (which would turn a soft prior into
a hard equality).
"""
from __future__ import annotations
import numpy as np
import pytest
from rdkit import Chem
from tebnet.data.homologs import (
    TOL_MIN,
    HomologPairDataset,
    HomologPairTable,
    build_homolog_pairs,
    canonical_smiles,
    insert_ch2,
    insertion_allowed,
    site_class,
)
@pytest.mark.parametrize("smiles", ["CCCC", "CCCCO", "CC(C)C", "CCN", "CCOC"])
def test_saturated_chain_bonds_are_eligible(smiles):
    """Chain extension is what the prior is for: every acyclic saturated C-C/C-X bond counts."""
    mol = Chem.MolFromSmiles(smiles)
    bonds = [(b.GetBeginAtom(), b.GetEndAtom()) for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    assert bonds, f"{smiles} has no acyclic single bond"
    assert any(insertion_allowed(a1, a2) for a1, a2 in bonds), smiles
def test_ester_bond_is_rejected():
    """Inserting into C(=O)-O destroys the ester; the measured increment collapses to ~0."""
    mol = Chem.MolFromSmiles("CC(=O)OC")
    carbonyl = mol.GetAtomWithIdx(2)
    ester_o = mol.GetAtomWithIdx(3)
    assert not insertion_allowed(carbonyl, ester_o)
def test_alkoxy_oxygen_of_an_ester_is_also_rejected():
    """RDKit reports the alkoxy O of an ester as SP2; the filter must not rely on that."""
    mol = Chem.MolFromSmiles("CC(=O)OCC")
    ester_o = mol.GetAtomWithIdx(3)
    alkyl_c = mol.GetAtomWithIdx(4)
    assert not insertion_allowed(ester_o, alkyl_c)
def test_peroxide_bond_is_rejected():
    mol = Chem.MolFromSmiles("CCOOC")
    o1, o2 = mol.GetAtomWithIdx(2), mol.GetAtomWithIdx(3)
    assert not insertion_allowed(o1, o2)
def test_two_unsaturated_atoms_are_rejected():
    """A diene insertion breaks the conjugation pattern rather than extending a chain."""
    mol = Chem.MolFromSmiles("C=CC=C")
    c1, c2 = mol.GetAtomWithIdx(1), mol.GetAtomWithIdx(2)
    assert not insertion_allowed(c1, c2)
def test_bond_to_hydrogen_is_rejected():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    h = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    c = h.GetNeighbors()[0]
    assert not insertion_allowed(c, h)
def test_radical_centre_is_rejected():
    mol = Chem.MolFromSmiles("[CH2]CC")
    assert not insertion_allowed(mol.GetAtomWithIdx(0), mol.GetAtomWithIdx(1))
def test_insert_ch2_grows_the_chain_by_one_carbon():
    mol = Chem.MolFromSmiles("CCC")
    products = {smi for smi, _ in insert_ch2(mol)}
    assert products == {"CCCC"}
def test_insert_ch2_labels_the_flanking_site_classes():
    """The class is the *bond* that was extended, so it survives the dedup of products."""
    assert insert_ch2(Chem.MolFromSmiles("CC")) == [("CCC", "C4|C4")]
    assert insert_ch2(Chem.MolFromSmiles("CO")) == [("CCO", "C4|O4")]
    assert insert_ch2(Chem.MolFromSmiles("CCO")) == [("CCCO", "C4|C4")]
def test_insert_ch2_deduplicates_symmetry_equivalent_sites():
    """Every C-C bond of ethane gives the same product; the pair must appear once."""
    assert len(insert_ch2(Chem.MolFromSmiles("CC"))) == 1
    assert len(insert_ch2(Chem.MolFromSmiles("CCC"))) == 1
def test_insert_ch2_never_touches_ring_bonds():
    assert insert_ch2(Chem.MolFromSmiles("C1CCCCC1")) == []
    assert len(insert_ch2(Chem.MolFromSmiles("CC1CCCCC1"))) == 1
def test_site_class_distinguishes_pi_state():
    mol = Chem.MolFromSmiles("c1ccccc1C=CC")
    classes = [site_class(a) for a in mol.GetAtoms()]
    assert "C3b" in classes
    assert "C3" in classes
    assert "C4" in classes
def test_canonical_smiles_is_isomer_free():
    assert canonical_smiles("C/C=C/C") == canonical_smiles("CC=CC")
def test_table_finds_the_expected_pairs(table):
    assert len(table) > 0
    assert table.n_classes == len(table.classes)
    assert (table.partner_idx != table.anchor_idx).all()
    assert table.cls_idx.shape == (len(table),)
    assert table.increments.shape == (table.n_classes, 3)
    assert table.tolerances.shape == (table.n_classes, 3)
    assert table.n_pairs_total == len(table) + table.n_pairs_skipped
def test_increment_signs_match_the_physics(table):
    """Adding a CH2 lowers dHf, raises S and raises Cp -- the signs are universal."""
    inc = table.increments
    assert (inc[:, 0] < 0).all(), "dHf increment should be negative"
    assert (inc[:, 1] > 0).all(), "S increment should be positive"
    assert (inc[:, 2] > 0).all(), "Cp increment should be positive"
    assert np.all(inc[:, 0] > -15.0 * 4184.0) and np.all(inc[:, 0] < -2.0 * 4184.0)
    assert np.all(inc[:, 1] > 3.0 * 4.184) and np.all(inc[:, 1] < 25.0 * 4.184)
def test_tolerances_are_positive_and_finite(table):
    assert np.isfinite(table.tolerances).all()
    assert (table.tolerances > 0).all()
    assert (table.tolerances >= TOL_MIN).all()
def test_tolerance_is_floored_when_the_split_is_degenerate():
    """A perfectly linear homolog series has zero spread; the tolerance must not be 0.

    Without the floor the normalised residual ``resid / tol`` reaches ~1e9 and the penalty
    stops being a soft constraint, which is how this was found.
    """
    from tebnet.data.dataset import ThermochemDataset
    linear = [
        {"smiles": "C" * n,
         "dH_f_298": -74600.0 - 21000.0 * i,
         "S_298": 186.0 + 40.0 * i,
         "Cp": ((35.7 + 20.0 * i) + 0.05 * (np.array([298.15, 300.0, 400.0]) - 300.0)).tolist(),
         "weight": 1.0}
        for i, n in enumerate(range(1, 7))
    ]
    ds = ThermochemDataset(linear, T_grid=np.array([298.15, 300.0, 400.0]),
                           aux_feature_set="v15", compute_aux=False,
                           compute_comp=False, use_3d=False)
    tbl = build_homolog_pairs(ds)
    assert len(tbl) > 0
    assert (tbl.tolerances >= TOL_MIN).all()
    assert (tbl.tolerances > 0).all()
def test_empty_table_is_safe_and_does_not_raise():
    from tebnet.data.dataset import ThermochemDataset
    ds = ThermochemDataset(
        [{"smiles": "C1CCCCC1", "dH_f_298": -123000.0, "S_298": 298.0,
          "Cp": [100.0, 101.0], "weight": 1.0}],
        T_grid=np.array([300.0, 400.0]), aux_feature_set="v15",
        compute_aux=False, compute_comp=False, use_3d=False,
    )
    tbl = build_homolog_pairs(ds)
    assert len(tbl) == 0
    assert tbl.n_classes == 0
    assert tbl.tolerances.shape == (0, 3)
    assert "0 CH2-homolog pairs" in tbl.summary()
    assert tbl.to_json()["n_pairs"] == 0
    assert len(HomologPairDataset(ds, tbl)) == 0
def test_shrinkage_pulls_small_classes_towards_the_global_increment(dataset):
    """n_shrink is a prior strength: with n_shrink=0 the class medians are untouched."""
    raw = build_homolog_pairs(dataset, n_shrink=0.0)
    shrunk = build_homolog_pairs(dataset, n_shrink=1e6)
    assert not np.allclose(raw.increments, shrunk.increments)
    for k in range(1, shrunk.n_classes):
        assert np.allclose(shrunk.increments[k], shrunk.increments[0])
def test_table_tensors_and_json_round_trip(table):
    t = table.tensors()
    assert set(t) == {"anchor_idx", "partner_idx", "cls_idx", "increments", "tolerances"}
    assert t["increments"].dtype.is_floating_point
    assert t["increments"].shape == (table.n_classes, 3)
    assert not t["cls_idx"].dtype.is_floating_point
    js = table.to_json()
    assert js["n_pairs"] == len(table)
    assert js["classes"] == table.classes
    assert len(js["increments"]["h_kcal_per_mol"]) == table.n_classes
    assert abs(js["increments"]["h_kcal_per_mol"][0] * 4184.0 - table.increments[0, 0]) < 1.0
def test_pair_dataset_indexes_through_the_training_dataset(table, dataset):
    ds = HomologPairDataset(dataset, table)
    assert len(ds) == len(table)
    item = ds[0]
    assert set(item) == {"anchor", "partner", "cls_idx"}
    a = dataset.datapoints[int(table.anchor_idx[0])]
    assert item["anchor"]["smiles"] == a.smiles
    assert np.allclose(item["anchor"]["x_physic"].numpy(), a.x_physic)
def test_pair_dataset_repeat_multiplies_length_without_changing_content(table, dataset):
    once = HomologPairDataset(dataset, table)
    thrice = HomologPairDataset(dataset, table, repeat=3)
    assert len(thrice) == 3 * len(once)
    assert thrice[len(once)]["anchor"]["smiles"] == once[0]["anchor"]["smiles"]
    assert thrice[len(once)]["cls_idx"] == once[0]["cls_idx"]
def test_empty_pair_table_tensors_have_zero_rows():
    empty = HomologPairTable(
        anchor_idx=np.zeros(0, dtype=np.int64), partner_idx=np.zeros(0, dtype=np.int64),
        cls_idx=np.zeros(0, dtype=np.int64), increments=np.zeros((0, 3)),
        tolerances=np.zeros((0, 3)), classes=[],
    )
    t = empty.tensors()
    assert t["increments"].shape == (0, 3)
    assert t["cls_idx"].shape == (0,)
    assert empty.class_counts() == {}
    assert canonical_smiles("not-a-smiles") is None
