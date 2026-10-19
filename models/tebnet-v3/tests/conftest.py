"""Shared fixtures for the CH2-homolog consistency regulariser.

The suite runs against a 20-species toy split built in-process, so it needs no data file
and no GPU.  The numbers are real tabulated values (NIST/DeepTherm-like, SI units), which
matters: a synthetic *perfectly* linear series would make every increment identical and
therefore every tolerance zero -- a degenerate case that is tested separately in
``test_homologs.py`` precisely because it is not representative.

Run from ``models/tebnet-v3``::

    PYTHONPATH=. ../../.venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 18-point DeepTherm-style grid would be slow to featurise 20 molecules for; the tests only
# need the 300 K column (``cp_index`` auto-selection) and a couple of others.
T_GRID = np.array([298.15, 300.0, 400.0, 500.0, 1000.0, 1500.0], dtype=np.float64)

# smiles -> (dHf298 J/mol, S298 J/(mol K), Cp300 J/(mol K), dCp/dT J/(mol K^2))
_RECORDS: dict[str, tuple[float, float, float, float]] = {
    # n-alkanes: the C4|C4 chain-extension class
    "C": (-74600.0, 186.25, 35.69, 0.030),
    "CC": (-83820.0, 229.60, 52.50, 0.045),
    "CCC": (-104680.0, 269.90, 73.60, 0.060),
    "CCCC": (-125790.0, 310.20, 98.50, 0.075),
    "CCCCC": (-146760.0, 348.40, 120.00, 0.090),
    "CCCCCC": (-167190.0, 388.80, 143.10, 0.105),
    "CCCCCCC": (-187780.0, 427.90, 165.60, 0.120),
    "CCCCCCCC": (-208400.0, 466.70, 188.40, 0.135),
    # 1-alkanols: the C4|O4 class (extension onto an sp3 oxygen)
    "CO": (-201000.0, 239.80, 43.90, 0.035),
    "CCO": (-235100.0, 282.70, 65.40, 0.050),
    "CCCO": (-255200.0, 322.60, 85.40, 0.065),
    "CCCCO": (-274900.0, 362.80, 105.00, 0.080),
    # branched species: extension at a secondary carbon / onto the alcohol oxygen
    "CC(C)O": (-272600.0, 309.20, 89.30, 0.070),
    "CC(C)CO": (-283700.0, 352.00, 110.00, 0.085),
    "CC(C)C": (-134500.0, 294.60, 96.70, 0.075),
    "CC(C)CC": (-154500.0, 343.90, 118.80, 0.090),
    # heteroatom chains that still have an eligible C-C bond
    "CCN": (-47000.0, 283.10, 73.30, 0.060),
    "CCCN": (-70000.0, 322.60, 94.00, 0.075),
    "CCOC": (-252000.0, 342.70, 113.60, 0.085),
    "CCOCC": (-272000.0, 380.30, 134.00, 0.100),
}


def _records() -> list[dict]:
    """Dataset records with a Cp grid on :data:`T_GRID`."""
    out: list[dict] = []
    for smiles, (h, s, cp300, slope) in _RECORDS.items():
        cp = cp300 + slope * (T_GRID - 300.0)
        out.append({"smiles": smiles, "dH_f_298": h, "S_298": s,
                    "Cp": cp.tolist(), "weight": 1.0})
    return out


@pytest.fixture(scope="session")
def t_grid() -> np.ndarray:
    return T_GRID


@pytest.fixture(scope="session")
def records() -> list[dict]:
    return _records()


@pytest.fixture(scope="session")
def dataset(records):
    """20-species training split with v15 physics features (no 3-D descriptors)."""
    from tebnet.data.dataset import ThermochemDataset

    return ThermochemDataset(
        records, T_grid=T_GRID, aux_feature_set="v15",
        compute_aux=True, compute_comp=False, use_3d=False,
    )


@pytest.fixture(scope="session")
def table(dataset):
    """Pair table for :func:`dataset`, Cp column auto-selected (nearest to 300 K)."""
    from tebnet.data.homologs import build_homolog_pairs

    return build_homolog_pairs(dataset)


@pytest.fixture(scope="session")
def atom_featurizer():
    from tebnet.data.featurizer import ThermoAtomFeaturizerV3

    return ThermoAtomFeaturizerV3.thermo()


@pytest.fixture()
def collater(atom_featurizer):
    """Fresh collater per test: the graph cache is state, and tests must not share it."""
    from tebnet.data.collate import ThermoCollater

    return ThermoCollater(atom_featurizer=atom_featurizer)


@pytest.fixture()
def model(atom_featurizer):
    """Small TEBNetV20 with the homolog term enabled but no baselines fitted.

    ``batch_norm=False`` keeps single-item batches legal, so a test can exercise one pair.

    Function-scoped on purpose: the tests mutate hyper-parameters (``homolog_weight``,
    ``homolog_ramp_epochs``) and run ``backward()``, so a shared instance would make the
    suite order-dependent and could leak stale gradients into the next test.
    """
    from tebnet.data.featurizer import (CleanBondFeaturizer, V15_NAMES, v15_base_names,
                                        v15_route_names)
    from tebnet.models.tebnet import TEBNetV20

    return TEBNetV20(
        feature_names=list(V15_NAMES),
        route_h=v15_route_names("h", "grouped", False),
        route_s=v15_route_names("s", "grouped", False),
        route_cp=v15_route_names("cp", "grouped", False),
        base_h=v15_base_names("h", "physics", False),
        base_s=v15_base_names("s", "physics", False),
        base_cp=v15_base_names("cp", "physics", False),
        mp_hidden_dim=16, mp_depth=2, mp_dropout=0.0,
        branch_hidden=16, branch_layers=2, branch_dropout=0.0,
        cp_branch_hidden=16, cp_branch_dropout=0.0,
        batch_norm=False,
        d_v=len(atom_featurizer), d_e=len(CleanBondFeaturizer()),
        t_low_dom=300.0, t_mid=1000.0, t_high_dom=3000.0,
        homolog_weight=0.5, homolog_ramp_epochs=4, homolog_cp_index=1,
    )
