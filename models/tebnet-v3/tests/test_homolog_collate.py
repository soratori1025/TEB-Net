"""Tests for the paired collater and the CH2-homolog consistency penalty.

Two invariants matter here.  The pair branch must featurise exactly like the per-molecule
branch (otherwise part of the "inconsistency" is a featurisation artefact), and the
penalty must be a well-behaved function of the model output: finite, differentiable, and
increasing with the deviation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tebnet.data.collate import HomologPairBatch, HomologPairCollater, ThermoCollater
from tebnet.data.homologs import HomologPairDataset
from tebnet.nn.loss import HomologConsistencyLoss


def _batch(table, dataset, collater, idx=None):
    ds = HomologPairDataset(dataset, table)
    items = [ds[i] for i in range(len(ds))] if idx is None else [ds[i] for i in idx]
    return HomologPairCollater(base=collater, table=table)(items)


# ────────────────────────────── collation ──────────────────────────────
def test_collater_round_trips_both_members(table, dataset, collater):
    pb = _batch(table, dataset, collater)
    assert isinstance(pb, HomologPairBatch)
    assert len(pb) == len(table)
    # both members are ordinary batches with the same batch size
    assert len(pb.anchor) == len(pb.partner) == len(pb)
    assert pb.anchor.cp_grid.shape == pb.partner.cp_grid.shape
    assert pb.anchor.x_physic.shape[1] == pb.partner.x_physic.shape[1]
    # the targets are the real tabulated values, not the increment
    assert not torch.allclose(pb.anchor.dh_f_298, pb.partner.dh_f_298)


def test_collater_attaches_the_increment_and_tolerance_of_each_class(table, dataset, collater):
    pb = _batch(table, dataset, collater)
    assert pb.increments is not None and pb.tolerances is not None
    assert pb.increments.shape == (len(pb), 3)
    assert pb.tolerances.shape == (len(pb), 3)
    for i, k in enumerate(pb.cls_idx.tolist()):
        assert torch.allclose(pb.increments[i], torch.as_tensor(table.increments[k], dtype=torch.float32))
        assert torch.allclose(pb.tolerances[i], torch.as_tensor(table.tolerances[k], dtype=torch.float32))


def test_collater_without_a_table_omits_the_priors(table, dataset, collater, model):
    """The priors are optional, and a missing one must be reported, not defaulted to zero."""
    ds = HomologPairDataset(dataset, table)
    pb = HomologPairCollater(base=collater)([ds[0]])
    assert pb.increments is None and pb.tolerances is None
    with pytest.raises(ValueError, match="increments/tolerances"):
        model.homolog_loss(pb)


def test_pair_batch_to_device_moves_every_tensor(table, dataset, collater):
    pb = _batch(table, dataset, collater)
    moved = pb.to(torch.device("cpu"))
    assert moved is not pb
    assert moved.anchor.cp_grid.device.type == "cpu"
    assert moved.increments.device.type == "cpu"
    assert moved.cls_idx.tolist() == pb.cls_idx.tolist()
    assert moved.anchor.smiles == pb.anchor.smiles


def test_collater_reuses_the_shared_graph_cache(table, dataset, collater):
    """Pair and molecule branches must not featurise the same species twice, differently."""
    ds = HomologPairDataset(dataset, table)
    item = ds[0]
    single = collater([item["anchor"]])          # populates the cache for the anchor
    n_cached = len(collater._graph_cache)
    assert n_cached >= 1
    pair = HomologPairCollater(base=collater, table=table)([item])
    # only the partner (and any species new to this batch) may be added -- the anchor was hit
    assert len(collater._graph_cache) <= n_cached + 2
    assert pair.anchor.bmg.V.shape == single.bmg.V.shape
    assert torch.allclose(pair.anchor.bmg.V, single.bmg.V)


# ────────────────────────────── penalty ──────────────────────────────
def _val(t) -> float:
    """Scalar value of a (possibly grad-tracking) 0-d tensor."""
    return float(t.detach())


def _preds(b: int, n_t: int, h: float, s: float, cp: float) -> dict[str, torch.Tensor]:
    return {"dh_f_298": torch.full((b,), h, requires_grad=True),
            "s_298": torch.full((b,), s, requires_grad=True),
            "cp": torch.full((b, n_t), cp, requires_grad=True)}


def test_penalty_is_zero_at_the_prior_increment():
    """If the model reproduces the increment exactly, the regulariser contributes nothing."""
    loss = HomologConsistencyLoss(cp_index=1)
    inc = torch.tensor([[-20000.0, 40.0, 20.0]])
    tol = torch.tensor([[1000.0, 4.0, 2.0]])
    a = _preds(1, 3, 0.0, 0.0, 0.0)
    p = _preds(1, 3, -20000.0, 40.0, 20.0)
    out = loss(a, p, inc, tol)
    assert _val(out["total"]) == pytest.approx(0.0, abs=1e-8)
    assert _val(out["r_h"]) == pytest.approx(0.0, abs=1e-6)


def test_penalty_grows_with_the_deviation():
    """Monotone in the normalised residual: twice the error, more than twice the penalty."""
    loss = HomologConsistencyLoss(cp_index=0, w_s=0.0, w_cp=0.0)
    inc = torch.tensor([[-20000.0, 0.0, 0.0]])
    tol = torch.tensor([[1000.0, 1.0, 1.0]])
    a = _preds(1, 2, 0.0, 0.0, 0.0)
    small = _val(loss(a, _preds(1, 2, -19000.0, 0.0, 0.0), inc, tol)["total"])
    large = _val(loss(a, _preds(1, 2, -18000.0, 0.0, 0.0), inc, tol)["total"])
    assert 0.0 < small < large


def test_penalty_is_quadratic_within_one_tolerance_and_linear_beyond():
    """Pseudo-Huber: ~r^2 near 0, ~2|r|-2 far out.  This is what makes it robust to outliers."""
    loss = HomologConsistencyLoss(cp_index=0, w_s=0.0, w_cp=0.0, beta=1.0)
    inc = torch.tensor([[-20000.0, 0.0, 0.0]])
    tol = torch.tensor([[1000.0, 1.0, 1.0]])
    a = _preds(1, 2, 0.0, 0.0, 0.0)

    near = _val(loss(a, _preds(1, 2, -19900.0, 0.0, 0.0), inc, tol)["total"])   # r = 0.1
    far = _val(loss(a, _preds(1, 2, -15000.0, 0.0, 0.0), inc, tol)["total"])     # r = 5

    # exact: 2 b^2 (sqrt(1 + (r/b)^2) - 1) with b = 1
    assert near == pytest.approx(2.0 * (np.sqrt(1.01) - 1.0), rel=1e-4)
    assert far == pytest.approx(2.0 * (np.sqrt(26.0) - 1.0), rel=1e-4)
    # asymptotic regimes: r^2 close in, 2|r| - 2 far out
    assert near == pytest.approx(0.1 ** 2, rel=1e-2)
    assert far == pytest.approx(2.0 * 5.0 - 2.0, rel=3e-2)

    # and the far branch is linear, not quadratic: 10x the residual -> ~10x the penalty
    farther = _val(loss(a, _preds(1, 2, -10000.0, 0.0, 0.0), inc, tol)["total"])  # r = 10
    assert farther / far == pytest.approx(18.0 / 8.198, rel=2e-2)


def test_penalty_is_invariant_to_a_shared_offset():
    """Only the *difference* between the two members is penalised, never their level."""
    loss = HomologConsistencyLoss(cp_index=0)
    inc = torch.tensor([[-20000.0, 40.0, 20.0]])
    tol = torch.tensor([[1000.0, 4.0, 2.0]])
    a = _preds(2, 3, -100000.0, 200.0, 100.0)
    p = _preds(2, 3, -120000.0, 240.0, 120.0)
    base = _val(loss(a, p, inc, tol)["total"])
    shifted = _val(loss(_preds(2, 3, -90000.0, 210.0, 130.0),
                        _preds(2, 3, -110000.0, 250.0, 150.0), inc, tol)["total"])
    assert base == pytest.approx(0.0, abs=1e-6)
    assert shifted == pytest.approx(base, abs=1e-6)


def test_penalty_ignores_nan_rows_and_reports_the_valid_count():
    """NaN targets (missing tabulated data) must not poison the term or the gradient."""
    loss = HomologConsistencyLoss(cp_index=0)
    inc = torch.tensor([[-20000.0, 40.0, 20.0], [float("nan")] * 3])
    tol = torch.tensor([[1000.0, 4.0, 2.0], [1000.0, 4.0, 2.0]])
    a = _preds(2, 3, 0.0, 0.0, 0.0)
    p = _preds(2, 3, -20000.0, 40.0, 20.0)
    out = loss(a, p, inc, tol)
    assert torch.isfinite(out["total"])
    assert out["n"][0].item() == 1          # one of the two rows is NaN for every property


def test_penalty_keeps_the_graph_on_an_all_nan_batch():
    """A degenerate batch must return a zero that still backpropagates (no None grad)."""
    loss = HomologConsistencyLoss(cp_index=0)
    inc = torch.full((2, 3), float("nan"))
    tol = torch.ones((2, 3))
    a = _preds(2, 3, 0.0, 0.0, 0.0)
    p = _preds(2, 3, -20000.0, 40.0, 20.0)
    out = loss(a, p, inc, tol)
    out["total"].backward()
    for key in ("dh_f_298", "s_298", "cp"):
        assert p[key].grad is not None
        assert torch.isfinite(p[key].grad).all()


def test_penalty_backpropagates_finite_gradients():
    loss = HomologConsistencyLoss(cp_index=1)
    inc = torch.tensor([[-20000.0, 40.0, 20.0]])
    tol = torch.tensor([[1000.0, 4.0, 2.0]])
    a = _preds(1, 3, 0.0, 0.0, 0.0)
    p = _preds(1, 3, -15000.0, 20.0, 5.0)
    out = loss(a, p, inc, tol)
    out["total"].backward()
    assert p["dh_f_298"].grad is not None and torch.isfinite(p["dh_f_298"].grad).all()
    assert p["cp"].grad is not None and torch.isfinite(p["cp"].grad).all()
    # a positive residual (partner too high) pushes the partner prediction down
    assert p["dh_f_298"].grad.item() > 0


def test_penalty_only_reads_cp_at_the_selected_temperature():
    """The increment table has no T dependence; constraining other columns would be invented."""
    loss = HomologConsistencyLoss(cp_index=2)
    inc = torch.tensor([[0.0, 0.0, 20.0]])
    tol = torch.tensor([[1.0, 1.0, 2.0]])
    a = {"dh_f_298": torch.zeros(1), "s_298": torch.zeros(1), "cp": torch.zeros(1, 4)}
    p = {"dh_f_298": torch.zeros(1), "s_298": torch.zeros(1),
         "cp": torch.tensor([[99.0, 99.0, 20.0, 99.0]])}
    assert _val(loss(a, p, inc, tol)["total"]) == pytest.approx(0.0, abs=1e-6)


def test_penalty_rejects_a_cp_index_outside_the_grid():
    loss = HomologConsistencyLoss(cp_index=5)
    a = _preds(1, 3, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="cp_index"):
        loss(a, a, torch.zeros((1, 3)), torch.ones((1, 3)))


def test_property_weights_scale_their_own_property():
    """w_h/w_s/w_cp are the only free weighting; tolerances normalise the rest."""
    inc = torch.tensor([[-20000.0, 40.0, 20.0]])
    tol = torch.tensor([[1000.0, 4.0, 2.0]])
    a = _preds(1, 2, 0.0, 0.0, 0.0)
    p = _preds(1, 2, -19000.0, 40.0, 20.0)     # only dHf is off
    one = HomologConsistencyLoss(cp_index=0, w_h=1.0, w_s=0.0, w_cp=0.0)
    two = HomologConsistencyLoss(cp_index=0, w_h=2.0, w_s=0.0, w_cp=0.0)
    h_only = _val(one(a, p, inc, tol)["total"])
    doubled = _val(two(a, p, inc, tol)["total"])
    assert h_only > 0.0
    assert doubled == pytest.approx(2.0 * h_only, rel=1e-6)


def test_weight_buffers_are_not_persistent():
    """The penalty is a hyperparameter, not model state: old checkpoints must still load."""
    assert "w" not in HomologConsistencyLoss().state_dict()


def test_penalty_is_invariant_to_the_batch_size():
    """A mean over a batch must not grow when the same batch is replicated.

    Regression test: the property weights were once broadcast against the *already reduced*
    per-property means, which produced a (B, 3) tensor whose sum counted every property B
    times.  The penalty then scaled linearly with the batch size (32x at the training batch
    size), swamping the per-molecule loss and preventing the model from learning at all.
    """
    loss = HomologConsistencyLoss(cp_index=0)
    inc1 = torch.tensor([[-20000.0, 40.0, 20.0]])
    tol1 = torch.tensor([[1000.0, 4.0, 2.0]])

    a1 = _preds(1, 3, 0.0, 0.0, 0.0)
    p1 = _preds(1, 3, -19000.0, 44.0, 23.0)
    one = _val(loss(a1, p1, inc1, tol1)["total"])

    # The same single pair, replicated B times, must give exactly the same penalty.
    for b in (2, 8, 32):
        a = {k: v.repeat(b, *([1] * (v.dim() - 1))) for k, v in a1.items()}
        p = {k: v.repeat(b, *([1] * (v.dim() - 1))) for k, v in p1.items()}
        inc = inc1.repeat(b, 1)
        tol = tol1.repeat(b, 1)
        got = _val(loss(a, p, inc, tol)["total"])
        assert got == pytest.approx(one, rel=1e-6), f"penalty grew with batch size {b}"


def test_total_equals_the_sum_of_the_per_property_contributions():
    """``total`` is the weighted sum of the three logged parts -- the logged diagnostics
    must add up to the term that is actually added to the loss."""
    loss = HomologConsistencyLoss(cp_index=0)
    b = 4
    inc = torch.tensor([[-20000.0, 40.0, 20.0]]).repeat(b, 1)
    tol = torch.tensor([[1000.0, 4.0, 2.0]]).repeat(b, 1)
    a = _preds(b, 3, 0.0, 0.0, 0.0)
    p = _preds(b, 3, -18500.0, 35.0, 18.0)
    out = loss(a, p, inc, tol)
    parts = sum(_val(out[k]) for k in ("h", "s", "cp"))
    assert _val(out["total"]) == pytest.approx(parts, rel=1e-6)

