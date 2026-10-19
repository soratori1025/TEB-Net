"""Tests for the paired forward pass, the batch split and the CombinedLoader integration.

The integration tests drive a real (tiny) ``lightning.Trainer`` so that the two pieces most
likely to break silently are exercised: ``CombinedLoader(mode="max_size_cycle")`` handing
``training_step`` a dict, and ``transfer_batch_to_device`` moving a ``HomologPairBatch``
(whose members are themselves dataclasses, not tensors) to the accelerator.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tebnet.data.collate import HomologPairBatch, HomologPairCollater, ThermoCollater
from tebnet.data.homologs import HomologPairDataset


def _pair_batch(table, dataset, collater, idx=None):
    ds = HomologPairDataset(dataset, table)
    items = [ds[i] for i in range(len(ds))] if idx is None else [ds[i] for i in idx]
    return HomologPairCollater(base=collater, table=table)(items)


# ────────────────────────────── forward dispatch ──────────────────────────────
def test_forward_dispatches_on_the_batch_type(model, table, dataset, collater):
    """A paired batch goes through the same ``__call__`` as a molecule batch."""
    pb = _pair_batch(table, dataset, collater, idx=[0])
    out = model(pb)
    assert set(out) == {"anchor", "partner"}
    for member in ("anchor", "partner"):
        assert set(out[member]) == {"cp", "H", "S", "coeffs_low", "coeffs_high",
                                    "dh_f_298", "s_298"}
        assert out[member]["cp"].shape[0] == 1
        assert out[member]["dh_f_298"].shape == (1,)
        assert torch.isfinite(out[member]["cp"]).all()


def test_paired_forward_equals_two_single_forwards(model, table, dataset, collater):
    """The dispatch must not change the numbers: it is the same encoder, called twice."""
    ds = HomologPairDataset(dataset, table)
    items = [ds[0], ds[1]]
    pb = HomologPairCollater(base=collater, table=table)(items)
    paired = model(pb)
    single = model(collater([it["anchor"] for it in items]))
    assert torch.allclose(paired["anchor"]["dh_f_298"], single["dh_f_298"], rtol=1e-5, atol=1e-4)
    assert torch.allclose(paired["anchor"]["cp"], single["cp"], rtol=1e-5, atol=1e-4)


def test_homolog_loss_requires_the_priors(model, table, dataset, collater):
    ds = HomologPairDataset(dataset, table)
    pb = HomologPairCollater(base=collater)([ds[0]])
    with pytest.raises(ValueError, match="increments/tolerances"):
        model.homolog_loss(pb)


def test_homolog_loss_is_differentiable_end_to_end(model, table, dataset, collater):
    """Gradients must reach the encoder through *both* members of the pair."""
    model.zero_grad(set_to_none=True)
    diag = model.homolog_loss(_pair_batch(table, dataset, collater))
    assert torch.isfinite(diag["total"])
    diag["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient from the consistency term"
    assert all(torch.isfinite(g).all() for g in grads)


def test_homolog_weight_ramp(model):
    model.homolog_weight = 0.8
    model.homolog_ramp_epochs = 4
    assert model.homolog_weight_at(0) == pytest.approx(0.2)
    assert model.homolog_weight_at(1) == pytest.approx(0.4)
    assert model.homolog_weight_at(3) == pytest.approx(0.8)
    assert model.homolog_weight_at(10) == pytest.approx(0.8)     # clamped, never overshoots
    model.homolog_ramp_epochs = 0                                 # ramp disabled -> full weight
    assert model.homolog_weight_at(0) == pytest.approx(0.8)
    model.homolog_weight = 0.0                                    # term off
    assert model.homolog_weight_at(99) == 0.0


# ────────────────────────────── batch split ──────────────────────────────
def test_split_homolog_batch_accepts_every_batch_shape(table, dataset, collater):
    from tebnet.models.tebnet import TEBNetV20

    ds = HomologPairDataset(dataset, table)
    main = collater([ds[0]["anchor"]])
    pairs = HomologPairCollater(base=collater, table=table)([ds[0]])

    assert TEBNetV20.split_homolog_batch(main) == (main, None)
    assert TEBNetV20.split_homolog_batch(pairs) == (None, pairs)
    assert TEBNetV20.split_homolog_batch({"main": main, "pairs": pairs}) == (main, pairs)
    assert TEBNetV20.split_homolog_batch({"main": main}) == (main, None)
    assert TEBNetV20.split_homolog_batch({"pairs": pairs}) == (None, pairs)

def test_split_homolog_batch_rejects_a_mistyped_dict(table, dataset, collater):
    from tebnet.models.tebnet import TEBNetV20

    main = collater([dataset[0]])
    with pytest.raises(TypeError, match="batch\\['main'\\]"):
        TEBNetV20.split_homolog_batch({"main": {"not": "a batch"}})
    with pytest.raises(TypeError, match="batch\\['pairs'\\]"):
        TEBNetV20.split_homolog_batch({"main": main, "pairs": main})


def test_transfer_batch_to_device_handles_a_paired_batch(model, table, dataset, collater):
    pb = _pair_batch(table, dataset, collater, idx=[0])
    moved = model.transfer_batch_to_device(pb, torch.device("cpu"), 0)
    assert isinstance(moved, HomologPairBatch)
    assert moved.anchor.cp_grid.device.type == "cpu"
    assert moved.increments.device.type == "cpu"
    # an ordinary batch still takes the same path
    main = collater([dataset[0]])
    assert model.transfer_batch_to_device(main, torch.device("cpu"), 0).smiles == main.smiles


# ────────────────────────────── CombinedLoader integration ──────────────────────────────
def _loaders(dataset, table, collater, batch_size: int = 4, repeat: int = 1):
    """The two loaders exactly as ``scripts/train.py`` builds them."""
    main = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                      collate_fn=collater)
    pairs = DataLoader(HomologPairDataset(dataset, table, repeat=repeat),
                       batch_size=batch_size, shuffle=False, drop_last=True,
                       collate_fn=HomologPairCollater(base=collater, table=table))
    return main, pairs


def test_combined_loader_yields_a_dict_with_both_objectives(dataset, table, collater):
    """``mode="max_size_cycle"`` must give one dict per step, never a bare batch."""
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    main, pairs = _loaders(dataset, table, collater)
    combined = CombinedLoader({"main": main, "pairs": pairs}, mode="max_size_cycle")
    steps = 0
    # iterating yields (batch, batch_idx, dataloader_idx); the Trainer passes only `batch`
    for batch, batch_idx, _ in combined:
        assert isinstance(batch, dict) and set(batch) == {"main", "pairs"}
        assert batch_idx == steps
        assert batch["main"].x_physic.shape[1] == batch["pairs"].anchor.x_physic.shape[1]
        steps += 1
    # the step count is set by the *longer* loader, so the shorter one is cycled
    assert steps == max(len(main), len(pairs))


def test_combined_loader_cycles_the_pair_loader_when_it_is_shorter(dataset, table, collater):
    """A pair set smaller than the molecule set must still be covered every epoch."""
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    main, pairs = _loaders(dataset, table, collater, batch_size=1)
    combined = CombinedLoader({"main": main, "pairs": pairs}, mode="max_size_cycle")
    # ``len()`` is only valid once the iterator exists -- it is computed from the mode
    with pytest.raises(RuntimeError, match="call `iter"):
        len(combined)
    it = iter(combined)
    assert len(combined) == max(len(main), len(pairs))
    seen = 0
    for batch, _, _ in it:
        # the pair loader is restarted from its first batch once exhausted, so no
        # ``pairs`` slot may ever be None in this mode (unlike mode="max_size")
        assert batch["pairs"] is not None
        seen += len(batch["pairs"])
    assert seen >= len(table)                              # every pair appears at least once


def test_training_step_combines_the_two_losses(model, dataset, table, collater):
    """The returned loss must include the pair term when the weight is non-zero."""
    import warnings

    model.homolog_ramp_epochs = 0
    main = collater([dataset[i] for i in range(4)])
    pairs = _pair_batch(table, dataset, collater, idx=[0, 1, 2, 3])
    combined = {"main": main, "pairs": pairs}

    # outside a Trainer ``self.log`` has nowhere to go and Lightning says so once; the
    # step itself must still work, which is exactly what is being asserted here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.homolog_weight = 0.0
        only_main = model.training_step(combined, 0)
        model.homolog_weight = 0.5
        with_pairs = model.training_step(combined, 0)
    assert torch.isfinite(only_main) and torch.isfinite(with_pairs)
    assert not torch.allclose(only_main, with_pairs)
    # the difference must be exactly the weighted penalty of this batch, up to the float32
    # resolution of adding it to the (much larger) per-molecule loss
    model.homolog_weight = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        penalty = model._homolog_term(pairs, "train")
    eps = float(only_main.abs().item()) * torch.finfo(torch.float32).eps
    assert (with_pairs - only_main).item() == pytest.approx(penalty.item(), abs=2 * eps)
    # ...and the term is genuinely small next to the data loss, i.e. a regulariser
    assert abs(penalty.item()) < abs(only_main.item())


def test_training_step_rejects_an_empty_combined_batch(model):
    with pytest.raises(RuntimeError, match="neither molecules nor homolog pairs"):
        model.training_step({}, 0)


def test_trainer_fit_runs_one_epoch_with_the_regulariser(dataset, table, collater, model,
                                                         tmp_path):
    """End-to-end smoke test: fit() must accept the CombinedLoader and log the prior."""
    import lightning.pytorch as pl

    main, pairs = _loaders(dataset, table, collater, batch_size=4)
    combined = pl.utilities.combined_loader.CombinedLoader(
        {"main": main, "pairs": pairs}, mode="max_size_cycle")
    trainer = pl.Trainer(
        max_epochs=1, limit_train_batches=2, limit_val_batches=0,
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    model.homolog_weight = 0.5
    model.homolog_ramp_epochs = 0
    trainer.fit(model, train_dataloaders=combined)
    # the regulariser actually ran and its diagnostics were recorded
    assert "train/homolog_loss" in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics["train/homolog_loss"])
    assert trainer.callback_metrics["train/homolog_weight"].item() == pytest.approx(0.5)

