#!/usr/bin/env python3
"""Diagnose the CH2-homolog regulariser's magnitude relative to the main loss.

Reads the TensorBoard scalars written by ``TEBNetV20.training_step`` and prints, per
epoch, the per-molecule loss and the homolog penalty side by side.  The two are added
without rescaling in ``training_step`` (``total + diag["total"] * w``), so this script
answers the only question that matters when the regularised run does not learn: how
large is the penalty compared with the objective it is supposed to perturb?

Usage
-----
    python check/diagnose_homolog_scale.py <run_dir> [--baseline <baseline_run_dir>]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load(run_dir: str) -> EventAccumulator:
    path = Path(run_dir) / "lightning_logs" / "version_0"
    if not path.is_dir():
        cands = sorted(glob.glob(str(Path(run_dir) / "lightning_logs" / "version_*")))
        if not cands:
            sys.exit(f"no lightning_logs/version_* under {run_dir}")
        path = Path(cands[-1])
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _by_epoch(ea: EventAccumulator, tag: str) -> dict[int, float]:
    """Map epoch -> mean value for ``tag``.

    Both train and val scalars carry the *global step*, not the epoch: a 10-epoch run with
    126 train batches writes ``train/*`` at steps 9, 19, ... and ``val/*`` at 125, 251, ....
    The epoch is recovered by dividing by the steps-per-epoch implied by the val cadence
    (val runs exactly once per epoch), so the two series line up.
    """
    if tag not in ea.Tags()["scalars"]:
        return {}
    scalars = ea.Scalars(tag)
    val = ea.Scalars("val/loss_total") if "val/loss_total" in ea.Tags()["scalars"] else []
    steps_per_epoch = (val[0].step + 1) if val else 1
    acc: dict[int, list[float]] = {}
    for e in scalars:
        acc.setdefault(int(e.step) // steps_per_epoch, []).append(float(e.value))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="regularised run directory (…/member0_seed0)")
    ap.add_argument("--baseline", default=None, help="unregularised run for the same split")
    ap.add_argument("--epochs", type=int, default=12, help="how many epochs to print")
    args = ap.parse_args()

    ea = _load(args.run)
    main_loss = _by_epoch(ea, "train/loss_total")
    pen = _by_epoch(ea, "train/homolog_loss")
    w = _by_epoch(ea, "train/homolog_weight")
    val_h = _by_epoch(ea, "val/mae_h298")

    print(f"run: {args.run}")
    print(f"{'ep':>3} {'main':>12} {'hom_pen':>12} {'w':>8} {'w*pen':>12} "
          f"{'pen/main':>10} {'val_mae_H':>12} {'val_H_kcal':>11}")
    for ep in sorted(set(main_loss) | set(pen))[: args.epochs]:
        m = main_loss.get(ep, float("nan"))
        p = pen.get(ep, float("nan"))
        ww = w.get(ep, 0.0)
        v = val_h.get(ep, float("nan"))
        ratio = (ww * p) / m if m else float("nan")
        print(f"{ep:>3} {m:>12.4f} {p:>12.4f} {ww:>8.3g} {ww * p:>12.4f} "
              f"{ratio:>10.3g} {v:>12.1f} {v / 4184.0:>11.4f}")

    # The residual scale the main loss divides by -- the yardstick the penalty must match.
    hp = Path(args.run) / "lightning_logs" / "version_0" / "hparams.yaml"
    if hp.is_file():
        import yaml

        h = yaml.safe_load(hp.read_text())
        print("\nhparams relevant to the loss scale:")
        for k in ("loss_scale_h", "loss_scale_s", "loss_scale_cp", "huber_beta_h",
                  "huber_beta_s", "huber_beta_cp", "homolog_weight",
                  "homolog_ramp_epochs", "homolog_w_h", "homolog_w_s",
                  "homolog_w_cp", "homolog_beta"):
            if k in h:
                print(f"  {k:<22s}{h[k]}")

    if args.baseline:
        eb = _load(args.baseline)
        vb = _by_epoch(eb, "val/mae_h298")
        print(f"\nbaseline val ΔH298 MAE (J/mol), first {args.epochs} epochs:")
        print("  " + "  ".join(f"{vb[e]:.0f}" for e in sorted(vb)[: args.epochs]))


if __name__ == "__main__":
    main()
