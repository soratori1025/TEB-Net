#!/usr/bin/env python3
"""Dump the raw TensorBoard series for the main loss and the CH2-homolog penalty.

``diagnose_homolog_scale.py`` aggregates by epoch by dividing the global step by the
val cadence.  That is fine for ``val/*`` but it assumes ``train/*`` is logged once per
step with the *instantaneous* value.  Lightning, however, writes the running epoch mean
for a scalar logged with ``batch_size=...``.  The two conventions disagree, so the
epoch-mean of ``train/homolog_loss`` can come out hundreds of times larger than the
last ``train/homolog_pen_*`` value of the same epoch.

This script sidesteps the ambiguity: it prints the first/last few raw (step, value)
pairs of every tag so the logging convention can be read off the data itself.

Usage
-----
    python check/dump_homolog_series.py <run_dir> [--n 5]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = (
    "train/loss_total", "train/loss_h298", "train/loss_s298", "train/loss_cp",
    "train/homolog_loss", "train/homolog_pen_h", "train/homolog_pen_s",
    "train/homolog_pen_cp", "train/homolog_r_h", "train/homolog_weight",
    "val/loss_total", "val/mae_h298",
)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=5, help="how many head/tail points to print")
    args = ap.parse_args()

    ea = _load(args.run)
    present = set(ea.Tags()["scalars"])
    print(f"run: {args.run}")
    print(f"scalar tags: {len(present)}")

    for tag in TAGS:
        if tag not in present:
            print(f"\n{tag}: <absent>")
            continue
        s = ea.Scalars(tag)
        print(f"\n{tag}: {len(s)} points, steps {s[0].step}..{s[-1].step}")
        head = s[: args.n]
        tail = s[-args.n:]
        print("  head: " + "  ".join(f"({e.step},{e.value:.6g})" for e in head))
        print("  tail: " + "  ".join(f"({e.step},{e.value:.6g})" for e in tail))

    # Cross-check the two homolog summaries at the same steps.
    if "train/homolog_loss" in present and "train/homolog_pen_h" in present:
        tot = {e.step: e.value for e in ea.Scalars("train/homolog_loss")}
        ph = {e.step: e.value for e in ea.Scalars("train/homolog_pen_h")}
        ps = {e.step: e.value for e in ea.Scalars("train/homolog_pen_s")}
        pc = {e.step: e.value for e in ea.Scalars("train/homolog_pen_cp")}
        common = sorted(set(tot) & set(ph) & set(ps) & set(pc))
        print(f"\nstep-aligned check ({len(common)} common steps):")
        print(f"{'step':>6} {'loss_total':>12} {'pen_h':>10} {'pen_s':>10} "
              f"{'pen_cp':>10} {'sum(pen)':>10} {'homolog_loss':>13}")
        picks = common[:3] + common[-3:] if len(common) > 6 else common
        for st in picks:
            sm = ph[st] + ps[st] + pc[st]
            print(f"{st:>6} {'':>12} {ph[st]:>10.4g} {ps[st]:>10.4g} "
                  f"{pc[st]:>10.4g} {sm:>10.4g} {tot[st]:>13.6g}")


if __name__ == "__main__":
    main()
