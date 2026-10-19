#!/usr/bin/env python3
"""Per-property breakdown of the CH2-homolog penalty vs. the main loss yardsticks.

The main loss divides every property by its *train target std* (``loss_scale_*``),
whereas :class:`HomologConsistencyLoss` divides by the *measured CH2 increment
tolerance*.  Those are two different yardsticks, and this script prints both so the
ratio between them -- the reason a given ``--homolog_weight`` is or is not a
perturbation -- is explicit rather than guessed.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184


def _load(run_dir: str) -> EventAccumulator:
    path = Path(run_dir) / "lightning_logs" / "version_0"
    if not path.is_dir():
        cands = sorted(glob.glob(str(Path(run_dir) / "lightning_logs" / "version_*")))
        path = Path(cands[-1])
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _series(ea: EventAccumulator, tag: str) -> list[float]:
    if tag not in ea.Tags()["scalars"]:
        return []
    return [float(e.value) for e in ea.Scalars(tag)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--results", default=None,
                    help="results.json of the run (holds the increment/tolerance table)")
    args = ap.parse_args()

    ea = _load(args.run)
    print("=== last-epoch means (regularised run) ===")
    for tag in ("train/loss_total", "train/loss_h298", "train/loss_s298", "train/loss_cp",
                "train/homolog_loss", "train/homolog_weight",
                "train/homolog_pen_h", "train/homolog_pen_s", "train/homolog_pen_cp",
                "train/homolog_r_h", "train/homolog_r_s", "train/homolog_r_cp",
                "train/homolog_resid_h", "train/homolog_resid_s", "train/homolog_resid_cp"):
        v = _series(ea, tag)
        if v:
            tail = v[-5:]
            print(f"  {tag:<26s} {sum(tail) / len(tail):>12.4f}   (n={len(v)})")

    print("\n=== validation ΔH298 MAE trajectory ===")
    v = _series(ea, "val/mae_h298")
    if v:
        print("  J/mol :", " ".join(f"{x:.0f}" for x in v[:12]))
        print("  kcal  :", " ".join(f"{x * J_TO_KCAL:.2f}" for x in v[:12]))

    print("\n=== yardstick comparison ===")
    import json

    hp = Path(args.run) / "lightning_logs" / "version_0" / "hparams.yaml"
    import yaml

    h = yaml.safe_load(hp.read_text())
    scales = {"h": h["loss_scale_h"], "s": h["loss_scale_s"], "cp": h["loss_scale_cp"]}

    rp = args.results
    if rp is None:
        cands = sorted(glob.glob(str(Path(args.run).parents[0] / "**" / "results.json"),
                                 recursive=True))
        rp = cands[0] if cands else None
    tols = None
    if rp and Path(rp).is_file():
        d = json.loads(Path(rp).read_text())
        tol = d.get("homolog", {}).get("tolerances", {})
        if tol:
            tols = {"h": tol["h_kcal_per_mol"][0] * 4184.0,
                    "s": tol["s_cal_per_molK"][0] * 4.184,
                    "cp": tol["cp_cal_per_molK"][0] * 4.184}
            print(f"  (tolerances from {rp})")

    print(f"  {'prop':>4} {'main yardstick':>16} {'pen yardstick':>15} {'ratio':>9} {'ratio^2':>10}")
    for k in ("h", "s", "cp"):
        sc = scales[k]
        if tols:
            tl = tols[k]
            print(f"  {k:>4} {sc:>16.2f} {tl:>15.2f} {sc / tl:>9.2f} {(sc / tl) ** 2:>10.1f}")
        else:
            print(f"  {k:>4} {sc:>16.2f} {'n/a':>15}")

    print("\n  main yardstick = train target std (J/mol or J/(mol K)), the divisor of the"
          "\n  per-molecule loss; pen yardstick = CH2 increment tolerance in the same units."
          "\n  For the same absolute error the penalty enters (ratio)^2 times harder, because"
          "\n  both terms are quadratic in the small-residual regime.")
    print(f"\n  loss_scale_h = {scales['h']:.1f} J/mol = {scales['h'] * J_TO_KCAL:.2f} kcal/mol")
    if tols:
        print(f"  tol_h        = {tols['h']:.1f} J/mol = {tols['h'] * J_TO_KCAL:.2f} kcal/mol")


if __name__ == "__main__":
    main()
