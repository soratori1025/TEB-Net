"""Side-by-side test metrics of a baseline ensemble and a CH2-homolog run (same split, same seeds).

Reads results.json from two benchmark trees and prints, per member and as an ensemble, the DeepTherm-set
test MAE of dHf, S298 and Cp(300 K), so a change in training can be judged on the benchmark the manuscript
reports rather than on the regulariser's own loss.

Usage
  python compare_homolog_run.py \
      --baseline benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact \
      --run      benchmark_results_homolog/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact_hom0.1 \
      --split    random0
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
PROPS = (("H298_kcal", "dHf (kcal/mol)"), ("S298_cal", "S298 (cal/mol/K)"), ("Cp300_cal", "Cp300 (cal/mol/K)"))
def load(tree, split):
    """{member label: (test metrics dict, val metrics dict)} for every results.json under tree/split."""
    out = {}
    for f in sorted(glob.glob(os.path.join(tree, split, "member*_seed*", "results.json"))):
        d = json.load(open(f))
        out[os.path.basename(os.path.dirname(f))] = (d["test"], d["val"], d)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="random0")
    args = ap.parse_args()
    base, run = load(args.baseline, args.split), load(args.run, args.split)
    if not base or not run:
        raise SystemExit(f"no results.json found (baseline={len(base)}, run={len(run)})")
    print(f"split {args.split}: baseline {len(base)} member(s), run {len(run)} member(s)\n")
    for prop, label in PROPS:
        print(f"### {label}\n")
        print("| member | baseline | regulariser | delta |")
        print("|---|---|---|---|")
        shared = [m for m in base if m in run]
        for m in shared:
            b, r = base[m][0][prop]["mae"], run[m][0][prop]["mae"]
            print(f"| {m} | {b:.3f} | {r:.3f} | {r - b:+.3f} |")
        if shared:
            bm = np.mean([base[m][0][prop]["mae"] for m in shared])
            rm = np.mean([run[m][0][prop]["mae"] for m in shared])
            sd = np.std([base[m][0][prop]["mae"] for m in shared], ddof=1) if len(shared) > 1 else float("nan")
            print(f"| mean over {len(shared)} | {bm:.3f} | {rm:.3f} | {rm - bm:+.3f} (baseline s.d. {sd:.3f}) |")
        print()
    print("### validation monitor (best checkpoint selection)\n")
    for m in sorted(set(base) & set(run)):
        b, r = base[m][1], run[m][1]
        print(f"  {m}: baseline H {b['H298_kcal']['mae']:.3f} S {b['S298_cal']['mae']:.3f} "
              f"Cp {b['Cp_mean_MAE_cal']:.3f} | run H {r['H298_kcal']['mae']:.3f} S {r['S298_cal']['mae']:.3f} "
              f"Cp {r['Cp_mean_MAE_cal']:.3f}")
    print("\n### training metadata\n")
    for m in sorted(set(base) & set(run)):
        b, r = base[m][2], run[m][2]
        hom = r.get("homolog") or {}
        print(f"  {m}: baseline {b['epochs_trained']} ep (best {b['best_epoch']}, {b['training_time_sec']:.0f} s) | "
              f"run {r['epochs_trained']} ep (best {r['best_epoch']}, {r['training_time_sec']:.0f} s) | "
              f"homolog weight {hom.get('weight', 0)} ramp {hom.get('ramp_epochs', 0)} "
              f"pairs {hom.get('n_pairs', 0)} classes {hom.get('n_classes', 0)}")
if __name__ == "__main__":
    main()
