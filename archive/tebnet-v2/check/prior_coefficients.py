"""C6: are the coefficients of TEB-Net's frozen ridge prior chemically meaningful?

Reads `baseline_coefficients` from the results.json files written by scripts/train.py (no training, numpy
only) and compares them with textbook values, as mean ± std over the random splits found.

Identifiability. In the ΔH prior, the ring counts sum to the cyclomatic number = (#heavy-atom bonds) −
(#heavy atoms) + 1, which is already a linear combination of the bond-type and element-count columns. The
same holds for the radical-centre types, which sum to n_rad_e (also a column). The ridge solution therefore
fixes only DIFFERENCES between ring sizes and between radical types; the absolute values are arbitrary.
So the script reports
  * ring_k − ring6                 vs Benson ring-strain corrections (cyclohexane = 0)
  * rad_type − rad_alkyl_p         vs BDE(R–H of that type) − BDE(primary C–H)
  * rad_type − rad_alkoxy (O-centred) vs BDE(O–H of that type) − BDE(CH3O–H)
  * S coefficients in units of R   vs ideal-gas statistical mechanics: −ln σ, +ln(2S+1), +ln 2 (optical isomers)

Reference values are typical literature numbers to be CHECKED against the sources cited in the paper:
Benson, Thermochemical Kinetics (1976) for ring strain; Blanksby & Ellison, Acc. Chem. Res. 36 (2003) 255
for bond dissociation enthalpies at 298 K.

Usage
  python scripts/prior_coefficients.py --runs benchmark_results_deeptherm
  (uses one run per random split with the default 81/9/10 fractions, so learning-curve runs are ignored)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
RING_REF = {"ring3": 27.6, "ring4": 26.2, "ring5": 6.3, "ring7p": 6.4}
BDE_C = {"rad_alkyl_p": 101.1, "rad_alkyl_s": 98.6, "rad_alkyl_t": 96.5, "rad_allylic": 88.8,
         "rad_vinyl": 110.7, "rad_aryl": 112.9, "rad_ethynyl": 133.3, "rad_acyl": 89.0,
         "rad_alpha_O": 96.1, "rad_alpha_CO": 96.0}
BDE_O = {"rad_alkoxy": 104.6, "rad_peroxy": 88.0, "rad_vinoxy": 87.8}
S_REF = {"log_sigma": (-1.0, "−R ln σ (rotational symmetry)"),
         "ln_mult": (1.0, "+R ln(2S+1) (spin multiplicity)"),
         "is_chiral": (float(np.log(2.0)), "+R ln 2 (pair of optical isomers)"),
         "log_I_rot": (1.0, "+R·½ ln(I_A I_B I_C) (rotation; collinear with size)")}
DEFAULT_FRACTIONS = (0.81, 0.09, 0.10)
def load(roots: list[str], split_type: str, tag_contains: str) -> dict[int, dict]:
    per_seed: dict[int, dict] = {}
    for root in roots:
        for rp in sorted(Path(root).rglob("results.json")):
            try:
                res = json.loads(rp.read_text())
            except Exception:
                continue
            coef = res.get("baseline_coefficients") or {}
            sp = res.get("split", {})
            if (not coef or sp.get("type") != split_type or res.get("member", 0) != 0
                    or tag_contains not in res.get("tag", "")
                    or not np.allclose(sp.get("fractions", DEFAULT_FRACTIONS), DEFAULT_FRACTIONS)):
                continue
            per_seed.setdefault(int(sp["seed"]), coef)
    return per_seed
def stats(per_seed: dict[int, dict], key_fn) -> tuple[float, float, int]:
    vals = [v for v in (key_fn(c) for c in per_seed.values()) if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"), len(vals)
def diff(c: dict, a: str, b: str):
    return c[a] - c[b] if a in c and b in c else None
def fmt(m, s):
    return f"{m:+.2f}" + (f" ± {s:.2f}" if np.isfinite(s) else "")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["benchmark_results_deeptherm"])
    ap.add_argument("--split_type", default="random")
    ap.add_argument("--tag_contains", default="thermo_physics_chebyshev_3d",
                    help="substring of the run tag; keeps runs that used the full physics prior")
    args = ap.parse_args()
    per_seed = load(args.runs, args.split_type, args.tag_contains)
    if not per_seed:
        raise SystemExit("no matching results.json with baseline_coefficients")
    print(f"prior coefficients from {len(per_seed)} {args.split_type} split(s): seeds {sorted(per_seed)}\n")
    print("Ring strain, ΔH prior (kcal/mol per ring, relative to ring6)")
    print("| Ring | Fitted (ring_k − ring6) | Benson ring-strain correction | Difference |")
    print("|---|---|---|---|")
    for k, ref in RING_REF.items():
        m, s, _ = stats(per_seed, lambda c, k=k: diff(c, f"H:{k}", "H:ring6"))
        print(f"| {k} | {fmt(m, s)} | {ref:+.1f} | {m - ref:+.1f} |")
    print("\nCarbon-centred radicals, ΔH prior (kcal/mol, relative to a primary alkyl radical)")
    print("| Radical centre | Fitted (type − rad_alkyl_p) | ΔBDE(C–H) vs primary | Difference |")
    print("|---|---|---|---|")
    for k, bde in BDE_C.items():
        if k == "rad_alkyl_p":
            continue
        ref = bde - BDE_C["rad_alkyl_p"]
        m, s, n = stats(per_seed, lambda c, k=k: diff(c, f"H:{k}", "H:rad_alkyl_p"))
        if n:
            print(f"| {k} | {fmt(m, s)} | {ref:+.1f} | {m - ref:+.1f} |")
    print("\nOxygen-centred radicals, ΔH prior (kcal/mol, relative to an alkoxy radical)")
    print("| Radical centre | Fitted (type − rad_alkoxy) | ΔBDE(O–H) vs CH3O–H | Difference |")
    print("|---|---|---|---|")
    for k, bde in BDE_O.items():
        if k == "rad_alkoxy":
            continue
        ref = bde - BDE_O["rad_alkoxy"]
        m, s, n = stats(per_seed, lambda c, k=k: diff(c, f"H:{k}", "H:rad_alkoxy"))
        if n:
            print(f"| {k} | {fmt(m, s)} | {ref:+.1f} | {m - ref:+.1f} |")
    print("\nEntropy prior (units of R)")
    print("| Feature | Fitted | Expected | Meaning |")
    print("|---|---|---|---|")
    for k, (ref, meaning) in S_REF.items():
        m, s, n = stats(per_seed, lambda c, k=k: c.get(f"S:{k} [R]"))
        if n:
            print(f"| {k} | {fmt(m, s)} | {ref:+.2f} | {meaning} |")
if __name__ == "__main__":
    main()
