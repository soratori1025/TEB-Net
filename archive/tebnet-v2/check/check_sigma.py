"""Check the symmetry numbers stored in the physics cache against textbook values.

  python scripts/check_sigma.py --cache deeptherm_data.physics_v15.csv

Expected values are TOTAL symmetry numbers (external × internal rotors, e.g. ethane 6 × 3 = 18),
the convention of RMG / Benson group additivity and of 1-D hindered-rotor thermochemistry.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
REF = {
    "C=O": 2, "CO": 3, "C#C": 2, "C=C": 4, "CC": 18, "COC": 18, "CC=O": 3, "C1CO1": 2, "C=C=O": 2,
    "O=CC=O": 2, "CCO": 3, "CC(C)=O": 18, "CCC": 18, "CC(C)C": 81, "CCCCC": 18, "c1ccccc1": 12,
    "C1CCCCC1": 6, "O=C=O": 2, "[CH3]": 6, "CC(C)(C)C": 972, "OO": 2, "COO": 3, "C=CC": 3,
}
NOTE = {"C1CCCCC1": "chair D3d = 6; a graph-only method may give 12"}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.cache)
    d["smiles"] = d["smiles"].astype(str).str.strip()
    d = d.drop_duplicates("smiles").set_index("smiles")
    sig = np.exp(d["log_sigma"].astype(float))
    print(f"{len(d)} species | σ = 1 for {100 * (sig < 1.5).mean():.1f}% | "
          f"median σ {np.median(sig):.0f} | 95th pct {np.percentile(sig, 95):.0f} | max {sig.max():.0f}")
    print(f"\n  {'SMILES':<12s} {'expected':>9s} {'cache':>9s}")
    n_ok = n_tot = 0
    for s, ref in REF.items():
        if s not in d.index:
            continue
        got = float(sig[s])
        ok = abs(got - ref) < 0.5
        n_ok += ok
        n_tot += 1
        print(f"  {s:<12s} {ref:>9d} {got:>9.0f}   {'' if ok else '<-- ' + NOTE.get(s, 'mismatch')}")
    print(f"\n{n_ok}/{n_tot} match")
if __name__ == "__main__":
    main()
