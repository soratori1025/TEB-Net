"""Equilibrium constant of R + O2 <=> RO2 from exported NASA-7 files against library references.

Reads a Chemkin thermo file written by check/nasa7_check.py --write_dat (the same file that goes
into the mechanism) and the evaluated NASA-7 polynomials of data/rmg_data/libraries/rmg_thermo.csv,
then evaluates dH, dS, dG, log10 Kp (bar^-1) and Kc (cm3/mol) for R + O2 <=> RO2.

O2 is always taken from the reference library (GRI-Mech 3.0), so a difference is attributable to
the two model species. Kp uses the ideal-gas standard state of the NASA-7 polynomials (1 bar) and
Kc = Kp * R T (bar^-1 -> cm3/mol: Kc[cm3/mol] = Kp[bar^-1] * R T / 10).

Usage (from models/tebnet-v3, as check/predict_smiles.py)
    python check/kc_equilibrium.py --thermo check/kinetics/kin6.dat \
        --smiles_csv check/kinetics/kin6.smiles.csv \
        --as_trained check/kinetics/kin6.as_trained.dat
The three files in check/kinetics/ are the ones behind Table 7 of the manuscript; they are written by
    python check/nasa7_check.py --h_runs ... --scp_runs ... --smiles_csv check/kinetics/species.csv \
        --complete_high --write_dat check/kinetics/kin6.dat
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
R_GAS = 8.314462618
ROOT = Path(__file__).resolve().parents[3]
RMG = ROOT / "data/rmg_data/libraries/rmg_thermo.csv"
SPECIES = {"CH3": "[CH3]", "C2H5": "[CH2]C", "CH3OO": "CO[O]", "C2H5OO": "CCO[O]",
           "C6H11": "[CH]1CCCCC1", "C6H11OO": "[O]OC1CCCCC1", "O2": "[O][O]"}
REACTIONS = [("CH3 + O2 <=> CH3OO", "CH3", "CH3OO"),
             ("C2H5 + O2 <=> C2H5OO", "C2H5", "C2H5OO"),
             ("c-C6H11 + O2 <=> c-C6H11OO", "C6H11", "C6H11OO")]
GRID = [300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0]
def read_chemkin(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """{name: (low a1-a7, high a1-a7, T_mid)}; Chemkin puts the HIGH range first."""
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    out, i = {}, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("THERMO") or line.strip().startswith("300.000") or \
                not line.strip() or line.strip().upper() == "END":
            i += 1
            continue
        t_mid = float(line[65:75])
        vals = []
        for k in range(1, 4):
            chunk = lines[i + k][:75]
            vals.extend(float(chunk[j:j + 15]) for j in range(0, 75, 15) if chunk[j:j + 15].strip())
        out[line.split()[0]] = (np.array(vals[7:14]), np.array(vals[:7]), t_mid)
        i += 4
    return out
def nasa(a: np.ndarray, T: float) -> tuple[float, float]:
    """(H [J/mol], S [J/(mol K)]) of one NASA-7 coefficient set at T."""
    a1, a2, a3, a4, a5, a6, a7 = a
    h_rt = a1 + a2 * T / 2 + a3 * T**2 / 3 + a4 * T**3 / 4 + a5 * T**4 / 5 + a6 / T
    s_r = a1 * np.log(T) + a2 * T + a3 * T**2 / 2 + a4 * T**3 / 3 + a5 * T**4 / 4 + a7
    return h_rt * R_GAS * T, s_r * R_GAS
def ev(entry, T: float) -> tuple[float, float]:
    low, high, t_mid = entry[:3]
    return nasa(low if T <= t_mid else high, T)
def read_reference(wanted: set[str]) -> dict[str, tuple]:
    cols = [f"a{k}_{rng}" for rng in ("low", "high") for k in range(1, 8)]
    out = {}
    with RMG.open() as fh:
        for row in csv.DictReader(fh):
            if row["smiles"] in wanted:
                v = np.array([float(row[c]) for c in cols])
                out[row["smiles"]] = (v[:7], v[7:], float(row["tmid_K"]), row["library"])
    return out
def log10_kp(dh_j: float, ds: float, T: float) -> float:
    return -(dh_j - T * ds) / (R_GAS * T * np.log(10.0))
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermo", required=True)
    ap.add_argument("--smiles_csv", required=True)
    ap.add_argument("--as_trained", default=None)
    args = ap.parse_args()
    ident = {}
    with Path(args.smiles_csv).open() as fh:
        for row in csv.DictReader(fh):
            ident[row["smiles"]] = row["name"]
    teb = read_chemkin(Path(args.thermo))
    raw = read_chemkin(Path(args.as_trained)) if args.as_trained else None
    ref = read_reference(set(SPECIES.values()))
    for rxn, rad, ro2 in REACTIONS:
        if SPECIES[rad] not in ident or SPECIES[ro2] not in ident:
            print(f"\n=== {rxn} === (not in the exported file, skipped)")
            continue
        print(f"\n=== {rxn} ===  reference: {ref[SPECIES[ro2]][3]}")
        print(f"{'T/K':>6s} {'dHref':>8s} {'dHTEB':>8s} {'dSref':>8s} {'dSTEB':>8s} "
              f"{'logKref':>8s} {'logKTEB':>8s} {'ratio':>7s} {'Kc':>10s}")
        for T in GRID:
            h_o, s_o = ev(ref[SPECIES["O2"]], T)
            h_r, s_r = ev(ref[SPECIES[rad]], T)
            h_p, s_p = ev(ref[SPECIES[ro2]], T)
            dh_ref, ds_ref = h_p - h_r - h_o, s_p - s_r - s_o
            h_r2, s_r2 = ev(teb[ident[SPECIES[rad]]], T)
            h_p2, s_p2 = ev(teb[ident[SPECIES[ro2]]], T)
            dh_teb, ds_teb = h_p2 - h_r2 - h_o, s_p2 - s_r2 - s_o
            lk_ref, lk_teb = log10_kp(dh_ref, ds_ref, T), log10_kp(dh_teb, ds_teb, T)
            kc = 10 ** lk_teb * (R_GAS * T) / 10.0
            line = (f"{T:6.0f} {dh_ref / 1000:8.2f} {dh_teb / 1000:8.2f} {ds_ref:8.2f} "
                    f"{ds_teb:8.2f} {lk_ref:8.2f} {lk_teb:8.2f} {10 ** (lk_teb - lk_ref):7.2f} {kc:10.2e}")
            if raw is not None:
                h_r3, s_r3 = ev(raw[ident[SPECIES[rad]]], T)
                h_p3, s_p3 = ev(raw[ident[SPECIES[ro2]]], T)
                lk_raw = log10_kp(h_p3 - h_r3 - h_o, s_p3 - s_r3 - s_o, T)
                line += f"  as-trained ratio {10 ** (lk_raw - lk_ref):6.2f}"
            print(line)
        T = 298.15
        h_o, s_o = ev(ref[SPECIES["O2"]], T)
        h_r, s_r = ev(ref[SPECIES[rad]], T)
        h_p, s_p = ev(ref[SPECIES[ro2]], T)
        h_r2, s_r2 = ev(teb[ident[SPECIES[rad]]], T)
        h_p2, s_p2 = ev(teb[ident[SPECIES[ro2]]], T)
        dg_ref = (h_p - h_r - h_o) - T * (s_p - s_r - s_o)
        dg_teb = (h_p2 - h_r2 - h_o) - T * (s_p2 - s_r2 - s_o)
        print(f"  298.15 K: dH {(h_p - h_r - h_o) / 1000:7.2f} -> {(h_p2 - h_r2 - h_o) / 1000:7.2f} kJ/mol, "
              f"dG {dg_ref / 1000:7.2f} -> {dg_teb / 1000:7.2f} kJ/mol, "
              f"K ratio {10 ** (-(dg_teb - dg_ref) / (R_GAS * T * np.log(10.0))):.2f}")
if __name__ == "__main__":
    main()
