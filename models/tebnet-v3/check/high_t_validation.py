"""Do TEB-Net's exported NASA-7 polynomials follow reference Cp(T), H(T) and S(T) above the data (1500-3000 K)?

Inputs
  <dat>                 polynomials from
                        check/nasa7_check.py --smiles_csv rmg_cp_reference.csv --complete_high --write_dat <dat>
                        (with <dat>.as_trained.dat and <dat>.smiles.csv written next to it)
  --reference           rmg_cp_reference.csv from check/rmg_high_t_reference.py
  --deeptherm_csv       optional: Cp of the reference against the DeepTherm data at 300-1500 K for the species that
                        are in DeepTherm (the offset between the two sources, which TEB-Net inherits)

Species: rank-0 reference rows whose SMILES carries the right radical count (no singlet carbenes) and whose
multiplicity is <= --max_multiplicity, matched to TEB-Net by SMILES. Every match is reported; none is selected by
its error.

Metrics, per species, then averaged over groups (all; split train / val / test / none; source evaluated / qm;
heavy atoms)
  Cp MAE on 300-1500 K, 1500-2000 K and 2000-3000 K, in J/(mol K) and in % of the reference. On 300-1500 K
     TEB-Net was trained on DeepTherm's Cp and the reference comes from another source, so this band measures the
     offset between the two, not the extrapolation.
  drift: mean over 1500-3000 K of |e(T) - e(1500 K)|, e = TEB-Net - reference: the error the extrapolation adds
     on top of the offset already present at the last data temperature.
  H(3000 K) - H(298.15 K) and S(3000 K) - S(298.15 K): the integrals a kinetics code uses.
  completed = the exported polynomials (--complete_high), as trained = straight high range.

Usage
  python check/high_t_validation.py c5_rmg.dat --reference rmg_cp_reference.csv \
      [--deeptherm_csv models-baseline/deeptherm_data.csv] --out_csv high_t_validation.csv
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rmg_cp_reference import cp_j, h_j, s_j
R = 8.314462618
CAL = 4.184
T_REF = 298.15
T_DATA = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
DT_CP = ["cp_300K_cal/(molK)", "cp_400K_cal/(molK)", "cp_500K_cal/(molK)", "cp_600K_cal/(molK)",
         "cp_800K_cal/(molK)", "cp_1000K_cal/(molK)", "cp_1500K_cal/(molK)"]
NUM = re.compile(r"[+-]?\d\.\d+[Ee][+-]\d+")
BANDS = [("300–1500 K", "300_1500", 300.0, 1500.0), ("1500–2000 K", "1500_2000", 1500.0, 2000.0),
         ("2000–3000 K", "2000_3000", 2000.0, 3000.0)]
def parse_chemkin(path):
    """Chemkin THERMO file -> {name: (t_mid, a_low[7], a_high[7])} (same reader as figures/plot_cp_completion.py)."""
    lines = Path(path).read_text().splitlines()
    out, i = {}, 0
    while i + 3 < len(lines):
        l = lines[i]
        if len(l) >= 80 and l[79] == "1":
            c = [float(x) for x in NUM.findall(lines[i + 1][:75] + lines[i + 2][:75] + lines[i + 3][:60])]
            if len(c) == 14:
                out[l[:18].strip()] = (float(l[65:73]), np.array(c[7:]), np.array(c[:7]))
            i += 4
        else:
            i += 1
    return out
def teb_props(entry, T):
    """Cp, H, S (J units) of a NASA-7 pair on the array T."""
    t_mid, lo, hi = entry
    a = np.where((T <= t_mid)[:, None], lo[None, :], hi[None, :])
    cp = R * sum(a[:, k] * T ** k for k in range(5))
    h = R * T * (sum(a[:, k] * T ** k / (k + 1) for k in range(5)) + a[:, 5] / T)
    s = R * (a[:, 0] * np.log(T) + sum(a[:, k] * T ** k / k for k in range(1, 5)) + a[:, 6])
    return cp, h, s
def canon_fn():
    try:
        from rdkit import Chem
    except ImportError:
        return lambda s: s
    def can(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else s
    return can
def size_bin(n):
    return "≤ 5 heavy atoms" if n <= 5 else ("6–10 heavy atoms" if n <= 10 else "> 10 heavy atoms")
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("completed_dat")
    ap.add_argument("--as_trained", default=None, help="default: <dat>.as_trained.dat")
    ap.add_argument("--smiles_csv", default=None, help="default: <dat>.smiles.csv")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--max_multiplicity", type=int, default=2)
    ap.add_argument("--t_data_max", type=float, default=1500.0)
    ap.add_argument("--t_high", type=float, default=3000.0)
    ap.add_argument("--deeptherm_csv", default=None)
    ap.add_argument("--out_csv", default="high_t_validation.csv")
    args = ap.parse_args()
    p = Path(args.completed_dat)
    comp = parse_chemkin(p)
    trained = parse_chemkin(args.as_trained or p.with_suffix(".as_trained.dat"))
    names = pd.read_csv(args.smiles_csv or p.with_suffix(".smiles.csv"))
    if "split" not in names:
        names["split"] = "test"
    can = canon_fn()
    names["_key"] = names.smiles.astype(str).map(can)
    ref = pd.read_csv(args.reference)
    n_ref = int((ref["rank"] == 0).sum())
    ref = ref[(ref["rank"] == 0) & (ref.smiles_roundtrip == 1) & (ref.multiplicity <= args.max_multiplicity)].copy()
    ref["_key"] = ref.smiles.astype(str).map(can)
    ref = ref.drop_duplicates("_key")
    m = names.merge(ref.drop(columns="smiles"), on="_key", how="inner")
    m = m[m.name.isin(comp) & m.name.isin(trained)]
    print(f"reference species (rank 0): {n_ref}; usable (SMILES round-trips, multiplicity <= {args.max_multiplicity}): "
          f"{len(ref)}; with TEB-Net polynomials: {len(m)}")
    if m.empty:
        raise SystemExit("no species matched: run nasa7_check.py --smiles_csv on the same reference CSV")
    T = np.arange(300.0, args.t_high + 0.5, 10.0)
    i1500 = int(np.argmin(np.abs(T - args.t_data_max)))
    above = T > args.t_data_max
    rows = []
    for _, r in m.iterrows():
        pieces = [(lo, hi, a) for lo, hi, a in json.loads(r.nasa)]
        cp_ref = np.array([cp_j(pieces, t) for t in T])
        dh_ref = (h_j(pieces, args.t_high) - h_j(pieces, T_REF)) / 1000.0
        ds_ref = s_j(pieces, args.t_high) - s_j(pieces, T_REF)
        row = {"name": r["name"], "smiles": r.smiles, "label": r.label, "library": r.library, "source": r.source,
               "split": r.split, "n_atoms": r.n_atoms, "n_heavy": r.n_heavy, "multiplicity": r.multiplicity,
               "cp_ref_1500": cp_ref[i1500], "cp_ref_3000": cp_ref[-1]}
        for tag, ent in (("c", comp[r["name"]]), ("t", trained[r["name"]])):
            cp, h, s = teb_props(ent, np.concatenate([T, [T_REF]]))
            cp, h_end, h0, s_end, s0 = cp[:-1], h[-2], h[-1], s[-2], s[-1]
            e = cp - cp_ref
            for _, key, lo, hi in BANDS:
                sel = (T >= lo) & (T <= hi) if lo <= T[0] else (T > lo) & (T <= hi)
                row[f"mae_{key}_{tag}"] = np.abs(e[sel]).mean()
                row[f"mape_{key}_{tag}"] = 100 * np.abs(e[sel] / cp_ref[sel]).mean()
            row[f"e1500_{tag}"] = e[i1500]
            row[f"e3000_{tag}"] = e[-1]
            row[f"drift_{tag}"] = np.abs(e[above] - e[i1500]).mean()
            row[f"dH3000_{tag}"] = (h_end - h0) / 1000.0 - dh_ref
            row[f"dS3000_{tag}"] = (s_end - s0) - ds_ref
            row[f"cp3000_{tag}"] = cp[-1]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, float_format="%.4g")
    groups = [("all", df.index == df.index)]
    groups += [(f"split: {k}", df.split == k) for k in ("train", "val", "test", "none") if (df.split == k).any()]
    groups += [(f"source: {k}", df.source == k) for k in ("evaluated", "qm") if (df.source == k).any()]
    sb = df.n_heavy.map(size_bin)
    groups += [(k, sb == k) for k in ("≤ 5 heavy atoms", "6–10 heavy atoms", "> 10 heavy atoms") if (sb == k).any()]
    def cell(g, col, f="{:.1f}"):
        return f"{f.format(g[col + '_c'].abs().mean())} / {f.format(g[col + '_t'].abs().mean())}"
    print(f"\n### Cp error against the reference, J/(mol K): completed / as trained (no data above "
          f"{args.t_data_max:g} K)\n")
    print("| Group | n | MAE " + " | MAE ".join(b[0] for b in BANDS) + f" | drift above {args.t_data_max:g} K "
          f"| \\|Δ[H(3000)−H(298)]\\| kJ/mol | \\|Δ[S(3000)−S(298)]\\| J/(mol K) |")
    print("|---|---|" + "---|" * (len(BANDS) + 3))
    for gname, sel in groups:
        g = df[sel]
        print(f"| {gname} | {len(g)} | " + " | ".join(cell(g, f"mae_{b[1]}") for b in BANDS)
              + f" | {cell(g, 'drift')} | {cell(g, 'dH3000')} | {cell(g, 'dS3000')} |")
    print("\n### Same, in % of the reference Cp\n")
    print("| Group | n | " + " | ".join(b[0] for b in BANDS) + " |")
    print("|---|---|" + "---|" * len(BANDS))
    for gname, sel in groups:
        g = df[sel]
        print(f"| {gname} | {len(g)} | " + " | ".join(cell(g, f"mape_{b[1]}") for b in BANDS) + " |")
    rise = df.cp_ref_3000 - df.cp_ref_1500
    print(f"\nreference rise of Cp from {args.t_data_max:g} to {args.t_high:g} K: median {rise.median():.1f} J/(mol K) "
          f"({100 * (rise / df.cp_ref_1500).median():.0f} % of Cp({args.t_data_max:g} K))")
    print(f"signed error at {args.t_data_max:g} K (last data temperature, same for both): median "
          f"{df.e1500_c.median():+.1f}, at {args.t_high:g} K: completed {df.e3000_c.median():+.1f}, "
          f"as trained {df.e3000_t.median():+.1f} J/(mol K)")
    better = (df.mae_2000_3000_c < df.mae_2000_3000_t).mean()
    print(f"completed closer to the reference than as trained on 2000–3000 K: {100 * better:.0f} % of species")
    print("\nlargest MAE on 2000–3000 K (completed), for the discussion, not for the figure:")
    for _, r in df.nlargest(8, "mae_2000_3000_c").iterrows():
        print(f"  {r['name']:<7s} {r.smiles:<32s} {r.library:<28s} {r.split:<5s} "
              f"MAE {r.mae_2000_3000_c:.1f} (as trained {r.mae_2000_3000_t:.1f}), e(1500 K) {r.e1500_c:+.1f}")
    if args.deeptherm_csv:
        d = pd.read_csv(args.deeptherm_csv)
        col = next(c for c in d.columns if c.strip().lower() == "smiles")
        d["_key"] = d[col].astype(str).str.strip().map(can)
        d = d.drop_duplicates("_key").set_index("_key")
        both = df.assign(_key=df.smiles.map(can))
        both = both[both._key.isin(d.index)]
        if len(both):
            err = []
            for _, r in both.iterrows():
                pieces = [(lo, hi, a) for lo, hi, a in json.loads(ref.set_index("_key").loc[r._key, "nasa"])]
                ref_cp = np.array([cp_j(pieces, t) for t in T_DATA])
                err.append(np.abs(d.loc[r._key, DT_CP].astype(float).values * CAL - ref_cp))
            err = np.array(err)
            print(f"\nreference vs DeepTherm data at 300–1500 K ({len(both)} species in both): MAE "
                  + ", ".join(f"{t:.0f} K {e:.1f}" for t, e in zip(T_DATA, err.mean(0))) + " J/(mol K)")
    print(f"\nwrote {args.out_csv}")
if __name__ == "__main__":
    main()
