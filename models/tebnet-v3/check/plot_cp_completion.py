"""Figure 4 of the manuscript: heat capacity from 300 to 3000 K before and after the high-temperature completion.

Reads the two Chemkin files written by nasa7_check.py --write_dat ... --complete_high:
  <name>.dat              completed polynomials (the exported file)
  <name>.as_trained.dat   the same species with the as-trained (uncompleted) high range
  <name>.smiles.csv       name -> SMILES (and split, if written by nasa7_check.py --smiles_csv)
and draws, for three species, Cp(T) of both polynomials, the predictions at the data temperatures (identical in the
two files), the classical limit (3N - 2)R ((3N - 3/2)R for linear molecules) and, optionally, reference data.

High-temperature reference (--reference rmg_cp_reference.csv from check/rmg_high_t_reference.py, with the .dat
written by nasa7_check.py --smiles_csv on the same CSV): adds the reference Cp(T) up to 3000 K to each panel and,
with --error_panel, a last panel with the error (TEB-Net - reference) / reference against T over ALL matched
species (median and 10th-90th percentile band), so that the three examples can be read against the whole set.
Without --species the examples are picked by a fixed rule before looking at any error: among the matched species
outside the training and validation splits (--pick_from unseen), the ones at the 10th, 50th and 90th percentile of
the atom count.

Usage
  python plot_cp_completion.py c5_random0.dat --out fig_cp_completion
  python plot_cp_completion.py c5_random0.dat --species TEB123 TEB456 TEB789
  python plot_cp_completion.py c5_random0.dat --data_csv deeptherm_curated.csv     # adds the DeepTherm data points
  python plot_cp_completion.py c5_rmg.dat --reference rmg_cp_reference.csv --error_panel --out fig_cp_completion
Without --species, the script picks the species at the 10th, 50th and 90th percentile of the atom count.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = 8.314462618
CAL = 4.184
T_DATA = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
DT_CP = ["cp_300K_cal/(molK)", "cp_400K_cal/(molK)", "cp_500K_cal/(molK)", "cp_600K_cal/(molK)",
         "cp_800K_cal/(molK)", "cp_1000K_cal/(molK)", "cp_1500K_cal/(molK)"]
NUM = re.compile(r"[+-]?\d\.\d+[Ee][+-]\d+")
BLUE, VERMILION, REF_GREY = "#0072B2", "#D55E00", "0.62"


def parse(path):
    """Chemkin THERMO file -> {name: (elements, t_low, t_high, t_mid, a_low[7], a_high[7])}."""
    lines = Path(path).read_text().splitlines()
    out, i = {}, 0
    while i + 3 < len(lines):
        l = lines[i]
        if len(l) >= 80 and l[79] == "1":
            c = [float(x) for x in NUM.findall(lines[i + 1][:75] + lines[i + 2][:75] + lines[i + 3][:60])]
            el = {e.capitalize(): int(n) for e, n in re.findall(r"([A-Za-z]{1,2})\s*(\d+)", l[24:44]) if int(n) > 0}
            t_lo, t_hi, t_mid = float(l[45:55]), float(l[55:65]), float(l[65:73])
            if len(c) == 14:
                out[l[:18].strip()] = (el, t_lo, t_hi, t_mid, np.array(c[7:]), np.array(c[:7]))
            i += 4
        else:
            i += 1
    return out


def cp_of(entry, T):
    _, _, _, t_mid, lo, hi = entry
    a = np.where((T <= t_mid)[:, None], lo[None, :5], hi[None, :5])
    return R * (a[:, 0] + a[:, 1] * T + a[:, 2] * T**2 + a[:, 3] * T**3 + a[:, 4] * T**4)


def cp_ref(nasa_json, T):
    """Cp of a reference entry (JSON list of [t_min, t_max, 7 or 9 coefficients]) on T; NaN outside its range."""
    out = np.full(len(T), np.nan)
    for lo, hi, a in json.loads(nasa_json):
        sel = (T >= lo - 1e-6) & (T <= hi + 1e-6) & np.isnan(out)
        t = T[sel]
        if len(a) == 7:
            out[sel] = R * (a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4)
        else:
            out[sel] = R * (a[0] / t**2 + a[1] / t + a[2] + a[3] * t + a[4] * t**2 + a[5] * t**3 + a[6] * t**4)
    return out


def is_linear(smiles):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if m.GetNumAtoms() <= 2:
            return True
        AllChem.EmbedMolecule(m, randomSeed=0)
        pos = m.GetConformer().GetPositions()
        v = pos - pos[0]
        return np.linalg.matrix_rank(v, tol=1e-2) <= 1
    except Exception:
        return False


def limit(el, smiles):
    n = sum(el.values())
    if n == 1:
        return 2.5 * R
    return (3 * n - 1.5) * R if is_linear(smiles) else (3 * n - 2) * R


def formula(el):
    order = ["C", "H"] + sorted(k for k in el if k not in ("C", "H"))
    sub = lambda n: "" if n == 1 else "$_{%d}$" % n
    return "".join(f"{k}{sub(el[k])}" for k in order if k in el)


def canon(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, isomericSmiles=False) if m else s
    except Exception:
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("completed_dat")
    ap.add_argument("--as_trained", default=None, help="default: <completed>.as_trained.dat")
    ap.add_argument("--smiles_csv", default=None, help="default: <completed>.smiles.csv")
    ap.add_argument("--species", nargs="*", default=None, help="names in the .dat file (default: automatic pick)")
    ap.add_argument("--data_csv", default=None, help="optional reference data (SMILES + DeepTherm cp columns in cal)")
    ap.add_argument("--reference", default=None, help="rmg_cp_reference.csv (high-temperature reference Cp)")
    ap.add_argument("--pick_from", choices=["unseen", "test", "any"], default="unseen",
                    help="automatic pick with --reference: species outside train/val (default), test only, or any")
    ap.add_argument("--max_multiplicity", type=int, default=2)
    ap.add_argument("--error_panel", action="store_true", help="with --reference: add the all-species error panel")
    ap.add_argument("--t_max", type=float, default=3000.0)
    ap.add_argument("--out", default="fig_cp_completion")
    args = ap.parse_args()

    p = Path(args.completed_dat)
    comp = parse(p)
    trained = parse(args.as_trained or p.with_suffix(".as_trained.dat"))
    tab = pd.read_csv(args.smiles_csv or p.with_suffix(".smiles.csv"))
    if "split" not in tab:
        tab["split"] = "test"
    smi = tab.set_index("name")["smiles"].to_dict()
    split = tab.set_index("name")["split"].to_dict()

    hiref = {}                                         # name -> reference row
    if args.reference:
        r = pd.read_csv(args.reference)
        r = r[(r["rank"] == 0) & (r.smiles_roundtrip == 1) & (r.multiplicity <= args.max_multiplicity)].copy()
        r["_key"] = r.smiles.astype(str).map(canon)
        r = r.drop_duplicates("_key").set_index("_key")
        for name, s in smi.items():
            k = canon(str(s))
            if k in r.index and name in comp and name in trained:
                hiref[name] = r.loc[k]
        print(f"{len(hiref)} species with a high-temperature reference")
        if not hiref:
            raise SystemExit("no species matched: run nasa7_check.py --smiles_csv on the same reference CSV")

    names = args.species
    if not names:
        pool = list(hiref) if hiref else list(comp)
        if hiref and args.pick_from != "any":
            ok = ("test", "none") if args.pick_from == "unseen" else ("test",)
            pool = [n for n in pool if split.get(n, "test") in ok]
        n_atoms = pd.Series({k: sum(comp[k][0].values()) for k in pool}).sort_values(kind="stable")
        picks = [int(round(q * (len(n_atoms) - 1))) for q in (0.10, 0.50, 0.90)]
        names = list(dict.fromkeys(n_atoms.index[i] for i in picks))
    ref = None
    if args.data_csv:
        d = pd.read_csv(args.data_csv)
        col = next(c for c in d.columns if c.lower() == "smiles")
        d["_key"] = d[col].astype(str).map(canon)
        ref = d.set_index("_key")

    err_panel = bool(args.error_panel and hiref)
    n_pan = len(names) + int(err_panel)
    narrow = n_pan > 3
    plt.rcParams.update({"font.size": 7.5 if narrow else 8, "axes.linewidth": 0.8, "xtick.direction": "in",
                         "ytick.direction": "in", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, n_pan, figsize=(min(2.45 * n_pan, 7.4), 2.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    ticks = [1000, 2000, 3000] if narrow else [500, 1000, 1500, 2000, 2500, 3000]
    T = np.arange(300.0, args.t_max + 1.0, 5.0)
    for k, (ax, name) in enumerate(zip(axes, names)):
        e_c, e_t = comp[name], trained[name]
        s = smi.get(name, "")
        lim = limit(e_c[0], s)
        ax.axvspan(1500.0, args.t_max, color="0.93", lw=0, zorder=0)
        ax.axhline(lim, color="black", ls=":", lw=1.0, zorder=1)
        top = max(cp_of(e_t, T).max(), lim)
        if name in hiref:
            c_ref = cp_ref(hiref[name].nasa, T)
            ax.plot(T, c_ref, "-", color=REF_GREY, lw=3.2, alpha=0.75, solid_capstyle="butt", zorder=1.5)
            top = max(top, np.nanmax(c_ref))
        ax.plot(T, cp_of(e_t, T), "--", color=VERMILION, lw=1.4, zorder=2)
        ax.plot(T, cp_of(e_c, T), "-", color=BLUE, lw=1.6, zorder=3)
        ax.plot(T_DATA, cp_of(e_c, T_DATA), "o", color="black", ms=3.5, zorder=4)
        if ref is not None and canon(s) in ref.index:
            r = ref.loc[canon(s)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            ax.plot(T_DATA, r[DT_CP].astype(float).values * CAL, "s", mfc="white", mec="black", ms=4.5, zorder=5)
        top *= 1.06
        ax.set_ylim(0, top)
        ax.set_xlim(300, args.t_max)
        ax.set_xticks(ticks)
        ax.set_xlabel("Temperature (K)")
        if k == 0:
            ax.set_ylabel(r"$C_p$ (J mol$^{-1}$ K$^{-1}$)")
        ax.set_title(f"({chr(97 + k)}) {formula(e_c[0])}", loc="left", fontsize=8.5)
        ax.text(0.03, 0.04, s, transform=ax.transAxes, fontsize=5.5, color="0.35")
        if name in hiref:
            ax.text(0.03, 0.10, f"ref.: {hiref[name].library}", transform=ax.transAxes, fontsize=5.0, color="0.45")
        ax.text(340, lim * 1.012, "classical limit", fontsize=6.5, va="bottom", color="0.2")
        if k == 0:
            ax.text(1530, top * 0.06 + (0.1 * top if name in hiref else 0.0), "no data\nabove 1500 K",
                    fontsize=6.5, color="0.4", va="bottom")

    if err_panel:
        ax = axes[-1]
        Tg = np.arange(300.0, args.t_max + 1.0, 10.0)
        rel = {"c": [], "t": []}
        for name, row in hiref.items():
            c_ref = cp_ref(row.nasa, Tg)
            rel["c"].append(100 * (cp_of(comp[name], Tg) - c_ref) / c_ref)
            rel["t"].append(100 * (cp_of(trained[name], Tg) - c_ref) / c_ref)
        ax.axvspan(1500.0, args.t_max, color="0.93", lw=0, zorder=0)
        ax.axhline(0.0, color="black", lw=0.8, zorder=1)
        for tag, col, ls in (("t", VERMILION, "--"), ("c", BLUE, "-")):
            a = np.array(rel[tag])
            lo_, med, hi_ = np.nanpercentile(a, [10, 50, 90], axis=0)
            ax.fill_between(Tg, lo_, hi_, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(Tg, med, ls, color=col, lw=1.5, zorder=3)
        ax.set_xlim(300, args.t_max)
        ax.set_xticks(ticks)
        c_hi = np.nanpercentile(np.array(rel["c"]), 90, axis=0)
        t_hi = np.nanpercentile(np.array(rel["t"]), 90, axis=0)
        ymax = max(10.0, min(float(np.nanmax(t_hi)), 40.0), float(np.nanmax(c_hi)))
        ax.set_ylim(-ymax * 0.5, ymax * 1.1)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$(C_p - C_p^\mathrm{ref})/C_p^\mathrm{ref}$ (%)")
        ax.set_title(f"({chr(97 + len(names))}) all {len(hiref)} species", loc="left", fontsize=8.5)
        ax.text(0.03, 0.04, "median, 10th–90th percentile", transform=ax.transAxes, fontsize=5.5, color="0.35")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="--", color=VERMILION, lw=1.4), Line2D([], [], color=BLUE, lw=1.6),
               Line2D([], [], ls="", marker="o", color="black", ms=3.5)]
    labels = ["as trained (straight high range)", "after completion (exported)", "predictions at the data temperatures"]
    if hiref:
        handles.append(Line2D([], [], color=REF_GREY, lw=3.2, alpha=0.75))
        labels.append("reference (RMG-database libraries)")
    if ref is not None:
        handles.append(Line2D([], [], ls="", marker="s", mfc="white", mec="black", ms=4.5))
        labels.append("DeepTherm data" if hiref else "reference data")
    handles.append(Line2D([], [], ls=":", color="black", lw=1.0))
    labels.append("classical limit")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    print("species:", ", ".join(f"{n} ({smi.get(n, '?')}, {split.get(n, '?')}"
                                + (f", {hiref[n].library}" if n in hiref else "") + ")" for n in names))
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()