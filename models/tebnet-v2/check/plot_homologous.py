"""Figure 4 of the manuscript: size extrapolation on homologous series.

Reads the CSV written by predict_smiles.py (homologous_predictions.csv) and draws
  (a) ΔfH°298 of the n-alkanes against carbon number: values in the DeepTherm set, the straight line fitted to
      the set members with n >= 3 and extrapolated, TEB-Net, the published DeepTherm predictions and the NIST value,
      plus optional controls (--controls label=csv, e.g. the prior alone and the network without the prior);
  (b-d) deviation from that line of ΔfH°298, S°298 and Cp(300 K) for the members of each series that are not in
      the DeepTherm set (TEB-Net solid, DeepTherm dashed), with the NIST value of n-hexadecane where given.
The line is the same as in predict_smiles.py (np.polyfit on in-set members with n >= 3).

Usage
  python plot_homologous.py homologous_predictions.csv --out fig_homologous \
      --controls "prior only=preds_prior_v1.csv" "network without prior=no_prior_h.csv"
  -> fig_homologous.pdf (vector, for the journal) and fig_homologous.png (300 dpi, for checking)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Okabe-Ito colours (colour-blind safe), one per series, in the order of the CSV
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
MARKERS = ["o", "s", "^", "D", "v"]   # second cue next to colour (series identity)
PROPS = {  # key: (data column, TEB-Net column, DeepTherm column, reference column, axis label)
    "H": ("data_H_kcal", "teb_H_kcal", "deeptherm_H_kcal", "ref_H_kcal",
          r"$\Delta_\mathrm{f}H^\circ_{298}$ $-$ line (kcal mol$^{-1}$)"),
    "S": ("data_S_cal", "teb_S_cal", "deeptherm_S_cal", "ref_S_cal",
          r"$S^\circ_{298}$ $-$ line (cal mol$^{-1}$ K$^{-1}$)"),
    "Cp300": ("data_Cp300_cal", "teb_Cp300_cal", "deeptherm_Cp300_cal", None,
              r"$C_p$(300 K) $-$ line (cal mol$^{-1}$ K$^{-1}$)"),
}
# controls in panel (a): colour, line style, marker (distinct from TEB-Net blue and the DeepTherm grey)
CONTROL_STYLE = [("#E69F00", "-.", "x"), ("#009E73", (0, (1, 1.2)), "+"), ("#CC79A7", (0, (4, 1.5)), "1")]
SERIES_LABEL = {"n-alkane": "n-alkanes", "1-alkyl radical": "1-alkyl radicals",
                "1-alkyl hydroperoxide": "1-alkyl hydroperoxides", "1-alkene": "1-alkenes",
                "n-aldehyde": "n-aldehydes"}


def trend_line(g, data_col):
    fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3) & g[data_col].notna()]
    if len(fit) < 3:
        return None
    slope, icpt = np.polyfit(fit.n_carbon, fit[data_col], 1)
    return lambda n: slope * np.asarray(n, dtype=float) + icpt


def control_column(df: pd.DataFrame, prop: str = "H") -> str | None:
    suf = "kcal" if prop == "H" else "cal"
    for c in (f"{prop}_{suf}", f"teb_{prop}_{suf}", f"pred_{prop}_{suf}"):
        if c in df.columns:
            return c
    return None


def panel_absolute(ax, g, controls=()):
    """(a) n-alkane ΔfH°298 against n."""
    data, teb, dt, ref, _ = PROPS["H"]
    line = trend_line(g, data)
    n_in = g[g.in_deeptherm_set == 1].n_carbon.max()
    n_all = np.arange(1, g.n_carbon.max() + 1)
    ax.axvspan(0.5, n_in + 0.5, color="0.92", zorder=0)
    ax.text(1.0, 0.04, "in the DeepTherm set", transform=ax.get_xaxis_transform(), fontsize=7, color="0.35")
    ax.plot(n_all, line(n_all), color="0.45", ls=":", lw=1.2, label="line fitted to the set")
    ins = g[g.in_deeptherm_set == 1]
    ax.plot(ins.n_carbon, ins[data], "o", color="black", ms=4, label="DeepTherm set")
    if teb in g:
        ax.plot(g.n_carbon, g[teb], "-s", color=COLOURS[0], ms=3.5, lw=1.2, label="TEB-Net")
    for k, (label, values) in enumerate(controls):
        c, ls, mk = CONTROL_STYLE[k % len(CONTROL_STYLE)]
        v = g.smiles.astype(str).str.strip().map(values)
        ok = v.notna()
        ax.plot(g.n_carbon[ok], v[ok], color=c, ls=ls, marker=mk, ms=4, lw=1.1, label=label)
    ax.plot(g.n_carbon, g[dt], "--^", color="0.25", mfc="white", ms=4, lw=1.0, label="DeepTherm (published)")
    r = g[g[ref].notna()] if ref in g else g.iloc[:0]
    if len(r):
        ax.plot(r.n_carbon, r[ref], "*", color="#B2182B", ms=10, label="experiment (NIST)", zorder=5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi)
    ax.set_xlabel("Number of carbon atoms")
    ax.set_ylabel(r"$\Delta_\mathrm{f}H^\circ_{298}$ (kcal mol$^{-1}$)")
    ax.set_title("(a) n-alkanes", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.07), handlelength=1.8)


def panel_deviation(ax, df, prop, title):
    data, teb, dt, ref, ylabel = PROPS[prop]
    ax.axhline(0.0, color="0.45", lw=0.8, ls=":")
    for k, (name, g) in enumerate(df.groupby("series", sort=False)):
        g = g.sort_values("n_carbon")
        line = trend_line(g, data)
        if line is None:
            continue
        out = g[g.in_deeptherm_set == 0]
        c, mk = COLOURS[k % len(COLOURS)], MARKERS[k % len(MARKERS)]
        if teb in out:
            ax.plot(out.n_carbon, out[teb] - line(out.n_carbon), "-", marker=mk, color=c, ms=3.5, lw=1.2)
        o = out[out[dt].notna()]
        ax.plot(o.n_carbon, o[dt] - line(o.n_carbon), "--", marker=mk, color=c, mfc="white", ms=3.5, lw=1.0)
        if ref and ref in out:
            r = out[out[ref].notna()]
            if len(r):
                ax.plot(r.n_carbon, r[ref] - line(r.n_carbon), "*", color="#B2182B", ms=9, zorder=5)
    ax.set_xlabel("Number of carbon atoms")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("--out", default="fig_homologous")
    ap.add_argument("--controls", nargs="*", default=[],
                    help="label=csv with smiles and H_kcal (or teb_H_kcal / pred_H_kcal), drawn in panel (a)")
    args = ap.parse_args()
    df = pd.read_csv(args.predictions_csv)
    controls = []
    for spec in args.controls:
        label, path = spec.split("=", 1)
        c = pd.read_csv(path)
        col = control_column(c, "H")
        if col is None:
            raise SystemExit(f"{path}: no H_kcal / teb_H_kcal / pred_H_kcal column")
        controls.append((label.strip(), dict(zip(c["smiles"].astype(str).str.strip(), c[col]))))
    missing = [c for c in ("teb_H_kcal", "teb_S_cal", "teb_Cp300_cal") if c not in df]
    if missing:
        print(f"warning: {missing} not in the CSV; those TEB-Net curves are left out")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    panel_absolute(axes[0, 0], df[df.series == "n-alkane"].sort_values("n_carbon"), controls)
    panel_deviation(axes[0, 1], df, "H", r"(b) $\Delta_\mathrm{f}H^\circ_{298}$, outside the set")
    panel_deviation(axes[1, 0], df, "S", r"(c) $S^\circ_{298}$, outside the set")
    panel_deviation(axes[1, 1], df, "Cp300", r"(d) $C_p$(300 K), outside the set")
    for ax in axes.flat[1:]:
        ax.set_xlim(6.5, df.n_carbon.max() + 0.5)
    for ax in axes.flat:
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))

    names = [SERIES_LABEL.get(s, s) for s in df.series.drop_duplicates()]
    handles = [Line2D([], [], color=COLOURS[k % len(COLOURS)], marker=MARKERS[k % len(MARKERS)], ms=4, lw=1.5)
               for k in range(len(names))]
    handles += [Line2D([], [], color="0.2", ls="-", marker="o", ms=3),
                Line2D([], [], color="0.2", ls="--", marker="o", mfc="white", ms=3),
                Line2D([], [], color="#B2182B", ls="", marker="*", ms=8)]
    names += ["TEB-Net", "DeepTherm (published)", "experiment (NIST)"]
    fig.legend(handles, names, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()