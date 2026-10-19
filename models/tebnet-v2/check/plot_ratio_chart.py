"""Figure 2 of the manuscript: TEB-Net MAE relative to the best published MAE, for all 24 comparisons.

The numbers are those of the DeepTherm, Ureel, WUDILY-CHO and hold-out tables of the manuscript (copied below;
edit them here if a table changes). WUDILY-CHO: single models, 5-fold CV, against CDS-RF of Bruce et al. (2025),
with Cp averaged over 300-1500 K as in their Table 4.
Values that DeepTherm reports only in figures are given as (low, high) ranges; the point is drawn against the LOW
end of the range (the comparison least favourable to TEB-Net) and the whisker spans the whole range. This is the
convention behind the counts in Section 3.1 (11 within +/-10 %, 4 better, 9 worse).

Usage
  python plot_ratio_chart.py --out fig_ratio      ->  fig_ratio.pdf, fig_ratio.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

H, S, CP = r"$\Delta_\mathrm{f}H^\circ_{298}$", r"$S^\circ_{298}$", r"$C_p$(300 K)"
CPM = r"$C_p$(300–1500 K)"      # WUDILY-CHO: MAE averaged over the seven temperatures
# (group, property, TEB-Net MAE, best published MAE or (low, high), published model)
ROWS = [
    ("DeepTherm set, random split", H, 1.28, 1.25, "DeepTherm"),
    ("DeepTherm set, random split", S, 1.41, 1.32, "DeepTherm"),
    ("DeepTherm set, random split", CP, 0.82, 0.87, "DeepTherm"),
    ("DeepTherm set, complexity split", H, 1.89, (2.05, 2.2), "DeepTherm"),
    ("DeepTherm set, complexity split", S, 2.81, (2.17, 2.4), "DeepTherm"),
    ("DeepTherm set, complexity split", CP, 1.41, (1.45, 2.0), "DeepTherm"),
    ("Ureel hydrocarbons", H, 2.53, 2.52, "transfer learning"),
    ("Ureel hydrocarbons", S, 2.97, 2.97, "transfer learning"),
    ("Ureel hydrocarbons", CP, 1.73, 1.66, "transfer learning"),
    ("Ureel radicals", H, 2.17, 2.17, "transfer learning"),
    ("Ureel radicals", S, 2.80, 2.79, "transfer learning"),
    ("Ureel radicals", CP, 1.66, 1.65, "transfer learning"),
    ("Ureel carbenium ions", H, 5.15, 3.67, "transfer learning"),
    ("Ureel carbenium ions", S, 8.53, 6.04, "transfer learning"),
    ("Ureel carbenium ions", CP, 2.55, 1.97, "transfer learning"),
    ("WUDILY-CHO set, 5-fold CV", H, 1.25, 0.84, "CDS-RF"),
    ("WUDILY-CHO set, 5-fold CV", S, 1.07, 0.83, "CDS-RF"),
    ("WUDILY-CHO set, 5-fold CV", CPM, 0.53, 0.86, "CDS-RF"),
    ("Hold-out: ether", H, 5.80, 1.38, "DeepTherm"),
    ("Hold-out: ether", S, 2.75, 1.77, "DeepTherm"),
    ("Hold-out: epoxide", H, 3.63, 1.85, "DeepTherm"),
    ("Hold-out: epoxide", S, 1.95, 2.57, "DeepTherm"),
    ("Hold-out: enol", H, 3.35, 3.86, "DeepTherm"),
    ("Hold-out: enol", S, 1.95, 3.65, "DeepTherm"),
]
BAND = 0.10
# outcome -> (colour, marker); blue / neutral grey / vermilion, with the marker shape as second cue
STYLE = {"better": ("#0072B2", "v"), "parity": ("#8A8A8A", "o"), "worse": ("#D55E00", "^")}


def outcome(r):
    return "better" if r < 1 - BAND else "worse" if r > 1 + BAND else "parity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_ratio")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                         "pdf.fonttype": 42})

    # y positions: rows top to bottom, a gap between groups
    ys, y, prev, group_mid = [], 0.0, None, {}
    for g, *_ in ROWS:
        if prev is not None and g != prev:
            y += 0.8
        ys.append(y)
        group_mid.setdefault(g, []).append(y)
        prev = g
        y += 1.0
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(6.3, 6.0), constrained_layout=True)
    ax.axvspan(1 - BAND, 1 + BAND, color="0.90", zorder=0, lw=0)
    ax.axvline(1.0, color="0.45", lw=0.8, zorder=1)
    counts = {"better": 0, "parity": 0, "worse": 0}
    for yi, (g, prop, teb, pub, model) in zip(ys, ROWS):
        if isinstance(pub, tuple):
            r_lo, r_hi = teb / pub[1], teb / pub[0]       # against the high and the low end of the range
            ax.plot([r_lo, r_hi], [yi, yi], color="0.35", lw=1.0, zorder=2, solid_capstyle="butt")
            r = r_hi
        else:
            r = teb / pub
        o = outcome(r)
        counts[o] += 1
        c, m = STYLE[o]
        ax.plot(r, yi, m, ms=6.5, color=c, mec="white", mew=0.8, zorder=3)
        ax.annotate(f"{teb:.2f} / " + (f"{pub[0]:.2f}–{pub[1]:.2f}" if isinstance(pub, tuple) else f"{pub:.2f}"),
                    xy=(1.0, yi), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.5, color="0.30")
    ax.set_yticks(ys, [p for _, p, *_ in ROWS], fontsize=7)
    ax.set_ylim(ys.max() + 0.8, -0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 5.0)
    ticks = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("TEB-Net MAE / best published MAE (log scale)")
    # group names to the left of the tick labels
    for g, yy in group_mid.items():
        ax.annotate(g, xy=(0, np.mean(yy)), xycoords=("axes fraction", "data"), xytext=(-62, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.annotate("MAE\nTEB-Net / published", xy=(1.0, -0.8), xycoords=("axes fraction", "data"), xytext=(6, 2),
                textcoords="offset points", va="bottom", ha="left", fontsize=6.5, color="0.30")
    handles = [Line2D([], [], ls="", marker=STYLE[k][1], color=STYLE[k][0], mec="white", ms=7) for k in STYLE]
    labels = [f"more than 10 % better ({counts['better']})", f"within $\\pm$10 % ({counts['parity']})",
              f"more than 10 % worse ({counts['worse']})"]
    handles.append(Line2D([], [], color="0.35", lw=1.0))
    labels.append("range of a value read from a figure")
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    print("counts:", counts)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    print(f"wrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()