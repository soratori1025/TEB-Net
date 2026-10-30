"""Render the five-split divergence sweep as the LaTeX table quoted in the manuscript.

Reads what check/divergence_report.py computes (per-member increment range, C16 error,
validation MAE) and writes a table in the style of Tables S9/S10. Refuses to write a
partial table unless --allow-partial is given, so a half-finished sweep cannot end up in
the supplement by accident.

Usage
    python check/divergence_table.py --out ../teb_manuscript_overleaf/si_SM3c_table_S12_sweep.tex
    python check/divergence_table.py --out /tmp/t.tex --allow-partial   # while running
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import divergence_report as dr
def fmt_range(lo: float, hi: float) -> str:
    """A range that collapses to a single value when both ends coincide."""
    if abs(hi - lo) < 5e-3:
        return f"{lo:.2f}"
    return f"{lo:.2f}--{hi:.2f}"
def fmt_signed(x: float) -> str:
    """Signed number in math mode, so that the minus sign is a minus and not a hyphen."""
    return f"${x:+.2f}$"
def collect(splits: list[str]) -> tuple[pd.DataFrame, dict]:
    series = dr.load_series()
    ref_inc = dr.reference_increments(series)
    rows = []
    for split in splits:
        for run in dr.member_dirs(split):
            k = int(run.name.split("member")[1].split("_")[0])
            pred = dr.member_h(split, k, series)
            st = dr.member_stats(series, pred, ref_inc)
            v, t = dr.val_test_mae(run)
            rows.append({"split": split, "member": k, "val": v, "test": t,
                         "inc_min": min(st["increment"].values()),
                         "inc_max": max(st["increment"].values()),
                         "c16": st["c16_error"], "inverted": st["inverted"]})
    return pd.DataFrame(rows), ref_inc
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()
    splits = args.splits or sorted(p.name for p in dr.ROOT.glob("random*") if p.is_dir())
    df, ref_inc = collect(splits)
    if df.empty:
        raise SystemExit("no completed members")
    per_split = df.groupby("split").size()
    complete = (per_split == dr.N_MEMBERS).all() and len(per_split) == 5
    if not complete and not args.allow_partial:
        raise SystemExit(f"sweep incomplete ({dict(per_split)}); pass --allow-partial to render anyway")
    n, k = len(df), int(df.inverted.sum())
    lo, hi = dr.jeffreys(k, n)
    inc_lo, inc_hi = df.inc_min.min(), df.inc_max.max()
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\renewcommand{\thetable}{S12}",
        r"\caption{Frequency of the divergent member across the five random splits. Each split has an "
        r"independent ten-member $\Delta_\mathrm{f}H^\circ_{298}$ ensemble (reshuffled training and "
        r"validation sets, fixed 501-species test set); every member is scored on the 45 homologues "
        r"outside the DeepTherm set. Inverted: the sign of the increment per CH$_2$ is opposite to that "
        r"of the reference line ($-4.79$ to $-5.03$~\kcal), i.e.\ the predicted enthalpy rises with chain "
        r"length. C$_{16}$: absolute error for \textit{n}-hexadecane against NIST.}",
        r"\label{tab:S12}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Split & Members & Inverted & \thead{Validation MAE of\\the inverted members} & \thead{C$_{16}$ error of\\the inverted members} \\",
        r"\midrule",
    ]
    for split in sorted(df.split.unique()):
        g = df[df.split == split]
        bad = g[g.inverted]
        v = "--" if bad.empty else fmt_range(bad.val.min(), bad.val.max())
        c = "--" if bad.empty else fmt_range(bad.c16.min(), bad.c16.max())
        lines.append(f"{split.replace('random', 'Random split ')} & {len(g)} & {len(bad)} & {v} & {c} \\\\")
    lines += [
        r"\midrule",
        f"All splits & {n} & {k} & \\multicolumn{{2}}{{c}}{{{100 * k / n:.1f}\\% (Jeffreys 95\\% CI "
        f"{100 * lo:.1f}--{100 * hi:.1f}\\%)}} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip",
        r"\begin{minipage}{\linewidth}\footnotesize",
        f"Members of every split have in-distribution validation MAE between {df.val.min():.2f} and "
        f"{df.val.max():.2f}~\\kcal{{}} and test MAE between {df.test.min():.2f} and {df.test.max():.2f}~\\kcal{{}}, "
        f"so no member is an obvious training failure. The increment of the ensemble lies between "
        f"{fmt_signed(inc_lo)} and {fmt_signed(inc_hi)}~\\kcal{{}} per CH$_2$ over all {n} members and five series. "
        r"Produced by \texttt{check/divergence\_table.py} from \texttt{check/divergence\_report.py}.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {n} members, {k} inverted, CI {100 * lo:.1f}-{100 * hi:.1f}%")
if __name__ == "__main__":
    main()
