"""Ensemble aggregation on the homologous series: weighted mean, trimmed mean, median.

Reproduces Table 5 of the manuscript and Tables S9 / S10 of SM3 from the stored per-member
predictions (check/homologous_members.csv, exported by check/export_homologous_members.py).

Definitions (as in the manuscript):
  * reference line   straight line in n fitted to the members of each series that are in the
                     DeepTherm set with n >= 3, extrapolated to the members outside the set;
  * increment        slope per CH2 group, fitted over the members outside the set;
  * MAE from lines   absolute deviation from the line, averaged over the members outside the
                     set, then averaged over the five series (unweighted, as in Table 5);
  * C16 error        absolute error of n-hexadecane against NIST (-89.60 kcal/mol).

Usage
    python check/aggregation_report.py                 # full report
    python check/aggregation_report.py --csv out.csv   # also write the per-query table
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark_results_deeptherm"
H_ROOT = BENCH / "thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH" / "random0"
SCP_ROOT = BENCH / "thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp_sp-v2_exact_splitSCp" / "random0"
N_MEMBERS = 10
REF_C16_H = -89.60
REF_C16_S = 186.0
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
COLS = {"H": "data_H_kcal", "S": "data_S_cal", "Cp300": "data_Cp300_cal"}
def trimmed_mean(a: np.ndarray) -> np.ndarray:
    """Trimmed mean over axis 0, discarding the lowest and the highest value at each point."""
    return np.sort(a, axis=0)[1:-1].mean(axis=0)
def median(a: np.ndarray) -> np.ndarray:
    return np.median(a, axis=0)
AGGS = {"weighted mean": None, "trimmed mean": trimmed_mean, "median": median}
def load_members(csv: Path | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read the per-member predictions; returns the table and {prop: (k, n_queries)}."""
    path = csv or (HERE / "homologous_members.csv")
    df = pd.read_csv(path)
    mats = {p: np.stack([df[f"m{k}_{p}"].to_numpy(float) for k in range(N_MEMBERS)])
            for p in ("H", "S", "Cp300")}
    return df, mats
def reference_lines(df: pd.DataFrame, prop: str) -> np.ndarray:
    line = np.full(len(df), np.nan)
    for _, g in df.groupby("series", sort=False):
        fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3)]
        slope, icpt = np.polyfit(fit.n_carbon, fit[COLS[prop]], 1)
        out = g[g.in_deeptherm_set == 0].index.to_numpy()
        line[out] = slope * g.loc[out, "n_carbon"].to_numpy() + icpt
    return line
def validation_weights(prop: str = "H") -> np.ndarray:
    """Inverse validation MAE, normalised (as in DeepTherm and in the manuscript).

    Delta-H models are weighted by their validation enthalpy MAE, S/Cp models by the mean of
    the validation S and Cp MAEs over the nine targets (as in train.py / deeptherm_eval.py).
    """
    root = H_ROOT if prop == "H" else SCP_ROOT
    v = []
    for k in range(N_MEMBERS):
        val = json.loads((root / f"member{k}_seed{k}" / "results.json").read_text())["val"]
        if prop == "H":
            v.append(val["H298_kcal"]["mae"])
        else:
            n_t = len(json.loads((root / f"member{k}_seed{k}" / "results.json").read_text())["t_grid"])
            v.append((val["S298_cal"]["mae"] + n_t * val["Cp_mean_MAE_cal"]) / (1 + n_t))
    w = 1.0 / np.array(v, float)
    return w / w.sum()
def summarise(df: pd.DataFrame, agg: np.ndarray, prop: str, line: np.ndarray) -> dict:
    """Increment range, mean MAE from the lines and (for H and S) the C16 error."""
    out = (df.in_deeptherm_set == 0).to_numpy() & np.isfinite(line)
    inc, per_series = [], []
    for _, g in df.groupby("series", sort=False):
        idx = g[g.in_deeptherm_set == 0].index.to_numpy()
        inc.append(np.polyfit(g.loc[idx, "n_carbon"].to_numpy(), agg[idx], 1)[0])
        per_series.append(np.abs(agg[idx] - line[idx]).mean())
    ref = {"H": REF_C16_H, "S": REF_C16_S}.get(prop)
    c16 = np.nan
    if ref is not None:
        j = int(np.flatnonzero(df.smiles.astype(str).to_numpy() == "C" * 16)[0])
        c16 = abs(agg[j] - ref)
    return {"inc_min": min(inc), "inc_max": max(inc), "mae": float(np.mean(per_series)),
            "c16": c16, "worst": float(np.abs(agg[out] - line[out]).max())}
def in_distribution(prop: str, members: list[int] | None = None) -> tuple[dict[str, float], int]:
    """MAE of each aggregator on the common test set of the decoupled trees.

    member 0 of the Delta-H tree was written with the 60/10/30 fractions (1500 test rows,
    2998 training rows); every member is aligned on test_indices before aggregating, and
    `members` selects which of them enter the ensemble.
    """
    root = H_ROOT if prop == "H" else SCP_ROOT
    keys = {"H": ("test_h_pred", "test_h_true"), "S": ("test_s_pred", "test_s_true"),
            "Cp300": ("test_cp_pred", "test_cp_true")}[prop]
    keep = list(range(N_MEMBERS)) if members is None else members
    per = []
    for k in keep:
        z = np.load(root / f"member{k}_seed{k}" / "predictions.npz", allow_pickle=True)
        per.append((np.asarray(z[keys[0]], float), np.asarray(z[keys[1]], float),
                    np.asarray(z["test_indices"]).ravel()))
    common = per[0][2]
    for _, _, idx in per[1:]:
        common = np.intersect1d(common, idx)
    preds, true = [], None
    for p, t, idx in per:
        pos = {v: j for j, v in enumerate(idx)}
        sel = np.array([pos[v] for v in common])
        preds.append(p[sel])
        true = t[sel]
    P = np.stack(preds)
    scale = J_TO_KCAL if prop == "H" else J_TO_CAL
    P, true = P * scale, true * scale
    if P.ndim == 3:
        P, true = P[:, :, 0], true[:, 0]
    w = validation_weights(prop)[keep]
    w = w / w.sum()
    res = {"weighted mean": float(np.abs(np.tensordot(w, P, axes=(0, 0)) - true).mean()),
           "trimmed mean": float(np.abs(trimmed_mean(P) - true).mean()),
           "median": float(np.abs(median(P) - true).mean())}
    return res, len(common)
def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble aggregation on the homologous series.")
    ap.add_argument("--csv", default=None, help="write the per-query aggregated table here")
    args = ap.parse_args()
    df, mats = load_members()
    lines = {p: reference_lines(df, p) for p in mats}
    weights = {p: validation_weights(p) for p in mats}
    print(f"Ensemble of {N_MEMBERS} members; {int((df.in_deeptherm_set == 0).sum())} homologues "
          f"outside the DeepTherm set; H weights {weights['H'].min():.3f} - {weights['H'].max():.3f}\n")
    for prop, unit in (("H", "kcal/mol"), ("S", "cal/(mol K)"), ("Cp300", "cal/(mol K)")):
        print(f"### {prop} ({unit})")
        print(f"  {'aggregation':14s} {'increment per CH2':>20s} {'MAE from lines':>15s} "
              f"{'worst':>7s} {'C16 error':>10s}")
        for name, fn in AGGS.items():
            agg = np.tensordot(weights[prop], mats[prop], axes=(0, 0)) if fn is None else fn(mats[prop])
            st = summarise(df, agg, prop, lines[prop])
            c16 = "        --" if np.isnan(st["c16"]) else f"{st['c16']:10.2f}"
            print(f"  {name:14s} {st['inc_min']:+8.2f} to {st['inc_max']:+.2f} "
                  f"{st['mae']:15.2f} {st['worst']:7.2f} {c16}")
        print()
    print("### In-distribution (common test set)")
    for prop in ("H", "S", "Cp300"):
        res, n = in_distribution(prop)
        print(f"  {prop:6s} n={n}: " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))
    res, n = in_distribution("H", members=[k for k in range(N_MEMBERS) if k != 0])
    print(f"  H      n={n} (excl. member 0, which was trained on the 60/10/30 split): "
          + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))
    c16 = int(np.flatnonzero(df.smiles.astype(str).to_numpy() == "C" * 16)[0])
    err = np.abs(mats["H"][:, c16] - REF_C16_H)
    print("\n### Per-member C16 error (H)")
    print("  " + "  ".join(f"m{k}={e:.2f}" for k, e in enumerate(err)))
    print(f"  mean {err.mean():.2f} +- {err.std(ddof=1):.2f}, median {np.median(err):.2f}, "
          f"nine members (excl. m7) {np.delete(err, 7).mean():.2f}")
    if args.csv:
        out = df[["series", "n_carbon", "smiles", "in_deeptherm_set"]].copy()
        for prop in mats:
            out[f"line_{prop}"] = lines[prop]
            out[f"weighted_{prop}"] = np.tensordot(w, mats[prop], axes=(0, 0))
            out[f"trimmed_{prop}"] = trimmed_mean(mats[prop])
            out[f"median_{prop}"] = median(mats[prop])
        out.to_csv(args.csv, index=False, float_format="%.4f")
        print(f"\nwrote {args.csv}")
if __name__ == "__main__":
    main()
