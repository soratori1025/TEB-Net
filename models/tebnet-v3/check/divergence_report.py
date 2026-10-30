"""Success frequency of the divergent-member failure across split x member ensembles.

Reproduces Section 3.3 of the manuscript ("one ensemble member in ten inverts the
enthalpy increment") and replaces that n = 10 statement by a frequency with a Jeffreys
interval once the five-split sweep is complete.

Definitions (the increment and C16 error are identical to check/aggregation_report.py,
which reproduces Table 5):

  * reference line   straight line in n fitted to the members of each series that are in
                     the DeepTherm set with n >= 3, extrapolated to the members outside;
  * increment        slope per CH2 group, fitted over the members outside the set;
  * divergent member the sign of the increment is opposite to the reference line, i.e.
                     the predicted enthalpy *rises* with chain length.  All reference
                     increments are negative (-4.79 to -5.03 kcal/mol), so a positive
                     slope is a qualitative failure, not a small error;
  * C16 error        |prediction - (-89.60)| kcal/mol for n-hexadecane (NIST).

Usage
    python check/divergence_report.py                  # every split with >= 1 member
    python check/divergence_report.py --json out.json  # machine-readable summary
    python check/divergence_report.py --split random0  # one split only

random0 members are read from check/homologous_members.csv (the file Table 5 and
Tables S9/S10 are built from), so those numbers reproduce without touching a checkpoint.
Other splits are evaluated with check/predict_smiles.py and cached in
check/sweep_predictions/<split>_m<k>.csv, so the script can be re-run while the sweep is
still running and only pays for the new members.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark_results_deeptherm"
TAG = "thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH"
ROOT = BENCH / TAG
SERIES_CSV = HERE / "homologous_series.csv"
CACHE_DIR = HERE / "sweep_predictions"
REF_C16_H = -89.60
N_MEMBERS = 10
def load_series() -> pd.DataFrame:
    df = pd.read_csv(SERIES_CSV)
    df["smiles"] = df["smiles"].astype(str).str.strip()
    return df
def member_dirs(split: str) -> list[Path]:
    d = ROOT / split
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("member*_seed*")) if (p / "results.json").exists()]
def member_h(split: str, member: int, series: pd.DataFrame) -> np.ndarray:
    """Delta-H prediction (kcal/mol) of one member on the 78 series species."""
    cache = CACHE_DIR / f"{split}_m{member}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        col = "teb_H_kcal" if "teb_H_kcal" in df.columns else "H_kcal"
        m = dict(zip(df.smiles.astype(str).str.strip(), df[col].to_numpy(float)))
        return np.array([m.get(s, np.nan) for s in series.smiles], dtype=float)
    if split == "random0":
        legacy = HERE / "homologous_members.csv"
        if legacy.exists():
            df = pd.read_csv(legacy)
            if f"m{member}_H" in df.columns:
                return df[f"m{member}_H"].to_numpy(float)
    import predict_smiles as ps
    ps.setup_paths()
    run = ROOT / split / f"member{member}_seed{member}"
    preds, _, _ = ps.predict([run], list(series.smiles))
    h = np.asarray(preds[0]["h"], dtype=float)
    CACHE_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"smiles": series.smiles, "teb_H_kcal": h}).to_csv(cache, index=False)
    return h
def reference_increments(series: pd.DataFrame) -> dict[str, float]:
    """Slope per CH2 of the reference line of each series (fitted inside the set, n >= 3)."""
    out = {}
    for name, g in series.groupby("series", sort=False):
        fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3)]
        if len(fit) < 2:
            continue
        out[name] = float(np.polyfit(fit.n_carbon.to_numpy(float),
                                     fit.data_H_kcal.to_numpy(float), 1)[0])
    return out
def member_stats(series: pd.DataFrame, pred: np.ndarray, ref_inc: dict[str, float]) -> dict:
    inc = {}
    for name, g in series.groupby("series", sort=False):
        out = g[g.in_deeptherm_set == 0]
        if len(out) < 2:
            continue
        y = pred[out.index.to_numpy()]
        inc[name] = float(np.polyfit(out.n_carbon.to_numpy(float), y, 1)[0]) \
            if np.all(np.isfinite(y)) else float("nan")
    c16 = np.flatnonzero(series.smiles.to_numpy() == "C" * 16)
    c16_err = float(abs(pred[c16[0]] - REF_C16_H)) if len(c16) else float("nan")
    inverted = [k for k, v in inc.items()
                if np.isfinite(v) and k in ref_inc and np.sign(v) != np.sign(ref_inc[k])]
    return {"increment": inc, "c16_error": c16_err,
            "n_series_inverted": len(inverted), "inverted": bool(inverted),
            "worst_abs_increment": max((abs(v) for v in inc.values() if np.isfinite(v)),
                                       default=float("nan"))}
def jeffreys(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Jeffreys interval for a binomial proportion (k events in n trials)."""
    try:
        from scipy.stats import beta
    except ImportError:
        if k == 0:
            return 0.0, 1 - (1 - level) ** (1 / n)
        if k == n:
            return (1 - level) ** (1 / n), 1.0
        raise
    lo = 0.0 if k == 0 else float(beta.ppf((1 - level) / 2, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - level) / 2, k + 0.5, n - k + 0.5))
    return lo, hi
def val_test_mae(run: Path) -> tuple[float, float]:
    res = json.loads((run / "results.json").read_text())
    return (float(res["val"]["H298_kcal"]["mae"]), float(res["test"]["H298_kcal"]["mae"]))
def main() -> None:
    ap = argparse.ArgumentParser(description="Divergent-member frequency over splits x members.")
    ap.add_argument("--split", default=None, help="only this split (default: all)")
    ap.add_argument("--json", default=None, help="write the summary here")
    args = ap.parse_args()
    series = load_series()
    ref_inc = reference_increments(series)
    splits = [args.split] if args.split else sorted(p.name for p in ROOT.glob("random*") if p.is_dir())
    rows, summary = [], {}
    for split in splits:
        runs = member_dirs(split)
        if not runs:
            continue
        done = []
        for run in runs:
            k = int(run.name.split("member")[1].split("_")[0])
            pred = member_h(split, k, series)
            st = member_stats(series, pred, ref_inc)
            v, t = val_test_mae(run)
            st.update(split=split, member=k, val_mae=v, test_mae=t)
            done.append(st)
        n = len(done)
        k_bad = sum(d["inverted"] for d in done)
        lo, hi = jeffreys(k_bad, n)
        summary[split] = {"n_members": n, "n_inverted": k_bad,
                          "rate": k_bad / n, "ci95": [lo, hi],
                          "members": done}
        rows.extend(done)
    if not rows:
        raise SystemExit("no completed members found")
    df = pd.DataFrame(rows)
    print(f"### Per member ({len(df)} members, {df.split.nunique()} split(s))")
    print(f"  {'split':8s} {'mem':>3s} {'val':>6s} {'test':>6s} {'inc range':>18s} "
          f"{'C16':>7s}  inverted")
    for r in rows:
        inc = r["increment"]
        rng = f"{min(inc.values()):+.2f} to {max(inc.values()):+.2f}"
        print(f"  {r['split']:8s} {r['member']:3d} {r['val_mae']:6.3f} {r['test_mae']:6.3f} "
              f"{rng:>18s} {r['c16_error']:7.2f}  {'YES' if r['inverted'] else ''}")
    n_tot = int(df.shape[0])
    k_tot = int(df.inverted.sum())
    lo, hi = jeffreys(k_tot, n_tot)
    print(f"\n### Frequency of the failure")
    print(f"  pooled: {k_tot}/{n_tot} = {100 * k_tot / n_tot:.1f}%  "
          f"(Jeffreys 95% CI {100 * lo:.1f}-{100 * hi:.1f}%)")
    for split, s in summary.items():
        l2, h2 = s["ci95"]
        print(f"  {split}: {s['n_inverted']}/{s['n_members']} = {100 * s['rate']:.1f}%  "
              f"(CI {100 * l2:.1f}-{100 * h2:.1f}%)")
    good, bad = df[~df.inverted], df[df.inverted]
    if len(bad):
        print(f"\n### Validation error does not separate them")
        print(f"  inverted members : val MAE {bad.val_mae.min():.2f}-{bad.val_mae.max():.2f} "
              f"(median {bad.val_mae.median():.2f}), test MAE {bad.test_mae.min():.2f}-"
              f"{bad.test_mae.max():.2f}")
        print(f"  other members    : val MAE {good.val_mae.min():.2f}-{good.val_mae.max():.2f} "
              f"(median {good.val_mae.median():.2f}), test MAE {good.test_mae.min():.2f}-"
              f"{good.test_mae.max():.2f}")
        rank = (df.val_mae.rank() <= df.val_mae.median()).sum()
        print(f"  members in the better half on validation: {int(rank)} of {n_tot}")
        from scipy.stats import spearmanr
        rho, p = spearmanr(df.val_mae, df.c16_error)
        print(f"  Spearman(val MAE, C16 error) = {rho:+.3f} (p = {p:.3f}, n = {n_tot})")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=float))
        print(f"\nwrote {args.json}")
if __name__ == "__main__":
    main()
