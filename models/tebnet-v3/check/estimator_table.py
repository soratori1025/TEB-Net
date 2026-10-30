"""Five splits x three aggregators on the homologous series (Table S12, panel b).

Definitions follow check/aggregation_report.py and check/divergence_report.py:
  * reference line  straight line in n fitted to the in-set members of each series with n >= 3;
  * increment       slope per CH2 over the members outside the set;
  * MAE from lines  mean absolute deviation from the line over the 45 out-of-set homologues,
                    averaged over the five series;
  * C16 error       |prediction - (-89.60)| kcal/mol for n-hexadecane.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
BENCH = Path("benchmark_results_deeptherm")
H_TAG = "thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH"
CACHE = Path("check/sweep_predictions")
SERIES = Path("check/homologous_series.csv")
LEGACY = Path("check/homologous_members.csv")
REF_C16 = -89.60
K = 10
def reference_lines(df: pd.DataFrame) -> np.ndarray:
    line = np.full(len(df), np.nan)
    for _, g in df.groupby("series", sort=False):
        fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3)]
        slope, icpt = np.polyfit(fit.n_carbon.astype(float), fit.data_H_kcal.astype(float), 1)
        out = g[g.in_deeptherm_set == 0].index.to_numpy()
        line[out] = slope * g.loc[out, "n_carbon"].to_numpy(float) + icpt
    return line
def member_matrix(split: str, df: pd.DataFrame) -> np.ndarray:
    rows = []
    for k in range(K):
        cache = CACHE / f"{split}_m{k}.csv"
        if cache.exists():
            d = pd.read_csv(cache)
            col = "teb_H_kcal" if "teb_H_kcal" in d else "H_kcal"
            m = dict(zip(d.smiles.astype(str).str.strip(), d[col].astype(float)))
            rows.append([m.get(s, np.nan) for s in df.smiles])
        elif split == "random0":
            rows.append(df[f"m{k}_H"].to_numpy(float))
        else:
            raise SystemExit(f"missing cache for {split} member {k}")
    return np.stack(rows)
def weights(split: str) -> np.ndarray:
    v = []
    for k in range(K):
        res = json.loads((BENCH / H_TAG / split / f"member{k}_seed{k}" / "results.json").read_text())
        v.append(res["val"]["H298_kcal"]["mae"])
    w = 1.0 / np.asarray(v, float)
    return w / w.sum()
def summarise(df: pd.DataFrame, pred: np.ndarray, line: np.ndarray) -> tuple[str, float, float]:
    incs, series_mae = [], []
    for _, g in df.groupby("series", sort=False):
        out = g[g.in_deeptherm_set == 0]
        idx = out.index.to_numpy()
        incs.append(np.polyfit(out.n_carbon.astype(float), pred[idx], 1)[0])
        series_mae.append(float(np.abs(pred[idx] - line[idx]).mean()))
    mae = float(np.mean(series_mae))
    c16 = float(np.flatnonzero(df.smiles.to_numpy() == "C" * 16)[0])
    return f"{min(incs):+.2f} to {max(incs):+.2f}", mae, abs(pred[int(c16)] - REF_C16)
def main() -> None:
    df = pd.read_csv(SERIES)
    df["smiles"] = df["smiles"].astype(str).str.strip()
    legacy = pd.read_csv(LEGACY)
    df = df.merge(legacy[[c for c in legacy.columns if c.startswith("m")] +
                         ["smiles"]], on="smiles", how="left")
    line = reference_lines(df)
    print(f"{'split':8s} {'aggregator':15s} {'increment/CH2':>20s} {'MAE lines':>9s} {'C16 err':>8s}")
    for split in ["random0", "random1", "random2", "random3", "random4"]:
        mat = member_matrix(split, df)
        w = weights(split)
        preds = {
            "weighted mean": w @ mat,
            "trimmed mean": np.sort(mat, axis=0)[1:-1].mean(axis=0),
            "median": np.median(mat, axis=0),
        }
        for name, p in preds.items():
            inc, mae, c16 = summarise(df, p, line)
            print(f"{split:8s} {name:15s} {inc:>20s} {mae:9.2f} {c16:8.2f}")
if __name__ == "__main__":
    main()
