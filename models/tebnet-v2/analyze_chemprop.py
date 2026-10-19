import pandas as pd
import numpy as np
from pathlib import Path

def fit_line(df, col):
    d = df[df.in_deeptherm_set == 1].dropna(subset=[col])
    if len(d) < 2: return None
    return np.polyfit(d.n_carbon, d[col], 1)

def get_maes(df, pred_col, line_poly):
    d = df[(df.in_deeptherm_set == 0) & df[pred_col].notna()]
    if d.empty: return np.nan
    line_preds = np.polyval(line_poly, d.n_carbon)
    return np.mean(np.abs(d[pred_col] - line_preds))

def get_inc(df, pred_col):
    d = df[(df.in_deeptherm_set == 0) & df[pred_col].notna()]
    if len(d) < 2: return np.nan
    return np.polyfit(d.n_carbon, d[pred_col], 1)[0]

ser = pd.read_csv("check/homologous_series.csv")

for agg in ["mean", "sum"]:
    print(f"\n--- chemprop {agg} ---")
    for s in [0, 1, 2]:
        p = Path(f"runs_agg/agg_{agg}_H_s{s}/preds_homologous.csv")
        if not p.exists(): continue
        pred = pd.read_csv(p)
        merged = ser.merge(pred, on="smiles", how="left")
        
        print(f"  Seed {s}:")
        for name in ["n-alkane", "1-alkyl radical", "1-alkyl hydroperoxide", "1-alkene", "n-aldehyde"]:
            sub = merged[merged.series == name]
            poly = fit_line(sub, "data_H_kcal")
            inc = get_inc(sub, "H_kcal")
            mae = get_maes(sub, "H_kcal", poly) if poly is not None else np.nan
            print(f"    {name}: inc = {inc:.2f}, mae = {mae:.2f}")

