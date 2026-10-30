import pandas as pd
import numpy as np
for agg in ["mean", "sum"]:
    print(f"\n--- chemprop {agg} ---")
    dfs = []
    for s in [0, 1, 2]:
        path = f"runs_agg/agg_{agg}_H_s{s}/preds_homologous.csv"
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except:
            pass
    for name in ["n-alkane", "1-alkyl radical", "1-alkyl hydroperoxide", "1-alkene", "n-aldehyde"]:
        incs = []
        maes = []
        for df in dfs:
            g = df[df['series'] == name].copy()
            g = g[g.in_deeptherm_set == 0]
            if g.empty: continue
            o = g[g.teb_H_kcal.notna()]
            if len(o) >= 2:
                inc = np.polyfit(o.n_carbon, o.teb_H_kcal, 1)[0]
                incs.append(inc)
            if 'line_H_kcal' in g:
                m = g.teb_H_kcal.notna() & g.line_H_kcal.notna()
                if m.any():
                    maes.append(np.abs(g.loc[m, 'teb_H_kcal'] - g.loc[m, 'line_H_kcal']).mean())
        if incs:
            print(f"{name}: inc = [{min(incs):.2f} .. {max(incs):.2f}], MAE = [{min(maes):.2f} .. {max(maes):.2f}]")
