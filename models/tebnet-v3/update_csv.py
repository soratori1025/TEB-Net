import pandas as pd

h_alkanes = {5: -35.0, 6: -39.96, 7: -44.89, 8: -49.82, 9: -54.76, 10: -59.68, 11: -64.60, 12: -69.52, 13: -74.49, 14: -79.47, 15: -84.50, 16: -89.60}
s_alkanes = {5: 83.25, 6: 92.83, 7: 102.22, 8: 111.55, 9: 121.05, 10: 130.45, 11: 139.73, 12: 148.97, 13: 158.26, 14: 167.54, 15: 176.78, 16: 186.02}

h_alkenes = {5: -5.26, 6: -9.97, 7: -14.89, 8: -19.82, 9: -24.75, 10: -29.78, 11: -34.50, 12: -39.46, 13: -44.40, 14: -49.33, 15: -54.26, 16: -59.19}

df = pd.read_csv("check/homologous_series.csv")
for idx, row in df.iterrows():
    n = row['n_carbon']
    if row['series'] == 'n-alkane' and n in h_alkanes:
        df.at[idx, 'ref_H_kcal'] = h_alkanes[n]
        df.at[idx, 'ref_S_cal'] = s_alkanes[n]
        df.at[idx, 'ref_source'] = "NIST WebBook"
    elif row['series'] == '1-alkene' and n in h_alkenes:
        df.at[idx, 'ref_H_kcal'] = h_alkenes[n]
        df.at[idx, 'ref_source'] = "NIST WebBook"

df.to_csv("check/homologous_series.csv", index=False)
print("Updated CSV!")
