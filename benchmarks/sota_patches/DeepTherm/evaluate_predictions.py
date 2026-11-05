import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = pd.read_csv('predictions_new.csv')
targets = ['Hf_298', 'S_298', 'Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000', 'Cp_1500']
results = []

for t in targets:
    pred_col = f'pred_{t}'
    if t in df.columns and pred_col in df.columns:
        valid = df[[t, pred_col]].dropna()
        if len(valid) > 0:
            mae = mean_absolute_error(valid[t], valid[pred_col])
            rmse = np.sqrt(mean_squared_error(valid[t], valid[pred_col]))
            results.append({'Target': t, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)})

res_df = pd.DataFrame(results)
print(res_df.to_markdown(index=False))
print(f'\n**Mean MAE across all targets**: {res_df["MAE"].mean():.4f}')
print(f'**Mean RMSE across all targets**: {res_df["RMSE"].mean():.4f}')
