import pandas as pd
import numpy as np
import subprocess
import os
from sklearn.metrics import mean_absolute_error

def main():
    print("Running Chemprop predict on all data using replicate_0 model...")
    # Predict using the first fold model on a single GPU to avoid DDP index bug
    cmd = [
        "../.venv/bin/chemprop", "predict",
        "--test-path", "hc_sota_clean.csv",
        "--smiles-column", "smiles",
        "--model-path", "chemprop_checkpoints/replicate_0/model_0/best.pt",
        "--preds-path", "chemprop_preds.csv"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    
    df_true = pd.read_csv("hc_sota_clean.csv")
    df_pred = pd.read_csv("chemprop_preds.csv")
    
    # Merge on smiles to ensure order
    df_merged = pd.merge(df_true, df_pred, on='smiles', suffixes=('_true', '_pred'))
    
    # Calculate MAE for H298
    h298_mae = mean_absolute_error(df_merged['dH_f_298_true'], df_merged['dH_f_298_pred'])
    
    # Calculate MAE for S298
    s298_mae = mean_absolute_error(df_merged['S_298_true'], df_merged['S_298_pred'])
    
    # Calculate MAE for Cp (18 columns)
    cp_cols = [c for c in df_true.columns if c.startswith('Cp_')]
    cp_true = df_merged[[c + '_true' for c in cp_cols]].values
    cp_pred = df_merged[[c + '_pred' for c in cp_cols]].values
    cp_mae = mean_absolute_error(cp_true, cp_pred)
    
    print("="*50)
    print("Chemprop Baseline Results (Replicate 0, Eval on ALL data)")
    print(f"H298 MAE : {h298_mae:.2f} J/mol")
    print(f"S298 MAE : {s298_mae:.2f} J/(mol K)")
    print(f"Cp MAE   : {cp_mae:.2f} J/(mol K)")
    print("="*50)

if __name__ == "__main__":
    main()
