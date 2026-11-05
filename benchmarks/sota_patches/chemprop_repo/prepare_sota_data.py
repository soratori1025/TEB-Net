import pandas as pd
import numpy as np

def main():
    print("Preparing clean dataset for SOTA models...")
    df = pd.read_csv("../data/hc_thermo_combined_v7.csv")
    
    # Extract only smiles and target columns
    target_cols = ['dH_f_298', 'S_298'] + [c for c in df.columns if c.startswith('Cp_')]
    assert len(target_cols) == 20
    
    keep_cols = ['smiles'] + target_cols
    df_clean = df[keep_cols].copy()
    
    # Drop any row with NaN targets
    df_clean = df_clean.dropna(subset=target_cols).reset_index(drop=True)
    
    # Chemprop is better with scaled targets when there's a mix of scales.
    # dH_f_298 is in kJ/mol (or J/mol?), Cp is J/(mol K).
    # Wait, our code says H298 is in J/mol.
    # We will let Chemprop do its own target scaling (default is standard scaler).
    
    out_path = "hc_sota_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"Saved {len(df_clean)} valid entries to {out_path}")
    print("Columns:", list(df_clean.columns))

if __name__ == "__main__":
    main()
