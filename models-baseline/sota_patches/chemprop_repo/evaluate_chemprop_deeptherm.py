import pandas as pd
import numpy as np
import subprocess
import os
import json
from pathlib import Path

def main():
    csv_path = "../../deeptherm_data.csv"
    splits_path = "splits_random.json"
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    df_true = pd.read_csv(csv_path)
    
    targets = [
        'Hf_298.15K_kcal/mol',
        'S_298.15K_cal/(molK)',
        'cp_300K_cal/(molK)',
        'cp_400K_cal/(molK)',
        'cp_500K_cal/(molK)',
        'cp_600K_cal/(molK)',
        'cp_800K_cal/(molK)',
        'cp_1000K_cal/(molK)',
        'cp_1500K_cal/(molK)'
    ]
    
    results = []
    
    print("=" * 86)
    print(f" {'n':>40s} {'ΔH298':>8s} {'S298':>7s} {'Cp300':>7s} {'Cp mean':>8s} {'mean-9':>8s}")
    
    if os.path.exists("../../../models-s3/tebnet-v1/excluded_rows.txt"):
        ex = [int(t) for line in open("../../../models-s3/tebnet-v1/excluded_rows.txt") for t in line.split("#")[0].split()]
        bad = set(df_true["smiles"].iloc[ex])
    else:
        bad = set()
        
    for i, split in enumerate(splits):
        test_idx = split["test"]
        model_path = f"chemprop_dmpnn_random/replicate_{i}/model_0/best.pt"
        preds_path = f"chemprop_dmpnn_random/preds_{i}.csv"
        
        # Run predict if not already run
        if not os.path.exists(preds_path):
            cmd = [
                "chemprop", "predict",
                "--test-path", csv_path,
                "--smiles-columns", "smiles",
                "--model-path", model_path,
                "--preds-path", preds_path
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        
        df_pred = pd.read_csv(preds_path)
        
        true_test = df_true.iloc[test_idx]
        pred_test = df_pred.iloc[test_idx]
        
        keep = ~true_test["smiles"].isin(bad)
        true_test = true_test[keep]
        pred_test = pred_test[keep]
        
        # H298
        h_mae = np.abs(true_test['Hf_298.15K_kcal/mol'].values - pred_test['Hf_298.15K_kcal/mol'].values).mean()
        
        # S298
        s_mae = np.abs(true_test['S_298.15K_cal/(molK)'].values - pred_test['S_298.15K_cal/(molK)'].values).mean()
        
        # Cp
        cp_cols = [c for c in targets if c.startswith('cp_')]
        cp_maes = []
        for c in cp_cols:
            cp_maes.append(np.abs(true_test[c].values - pred_test[c].values).mean())
        
        cp300_mae = cp_maes[0] # cp_300 is the first one
        cp_mean = np.mean(cp_maes)
        
        # Mean 9
        all_maes = [h_mae, s_mae] + cp_maes
        mean9 = np.mean(all_maes)
        
        print(f" split random{i} {' '*24} {len(true_test):>3d} {h_mae:>8.3f} {s_mae:>7.3f} {cp300_mae:>7.3f} {cp_mean:>8.3f} {mean9:>8.3f}")
        
        results.append({
            "h": h_mae,
            "s": s_mae,
            "cp300": cp300_mae,
            "cp_mean": cp_mean,
            "mean9": mean9
        })
        
    print("-" * 86)
    print(f" mean {' '*35} {np.mean([r['h'] for r in results]):>8.3f} {np.mean([r['s'] for r in results]):>7.3f} {np.mean([r['cp300'] for r in results]):>7.3f} {np.mean([r['cp_mean'] for r in results]):>8.3f} {np.mean([r['mean9'] for r in results]):>8.3f}")
    print(f" std  {' '*35} {np.std([r['h'] for r in results], ddof=1):>8.3f} {np.std([r['s'] for r in results], ddof=1):>7.3f} {np.std([r['cp300'] for r in results], ddof=1):>7.3f} {np.std([r['cp_mean'] for r in results], ddof=1):>8.3f} {np.std([r['mean9'] for r in results], ddof=1):>8.3f}")
    print("=" * 86)

if __name__ == "__main__":
    main()
