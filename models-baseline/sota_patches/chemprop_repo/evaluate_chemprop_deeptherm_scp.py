import pandas as pd
import numpy as np
import subprocess
import os
import json

def eval_target(prefix, targets, csv_path, splits, out_label):
    results = []
    print("=" * 60)
    print(f" {out_label}")
    print(f" {'n':>20s} {'MAE':>8s}")
    
    df_true = pd.read_csv(csv_path)
    
    if os.path.exists("../../../models-s3/tebnet-v1/excluded_rows.txt"):
        ex = [int(t) for line in open("../../../models-s3/tebnet-v1/excluded_rows.txt") for t in line.split("#")[0].split()]
        bad = set(df_true["smiles"].iloc[ex])
    else:
        bad = set()
    
    for i, split in enumerate(splits):
        test_idx = split["test"]
        model_path = f"{prefix}/replicate_{i}/model_0/best.pt"
        preds_path = f"{prefix}/preds_{i}.csv"
        
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
        true_test = true_test[keep][targets].values
        pred_test = pred_test[keep][targets].values
        
        mae = np.abs(true_test - pred_test).mean()
        
        print(f" split random{i} {' '*4} {len(true_test):>3d} {mae:>8.3f}")
        results.append(mae)
        
    print("-" * 60)
    print(f" mean {' '*15} {np.mean(results):>8.3f}")
    print(f" std  {' '*15} {np.std(results, ddof=1):>8.3f}")
    print("=" * 60)

def main():
    csv_path = "../../deeptherm_data.csv"
    splits_path = "splits_random.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    eval_target("chemprop_dmpnn_random_s", ["S_298.15K_cal/(molK)"], csv_path, splits, "S_298")
    
    cp_targets = [f'cp_{int(t)}K_cal/(molK)' for t in [300, 400, 500, 600, 800, 1000, 1500]]
    eval_target("chemprop_dmpnn_random_cp", cp_targets, csv_path, splits, "Cp_mean (7 columns)")
    eval_target("chemprop_dmpnn_random_cp", [cp_targets[0]], csv_path, splits, "Cp_300")

if __name__ == "__main__":
    main()
