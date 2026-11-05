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
    
    # We only care about the single split generated for this FG
    test_idx = splits[0]["test"]
    model_path = f"{prefix}/model_0/best.pt"
    preds_path = f"{prefix}/preds_0.csv"
    
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
    true_test = df_true.iloc[test_idx][targets].values
    pred_test = df_pred.iloc[test_idx][targets].values
    
    mae = np.abs(true_test - pred_test).mean()
    
    print(f" split test {' '*4} {len(test_idx):>3d} {mae:>8.3f}")
    results.append(mae)
        
    print("=" * 60)

def main():
    import sys
    group = sys.argv[1]
    gname = group.replace(' ', '_')
    csv_path = "../../deeptherm_data.csv"
    splits_path = f"splits_fg_{gname}.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    eval_target(f"chemprop_fg_h_{gname}", ["Hf_298.15K_kcal/mol"], csv_path, splits, "H_298")
    eval_target(f"chemprop_fg_scp_{gname}", ["S_298.15K_cal/(molK)"], csv_path, splits, "S_298")
    
    cp_targets = [f'cp_{int(t)}K_cal/(molK)' for t in [300, 400, 500, 600, 800, 1000, 1500]]
    eval_target(f"chemprop_fg_scp_{gname}", cp_targets, csv_path, splits, "Cp_mean (7 columns)")
    eval_target(f"chemprop_fg_scp_{gname}", [cp_targets[0]], csv_path, splits, "Cp_300")

if __name__ == "__main__":
    main()
