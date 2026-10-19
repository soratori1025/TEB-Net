import os
import subprocess
import time
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.evaluation_metrics import calculate_mae, calculate_rmse, check_high_T_validity
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    print("=== Testing Chemprop on Goldsmith (OOD) ===")
    
    test_csv = "data/goldsmith_data/goldsmith_deduped.csv"
    save_dir = "sota/chemprop_repo/checkpoints_chemprop/Scaffold"
    preds_path = "sota/chemprop_repo/checkpoints_chemprop/Scaffold/goldsmith_preds.csv"
    
    # Predicting Step
    print("Running Predictions...")
    pred_cmd = [
        ".venv/bin/chemprop", "predict",
        "--test-path", test_csv,
        "--model-path", f"{save_dir}/model_0/best.pt",
        "--preds-path", preds_path
    ]
    subprocess.run(pred_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Evaluation
    test_df = pd.read_csv(test_csv)
    preds_df = pd.read_csv(preds_path)
    
    TARGETS = ['dH_f_298', 'S_298', 'Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']
    T_ARRAY = [300, 400, 500, 600, 800, 1000]
    
    H_true = test_df['H298_J_mol'].values
    S_true = test_df['S298_J_mol_K'].values
    Cp_true = test_df[['Cp300_J_mol_K', 'Cp400_J_mol_K', 'Cp500_J_mol_K', 'Cp600_J_mol_K', 'Cp800_J_mol_K', 'Cp1000_J_mol_K']].values
    
    H_pred = preds_df['dH_f_298'].values
    S_pred = preds_df['S_298'].values
    Cp_pred = preds_df[['Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']].values
    
    metrics = {
        'Model': 'Chemprop',
        'Test_Set': 'Goldsmith (OOD)',
        'H298_MAE': calculate_mae(H_true, H_pred),
        'H298_RMSE': calculate_rmse(H_true, H_pred),
        'S298_MAE': calculate_mae(S_true, S_pred),
        'S298_RMSE': calculate_rmse(S_true, S_pred),
        'Cp_MAE': calculate_mae(Cp_true, Cp_pred),
        'Cp_RMSE': calculate_rmse(Cp_true, Cp_pred),
    }
    
    high_t = check_high_T_validity(Cp_pred, T_ARRAY)
    metrics.update(high_t)
    
    res_df = pd.DataFrame([metrics])
    res_df.to_csv('sota/chemprop_repo/chemprop_goldsmith_results.csv', index=False)
    print("Done! Results saved to sota/chemprop_repo/chemprop_goldsmith_results.csv")
    print(res_df)

if __name__ == '__main__':
    main()
