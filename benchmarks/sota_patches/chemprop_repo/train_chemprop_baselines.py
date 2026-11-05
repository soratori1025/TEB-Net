import os
import subprocess
import time
import pandas as pd
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.evaluation_metrics import calculate_mae, calculate_rmse, check_high_T_validity
import warnings
warnings.filterwarnings('ignore')

splits = [
    ('Random_Fold0', 'data/chemprop_data/stable_random_train.csv', 'data/chemprop_data/stable_random_test.csv'),
    ('Scaffold', 'data/chemprop_data/stable_scaffold_train.csv', 'data/chemprop_data/stable_scaffold_test.csv'),
    ('Size_Extrapolation', 'data/chemprop_data/stable_size_train.csv', 'data/chemprop_data/stable_size_test.csv')
]

TARGETS = ['dH_f_298', 'S_298', 'Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']
T_ARRAY = [300, 400, 500, 600, 800, 1000]

results = []
os.makedirs('data/results', exist_ok=True)
os.makedirs('sota/chemprop_repo/checkpoints_chemprop', exist_ok=True)

for split_name, train_csv, test_csv in splits:
    print(f"=== Starting Chemprop on {split_name} ===")
    
    config_path = f"sota/chemprop_repo/checkpoints_chemprop/{split_name}_config.json"
    save_dir = f"sota/chemprop_repo/checkpoints_chemprop/{split_name}"
    preds_path = f"sota/chemprop_repo/checkpoints_chemprop/{split_name}/test_preds.csv"
    
    # HPO Step (Limit to 5 iters to save time, production should be 20-30)
    print("Running Hyperopt...")
    hpo_cmd = [
        ".venv/bin/chemprop", "hyperopt",
        "--data-path", train_csv,
        "--dataset-type", "regression",
        "--epochs", "20"
    ]
    subprocess.run(hpo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Training Step
    print("Running Training...")
    start_time = time.time()
    train_cmd = [
        ".venv/bin/chemprop", "train",
        "--data-path", train_csv,
        "--task-type", "regression",
        "--output-dir", save_dir,
        "--epochs", "30"
    ]
    subprocess.run(train_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    train_time = time.time() - start_time
    
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
    
    H_true = test_df['dH_f_298'].values
    S_true = test_df['S_298'].values
    Cp_true = test_df[['Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']].values
    
    H_pred = preds_df['dH_f_298'].values
    S_pred = preds_df['S_298'].values
    Cp_pred = preds_df[['Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']].values
    
    metrics = {
        'Model': 'Chemprop',
        'Split': split_name,
        'H298_MAE': calculate_mae(H_true, H_pred),
        'H298_RMSE': calculate_rmse(H_true, H_pred),
        'S298_MAE': calculate_mae(S_true, S_pred),
        'S298_RMSE': calculate_rmse(S_true, S_pred),
        'Cp_MAE': calculate_mae(Cp_true, Cp_pred),
        'Cp_RMSE': calculate_rmse(Cp_true, Cp_pred),
        'TrainTime_s': train_time
    }
    
    high_t = check_high_T_validity(Cp_pred, T_ARRAY)
    metrics.update(high_t)
    results.append(metrics)
    
res_df = pd.DataFrame(results)
res_df.to_csv('sota/chemprop_repo/chemprop_baselines_stable_hc.csv', index=False)
print("Done! Results saved to sota/chemprop_repo/chemprop_baselines_stable_hc.csv")
print(res_df)
