import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import json
J_PER_KCAL = 4184.0
def evaluate_tebnet_loader(model, loader, device='cuda'):
    model.eval()
    model.to(device)
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred_dh_j = out["dh_f_298"].detach().cpu().numpy().ravel()
            true_dh_j = batch.dh_f_298.detach().cpu().numpy().ravel()
            pred_dh_kcal = pred_dh_j / J_PER_KCAL
            true_dh_kcal = true_dh_j / J_PER_KCAL
            y_pred_list.extend(pred_dh_kcal)
            y_true_list.extend(true_dh_kcal)
    return np.array(y_true_list), np.array(y_pred_list)
def run_tebnet_10fold_cv(tebnet_dir="/home/nvsinh1/brats_segmentation/teb-net-model/tebnet-v12(fix-bug)", model_name_label="TEBNet-v12 (Custom Feat)", max_epochs=200, n_splits=10, seed=42):
    for mod in list(sys.modules.keys()):
        if mod.startswith('tebnet'):
            del sys.modules[mod]
    if tebnet_dir not in sys.path:
        sys.path.insert(0, tebnet_dir)
    print("\n=========================================================================", flush=True)
    print(f"RUNNING {model_name_label} {n_splits}-FOLD CV (EPOCHS={max_epochs}, UNCERTAINTY LOSS WEIGHTING)", flush=True)
    print(f"TEBNet Directory: {tebnet_dir}", flush=True)
    print("=========================================================================\n", flush=True)
    from dataset import WudilyCHODataset
    from utils import evaluate_predictions
    from tebnet.training.train import train, load_config
    from tebnet.data.collate import ThermoCollater
    from tebnet.data.dataset import ThermochemDataset
    ds = WudilyCHODataset()
    df = ds.df
    splits = ds.get_kfold_splits(n_splits=n_splits, seed=seed)
    cp_cols = [c for c in df.columns if c.startswith('cp_')]
    all_records = []
    for idx, row in df.iterrows():
        smi = str(row['Canonical_SMILES']).strip()
        hf_kcal = float(row['Hf_298.15K_kcal/mol'])
        s_val = float(row['S_298.15K_cal/(molK)']) if 'S_298.15K_cal/(molK)' in row and pd.notna(row['S_298.15K_cal/(molK)']) else 0.0
        cp_vals = row[cp_cols].values.astype(np.float32) if cp_cols else np.zeros(18, dtype=np.float32)
        all_records.append({
            'smiles': smi,
            'dH_f_298': hf_kcal * J_PER_KCAL,
            'S_298': s_val,
            'Cp': cp_vals
        })
    safe_model_name = model_name_label.replace(' ', '_').replace('(', '').replace(')', '')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, f"{safe_model_name}_cv_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            fold_metrics = json.load(f)
        print(f"Loaded existing progress: {len(fold_metrics)} folds completed.", flush=True)
    else:
        fold_metrics = []
    start_total_time = time.time()
    config_path = os.path.join(tebnet_dir, "configs", "stage3_hc.yaml")
    base_config = load_config(config_path)
    for fold, (train_idx, test_idx) in enumerate(splits):
        if fold < len(fold_metrics):
            print(f"\n--- Fold {fold+1:02d}/{n_splits:02d} --- [ALREADY COMPLETED, SKIPPING]", flush=True)
            continue
        print(f"\n--- Fold {fold+1:02d}/{n_splits:02d} ---", flush=True)
        np.random.seed(seed + fold)
        shuffled_train_idx = np.random.permutation(train_idx)
        val_size = int(0.1 * len(shuffled_train_idx))
        val_sub_idx = shuffled_train_idx[:val_size]
        train_sub_idx = shuffled_train_idx[val_size:]
        train_records = [all_records[i] for i in train_sub_idx]
        val_records = [all_records[i] for i in val_sub_idx]
        test_records = [all_records[i] for i in test_idx]
        fold_config = base_config.copy()
        fold_config['training']['max_epochs'] = max_epochs
        fold_config['training']['batch_size'] = 64
        fold_config['training']['weight_strategy'] = 'uncertainty'
        fold_config['training']['patience'] = 30
        fold_config['training']['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
        fold_config['logging']['save_dir'] = os.path.join(os.path.dirname(__file__), 'checkpoints', safe_model_name, f"fold_{fold+1}")
        model, _ = train(fold_config, train_records=train_records, val_records=val_records, test_records=test_records)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        full_train_records = [all_records[i] for i in train_idx]
        collater = ThermoCollater()
        train_ds = ThermochemDataset(full_train_records)
        test_ds = ThermochemDataset(test_records)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=collater)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collater)
        y_train_true, y_train_pred = evaluate_tebnet_loader(model, train_loader, device=device)
        y_test_true, y_test_pred = evaluate_tebnet_loader(model, test_loader, device=device)
        train_eval = evaluate_predictions(y_train_true, y_train_pred)
        test_eval = evaluate_predictions(y_test_true, y_test_pred)
        fold_metrics.append({
            'Train_MAE': float(train_eval['MAE']),
            'Test_MAE': float(test_eval['MAE']),
            'Train_RMSE': float(train_eval['RMSE']),
            'Test_RMSE': float(test_eval['RMSE']),
            'Train_R2': float(train_eval['R2']),
            'Test_R2': float(test_eval['R2'])
        })
        with open(progress_file, 'w') as f:
            json.dump(fold_metrics, f, indent=4)
        print(f"Fold {fold+1:02d} Results | Train MAE: {train_eval['MAE']:.2f}, Test MAE: {test_eval['MAE']:.2f} | Train RMSE: {train_eval['RMSE']:.2f}, Test RMSE: {test_eval['RMSE']:.2f} | Test R2: {test_eval['R2']:.4f}", flush=True)
    if len(fold_metrics) < n_splits:
        print("Not all folds completed yet. Exiting.")
        return None
    total_time = time.time() - start_total_time
    mean_train_mae = np.mean([m['Train_MAE'] for m in fold_metrics])
    mean_test_mae = np.mean([m['Test_MAE'] for m in fold_metrics])
    mean_train_rmse = np.mean([m['Train_RMSE'] for m in fold_metrics])
    mean_test_rmse = np.mean([m['Test_RMSE'] for m in fold_metrics])
    mean_train_r2 = np.mean([m['Train_R2'] for m in fold_metrics])
    mean_test_r2 = np.mean([m['Test_R2'] for m in fold_metrics])
    print("\n=========================================================================", flush=True)
    print(f"{model_name_label} FINAL {n_splits}-FOLD CV SUMMARY:", flush=True)
    print(f"  Train MAE  : {mean_train_mae:.2f} kcal/mol", flush=True)
    print(f"  Test MAE   : {mean_test_mae:.2f} kcal/mol", flush=True)
    print(f"  Train RMSE : {mean_train_rmse:.2f} kcal/mol", flush=True)
    print(f"  Test RMSE  : {mean_test_rmse:.2f} kcal/mol", flush=True)
    print(f"  Train R2   : {mean_train_r2:.4f}", flush=True)
    print(f"  Test R2    : {mean_test_r2:.4f}", flush=True)
    print(f"Total time   : {total_time:.2f}s", flush=True)
    print("=========================================================================\n", flush=True)
    tebnet_res = {
        'Model': model_name_label,
        'Train_MAE': mean_train_mae,
        'Test_MAE': mean_test_mae,
        'Train_RMSE': mean_train_rmse,
        'Test_RMSE': mean_test_rmse,
        'Train_R2': mean_train_r2,
        'Test_R2': mean_test_r2,
        'Time_sec': total_time
    }
    csv_path = os.path.join(output_dir, 'benchmark_table_3.csv')
    md_path = os.path.join(output_dir, 'benchmark_table_3.md')
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)
        df_bench = df_bench[df_bench['Model'] != model_name_label]
        df_combined = pd.concat([df_bench, pd.DataFrame([tebnet_res])], ignore_index=True)
    else:
        df_combined = pd.DataFrame([tebnet_res])
    df_sorted = df_combined.sort_values(by='Test_MAE').reset_index(drop=True)
    df_sorted.to_csv(csv_path, index=False)
    md_content = "# Re-implemented Benchmark Results: Bruce et al. (2025) vs TEBNet\n\n"
    md_content += f"Target Property: Standard Enthalpy of Formation (\\(\\Delta H^\\circ_{{f,298}}\\)) in `kcal/mol`\n"
    md_content += f"Dataset: WUDILY-CHO (6,074 aliphatic C, H, O species)\n"
    md_content += f"Evaluation Protocol: 10-Fold Cross-Validation\n\n"
    md_content += "| Model (Feature-Method) | Train_MAE | Test_MAE | Train_RMSE | Test_RMSE | Train_R2 | Test_R2 | Training Time (s) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for idx, row in df_sorted.iterrows():
        is_tebnet = ('TEBNet' in str(row['Model']))
        bold_prefix = "**" if is_tebnet else ""
        bold_suffix = "**" if is_tebnet else ""
        md_content += f"| {bold_prefix}{row['Model']}{bold_suffix} | {row['Train_MAE']:.2f} | {bold_prefix}{row['Test_MAE']:.2f}{bold_suffix} | {row['Train_RMSE']:.2f} | {bold_prefix}{row['Test_RMSE']:.2f}{bold_suffix} | {row['Train_R2']:.4f} | {row['Test_R2']:.4f} | {row['Time_sec']:.2f} |\n"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Updated benchmark results saved to {csv_path} and {md_path}", flush=True)
    return df_sorted
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TEBNet CV")
    parser.add_argument('--tebnet_dir', type=str, required=True, help="Path to TEBNet directory")
    parser.add_argument('--model_name_label', type=str, required=True, help="Model name label for results")
    parser.add_argument('--max_epochs', type=int, default=200, help="Max epochs")
    parser.add_argument('--n_splits', type=int, default=10, help="Number of splits")
    args = parser.parse_args()
    run_tebnet_10fold_cv(
        tebnet_dir=args.tebnet_dir,
        model_name_label=args.model_name_label,
        max_epochs=args.max_epochs,
        n_splits=args.n_splits
    )
