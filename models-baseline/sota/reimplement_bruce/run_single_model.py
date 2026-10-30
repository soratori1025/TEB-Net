import argparse
import numpy as np
import time
from dataset import WudilyCHODataset
from featurizers import (
    CustomDescriptorSetExtractor,
    ECFP4Extractor,
    MordredExtractor,
    CoulombMatrixExtractor,
    SOAPMBTRRDKitExtractor,
    MoleculeGraphBuilder
)
from utils import FeatureMinMaxScaler, TargetNormalizer, evaluate_predictions
from models import get_traditional_model, ANNTrainer, GNNTrainer
def run_single_model_cv(model_name, n_splits=10, seed=42, epochs=400):
    print(f"\n==================================================")
    print(f"Running {n_splits}-Fold CV for model: {model_name}")
    print(f"==================================================")
    ds = WudilyCHODataset()
    smiles_list = ds.smiles_list
    y_raw = ds.targets['Hf_298']
    print(f"Extracting features for {model_name}...")
    start_feat = time.time()
    is_gnn = model_name.startswith('GNF-')
    is_ann = 'ANN' in model_name
    if is_gnn:
        builder = MoleculeGraphBuilder()
        raw_graphs = [builder.smiles_to_graph(smi) for smi in smiles_list]
        X = raw_graphs
    elif 'CDS' in model_name:
        extractor = CustomDescriptorSetExtractor()
        X = extractor.transform_smiles_list(smiles_list)
    elif 'ECFP' in model_name:
        extractor = ECFP4Extractor()
        X = extractor.transform_smiles_list(smiles_list)
    elif 'MRD' in model_name:
        extractor = MordredExtractor()
        X = extractor.fit_transform_smiles_list(smiles_list)
    elif 'CM' in model_name:
        extractor = CoulombMatrixExtractor()
        X = extractor.transform_smiles_list(smiles_list)
    else:
        extractor = SOAPMBTRRDKitExtractor()
        X = extractor.transform_smiles_list(smiles_list)
    feat_time = time.time() - start_feat
    print(f"Feature extraction completed in {feat_time:.2f}s.")
    splits = ds.get_kfold_splits(n_splits=n_splits, seed=seed)
    fold_metrics = []
    start_train = time.time()
    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]
        if is_gnn:
            target_scaler = TargetNormalizer(method='standard')
            y_train_norm = target_scaler.fit_transform(y_train_raw)
            import copy
            train_graphs = []
            for idx in train_idx:
                g = copy.deepcopy(X[idx])
                import torch
                g.y = torch.tensor([y_train_norm[len(train_graphs)]], dtype=torch.float)
                train_graphs.append(g)
            test_graphs = [copy.deepcopy(X[idx]) for idx in test_idx]
            trainer = GNNTrainer(model_type=model_name, epochs=epochs, batch_size=128)
            trainer.fit(train_graphs)
            train_preds_norm = trainer.predict(train_graphs)
            test_preds_norm = trainer.predict(test_graphs)
            train_preds = target_scaler.inverse_transform(train_preds_norm)
            test_preds = target_scaler.inverse_transform(test_preds_norm)
        elif is_ann:
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            feature_scaler = FeatureMinMaxScaler()
            X_train = feature_scaler.fit_transform(X_train_raw)
            X_test = feature_scaler.transform(X_test_raw)
            target_scaler = TargetNormalizer(method='standard')
            y_train_norm = target_scaler.fit_transform(y_train_raw)
            trainer = ANNTrainer(input_dim=X_train.shape[1], epochs=epochs, batch_size=128)
            trainer.fit(X_train, y_train_norm)
            train_preds_norm = trainer.predict(X_train)
            test_preds_norm = trainer.predict(X_test)
            train_preds = target_scaler.inverse_transform(train_preds_norm)
            test_preds = target_scaler.inverse_transform(test_preds_norm)
        else:
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            feature_scaler = FeatureMinMaxScaler()
            X_train = feature_scaler.fit_transform(X_train_raw)
            X_test = feature_scaler.transform(X_test_raw)
            model = get_traditional_model(model_name, random_state=seed)
            model.fit(X_train, y_train_raw)
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
        train_eval = evaluate_predictions(y_train_raw, train_preds)
        test_eval = evaluate_predictions(y_test_raw, test_preds)
        fold_metrics.append({
            'Train_MAE': train_eval['MAE'],
            'Test_MAE': test_eval['MAE'],
            'Train_RMSE': train_eval['RMSE'],
            'Test_RMSE': test_eval['RMSE'],
            'Train_R2': train_eval['R2'],
            'Test_R2': test_eval['R2']
        })
        print(f"  Fold {fold+1:02d}/{n_splits:02d} | Train MAE: {train_eval['MAE']:.2f}, Test MAE: {test_eval['MAE']:.2f} | Train RMSE: {train_eval['RMSE']:.2f}, Test RMSE: {test_eval['RMSE']:.2f} | Test R2: {test_eval['R2']:.4f}")
    train_time = time.time() - start_train
    mean_train_mae = np.mean([m['Train_MAE'] for m in fold_metrics])
    mean_test_mae = np.mean([m['Test_MAE'] for m in fold_metrics])
    mean_train_rmse = np.mean([m['Train_RMSE'] for m in fold_metrics])
    mean_test_rmse = np.mean([m['Test_RMSE'] for m in fold_metrics])
    mean_train_r2 = np.mean([m['Train_R2'] for m in fold_metrics])
    mean_test_r2 = np.mean([m['Test_R2'] for m in fold_metrics])
    print("\n--------------------------------------------------")
    print(f"FINAL CV SUMMARY ({n_splits}-Fold) for {model_name}:")
    print(f"  Train MAE  : {mean_train_mae:.2f} kcal/mol")
    print(f"  Test MAE   : {mean_test_mae:.2f} kcal/mol")
    print(f"  Train RMSE : {mean_train_rmse:.2f} kcal/mol")
    print(f"  Test RMSE  : {mean_test_rmse:.2f} kcal/mol")
    print(f"  Train R2   : {mean_train_r2:.4f}")
    print(f"  Test R2    : {mean_test_r2:.4f}")
    print(f"Total training time: {train_time:.2f}s")
    print("--------------------------------------------------\n")
    return {
        'Model': model_name,
        'Train_MAE': mean_train_mae,
        'Test_MAE': mean_test_mae,
        'Train_RMSE': mean_train_rmse,
        'Test_RMSE': mean_test_rmse,
        'Train_R2': mean_train_r2,
        'Test_R2': mean_test_r2,
        'Time_sec': train_time
    }
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a single model on WUDILY-CHO dataset.")
    parser.add_argument('--model', type=str, default='CDS-RF', help='Model name (e.g., CDS-RF, GNF-MPNN, MRD-RF, CM-SVR)')
    parser.add_argument('--folds', type=int, default=10, help='Number of CV folds (default: 10)')
    parser.add_argument('--epochs', type=int, default=400, help='Epochs for ANN/GNN models (default: 400)')
    args = parser.parse_args()
    run_single_model_cv(args.model, n_splits=args.folds, epochs=args.epochs)
