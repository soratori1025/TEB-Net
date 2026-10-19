import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/home/nvsinh1/brats_segmentation/teb-net-model")
sys.path.insert(0, str(PROJECT_ROOT))

from tebnet.training.train import create_dataloaders, load_config
from tebnet.models.tebnet import TEBNet

def analyze_dataset():
    print("=== 1. Checking dataset_combined.csv distribution and splits ===")
    df = pd.read_csv(PROJECT_ROOT / "data" / "dataset_combined.csv")
    
    # H298 distribution
    h298 = df["Hf298_kJ_mol"].dropna()
    print(f"H298 distribution (kJ/mol):")
    print(f"  Mean: {h298.mean():.2f}")
    print(f"  Std:  {h298.std():.2f}")
    print(f"  Median: {h298.median():.2f}")
    
    mad = (h298 - h298.median()).abs().median()
    print(f"  MAD:  {mad:.2f}")
    print(f"  Robust std (1.4826 * MAD): {1.4826 * mad:.2f}")
    
    # Train / Val / Test distribution
    print("\nH298 Stats by split:")
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split]["Hf298_kJ_mol"].dropna()
        print(f"  {split:5s} - Mean: {s.mean():6.2f}, Std: {s.std():6.2f}, Median: {s.median():6.2f}, MAD: {(s-s.median()).abs().median():6.2f}")

    print("\n=== 2. Checking duplicate InChIKeys ===")
    train_keys = set(df[df["split"] == "train"]["inchikey"].dropna())
    test_keys = set(df[df["split"] == "test"]["inchikey"].dropna())
    overlap = train_keys.intersection(test_keys)
    print(f"  Train InChIKeys: {len(train_keys)}")
    print(f"  Test InChIKeys: {len(test_keys)}")
    print(f"  Overlap: {len(overlap)}")

def evaluate_model():
    print("\n=== 3. Evaluating Model and MAE by Bins ===")
    ckpt_path = PROJECT_ROOT / "checkpoints/tebnet-epoch=009-val/loss_total=0.0219.ckpt"
    
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    # Load normalization stats from dataset manually to match training
    import pickle
    with open(PROJECT_ROOT / "data" / "train.pkl", "rb") as f:
        train_records = pickle.load(f)
    h_train = np.array([r["dH_f_298"] for r in train_records if not np.isnan(r["dH_f_298"])])
    s_train = np.array([r["S_298"] for r in train_records if not np.isnan(r["S_298"])])
    config["normalization"] = {
        "mean_h": float(h_train.mean()),
        "std_h": float(h_train.std()),
        "mean_s": float(s_train.mean()),
        "std_s": float(s_train.std()),
    }
    
    try:
        model = TEBNet.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        model.cuda()
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    print("\n=== 4. Uncertainty Weights ===")
    if hasattr(model.criterion, 'log_var_h298'):
        print(f"  log_var_cp: {model.criterion.log_var_cp.item():.4f} -> weight: {torch.exp(-model.criterion.log_var_cp).item():.4f}")
        print(f"  log_var_h298: {model.criterion.log_var_h298.item():.4f} -> weight: {torch.exp(-model.criterion.log_var_h298).item():.4f}")
        print(f"  log_var_s298: {model.criterion.log_var_s298.item():.4f} -> weight: {torch.exp(-model.criterion.log_var_s298).item():.4f}")
    else:
        print(f"  Using fixed weights: w_cp={model.criterion.w_cp.item():.4e}, w_h={model.criterion.w_h298.item():.4e}, w_s={model.criterion.w_s298.item():.4e}")

    # Load test data
    _, _, test_loader = create_dataloaders(config)
    
    true_h298 = []
    pred_h298 = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to("cuda")
            out = model(batch)
            
            mask = ~torch.isnan(batch.dh_f_298)
            true_h298.extend(batch.dh_f_298[mask].cpu().numpy() / 1000.0) # convert to kJ/mol
            pred_h298.extend(out["dh_f_298"][mask].cpu().numpy() / 1000.0)

    true_h298 = np.array(true_h298)
    pred_h298 = np.array(pred_h298)
    
    errors = np.abs(true_h298 - pred_h298)
    
    # Baselines
    print("\n=== 5. Baselines on Test Set (kJ/mol) ===")
    mae_mean = np.mean(np.abs(true_h298 - (config["normalization"]["mean_h"]/1000.0)))
    mae_model = np.mean(errors)
    print(f"  MAE (Predicting Train Mean): {mae_mean:.2f}")
    print(f"  MAE (Model): {mae_model:.2f}")

    print("\n=== 6. H298 MAE by Bins (kJ/mol) ===")
    bins = [-np.inf, -1000, -500, -100, 100, 500, 1000, np.inf]
    labels = ["< -1000", "-1000 to -500", "-500 to -100", "-100 to 100", "100 to 500", "500 to 1000", "> 1000"]
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (true_h298 >= low) & (true_h298 < high)
        n = np.sum(mask)
        if n > 0:
            mae_bin = np.mean(errors[mask])
            print(f"  {labels[i]:15s} | N={n:4d} | MAE={mae_bin:8.2f}")
        else:
            print(f"  {labels[i]:15s} | N=   0 | MAE=     N/A")


if __name__ == "__main__":
    analyze_dataset()
    evaluate_model()
