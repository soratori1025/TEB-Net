import argparse
import numpy as np
from pathlib import Path
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
def analyze_extremes(npz_path, prop="H", top_k=5):
    z = np.load(npz_path, allow_pickle=True)
    if prop == "H":
        pred = z["test_h_pred"] * J_TO_KCAL
        true = z["test_h_true"] * J_TO_KCAL
        unit = "kcal/mol"
    elif prop == "S":
        pred = z["test_s_pred"] * J_TO_CAL
        true = z["test_s_true"] * J_TO_CAL
        unit = "cal/(mol K)"
    elif prop == "Cp":
        pred = z["test_cp_pred"] * J_TO_CAL
        true = z["test_cp_true"] * J_TO_CAL
        unit = "cal/(mol K)"
    else:
        raise ValueError(f"Unknown property: {prop}")
    if prop == "Cp":
        err = np.mean(np.abs(pred - true), axis=1)
    else:
        err = np.abs(pred - true)
    ok = np.isfinite(err)
    err_ok = err[ok]
    if "test_smiles" in z:
        smiles = np.array([str(s) for s in z["test_smiles"]])[ok]
    else:
        smiles = np.array([f"Molecule_{i}" for i in range(len(err))])[ok]
    if prop != "Cp":
        pred_ok = pred[ok]
        true_ok = true[ok]
    order = np.argsort(err_ok)
    print(f"\n{'='*80}")
    print(f" Dataset: {npz_path}")
    print(f" Property: {prop} | Unit: {unit} | Valid molecules: {len(err_ok)}")
    print(f"{'='*80}")
    print(f"\n[ TOP {top_k} BEST (Lowest MAE) ]")
    for idx in order[:top_k]:
        smi = smiles[idx]
        if prop == "Cp":
            print(f"  {smi[:50]:<50s} MAE: {err_ok[idx]:>7.3f}")
        else:
            print(f"  {smi[:50]:<50s} True: {true_ok[idx]:>8.2f} | Pred: {pred_ok[idx]:>8.2f} | MAE: {err_ok[idx]:>7.3f}")
    print(f"\n[ TOP {top_k} WORST (Highest MAE) ]")
    for idx in order[-top_k:][::-1]:
        smi = smiles[idx]
        if prop == "Cp":
            print(f"  {smi[:50]:<50s} MAE: {err_ok[idx]:>7.3f}")
        else:
            print(f"  {smi[:50]:<50s} True: {true_ok[idx]:>8.2f} | Pred: {pred_ok[idx]:>8.2f} | MAE: {err_ok[idx]:>7.3f}")
def main():
    parser = argparse.ArgumentParser(description="Find 5 molecules with highest and lowest MAE.")
    parser.add_argument("datasets", nargs="+", help="Paths to predictions.npz files")
    parser.add_argument("--prop", choices=["H", "S", "Cp"], default="H", 
                        help="Property to analyze (H, S, or Cp). Default: H")
    parser.add_argument("--k", type=int, default=5, help="Number of extremes to show. Default: 5")
    args = parser.parse_args()
    for path in args.datasets:
        if not Path(path).exists():
            print(f"\nFile not found: {path}")
            continue
        analyze_extremes(path, prop=args.prop, top_k=args.k)
if __name__ == "__main__":
    main()
