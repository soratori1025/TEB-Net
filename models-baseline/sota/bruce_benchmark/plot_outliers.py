"""
Plot the molecular outliers for TEB-Net v14.2 predictions on the WUDILY-CHO dataset.
Finds the molecules with the highest MAE for Enthalpy (ΔHf298) and generates an image grid.
"""
import argparse
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

J_TO_KCAL = 1.0 / 4184.0

def get_ensemble_predictions(results_dir: Path):
    """Aggregate predictions across all available seeds."""
    all_seeds_preds = []
    
    # Load all available prediction files
    for pred_file in sorted(results_dir.glob("seed*/init*/predictions.npz")):
        data = np.load(pred_file, allow_pickle=True)
        all_seeds_preds.append({
            "h_pred": data["test_h_pred"],  # Already in kcal/mol
            "h_true": data["test_h_true"],
            "smiles": data["test_smiles"]
        })
        
    if not all_seeds_preds:
        raise ValueError(f"No prediction files found in {results_dir}")
        
    print(f"Loaded predictions from {len(all_seeds_preds)} seeds.")
    
    # Use the first seed's SMILES and truth values as reference
    ref_smiles = all_seeds_preds[0]["smiles"]
    ref_true = all_seeds_preds[0]["h_true"]
    
    # Ensure all seeds have the same test set (should be true if split_seed is fixed,
    # but in our benchmark we used random splits per seed. If splits vary per seed, 
    # we should just look at the outliers from the first completed seed to be safe).
    # Since we used split_seed=seed, the test sets are different!
    # So we'll just use the first seed's test set for outlier detection.
    print(f"Using predictions from the first seed for outlier detection.")
    return all_seeds_preds[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/outliers_Hf298.png")
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    preds = get_ensemble_predictions(results_path)
    
    h_pred = preds["h_pred"]
    h_true = preds["h_true"]
    smiles = preds["smiles"]
    
    # Compute absolute errors
    errors = np.abs(h_pred - h_true)
    
    # Sort indices by descending error
    sorted_idx = np.argsort(errors)[::-1]
    
    outlier_mols = []
    legends = []
    
    print(f"\nTop {args.top_k} Outliers for ΔHf298 (kcal/mol):")
    print(f"{'SMILES':<60} {'True':>10} {'Pred':>10} {'Error':>10}")
    print("-" * 93)
    
    for i in range(args.top_k):
        idx = sorted_idx[i]
        smi = smiles[idx]
        err = errors[idx]
        true_val = h_true[idx]
        pred_val = h_pred[idx]
        
        print(f"{smi:<60} {true_val:>10.2f} {pred_val:>10.2f} {err:>10.2f}")
        
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            outlier_mols.append(mol)
            # Create a clean legend
            legend = f"Error: {err:.1f} kcal/mol\n(True: {true_val:.1f}, Pred: {pred_val:.1f})"
            legends.append(legend)
    
    # Generate image
    img = Draw.MolsToGridImage(
        outlier_mols,
        molsPerRow=4,
        subImgSize=(400, 400),
        legends=legends,
        useSVG=False,
        returnPNG=False
    )
    
    # Save the image
    img.save(args.output)
    print(f"\nSaved outlier visualization to: {args.output}")


if __name__ == "__main__":
    main()
