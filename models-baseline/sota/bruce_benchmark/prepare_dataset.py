"""
Prepare the WUDILY-CHO dataset from Bruce et al. (Fuel 384, 133999, 2025)
for benchmarking TEB-Net v14.2.

Protocol (faithfully matching the paper):
  1. Load mmc5 (Supplementary Material 5) — 6,074 C/H/O species
  2. Canonicalize SMILES (RDKit)
  3. Remove duplicates by canonical SMILES
  4. Remove charged species
  5. Verify Kekulization
  6. 80/20 random train/test split (paper's protocol)
  7. Save processed CSV + split index files

Units in the saved CSV follow the paper's convention:
  - ΔHf(298.15 K): kcal/mol
  - S(298.15 K):    cal/(mol·K)
  - Cp(T):          cal/(mol·K)

For TEB-Net internal use, convert to SI at load time.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi, MolToInchi, InchiToInchiKey
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")
SEEDS = [42, 123, 2024, 8888, 9999]
TRAIN_FRAC = 0.80
SMILES_COL = "Canonical_SMILES"
HF_COL = "Hf_298.15K_kcal/mol"
S_COL = "S_298.15K_cal/(molK)"
CP_COLS = [
    "cp_300K_cal/(molK)",
    "cp_400K_cal/(molK)",
    "cp_500K_cal/(molK)",
    "cp_600K_cal/(molK)",
    "cp_800K_cal/(molK)",
    "cp_1000K_cal/(molK)",
    "cp_1500K_cal/(molK)",
]
T_GRID = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]
def load_mmc5(excel_path: str) -> pd.DataFrame:
    """Load mmc5 Excel file (Supplementary Material 5)."""
    logger.info(f"Loading dataset from {excel_path}")
    df = pd.read_excel(excel_path, sheet_name="therm", engine="openpyxl")
    logger.info(f"  Raw rows: {len(df)}")
    return df
def canonicalize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize SMILES, remove invalid/charged/duplicate species."""
    valid_rows = []
    seen_inchikeys = set()
    n_invalid = 0
    n_charged = 0
    n_dup = 0
    n_no_kekulize = 0
    for idx, row in df.iterrows():
        smiles = str(row[SMILES_COL]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_invalid += 1
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=False)
        except Exception:
            n_no_kekulize += 1
            continue
        charge = Chem.GetFormalCharge(mol)
        if charge != 0:
            n_charged += 1
            continue
        canon_smi = Chem.MolToSmiles(mol, canonical=True)
        inchi = MolToInchi(mol)
        if inchi is not None:
            inchikey = InchiToInchiKey(inchi)
        else:
            inchikey = canon_smi
        if inchikey in seen_inchikeys:
            n_dup += 1
            continue
        seen_inchikeys.add(inchikey)
        clean_row = {
            "smiles": canon_smi,
            "Hf_298_kcal_mol": float(row[HF_COL]),
            "S_298_cal_molK": float(row[S_COL]),
            "inchikey": inchikey,
        }
        for cp_col, T in zip(CP_COLS, T_GRID):
            clean_row[f"Cp_{T:.0f}_cal_molK"] = float(row[cp_col])
        clean_row["Molecular_Formula"] = str(row.get("Molecular_Formula", ""))
        clean_row["Molecular_Weight"] = float(row.get("Molecular_Weight", 0))
        clean_row["source"] = str(row.get("Source or Reference", ""))
        clean_row["n_C"] = int(row.get("C", 0)) if pd.notna(row.get("C")) else 0
        clean_row["n_H"] = int(row.get("H", 0)) if pd.notna(row.get("H")) else 0
        clean_row["n_O"] = int(row.get("O", 0)) if pd.notna(row.get("O")) else 0
        valid_rows.append(clean_row)
    logger.info(
        f"  Cleaning: {n_invalid} invalid SMILES, {n_no_kekulize} failed Kekulize, "
        f"{n_charged} charged, {n_dup} duplicates removed"
    )
    logger.info(f"  Clean dataset: {len(valid_rows)} molecules")
    return pd.DataFrame(valid_rows)
def generate_splits(n_total: int, seeds: list[int], train_frac: float, out_dir: str):
    """Generate 80/20 random train/test splits for multiple seeds.

    Paper protocol: simple random split (no scaffold/stratified).
    Validation set: carved from training set (10% of train pool) for early stopping.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_indices = np.arange(n_total)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(all_indices)
        n_train_pool = int(train_frac * n_total)
        train_pool = perm[:n_train_pool].tolist()
        test_idx = perm[n_train_pool:].tolist()
        n_val = int(0.1 * len(train_pool))
        rng_val = np.random.default_rng(seed + 10007)
        pool_perm = rng_val.permutation(train_pool)
        val_idx = pool_perm[:n_val].tolist()
        train_idx = pool_perm[n_val:].tolist()
        split_info = {
            "seed": seed,
            "train_frac": train_frac,
            "n_total": n_total,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }
        split_file = os.path.join(out_dir, f"split_seed_{seed}.json")
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        logger.info(
            f"  Seed {seed}: {len(train_idx)} train / {len(val_idx)} val / "
            f"{len(test_idx)} test"
        )
def main():
    parser = argparse.ArgumentParser(description="Prepare WUDILY-CHO dataset (Bruce et al. 2025)")
    parser.add_argument(
        "--excel",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "Supplementary-data-ML"
            / "1-s2.0-S0016236124031491-mmc5.xlsx"
        ),
        help="Path to mmc5 Excel file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    df_raw = load_mmc5(args.excel)
    df_clean = canonicalize_and_clean(df_raw)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "wudily_cho_processed.csv")
    df_clean.to_csv(csv_path, index=False)
    logger.info(f"Saved processed dataset: {csv_path}")
    splits_dir = os.path.join(args.out_dir, "splits")
    logger.info(f"Generating {len(SEEDS)} random splits (80/20)...")
    generate_splits(len(df_clean), SEEDS, TRAIN_FRAC, splits_dir)
    print("\n" + "=" * 60)
    print(" WUDILY-CHO Dataset Preparation Summary")
    print("=" * 60)
    print(f"  Source: {args.excel}")
    print(f"  Molecules: {len(df_clean)}")
    print(f"  Targets: ΔHf(298K), S(298K), Cp at {len(T_GRID)} temperatures")
    print(f"  ΔHf range: [{df_clean['Hf_298_kcal_mol'].min():.2f}, {df_clean['Hf_298_kcal_mol'].max():.2f}] kcal/mol")
    print(f"  S298 range: [{df_clean['S_298_cal_molK'].min():.2f}, {df_clean['S_298_cal_molK'].max():.2f}] cal/(mol·K)")
    print(f"  Splits: {len(SEEDS)} seeds, 80/20 train/test")
    print(f"  Output: {csv_path}")
    print(f"  Splits: {splits_dir}/")
    print("=" * 60)
if __name__ == "__main__":
    main()
