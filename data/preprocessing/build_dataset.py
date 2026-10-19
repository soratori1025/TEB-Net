#!/usr/bin/env python3
"""
Build unified thermodynamic dataset for TEB-Net v2 training.

Merges multiple data sources:
    - Burcat (cleaned + SMILES mapped)
    - QM9 + RRHO (computed thermodynamics)
    - RMG thermo libraries

Pipeline:
    1. Load all source datasets
    2. Normalize columns to common schema
    3. Assign quality weights
    4. Deduplicate across sources by InChIKey
    5. Compute Cp(T) grid from NASA-7 at standard temperatures
    6. Validate all SMILES with RDKit
    7. Scaffold split → train / val / test
    8. Export pickle files

Usage:
    python build_dataset.py \\
        --burcat burcat_with_smiles_valid.csv \\
        --qm9 qm9_thermo.csv \\
        --rmg rmg_thermo.csv \\
        --output-dir ../data
"""

import argparse
import hashlib
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from thermo_utils import DEFAULT_T_GRID, R_GAS, nasa7_cp_grid

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARNING] RDKit not available. SMILES validation will be limited.")

try:
    from sklearn.model_selection import GroupShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# CONFIGURATION
# ============================================================

# Quality weights by source
SOURCE_WEIGHTS = {
    "burcat": 5.0,   # Experimental / reviewed polynomials
    "rmg": 2.0,      # Curated from various QM methods
    "qm9": 1.0,      # B3LYP/6-31G(2df,p) — DFT level
}

# Split ratios
SPLIT_SIZES = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


# ============================================================
# DATA LOADING & NORMALIZATION
# ============================================================

def load_burcat(csv_path, T_grid):
    """Load cleaned Burcat dataset with SMILES."""
    df = pd.read_csv(csv_path)
    print(f"  Burcat: {len(df):,} records from {csv_path}")

    # Compute Cp at temperature grid from NASA coefficients
    nasa_cols_low = [f"a{i}_low" for i in range(1, 8)]
    nasa_cols_high = [f"a{i}_high" for i in range(1, 8)]

    has_nasa = (
        all(c in df.columns for c in nasa_cols_low) and
        all(c in df.columns for c in nasa_cols_high)
    )

    cp_grid_data = []
    for _, row in df.iterrows():
        if has_nasa and not any(pd.isna(row[c]) for c in nasa_cols_low):
            coeffs_low = np.array([row[c] for c in nasa_cols_low])
            coeffs_high = np.array([row[c] for c in nasa_cols_high])
            tmid = row.get("tmid_K", 1000.0)
            if pd.isna(tmid):
                tmid = 1000.0
            cp = nasa7_cp_grid(coeffs_low, coeffs_high, T_grid, T_mid=tmid)
        else:
            cp = np.full(len(T_grid), np.nan)
        cp_grid_data.append(cp)

    # Normalize schema
    result_dict = {
        "source": "burcat",
        "smiles": df.get("smiles"),
        "name": df.get("species_label", df.get("species_name_clean", "")),
        "molecular_weight": df.get("molecular_weight"),
        "Cp_298_J_mol_K": df.get("Cp_298_J_mol_K"),
        "Hf298_kJ_mol": df.get("Hf298_kJ_mol"),
        "S_298_J_mol_K": df.get("S_298_J_mol_K"),
        "weight": SOURCE_WEIGHTS["burcat"],
    }
    if "x_physic" in df.columns:
        result_dict["x_physic"] = df["x_physic"]
    result = pd.DataFrame(result_dict)

    # Add Cp grid columns
    for j, T_val in enumerate(T_grid):
        result[f"Cp_{T_val:.2f}"] = [cp[j] for cp in cp_grid_data]

    # Add NASA coefficients
    if has_nasa:
        for col in nasa_cols_low + nasa_cols_high:
            result[col] = df[col].values

    # Add InChIKey if not present
    if "inchikey" in df.columns:
        result["inchikey"] = df["inchikey"].values

    return result


def load_qm9(csv_path, T_grid):
    """Load QM9 processed dataset."""
    df = pd.read_csv(csv_path)
    print(f"  QM9: {len(df):,} records from {csv_path}")

    result_dict = {
        "source": "qm9",
        "smiles": df.get("smiles"),
        "name": df.get("gdb_idx", "").astype(str).apply(lambda x: f"qm9_{x}"),
        "molecular_weight": df.get("molecular_weight"),
        "Cp_298_J_mol_K": df.get("Cp_298_J_mol_K"),
        "Hf298_kJ_mol": df.get("Hf298_kJ_mol"),
        "S_298_J_mol_K": df.get("S_298_J_mol_K"),
        "weight": df.get("weight", SOURCE_WEIGHTS["qm9"]),
    }
    if "x_physic" in df.columns:
        result_dict["x_physic"] = df["x_physic"]
    result = pd.DataFrame(result_dict)

    # Copy Cp grid columns
    for T_val in T_grid:
        col = f"Cp_{T_val:.2f}"
        if col in df.columns:
            result[col] = df[col].values

    # Copy NASA coefficients if present
    for col in [f"a{i}_{rng}" for rng in ["low", "high"] for i in range(1, 8)]:
        if col in df.columns:
            result[col] = df[col].values

    if "inchikey" in df.columns:
        result["inchikey"] = df["inchikey"].values

    return result


def load_rmg(csv_path, T_grid):
    """Load RMG thermo dataset."""
    df = pd.read_csv(csv_path)
    print(f"  RMG: {len(df):,} records from {csv_path}")

    result_dict = {
        "source": "rmg",
        "smiles": df.get("smiles"),
        "name": df.get("label", ""),
        "molecular_weight": df.get("molecular_weight"),
        "Cp_298_J_mol_K": df.get("Cp_298_J_mol_K"),
        "Hf298_kJ_mol": df.get("Hf298_kJ_mol"),
        "S_298_J_mol_K": df.get("S_298_J_mol_K"),
        "weight": df.get("weight", SOURCE_WEIGHTS["rmg"]),
    }
    if "x_physic" in df.columns:
        result_dict["x_physic"] = df["x_physic"]
    result = pd.DataFrame(result_dict)

    # Copy Cp grid columns
    for T_val in T_grid:
        col = f"Cp_{T_val:.2f}"
        if col in df.columns:
            result[col] = df[col].values

    # Copy NASA coefficients
    for col in [f"a{i}_{rng}" for rng in ["low", "high"] for i in range(1, 8)]:
        if col in df.columns:
            result[col] = df[col].values

    if "inchikey" in df.columns:
        result["inchikey"] = df["inchikey"].values

    return result


# ============================================================
# SMILES VALIDATION & INCHIKEY COMPUTATION
# ============================================================

def validate_and_canonicalize(df):
    """
    Validate SMILES, canonicalize, compute InChIKey.
    Remove rows with invalid SMILES.
    """
    if not HAS_RDKIT:
        print("[WARNING] RDKit not available, skipping SMILES validation")
        return df

    print("\nValidating SMILES with RDKit...")

    valid_mask = []
    canonical_smiles = []
    inchikeys = []
    n_atoms_list = []

    for smiles in df["smiles"]:
        if pd.isna(smiles) or not isinstance(smiles, str):
            valid_mask.append(False)
            canonical_smiles.append(None)
            inchikeys.append(None)
            n_atoms_list.append(0)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            valid_mask.append(False)
            canonical_smiles.append(None)
            inchikeys.append(None)
            n_atoms_list.append(0)
            continue

        can_smi = Chem.MolToSmiles(mol)
        canonical_smiles.append(can_smi)

        try:
            inchi = MolToInchi(mol)
            ik = InchiToInchiKey(inchi) if inchi else None
        except Exception:
            ik = None

        inchikeys.append(ik)

        mol_h = Chem.AddHs(mol)
        n_atoms_list.append(mol_h.GetNumAtoms())

        valid_mask.append(True)

    df = df.copy()
    df["smiles"] = canonical_smiles
    df["inchikey"] = inchikeys
    df["n_atoms_total"] = n_atoms_list

    before = len(df)
    df = df[valid_mask].reset_index(drop=True)

    print(f"  Valid SMILES: {len(df):,} / {before:,}")
    print(f"  Invalid removed: {before - len(df):,}")

    return df


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_by_inchikey(df):
    """
    Remove duplicate molecules across sources.

    When the same molecule appears in multiple sources,
    keep the record with the highest quality weight.
    """
    if "inchikey" not in df.columns:
        print("[WARNING] No InChIKey column — skipping dedup")
        return df

    before = len(df)

    # Remove rows without InChIKey
    has_key = df["inchikey"].notna()
    df_with_key = df[has_key].copy()
    df_no_key = df[~has_key].copy()

    # Sort by weight (descending) to keep highest quality
    df_with_key = df_with_key.sort_values("weight", ascending=False)
    df_dedup = df_with_key.drop_duplicates(
        subset="inchikey",
        keep="first",
    )

    # Re-combine with records that had no InChIKey
    df_result = pd.concat([df_dedup, df_no_key], ignore_index=True)

    print(f"\nDeduplication by InChIKey:")
    print(f"  Before: {before:,}")
    print(f"  After: {len(df_result):,}")
    print(f"  Duplicates removed: {before - len(df_result):,}")

    # Source distribution after dedup
    source_counts = df_result["source"].value_counts()
    for src, cnt in source_counts.items():
        print(f"    {src}: {cnt:,}")

    return df_result


# ============================================================
# SCAFFOLD SPLITTING
# ============================================================

def get_murcko_scaffold(smiles):
    """Get Murcko scaffold SMILES for a molecule."""
    if not HAS_RDKIT or pd.isna(smiles):
        return "no_scaffold"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "no_scaffold"
        scaffold = GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) or "no_scaffold"
    except Exception:
        return "no_scaffold"


def scaffold_split(df, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Split dataset by Murcko scaffold.

    All molecules with the same scaffold go into the same split.
    This tests generalization to novel scaffolds.
    """
    print(f"\nScaffold splitting (train={train_frac}, val={val_frac}, test={1-train_frac-val_frac})...")

    # Compute scaffolds
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(get_murcko_scaffold)

    scaffolds = df["scaffold"].values
    unique_scaffolds = list(set(scaffolds))
    n_scaffolds = len(unique_scaffolds)
    print(f"  Unique scaffolds: {n_scaffolds:,}")

    # Assign scaffold index
    scaffold_to_idx = {s: i for i, s in enumerate(unique_scaffolds)}
    scaffold_indices = np.array([scaffold_to_idx[s] for s in scaffolds])

    # Group scaffolds by size
    scaffold_counts = Counter(scaffolds)
    sorted_scaffolds = sorted(
        unique_scaffolds,
        key=lambda s: scaffold_counts[s],
        reverse=True,
    )

    # Greedy assignment: put scaffolds into splits
    np.random.seed(seed)

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_indices = []
    val_indices = []
    test_indices = []

    # Shuffle scaffolds (but keep deterministic)
    shuffled = sorted_scaffolds.copy()
    np.random.shuffle(shuffled)

    for scaffold in shuffled:
        idxs = df.index[df["scaffold"] == scaffold].tolist()

        if len(train_indices) < n_train:
            train_indices.extend(idxs)
        elif len(val_indices) < n_val:
            val_indices.extend(idxs)
        else:
            test_indices.extend(idxs)

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # Check source distribution in each split
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        src_counts = split_df["source"].value_counts()
        parts = [f"{s}={c:,}" for s, c in src_counts.items()]
        print(f"    {name}: {', '.join(parts)}")

    return train_df, val_df, test_df


# ============================================================
# EXPORT TO PICKLE
# ============================================================

def df_to_records(df, T_grid):
    """
    Convert DataFrame to list of dicts for ThermochemDataset.

    Each record has:
        - smiles: str
        - Cp: np.ndarray of shape (n_temps,) in J/(mol·K)
        - dH_f_298: float in J/mol  (NOTE: J/mol, not kJ/mol)
        - S_298: float in J/(mol·K)
        - weight: float
        - name: str
    """
    records = []

    cp_cols = [f"Cp_{T_val:.2f}" for T_val in T_grid]

    for _, row in df.iterrows():
        smiles = row.get("smiles")
        if pd.isna(smiles):
            continue

        # Extract Cp grid
        cp_values = []
        for col in cp_cols:
            if col in row and not pd.isna(row[col]):
                cp_values.append(float(row[col]))
            else:
                cp_values.append(np.nan)

        cp_array = np.array(cp_values, dtype=np.float32)

        # Convert Hf298 from kJ/mol to J/mol for model
        hf298_kj = row.get("Hf298_kJ_mol", 0.0)
        hf298_j = float(hf298_kj) * 1000.0 if not pd.isna(hf298_kj) else 0.0

        s_298 = float(row.get("S_298_J_mol_K", 0.0))
        if pd.isna(s_298):
            s_298 = 0.0

        weight = float(row.get("weight", 1.0))
        name = str(row.get("name", smiles))

        record = {
            "smiles": smiles,
            "Cp": cp_array,
            "dH_f_298": hf298_j,
            "S_298": s_298,
            "weight": weight,
            "name": name,
        }
        if "x_physic" in row and pd.notna(row["x_physic"]):
            record["x_physic"] = row["x_physic"]
        else:
            from tebnet.data.featurizer import compute_physic_features
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    record["x_physic"] = str(compute_physic_features(mol))
            except Exception:
                pass

        records.append(record)

    return records


# ============================================================
# MAIN
# ============================================================

def build_dataset(
    burcat_path=None,
    qm9_path=None,
    rmg_path=None,
    output_dir=".",
    T_grid=None,
    seed=42,
):
    """
    Build unified dataset from all sources.
    """
    if T_grid is None:
        T_grid = DEFAULT_T_GRID

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("BUILDING UNIFIED THERMODYNAMIC DATASET")
    print("=" * 80)

    # --- Load all sources ---
    print("\nLoading data sources...")

    dfs = []

    if burcat_path and Path(burcat_path).exists():
        dfs.append(load_burcat(burcat_path, T_grid))

    if qm9_path and Path(qm9_path).exists():
        dfs.append(load_qm9(qm9_path, T_grid))

    if rmg_path and Path(rmg_path).exists():
        dfs.append(load_rmg(rmg_path, T_grid))

    if not dfs:
        print("[ERROR] No data sources provided or found!")
        return

    # --- Concatenate ---
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df):,} records")

    source_counts = df["source"].value_counts()
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt:,}")

    # --- Validate SMILES ---
    df = validate_and_canonicalize(df)

    # --- Deduplicate ---
    df = deduplicate_by_inchikey(df)

    # --- Basic quality filters ---
    print("\nApplying quality filters...")

    before = len(df)

    # Remove records with missing key properties
    required = ["smiles", "Cp_298_J_mol_K"]
    for col in required:
        if col in df.columns:
            df = df[df[col].notna()]

    # Remove obviously wrong Cp values (negative or extremely large)
    if "Cp_298_J_mol_K" in df.columns:
        cp = pd.to_numeric(df["Cp_298_J_mol_K"], errors="coerce")
        df = df[(cp > 0) & (cp < 5000)]

    # Remove extremely large/small Hf298
    if "Hf298_kJ_mol" in df.columns:
        hf = pd.to_numeric(df["Hf298_kJ_mol"], errors="coerce")
        df = df[hf.isna() | ((hf > -10000) & (hf < 10000))]

    df = df.reset_index(drop=True)

    print(f"  After quality filters: {len(df):,} (removed {before - len(df):,})")

    # --- Scaffold split ---
    train_df, val_df, test_df = scaffold_split(
        df,
        train_frac=SPLIT_SIZES["train"],
        val_frac=SPLIT_SIZES["val"],
        seed=seed,
    )

    # --- Export ---
    print("\nExporting datasets...")

    # Convert to records
    train_records = df_to_records(train_df, T_grid)
    val_records = df_to_records(val_df, T_grid)
    test_records = df_to_records(test_df, T_grid)

    # Save pickle
    for name, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        pkl_path = output_dir / f"{name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        print(f"  {pkl_path}: {len(records):,} records")

    # Save combined CSV for reference
    csv_path = output_dir / "dataset_combined.csv"

    # Add split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"  {csv_path}: {len(combined):,} records")

    # --- Final statistics ---
    print()
    print("=" * 80)
    print("FINAL DATASET STATISTICS")
    print("=" * 80)
    print(f"  Total molecules: {len(combined):,}")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    print()
    for col in ["Cp_298_J_mol_K", "Hf298_kJ_mol", "S_298_J_mol_K", "molecular_weight"]:
        if col in combined.columns:
            vals = pd.to_numeric(combined[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(
                    f"  {col:24s}: "
                    f"n={len(vals):,}, "
                    f"min={vals.min():.4g}, "
                    f"median={vals.median():.4g}, "
                    f"mean={vals.mean():.4g}, "
                    f"max={vals.max():.4g}"
                )

    print()
    print("Source distribution:")
    for split_name in ["train", "val", "test"]:
        split_data = combined[combined["split"] == split_name]
        src_counts = split_data["source"].value_counts()
        parts = [f"{s}={c:,}" for s, c in src_counts.items()]
        print(f"  {split_name:6s}: {', '.join(parts)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build unified thermodynamic dataset for TEB-Net v2."
    )
    parser.add_argument(
        "--burcat",
        default=None,
        help="Path to cleaned Burcat CSV with SMILES",
    )
    parser.add_argument(
        "--qm9",
        default=None,
        help="Path to QM9 thermo CSV",
    )
    parser.add_argument(
        "--rmg",
        default=None,
        help="Path to RMG thermo CSV",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for train/val/test pkl files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    build_dataset(
        burcat_path=args.burcat,
        qm9_path=args.qm9,
        rmg_path=args.rmg,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
