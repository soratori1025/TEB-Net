import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# Setup paths to import project modules
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "data"))

try:
    from symmetry import compute_symmetry_number
    from molecular_geometry import compute_is_linear
except ImportError:
    from data.symmetry import compute_symmetry_number
    from data.molecular_geometry import compute_is_linear


def compute_fast_symmetry(smiles: str) -> tuple[float, float, float] | None:
    """Compute (log_sigma, N, M_scaled) for SMILES string.
    
    Fast topological calculation (milliseconds).
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        mol_h = Chem.AddHs(mol)
        N = float(mol_h.GetNumAtoms())
        M = float(Descriptors.MolWt(mol))

        try:
            sigma = compute_symmetry_number(mol)
        except Exception:
            sigma = 1

        log_sigma = float(round(float(np.log(max(sigma, 1))), 4))
        M_scaled = float(round(0.01 * M, 4))
        return (log_sigma, N, M_scaled)
    except Exception:
        return None


def compute_linearity_fast(smiles: str) -> float:
    """Compute is_linear (0.0 or 1.0) using topological shortcuts first,
    falling back to 3D conformer moments for unbranched non-cyclic candidates.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return 0.0

        # Shortcut 1: <= 2 total atoms (including H) are always linear
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() <= 2:
            return 1.0

        # Shortcut 2: Cyclic molecules are never linear
        if mol.GetRingInfo().NumRings() > 0:
            return 0.0

        # Shortcut 3: Molecules with heavy-atom degree > 2 (branching) are never linear
        if any(atom.GetDegree() > 2 for atom in mol.GetAtoms()):
            return 0.0

        # Slow path: 3D conformer embedding for unbranched non-cyclic species >= 3 atoms
        try:
            return float(compute_is_linear(mol))
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def parse_x_physic(val) -> list[float] | None:
    """Safely parse x_physic JSON string or return None."""
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return None
    try:
        data = json.loads(val)
        if isinstance(data, list) and len(data) == 4:
            return [float(x) for x in data]
    except Exception:
        pass
    return None


def process_dataset_symmetry(csv_path: Path, smiles_col: str = "smiles", n_workers: int = 8) -> None:
    """Phase 1: Compute symmetry number, atom count N, and molecular weight M.
    Save results immediately to disk.
    """
    print(f"\n==========================================")
    print(f"[PHASE 1: SYMMETRY & FAST FEATURES] {csv_path.name}")
    print(f"==========================================")

    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df):,}")

    if smiles_col not in df.columns:
        print(f"[ERROR] Column '{smiles_col}' not found in {csv_path}")
        return

    smiles_list = df[smiles_col].tolist()
    existing_x_physic = df["x_physic"].tolist() if "x_physic" in df.columns else [None] * len(df)

    start_time = time.time()
    chunksize = max(1, min(500, len(smiles_list) // (n_workers * 4)))

    with mp.Pool(processes=n_workers) as pool:
        fast_results = list(
            tqdm(
                pool.imap(compute_fast_symmetry, smiles_list, chunksize=chunksize),
                total=len(smiles_list),
                desc=f"Computing Symmetry/N/M for {csv_path.name}",
            )
        )

    updated_x_physic = []
    n_updated = 0
    for idx, fast_res in enumerate(fast_results):
        if fast_res is None:
            updated_x_physic.append(existing_x_physic[idx])
            continue

        log_sigma, N, M_scaled = fast_res
        parsed = parse_x_physic(existing_x_physic[idx])
        is_linear = parsed[2] if parsed is not None else 0.0

        feat_vector = [log_sigma, N, is_linear, M_scaled]
        updated_x_physic.append(json.dumps(feat_vector))
        n_updated += 1

    df["x_physic"] = updated_x_physic
    elapsed = time.time() - start_time
    print(f"Phase 1 finished in {elapsed:.2f}s ({elapsed/60:.2f} min). Updated {n_updated}/{len(df)} records.")

    print(f"Saving dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved symmetry features for {csv_path.name}!")


def process_dataset_linearity(csv_path: Path, smiles_col: str = "smiles", n_workers: int = 8, save_interval: int = 500) -> None:
    """Phase 2: Compute molecular linearity (is_linear) with shortcuts & incremental saving."""
    print(f"\n==========================================")
    print(f"[PHASE 2: MOLECULAR LINEARITY] {csv_path.name}")
    print(f"==========================================")

    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df):,}")

    if smiles_col not in df.columns:
        print(f"[ERROR] Column '{smiles_col}' not found in {csv_path}")
        return

    if "x_physic" not in df.columns:
        print(f"[WARNING] Column 'x_physic' not found. Run Phase 1 (--mode symmetry) first!")
        df["x_physic"] = None

    smiles_list = df[smiles_col].tolist()
    x_physic_list = df["x_physic"].tolist()

    start_time = time.time()

    items_to_compute = list(enumerate(smiles_list))
    if save_interval is None or save_interval <= 0:
        save_interval = max(500, min(10000, len(items_to_compute) // 20))

    print(f"Processing linearity for {len(items_to_compute):,} records (save every {save_interval} items)...")

    chunksize = max(1, min(200, len(items_to_compute) // (n_workers * 4)))

    with mp.Pool(processes=n_workers) as pool:
        for i in range(0, len(items_to_compute), save_interval):
            batch_items = items_to_compute[i : i + save_interval]
            batch_smiles = [item[1] for item in batch_items]

            batch_results = list(
                tqdm(
                    pool.imap(compute_linearity_fast, batch_smiles, chunksize=chunksize),
                    total=len(batch_smiles),
                    desc=f"Linearity batch {i // save_interval + 1}/{(len(items_to_compute) + save_interval - 1) // save_interval}",
                )
            )

            for (orig_idx, _), is_lin in zip(batch_items, batch_results):
                parsed = parse_x_physic(x_physic_list[orig_idx])
                if parsed is not None:
                    parsed[2] = float(is_lin)
                    x_physic_list[orig_idx] = json.dumps(parsed)
                else:
                    x_physic_list[orig_idx] = json.dumps([0.0, 0.0, float(is_lin), 0.0])

            df["x_physic"] = x_physic_list
            df.to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    print(f"Phase 2 finished for {csv_path.name} in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    print(f"[SUCCESS] Linearity saved for {csv_path.name}!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute physics features (symmetry, linearity, N, M) for thermodynamic datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["symmetry", "linear", "all"],
        default="all",
        help="Calculation mode: 'symmetry' (fast phase 1), 'linear' (linearity phase 2), or 'all' (both). Default: 'all'",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Number of CPU multiprocessing workers. Default: max(1, CPU_COUNT - 2)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save CSV to disk after every N items in linearity phase. Default: 500",
    )
    parser.add_argument(
        "--dataset",
        choices=["burcat", "rmg", "qm9", "all"],
        default="all",
        help="Dataset to process: 'burcat', 'rmg', 'qm9', or 'all'. Default: 'all'",
    )

    args = parser.parse_args()
    data_dir = Path(__file__).resolve().parent

    dataset_map = {
        "burcat": data_dir / "burcat_with_smiles_valid.csv",
        "rmg": data_dir / "rmg_thermo.csv",
        "qm9": data_dir / "qm9_thermo.csv",
    }

    if args.dataset == "all":
        target_files = list(dataset_map.values())
    else:
        target_files = [dataset_map[args.dataset]]

    print(f"Execution Mode: {args.mode.upper()}")
    print(f"CPU Workers: {args.n_workers}")
    print(f"Save Interval: {args.save_interval}")

    # Phase 1: Symmetry & Fast Features
    if args.mode in ["symmetry", "all"]:
        for ds in target_files:
            process_dataset_symmetry(ds, n_workers=args.n_workers)

    # Phase 2: Linearity
    if args.mode in ["linear", "all"]:
        for ds in target_files:
            process_dataset_linearity(ds, n_workers=args.n_workers, save_interval=args.save_interval)


if __name__ == "__main__":
    main()
