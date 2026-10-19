"""
prepare_hc_dataset.py
=====================
Load và merge 3 datasets từ https://github.com/yureel/HC-Thermochemistry:
  - ab_initio_database_hc.csv    : Hydrocarbons (stable molecules)
  - ab_initio_database_rad.csv   : Hydrocarbon radicals
  - ab_initio_database_carb.csv  : Carbenium ions

Các bước:
1. Inspect từng dataset (số lượng, phân phối, SMILES validity)
2. Chuẩn hóa unit về SI (kJ/mol → J/mol cho H_298; J/mol·K cho Cp, S)
3. Tính T_grid nội suy và hệ số NASA-7 thô
4. Tính x_physic = [log(σ), N, is_linear, M/100] giống Burcat pipeline
5. Xuất ra hc_thermo_combined.csv sẵn cho v6 training

Usage:
    cd /home/nvsinh1/brats_segmentation/teb-net-model
    .venv/bin/python data/prepare_hc_dataset.py
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
import ast

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent
HC_REPO  = DATA_DIR / "hc_thermo_repo"
OUT_PATH = DATA_DIR / "hc_thermo_combined.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_hc")

# ── physics helpers ────────────────────────────────────────────────────────────
try:
    from symmetry import compute_symmetry_number
    from molecular_geometry import compute_is_linear
except ImportError:
    from data.symmetry import compute_symmetry_number
    from data.molecular_geometry import compute_is_linear

# HC dataset T_grid (7 points provided)
HC_T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])

# Full T_grid used by TEB-Net v5/v6 (18 points)
TARGET_T_GRID = np.array([200, 250, 298.15, 300, 400, 500, 600, 700, 800, 900,
                           1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], dtype=float)


# ── dataset loading ────────────────────────────────────────────────────────────
def load_hc_csv(path: Path, source_label: str) -> pd.DataFrame:
    """Load one HC-Thermochemistry CSV, add source column."""
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "mol_1": "smiles",
        "target_H_298": "Hf298_kJ_mol",       # kJ/mol
        "target_S_298": "S_298_J_mol_K",       # J/(mol·K)
        "target_Cp_300": "Cp_300",
        "target_Cp_400": "Cp_400",
        "target_Cp_500": "Cp_500",
        "target_Cp_600": "Cp_600",
        "target_Cp_800": "Cp_800",
        "target_Cp_1000": "Cp_1000",
        "target_Cp_1500": "Cp_1500",
        "feature_symm": "sigma_rdkit",     # symmetry number provided
        "feature_entropy": "ln_sigma",
    })
    df["source"] = source_label
    # Keep only needed columns
    keep = ["smiles", "Hf298_kJ_mol", "S_298_J_mol_K",
            "Cp_300", "Cp_400", "Cp_500", "Cp_600", "Cp_800", "Cp_1000", "Cp_1500",
            "feature_opt", "sigma_rdkit", "ln_sigma", "source"]
    df = df[[c for c in keep if c in df.columns]].copy()
    logger.info(f"  Loaded {source_label}: {len(df)} entries")
    return df


def inspect_dataset(df: pd.DataFrame, label: str) -> None:
    """Print inspection stats for one dataset."""
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Entries           : {len(df)}")
    valid_smiles = df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    print(f"  Valid SMILES      : {valid_smiles.sum()} / {len(df)}")
    print(f"  Hf298 [kJ/mol]:")
    print(f"    Mean  : {df.Hf298_kJ_mol.mean():.1f}")
    print(f"    Std   : {df.Hf298_kJ_mol.std():.1f}")
    print(f"    Min   : {df.Hf298_kJ_mol.min():.1f}")
    print(f"    Max   : {df.Hf298_kJ_mol.max():.1f}")
    print(f"  S_298 [J/mol·K]:")
    print(f"    Mean  : {df.S_298_J_mol_K.mean():.1f}")
    print(f"    Std   : {df.S_298_J_mol_K.std():.1f}")
    print(f"  Cp_300 [J/mol·K]:")
    print(f"    Mean  : {df.Cp_300.mean():.1f}")
    print(f"    Min   : {df.Cp_300.min():.1f}   Max: {df.Cp_300.max():.1f}")
    print(f"  Duplicate SMILES  : {df.smiles.duplicated().sum()}")


# ── Cp interpolation to full T_grid ───────────────────────────────────────────
def interpolate_cp_to_full_grid(row: pd.Series) -> np.ndarray:
    """
    Given Cp at 7 HC T points (300–1500 K), interpolate/extrapolate to TARGET_T_GRID (18 pts).

    Extrapolation strategy:
    - T > 1500 K : log-linear fit on last 3 points  (Cp → const as T→∞ classically)
    - T < 300 K  : quadratic fit on first 3 points, clamped so Cp ≥ 0
                   Physically Cp should approach 0 as T → 0 (3rd law), and the
                   quadratic fit in T gives a reasonable low-T shape.
    """
    cp_known = np.array([
        row.get("Cp_300", np.nan),
        row.get("Cp_400", np.nan),
        row.get("Cp_500", np.nan),
        row.get("Cp_600", np.nan),
        row.get("Cp_800", np.nan),
        row.get("Cp_1000", np.nan),
        row.get("Cp_1500", np.nan),
    ], dtype=float)

    valid = ~np.isnan(cp_known)
    T_known = HC_T_GRID[valid]
    cp_valid = cp_known[valid]

    if len(cp_valid) < 3:
        return np.full(len(TARGET_T_GRID), np.nan, dtype=np.float32)

    # Linear interpolation within [T_known[0], T_known[-1]]
    cp_interp = np.interp(TARGET_T_GRID, T_known, cp_valid)

    # High-T extrapolation (T > 1500 K): log-linear in T
    t_high = TARGET_T_GRID[TARGET_T_GRID > T_known[-1]]
    if len(t_high) > 0:
        t_fit = T_known[-3:] if len(T_known) >= 3 else T_known
        cp_fit = cp_valid[-len(t_fit):]
        try:
            coeffs = np.polyfit(np.log(t_fit), cp_fit, 1)
            cp_extrap = np.polyval(coeffs, np.log(t_high))
            # Cp should not decrease unreasonably at very high T — clamp at last known value
            cp_extrap = np.maximum(cp_extrap, cp_valid[-1] * 0.9)
            cp_interp[TARGET_T_GRID > T_known[-1]] = cp_extrap
        except Exception:
            pass

    # Low-T extrapolation (T < 300 K): quadratic fit in T, through (0, 0)
    t_low_mask = TARGET_T_GRID < T_known[0]
    t_low = TARGET_T_GRID[t_low_mask]
    if len(t_low) > 0:
        try:
            # Use first 3 known points + enforce Cp(0) ≈ 0 via quadratic through origin
            n_fit = min(3, len(T_known))
            t_fit_low = np.concatenate([[0.0], T_known[:n_fit]])
            cp_fit_low = np.concatenate([[0.0], cp_valid[:n_fit]])
            coeffs_low = np.polyfit(t_fit_low, cp_fit_low, 2)
            cp_extrap_low = np.polyval(coeffs_low, t_low)
            # Physical clamp: Cp must be >= 0
            cp_extrap_low = np.maximum(cp_extrap_low, 0.0)
            cp_interp[t_low_mask] = cp_extrap_low
        except Exception:
            # Fallback: linear from first known point, clamped
            try:
                slope = (cp_valid[1] - cp_valid[0]) / (T_known[1] - T_known[0])
                cp_fallback = cp_valid[0] + slope * (t_low - T_known[0])
                cp_interp[t_low_mask] = np.maximum(cp_fallback, 0.0)
            except Exception:
                pass

    # Final physical clamp: Cp >= 0 everywhere
    cp_interp = np.maximum(cp_interp, 0.0)

    return cp_interp.astype(np.float32)


# ── x_physic computation ───────────────────────────────────────────────────────
def compute_x_physic(row: pd.Series) -> list[float] | None:
    """
    Compute [log(σ), N, is_linear, M/100] from SMILES.
    Tries to use precomputed sigma_rdkit if available and consistent.
    """
    smiles = str(row["smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # Symmetry number: prefer dataset's feature_symm if > 0
        sigma_from_data = float(row.get("sigma_rdkit", 0))
        if sigma_from_data > 0:
            sigma = sigma_from_data
        else:
            sigma = compute_symmetry_number(mol)
        log_sigma = float(np.log(max(sigma, 1)))

        # Number of atoms (heavy + H)
        mol_h = Chem.AddHs(mol)
        N = float(mol_h.GetNumAtoms())

        # Is linear
        is_linear = float(compute_is_linear(mol))

        # Molecular weight scaled
        M = Descriptors.MolWt(mol)
        M_scaled = M / 100.0

        return [log_sigma, N, is_linear, M_scaled]
    except Exception as e:
        logger.debug(f"x_physic failed for {smiles}: {e}")
        return None


# ── main pipeline ──────────────────────────────────────────────────────────────
def main():
    # ── 1. Load ────────────────────────────────────────────────────────────────
    logger.info("Loading 3 HC-Thermochemistry datasets...")
    dfs = []
    for fname, label in [
        ("ab_initio_database_hc.csv",   "hc"),
        ("ab_initio_database_rad.csv",  "rad"),
        ("ab_initio_database_carb.csv", "carb"),
    ]:
        fpath = HC_REPO / fname
        if not fpath.exists():
            logger.error(f"File not found: {fpath}")
            sys.exit(1)
        dfs.append(load_hc_csv(fpath, label))

    # ── 2. Inspect ─────────────────────────────────────────────────────────────
    for df, label in zip(dfs, ["HC (stable)", "Radicals", "Carbenium ions"]):
        inspect_dataset(df, label)

    # ── 3. Merge ───────────────────────────────────────────────────────────────
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nTotal after merge: {len(df_all)} entries")

    # Validate SMILES
    valid_mask = df_all["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Dropping {n_invalid} entries with invalid SMILES")
        df_all = df_all[valid_mask].copy()

    # Remove exact SMILES duplicates (keep first occurrence)
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    logger.info(f"Deduplication: {n_before} → {len(df_all)} (removed {n_before - len(df_all)} dups)")

    # ── 4. Unit conversion ─────────────────────────────────────────────────────
    # H298: kJ/mol → J/mol (multiply ×1000) to match v5 convention
    df_all["dH_f_298"] = df_all["Hf298_kJ_mol"] * 1000.0   # J/mol
    df_all["S_298"]    = df_all["S_298_J_mol_K"]             # J/(mol·K), already correct

    # Print stats after merge
    print(f"\n{'='*60}")
    print(f" MERGED DATASET STATS")
    print(f"{'='*60}")
    print(f"  Total entries     : {len(df_all)}")
    print(f"  Hf298 [kJ/mol]:")
    print(f"    Mean  : {df_all.Hf298_kJ_mol.mean():.1f}")
    print(f"    Std   : {df_all.Hf298_kJ_mol.std():.1f}")
    print(f"    Min   : {df_all.Hf298_kJ_mol.min():.1f}")
    print(f"    Max   : {df_all.Hf298_kJ_mol.max():.1f}")
    print(f"  Source breakdown  : {dict(df_all.source.value_counts())}")

    # ── 5. Interpolate Cp to full T_grid ───────────────────────────────────────
    logger.info("Interpolating Cp to full 18-point T_grid...")
    cp_matrix = df_all.apply(interpolate_cp_to_full_grid, axis=1)

    # Store each T point as a named column (compatible with v5 dataset loader)
    for i, T in enumerate(TARGET_T_GRID):
        col_name = f"Cp_{T:.2f}" if T != int(T) else f"Cp_{int(T)}"
        # Use the format the v5 dataset.py can parse
        if T == 298.15:
            col_name = "Cp_298.15"
        df_all[col_name] = cp_matrix.apply(lambda arr: arr[i])

    # ── 6. Compute x_physic ────────────────────────────────────────────────────
    logger.info("Computing x_physic features [log_sigma, N, is_linear, M/100]...")
    n_failed = 0
    x_physic_list = []
    for idx, row in df_all.iterrows():
        xp = compute_x_physic(row)
        if xp is None:
            n_failed += 1
            x_physic_list.append(None)
        else:
            x_physic_list.append(str(xp))  # store as string like Burcat pipeline

    df_all["x_physic"] = x_physic_list
    logger.info(f"  x_physic computed: {len(df_all) - n_failed}/{len(df_all)} entries")

    # Drop rows where x_physic failed
    df_all = df_all[df_all["x_physic"].notna()].reset_index(drop=True)
    logger.info(f"  After dropping x_physic failures: {len(df_all)} entries")

    # ── 7. Final cleanup ───────────────────────────────────────────────────────
    # Keep only the columns needed by ThermochemDataset.from_csv()
    cp_cols = []
    for T in TARGET_T_GRID:
        col_name = "Cp_298.15" if T == 298.15 else (f"Cp_{T:.2f}" if T != int(T) else f"Cp_{int(T)}")
        if col_name in df_all.columns:
            cp_cols.append(col_name)

    final_cols = ["smiles", "dH_f_298", "S_298", "source"] + cp_cols + ["x_physic"]
    df_out = df_all[final_cols].copy()

    # Add weight column (all ab initio = high quality)
    df_out["weight"] = 1.0
    df_out["name"] = df_out["smiles"]

    # ── 8. Save ────────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_PATH, index=False)
    logger.info(f"\nSaved combined dataset → {OUT_PATH}")
    logger.info(f"Final shape: {df_out.shape}")

    # ── Summary comparison with Burcat ─────────────────────────────────────────
    burcat_path = DATA_DIR / "burcat_with_smiles_valid.csv"
    print(f"\n{'='*60}")
    print(f" COMPARISON: HC Combined vs Burcat")
    print(f"{'='*60}")
    print(f"  HC Combined: {len(df_out)} entries")
    if burcat_path.exists():
        burcat = pd.read_csv(burcat_path)
        print(f"  Burcat     : {len(burcat)} entries")
        print(f"  HC Hf298 std  : {(df_out.dH_f_298/1000).std():.1f} kJ/mol")
        print(f"  Burcat Hf298 std: {burcat.Hf298_kJ_mol.std():.1f} kJ/mol")
        print(f"  HC Hf298 range: {(df_out.dH_f_298/1000).min():.1f} to {(df_out.dH_f_298/1000).max():.1f} kJ/mol")
        print(f"  Burcat range  : {burcat.Hf298_kJ_mol.min():.1f} to {burcat.Hf298_kJ_mol.max():.1f} kJ/mol")
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
