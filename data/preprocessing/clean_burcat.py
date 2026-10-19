#!/usr/bin/env python3
"""
Clean Burcat parsed dataset for TEB-Net v2 training.

Pipeline:
    1. Remove parse failures (missing_header, bad_coefficient_count)
    2. Filter gas phase only (phase_code == 'G')
    3. Classify species and remove ions, atoms, electrons
    4. Resolve duplicate CAS (keep widest T-range covering 298K)
    5. Remove thermodynamic anomalies (Cp=0, S=0)
    6. Output cleaned CSV with statistics

Usage:
    python clean_burcat.py [--input burcat_parsed.csv] [--output burcat_cleaned.csv]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SPECIES CLASSIFICATION (reuse logic from inspect script)
# ============================================================

def clean_species_label(label):
    """Extract the first token as a basic species name."""
    if pd.isna(label):
        return ""
    label = str(label).strip()
    match = re.match(r"^(\S+)", label)
    return match.group(1) if match else label


def classify_species(label):
    """
    Heuristic classification of BURCAT species.

    Returns one of:
        electron, ion, radical, atom, molecule, unknown
    """
    if not label:
        return "unknown"

    label = str(label).strip()

    # Electron
    if label in {"e-", "e", "electron"}:
        return "electron"

    # Ion: trailing +/- or +N/-N
    if re.search(r"[+-]$", label):
        return "ion"
    if re.search(r"[+-]\d+$", label):
        return "ion"

    # Radical markers
    if re.search(r"\brad(ical)?\b", label, flags=re.IGNORECASE):
        return "radical"

    # Single atom (e.g. H, He, Ag)
    if re.fullmatch(r"[A-Z][a-z]?", label):
        return "atom"

    # Molecule: contains digits, parentheses, dots, or multiple elements
    if re.search(r"\d|[().]", label):
        return "molecule"
    if re.search(r"[A-Z][a-z].*[A-Z]", label):
        return "molecule"

    return "unknown"


# ============================================================
# REFERENCE CODE PRIORITY
# ============================================================

# Higher number = higher priority
# Burcat reference codes (observed in dataset):
#   A = ATcT (highest accuracy)
#   B = Burcat (reviewed polynomial)
#   C, D = other literature
#   E, F = estimates
REFERENCE_PRIORITY = {
    "A": 10,    # ATcT — highest accuracy
    "B": 7,     # Burcat reviewed
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "X": 1,
    "b": 6,
}


def get_reference_priority(ref_code):
    """Return numeric priority for a reference code."""
    if pd.isna(ref_code):
        return 0
    return REFERENCE_PRIORITY.get(str(ref_code).strip(), 0)


# ============================================================
# DUPLICATE CAS RESOLUTION
# ============================================================

def resolve_duplicate_cas(df):
    """
    For each CAS with multiple records, keep the best one.

    Selection criteria (in order):
        1. Temperature range must include 298.15 K
        2. Widest temperature range (tmax - tmin)
        3. Higher reference code priority
        4. First occurrence (tiebreaker)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with CAS duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame with one record per CAS.
    """
    # Add scoring columns
    df = df.copy()

    df["_covers_298"] = (
        (df["tmin_K"] <= 298.15) &
        (df["tmax_K"] >= 298.15)
    ).astype(int)

    df["_t_range"] = df["tmax_K"] - df["tmin_K"]

    df["_ref_priority"] = df["reference_code"].apply(
        get_reference_priority
    )

    # Sort by preference: covers_298 (desc), T_range (desc), ref_priority (desc)
    df = df.sort_values(
        by=["cas", "_covers_298", "_t_range", "_ref_priority"],
        ascending=[True, False, False, False],
    )

    # Keep first (best) record per CAS
    df_dedup = df.drop_duplicates(
        subset=["cas"],
        keep="first",
    )

    # Drop scoring columns
    df_dedup = df_dedup.drop(
        columns=["_covers_298", "_t_range", "_ref_priority"],
    )

    return df_dedup


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def clean_burcat(input_path, output_path=None, output_dir=None):
    """
    Run full cleaning pipeline on Burcat parsed CSV.

    Returns the cleaned DataFrame.
    """

    print()
    print("=" * 80)
    print("BURCAT CLEANING PIPELINE")
    print("=" * 80)

    # ----------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------

    df = pd.read_csv(input_path)
    total = len(df)

    print(f"\nLoaded: {total:,} records from {input_path}")

    # ----------------------------------------------------------
    # STEP 1: Remove parse failures
    # ----------------------------------------------------------

    df_ok = df[df["parse_status"] == "ok"].copy()
    n_removed_1 = total - len(df_ok)

    print(f"\n[Step 1] Remove parse failures")
    print(f"  Removed: {n_removed_1:,}")
    print(f"    missing_header: {(df['parse_status'] == 'missing_header').sum():,}")
    print(f"    bad_coefficient_count: {(df['parse_status'] == 'bad_coefficient_count').sum():,}")
    print(f"  Remaining: {len(df_ok):,}")

    # ----------------------------------------------------------
    # STEP 2: Gas phase only
    # ----------------------------------------------------------

    before = len(df_ok)
    df_gas = df_ok[df_ok["phase_code"] == "G"].copy()
    n_removed_2 = before - len(df_gas)

    print(f"\n[Step 2] Filter gas phase (G) only")
    print(f"  Removed: {n_removed_2:,}")
    if "phase_code" in df_ok.columns:
        for phase in df_ok["phase_code"].unique():
            if phase != "G":
                count = (df_ok["phase_code"] == phase).sum()
                print(f"    {phase}: {count:,}")
    print(f"  Remaining: {len(df_gas):,}")

    # ----------------------------------------------------------
    # STEP 3: Classify & remove non-molecules
    # ----------------------------------------------------------

    df_gas["species_name_clean"] = (
        df_gas["species_label"].apply(clean_species_label)
    )
    df_gas["species_class"] = (
        df_gas["species_name_clean"].apply(classify_species)
    )

    before = len(df_gas)

    # Keep only molecules, radicals, and unknowns that look molecular
    # Remove: electron, ion, atom
    keep_classes = {"molecule", "radical", "unknown"}
    remove_classes = {"electron", "ion", "atom"}

    df_mol = df_gas[
        df_gas["species_class"].isin(keep_classes)
    ].copy()

    n_removed_3 = before - len(df_mol)

    print(f"\n[Step 3] Remove non-molecular species (ions, atoms, electrons)")
    for cls in remove_classes:
        count = (df_gas["species_class"] == cls).sum()
        print(f"    {cls}: {count:,}")
    print(f"  Total removed: {n_removed_3:,}")
    print(f"  Remaining: {len(df_mol):,}")

    class_dist = df_mol["species_class"].value_counts()
    for cls, count in class_dist.items():
        print(f"    Kept {cls}: {count:,}")

    # ----------------------------------------------------------
    # STEP 4: Resolve duplicate CAS
    # ----------------------------------------------------------

    before = len(df_mol)
    n_dup_cas = df_mol["cas"].duplicated().sum()

    df_dedup = resolve_duplicate_cas(df_mol)
    n_removed_4 = before - len(df_dedup)

    print(f"\n[Step 4] Resolve duplicate CAS numbers")
    print(f"  Duplicated CAS entries: {n_dup_cas:,}")
    print(f"  Removed: {n_removed_4:,}")
    print(f"  Remaining: {len(df_dedup):,}")

    # ----------------------------------------------------------
    # STEP 5: Thermodynamic anomaly filters
    # ----------------------------------------------------------

    before = len(df_dedup)

    # Convert to numeric
    for col in ["Cp_298_J_mol_K", "S_298_J_mol_K", "molecular_weight"]:
        if col in df_dedup.columns:
            df_dedup[col] = pd.to_numeric(
                df_dedup[col], errors="coerce"
            )

    # Filters
    mask_cp_zero = (
        df_dedup["Cp_298_J_mol_K"].notna() &
        (df_dedup["Cp_298_J_mol_K"] == 0)
    )
    mask_s_zero = (
        df_dedup["S_298_J_mol_K"].notna() &
        (df_dedup["S_298_J_mol_K"] == 0)
    )
    mask_mw_tiny = (
        df_dedup["molecular_weight"].notna() &
        (df_dedup["molecular_weight"] < 2.0)
    )

    # Cp < 0 and S < 0 are unphysical for gas-phase molecules
    mask_cp_neg = (
        df_dedup["Cp_298_J_mol_K"].notna() &
        (df_dedup["Cp_298_J_mol_K"] < 0)
    )
    mask_s_neg = (
        df_dedup["S_298_J_mol_K"].notna() &
        (df_dedup["S_298_J_mol_K"] < 0)
    )

    combined_mask = (
        mask_cp_zero | mask_s_zero |
        mask_mw_tiny |
        mask_cp_neg | mask_s_neg
    )

    df_clean = df_dedup[~combined_mask].copy()
    n_removed_5 = before - len(df_clean)

    print(f"\n[Step 5] Remove thermodynamic anomalies")
    print(f"  Cp_298 == 0: {mask_cp_zero.sum():,}")
    print(f"  Cp_298 < 0: {mask_cp_neg.sum():,}")
    print(f"  S_298 == 0: {mask_s_zero.sum():,}")
    print(f"  S_298 < 0: {mask_s_neg.sum():,}")
    print(f"  MW < 2: {mask_mw_tiny.sum():,}")
    print(f"  Total removed: {n_removed_5:,}")
    print(f"  Remaining: {len(df_clean):,}")

    # ----------------------------------------------------------
    # STEP 6: Verify NASA coefficients are complete
    # ----------------------------------------------------------

    nasa_cols = [
        f"a{i}_{rng}"
        for rng in ["low", "high"]
        for i in range(1, 8)
    ]

    existing_nasa = [c for c in nasa_cols if c in df_clean.columns]

    before = len(df_clean)
    nasa_complete = df_clean[existing_nasa].notna().all(axis=1)
    df_clean = df_clean[nasa_complete].copy()
    n_removed_6 = before - len(df_clean)

    print(f"\n[Step 6] Verify complete NASA-7 coefficients")
    print(f"  Incomplete removed: {n_removed_6:,}")
    print(f"  Remaining: {len(df_clean):,}")

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------

    print()
    print("=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"  Original records:     {total:,}")
    print(f"  Parse failures:       -{n_removed_1:,}")
    print(f"  Non-gas phase:        -{n_removed_2:,}")
    print(f"  Non-molecular:        -{n_removed_3:,}")
    print(f"  CAS duplicates:       -{n_removed_4:,}")
    print(f"  Thermo anomalies:     -{n_removed_5:,}")
    print(f"  Incomplete NASA:      -{n_removed_6:,}")
    print(f"  ─────────────────────────────────")
    print(f"  FINAL cleaned:        {len(df_clean):,}")
    print(f"  Unique CAS:           {df_clean['cas'].nunique():,}")
    print()

    # Statistics
    for col in ["Cp_298_J_mol_K", "Hf298_kJ_mol", "S_298_J_mol_K", "molecular_weight"]:
        if col in df_clean.columns:
            vals = pd.to_numeric(df_clean[col], errors="coerce").dropna()
            print(
                f"  {col:24s}: "
                f"n={len(vals):,}, "
                f"min={vals.min():.4g}, "
                f"median={vals.median():.4g}, "
                f"mean={vals.mean():.4g}, "
                f"max={vals.max():.4g}"
            )

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------

    # Select output columns (drop raw/debug columns)
    output_columns = [
        "record_index", "cas",
        "species_label", "species_name_clean", "species_class",
        "phase_code", "reference_code",
        "molecular_weight",
        "tmin_K", "tmax_K", "tmid_K",
        # NASA coefficients
        "a1_high", "a2_high", "a3_high", "a4_high",
        "a5_high", "a6_high", "a7_high",
        "a1_low", "a2_low", "a3_low", "a4_low",
        "a5_low", "a6_low", "a7_low",
        # Thermodynamic values
        "Hf298_over_R", "Hf298_kJ_mol",
        "Cp_298_J_mol_K", "H_298_kJ_mol", "S_298_J_mol_K",
    ]
    output_columns = [c for c in output_columns if c in df_clean.columns]

    df_out = df_clean[output_columns].copy()
    df_out = df_out.reset_index(drop=True)

    if output_path:
        df_out.to_csv(output_path, index=False)
        print(f"\nSaved cleaned dataset: {output_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed reports
        df_out.describe().to_csv(output_dir / "cleaned_statistics.csv")

        # Species class distribution
        df_out["species_class"].value_counts().to_csv(
            output_dir / "cleaned_species_classes.csv"
        )

        print(f"Reports saved to: {output_dir}")

    return df_out


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean Burcat parsed dataset for TEB-Net v2."
    )
    parser.add_argument(
        "--input", "-i",
        default="burcat_parsed.csv",
        help="Path to burcat_parsed.csv (default: burcat_parsed.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="burcat_cleaned.csv",
        help="Output CSV path (default: burcat_cleaned.csv)",
    )
    parser.add_argument(
        "--report-dir",
        default="burcat_cleaned_analysis",
        help="Directory for analysis reports (default: burcat_cleaned_analysis)",
    )
    args = parser.parse_args()

    clean_burcat(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
