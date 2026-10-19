import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NASA_COEFFICIENTS = [
    "a1_high",
    "a2_high",
    "a3_high",
    "a4_high",
    "a5_high",
    "a6_high",
    "a7_high",
    "a1_low",
    "a2_low",
    "a3_low",
    "a4_low",
    "a5_low",
    "a6_low",
    "a7_low",
]

THERMO_COLUMNS = [
    "Cp_298_J_mol_K",
    "Hf298_kJ_mol",
    "H_298_kJ_mol",
    "S_298_J_mol_K",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_species_label(label):
    """
    Clean BURCAT fixed-width species label.

    Example:
        'Ag                g10/97AG 1.   0.   0.   0.G'

    becomes approximately:
        'Ag'
    """

    if pd.isna(label):
        return ""

    label = str(label).strip()

    # BURCAT comments/reference information often starts
    # after the first block of whitespace.
    #
    # We intentionally keep this conservative.
    match = re.match(r"^(\S+)", label)

    if match:
        return match.group(1)

    return label


def normalize_formula(formula):
    """
    Basic normalization of a chemical formula.

    This is NOT a full chemical parser.

    It is only used for heuristic inspection.
    """

    if not formula:
        return ""

    formula = str(formula).strip()

    # Remove common phase suffixes
    formula = re.sub(
        r"\((?:cr|s|l|g|gas|liq|solid)\)$",
        "",
        formula,
        flags=re.IGNORECASE
    )

    formula = formula.strip()

    return formula


def classify_species(label):
    """
    Heuristic classification of BURCAT species.

    Categories:
        electron
        ion
        radical
        atom
        molecule
        unknown

    NOTE:
        This is intentionally conservative.
        It is NOT a chemically rigorous parser.
    """

    if not label:
        return "unknown"

    label = str(label).strip()

    # --------------------------------------------------------
    # Electron
    # --------------------------------------------------------

    if label in {"e-", "e", "electron"}:
        return "electron"

    # --------------------------------------------------------
    # Ion
    # --------------------------------------------------------

    # Common BURCAT ion notation:
    # H+
    # OH-
    # SiD+
    # etc.

    if re.search(r"[+-]$", label):
        return "ion"

    # Charges written as:
    # X+1, X-1, X+2, X-2

    if re.search(r"[+-]\d+$", label):
        return "ion"

    # --------------------------------------------------------
    # Radical
    # --------------------------------------------------------

    # Common radical notation:
    # CH3
    # OH
    # HO2
    #
    # This cannot be reliably detected from the formula alone,
    # so only explicit radical markers are checked here.

    radical_patterns = [
        r"\brad\b",
        r"\bradical\b",
        r"\bRAD\b",
    ]

    for pattern in radical_patterns:

        if re.search(
            pattern,
            label,
            flags=re.IGNORECASE
        ):
            return "radical"

    # --------------------------------------------------------
    # Atom
    # --------------------------------------------------------

    # Single chemical element:
    #
    # H
    # He
    # Li
    # C
    # O
    # Ag
    #
    # Optional charge has already been handled.

    if re.fullmatch(
        r"[A-Z][a-z]?",
        label
    ):
        return "atom"

    # --------------------------------------------------------
    # Molecule
    # --------------------------------------------------------

    # Anything containing:
    # multiple atoms
    # digits
    # parentheses
    # dots
    #
    # is tentatively considered a molecule.

    if re.search(
        r"\d|[().]",
        label
    ):
        return "molecule"

    # Multi-element formula:
    if re.search(
        r"[A-Z][a-z].*[A-Z]",
        label
    ):
        return "molecule"

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return "unknown"


def safe_numeric(df, column):
    """
    Convert a column to numeric safely.
    """
    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 1. SPECIES CLASSIFICATION
# ============================================================

def analyze_species_classes(df, output_dir):

    print()
    print("=" * 80)
    print("1. SPECIES CLASSIFICATION")
    print("=" * 80)

    if "species_label" not in df.columns:
        print("[WARNING] species_label not found.")
        return df

    df = df.copy()

    df["species_name_basic"] = (
        df["species_label"]
        .apply(clean_species_label)
    )

    df["species_class"] = (
        df["species_name_basic"]
        .apply(classify_species)
    )

    counts = (
        df["species_class"]
        .value_counts()
        .rename_axis("species_class")
        .reset_index(name="count")
    )

    counts["percentage"] = (
        counts["count"] /
        len(df) *
        100
    )

    print(
        counts.to_string(
            index=False
        )
    )

    counts.to_csv(
        output_dir /
        "species_class_distribution.csv",
        index=False
    )

    # Save classified records
    columns = [
        "record_index",
        "cas",
        "species_label",
        "species_name_basic",
        "species_class",
        "phase_code",
        "molecular_weight",
        "Hf298_kJ_mol",
        "Cp_298_J_mol_K",
        "S_298_J_mol_K",
    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    df[columns].to_csv(
        output_dir /
        "species_classification.csv",
        index=False
    )

    return df


# ============================================================
# 2. DUPLICATE CAS ANALYSIS
# ============================================================

def analyze_duplicate_cas(df, output_dir):

    print()
    print("=" * 80)
    print("2. DUPLICATE CAS ANALYSIS")
    print("=" * 80)

    if "cas" not in df.columns:
        print("[WARNING] CAS column not found.")
        return

    cas_counts = (
        df["cas"]
        .value_counts()
        .rename_axis("cas")
        .reset_index(name="record_count")
    )

    duplicate_cas = cas_counts[
        cas_counts["record_count"] > 1
    ].copy()

    print(
        f"Unique CAS numbers        : "
        f"{df['cas'].nunique():,}"
    )

    print(
        f"CAS with multiple records : "
        f"{len(duplicate_cas):,}"
    )

    print(
        f"Records involved           : "
        f"{df['cas'].duplicated(keep=False).sum():,}"
    )

    duplicate_cas.to_csv(
        output_dir /
        "duplicate_cas_summary.csv",
        index=False
    )

    # Full duplicate records

    duplicate_records = df[
        df["cas"].duplicated(
            keep=False
        )
    ].copy()

    duplicate_records = duplicate_records.sort_values(
        by=["cas", "phase_code"]
    )

    columns = [
        "record_index",
        "cas",
        "species_label",
        "species_name_basic",
        "species_class",
        "phase_code",
        "reference_code",
        "molecular_weight",
        "tmin_K",
        "tmax_K",
        "Hf298_kJ_mol",
        "Cp_298_J_mol_K",
        "H_298_kJ_mol",
        "S_298_J_mol_K",
    ]

    columns = [
        c for c in columns
        if c in duplicate_records.columns
    ]

    duplicate_records[
        columns
    ].to_csv(
        output_dir /
        "duplicate_cas_records.csv",
        index=False
    )


# ============================================================
# 3. MULTIPLE PHASE ANALYSIS
# ============================================================

def analyze_multiple_phases(df, output_dir):

    print()
    print("=" * 80)
    print("3. MULTIPLE PHASE ANALYSIS")
    print("=" * 80)

    if "cas" not in df.columns:
        return

    phase_counts = (
        df.groupby("cas")["phase_code"]
        .nunique()
    )

    multi_phase_cas = (
        phase_counts[
            phase_counts > 1
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        f"CAS numbers appearing in "
        f"multiple phases: "
        f"{len(multi_phase_cas):,}"
    )

    result = (
        multi_phase_cas
        .rename("number_of_phases")
        .reset_index()
    )

    result.to_csv(
        output_dir /
        "multiple_phase_cas.csv",
        index=False
    )

    if len(multi_phase_cas) > 0:

        records = df[
            df["cas"].isin(
                multi_phase_cas.index
            )
        ].copy()

        records = records.sort_values(
            ["cas", "phase_code"]
        )

        columns = [
            "record_index",
            "cas",
            "species_label",
            "species_name_basic",
            "phase_code",
            "reference_code",
            "molecular_weight",
            "Hf298_kJ_mol",
            "Cp_298_J_mol_K",
            "S_298_J_mol_K",
        ]

        columns = [
            c for c in columns
            if c in records.columns
        ]

        records[
            columns
        ].to_csv(
            output_dir /
            "multiple_phase_records.csv",
            index=False
        )


# ============================================================
# 4. NASA COEFFICIENT DUPLICATE ANALYSIS
# ============================================================

def analyze_nasa_duplicates(df, output_dir):

    print()
    print("=" * 80)
    print("4. NASA COEFFICIENT DUPLICATE ANALYSIS")
    print("=" * 80)

    missing_columns = [
        c for c in NASA_COEFFICIENTS
        if c not in df.columns
    ]

    if missing_columns:

        print(
            "[WARNING] Missing coefficient columns:"
        )

        print(
            missing_columns
        )

        return

    valid = df.dropna(
        subset=NASA_COEFFICIENTS
    ).copy()

    print(
        f"Records with complete NASA coefficients: "
        f"{len(valid):,}"
    )

    # Exact duplicates

    coefficient_counts = (
        valid.groupby(
            NASA_COEFFICIENTS,
            dropna=False
        )
        .size()
        .reset_index(
            name="record_count"
        )
    )

    exact_duplicates = coefficient_counts[
        coefficient_counts["record_count"] > 1
    ].copy()

    print(
        f"Exact duplicated coefficient sets: "
        f"{len(exact_duplicates):,}"
    )

    exact_duplicates.to_csv(
        output_dir /
        "duplicate_nasa_coefficient_sets.csv",
        index=False
    )

    # --------------------------------------------------------
    # Coefficient fingerprint
    # --------------------------------------------------------

    def fingerprint(row):

        values = []

        for col in NASA_COEFFICIENTS:

            value = row[col]

            if pd.isna(value):
                values.append("NaN")
            else:
                values.append(
                    f"{float(value):.12e}"
                )

        return "|".join(values)

    valid["nasa_fingerprint"] = (
        valid.apply(
            fingerprint,
            axis=1
        )
    )

    fp_counts = (
        valid["nasa_fingerprint"]
        .value_counts()
    )

    duplicated_fp = fp_counts[
        fp_counts > 1
    ]

    if len(duplicated_fp) > 0:

        records = valid[
            valid["nasa_fingerprint"].isin(
                duplicated_fp.index
            )
        ]

        columns = [
            "record_index",
            "cas",
            "species_label",
            "phase_code",
            "nasa_fingerprint",
        ]

        records[
            columns
        ].to_csv(
            output_dir /
            "nasa_duplicate_records.csv",
            index=False
        )


# ============================================================
# 5. ZERO / SUSPICIOUS THERMO VALUES
# ============================================================

def analyze_thermo_anomalies(df, output_dir):

    print()
    print("=" * 80)
    print("5. THERMODYNAMIC ANOMALIES")
    print("=" * 80)

    anomaly_rows = []

    for column in THERMO_COLUMNS:

        if column not in df.columns:
            continue

        values = safe_numeric(
            df,
            column
        )

        zero_mask = (
            values == 0
        )

        negative_mask = (
            values < 0
        )

        print()
        print(column)

        print(
            f"  zero      : "
            f"{zero_mask.sum():,}"
        )

        print(
            f"  negative  : "
            f"{negative_mask.sum():,}"
        )

        for idx in df.index[
            zero_mask
        ]:

            anomaly_rows.append({
                "record_index":
                    df.loc[idx, "record_index"],
                "cas":
                    df.loc[idx, "cas"],
                "species_label":
                    df.loc[idx, "species_label"],
                "phase_code":
                    df.loc[idx, "phase_code"],
                "property":
                    column,
                "value":
                    values.loc[idx],
                "anomaly":
                    "zero"
            })

    anomaly_df = pd.DataFrame(
        anomaly_rows
    )

    anomaly_df.to_csv(
        output_dir /
        "thermodynamic_zero_values.csv",
        index=False
    )


# ============================================================
# 6. EXTREME THERMODYNAMIC RECORDS
# ============================================================

def analyze_extreme_thermo(df, output_dir):

    print()
    print("=" * 80)
    print("6. EXTREME THERMODYNAMIC RECORDS")
    print("=" * 80)

    rows = []

    for column in THERMO_COLUMNS:

        if column not in df.columns:
            continue

        values = safe_numeric(
            df,
            column
        )

        valid = values.dropna()

        if len(valid) == 0:
            continue

        # 10 lowest
        lowest = valid.nsmallest(10)

        # 10 highest
        highest = valid.nlargest(10)

        for kind, subset in [
            ("lowest", lowest),
            ("highest", highest)
        ]:

            for idx, value in subset.items():

                rows.append({
                    "property": column,
                    "rank_type": kind,
                    "value": value,
                    "record_index":
                        df.loc[idx, "record_index"],
                    "cas":
                        df.loc[idx, "cas"],
                    "species_label":
                        df.loc[idx, "species_label"],
                    "phase_code":
                        df.loc[idx, "phase_code"],
                    "molecular_weight":
                        df.loc[idx, "molecular_weight"],
                })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        output_dir /
        "extreme_thermodynamic_records.csv",
        index=False
    )

    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# 7. MOLECULAR WEIGHT INSPECTION
# ============================================================

def analyze_molecular_weight(df, output_dir):

    print()
    print("=" * 80)
    print("7. MOLECULAR WEIGHT INSPECTION")
    print("=" * 80)

    if "molecular_weight" not in df.columns:
        return

    mw = safe_numeric(
        df,
        "molecular_weight"
    )

    print(
        f"MW < 1             : "
        f"{(mw < 1).sum():,}"
    )

    print(
        f"MW < 5             : "
        f"{(mw < 5).sum():,}"
    )

    print(
        f"MW > 500           : "
        f"{(mw > 500).sum():,}"
    )

    print(
        f"MW > 1000          : "
        f"{(mw > 1000).sum():,}"
    )

    suspicious = df[
        (mw < 1) |
        (mw > 500)
    ].copy()

    columns = [
        "record_index",
        "cas",
        "species_label",
        "species_name_basic",
        "species_class",
        "phase_code",
        "molecular_weight",
        "Hf298_kJ_mol",
        "Cp_298_J_mol_K",
        "S_298_J_mol_K",
    ]

    columns = [
        c for c in columns
        if c in suspicious.columns
    ]

    suspicious[
        columns
    ].sort_values(
        "molecular_weight"
    ).to_csv(
        output_dir /
        "suspicious_molecular_weights.csv",
        index=False
    )


# ============================================================
# 8. INCOMPLETE RECORDS
# ============================================================

def analyze_incomplete_records(df, output_dir):

    print()
    print("=" * 80)
    print("8. INCOMPLETE RECORDS")
    print("=" * 80)

    # Missing header

    missing_header = df[
        df["parse_status"] ==
        "missing_header"
    ].copy()

    print(
        f"missing_header records: "
        f"{len(missing_header):,}"
    )

    missing_header.to_csv(
        output_dir /
        "missing_header_records.csv",
        index=False
    )

    # Bad coefficient count

    bad_coeff = df[
        df["parse_status"] ==
        "bad_coefficient_count"
    ].copy()

    print(
        f"bad_coefficient_count records: "
        f"{len(bad_coeff):,}"
    )

    bad_coeff.to_csv(
        output_dir /
        "bad_coefficient_records.csv",
        index=False
    )


# ============================================================
# 9. PHASE × SPECIES CLASS
# ============================================================

def analyze_phase_species_class(df, output_dir):

    print()
    print("=" * 80)
    print("9. PHASE × SPECIES CLASS")
    print("=" * 80)

    if (
        "phase_code" not in df.columns
        or "species_class" not in df.columns
    ):
        return

    table = pd.crosstab(
        df["species_class"],
        df["phase_code"],
        margins=True
    )

    print(
        table.to_string()
    )

    table.to_csv(
        output_dir /
        "phase_species_class_crosstab.csv"
    )


# ============================================================
# 10. REFERENCE × PHASE
# ============================================================

def analyze_reference_phase(df, output_dir):

    print()
    print("=" * 80)
    print("10. REFERENCE CODE × PHASE")
    print("=" * 80)

    if (
        "reference_code" not in df.columns
        or "phase_code" not in df.columns
    ):
        return

    table = pd.crosstab(
        df["reference_code"],
        df["phase_code"],
        margins=True
    )

    print(
        table.to_string()
    )

    table.to_csv(
        output_dir /
        "reference_phase_crosstab.csv"
    )


# ============================================================
# 11. THERMODYNAMIC CONSISTENCY CHECK
# ============================================================

def analyze_thermo_consistency(df, output_dir):

    print()
    print("=" * 80)
    print("11. THERMODYNAMIC CONSISTENCY CHECK")
    print("=" * 80)

    required = [
        "Hf298_over_R",
        "Hf298_kJ_mol",
    ]

    if not all(
        col in df.columns
        for col in required
    ):
        return

    R = 8.31446261815324

    calculated = (
        safe_numeric(
            df,
            "Hf298_over_R"
        )
        * R
        / 1000
    )

    stored = safe_numeric(
        df,
        "Hf298_kJ_mol"
    )

    difference = (
        calculated -
        stored
    )

    result = pd.DataFrame({
        "record_index":
            df["record_index"],
        "cas":
            df["cas"],
        "species_label":
            df["species_label"],
        "Hf298_stored":
            stored,
        "Hf298_recalculated":
            calculated,
        "difference_kJ_mol":
            difference.abs(),
    })

    valid = result[
        result["difference_kJ_mol"].notna()
    ]

    print(
        f"Records checked: "
        f"{len(valid):,}"
    )

    if len(valid) > 0:

        print(
            f"Maximum difference: "
            f"{valid['difference_kJ_mol'].max():.6e} "
            f"kJ/mol"
        )

        print(
            f"Mean difference: "
            f"{valid['difference_kJ_mol'].mean():.6e} "
            f"kJ/mol"
        )

    result.to_csv(
        output_dir /
        "thermo_consistency_Hf298.csv",
        index=False
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

def write_summary(df, output_dir):

    summary_path = (
        output_dir /
        "chemistry_inspection_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BURCAT CHEMISTRY INSPECTION\n"
        )

        f.write(
            "=" * 80 +
            "\n\n"
        )

        f.write(
            f"Total records: "
            f"{len(df):,}\n"
        )

        if "cas" in df.columns:

            f.write(
                f"Unique CAS: "
                f"{df['cas'].nunique():,}\n"
            )

        if "species_name_basic" in df.columns:

            f.write(
                f"Unique basic species names: "
                f"{df['species_name_basic'].nunique():,}\n"
            )

        f.write("\n")

        if "species_class" in df.columns:

            f.write(
                "Species classification:\n"
            )

            f.write(
                df["species_class"]
                .value_counts()
                .to_string()
            )

            f.write("\n\n")

        if "phase_code" in df.columns:

            f.write(
                "Phase distribution:\n"
            )

            f.write(
                df["phase_code"]
                .fillna("MISSING")
                .value_counts()
                .to_string()
            )

            f.write("\n\n")

        if "parse_status" in df.columns:

            f.write(
                "Parse status:\n"
            )

            f.write(
                df["parse_status"]
                .value_counts()
                .to_string()
            )

            f.write("\n\n")

        if "molecular_weight" in df.columns:

            mw = safe_numeric(
                df,
                "molecular_weight"
            )

            f.write(
                "Molecular weight:\n"
            )

            f.write(
                f"  min    = {mw.min()}\n"
            )

            f.write(
                f"  median = {mw.median()}\n"
            )

            f.write(
                f"  mean   = {mw.mean()}\n"
            )

            f.write(
                f"  max    = {mw.max()}\n"
            )

            f.write("\n")

        f.write(
            "NOTE:\n"
        )

        f.write(
            "This inspection does not perform filtering.\n"
        )

        f.write(
            "All decisions about removing records should "
            "be made after reviewing these reports.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Inspect chemical structure and "
        "thermodynamic characteristics "
        "of a parsed BURCAT dataset."
    )

    parser.add_argument(
        "input_csv",
        help="Path to burcat_parsed.csv"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="burcat_chemistry",
        help=
        "Output directory "
        "(default: burcat_chemistry)"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_csv
    )

    output_dir = Path(
        args.output
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        f"Loading dataset: "
        f"{input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Loaded {len(df):,} records."
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    df = analyze_species_classes(
        df,
        output_dir
    )

    analyze_duplicate_cas(
        df,
        output_dir
    )

    analyze_multiple_phases(
        df,
        output_dir
    )

    analyze_nasa_duplicates(
        df,
        output_dir
    )

    analyze_thermo_anomalies(
        df,
        output_dir
    )

    analyze_extreme_thermo(
        df,
        output_dir
    )

    analyze_molecular_weight(
        df,
        output_dir
    )

    analyze_incomplete_records(
        df,
        output_dir
    )

    analyze_phase_species_class(
        df,
        output_dir
    )

    analyze_reference_phase(
        df,
        output_dir
    )

    analyze_thermo_consistency(
        df,
        output_dir
    )

    write_summary(
        df,
        output_dir
    )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CHEMISTRY INSPECTION COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Results saved to:\n"
        f"{output_dir.resolve()}"
    )

    print()
    print(
        "IMPORTANT: No records were removed or modified."
    )


if __name__ == "__main__":
    main()