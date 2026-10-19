import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# CONFIGURATION
# ============================================================

NUMERIC_COLUMNS = [
    "molecular_weight",
    "tmin_K",
    "tmax_K",
    "tmid_K",
    "Cp_298_J_mol_K",
    "Hf298_kJ_mol",
    "H_298_kJ_mol",
    "S_298_J_mol_K",
]

THERMO_COLUMNS = [
    "Cp_298_J_mol_K",
    "Hf298_kJ_mol",
    "H_298_kJ_mol",
    "S_298_J_mol_K",
]

PERCENTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_value_counts(df, column, output_path):
    """
    Save value counts of a categorical column.
    """
    if column not in df.columns:
        return None

    counts = (
        df[column]
        .fillna("MISSING")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    counts["percentage"] = (
        counts["count"] / len(df) * 100
    )

    counts.to_csv(output_path, index=False)

    return counts


def numeric_statistics(df, columns):
    """
    Calculate detailed statistics for numeric columns.
    """
    rows = []

    for column in columns:

        if column not in df.columns:
            continue

        x = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid = x.dropna()

        if len(valid) == 0:
            continue

        stats = {
            "column": column,
            "count": len(valid),
            "missing": x.isna().sum(),
            "missing_percent": x.isna().mean() * 100,

            "min": valid.min(),

            "p01": valid.quantile(0.01),
            "p05": valid.quantile(0.05),
            "p10": valid.quantile(0.10),

            "p25": valid.quantile(0.25),
            "median": valid.quantile(0.50),
            "mean": valid.mean(),

            "p75": valid.quantile(0.75),

            "p90": valid.quantile(0.90),
            "p95": valid.quantile(0.95),
            "p99": valid.quantile(0.99),

            "max": valid.max(),

            "std": valid.std(),
        }

        rows.append(stats)

    return pd.DataFrame(rows)


def plot_histogram(
    data,
    title,
    xlabel,
    output_path,
    bins=100
):
    """
    Plot a histogram.

    This function does NOT remove outliers.
    The full raw distribution is plotted.
    """

    data = pd.to_numeric(
        data,
        errors="coerce"
    ).dropna()

    if len(data) == 0:
        print(f"[WARNING] No valid data for: {title}")
        return

    plt.figure(figsize=(10, 6))

    plt.hist(
        data,
        bins=bins,
        edgecolor="black"
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of records")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def plot_bar(
    counts,
    category_column,
    title,
    xlabel,
    output_path
):
    """
    Plot categorical distribution.
    """

    if counts is None or len(counts) == 0:
        return

    plt.figure(figsize=(10, 6))

    plt.bar(
        counts[category_column].astype(str),
        counts["count"]
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of records")

    plt.xticks(rotation=45)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def print_distribution(name, series):
    """
    Print a categorical distribution to terminal.
    """

    print()
    print(name)
    print("-" * 70)

    counts = (
        series
        .fillna("MISSING")
        .value_counts(dropna=False)
    )

    total = len(series)

    for value, count in counts.items():

        percent = count / total * 100

        print(
            f"{str(value):20s}"
            f"{count:8,d}"
            f"   {percent:6.2f}%"
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_dataset(df, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_records = len(df)

    print()
    print("=" * 80)
    print("BURCAT DATASET ANALYSIS")
    print("=" * 80)

    print()
    print(f"Total records: {total_records:,}")

    print()
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    # --------------------------------------------------------
    # 1. PARSE STATUS
    # --------------------------------------------------------

    print_distribution(
        "PARSE STATUS DISTRIBUTION",
        df["parse_status"]
    )

    parse_counts = save_value_counts(
        df,
        "parse_status",
        output_dir / "parse_status_distribution.csv"
    )

    # --------------------------------------------------------
    # 2. PHASE DISTRIBUTION
    # --------------------------------------------------------

    if "phase_code" in df.columns:

        print_distribution(
            "PHASE DISTRIBUTION",
            df["phase_code"]
        )

        phase_counts = save_value_counts(
            df,
            "phase_code",
            output_dir / "phase_distribution.csv"
        )

        plot_bar(
            phase_counts,
            "phase_code",
            "BURCAT Phase Distribution",
            "Phase",
            output_dir / "phase_distribution.png"
        )

    # --------------------------------------------------------
    # 3. REFERENCE DISTRIBUTION
    # --------------------------------------------------------

    if "reference_code" in df.columns:

        print_distribution(
            "REFERENCE CODE DISTRIBUTION",
            df["reference_code"]
        )

        reference_counts = save_value_counts(
            df,
            "reference_code",
            output_dir / "reference_distribution.csv"
        )

        plot_bar(
            reference_counts,
            "reference_code",
            "BURCAT Reference Distribution",
            "Reference Code",
            output_dir / "reference_distribution.png"
        )

    # --------------------------------------------------------
    # 4. UNIQUE SPECIES / CAS
    # --------------------------------------------------------

    print()
    print("UNIQUE IDENTIFIERS")
    print("-" * 70)

    if "cas" in df.columns:

        unique_cas = df["cas"].nunique(
            dropna=True
        )

        duplicated_cas = (
            df["cas"]
            .duplicated()
            .sum()
        )

        print(
            f"Unique CAS numbers       : "
            f"{unique_cas:,}"
        )

        print(
            f"Duplicated CAS records   : "
            f"{duplicated_cas:,}"
        )

    if "species_label" in df.columns:

        unique_species = (
            df["species_label"]
            .nunique(dropna=True)
        )

        print(
            f"Unique species labels    : "
            f"{unique_species:,}"
        )

    # --------------------------------------------------------
    # 5. MISSING VALUES
    # --------------------------------------------------------

    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": [
            df[col].isna().sum()
            for col in df.columns
        ],
        "missing_percent": [
            df[col].isna().mean() * 100
            for col in df.columns
        ]
    })

    missing = missing.sort_values(
        "missing_count",
        ascending=False
    )

    missing.to_csv(
        output_dir / "missing_values.csv",
        index=False
    )

    print()
    print("MISSING VALUES")
    print("-" * 70)

    print(
        missing.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 6. NUMERIC STATISTICS
    # --------------------------------------------------------

    stats = numeric_statistics(
        df,
        NUMERIC_COLUMNS
    )

    stats.to_csv(
        output_dir / "numeric_statistics.csv",
        index=False
    )

    print()
    print("NUMERIC STATISTICS")
    print("-" * 70)

    if len(stats) > 0:

        print(
            stats.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 7. THERMODYNAMIC AVAILABILITY
    # --------------------------------------------------------

    print()
    print("THERMODYNAMIC PROPERTY AVAILABILITY")
    print("-" * 70)

    thermo_summary = []

    for col in THERMO_COLUMNS:

        if col not in df.columns:
            continue

        available = df[col].notna().sum()

        percentage = (
            available /
            total_records *
            100
        )

        print(
            f"{col:30s}"
            f"{available:8,d}"
            f"   {percentage:6.2f}%"
        )

        thermo_summary.append({
            "property": col,
            "available": available,
            "missing": total_records - available,
            "percentage": percentage
        })

    pd.DataFrame(
        thermo_summary
    ).to_csv(
        output_dir /
        "thermodynamic_availability.csv",
        index=False
    )

    # --------------------------------------------------------
    # 8. FULL THERMODYNAMIC RECORDS
    # --------------------------------------------------------

    available_columns = [
        col
        for col in THERMO_COLUMNS
        if col in df.columns
    ]

    if available_columns:

        complete_mask = (
            df[available_columns]
            .notna()
            .all(axis=1)
        )

        complete_records = (
            complete_mask.sum()
        )

        print()
        print("COMPLETE THERMODYNAMIC RECORDS")
        print("-" * 70)

        print(
            f"Records with ALL of "
            f"Cp + Hf + H + S available: "
            f"{complete_records:,} "
            f"({complete_records / total_records * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # 9. OUTLIER / EXTREME RECORD REPORT
    # --------------------------------------------------------

    print()
    print("EXTREME VALUE RECORDS")
    print("-" * 70)

    extreme_rows = []

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid = values.dropna()

        if len(valid) == 0:
            continue

        min_idx = valid.idxmin()
        max_idx = valid.idxmax()

        for kind, idx in [
            ("minimum", min_idx),
            ("maximum", max_idx)
        ]:

            row = {
                "column": column,
                "type": kind,
                "value": values.loc[idx],
                "record_index": df.loc[idx].get(
                    "record_index",
                    None
                ),
                "cas": df.loc[idx].get(
                    "cas",
                    None
                ),
                "species_label": df.loc[idx].get(
                    "species_label",
                    None
                ),
                "phase_code": df.loc[idx].get(
                    "phase_code",
                    None
                )
            }

            extreme_rows.append(row)

            print(
                f"{column:25s} "
                f"{kind:8s} "
                f"value={row['value']:.8g} | "
                f"CAS={row['cas']} | "
                f"Species={row['species_label']} | "
                f"Phase={row['phase_code']}"
            )

    extreme_df = pd.DataFrame(
        extreme_rows
    )

    extreme_df.to_csv(
        output_dir /
        "extreme_value_records.csv",
        index=False
    )

    # --------------------------------------------------------
    # 10. ZERO VALUE REPORT
    # --------------------------------------------------------

    zero_summary = []

    for column in THERMO_COLUMNS:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        zero_count = (
            values == 0
        ).sum()

        negative_count = (
            values < 0
        ).sum()

        zero_summary.append({
            "column": column,
            "zero_count": zero_count,
            "negative_count": negative_count
        })

    zero_df = pd.DataFrame(
        zero_summary
    )

    zero_df.to_csv(
        output_dir /
        "zero_and_negative_values.csv",
        index=False
    )

    print()
    print("ZERO / NEGATIVE VALUE REPORT")
    print("-" * 70)

    print(
        zero_df.to_string(
            index=False
        )
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print()
    print("GENERATING PLOTS...")

    # Molecular weight

    if "molecular_weight" in df.columns:

        plot_histogram(
            df["molecular_weight"],
            "BURCAT Molecular Weight Distribution",
            "Molecular Weight",
            output_dir /
            "molecular_weight_histogram.png"
        )

    # Cp

    if "Cp_298_J_mol_K" in df.columns:

        plot_histogram(
            df["Cp_298_J_mol_K"],
            "Heat Capacity Distribution at 298.15 K",
            r"$C_p$ (J/mol/K)",
            output_dir /
            "cp_298_histogram.png"
        )

    # Hf

    if "Hf298_kJ_mol" in df.columns:

        plot_histogram(
            df["Hf298_kJ_mol"],
            "Heat of Formation Distribution at 298.15 K",
            r"$\Delta_f H^\circ$ (kJ/mol)",
            output_dir /
            "hf298_histogram.png"
        )

    # Entropy

    if "S_298_J_mol_K" in df.columns:

        plot_histogram(
            df["S_298_J_mol_K"],
            "Entropy Distribution at 298.15 K",
            r"$S^\circ$ (J/mol/K)",
            output_dir /
            "entropy_298_histogram.png"
        )

    # Tmin / Tmax

    if (
        "tmin_K" in df.columns
        and "tmax_K" in df.columns
    ):

        plt.figure(figsize=(10, 6))

        plt.hist(
            pd.to_numeric(
                df["tmin_K"],
                errors="coerce"
            ).dropna(),
            bins=100,
            alpha=0.6,
            label="Tmin"
        )

        plt.hist(
            pd.to_numeric(
                df["tmax_K"],
                errors="coerce"
            ).dropna(),
            bins=100,
            alpha=0.6,
            label="Tmax"
        )

        plt.title(
            "BURCAT Temperature Range Distribution"
        )

        plt.xlabel(
            "Temperature (K)"
        )

        plt.ylabel(
            "Number of records"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir /
            "temperature_range.png",
            dpi=200
        )

        plt.close()

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary_path = (
        output_dir /
        "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BURCAT DATASET ANALYSIS\n"
        )

        f.write(
            "=" * 80 +
            "\n\n"
        )

        f.write(
            f"Total records: "
            f"{total_records:,}\n"
        )

        if "cas" in df.columns:

            f.write(
                f"Unique CAS: "
                f"{df['cas'].nunique(dropna=True):,}\n"
            )

        if "species_label" in df.columns:

            f.write(
                f"Unique species labels: "
                f"{df['species_label'].nunique(dropna=True):,}\n"
            )

        f.write("\n")

        f.write(
            "Parse status:\n"
        )

        if "parse_status" in df.columns:

            f.write(
                df["parse_status"]
                .fillna("MISSING")
                .value_counts()
                .to_string()
            )

        f.write("\n\n")

        f.write(
            "Phase distribution:\n"
        )

        if "phase_code" in df.columns:

            f.write(
                df["phase_code"]
                .fillna("MISSING")
                .value_counts()
                .to_string()
            )

        f.write("\n\n")

        f.write(
            "Numeric statistics:\n"
        )

        f.write(
            stats.to_string(
                index=False
            )
        )

        f.write("\n")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Results saved to:\n"
        f"{output_dir.resolve()}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Analyze parsed BURCAT thermodynamic dataset."
    )

    parser.add_argument(
        "input_csv",
        help=
        "Path to burcat_parsed.csv"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="burcat_analysis",
        help=
        "Output directory "
        "(default: burcat_analysis)"
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_csv
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    print(
        f"Loading dataset: "
        f"{input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    analyze_dataset(
        df,
        args.output
    )


if __name__ == "__main__":
    main()