#!/usr/bin/env python3
"""
Parse BURCAT.THR / BURCAT.THR.txt into a faithful CSV.

IMPORTANT:
- This version does NOT filter chemistry.
- Every BURCAT thermochemical record is retained if a record can be detected.
- It parses the NASA-7 coefficients correctly:
    line 2: high-T a1..a5
    line 3: high-T a6..a7 + low-T a1..a3
    line 4: low-T a4..a7 + Hf298/R
- It also calculates Cp, H, S at 298.15 K from the NASA polynomial.
- Hf298 is retained separately from the 15th BURCAT value.

Output:
    burcat_parsed_v2.csv

This is a RAW/PARSED dataset, not the final ML dataset.
"""

import argparse
import math
import re
from pathlib import Path

import pandas as pd


CAS_RE = re.compile(r"^\s*(\d{2,7}-\d{2}-\d)\b")

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"

# BURCAT header can occur in two forms:
#   ... G Tmin Tmax REF MW 1
# or
#   ... G Tmin Tmax Tmid REF MW 1
HEADER_RE = re.compile(
    rf"(?P<phase>[GSL])\s+"
    rf"(?P<tmin>{NUM})\s+"
    rf"(?P<tmax>{NUM})\s+"
    rf"(?:(?P<tmid>{NUM})\s+)?"
    rf"(?P<ref>[A-Za-z])\s+"
    rf"(?P<mw>{NUM})\s+1\s*$"
)

def to_float(x):
    """Convert BURCAT scientific notation, including D/d exponents."""
    return float(x.replace("D", "E").replace("d", "e"))

def extract_numbers(line):
    """Extract numeric fields, ignoring the trailing NASA line number 2/3/4."""
    nums = re.findall(NUM, line)
    if nums and nums[-1] in {"2", "3", "4"}:
        nums = nums[:-1]
    return [to_float(x) for x in nums]

def parse_header(line):
    m = HEADER_RE.search(line.rstrip())
    if not m:
        return None

    d = m.groupdict()
    return {
        "phase_code": d["phase"],
        "tmin_K": to_float(d["tmin"]),
        "tmax_K": to_float(d["tmax"]),
        "tmid_K": to_float(d["tmid"]) if d["tmid"] else None,
        "reference_code": d["ref"],
        "molecular_weight": to_float(d["mw"]),
    }

def nasa7_values(values):
    """
    BURCAT's 15 thermodynamic values are ordered as:

      high: a1 a2 a3 a4 a5
      high: a6 a7
      low : a1 a2 a3
      low : a4 a5 a6 a7
      Hf298/R

    => total = 15 values.
    """
    if len(values) != 15:
        return None

    return {
        "a1_high": values[0],
        "a2_high": values[1],
        "a3_high": values[2],
        "a4_high": values[3],
        "a5_high": values[4],
        "a6_high": values[5],
        "a7_high": values[6],
        "a1_low": values[7],
        "a2_low": values[8],
        "a3_low": values[9],
        "a4_low": values[10],
        "a5_low": values[11],
        "a6_low": values[12],
        "a7_low": values[13],
        "Hf298_over_R": values[14],
    }

def thermo_from_nasa(a, T=298.15, use_high=False):
    """Return Cp [J/mol/K], H [kJ/mol], S [J/mol/K]."""
    suffix = "_high" if use_high else "_low"

    a1 = a["a1" + suffix]
    a2 = a["a2" + suffix]
    a3 = a["a3" + suffix]
    a4 = a["a4" + suffix]
    a5 = a["a5" + suffix]
    a6 = a["a6" + suffix]
    a7 = a["a7" + suffix]

    R = 8.31446261815324  # J mol-1 K-1

    cp_over_R = (
        a1
        + a2*T
        + a3*T**2
        + a4*T**3
        + a5*T**4
    )

    h_over_RT = (
        a1
        + a2*T/2
        + a3*T**2/3
        + a4*T**3/4
        + a5*T**4/5
        + a6/T
    )

    s_over_R = (
        a1*math.log(T)
        + a2*T
        + a3*T**2/2
        + a4*T**3/3
        + a5*T**4/4
        + a7
    )

    return {
        "Cp_J_mol_K": cp_over_R * R,
        "H_kJ_mol": h_over_RT * R * T / 1000.0,
        "S_J_mol_K": s_over_R * R,
    }

def choose_temperature_range(header, T):
    """
    Select NASA low/high range.

    If Tmid is explicitly present, use it.
    Otherwise BURCAT records commonly use 1000 K as the NASA split.
    """
    tmid = header["tmid_K"] if header["tmid_K"] is not None else 1000.0
    return T >= tmid

def parse_file(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    records = []
    i = 0
    record_index = 0
    preamble_lines = 0

    while i < len(lines):
        line = lines[i]

        # A BURCAT record starts with a CAS line.
        cas_match = CAS_RE.match(line)
        if not cas_match:
            preamble_lines += 1
            i += 1
            continue

        cas = cas_match.group(1)
        comment_lines = [line]
        j = i + 1
        header_idx = None
        header = None

        # Find the species/NASA header before the next CAS record.
        while j < len(lines):
            if CAS_RE.match(lines[j]):
                break

            h = parse_header(lines[j])
            if h is not None:
                header_idx = j
                header = h
                break

            comment_lines.append(lines[j])
            j += 1

        record_index += 1

        row = {
            "record_index": record_index,
            "cas": cas,
            "species_label": None,
            "phase_code": None,
            "tmin_K": None,
            "tmax_K": None,
            "tmid_K": None,
            "reference_code": None,
            "molecular_weight": None,

            # NASA coefficients
            "a1_high": None,
            "a2_high": None,
            "a3_high": None,
            "a4_high": None,
            "a5_high": None,
            "a6_high": None,
            "a7_high": None,
            "a1_low": None,
            "a2_low": None,
            "a3_low": None,
            "a4_low": None,
            "a5_low": None,
            "a6_low": None,
            "a7_low": None,

            # Explicit BURCAT thermochemical value
            "Hf298_over_R": None,
            "Hf298_kJ_mol": None,

            # Thermodynamic functions evaluated at 298.15 K
            "Cp_298_J_mol_K": None,
            "H_298_kJ_mol": None,
            "S_298_J_mol_K": None,

            "parse_status": None,
            "parse_error": None,
            "raw_header": None,
            "raw_coefficient_lines": None,
            "raw_comments": "\n".join(comment_lines),
        }

        if header_idx is None:
            row["parse_status"] = "missing_header"
            row["parse_error"] = "Could not find BURCAT NASA header."
            records.append(row)
            i = j
            continue

        # Species label is the text before the NASA date/reference section.
        header_line = lines[header_idx]
        row["raw_header"] = header_line

        # Usually the species name is the first field(s) before the date code.
        # Keep a conservative label: text before the temperature region.
        m_label = re.match(r"^\s*(.*?)\s+(?:[A-Za-z]?\s*)?[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?", header_line)
        if m_label:
            row["species_label"] = m_label.group(1).strip()
        else:
            row["species_label"] = header_line[:18].strip()

        row.update(header)

        # The 3 coefficient lines follow the header.
        coeff_lines = []
        k = header_idx + 1

        while k < len(lines) and len(coeff_lines) < 3:
            # Stop if another record starts unexpectedly.
            if CAS_RE.match(lines[k]):
                break

            stripped = lines[k].strip()

            # Skip blank/comment lines only if they are not coefficient lines.
            nums = extract_numbers(stripped)
            if nums:
                coeff_lines.append(lines[k])

            k += 1

        row["raw_coefficient_lines"] = "\n".join(coeff_lines)

        values = []
        for cl in coeff_lines:
            values.extend(extract_numbers(cl))

        if len(values) != 15:
            row["parse_status"] = "bad_coefficient_count"
            row["parse_error"] = (
                f"Expected 15 NASA values from 3 coefficient lines; found {len(values)}."
            )
            records.append(row)
            i = k
            continue

        nasa = nasa7_values(values)
        row.update(nasa)

        # Explicit Hf298/R is converted using R.
        R = 8.31446261815324
        row["Hf298_kJ_mol"] = row["Hf298_over_R"] * R / 1000.0

        try:
            use_high = choose_temperature_range(header, 298.15)
            thermo = thermo_from_nasa(nasa, T=298.15, use_high=use_high)
            row.update({
                "Cp_298_J_mol_K": thermo["Cp_J_mol_K"],
                "H_298_kJ_mol": thermo["H_kJ_mol"],
                "S_298_J_mol_K": thermo["S_J_mol_K"],
            })
            row["parse_status"] = "ok"
        except Exception as exc:
            row["parse_status"] = "thermo_calculation_error"
            row["parse_error"] = str(exc)

        records.append(row)
        i = k

    return pd.DataFrame(records), len(lines), preamble_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="BURCAT.THR or BURCAT.THR.txt")
    parser.add_argument("-o", "--output", default="burcat_parsed_v2.csv")
    args = parser.parse_args()

    df, n_lines, preamble = parse_file(args.input)

    print(f"Input file : {args.input}")
    print(f"Total lines: {n_lines}")
    print(f"Detected records: {len(df)}")
    print()
    print("=" * 70)
    print("BURCAT PARSE SUMMARY")
    print("=" * 70)
    print(f"Records parsed : {len(df):,}")
    print(f"Preamble lines : {preamble:,}")
    print()
    print("Parse status:")
    print(df["parse_status"].value_counts(dropna=False).to_string())
    print()
    print("Phase code:")
    print(df["phase_code"].fillna("UNKNOWN").value_counts().to_string())
    print()
    print("Thermodynamic values available:")
    for col in [
        "Cp_298_J_mol_K",
        "Hf298_kJ_mol",
        "H_298_kJ_mol",
        "S_298_J_mol_K",
    ]:
        n = df[col].notna().sum()
        print(f"  {col:24s}: {n:,}")
    print()
    print("Basic distributions:")
    for col in [
        "molecular_weight",
        "Cp_298_J_mol_K",
        "Hf298_kJ_mol",
        "H_298_kJ_mol",
        "S_298_J_mol_K",
    ]:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x):
            print(
                f"  {col:24s}: "
                f"n={len(x):,}, min={x.min():.5g}, "
                f"median={x.median():.5g}, mean={x.mean():.5g}, max={x.max():.5g}"
            )

    df.to_csv(args.output, index=False)
    print()
    print(f"CSV written to: {args.output}")

if __name__ == "__main__":
    main()
