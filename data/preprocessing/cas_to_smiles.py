
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import urllib.request
    import urllib.error
    import urllib.parse
except ImportError:
    pass


# ============================================================
# PUBCHEM API
# ============================================================

PUBCHEM_BASE = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
)

# Rate limit: ~5 requests/second for free usage
REQUEST_DELAY = 0.25  # seconds between requests


def query_pubchem_smiles(cas_number, timeout=10):
    """
    Query PubChem for canonical SMILES given a CAS number.

    Returns
    -------
    str or None
        Canonical SMILES, or None if not found.
    """
    url = (
        f"{PUBCHEM_BASE}/compound/name/"
        f"{urllib.parse.quote(cas_number)}"
        f"/property/CanonicalSMILES/JSON"
    )

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        props = data.get("PropertyTable", {}).get("Properties", [])

        if props:
            p = props[0]
            for key in ["CanonicalSMILES", "SMILES", "IsomericSMILES", "ConnectivitySMILES"]:
                if key in p:
                    return p[key]

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # CAS not found in PubChem
        # Other HTTP errors — log and skip
        pass
    except Exception:
        pass

    return None


def validate_smiles(smiles):
    """
    Validate a SMILES string with RDKit.

    Returns
    -------
    str or None
        Canonical SMILES if valid, None otherwise.
    """
    if not HAS_RDKIT:
        return smiles  # Cannot validate

    if smiles is None or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Re-canonicalize
        canonical = Chem.MolToSmiles(mol)
        return canonical

    except Exception:
        return None


def smiles_to_inchikey(smiles):
    """Compute InChIKey from SMILES for deduplication."""
    if not HAS_RDKIT or smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


# ============================================================
# BATCH MAPPING
# ============================================================

def load_cache(cache_path):
    """Load cached CAS → SMILES mapping."""
    if Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    """Save CAS → SMILES mapping to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def map_cas_to_smiles(cas_list, cache_path="cas_smiles_cache.json"):
    """
    Map a list of CAS numbers to SMILES using PubChem.

    Uses a local JSON cache to avoid repeated API calls.

    Parameters
    ----------
    cas_list : list[str]
        Unique CAS numbers to look up.
    cache_path : str
        Path to cache file.

    Returns
    -------
    dict
        {cas: smiles_or_None}
    """
    cache = load_cache(cache_path)

    # Determine which CAS numbers need lookup
    to_lookup = [
        cas for cas in cas_list
        if cas not in cache
    ]

    print(f"Total CAS numbers: {len(cas_list):,}")
    print(f"Already cached: {len(cas_list) - len(to_lookup):,}")
    print(f"To look up: {len(to_lookup):,}")

    if len(to_lookup) == 0:
        return {cas: cache.get(cas) for cas in cas_list}

    print()
    print("Querying PubChem API...")

    found = 0
    errors = 0

    for i, cas in enumerate(to_lookup):

        # Progress report every 50 items
        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(to_lookup)}] "
                f"found={found}, missing={i+1-found-errors}, "
                f"errors={errors}"
            )

        try:
            smiles = query_pubchem_smiles(cas)

            if smiles is not None:
                # Validate with RDKit
                validated = validate_smiles(smiles)
                cache[cas] = validated
                if validated:
                    found += 1
                else:
                    cache[cas] = None
            else:
                cache[cas] = None

        except Exception as e:
            cache[cas] = None
            errors += 1

        # Rate limiting
        time.sleep(REQUEST_DELAY)

        # Save cache periodically
        if (i + 1) % 100 == 0:
            save_cache(cache, cache_path)

    # Final save
    save_cache(cache, cache_path)

    print()
    print(f"Lookup complete:")
    print(f"  Found SMILES: {found:,}")
    print(f"  Not found: {len(to_lookup) - found - errors:,}")
    print(f"  Errors: {errors:,}")

    return {cas: cache.get(cas) for cas in cas_list}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Map CAS numbers to SMILES using PubChem."
    )
    parser.add_argument(
        "--input", "-i",
        default="burcat_cleaned.csv",
        help="Input CSV with 'cas' column",
    )
    parser.add_argument(
        "--output", "-o",
        default="burcat_with_smiles.csv",
        help="Output CSV with SMILES added",
    )
    parser.add_argument(
        "--cache",
        default="cas_smiles_cache.json",
        help="Path to CAS → SMILES cache file",
    )
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} records from {args.input}")

    if "cas" not in df.columns:
        raise ValueError("Input CSV must have a 'cas' column.")

    # Get unique CAS numbers
    unique_cas = df["cas"].dropna().unique().tolist()
    print(f"Unique CAS numbers: {len(unique_cas):,}")

    # Map CAS → SMILES
    mapping = map_cas_to_smiles(unique_cas, cache_path=args.cache)

    # Merge into DataFrame
    df["smiles"] = df["cas"].map(mapping)

    # Add InChIKey for future deduplication
    if HAS_RDKIT:
        df["inchikey"] = df["smiles"].apply(smiles_to_inchikey)

    # Statistics
    has_smiles = df["smiles"].notna().sum()
    no_smiles = df["smiles"].isna().sum()

    print()
    print("=" * 80)
    print("CAS → SMILES MAPPING RESULTS")
    print("=" * 80)
    print(f"  Records with SMILES:    {has_smiles:,} ({has_smiles/len(df)*100:.1f}%)")
    print(f"  Records without SMILES: {no_smiles:,} ({no_smiles/len(df)*100:.1f}%)")

    if HAS_RDKIT:
        has_inchikey = df["inchikey"].notna().sum()
        print(f"  Records with InChIKey:  {has_inchikey:,}")

    # Save — include all records (SMILES=NaN for missing)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Also save the version with only valid SMILES
    df_valid = df[df["smiles"].notna()].copy()
    valid_path = args.output.replace(".csv", "_valid.csv")
    df_valid.to_csv(valid_path, index=False)
    print(f"Saved (valid SMILES only): {valid_path} ({len(df_valid):,} records)")


if __name__ == "__main__":
    main()
