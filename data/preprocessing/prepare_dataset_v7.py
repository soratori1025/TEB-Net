"""
prepare_dataset_v7.py
=====================
Updates datasets for TEB-Net v7 by expanding the x_physic vector from 4D to 8D.
The new features from RDKit include:
  1. log(sigma)
  2. N (number of atoms including Hs)
  3. is_linear (1.0 or 0.0)
  4. M_scaled (MolWt / 100)
  5. TPSA (Topological Polar Surface Area)
  6. num_rotatable_bonds
  7. num_rings
  8. max_ring_size

Input files:
  - qm9_thermo.csv
  - rmg_thermo.csv
  - hc_thermo_combined.csv

Output files:
  - qm9_thermo_v7.csv
  - rmg_thermo_v7.csv
  - hc_thermo_combined_v7.csv
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prep_v7")

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(DATA_DIR.parent))

# Import helpers if available, else define simple fallbacks
try:
    from symmetry import compute_symmetry_number
    from molecular_geometry import compute_is_linear
except ImportError:
    try:
        from data.symmetry import compute_symmetry_number
        from data.molecular_geometry import compute_is_linear
    except ImportError:
        def compute_symmetry_number(mol): return 1.0
        def compute_is_linear(mol): return 0.0

def compute_x_physic_8d(smiles: str) -> list[float] | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        
        # Original 4 features
        sigma = compute_symmetry_number(mol)
        log_sigma = float(np.log(max(sigma, 1)))
        
        mol_h = Chem.AddHs(mol)
        N = float(mol_h.GetNumAtoms())
        
        is_linear = float(compute_is_linear(mol))
        
        M_scaled = Descriptors.MolWt(mol) / 100.0
        
        # New 4 features
        tpsa = Descriptors.TPSA(mol) / 100.0  # Scale TPSA
        n_rotatable = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        ri = mol.GetRingInfo()
        n_rings = float(ri.NumRings())
        
        max_ring_size = 0.0
        if n_rings > 0:
            max_ring_size = float(max(len(ring) for ring in ri.AtomRings()))
            
        return [log_sigma, N, is_linear, M_scaled, tpsa, n_rotatable, n_rings, max_ring_size]
    except Exception as e:
        logger.debug(f"Failed to compute 8D features for {smiles}: {e}")
        return None

def process_dataset(input_name: str, output_name: str):
    in_path = DATA_DIR / input_name
    out_path = DATA_DIR / output_name
    
    if not in_path.exists():
        logger.error(f"Input file {in_path} does not exist!")
        return
        
    logger.info(f"Processing {input_name} ...")
    df = pd.read_csv(in_path)
    
    # Check if SMILES column exists
    smiles_col = 'smiles'
    if smiles_col not in df.columns:
        if 'mol_1' in df.columns:
            smiles_col = 'mol_1'
        else:
            logger.error(f"Could not find smiles column in {input_name}")
            return
            
    # Compute new features
    logger.info(f"Computing 8D x_physic for {len(df)} entries...")
    tqdm.pandas()
    x_physic_list = df[smiles_col].progress_apply(compute_x_physic_8d)
    
    # Store as string to be parsed by dataset.py
    df['x_physic'] = x_physic_list.apply(lambda x: str(x) if x is not None else None)
    
    # Drop rows where feature computation failed
    n_before = len(df)
    df = df.dropna(subset=['x_physic']).reset_index(drop=True)
    n_failed = n_before - len(df)
    
    if n_failed > 0:
        logger.warning(f"Dropped {n_failed} entries due to failed x_physic computation.")
        
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} entries to {out_path}\n")

if __name__ == "__main__":
    datasets = [
        ("qm9_thermo.csv", "qm9_thermo_v7.csv"),
        ("rmg_thermo.csv", "rmg_thermo_v7.csv"),
        ("hc_thermo_combined.csv", "hc_thermo_combined_v7.csv")
    ]
    
    for in_file, out_file in datasets:
        process_dataset(in_file, out_file)
        
    logger.info("Dataset preparation for v7 complete!")
