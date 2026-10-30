import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, EState
from rdkit.Chem.EState import Fingerprinter
def get_rdkit_estate_descriptors():
    """
    Extracts all E-State / Electro-topological descriptors available in RDKit
    (EState_VSA1..11, VSA_EState1..10, EState atom type fingerprinter keys).
    """
    estate_desc_names = [name for name, _ in Descriptors.descList if 'EState' in name or 'VSA' in name]
    return estate_desc_names
def compute_estate_descriptors_single(smiles, desc_names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        res = {d: 0.0 for d in desc_names}
        res.update({f"EState_fp_{i}": 0.0 for i in range(79)})
        return res
    desc_dict = dict(Descriptors.descList)
    res = {}
    for d in desc_names:
        try:
            val = desc_dict[d](mol)
            res[d] = float(val) if np.isfinite(val) else 0.0
        except:
            res[d] = 0.0
    try:
        fp_counts, fp_sums = Fingerprinter.FingerprintMol(mol)
        for i, (cnt, sum_val) in enumerate(zip(fp_counts, fp_sums)):
            res[f"EState_sum_{i}"] = float(sum_val) if np.isfinite(sum_val) else 0.0
    except:
        for i in range(79):
            res[f"EState_sum_{i}"] = 0.0
    return res
def analyze_estate_features_on_wudily_cho(excel_path):
    print(f"Analyzing 70+ Electro-Topological (E-State) Descriptors for WUDILY-CHO dataset...\n")
    df_raw = pd.read_excel(excel_path)
    smiles_list = df_raw['Canonical_SMILES'].tolist()
    y_h298 = df_raw['Hf_298.15K_kcal/mol'].values
    sample_smiles = smiles_list[:1000]
    sample_y = y_h298[:1000]
    estate_desc_names = get_rdkit_estate_descriptors()
    records = []
    for smi in sample_smiles:
        records.append(compute_estate_descriptors_single(smi, estate_desc_names))
    df_feat = pd.DataFrame(records)
    print(f"Extracted {df_feat.shape[1]} total E-State & VSA features for {len(sample_smiles)} molecules.")
    stds = df_feat.std()
    non_zero_cols = stds[stds > 1e-6].index.tolist()
    zero_var_cols = stds[stds <= 1e-6].index.tolist()
    print(f"\n--- 1. E-STATE ZERO VARIANCE AUDIT ---")
    print(f"Active E-State features with non-zero variance: {len(non_zero_cols)}")
    print(f"Dead/Zero-variance E-State features: {len(zero_var_cols)}")
    df_active = df_feat[non_zero_cols]
    corr_matrix = df_active.corr().abs()
    high_corr_pairs = []
    cols = df_active.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr_matrix.loc[c1, c2]
            if val > 0.90:
                high_corr_pairs.append((c1, c2, val))
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n--- 2. E-STATE PAIRWISE CORRELATION AUDIT (> 0.90) ---")
    print(f"Found {len(high_corr_pairs)} pairs with high correlation (> 0.90). Top 10 pairs:")
    for c1, c2, val in high_corr_pairs[:10]:
        print(f"  {c1:25s} <--> {c2:25s} : {val:.4f}")
    target_corrs = {}
    for col in non_zero_cols:
        c = np.corrcoef(df_active[col].values, sample_y)[0, 1]
        target_corrs[col] = c if np.isfinite(c) else 0.0
    df_target_corr = pd.Series(target_corrs).sort_values(key=abs, ascending=False)
    print(f"\n--- 3. TOP E-STATE FEATURES CORRELATED WITH Hf_298 ---")
    for name, val in df_target_corr.head(15).items():
        print(f"  {name:25s} : {val:+.4f}")
    selected_estate = []
    for col in df_target_corr.index:
        is_collinear = False
        for s in selected_estate:
            if corr_matrix.loc[col, s] > 0.85:
                is_collinear = True
                break
        if not is_collinear:
            selected_estate.append(col)
        if len(selected_estate) >= 20:
            break
    print(f"\n--- 4. SELECTED TOP 20 NON-COLLINEAR E-STATE FEATURES FOR TEBNET ---")
    for idx, f in enumerate(selected_estate):
        print(f"  {idx+1:02d}. {f:25s} (corr: {df_target_corr[f]:+.4f})")
    return df_active, selected_estate
if __name__ == '__main__':
    excel_path = "/home/nvsinh1/brats_segmentation/teb-net-model/data/WUDILY-CHO/wudily-cho.xlsx"
    analyze_estate_features_on_wudily_cho(excel_path)
