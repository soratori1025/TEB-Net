import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# 34 Table S1 Descriptors from Bruce et al. (2025)
TABLE_S1_DESCRIPTORS = [
    'Eccentricity',
    'MaxPartialCharge',
    'MinPartialCharge',
    'Molecular weight',
    'MOLvolume',
    'nACC',
    'nAHC',
    'nC',
    'nC(NO2)',
    'nC(NO2)2',
    'nC(NO2)3',
    'nCH3',
    'nH',
    'nHbondA',
    'nHbondD',
    'nHC',
    'nN',
    'nNH2',
    'nNNO2',
    'nNO2',
    'nO',
    'nOCH3',
    'nONO2',
    'NPR1',
    'NPR2',
    'nR',
    'nRbond',
    'ob',
    'PBF',
    'PMI1',
    'PMI2',
    'PMI3',
    'Total energy',
    'TPSA'
]

def compute_table_s1_descriptors_single(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {d: 0.0 for d in TABLE_S1_DESCRIPTORS}
    
    mol_h = Chem.AddHs(mol)
    
    # Basic atom counts
    nC = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    nH = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    nO = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    nN = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    
    mw = Descriptors.MolWt(mol)
    
    # Partial charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms() if a.HasProp('_GasteigerCharge')]
        charges = [c for c in charges if np.isfinite(c)]
        max_q = max(charges) if charges else 0.0
        min_q = min(charges) if charges else 0.0
    except:
        max_q, min_q = 0.0, 0.0
        
    # Hydrogen bonds & Rings
    nHbondA = Descriptors.NumHAcceptors(mol)
    nHbondD = Descriptors.NumHDonors(mol)
    nR = rdMolDescriptors.CalcNumRings(mol)
    nRbond = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Oxygen balance: -1600/MW * (2C + H/2 - O)
    ob = -1600.0 / max(1.0, mw) * (2.0 * nC + 0.5 * nH - nO)
    
    # Functional groups SMARTS
    p_ch3 = Chem.MolFromSmarts('[CX4H3]')
    p_och3 = Chem.MolFromSmarts('[OX2][CX4H3]')
    p_nh2 = Chem.MolFromSmarts('[NX3H2]')
    p_no2 = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3](=O)[O-])]')
    p_c_no2 = Chem.MolFromSmarts('[CX4][$([NX3](=O)=O)]')
    p_c_no2_2 = Chem.MolFromSmarts('[CX4]($([NX3](=O)=O))$([NX3](=O)=O)')
    p_c_no2_3 = Chem.MolFromSmarts('[CX4]($([NX3](=O)=O))($([NX3](=O)=O))$([NX3](=O)=O)')
    p_n_no2 = Chem.MolFromSmarts('[NX3][$([NX3](=O)=O)]')
    p_ono2 = Chem.MolFromSmarts('[OX2][$([NX3](=O)=O)]')
    
    nCH3 = len(mol.GetSubstructMatches(p_ch3)) if p_ch3 else 0
    nOCH3 = len(mol.GetSubstructMatches(p_och3)) if p_och3 else 0
    nNH2 = len(mol.GetSubstructMatches(p_nh2)) if p_nh2 else 0
    nNO2 = len(mol.GetSubstructMatches(p_no2)) if p_no2 else 0
    nC_NO2 = len(mol.GetSubstructMatches(p_c_no2)) if p_c_no2 else 0
    nC_NO2_2 = len(mol.GetSubstructMatches(p_c_no2_2)) if p_c_no2_2 else 0
    nC_NO2_3 = len(mol.GetSubstructMatches(p_c_no2_3)) if p_c_no2_3 else 0
    nNNO2 = len(mol.GetSubstructMatches(p_n_no2)) if p_n_no2 else 0
    nONO2 = len(mol.GetSubstructMatches(p_ono2)) if p_ono2 else 0
    
    # Aromatic rings
    nACC = rdMolDescriptors.CalcNumAromaticRings(mol)
    nAHC = sum(1 for r in mol.GetRingInfo().AtomRings() if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in r) and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))
    nHC = sum(1 for r in mol.GetRingInfo().AtomRings() if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in r))

    # Fast 2D approximations for PMI and volume if 3D embedding fails
    pmi1, pmi2, pmi3, npr1, npr2, ecc, pbf, mol_vol, total_e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Descriptors.LabuteASA(mol), 0.0
    
    return {
        'Eccentricity': ecc,
        'MaxPartialCharge': max_q,
        'MinPartialCharge': min_q,
        'Molecular weight': mw,
        'MOLvolume': mol_vol,
        'nACC': nACC,
        'nAHC': nAHC,
        'nC': nC,
        'nC(NO2)': nC_NO2,
        'nC(NO2)2': nC_NO2_2,
        'nC(NO2)3': nC_NO2_3,
        'nCH3': nCH3,
        'nH': nH,
        'nHbondA': nHbondA,
        'nHbondD': nHbondD,
        'nHC': nHC,
        'nN': nN,
        'nNH2': nNH2,
        'nNNO2': nNNO2,
        'nNO2': nNO2,
        'nO': nO,
        'nOCH3': nOCH3,
        'nONO2': nONO2,
        'NPR1': npr1,
        'NPR2': npr2,
        'nR': nR,
        'nRbond': nRbond,
        'ob': ob,
        'PBF': pbf,
        'PMI1': pmi1,
        'PMI2': pmi2,
        'PMI3': pmi3,
        'Total energy': total_e,
        'TPSA': tpsa
    }

def analyze_wudily_cho_features(excel_path):
    print(f"Analyzing Table S1 descriptors for WUDILY-CHO dataset...")
    df_raw = pd.read_excel(excel_path)
    smiles_list = df_raw['Canonical_SMILES'].tolist()
    y_h298 = df_raw['Hf_298.15K_kcal/mol'].values
    
    sample_smiles = smiles_list[:1000]
    sample_y = y_h298[:1000]
    
    records = []
    for smi in sample_smiles:
        records.append(compute_table_s1_descriptors_single(smi))
        
    df_feat = pd.DataFrame(records)
    
    print("\n--- 1. ZERO VARIANCE / CONSTANT FEATURE AUDIT ---")
    stds = df_feat.std()
    zero_var_cols = stds[stds < 1e-8].index.tolist()
    print(f"Zero Variance Columns ({len(zero_var_cols)}):", zero_var_cols)
    
    print("\n--- 2. PAIRWISE CORRELATION AUDIT (> 0.90) ---")
    corr_matrix = df_feat.corr().abs()
    high_corr_pairs = []
    cols = df_feat.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr_matrix.loc[c1, c2]
            if val > 0.90:
                high_corr_pairs.append((c1, c2, val))
                
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"Found {len(high_corr_pairs)} pairs with correlation > 0.90:")
    for c1, c2, val in high_corr_pairs:
        print(f"  {c1:20s} <--> {c2:20s} : {val:.4f}")

    print("\n--- 3. CORRELATION WITH TARGET (Hf_298) ---")
    target_corrs = {}
    for col in df_feat.columns:
        if col not in zero_var_cols:
            c = np.corrcoef(df_feat[col].values, sample_y)[0, 1]
            target_corrs[col] = c if np.isfinite(c) else 0.0
            
    df_target_corr = pd.Series(target_corrs).sort_values(key=abs, ascending=False)
    print("\nTop Features Correlated with Hf_298:")
    for name, val in df_target_corr.items():
        print(f"  {name:20s} : {val:+.4f}")
    
    return df_feat, zero_var_cols, high_corr_pairs, df_target_corr

if __name__ == '__main__':
    excel_path = "/home/nvsinh1/brats_segmentation/teb-net-model/data/WUDILY-CHO/wudily-cho.xlsx"
    analyze_wudily_cho_features(excel_path)
