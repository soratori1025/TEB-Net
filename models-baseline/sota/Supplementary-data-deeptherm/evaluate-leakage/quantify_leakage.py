import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Tuple
def generate_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else "acyclic"
    except:
        return "invalid"
def create_butina_split(smiles_list: List[str], cutoff=0.4, train_frac=0.9, seed=42) -> Tuple[List[int], List[int]]:
    from rdkit.ML.Cluster import Butina
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_idx.append(i)
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    train_idx, test_idx = [], []
    rng = np.random.default_rng(seed)
    for cluster in clusters:
        cluster = list(cluster)
        rng.shuffle(cluster)
        orig_cluster = [valid_idx[i] for i in cluster]
        if len(train_idx) < int(train_frac * nfps):
            train_idx.extend(orig_cluster)
        else:
            test_idx.extend(orig_cluster)
    return train_idx, test_idx
def create_random_split(n, train_frac=0.9, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    split_point = int(n * train_frac)
    return indices[:split_point].tolist(), indices[split_point:].tolist()
def analyze_true_leakage(df, train_idx, test_idx, split_name="Random"):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_mols = [Chem.MolFromSmiles(s) for s in train_df['smiles']]
    test_mols = [Chem.MolFromSmiles(s) for s in test_df['smiles']]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in test_mols]
    max_sims = []
    closest_train_idx = []
    for test_fp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_idx = np.argmax(sims)
        max_sims.append(sims[max_idx])
        closest_train_idx.append(max_idx)
    sims = np.array(max_sims)
    pct_90 = np.mean(sims > 0.90) * 100
    pct_95 = np.mean(sims > 0.95) * 100
    mean_sim = np.mean(sims)
    test_h298 = test_df['Hf_298.15K_kcal/mol'].values
    train_h298_all = train_df['Hf_298.15K_kcal/mol'].values
    closest_train_h298 = np.array([train_h298_all[idx] for idx in closest_train_idx])
    delta_h = np.abs(test_h298 - closest_train_h298)
    near_dup_mask = sims > 0.90
    if np.sum(near_dup_mask) > 0:
        mean_delta_h_near_dup = np.mean(delta_h[near_dup_mask])
    else:
        mean_delta_h_near_dup = np.nan
    train_scaffolds = set(train_df[train_df['murcko'] != 'acyclic']['murcko'].tolist())
    test_scaffolds = test_df[test_df['murcko'] != 'acyclic']['murcko'].tolist()
    if len(test_scaffolds) > 0:
        overlap_scaffolds = [s for s in test_scaffolds if s in train_scaffolds]
        pct_scaffold_overlap = len(overlap_scaffolds) / len(test_scaffolds) * 100
    else:
        pct_scaffold_overlap = 0.0
    print(f"\n[{split_name} Split] Leakage Report")
    print("-" * 60)
    print(f"Mean Max-Tanimoto: {mean_sim:.4f}")
    print(f"% Test with Max-Tanimoto > 0.90 (Near-Duplicates): {pct_90:.2f}%")
    print(f"% Test with Max-Tanimoto > 0.95: {pct_95:.2f}%")
    if not np.isnan(mean_delta_h_near_dup):
        print(f"Mean |ΔH298| for Near-Duplicates (>0.90): {mean_delta_h_near_dup:.4f} kcal/mol")
    print(f"Mean |ΔH298| for all Test vs Closest Train: {np.mean(delta_h):.4f} kcal/mol")
    print(f"% Cyclic Test Mols with leaked Scaffold: {pct_scaffold_overlap:.2f}%")
    return sims
if __name__ == "__main__":
    df = pd.read_csv("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv")
    df['murcko'] = df['smiles'].apply(generate_murcko_scaffold)
    print("--- 1. Random Split ---")
    r_train, r_test = create_random_split(len(df), train_frac=0.9, seed=42)
    r_sims = analyze_true_leakage(df, r_train, r_test, "Random")
    print("\n--- 2. Butina Splits ---")
    b4_train, b4_test = create_butina_split(df['smiles'].tolist(), cutoff=0.4, train_frac=0.9, seed=42)
    b4_sims = analyze_true_leakage(df, b4_train, b4_test, "Butina (0.4)")
    b6_train, b6_test = create_butina_split(df['smiles'].tolist(), cutoff=0.6, train_frac=0.9, seed=42)
    b6_sims = analyze_true_leakage(df, b6_train, b6_test, "Butina (0.6)")
    plt.figure(figsize=(10, 6))
    plt.hist(r_sims, bins=50, alpha=0.4, label='Random Split', density=True)
    plt.hist(b6_sims, bins=50, alpha=0.4, label='Butina (0.6)', density=True)
    plt.hist(b4_sims, bins=50, alpha=0.4, label='Butina (0.4)', density=True)
    plt.title('Max Tanimoto Similarity Distribution')
    plt.xlabel('Max Tanimoto to Train Set')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/leakage_histogram_v2.png", dpi=300, bbox_inches='tight')
