import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import json
import os

def create_butina_split(smiles_list, cutoff, train_frac=0.9, seed=42):
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

def generate_unified_splits():
    df = pd.read_csv("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv")
    smiles_list = df['smiles'].tolist()
    n = len(df)
    
    splits = {}
    
    for seed in [42]: # Start with 1 seed for testing, expand to 5 seeds later if needed
        # 1. Random Split
        r_train, r_test = create_random_split(n, seed=seed)
        
        # 2. Butina 0.4 Split
        b4_train, b4_test = create_butina_split(smiles_list, cutoff=0.4, seed=seed)
        
        # 3. Butina 0.6 Split
        b6_train, b6_test = create_butina_split(smiles_list, cutoff=0.6, seed=seed)
        
        splits[f"seed_{seed}"] = {
            "random": {"train": r_train, "test": r_test},
            "butina_0.4": {"train": b4_train, "test": b4_test},
            "butina_0.6": {"train": b6_train, "test": b6_test}
        }
    
    os.makedirs("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/splits", exist_ok=True)
    with open("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/splits/unified_splits.json", "w") as f:
        json.dump(splits, f)
    
    print("Unified splits generated and saved.")

if __name__ == "__main__":
    generate_unified_splits()
