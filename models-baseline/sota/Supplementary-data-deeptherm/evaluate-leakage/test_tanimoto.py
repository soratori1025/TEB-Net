import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
df = pd.read_csv("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv")
rng = np.random.default_rng(42)
indices = np.arange(len(df))
rng.shuffle(indices)
split_point = int(len(df) * 0.9)
train_idx, test_idx = indices[:split_point], indices[split_point:]
train_smiles = df.iloc[train_idx]['smiles'].tolist()
test_smiles = df.iloc[test_idx]['smiles'].tolist()
train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols]
test_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in test_mols]
max_sims = []
for test_fp in test_fps:
    sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    max_sims.append(max(sims))
sims = np.array(max_sims)
print("Mean Max-Tanimoto:", np.mean(sims))
print("% > 0.90:", np.mean(sims > 0.90) * 100)
print("% > 0.95:", np.mean(sims > 0.95) * 100)
