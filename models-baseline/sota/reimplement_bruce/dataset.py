import os
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import KFold
DATASET_PATH = "/home/nvsinh1/brats_segmentation/teb-net-model/data/WUDILY-CHO/wudily-cho.xlsx"
class WudilyCHODataset:
    """
    Dataset loader for the WUDILY-CHO dataset (6,074 aliphatic C, H, O species)
    from Bruce et al. (2025).
    """
    def __init__(self, excel_path=DATASET_PATH):
        self.excel_path = excel_path
        self.df = None
        self.smiles_list = []
        self.targets = {}
        self._load_and_preprocess()
    def _load_and_preprocess(self):
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Dataset file not found at {self.excel_path}")
        print(f"Loading WUDILY-CHO dataset from {self.excel_path}...")
        self.df = pd.read_excel(self.excel_path)
        print(f"Loaded dataset with {len(self.df)} rows.")
        canonical_smiles = []
        valid_indices = []
        for idx, row in self.df.iterrows():
            smi = str(row['Canonical_SMILES']).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canon_smi)
                valid_indices.append(idx)
            else:
                print(f"Warning: Invalid SMILES at row {idx}: {smi}")
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        self.smiles_list = canonical_smiles
        self.targets['Hf_298'] = self.df['Hf_298.15K_kcal/mol'].values.astype(np.float32)
        if 'S_298.15K_cal/(molK)' in self.df.columns:
            self.targets['S_298'] = self.df['S_298.15K_cal/(molK)'].values.astype(np.float32)
        cp_cols = [c for c in self.df.columns if c.startswith('cp_')]
        if cp_cols:
            self.targets['Cp'] = self.df[cp_cols].values.astype(np.float32)
        print(f"Preprocessing completed. Valid molecules: {len(self.smiles_list)}")
    def get_kfold_splits(self, n_splits=10, seed=42):
        """
        Generates 10-fold cross validation split indices.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(len(self.smiles_list))))
if __name__ == '__main__':
    dataset = WudilyCHODataset()
    print("Dataset SMILES sample:", dataset.smiles_list[:3])
    print("Hf_298 target sample:", dataset.targets['Hf_298'][:3])
    splits = dataset.get_kfold_splits(n_splits=10)
    print(f"Generated {len(splits)} folds. Fold 1 train size: {len(splits[0][0])}, test size: {len(splits[0][1])}")
