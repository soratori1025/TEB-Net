import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
class MordredExtractor:
    """
    Mordred 2D Descriptors Extractor:
    Computes all 2D descriptors using mordred.
    Filters out missing/error values and drops highly correlated features (> 0.95).
    """
    def __init__(self, corr_threshold=0.95):
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.corr_threshold = corr_threshold
        self.valid_feature_mask = None
    def fit_transform_smiles_list(self, smiles_list):
        print(f"Calculating Mordred 2D descriptors for {len(smiles_list)} molecules...")
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        df = self.calc.pandas(mols, quiet=True)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        valid_cols = df_numeric.columns[df_numeric.isnull().mean() < 0.10].tolist()
        df_cleaned = df_numeric[valid_cols].fillna(0.0)
        variances = df_cleaned.var()
        non_constant_cols = variances[variances > 1e-10].index.tolist()
        df_cleaned = df_cleaned[non_constant_cols]
        corr_matrix = df_cleaned.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        selected_cols = [c for c in df_cleaned.columns if c not in to_drop]
        self.selected_cols = selected_cols
        print(f"Mordred extraction complete. Selected {len(selected_cols)} clean 2D descriptors.")
        return df_cleaned[selected_cols].values.astype(np.float32)
    def transform_smiles_list(self, smiles_list):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        df = self.calc.pandas(mols, quiet=True)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        df_cleaned = df_numeric.reindex(columns=self.selected_cols).fillna(0.0)
        return df_cleaned.values.astype(np.float32)
if __name__ == '__main__':
    extractor = MordredExtractor()
    X = extractor.fit_transform_smiles_list(['CCO', 'CC(=O)O', 'CCCC'])
    print(f"Mordred feature matrix shape: {X.shape}")
