import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
class ECFP4Extractor:
    """
    ECFP4 (Extended Connectivity Fingerprints) - 1024-bit binary vector
    Radius = 2 (ECFP4), nBits = 1024
    Uses joblib for parallel extraction.
    """
    def __init__(self, radius=2, n_bits=1024, n_jobs=-1):
        self.radius = radius
        self.n_bits = n_bits
        self.n_jobs = n_jobs
    def transform_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = np.zeros((0,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    def transform_smiles_list(self, smiles_list):
        X = Parallel(n_jobs=self.n_jobs, batch_size=256)(
            delayed(self.transform_smiles)(smi) for smi in smiles_list
        )
        return np.stack(X)
