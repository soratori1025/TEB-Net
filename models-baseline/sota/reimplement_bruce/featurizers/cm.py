import numpy as np
from rdkit import Chem
from joblib import Parallel, delayed

class CoulombMatrixExtractor:
    """
    2D/3D Coulomb Matrix Extractor:
    Calculates electrostatic interaction matrix C_ij = Z_i * Z_j / r_ij (for i != j)
    and 0.5 * Z_i^2.4 on diagonal, sorted by row norm and padded/eigenvalues extracted.
    Uses joblib for parallel extraction.
    """
    def __init__(self, max_atoms=50, return_eigenvalues=True, n_jobs=-1):
        self.max_atoms = max_atoms
        self.return_eigenvalues = return_eigenvalues
        self.n_jobs = n_jobs

    def transform_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            dim = self.max_atoms if self.return_eigenvalues else self.max_atoms * self.max_atoms
            return np.zeros(dim, dtype=np.float32)
        
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()
        atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        
        cm = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        dist_matrix = Chem.GetDistanceMatrix(mol)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                zi = atomic_nums[i]
                zj = atomic_nums[j]
                if i == j:
                    cm[i, j] = 0.5 * (zi ** 2.4)
                else:
                    d = dist_matrix[i, j]
                    if d > 0:
                        cm[i, j] = (zi * zj) / d
                    else:
                        cm[i, j] = 0.0

        row_norms = np.linalg.norm(cm, axis=1)
        sort_idx = np.argsort(row_norms)[::-1]
        cm_sorted = cm[sort_idx][:, sort_idx]

        if self.return_eigenvalues:
            eigvals = np.linalg.eigvalsh(cm_sorted)[::-1]
            padded = np.zeros(self.max_atoms, dtype=np.float32)
            k = min(len(eigvals), self.max_atoms)
            padded[:k] = eigvals[:k]
            return padded
        else:
            padded = np.zeros((self.max_atoms, self.max_atoms), dtype=np.float32)
            k = min(n_atoms, self.max_atoms)
            padded[:k, :k] = cm_sorted[:k, :k]
            return padded.ravel()

    def transform_smiles_list(self, smiles_list):
        X = Parallel(n_jobs=self.n_jobs, batch_size=256)(
            delayed(self.transform_smiles)(smi) for smi in smiles_list
        )
        return np.stack(X)
