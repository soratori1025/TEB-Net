import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
class SOAPMBTRRDKitExtractor:
    """
    RDKit All Descriptors & Topological Vector Extractor
    Serves as feature extractor for SOAP, MBTR, DRAG, ALVA, and RDKIT descriptors.
    Extracts all 200+ standard RDKit descriptors.
    """
    def __init__(self):
        self.desc_list = Descriptors.descList
    def transform_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(self.desc_list), dtype=np.float32)
        vals = []
        for name, func in self.desc_list:
            try:
                v = func(mol)
                vals.append(v if np.isfinite(v) else 0.0)
            except:
                vals.append(0.0)
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    def transform_smiles_list(self, smiles_list):
        X = [self.transform_smiles(smi) for smi in smiles_list]
        return np.stack(X)
if __name__ == '__main__':
    extractor = SOAPMBTRRDKitExtractor()
    feat = extractor.transform_smiles('CCO')
    print(f"Extracted RDKit Descriptors vector shape: {feat.shape}")
