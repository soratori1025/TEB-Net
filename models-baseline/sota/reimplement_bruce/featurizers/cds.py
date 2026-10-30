import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from joblib import Parallel, delayed
class CustomDescriptorSetExtractor:
    """
    CDS (Custom Descriptor Set) Featurizer - 104 bits:
    - 34 Domain-knowledge physicochemical/functional group features (including UFF Energy and 3D shapes)
    - 70 Electro-topological and 2D descriptors from RDKit
    Uses joblib parallel processing.
    """
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.smarts_patterns = {
            'nC(NO2)': Chem.MolFromSmarts('[CX4](-[NX3](=O)=O)'),
            'nC(NO2)2': Chem.MolFromSmarts('[CX4](-[NX3](=O)=O)(-[NX3](=O)=O)'),
            'nC(NO2)3': Chem.MolFromSmarts('[CX4](-[NX3](=O)=O)(-[NX3](=O)=O)(-[NX3](=O)=O)'),
            'nCH3': Chem.MolFromSmarts('[CX4H3]'),
            'nNH2': Chem.MolFromSmarts('[NX3H2]'),
            'nNNO2': Chem.MolFromSmarts('[NX3]-[NX3](=O)=O'),
            'nNO2': Chem.MolFromSmarts('[NX3](=O)=O'),
            'nOCH3': Chem.MolFromSmarts('[OX2]-[CX4H3]'),
            'nONO2': Chem.MolFromSmarts('[OX2]-[NX3](=O)=O')
        }
        self.rdkit_desc_names = [name for name, _ in Descriptors.descList]
        preferred = [d for d in self.rdkit_desc_names if 'EState' in d or 'VSA' in d or 'Chi' in d or 'HallKier' in d or 'Kappa' in d]
        remaining = [d for d in self.rdkit_desc_names if d not in preferred]
        self.selected_rdkit_70 = (preferred + remaining)[:70]
    def _extract_single(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(104, dtype=np.float32)
        mol_h = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol_h, randomSeed=42)
            ff = AllChem.UFFGetMoleculeForceField(mol_h)
            uff_energy = ff.CalcEnergy() if ff is not None else 0.0
            pmi1 = rdMolDescriptors.CalcPMI1(mol_h)
            pmi2 = rdMolDescriptors.CalcPMI2(mol_h)
            pmi3 = rdMolDescriptors.CalcPMI3(mol_h)
            npr1 = rdMolDescriptors.CalcNPR1(mol_h)
            npr2 = rdMolDescriptors.CalcNPR2(mol_h)
            pbf = rdMolDescriptors.CalcPBF(mol_h)
            eccentricity = rdMolDescriptors.CalcEccentricity(mol_h)
            mol_volume = AllChem.ComputeMolVolume(mol_h)
        except:
            uff_energy = 0.0
            pmi1, pmi2, pmi3 = 0.0, 0.0, 0.0
            npr1, npr2 = 0.0, 0.0
            pbf = 0.0
            eccentricity = 0.0
            mol_volume = 0.0
        mw = Descriptors.MolWt(mol)
        num_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        num_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
        num_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        num_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms() if a.HasProp('_GasteigerCharge')]
            charges = [c for c in charges if np.isfinite(c)]
            max_q = max(charges) if charges else 0.0
            min_q = min(charges) if charges else 0.0
        except:
            max_q, min_q = 0.0, 0.0
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_h_donors = rdMolDescriptors.CalcNumHBD(mol)
        n_h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        n_acc = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
        n_ahc = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        n_hc = rdMolDescriptors.CalcNumHeterocycles(mol)
        ob = -16.0 / max(1.0, mw) * (2.0 * num_c + 0.5 * num_h - num_o)
        smarts_counts = {}
        for name, pattern in self.smarts_patterns.items():
            if pattern is not None:
                smarts_counts[name] = len(mol.GetSubstructMatches(pattern))
            else:
                smarts_counts[name] = 0
        f34 = [
            eccentricity, max_q, min_q, mw, mol_volume,
            n_acc, n_ahc, num_c, smarts_counts['nC(NO2)'], smarts_counts['nC(NO2)2'],
            smarts_counts['nC(NO2)3'], smarts_counts['nCH3'], num_h, n_h_acceptors, n_h_donors,
            n_hc, num_n, smarts_counts['nNH2'], smarts_counts['nNNO2'], smarts_counts['nNO2'],
            num_o, smarts_counts['nOCH3'], smarts_counts['nONO2'], npr1, npr2,
            n_rings, n_rotatable, ob, pbf, pmi1, pmi2, pmi3, uff_energy, tpsa
        ]
        desc_dict = dict(Descriptors.descList)
        f70 = []
        for name in self.selected_rdkit_70:
            try:
                val = desc_dict[name](mol)
                f70.append(val if np.isfinite(val) else 0.0)
            except:
                f70.append(0.0)
        vector = np.array(f34 + f70, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    def transform_smiles_list(self, smiles_list):
        X = Parallel(n_jobs=self.n_jobs, batch_size=128)(
            delayed(self._extract_single)(smi) for smi in smiles_list
        )
        return np.stack(X)
