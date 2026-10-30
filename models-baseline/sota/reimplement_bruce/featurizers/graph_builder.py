import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding
def atom_to_feature_vector(atom):
    atomic_num_choices = [6, 1, 8, 7, 9, 16, 17, 35, 53, 'Other']
    degree_choices = [0, 1, 2, 3, 4, 5, 'Other']
    hybridization_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'Other'
    ]
    feats = []
    feats.extend(one_hot_encoding(atom.GetAtomicNum(), atomic_num_choices))
    feats.extend(one_hot_encoding(atom.GetDegree(), degree_choices))
    feats.extend([atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), 1.0 if atom.GetIsAromatic() else 0.0])
    feats.extend(one_hot_encoding(atom.GetHybridization(), hybridization_choices))
    return np.array(feats, dtype=np.float32)
def bond_to_feature_vector(bond):
    bond_type_choices = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'Other'
    ]
    feats = []
    feats.extend(one_hot_encoding(bond.GetBondType(), bond_type_choices))
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    stereo_choices = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
    feats.extend(one_hot_encoding(bond.GetStereo(), stereo_choices))
    return np.array(feats[:10], dtype=np.float32)
class MoleculeGraphBuilder:
    """
    Converts SMILES into PyTorch Geometric Data object.
    Node features: atom attributes
    Edge features: bond attributes (edge_dim=10)
    """
    def __init__(self):
        pass
    def smiles_to_graph(self, smiles, target=None):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        node_feats = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(node_feats), dtype=torch.float)
        edge_indices = []
        edge_feats = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_to_feature_vector(bond)
            edge_indices.append([i, j])
            edge_feats.append(bf)
            edge_indices.append([j, i])
            edge_feats.append(bf)
        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)
        y = torch.tensor([target], dtype=torch.float) if target is not None else None
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
        return data
if __name__ == '__main__':
    builder = MoleculeGraphBuilder()
    graph = builder.smiles_to_graph('CCO', target=10.5)
    print(f"Graph x shape: {graph.x.shape}, edge_index shape: {graph.edge_index.shape}, edge_attr shape: {graph.edge_attr.shape}")
