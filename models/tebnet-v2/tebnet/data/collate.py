from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from tebnet.data.featurizer import CleanBondFeaturizer
@dataclass
class ThermoBatch:
    bmg: BatchMolGraph
    x_physic: Tensor | None
    cp_grid: Tensor
    dh_f_298: Tensor
    s_298: Tensor
    weights: Tensor
    T_grid: Tensor
    smiles: list[str]
    comp: Tensor | None = None  
    def __len__(self) -> int:
        return len(self.smiles)
    def to(self, device: torch.device) -> "ThermoBatch":
        if hasattr(self.bmg, "to"):
            self.bmg.to(device)  
        return ThermoBatch(
            bmg=self.bmg,
            x_physic=self.x_physic.to(device) if self.x_physic is not None else None,
            cp_grid=self.cp_grid.to(device),
            dh_f_298=self.dh_f_298.to(device),
            s_298=self.s_298.to(device),
            weights=self.weights.to(device),
            T_grid=self.T_grid.to(device),
            smiles=self.smiles,
            comp=self.comp.to(device) if self.comp is not None else None,
        )
class ThermoCollater:
    def __init__(self, mol_graph_featurizer: SimpleMoleculeMolGraphFeaturizer | None = None, atom_featurizer=None, bond_featurizer=None):
        if mol_graph_featurizer is None:
            if bond_featurizer is None:
                bond_featurizer = CleanBondFeaturizer()
            if atom_featurizer is not None:
                mol_graph_featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer,)
            else:
                mol_graph_featurizer = SimpleMoleculeMolGraphFeaturizer(bond_featurizer=bond_featurizer,)
        self.mol_graph_featurizer = mol_graph_featurizer
    def __call__(self, batch: list[dict]) -> ThermoBatch:
        mols = [item["mol"] for item in batch]
        mol_graphs = [self.mol_graph_featurizer(mol) for mol in mols]
        bmg = BatchMolGraph(mol_graphs)
        cp_grid = torch.stack([item["cp_grid"] for item in batch])
        dh_f_298 = torch.stack([item["dh_f_298"] for item in batch])
        s_298 = torch.stack([item["s_298"] for item in batch])
        weights = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
        T_grid = batch[0]["T_grid"]
        x_physic_list = [item["x_physic"] for item in batch]
        if all(x is not None for x in x_physic_list):
            x_physic = torch.stack(x_physic_list)
        else:
            x_physic = None
        comp_list = [item.get("comp") for item in batch]
        if all(c is not None for c in comp_list):
            comp = torch.stack(comp_list)
        else:
            comp = None
        smiles = [item["smiles"] for item in batch]
        return ThermoBatch(
            bmg=bmg,
            x_physic=x_physic,
            cp_grid=cp_grid,
            dh_f_298=dh_f_298,
            s_298=s_298,
            weights=weights,
            T_grid=T_grid,
            smiles=smiles,
            comp=comp,
        )
