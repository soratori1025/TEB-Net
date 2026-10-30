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
@dataclass
class HomologPairBatch:
    """A batch of CH2-homolog pairs: two ordinary batches plus the pair's site class.

    ``anchor`` holds the shorter molecule, ``partner`` the longer one, so the increment
    the model should reproduce is ``pred(partner) - pred(anchor)``.  ``cls_idx`` indexes
    the increment/tolerance rows of :class:`tebnet.data.homologs.HomologPairTable`;
    ``increments`` / ``tolerances`` are those rows already gathered per pair, ``(B, 3)``
    in the order ``(dHf, dS, dCp)`` and in SI units.  Carrying them in the batch (instead
    of registering them on the model) keeps the prior out of the checkpoint: the table is
    a property of the training split, and a checkpoint must not silently carry one.
    """
    anchor: ThermoBatch
    partner: ThermoBatch
    cls_idx: Tensor
    increments: Tensor | None = None
    tolerances: Tensor | None = None
    def __len__(self) -> int:
        return len(self.cls_idx)
    def to(self, device: torch.device) -> "HomologPairBatch":
        return HomologPairBatch(
            anchor=self.anchor.to(device),
            partner=self.partner.to(device),
            cls_idx=self.cls_idx.to(device),
            increments=self.increments.to(device) if self.increments is not None else None,
            tolerances=self.tolerances.to(device) if self.tolerances is not None else None,
        )
class HomologPairCollater:
    """Collate ``{"anchor": item, "partner": item, "cls_idx": int}`` dicts.

    Wraps the ordinary :class:`ThermoCollater`, so the molecular graphs come from the same
    (cached) featuriser as the main loader.  Pass ``table`` (the
    :class:`tebnet.data.homologs.HomologPairTable` the pairs were built from) to attach
    each pair's target increment and tolerance to the batch.
    """
    def __init__(self, base: ThermoCollater | None = None, table=None):
        self.base = base if base is not None else ThermoCollater()
        self.increments = None
        self.tolerances = None
        if table is not None:
            self.increments = torch.as_tensor(table.increments, dtype=torch.float32)
            self.tolerances = torch.as_tensor(table.tolerances, dtype=torch.float32)
    def __call__(self, batch: list[dict]) -> HomologPairBatch:
        cls_idx = torch.tensor([item["cls_idx"] for item in batch], dtype=torch.long)
        increments = self.increments[cls_idx] if self.increments is not None else None
        tolerances = self.tolerances[cls_idx] if self.tolerances is not None else None
        return HomologPairBatch(
            anchor=self.base([item["anchor"] for item in batch]),
            partner=self.base([item["partner"] for item in batch]),
            cls_idx=cls_idx,
            increments=increments,
            tolerances=tolerances,
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
        self._graph_cache: dict[str, object] = {}
    def _graph(self, item: dict):
        key = item["smiles"]
        g = self._graph_cache.get(key)
        if g is None:
            g = self.mol_graph_featurizer(item["mol"])
            self._graph_cache[key] = g
        return g
    def __call__(self, batch: list[dict]) -> ThermoBatch:
        mol_graphs = [self._graph(item) for item in batch]
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
