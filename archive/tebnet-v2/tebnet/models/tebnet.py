"""TEB-Net v15.

Changes vs v14.4 (each one can be switched off from train_v15.py for the ablation table):
  1. Graph input: radical- and ring-aware atom features (ThermoAtomFeaturizerV3) —
     v14.4 silently used chemprop's default featurizer, which has no radical bit.
  2. Frozen linear baselines per property on NAMED physics columns
     (composition + bonds + ring sizes + radicals for H; + ln σ, ln(2S+1),
     chirality, rotors, inertia for S; Cp in Chebyshev space). The network
     learns residuals only.
  3. Cp is parameterised by Chebyshev coefficients on each NASA range and mapped
     EXACTLY to NASA-7 monomials (a = M c). Same function class and the same
     analytic NASA-7 layer as v14.4, but the output layer is well conditioned
     (cond ≈ 2 vs ≈ 4e3 for the scaled-monomial head) and starts at a
     physically sensible Cp instead of Cp ≈ 0.
  4. Feature normalisation is safe for features that are constant in the train
     split (std floor 1, clip ±10) — v14.4's clamp(1e-6) could multiply an unseen
     value by 1e6 on scaffold/size splits.
  5. Loss scaled by the RESIDUAL std of each property (after its baseline), so
     the three tasks start on the same footing.
"""
from __future__ import annotations
import math
import numpy as np
import lightning.pytorch as pl
import torch
from torch import Tensor, nn, optim
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import SumAggregation, MeanAggregation, NormAggregation
from chemprop.data import BatchMolGraph
from tebnet.nn.nasa_thermo import NASA7ThermodynamicLayer
from tebnet.nn.loss import NASA7ThermoLoss
from tebnet.data.collate import ThermoBatch
from tebnet.data.baselines import cheb_to_monomial_matrix, N_CHEB
def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, drop: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(max(n_layers - 2, 0)):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop)]
    layers += [nn.Linear(hidden, out_dim)]
    last = layers[-1]
    nn.init.xavier_uniform_(last.weight, gain=0.1) 
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers)
class TEBNetV20(pl.LightningModule):
    def __init__(
        self,
        feature_names: list[str],
        route_h: list[str],
        route_s: list[str],
        route_cp: list[str],
        base_h: list[str],
        base_s: list[str],
        base_cp: list[str],
        mp_hidden_dim: int = 300,
        mp_depth: int = 5,
        mp_dropout: float = 0.1,
        agg_type: str = "sum",
        batch_norm: bool = True,
        d_v: int | None = None,
        d_e: int | None = None,
        branch_hidden: int = 256,
        branch_layers: int = 2,
        branch_dropout: float = 0.1,
        cp_branch_hidden: int | None = None,
        cp_branch_dropout: float | None = None,
        t_ref: float = 298.15,
        t_mid: float = 1000.0,
        t_low_dom: float = 300.0,
        t_high_dom: float = 3000.0,
        cp_param: str = "chebyshev",
        loss_type: str = "huber",
        loss_weights: dict[str, float] | None = None,
        weight_strategy: str = "fixed",
        loss_scale_cp: float = 1.0,
        loss_scale_h: float = 1.0,
        loss_scale_s: float = 1.0,
        warmup_epochs: int = 3,
        init_lr: float = 1e-4,
        max_lr: float = 8e-4,
        final_lr: float = 1e-5,
        weight_decay: float = 5e-5,
        aux_clip: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        def _idx(names: list[str]) -> Tensor:
            missing = [n for n in names if n not in name_to_idx]
            if missing:
                raise ValueError(f"feature(s) {missing} not in feature_names")
            return torch.tensor([name_to_idx[n] for n in names], dtype=torch.long)
        for key, names in [("route_h", route_h), ("route_s", route_s), ("route_cp", route_cp),
                           ("base_h", base_h), ("base_s", base_s), ("base_cp", base_cp)]:
            self.register_buffer(f"{key}_idx", _idx(names), persistent=False)
        D = len(feature_names)
        self.aux_clip = aux_clip
        self.register_buffer("aux_mean", torch.zeros(D))
        self.register_buffer("aux_std", torch.ones(D))
        f64 = torch.float64
        self.register_buffer("base_h_w", torch.zeros(len(base_h), dtype=f64))
        self.register_buffer("base_h_b", torch.zeros((), dtype=f64))
        self.register_buffer("res_std_h", torch.ones((), dtype=f64))
        self.register_buffer("base_s_w", torch.zeros(len(base_s), dtype=f64))
        self.register_buffer("base_s_b", torch.zeros((), dtype=f64))
        self.register_buffer("res_std_s", torch.ones((), dtype=f64))
        self.register_buffer("base_cp_w", torch.zeros(len(base_cp), 2 * N_CHEB, dtype=f64))
        self.register_buffer("base_cp_b", torch.zeros(2 * N_CHEB, dtype=f64))
        self.register_buffer("cp_coef_std", torch.ones(2 * N_CHEB, dtype=f64))
        if cp_param == "chebyshev":
            M_low = cheb_to_monomial_matrix(t_low_dom, t_mid)
            M_high = cheb_to_monomial_matrix(t_mid, t_high_dom)
        elif cp_param == "monomial":
            M_low = M_high = np.diag([1.0, 1e-3, 1e-6, 1e-9, 1e-12])
        else:
            raise ValueError(f"cp_param must be 'chebyshev' or 'monomial', got {cp_param!r}")
        self.register_buffer("M_low", torch.tensor(M_low, dtype=f64), persistent=False)
        self.register_buffer("M_high", torch.tensor(M_high, dtype=f64), persistent=False)
        mp_kwargs = dict(d_h=mp_hidden_dim, depth=mp_depth, dropout=mp_dropout)
        if d_v is not None:
            mp_kwargs["d_v"] = d_v
        if d_e is not None:
            mp_kwargs["d_e"] = d_e
        self.message_passing = BondMessagePassing(**mp_kwargs)
        agg_map = {"sum": SumAggregation, "mean": MeanAggregation, "norm": NormAggregation}
        self.agg = agg_map.get(agg_type, SumAggregation)()
        self.bn = nn.BatchNorm1d(mp_hidden_dim) if batch_norm else nn.Identity()
        cp_h = cp_branch_hidden or branch_hidden
        cp_d = branch_dropout if cp_branch_dropout is None else cp_branch_dropout
        self.h_branch = _mlp(mp_hidden_dim + len(route_h), 1, branch_hidden, branch_layers, branch_dropout)
        self.s_branch = _mlp(mp_hidden_dim + len(route_s), 1, branch_hidden, branch_layers, branch_dropout)
        self.cp_branch = _mlp(mp_hidden_dim + len(route_cp), 2 * N_CHEB, cp_h, branch_layers, cp_d)
        self.thermo_layer = NASA7ThermodynamicLayer(t_ref=t_ref, t_mid=t_mid)
        lw = loss_weights or {"cp": 1.0, "h298": 2.0, "s298": 1.5}
        self.criterion = NASA7ThermoLoss(
            w_cp=lw.get("cp", 1.0), w_h298=lw.get("h298", 2.0), w_s298=lw.get("s298", 1.5),
            loss_type=loss_type, weight_strategy=weight_strategy,
            cp_scale=loss_scale_cp, h_scale=loss_scale_h, s_scale=loss_scale_s,
        )
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.weight_decay = weight_decay
    @torch.no_grad()
    def set_aux_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.aux_mean.copy_(torch.as_tensor(mean, dtype=self.aux_mean.dtype))
        self.aux_std.copy_(torch.as_tensor(std, dtype=self.aux_std.dtype))
    @torch.no_grad()
    def set_baselines(self, h: dict, s: dict, cp: dict) -> None:
        f64 = torch.float64
        self.base_h_w.copy_(torch.as_tensor(h["w"], dtype=f64))
        self.base_h_b.fill_(float(h["b"]))
        self.res_std_h.fill_(float(h["resid_std"]))
        self.base_s_w.copy_(torch.as_tensor(s["w"], dtype=f64))
        self.base_s_b.fill_(float(s["b"]))
        self.res_std_s.fill_(float(s["resid_std"]))
        self.base_cp_w.copy_(torch.as_tensor(cp["W"], dtype=f64))
        self.base_cp_b.copy_(torch.as_tensor(cp["b"], dtype=f64))
        self.cp_coef_std.copy_(torch.as_tensor(cp["coef_std"], dtype=f64))
    def encode(self, bmg: BatchMolGraph) -> Tensor:
        H_a = self.message_passing(bmg)
        H_m = self.agg(H_a, bmg.batch)
        return self.bn(H_m)
    def baselines(self, x_raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        xd = x_raw.double()
        h = xd.index_select(1, self.base_h_idx) @ self.base_h_w + self.base_h_b
        s = xd.index_select(1, self.base_s_idx) @ self.base_s_w + self.base_s_b
        c = xd.index_select(1, self.base_cp_idx) @ self.base_cp_w + self.base_cp_b
        return h, s, c
    def forward(self, batch: ThermoBatch) -> dict[str, Tensor]:
        if batch.x_physic is None:
            raise ValueError("TEBNetV15 needs x_physic (dataset with aux_feature_set='v15')")
        shared = self.encode(batch.bmg)
        x_raw = batch.x_physic
        xn = ((x_raw - self.aux_mean) / self.aux_std).clamp(-self.aux_clip, self.aux_clip)
        x_h = xn.index_select(1, self.route_h_idx)
        x_s = xn.index_select(1, self.route_s_idx)
        x_cp = xn.index_select(1, self.route_cp_idx)
        h_base, s_base, c_base = self.baselines(x_raw)
        raw_h = self.h_branch(torch.cat([shared, x_h], 1)).squeeze(-1).double()
        raw_s = self.s_branch(torch.cat([shared, x_s], 1)).squeeze(-1).double()
        raw_c = self.cp_branch(torch.cat([shared, x_cp], 1)).double()
        H298 = h_base + raw_h * self.res_std_h
        S298 = s_base + raw_s * self.res_std_s
        c = c_base + raw_c * self.cp_coef_std
        a1_5_low = c[:, :N_CHEB] @ self.M_low.T
        a1_5_high = c[:, N_CHEB:] @ self.M_high.T
        out = self.thermo_layer(a1_5_low, a1_5_high, H298, S298, batch.T_grid)
        return {k: v.float() for k, v in out.items()}
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, ThermoBatch):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
    @staticmethod
    def _nanmae(pred: Tensor, target: Tensor) -> Tensor | None:
        m = torch.isfinite(target)
        if m.sum() == 0:
            return None
        return (pred[m] - target[m]).abs().mean()
    def _compute_losses(self, batch, log_prefix, prog_bar=False, sample_weights=None):
        preds = self(batch)
        losses = self.criterion(
            pred_cp=preds["cp"], target_cp=batch.cp_grid,
            pred_dh_f=preds["dh_f_298"], target_dh_f=batch.dh_f_298,
            pred_s298=preds["s_298"], target_s298=batch.s_298,
            sample_weights=sample_weights,
        )
        bs = len(batch)
        self.log(f"{log_prefix}/loss_total", losses["total"], prog_bar=prog_bar, batch_size=bs)
        self.log(f"{log_prefix}/loss_cp", losses["cp"], batch_size=bs)
        self.log(f"{log_prefix}/loss_h298", losses["h298"], batch_size=bs)
        self.log(f"{log_prefix}/loss_s298", losses["s298"], batch_size=bs)
        return losses, preds
    def _log_maes(self, preds, batch, prefix):
        bs = len(batch)
        for key, p, t in [("cp", preds["cp"], batch.cp_grid),
                          ("h298", preds["dh_f_298"], batch.dh_f_298),
                          ("s298", preds["s_298"], batch.s_298)]:
            v = self._nanmae(p, t)
            if v is not None:
                self.log(f"{prefix}/mae_{key}", v, batch_size=bs)
    def training_step(self, batch: ThermoBatch, batch_idx: int) -> Tensor:
        losses, _ = self._compute_losses(batch, "train", prog_bar=True, sample_weights=batch.weights)
        return losses["total"]
    def validation_step(self, batch: ThermoBatch, batch_idx: int) -> None:
        _, preds = self._compute_losses(batch, "val", prog_bar=True, sample_weights=batch.weights)
        self._log_maes(preds, batch, "val")
    def test_step(self, batch: ThermoBatch, batch_idx: int) -> None:
        _, preds = self._compute_losses(batch, "test")
        self._log_maes(preds, batch, "test")
    def predict_step(self, batch: ThermoBatch, batch_idx: int) -> dict[str, Tensor]:
        return self(batch)
    def configure_optimizers(self) -> dict:
        optimizer = optim.AdamW(self.parameters(), lr=self.max_lr, weight_decay=self.weight_decay)
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except (AttributeError, RuntimeError):
            total_steps = 10000
        max_epochs = self.trainer.max_epochs if self.trainer else 300
        steps_per_epoch = max(1, total_steps // max(1, max_epochs))
        warmup_steps = self.warmup_epochs * steps_per_epoch
        start = self.init_lr / self.max_lr
        floor = self.final_lr / self.max_lr
        def lr_lambda(step):
            if step < warmup_steps:
                return start + (1.0 - start) * step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
