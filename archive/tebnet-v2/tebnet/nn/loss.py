from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
class NASA7ThermoLoss(nn.Module):
    def __init__(
        self,
        w_cp: float = 1.0,
        w_h298: float = 2.0,
        w_s298: float = 1.5,
        loss_type: str = "huber",
        weight_strategy: str = "fixed",
        cp_scale: float = 1.0,
        h_scale: float = 1.0,
        s_scale: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight_strategy = weight_strategy
        self.register_buffer("cp_scale", torch.tensor(float(cp_scale)))
        self.register_buffer("h_scale", torch.tensor(float(h_scale)))
        self.register_buffer("s_scale", torch.tensor(float(s_scale)))
        if weight_strategy == "uncertainty":
            self.log_var_cp = nn.Parameter(torch.zeros(1))
            self.log_var_h298 = nn.Parameter(torch.zeros(1))
            self.log_var_s298 = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("w_cp", torch.tensor(float(w_cp)))
            self.register_buffer("w_h298", torch.tensor(float(w_h298)))
            self.register_buffer("w_s298", torch.tensor(float(w_s298)))
    def _elementwise(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "mae":
            return (pred - target).abs()
        if self.loss_type == "mse":
            return (pred - target) ** 2
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unknown loss type: {self.loss_type}")
    def _weighted_mean(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        sample_w: Tensor | None = None,
    ) -> Tensor:
        err = self._elementwise(pred, target)
        if sample_w is None:
            w = torch.ones_like(err)
        elif err.dim() == 2:
            w = sample_w.unsqueeze(1).expand_as(err)
        else:
            w = sample_w.expand_as(err)
        if mask is not None:
            err = err[mask]
            w = w[mask]
        if err.numel() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)
        return (w * err).sum() / w.sum().clamp(min=1e-8)
    def forward(
        self,
        pred_cp: Tensor,
        target_cp: Tensor,
        pred_dh_f: Tensor,
        target_dh_f: Tensor,
        pred_s298: Tensor,
        target_s298: Tensor,
        sample_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        cp_mask = target_cp.isfinite()
        h_mask = target_dh_f.isfinite()
        s_mask = target_s298.isfinite()
        l_cp = self._weighted_mean(
            pred_cp / self.cp_scale, target_cp / self.cp_scale, cp_mask, sample_weights
        )
        l_h298 = self._weighted_mean(
            pred_dh_f / self.h_scale, target_dh_f / self.h_scale, h_mask, sample_weights
        )
        l_s298 = self._weighted_mean(
            pred_s298 / self.s_scale, target_s298 / self.s_scale, s_mask, sample_weights
        )
        if self.weight_strategy == "uncertainty":
            prec_cp = torch.exp(-self.log_var_cp)
            prec_h = torch.exp(-self.log_var_h298)
            prec_s = torch.exp(-self.log_var_s298)
            total = (
                0.5 * prec_cp * l_cp + 0.5 * self.log_var_cp
                + 0.5 * prec_h * l_h298 + 0.5 * self.log_var_h298
                + 0.5 * prec_s * l_s298 + 0.5 * self.log_var_s298
            ).squeeze()
        else:
            total = self.w_cp * l_cp + self.w_h298 * l_h298 + self.w_s298 * l_s298
        return {
            "total": total,
            "cp": l_cp.detach(),
            "h298": l_h298.detach(),
            "s298": l_s298.detach(),
        }
