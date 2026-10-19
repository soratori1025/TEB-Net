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
        huber_beta_cp: float = 1.0,
        huber_beta_h: float = 1.0,
        huber_beta_s: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight_strategy = weight_strategy
        # v3: Huber transition point per property, in the SCALED units (target / *_scale). The default 1.0
        # reproduces v1; with a global-std scale (~40 kcal/mol for ΔfH) beta = 1 makes the loss pure MSE.
        self.betas = {"cp": float(huber_beta_cp), "h": float(huber_beta_h), "s": float(huber_beta_s)}
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

    def _elementwise(self, pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
        if self.loss_type == "mae":
            return (pred - target).abs()
        if self.loss_type == "mse":
            return (pred - target) ** 2
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _weighted_mean(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        sample_w: Tensor | None = None,
        beta: float = 1.0,
    ) -> Tensor:
        err = self._elementwise(pred, target, beta)
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
        # Normalize by std to put all properties on comparable scale
        l_cp = self._weighted_mean(
            pred_cp / self.cp_scale, target_cp / self.cp_scale, cp_mask, sample_weights, self.betas["cp"]
        )
        l_h298 = self._weighted_mean(
            pred_dh_f / self.h_scale, target_dh_f / self.h_scale, h_mask, sample_weights, self.betas["h"]
        )
        l_s298 = self._weighted_mean(
            pred_s298 / self.s_scale, target_s298 / self.s_scale, s_mask, sample_weights, self.betas["s"]
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


class HomologConsistencyLoss(nn.Module):
    """CH2-homolog increment regulariser (the physics-informed part of the objective).

    Two species that differ by one CH2 group have thermochemistry that differs by a
    transferable increment.  A per-molecule regression never sees that: the two members of
    a pair are fitted independently, so nothing stops the model from giving a chain
    extension a wrong sign, the wrong magnitude, or a magnitude that drifts with chain
    length on extrapolation.  This term penalises exactly that.

    For a pair ``(anchor, partner)`` with ``partner = anchor + CH2`` the residual is

        r_p = [ pred_p(partner) - pred_p(anchor) - increment_p ] / tolerance_p

    for each property ``p`` in ``(dHf, S, Cp@T)``.  Both the increment and the tolerance
    come from the batch (:class:`tebnet.data.collate.HomologPairBatch`), i.e. from the
    *training split*: they are not model state, so a checkpoint cannot carry a prior.

    The penalty is a pseudo-Huber function of ``r``, ``2b^2 (sqrt(1 + (r/b)^2) - 1)``,
    which is ``r^2`` for small deviations and linear in ``|r|`` for large ones.  Squared
    error is right when the tolerance really is the scatter of the increment, but a
    handful of pairs in the table are chemically odd (a strained ring, a species whose
    tabulated value is off) and would otherwise set the gradient.  Because ``r`` is
    already in units of the measured within-class scatter, the three properties enter the
    sum with equal statistical weight and no extra scaling.

    Only Cp at ONE temperature is constrained: the increment table stores Cp at the
    benchmark temperature (300 K), and extrapolating that increment to the whole
    300-3000 K range would be inventing data.  ``cp_index`` selects the column.

    Parameters
    ----------
    w_h, w_s, w_cp
        Relative weight of the three properties (all in tolerance units).
    cp_index
        Index into the model's Cp grid at which the Cp increment is imposed.
    beta
        Transition point of the pseudo-Huber loss, in tolerance units.  ``beta=1`` means
        "quadratic while the model is within one measured scatter", which is the regime
        the prior is meant to police.
    """

    PROPERTIES = ("h", "s", "cp")

    def __init__(self, w_h: float = 1.0, w_s: float = 1.0, w_cp: float = 1.0,
                 cp_index: int = 0, beta: float = 1.0):
        super().__init__()
        # non-persistent: the property weights are hyperparameters, and keeping them out of
        # the state_dict means checkpoints written before this term existed still load.
        self.register_buffer("w", torch.tensor([float(w_h), float(w_s), float(w_cp)]),
                             persistent=False)
        self.cp_index = int(cp_index)
        self.beta = float(beta)

    @staticmethod
    def _pseudo_huber(r: Tensor, beta: float) -> Tensor:
        """``2 b^2 (sqrt(1 + (r/b)^2) - 1)``, written to avoid the catastrophic cancellation
        of ``sqrt(1 + a^2) - 1`` for small ``a``."""
        a2 = (r / beta) ** 2
        return 2.0 * beta * beta * a2 / ((1.0 + a2).sqrt() + 1.0)

    def forward(
        self,
        pred_anchor: dict[str, Tensor],
        pred_partner: dict[str, Tensor],
        increments: Tensor,
        tolerances: Tensor,
    ) -> dict[str, Tensor]:
        """Parameters
        ----------
        pred_anchor, pred_partner
            Model outputs for the shorter and the longer molecule, as returned by
            ``TEBNetV20.forward`` (``dh_f_298`` in J/mol, ``s_298`` in J/(mol K),
            ``cp`` on the temperature grid).
        increments, tolerances
            ``(B, 3)`` target increment and admissible deviation per pair, in the order
            ``(dHf, S, Cp)`` and in SI units (as carried by ``HomologPairBatch``).
        """
        if pred_anchor["cp"].shape[1] <= self.cp_index:
            raise ValueError(f"cp_index={self.cp_index} out of range for a Cp grid of "
                             f"{pred_anchor['cp'].shape[1]} points")
        d = torch.stack([
            pred_partner["dh_f_298"] - pred_anchor["dh_f_298"],
            pred_partner["s_298"] - pred_anchor["s_298"],
            pred_partner["cp"][:, self.cp_index] - pred_anchor["cp"][:, self.cp_index],
        ], dim=1)
        inc = increments.to(d.dtype)
        tol = tolerances.to(d.dtype).clamp_min(1e-8)
        resid = d - inc
        r = resid / tol
        pen = self._pseudo_huber(r, self.beta)              # (B, 3)
        # A pair is valid only when its increment, tolerance and prediction are all finite.
        # Masking has to go through torch.where, not multiplication: NaN * 0 is NaN, so a
        # single missing target would otherwise poison the sum and every gradient.
        valid = torch.isfinite(pen) & torch.isfinite(resid) & torch.isfinite(tol)
        zero = torch.zeros((), dtype=pen.dtype, device=pen.device)
        pen_m = torch.where(valid, pen, zero)
        resid_m = torch.where(valid, resid.abs(), zero)
        r_m = torch.where(valid, r.abs(), zero)
        n = valid.sum(0)                                    # per property
        denom = n.to(pen.dtype).clamp(min=1.0)
        per_prop = pen_m.sum(0) / denom                     # (3,): mean penalty per property
        # Combine with the (3,) property weights.  ``self.w`` must NOT be expanded to (B, 3)
        # here: ``per_prop`` is already a mean over the batch, so broadcasting a (B, 3) weight
        # against it would produce a (B, 3) tensor whose ``sum()`` counts every property B
        # times -- making the penalty (and its gradient) scale linearly with the batch size.
        total = (self.w.to(pen.dtype) * per_prop).sum()
        if not bool(valid.any()):
            # Keep the graph (and the optimiser) intact on a degenerate batch: the term is
            # zero, but it still has to be a differentiable zero, not a detached constant.
            total = torch.nan_to_num(d).sum() * 0.0
        out = {"total": total}
        for k, key in enumerate(self.PROPERTIES):
            out[key] = per_prop[k].detach()
            out[f"resid_{key}"] = (resid_m[:, k].sum() / denom[k]).detach()
            out[f"r_{key}"] = (r_m[:, k].sum() / denom[k]).detach()
        out["n"] = n.detach()
        return out
