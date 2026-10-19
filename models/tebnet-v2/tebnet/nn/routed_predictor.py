from __future__ import annotations
import torch
from torch import Tensor, nn

# =============================================================================
# TEB-Net v14.4 — task-specific feature-routing predictor
#
# Replaces v14.2's single shared_proj + two trunks with THREE small, heavily
# regularised branches. Each branch receives the SHARED D-MPNN latent
# concatenated with only the property-specific global features it needs:
#
#     H branch  : [ z_shared ⊕ X_H  ]  -> ΔH_f,298 residual
#     S branch  : [ z_shared ⊕ X_S  ]  -> S°298 residual
#     Cp branch : [ z_shared ⊕ X_Cp ]  -> a1..a5 (low, high)  -> NASA thermo layer
#
# Everything downstream (composition baselines, the analytic NASA-7 thermo layer
# that derives a6/a7 and enforces continuity, the multi-task loss) is unchanged
# from v14.2, so v14.4 vs v14.2 isolates exactly one variable: shared trunk ->
# routed branches. Branches are deliberately small (hidden 128) and high-dropout
# (0.3): three heads on ~4500 molecules make overfitting the first risk.
# =============================================================================

class RoutedNASA7Predictor(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        n_h_feat: int,
        n_s_feat: int,
        n_cp_feat: int,
        branch_hidden: int = 128,
        branch_layers: int = 2,
        branch_dropout: float = 0.3,
        cp_branch_hidden: int = None,
        cp_branch_dropout: float = None,
        norm_mean_h: float = 0.0,
        norm_std_h: float = 1.0,
        norm_mean_s: float = 0.0,
        norm_std_s: float = 1.0,
    ):
        super().__init__()
        # Residual-mode scale/centre buffers (identical semantics to v14.2):
        # in composition-baseline mode std_* holds the residual std and mean_* is
        # unused (a per-sample baseline is supplied at forward time).
        self.register_buffer("mean_h", torch.tensor(norm_mean_h))
        self.register_buffer("std_h", torch.tensor(norm_std_h))
        self.register_buffer("mean_s", torch.tensor(norm_mean_s))
        self.register_buffer("std_s", torch.tensor(norm_std_s))
        if cp_branch_hidden is None:
            cp_branch_hidden = branch_hidden
        if cp_branch_dropout is None:
            cp_branch_dropout = branch_dropout
        def _mlp(in_dim: int, out_dim: int, hidden: int, drop: float) -> nn.Sequential:
            layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop)]
            for _ in range(max(branch_layers - 2, 0)):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop)]
            layers += [nn.Linear(hidden, out_dim)]
            return nn.Sequential(*layers)
        self.h_branch = _mlp(shared_dim + n_h_feat, 1, branch_hidden, branch_dropout)
        self.s_branch = _mlp(shared_dim + n_s_feat, 1, branch_hidden, branch_dropout)
        self.cp_branch = _mlp(shared_dim + n_cp_feat, 10, cp_branch_hidden, cp_branch_dropout)  # a1..a5 low + a1..a5 high
        # Small init on the NASA output layer (stability with the T^4 scaling).
        last = self.cp_branch[-1]
        nn.init.xavier_uniform_(last.weight, gain=0.1)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        shared: Tensor,           # (B, shared_dim)  D-MPNN latent
        x_h: Tensor,              # (B, n_h_feat)
        x_s: Tensor,              # (B, n_s_feat)
        x_cp: Tensor,             # (B, n_cp_feat)
        h_baseline: Tensor | None = None,
        s_baseline: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raw_dh_f = self.h_branch(torch.cat([shared, x_h], dim=1)).squeeze(-1)
        center_h = self.mean_h if h_baseline is None else h_baseline
        dh_f_298 = raw_dh_f * self.std_h + center_h
        raw_s_298 = self.s_branch(torch.cat([shared, x_s], dim=1)).squeeze(-1)
        center_s = self.mean_s if s_baseline is None else s_baseline
        s_298 = raw_s_298 * self.std_s + center_s
        a1_10 = self.cp_branch(torch.cat([shared, x_cp], dim=1))
        scales = torch.tensor([1.0, 1e-3, 1e-6, 1e-9, 1e-12], device=shared.device)
        a1_5_low = a1_10[:, :5] * scales
        a1_5_high = a1_10[:, 5:] * scales
        return {
            "a1_5_low": a1_5_low,
            "a1_5_high": a1_5_high,
            "dh_f_298": dh_f_298,
            "s_298": s_298,
        }