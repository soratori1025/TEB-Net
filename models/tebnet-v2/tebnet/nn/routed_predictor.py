from __future__ import annotations
import torch
from torch import Tensor, nn
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
        self.cp_branch = _mlp(shared_dim + n_cp_feat, 10, cp_branch_hidden, cp_branch_dropout)
        last = self.cp_branch[-1]
        nn.init.xavier_uniform_(last.weight, gain=0.1)
        nn.init.zeros_(last.bias)
    def forward(
        self,
        shared: Tensor,
        x_h: Tensor,
        x_s: Tensor,
        x_cp: Tensor,
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
