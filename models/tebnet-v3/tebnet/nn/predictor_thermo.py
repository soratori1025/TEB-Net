from __future__ import annotations
import torch
from torch import Tensor, nn

class NASA7Predictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 2,
        dropout: float = 0.1,
        norm_mean_h: float = 0.0,
        norm_std_h: float = 1.0,
        norm_mean_s: float = 0.0,
        norm_std_s: float = 1.0,
        separate_trunks: bool = False,
    ):
        super().__init__()
        # In v14.1 composition-baseline mode: mean_h is unused (a per-sample
        # baseline is supplied at forward time) and std_h holds the RESIDUAL std
        # so the network output stays ~unit variance. In v14 mean mode these are
        # the global mean/std, exactly as before.
        self.register_buffer("mean_h", torch.tensor(norm_mean_h))
        self.register_buffer("std_h", torch.tensor(norm_std_h))
        self.register_buffer("mean_s", torch.tensor(norm_mean_s))
        self.register_buffer("std_s", torch.tensor(norm_std_s))
        # separate_trunks=True gives the scalar (H/S) FFN and the NASA-coefficient
        # FFN fully independent stacks (no shared first layer) — tests whether the
        # shared projection was bottlenecking the two objectives.
        self.separate_trunks = separate_trunks
        def _proj():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            )
        # ── Bottom projection(s) ──
        if separate_trunks:
            self.scalar_proj = _proj()
            self.nasa_proj = _proj()
        else:
            self.shared_proj = _proj()
        # ── Pathway 1: Scalar properties (H298, S298) ──
        scalar_layers = []
        for _ in range(n_layers - 1):
            scalar_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.trunk_scalar = nn.Sequential(*scalar_layers) if scalar_layers else nn.Identity()
        self.dh_f_head = nn.Linear(hidden_dim, 1)
        self.s_298_head = nn.Linear(hidden_dim, 1)
        # ── Pathway 2: NASA polynomial coefficients (a1-a5 for low and high) ──
        nasa_layers = []
        for _ in range(n_layers - 1):
            nasa_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.trunk_nasa = nn.Sequential(*nasa_layers) if nasa_layers else nn.Identity()
        self.nasa_head = nn.Linear(hidden_dim, 10)
        # Initialize NASA parameters with small weights for stability
        nn.init.xavier_uniform_(self.nasa_head.weight, gain=0.1)
        nn.init.zeros_(self.nasa_head.bias)

    def forward(
        self,
        fingerprint: Tensor,
        h_baseline: Tensor | None = None,
        s_baseline: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Bottom projection: shared, or one per pathway
        if self.separate_trunks:
            h_scalar_in = self.scalar_proj(fingerprint)
            h_nasa_in = self.nasa_proj(fingerprint)
        else:
            h_shared = self.shared_proj(fingerprint)
            h_scalar_in = h_nasa_in = h_shared
        # Pathway 1: Scalar properties
        h_scalar = self.trunk_scalar(h_scalar_in)
        raw_dh_f = self.dh_f_head(h_scalar).squeeze(-1)
        # H298 = raw * scale + center.
        #   composition mode : center = per-sample baseline, scale = residual std
        #   mean mode (v14)  : center = global mean,         scale = global std
        center_h = self.mean_h if h_baseline is None else h_baseline
        dh_f_298 = raw_dh_f * self.std_h + center_h
        raw_s_298 = self.s_298_head(h_scalar).squeeze(-1)
        # S298 mirrors H298 (v14.2): per-sample entropy baseline + residual, or
        # global mean when no baseline is supplied (v14/v14.1 behavior).
        center_s = self.mean_s if s_baseline is None else s_baseline
        s_298 = raw_s_298 * self.std_s + center_s
        # Pathway 2: NASA coefficients
        h_nasa = self.trunk_nasa(h_nasa_in)
        a1_10 = self.nasa_head(h_nasa)
        # Scale parameters to avoid numerical explosion with T^4
        scales = torch.tensor([1.0, 1e-3, 1e-6, 1e-9, 1e-12], device=h_nasa_in.device)
        a1_5_low = a1_10[:, :5] * scales
        a1_5_high = a1_10[:, 5:] * scales
        return {
            "a1_5_low": a1_5_low,
            "a1_5_high": a1_5_high,
            "dh_f_298": dh_f_298,
            "s_298": s_298,
        }