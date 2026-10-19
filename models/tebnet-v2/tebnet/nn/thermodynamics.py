from __future__ import annotations
import torch
from torch import Tensor

# Universal gas constant in J/(mol·K)
R_GAS = 8.314462618
# Boltzmann constant in J/K
K_B = 1.380649e-23
# Planck constant in J·s
H_PLANCK = 6.62607015e-34
# Reference temperature in K
T_REF = 298.15

def cp_trans_rot(N: Tensor, is_linear: Tensor) -> Tensor:
    # Monatomic: Cp = 5/2 R (no rotation)
    is_monatomic = (N == 1).float()
    is_lin = is_linear.float() * (1.0 - is_monatomic)
    is_nonlin = (1.0 - is_linear.float()) * (1.0 - is_monatomic)
    return 2.5 * is_monatomic + 3.5 * is_lin + 4.0 * is_nonlin

def n_vib_modes(N: Tensor, is_linear: Tensor) -> Tensor:
    is_monatomic = (N == 1).float()
    is_lin = is_linear.float() * (1.0 - is_monatomic)
    is_nonlin = (1.0 - is_linear.float()) * (1.0 - is_monatomic)
    return (3 * N - 5) * is_lin + (3 * N - 6) * is_nonlin

def cp_limit_high_t(N: Tensor, is_linear: Tensor) -> Tensor:
    return cp_trans_rot(N, is_linear) + n_vib_modes(N, is_linear)

# Variant A: Numerical integration from Cp grid (for MVP)
def integrate_cp_trapezoidal(cp_grid: Tensor, T_grid: Tensor, dh_f_298: Tensor, T_ref: float = T_REF) -> Tensor:
    batch_size, n_temps = cp_grid.shape
    # Find index closest to T_ref
    ref_idx = torch.argmin(torch.abs(T_grid - T_ref))
    # dT and average Cp for each interval
    dT = T_grid[1:] - T_grid[:-1]  # (n_temps - 1,)
    cp_avg = 0.5 * (cp_grid[:, :-1] + cp_grid[:, 1:])  # (batch, n_temps - 1)
    dH_intervals = cp_avg * dT.unsqueeze(0)  # (batch, n_temps - 1)
    # Cumulative sum from the beginning
    cumsum = torch.cumsum(dH_intervals, dim=1)  # (batch, n_temps - 1)
    # H values at each grid point (H at T_grid[0] is defined as cumsum starting point)
    # We need cumsum relative to ref_idx
    H = torch.zeros_like(cp_grid)  # (batch, n_temps)
    H[:, 0] = 0.0  # will be adjusted
    # Compute cumulative integral from T_grid[0]
    H[:, 1:] = cumsum
    # Shift so H[ref_idx] = dh_f_298
    H = H - H[:, ref_idx:ref_idx+1] + dh_f_298.unsqueeze(-1)
    return H

def integrate_cp_over_t_trapezoidal(cp_grid: Tensor, T_grid: Tensor, s_298: Tensor, T_ref: float = T_REF) -> Tensor:
    # Cp/T at each grid point
    cp_over_t = cp_grid / T_grid.unsqueeze(0)  # (batch, n_temps)
    # dT
    dT = T_grid[1:] - T_grid[:-1]  # (n_temps - 1,)
    avg = 0.5 * (cp_over_t[:, :-1] + cp_over_t[:, 1:])  # (batch, n_temps - 1)
    dS_intervals = avg * dT.unsqueeze(0)
    # Cumulative
    ref_idx = torch.argmin(torch.abs(T_grid - T_ref))
    cumsum = torch.cumsum(dS_intervals, dim=1)
    S = torch.zeros_like(cp_grid)
    S[:, 1:] = cumsum
    S = S - S[:, ref_idx:ref_idx+1] + s_298.unsqueeze(-1)
    return S
