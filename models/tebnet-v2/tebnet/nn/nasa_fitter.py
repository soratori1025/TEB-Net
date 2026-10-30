from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from tebnet.nn.thermodynamics import R_GAS
def nasa7_cp_over_r(T: Tensor, coeffs: Tensor) -> Tensor:
    a1, a2, a3, a4, a5 = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3], coeffs[..., 4]
    return a1 + a2 * T + a3 * T**2 + a4 * T**3 + a5 * T**4
def nasa7_h_over_rt(T: Tensor, coeffs: Tensor) -> Tensor:
    a1, a2, a3, a4, a5, a6 = (
        coeffs[..., 0], coeffs[..., 1], coeffs[..., 2],
        coeffs[..., 3], coeffs[..., 4], coeffs[..., 5],
    )
    return a1 + a2 * T / 2 + a3 * T**2 / 3 + a4 * T**3 / 4 + a5 * T**4 / 5 + a6 / T
def nasa7_s_over_r(T: Tensor, coeffs: Tensor) -> Tensor:
    a1, a2, a3, a4, a5, a7 = (
        coeffs[..., 0], coeffs[..., 1], coeffs[..., 2],
        coeffs[..., 3], coeffs[..., 4], coeffs[..., 6],
    )
    return a1 * torch.log(T) + a2 * T + a3 * T**2 / 2 + a4 * T**3 / 3 + a5 * T**4 / 4 + a7
def fit_nasa7_single_range(
    T_grid: Tensor,
    cp_over_r: Tensor,
    h_over_rt: Tensor,
    s_over_r: Tensor,
    w_cp: float = 1.0,
    w_h: float = 1.0,
    w_s: float = 1.0,
) -> Tensor:
    single = cp_over_r.dim() == 1
    if single:
        cp_over_r = cp_over_r.unsqueeze(0)
        h_over_rt = h_over_rt.unsqueeze(0)
        s_over_r = s_over_r.unsqueeze(0)
    batch_size = cp_over_r.shape[0]
    n_t = T_grid.shape[0]
    device = T_grid.device
    dtype = T_grid.dtype
    T1 = T_grid
    T2 = T_grid**2
    T3 = T_grid**3
    T4 = T_grid**4
    A_cp = torch.zeros(n_t, 7, dtype=dtype, device=device)
    A_cp[:, 0] = 1.0
    A_cp[:, 1] = T1
    A_cp[:, 2] = T2
    A_cp[:, 3] = T3
    A_cp[:, 4] = T4
    A_h = torch.zeros(n_t, 7, dtype=dtype, device=device)
    A_h[:, 0] = 1.0
    A_h[:, 1] = T1 / 2
    A_h[:, 2] = T2 / 3
    A_h[:, 3] = T3 / 4
    A_h[:, 4] = T4 / 5
    A_h[:, 5] = 1.0 / T1
    A_s = torch.zeros(n_t, 7, dtype=dtype, device=device)
    A_s[:, 0] = torch.log(T1)
    A_s[:, 1] = T1
    A_s[:, 2] = T2 / 2
    A_s[:, 3] = T3 / 3
    A_s[:, 4] = T4 / 4
    A_s[:, 6] = 1.0
    sqrt_wc = w_cp ** 0.5
    sqrt_wh = w_h ** 0.5
    sqrt_ws = w_s ** 0.5
    A = torch.cat([sqrt_wc * A_cp, sqrt_wh * A_h, sqrt_ws * A_s], dim=0)
    b = torch.cat([
        sqrt_wc * cp_over_r,
        sqrt_wh * h_over_rt,
        sqrt_ws * s_over_r,
    ], dim=1)
    result = torch.linalg.lstsq(A, b.T)
    coeffs = result.solution.T
    if single:
        coeffs = coeffs.squeeze(0)
    return coeffs
def fit_nasa7_two_ranges(
    T_grid: Tensor,
    cp: Tensor,
    H: Tensor,
    S: Tensor,
    T_mid: float = 1000.0,
    w_cp: float = 1.0,
    w_h: float = 1.0,
    w_s: float = 1.0,
) -> tuple[Tensor, Tensor]:
    cp_over_r = cp / R_GAS
    h_over_rt = H / (R_GAS * T_grid.unsqueeze(0))
    s_over_r = S / R_GAS
    low_mask = T_grid <= T_mid
    high_mask = T_grid >= T_mid
    T_low = T_grid[low_mask]
    T_high = T_grid[high_mask]
    coeffs_low = fit_nasa7_single_range(
        T_low,
        cp_over_r[:, low_mask],
        h_over_rt[:, low_mask],
        s_over_r[:, low_mask],
        w_cp, w_h, w_s,
    )
    coeffs_high = fit_nasa7_single_range(
        T_high,
        cp_over_r[:, high_mask],
        h_over_rt[:, high_mask],
        s_over_r[:, high_mask],
        w_cp, w_h, w_s,
    )
    return coeffs_low, coeffs_high
def format_nasa7_string(
    species_name: str,
    coeffs_high: Tensor,
    coeffs_low: Tensor,
    T_low: float = 200.0,
    T_mid: float = 1000.0,
    T_high: float = 3000.0,
    composition: dict[str, int] | None = None,
    phase: str = "G",
) -> str:
    ch = coeffs_high.detach().cpu().numpy()
    cl = coeffs_low.detach().cpu().numpy()
    comp_str = ""
    if composition:
        for elem, count in list(composition.items())[:4]:
            comp_str += f"{elem:2s}{count:3d}"
    line1 = f"{species_name:<24s}{comp_str:<20s}{phase:1s}{T_low:10.3f}{T_high:10.3f}{T_mid:8.3f}      1\n"
    line2 = f"{ch[0]:15.8E}{ch[1]:15.8E}{ch[2]:15.8E}{ch[3]:15.8E}{ch[4]:15.8E}    2\n"
    line3 = f"{ch[5]:15.8E}{ch[6]:15.8E}{cl[0]:15.8E}{cl[1]:15.8E}{cl[2]:15.8E}    3\n"
    line4 = f"{cl[3]:15.8E}{cl[4]:15.8E}{cl[5]:15.8E}{cl[6]:15.8E}                   4\n"
    return line1 + line2 + line3 + line4
def classical_cp_limits(n_atoms, is_linear, n_rotors) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = np.atleast_1d(np.asarray(n_atoms, dtype=np.float64))
    lin = np.atleast_1d(np.asarray(is_linear, dtype=np.float64)) >= 0.5
    rot = np.atleast_1d(np.asarray(n_rotors, dtype=np.float64))
    mono = n_atoms <= 1
    hard = np.where(mono, 2.5, np.where(lin, 3 * n_atoms - 1.5, 3 * n_atoms - 2.0)) * R_GAS
    est = hard - np.where(mono | lin, 0.0, 0.5 * rot) * R_GAS
    return hard, np.maximum(est, np.where(mono, 2.5, np.where(lin, 3.5, 4.0)) * R_GAS)
def _cp_r_np(a: np.ndarray, T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return a[:, :5] @ np.vstack([T ** k for k in range(5)])
def _dcp_r_np(a: np.ndarray, T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return a[:, 1:5] @ np.vstack([k * T ** (k - 1) for k in range(1, 5)])
def _fill_a6_a7_high_np(lo: np.ndarray, hi: np.ndarray, t_mid: float) -> np.ndarray:
    hi = hi.copy()
    h_basis = np.array([t_mid ** k / (k + 1) for k in range(5)])
    s_basis = np.array([np.log(t_mid), t_mid, t_mid ** 2 / 2, t_mid ** 3 / 3, t_mid ** 4 / 4])
    h_mid = lo[:, :5] @ h_basis + lo[:, 5] / t_mid
    s_mid = lo[:, :5] @ s_basis + lo[:, 6]
    hi[:, 5] = t_mid * (h_mid - hi[:, :5] @ h_basis)
    hi[:, 6] = s_mid - hi[:, :5] @ s_basis
    return hi
def _complete_high_np(lo, hi, t_mid, t_data, hard, est, t_high, n_fit=30, scale=1000.0):
    t_data = np.atleast_1d(np.asarray(t_data, dtype=np.float64))
    T1 = float(t_data.max())
    T_fit = np.linspace(T1, t_high, n_fit + 1)[1:]
    c1 = R_GAS * _cp_r_np(hi, [T1])[:, 0]
    s1 = R_GAS * _dcp_r_np(hi, [T1])[:, 0]
    cp_mid = _cp_r_np(lo, [t_mid])[:, 0]
    dcp_mid = _dcp_r_np(lo, [t_mid])[:, 0]
    cp_dat = _cp_r_np(hi, t_data)
    k = np.arange(5)
    def vrow(T):
        return (np.asarray(T, dtype=np.float64)[:, None] / scale) ** k
    def drow(T):
        return np.where(k > 0, k * (T / scale) ** np.maximum(k - 1, 0), 0.0) / scale
    end_as_equality = len(t_data) + 4 <= 5
    hi_new = hi.copy()
    n_flat = 0
    for i in range(len(hard)):
        cinf = est[i] if c1[i] < est[i] else hard[i]
        if c1[i] >= cinf or s1[i] <= 0:
            f_end, s_end = c1[i], 0.0
            y = np.full(len(T_fit), c1[i])
            n_flat += 1
        else:
            p = s1[i] * T1 / (cinf - c1[i])
            y = cinf - (cinf - c1[i]) * (T1 / T_fit) ** p
            f_end = y[-1]
            s_end = (cinf - c1[i]) * p * T1 ** p * t_high ** (-p - 1)
        C = [vrow([t_mid])[0], drow(t_mid), *vrow(t_data)]
        d = [cp_mid[i], dcp_mid[i], *cp_dat[i]]
        A, yy = vrow(T_fit), y / R_GAS
        if end_as_equality:
            C += [vrow([t_high])[0], drow(t_high)]
            d += [f_end / R_GAS, s_end / R_GAS]
        else:
            A = np.vstack([A, 10.0 * drow(t_high)])
            yy = np.concatenate([yy, [10.0 * s_end / R_GAS]])
        C, d = np.array(C), np.array(d)
        n_c = len(d)
        K = np.block([[2 * A.T @ A, C.T], [C, np.zeros((n_c, n_c))]])
        rhs = np.concatenate([2 * A.T @ yy, d])
        b = np.linalg.lstsq(K, rhs, rcond=None)[0][:5]
        hi_new[i, :5] = b / scale ** k
    return _fill_a6_a7_high_np(lo, hi_new, t_mid), n_flat
def complete_nasa7_high_range(
    coeffs_low: Tensor,
    coeffs_high: Tensor,
    n_atoms,
    is_linear,
    n_rotors,
    t_data=(1500.0,),
    T_mid: float = 1000.0,
    T_high: float = 3000.0,
) -> tuple[Tensor, int]:
    single = coeffs_high.dim() == 1
    lo = coeffs_low.detach().double().cpu().numpy().reshape(-1, 7)
    hi = coeffs_high.detach().double().cpu().numpy().reshape(-1, 7)
    def _np(x):
        return x.detach().double().cpu().numpy() if torch.is_tensor(x) else x
    hard, est = classical_cp_limits(_np(n_atoms), _np(is_linear), _np(n_rotors))
    if len(hard) != len(hi):
        raise ValueError(f"{len(hi)} coefficient rows but {len(hard)} molecules in n_atoms / is_linear / n_rotors")
    hi_new, n_flat = _complete_high_np(lo, hi, float(T_mid), t_data, hard, est, float(T_high))
    out = torch.as_tensor(hi_new, dtype=coeffs_high.dtype, device=coeffs_high.device)
    return (out[0] if single else out), n_flat
