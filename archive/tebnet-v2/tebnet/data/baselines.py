"""Frozen linear baselines for TEB-Net v15 (numpy only, fitted on the TRAIN split).

The network learns only the residual on top of these:
    ΔH_f,298 = x_H · w_H + b_H + σ_H · NN_H
    S°298    = x_S · w_S + b_S + σ_S · NN_S
    Cp(T)/R  = Σ_k c_k T_k(τ)  with  c = x_Cp · W_Cp + b_Cp + σ_c ⊙ NN_Cp
(Chebyshev T_k on each NASA range; converted exactly to NASA-7 monomials in the model).

Ridge on standardised columns (tiny λ) instead of plain lstsq: the bond / ring
counts are nearly collinear with the element counts, and plain OLS can then
produce large opposite coefficients that extrapolate badly on the size split.
Columns that are constant in the train split get weight 0, so an unseen feature
contributes nothing at test time instead of an arbitrary amount.
"""
from __future__ import annotations
import numpy as np
R_GAS = 8.314462618
N_CHEB = 5
def fit_ridge(X: np.ndarray, Y: np.ndarray, lam_rel: float = 1e-3):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    n, d = X.shape
    mu = X.mean(0)
    sd = X.std(0)
    keep = sd > 1e-8
    W = np.zeros((d, Y.shape[1]))
    ymu = Y.mean(0)
    if keep.any():
        Xs = (X[:, keep] - mu[keep]) / sd[keep]
        A = Xs.T @ Xs + lam_rel * n * np.eye(int(keep.sum()))
        Ws = np.linalg.solve(A, Xs.T @ (Y - ymu))
        W[keep] = Ws / sd[keep][:, None]
    b = ymu - mu @ W
    resid = Y - (X @ W + b)
    if squeeze:
        return W[:, 0], float(b[0]), resid[:, 0]
    return W, b, resid
def fit_scalar_baseline(X: np.ndarray, y: np.ndarray, lam_rel: float = 1e-3):
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 2:
        raise ValueError("not enough finite targets to fit the baseline")
    w, b, resid = fit_ridge(X[m], y[m], lam_rel)
    return {
        "w": w.astype(np.float64),
        "b": float(b),
        "resid_std": float(max(resid.std(), 1e-6)),
        "target_std": float(y[m].std()),
        "n": int(m.sum()),
    }
def fit_scalar_baseline_fixed(X: np.ndarray, y: np.ndarray, fixed: np.ndarray, lam_rel: float = 1e-3):
    """Like fit_scalar_baseline, but columns with a finite entry in `fixed` keep that coefficient (in target
    units per unit of the column) and only the other columns are fitted, on y minus the fixed part.

    The returned "w" covers all columns, so the frozen prior x·w + b of the model is unchanged in form
    (entropy prior v2: exact S_trans, S_rot, −R ln σ, R ln(2S+1) with coefficients R, R, −R, R)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.float64)
    if fixed.shape != (X.shape[1],):
        raise ValueError(f"`fixed` must have one entry per column ({X.shape[1]}), got {fixed.shape}")
    is_fixed = np.isfinite(fixed)
    m = np.isfinite(y) & np.isfinite(X).all(1)
    if m.sum() < 2:
        raise ValueError("not enough finite targets to fit the baseline")
    offset = X[m][:, is_fixed] @ fixed[is_fixed] if is_fixed.any() else np.zeros(int(m.sum()))
    w = np.zeros(X.shape[1])
    w[is_fixed] = fixed[is_fixed]
    if (~is_fixed).any():
        w_free, b, _ = fit_ridge(X[m][:, ~is_fixed], y[m] - offset, lam_rel)
        w[~is_fixed] = w_free
    else:
        b = float((y[m] - offset).mean())
    resid = y[m] - (X[m] @ w + b)
    return {
        "w": w.astype(np.float64),
        "b": float(b),
        "resid_std": float(max(resid.std(), 1e-6)),
        "target_std": float(y[m].std()),
        "n": int(m.sum()),
        "fixed_mask": is_fixed,
    }
def cheb_vander(T: np.ndarray, lo: float, hi: float, n: int = N_CHEB) -> np.ndarray:
    tau = (2.0 * np.asarray(T, dtype=np.float64) - (lo + hi)) / (hi - lo)
    return np.polynomial.chebyshev.chebvander(tau, n - 1)
def cheb_to_monomial_matrix(lo: float, hi: float, n: int = N_CHEB) -> np.ndarray:
    from numpy.polynomial import Chebyshev, Polynomial
    M = np.zeros((n, n))
    for k in range(n):
        p = Chebyshev.basis(k, domain=[lo, hi]).convert(kind=Polynomial)
        M[: len(p.coef), k] = p.coef
    return M
def cp_cheb_targets(cp: np.ndarray, T_grid: np.ndarray, t_mid: float, lo: float, hi: float, n: int = N_CHEB):
    cpR = np.asarray(cp, dtype=np.float64) / R_GAS
    T_grid = np.asarray(T_grid, dtype=np.float64)
    low = T_grid <= t_mid
    high = T_grid >= t_mid
    k_low, k_high = int(min(n, low.sum())), int(min(n, high.sum()))
    valid = np.isfinite(cpR[:, low]).all(1) & np.isfinite(cpR[:, high]).all(1)
    C = np.zeros((cpR.shape[0], 2 * n))
    if valid.any() and k_low > 0:
        B = cheb_vander(T_grid[low], lo, t_mid, k_low)
        C[valid, :k_low] = np.linalg.lstsq(B, cpR[valid][:, low].T, rcond=None)[0].T
    if valid.any() and k_high > 0:
        B = cheb_vander(T_grid[high], t_mid, hi, k_high)
        C[valid, n:n + k_high] = np.linalg.lstsq(B, cpR[valid][:, high].T, rcond=None)[0].T
    return C, valid, (k_low, k_high)
def cp_from_cheb(C: np.ndarray, T_grid: np.ndarray, t_mid: float, lo: float, hi: float, n: int = N_CHEB) -> np.ndarray:
    T_grid = np.asarray(T_grid, dtype=np.float64)
    out = np.zeros((C.shape[0], len(T_grid)))
    low = T_grid <= t_mid
    if low.any():
        out[:, low] = C[:, :n] @ cheb_vander(T_grid[low], lo, t_mid, n).T
    if (~low).any():
        out[:, ~low] = C[:, n:] @ cheb_vander(T_grid[~low], t_mid, hi, n).T
    return out * R_GAS
def fit_cp_cheb_baseline(X: np.ndarray, cp: np.ndarray, T_grid: np.ndarray, t_mid: float, lo: float, hi: float, lam_rel: float = 1e-3, n: int = N_CHEB, floor_frac: float = 0.02):
    C, valid, (k_low, k_high) = cp_cheb_targets(cp, T_grid, t_mid, lo, hi, n)
    if valid.sum() < 2:
        raise ValueError("not enough molecules with a complete Cp grid to fit the Cp baseline")
    W, b, resid = fit_ridge(X[valid], C[valid], lam_rel)
    s = resid.std(0)
    s_low = s[:n].copy()
    ref_low = float(s_low[:k_low].max()) if k_low > 0 else 1.0
    s_low = np.maximum(s_low, floor_frac * ref_low)
    s_low[k_low:] = floor_frac * ref_low
    s_high = s[n:].copy()
    ref_high = float(s_high[1:k_high].max()) if k_high > 1 else ref_low
    s_high = np.maximum(s_high, floor_frac * ref_high)
    s_high[k_high:] = floor_frac * ref_high
    coef_std = np.concatenate([s_low, s_high])
    cp_true = np.asarray(cp, dtype=np.float64)[valid]
    cp_base = cp_from_cheb(X[valid] @ W + b, T_grid, t_mid, lo, hi, n)
    d = (cp_true - cp_base)[np.isfinite(cp_true)]
    return {
        "W": W, "b": b, "coef_std": coef_std,
        "cp_resid_std": float(max(d.std(), 1e-6)),
        "cp_target_std": float(np.nanstd(cp_true)),
        "n": int(valid.sum()), "k_low": k_low, "k_high": k_high,
    }
