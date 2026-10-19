import math
import numpy as np
from scipy.optimize import least_squares

# PHYSICAL CONSTANTS
R_GAS = 8.314462618153244       # J/(mol·K)
K_B = 1.380649e-23              # J/K
H_PLANCK = 6.62607015e-34       # J·s
C_LIGHT = 2.99792458e10         # cm/s
N_A = 6.02214076e23             # 1/mol
HARTREE_TO_KJ = 2625.4996395    # kJ/mol per Hartree
T_REF = 298.15                  # K — Reference temperature

# DEFAULT TEMPERATURE GRID
DEFAULT_T_GRID = np.array([
    200, 250, 298.15, 300, 400, 500, 600, 700, 800, 900,
    1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000,
], dtype=np.float64)

# RRHO THERMODYNAMICS
def freq_cm_to_theta(freq_cm):
    return H_PLANCK * C_LIGHT * freq_cm / K_B

def einstein_cp(theta, T):
    T = np.asarray(T, dtype=np.float64)
    x = theta / np.maximum(T, 1.0)
    result = np.zeros_like(x)
    mask = x < 500.0
    xs = x[mask]
    exp_xs = np.exp(xs)
    result[mask] = R_GAS * xs**2 * exp_xs / (exp_xs - 1.0)**2
    # For very large x, contribution is negligible
    return result

def einstein_h_integral(theta, T):
    T = np.asarray(T, dtype=np.float64)
    x = theta / np.maximum(T, 1.0)

    result = np.zeros_like(x)
    mask = x < 500.0

    result[mask] = R_GAS * theta / (np.exp(x[mask]) - 1.0)

    return result


def einstein_s(theta, T):
    """
    Vibrational entropy for one mode.

    S_vib = R · [ (θ/T)/(exp(θ/T) - 1) - ln(1 - exp(-θ/T)) ]

    Returns J/(mol·K).
    """
    T = np.asarray(T, dtype=np.float64)
    x = theta / np.maximum(T, 1.0)

    result = np.zeros_like(x)
    mask = x < 500.0

    xs = x[mask]
    exp_neg = np.exp(-xs)
    result[mask] = R_GAS * (xs / (np.exp(xs) - 1.0) - np.log(1.0 - exp_neg))

    return result


def rrho_cp(freqs_cm, T, n_atoms, is_linear=False):
    """
    Compute RRHO ideal-gas Cp at temperature(s) T.

    Cp = Cp_trans + Cp_rot + Cp_vib
    where Cp = Cv + R for ideal gas.

    Parameters
    ----------
    freqs_cm : array-like
        Vibrational frequencies in cm⁻¹.
        Should be 3N-6 (nonlinear) or 3N-5 (linear) real frequencies.
    T : float or array-like
        Temperature(s) in K.
    n_atoms : int
        Number of atoms in the molecule.
    is_linear : bool
        Whether the molecule is linear.

    Returns
    -------
    np.ndarray
        Cp in J/(mol·K), shape matches T.
    """
    T = np.asarray(T, dtype=np.float64)
    freqs_cm = np.asarray(freqs_cm, dtype=np.float64)

    # Remove zero / imaginary frequencies
    freqs_cm = freqs_cm[freqs_cm > 10.0]

    # Translational + rotational Cv, then add R for Cp
    if n_atoms == 1:
        # Monatomic: Cv_trans = 3/2 R, no rotation
        cp_tr = 2.5 * R_GAS
    elif is_linear:
        # Linear: Cv_trans = 3/2 R, Cv_rot = R
        cp_tr = 3.5 * R_GAS
    else:
        # Nonlinear: Cv_trans = 3/2 R, Cv_rot = 3/2 R
        cp_tr = 4.0 * R_GAS

    # Vibrational contributions
    cp_vib = np.zeros_like(T)
    for freq in freqs_cm:
        theta = freq_cm_to_theta(freq)
        cp_vib += einstein_cp(theta, T)

    return cp_tr + cp_vib


def rrho_s(freqs_cm, T, n_atoms, molecular_weight,
           rot_constants_ghz=None, symmetry_number=1,
           is_linear=False):
    """
    Compute RRHO ideal-gas entropy at temperature(s) T.

    S_total = S_trans + S_rot + S_vib

    Parameters
    ----------
    freqs_cm : array-like
        Vibrational frequencies in cm⁻¹.
    T : float or array-like
        Temperature(s) in K.
    n_atoms : int
        Number of atoms.
    molecular_weight : float
        Molecular weight in g/mol (amu).
    rot_constants_ghz : array-like or None
        Rotational constants [A, B, C] in GHz.
        Required for multi-atom species.
    symmetry_number : int
        External symmetry number σ.
    is_linear : bool
        Whether the molecule is linear.

    Returns
    -------
    np.ndarray
        S in J/(mol·K), shape matches T.
    """
    T = np.asarray(T, dtype=np.float64)
    freqs_cm = np.asarray(freqs_cm, dtype=np.float64)
    freqs_cm = freqs_cm[freqs_cm > 10.0]

    # --- Translational entropy (Sackur-Tetrode) ---
    # S_trans/R = 5/2 + ln[ (2πmkT)^(3/2) / (h³ · P/kT) ]
    # At P = 1 bar = 1e5 Pa
    m_kg = molecular_weight * 1e-3 / N_A  # mass per molecule in kg
    P = 1e5  # Pa (1 bar standard pressure)

    s_trans = np.zeros_like(T)
    for i, Ti in enumerate(T.flat):
        q_trans_V = (2.0 * math.pi * m_kg * K_B * Ti / H_PLANCK**2)**1.5
        V_per_mol = R_GAS * Ti / P  # m³/mol at 1 bar
        V_per_molecule = V_per_mol / N_A  # m³/molecule
        s_trans.flat[i] = R_GAS * (
            2.5 + math.log(q_trans_V * V_per_molecule)
        )

    if n_atoms == 1:
        return s_trans

    # --- Rotational entropy ---
    s_rot = np.zeros_like(T)
    sigma = max(symmetry_number, 1)

    if rot_constants_ghz is not None:
        rot_constants_ghz = np.asarray(rot_constants_ghz, dtype=np.float64)
        rot_constants_ghz = rot_constants_ghz[rot_constants_ghz > 0]

        if is_linear and len(rot_constants_ghz) >= 1:
            # Linear: q_rot = kT / (σ · h · B)
            B_hz = rot_constants_ghz[0] * 1e9  # GHz → Hz
            for i, Ti in enumerate(T.flat):
                q_rot = K_B * Ti / (sigma * H_PLANCK * B_hz)
                s_rot.flat[i] = R_GAS * (1.0 + math.log(max(q_rot, 1e-30)))

        elif len(rot_constants_ghz) >= 3:
            # Nonlinear: q_rot = sqrt(π) / σ · (kT/h)^(3/2) / sqrt(A·B·C)
            A_hz = rot_constants_ghz[0] * 1e9
            B_hz = rot_constants_ghz[1] * 1e9
            C_hz = rot_constants_ghz[2] * 1e9
            ABC = A_hz * B_hz * C_hz

            for i, Ti in enumerate(T.flat):
                q_rot = (
                    math.sqrt(math.pi) / sigma
                    * (K_B * Ti / H_PLANCK)**1.5
                    / math.sqrt(max(ABC, 1e-90))
                )
                s_rot.flat[i] = R_GAS * (1.5 + math.log(max(q_rot, 1e-30)))
    else:
        # Fallback: estimate rotational entropy
        # Use classical approximation
        if is_linear:
            s_rot[:] = R_GAS * 5.0  # rough estimate
        else:
            s_rot[:] = R_GAS * 7.0  # rough estimate

    # --- Vibrational entropy ---
    s_vib = np.zeros_like(T)
    for freq in freqs_cm:
        theta = freq_cm_to_theta(freq)
        s_vib += einstein_s(theta, T)

    return s_trans + s_rot + s_vib


def rrho_h_increment(freqs_cm, T, n_atoms, is_linear=False):
    """
    Compute H(T) - H(0) from RRHO.

    This gives the thermal enthalpy increment, NOT ΔH_f.

    Returns J/mol.
    """
    T = np.asarray(T, dtype=np.float64)
    freqs_cm = np.asarray(freqs_cm, dtype=np.float64)
    freqs_cm = freqs_cm[freqs_cm > 10.0]

    # Translational + rotational: Cv·T + PV term = Cp·T for classical
    if n_atoms == 1:
        h_tr = 2.5 * R_GAS * T
    elif is_linear:
        h_tr = 3.5 * R_GAS * T
    else:
        h_tr = 4.0 * R_GAS * T

    # Vibrational: H_vib = Σ R·θ / (exp(θ/T) - 1)  + ZPE
    h_vib = np.zeros_like(T)
    zpe = 0.0
    for freq in freqs_cm:
        theta = freq_cm_to_theta(freq)
        h_vib += einstein_h_integral(theta, T)
        zpe += 0.5 * R_GAS * theta  # ZPE = Σ ½ R θ

    return h_tr + h_vib + zpe


# ============================================================
# NASA-7 POLYNOMIAL EVALUATION
# ============================================================

def nasa7_cp(coeffs, T):
    """
    Evaluate Cp/R from NASA-7 polynomial coefficients.

    Cp/R = a1 + a2·T + a3·T² + a4·T³ + a5·T⁴

    Parameters
    ----------
    coeffs : array-like, length 7
        [a1, a2, a3, a4, a5, a6, a7]
    T : float or array

    Returns
    -------
    Cp in J/(mol·K)
    """
    a = np.asarray(coeffs, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    cp_over_R = (
        a[0]
        + a[1] * T
        + a[2] * T**2
        + a[3] * T**3
        + a[4] * T**4
    )
    return cp_over_R * R_GAS


def nasa7_h(coeffs, T):
    """
    Evaluate H/RT from NASA-7 polynomial.

    H/RT = a1 + a2·T/2 + a3·T²/3 + a4·T³/4 + a5·T⁴/5 + a6/T

    Returns H in J/mol.
    """
    a = np.asarray(coeffs, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    h_over_RT = (
        a[0]
        + a[1] * T / 2
        + a[2] * T**2 / 3
        + a[3] * T**3 / 4
        + a[4] * T**4 / 5
        + a[5] / T
    )
    return h_over_RT * R_GAS * T


def nasa7_s(coeffs, T):
    """
    Evaluate S/R from NASA-7 polynomial.

    S/R = a1·ln(T) + a2·T + a3·T²/2 + a4·T³/3 + a5·T⁴/4 + a7

    Returns S in J/(mol·K).
    """
    a = np.asarray(coeffs, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    s_over_R = (
        a[0] * np.log(T)
        + a[1] * T
        + a[2] * T**2 / 2
        + a[3] * T**3 / 3
        + a[4] * T**4 / 4
        + a[6]
    )
    return s_over_R * R_GAS


def nasa7_cp_grid(coeffs_low, coeffs_high, T_grid, T_mid=1000.0):
    """
    Evaluate Cp at a temperature grid using two-range NASA-7.

    Parameters
    ----------
    coeffs_low : array, length 7
        Low-temperature coefficients [a1..a7].
    coeffs_high : array, length 7
        High-temperature coefficients [a1..a7].
    T_grid : array
        Temperature grid in K.
    T_mid : float
        Midpoint temperature (default 1000 K).

    Returns
    -------
    np.ndarray
        Cp values in J/(mol·K).
    """
    T_grid = np.asarray(T_grid, dtype=np.float64)
    cp = np.zeros_like(T_grid)

    low_mask = T_grid < T_mid
    high_mask = ~low_mask

    if low_mask.any():
        cp[low_mask] = nasa7_cp(coeffs_low, T_grid[low_mask])
    if high_mask.any():
        cp[high_mask] = nasa7_cp(coeffs_high, T_grid[high_mask])

    return cp


# ============================================================
# NASA-7 POLYNOMIAL FITTING
# ============================================================

def fit_nasa7(T_data, cp_data, H_ref, S_ref,
              T_ref=298.15, T_mid=1000.0,
              T_min=200.0, T_max=6000.0):
    """
    Fit NASA-7 two-range polynomial to Cp(T) data.

    Strategy:
        1. Fit a1-a5 for low and high ranges via least squares
           on Cp/R data.
        2. Enforce continuity of Cp at T_mid.
        3. Determine a6 from H_ref at T_ref.
        4. Determine a7 from S_ref at T_ref.

    Parameters
    ----------
    T_data : np.ndarray
        Temperature points where Cp is known.
    cp_data : np.ndarray
        Cp values in J/(mol·K) at T_data.
    H_ref : float
        Reference enthalpy ΔH_f(T_ref) in kJ/mol.
    S_ref : float
        Reference entropy S(T_ref) in J/(mol·K).
    T_ref : float
        Reference temperature (default 298.15 K).
    T_mid : float
        Midpoint temperature (default 1000 K).
    T_min : float
        Minimum temperature for low range.
    T_max : float
        Maximum temperature for high range.

    Returns
    -------
    dict with keys:
        'coeffs_low': array[7] — low-T NASA coefficients
        'coeffs_high': array[7] — high-T NASA coefficients
        'T_min', 'T_mid', 'T_max': temperature bounds
        'rmse_cp': RMSE of Cp fit in J/(mol·K)
    """
    T_data = np.asarray(T_data, dtype=np.float64)
    cp_data = np.asarray(cp_data, dtype=np.float64)

    # Convert Cp to Cp/R
    cp_over_R = cp_data / R_GAS

    # Separate into low and high ranges
    low_mask = T_data <= T_mid
    high_mask = T_data >= T_mid

    T_low = T_data[low_mask]
    cp_low = cp_over_R[low_mask]
    T_high = T_data[high_mask]
    cp_high = cp_over_R[high_mask]

    def _vandermonde(T):
        """Build design matrix for Cp/R = a1 + a2*T + a3*T² + a4*T³ + a5*T⁴"""
        T = np.asarray(T, dtype=np.float64)
        return np.column_stack([
            np.ones_like(T),
            T,
            T**2,
            T**3,
            T**4,
        ])

    # Fit low range
    if len(T_low) >= 5:
        A_low = _vandermonde(T_low)
        coeffs_low_5, _, _, _ = np.linalg.lstsq(A_low, cp_low, rcond=None)
    else:
        # Not enough points — use high range coefficients
        coeffs_low_5 = np.zeros(5)
        if len(cp_low) > 0:
            coeffs_low_5[0] = cp_low.mean()

    # Fit high range
    if len(T_high) >= 5:
        A_high = _vandermonde(T_high)
        coeffs_high_5, _, _, _ = np.linalg.lstsq(A_high, cp_high, rcond=None)
    else:
        coeffs_high_5 = coeffs_low_5.copy()

    # Enforce Cp continuity at T_mid
    # Adjust high-range a1 so that Cp_high(T_mid) = Cp_low(T_mid)
    T_mid_vec = _vandermonde(np.array([T_mid]))
    cp_low_at_mid = T_mid_vec @ coeffs_low_5
    cp_high_at_mid = T_mid_vec @ coeffs_high_5
    offset = (cp_low_at_mid - cp_high_at_mid)[0]
    coeffs_high_5[0] += offset

    # Determine a6 (enthalpy integration constant)
    # H(T)/RT = a1 + a2*T/2 + a3*T²/3 + a4*T³/4 + a5*T⁴/5 + a6/T
    # H_ref = H(T_ref) in J/mol
    H_ref_J = H_ref * 1000.0  # kJ/mol → J/mol

    # Use low-range coefficients at T_ref (since T_ref = 298.15 < T_mid)
    a = coeffs_low_5
    h_poly = (
        a[0]
        + a[1] * T_ref / 2
        + a[2] * T_ref**2 / 3
        + a[3] * T_ref**3 / 4
        + a[4] * T_ref**4 / 5
    )
    # H(T_ref) / (R * T_ref) = h_poly + a6 / T_ref
    # a6 = (H_ref / (R * T_ref) - h_poly) * T_ref
    a6_low = (H_ref_J / (R_GAS * T_ref) - h_poly) * T_ref

    # For high range, enforce H continuity at T_mid
    a_lo = coeffs_low_5
    a_hi = coeffs_high_5
    h_low_at_mid = (
        a_lo[0] * T_mid
        + a_lo[1] * T_mid**2 / 2
        + a_lo[2] * T_mid**3 / 3
        + a_lo[3] * T_mid**4 / 4
        + a_lo[4] * T_mid**5 / 5
        + a6_low
    ) * R_GAS

    h_high_poly_mid = (
        a_hi[0] * T_mid
        + a_hi[1] * T_mid**2 / 2
        + a_hi[2] * T_mid**3 / 3
        + a_hi[3] * T_mid**4 / 4
        + a_hi[4] * T_mid**5 / 5
    ) * R_GAS

    a6_high = (h_low_at_mid - h_high_poly_mid) / R_GAS

    # Determine a7 (entropy integration constant)
    # S(T)/R = a1*ln(T) + a2*T + a3*T²/2 + a4*T³/3 + a5*T⁴/4 + a7
    s_poly = (
        a[0] * np.log(T_ref)
        + a[1] * T_ref
        + a[2] * T_ref**2 / 2
        + a[3] * T_ref**3 / 3
        + a[4] * T_ref**4 / 4
    )
    a7_low = S_ref / R_GAS - s_poly

    # For high range, enforce S continuity at T_mid
    s_low_at_mid = (
        a_lo[0] * np.log(T_mid)
        + a_lo[1] * T_mid
        + a_lo[2] * T_mid**2 / 2
        + a_lo[3] * T_mid**3 / 3
        + a_lo[4] * T_mid**4 / 4
        + a7_low
    ) * R_GAS

    s_high_poly_mid = (
        a_hi[0] * np.log(T_mid)
        + a_hi[1] * T_mid
        + a_hi[2] * T_mid**2 / 2
        + a_hi[3] * T_mid**3 / 3
        + a_hi[4] * T_mid**4 / 4
    ) * R_GAS

    a7_high = (s_low_at_mid - s_high_poly_mid) / R_GAS

    # Assemble full 7-coefficient arrays
    coeffs_low_7 = np.array([*coeffs_low_5, a6_low, a7_low])
    coeffs_high_7 = np.array([*coeffs_high_5, a6_high, a7_high])

    # Compute RMSE
    cp_fit = nasa7_cp_grid(coeffs_low_7, coeffs_high_7, T_data, T_mid)
    rmse = np.sqrt(np.mean((cp_fit - cp_data)**2))

    return {
        "coeffs_low": coeffs_low_7,
        "coeffs_high": coeffs_high_7,
        "T_min": T_min,
        "T_mid": T_mid,
        "T_max": T_max,
        "rmse_cp": rmse,
    }


# ============================================================
# QM9 ATOMIC REFERENCE VALUES
# ============================================================
# B3LYP/6-31G(2df,p) atomic energies (U0, at 0K)
# from Ramakrishnan et al. and common computational chemistry references.

QM9_ATOMIC_U0 = {
    "H": -0.500273,     # Hartree
    "C": -37.846772,
    "N": -54.583861,
    "O": -75.064579,
    "F": -99.718730,
}

# Thermal corrections for atoms at 298.15K:
# H(298) - U(0) = 5/2 · k_B · T  (for monatomic ideal gas, per molecule)
# In Hartree: 5/2 · k_B · 298.15 / (4.3597447222071e-18) = 0.002360 Ha
QM9_ATOM_THERMAL_CORRECTION = 0.002360  # Ha, same for all atoms

# Experimental ΔH_f(298) for gaseous atoms in kJ/mol
# Sources: CODATA 2014, ATcT
EXPT_DHF_ATOMS = {
    "H": 217.998,
    "C": 716.68,
    "N": 472.68,
    "O": 249.18,
    "F": 79.38,
}


def compute_dhf_298_from_qm9(H_298_hartree, atom_counts):
    """
    Compute ΔH_f(298) from QM9 total enthalpy using
    atomization energy scheme.

    ΔH_f(mol, 298) = H_mol(298) - Σ H_atom(298) + Σ ΔH_f(atom, 298, expt)

    Parameters
    ----------
    H_298_hartree : float
        Total enthalpy of molecule at 298.15K in Hartree
        (from QM9 property 'H').
    atom_counts : dict
        {element: count}, e.g. {"C": 6, "H": 6}.

    Returns
    -------
    float
        ΔH_f(298) in kJ/mol.
    """
    # Sum of atomic H(298) in Hartree
    sum_H_atoms = 0.0
    sum_dhf_atoms = 0.0

    for elem, count in atom_counts.items():
        if elem not in QM9_ATOMIC_U0:
            raise ValueError(
                f"Unsupported element: {elem}. "
                f"QM9 only covers H, C, N, O, F."
            )
        H_atom_298 = (
            QM9_ATOMIC_U0[elem]
            + QM9_ATOM_THERMAL_CORRECTION
        )
        sum_H_atoms += count * H_atom_298
        sum_dhf_atoms += count * EXPT_DHF_ATOMS[elem]

    # Atomization enthalpy
    delta_H_at = (sum_H_atoms - H_298_hartree) * HARTREE_TO_KJ

    # ΔH_f = Σ ΔH_f(atoms) - ΔH_at
    dhf_298 = sum_dhf_atoms - delta_H_at

    return dhf_298
