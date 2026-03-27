from __future__ import annotations

import numpy as np
from scipy.linalg import eig


def solve_rayleigh_for_alpha(
    alpha: float,
    U_vals: np.ndarray,
    Upp_vals: np.ndarray,
    D2: np.ndarray,
    magnitude_threshold: float,
) -> dict[str, np.ndarray]:
    """Solve A phi = c B phi for one alpha and return filtered c and omega."""
    npts = U_vals.size
    I = np.eye(npts)

    lap = D2 - (alpha**2) * I
    A = np.diag(U_vals) @ lap - np.diag(Upp_vals)
    B = lap.copy()

    interior = slice(1, npts - 1)
    A_in = A[interior, interior]
    B_in = B[interior, interior]

    c_raw, _eigvecs = eig(A_in, B_in)
    omega_raw = alpha * c_raw

    finite = np.isfinite(c_raw.real) & np.isfinite(c_raw.imag)
    bounded = np.abs(c_raw) < magnitude_threshold
    keep = finite & bounded

    c_vals = c_raw[keep]
    omega_vals = omega_raw[keep]

    return {
        "c": c_vals,
        "omega": omega_vals,
    }
