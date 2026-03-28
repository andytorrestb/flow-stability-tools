from __future__ import annotations

import numpy as np

from components.rayleigh import solve_rayleigh_for_alpha


def alpha_scan(
    alpha_values: np.ndarray,
    U_vals: np.ndarray,
    Upp_vals: np.ndarray,
    D2: np.ndarray,
    magnitude_threshold: float,
) -> dict[str, list[float] | dict[str, float]]:
    """Scan alpha and extract dominant growth and frequency tracks."""
    dominant_growth: list[float] = []
    dominant_freq: list[float] = []
    dominant_omega_real: list[float] = []
    dominant_omega_imag: list[float] = []


    dominant_eigenfunctions: list[np.ndarray | None] = []
    for alpha in alpha_values:
        solve = solve_rayleigh_for_alpha(
            alpha=alpha,
            U_vals=U_vals,
            Upp_vals=Upp_vals,
            D2=D2,
            magnitude_threshold=magnitude_threshold,
        )
        omega_vals = solve["omega"]
        eigvecs = solve["eigvecs"] if "eigvecs" in solve else None

        if omega_vals.size == 0 or eigvecs is None or eigvecs.shape[1] == 0:
            dominant_growth.append(float("nan"))
            dominant_freq.append(float("nan"))
            dominant_omega_real.append(float("nan"))
            dominant_omega_imag.append(float("nan"))
            dominant_eigenfunctions.append(None)
            continue

        idx = int(np.argmax(np.imag(omega_vals)))
        omega_star = omega_vals[idx]
        phi_star = eigvecs[:, idx]

        dominant_growth.append(float(np.imag(omega_star)))
        dominant_freq.append(float(np.real(omega_star)))
        dominant_omega_real.append(float(np.real(omega_star)))
        dominant_omega_imag.append(float(np.imag(omega_star)))
        dominant_eigenfunctions.append(phi_star)

    growth_arr = np.asarray(dominant_growth, dtype=float)
    valid_mask = np.isfinite(growth_arr)

    def serialize_eigenfunction(ef):
        if ef is None:
            return None
        return [[float(np.real(x)), float(np.imag(x))] for x in ef]

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        local_idx = int(np.argmax(growth_arr[valid_mask]))
        global_idx = int(valid_idx[local_idx])
        alpha_star = float(alpha_values[global_idx])
        omega_r_star = float(dominant_omega_real[global_idx])
        omega_i_star = float(dominant_omega_imag[global_idx])
        eigenfunction_star = serialize_eigenfunction(dominant_eigenfunctions[global_idx])
    else:
        alpha_star = float("nan")
        omega_r_star = float("nan")
        omega_i_star = float("nan")
        eigenfunction_star = None

    return {
        "alpha": [float(a) for a in alpha_values],
        "dominant_growth": dominant_growth,
        "dominant_frequency": dominant_freq,
        "dominant_omega_real": dominant_omega_real,
        "dominant_omega_imag": dominant_omega_imag,
        "dominant_eigenfunctions": [serialize_eigenfunction(ef) for ef in dominant_eigenfunctions],
        "summary": {
            "alpha_star": alpha_star,
            "omega_star_real": omega_r_star,
            "omega_star_imag": omega_i_star,
            "growth_rate": omega_i_star,
            "frequency": omega_r_star,
            "eigenfunction_star": eigenfunction_star,
        },
    }
