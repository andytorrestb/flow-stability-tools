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

    for alpha in alpha_values:
        solve = solve_rayleigh_for_alpha(
            alpha=alpha,
            U_vals=U_vals,
            Upp_vals=Upp_vals,
            D2=D2,
            magnitude_threshold=magnitude_threshold,
        )
        omega_vals = solve["omega"]

        if omega_vals.size == 0:
            dominant_growth.append(float("nan"))
            dominant_freq.append(float("nan"))
            dominant_omega_real.append(float("nan"))
            dominant_omega_imag.append(float("nan"))
            continue

        idx = int(np.argmax(np.imag(omega_vals)))
        omega_star = omega_vals[idx]

        dominant_growth.append(float(np.imag(omega_star)))
        dominant_freq.append(float(np.real(omega_star)))
        dominant_omega_real.append(float(np.real(omega_star)))
        dominant_omega_imag.append(float(np.imag(omega_star)))

    growth_arr = np.asarray(dominant_growth, dtype=float)
    valid_mask = np.isfinite(growth_arr)

    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        local_idx = int(np.argmax(growth_arr[valid_mask]))
        global_idx = int(valid_idx[local_idx])
        alpha_star = float(alpha_values[global_idx])
        omega_r_star = float(dominant_omega_real[global_idx])
        omega_i_star = float(dominant_omega_imag[global_idx])
    else:
        alpha_star = float("nan")
        omega_r_star = float("nan")
        omega_i_star = float("nan")

    return {
        "alpha": [float(a) for a in alpha_values],
        "dominant_growth": dominant_growth,
        "dominant_frequency": dominant_freq,
        "dominant_omega_real": dominant_omega_real,
        "dominant_omega_imag": dominant_omega_imag,
        "summary": {
            "alpha_star": alpha_star,
            "omega_star_real": omega_r_star,
            "omega_star_imag": omega_i_star,
            "growth_rate": omega_i_star,
            "frequency": omega_r_star,
        },
    }
