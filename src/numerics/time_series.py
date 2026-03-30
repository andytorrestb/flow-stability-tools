from __future__ import annotations

import numpy as np

from numerics.spectral import chebyshev_matrices


def reconstruct_time_series(
    z: np.ndarray,
    eigenfunction: np.ndarray,
    omega: complex,
    t_grid: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Reconstruct q(z, t) = amplitude * phi(z) * exp(-i * omega * t)."""
    if eigenfunction.shape[0] != z.shape[0]:
        raise ValueError(
            f"eigenfunction length {eigenfunction.shape[0]} does not match z length {z.shape[0]}"
        )
    q0 = amplitude * eigenfunction
    qzt = np.outer(q0, np.exp(-1j * omega * t_grid))
    return qzt


def reconstruct_velocity_time_series(
    z: np.ndarray,
    eigenfunction: np.ndarray,
    omega: complex,
    t_grid: np.ndarray,
    amplitude: float = 1.0,
    D: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reconstruct u'(z, t) = amplitude * dphi/dz * exp(-i * omega * t).
    Uses Chebyshev derivative matrix if provided; otherwise infers it from z.
    """
    n = z.shape[0]
    if eigenfunction.shape[0] != n:
        raise ValueError(
            f"eigenfunction length {eigenfunction.shape[0]} does not match z length {n}"
        )

    if D is None:
        L = float(np.max(np.abs(z)))
        N = n - 1
        z_ref, D_ref, _D2 = chebyshev_matrices(N=N, L=L)
        if z_ref.shape != z.shape or not np.allclose(z_ref, z, atol=1e-12, rtol=1e-10):
            raise ValueError("Provided z does not match inferred Chebyshev grid for differentiation")
        D = D_ref

    phi_z = D @ eigenfunction
    phi_z[0] = 0.0
    phi_z[-1] = 0.0

    u0 = amplitude * phi_z
    uzt = np.outer(u0, np.exp(-1j * omega * t_grid))
    return uzt
