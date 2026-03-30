from __future__ import annotations

import numpy as np


def chebyshev_matrices(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Chebyshev collocation grid and first/second derivative matrices on [-L, L]."""
    if N < 2:
        raise ValueError("N must be at least 2")

    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)
    z = L * x

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D_x = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D_x = D_x - np.diag(np.sum(D_x, axis=1))

    D = D_x / L
    D2 = D @ D
    return z, D, D2
