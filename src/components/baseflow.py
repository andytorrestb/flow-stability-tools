from __future__ import annotations

import numpy as np
import sympy as sp


def baseflow_symbolic(profile_name: str) -> tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr]:
    """Return symbolic z, U, U', U'' for a supported baseflow profile."""
    z = sp.symbols("z", real=True)

    if profile_name == "tanh_shear":
        U = sp.Rational(1, 2) * (1 + sp.tanh(z))
    elif profile_name == "parabolic":
        U = z**2
    else:
        raise ValueError(f"Unsupported baseflow profile: {profile_name}")

    Up = sp.diff(U, z)
    Upp = sp.diff(Up, z)
    return z, U, Up, Upp


def evaluate_baseflow(profile_name: str, z_grid: np.ndarray) -> dict[str, np.ndarray | str]:
    """Evaluate U and U'' on the collocation grid, handling scalar lambdify outputs."""
    z_sym, U_expr, Up_expr, Upp_expr = baseflow_symbolic(profile_name)
    U_fun = sp.lambdify(z_sym, U_expr, "numpy")
    Upp_fun = sp.lambdify(z_sym, Upp_expr, "numpy")

    U_raw = np.asarray(U_fun(z_grid), dtype=np.complex128)
    Upp_raw = np.asarray(Upp_fun(z_grid), dtype=np.complex128)

    U_vals = np.broadcast_to(U_raw, z_grid.shape).astype(np.complex128, copy=False)
    Upp_vals = np.broadcast_to(Upp_raw, z_grid.shape).astype(np.complex128, copy=False)

    return {
        "U": U_vals,
        "Upp": Upp_vals,
        "symbolic": {
            "symbol": str(z_sym),
            "U": str(U_expr),
            "U_prime": str(Up_expr),
            "U_double_prime": str(Upp_expr),
            "U_pretty": sp.pretty(U_expr, use_unicode=False),
            "U_prime_pretty": sp.pretty(Up_expr, use_unicode=False),
            "U_double_prime_pretty": sp.pretty(Upp_expr, use_unicode=False),
            "U_latex": sp.latex(U_expr),
            "U_prime_latex": sp.latex(Up_expr),
            "U_double_prime_latex": sp.latex(Upp_expr),
        },
    }
