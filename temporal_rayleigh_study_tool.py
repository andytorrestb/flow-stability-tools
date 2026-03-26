"""
Temporal linear stability study tool for the inviscid Rayleigh equation.

This script is designed for early graduate-level learners and emphasizes readability.
It analyzes two parallel shear-flow profiles:
  1) U(z) = 0.5 * (1 + tanh(z))
  2) U(z) = z^2

Temporal normal-mode convention used here:
    q'(x,z,t) = q_hat(z) * exp(i * (alpha * x - omega * t))
where alpha is real and omega is complex.

Definitions:
  growth rate = omega_i = Im(omega)
  frequency   = omega_r = Re(omega)

The Rayleigh generalized eigenproblem is assembled as:
    A phi = c B phi,   with c = omega / alpha.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eig


# -----------------------------
# User-tunable study parameters
# -----------------------------
L_DEFAULT = 10.0
N_DEFAULT = 120
ALPHA_LIST = np.linspace(0.1, 1.5, 40)
EIGENVALUE_MAG_THRESHOLD = 1.0e3
PLOT_RESULTS = True


def print_temporal_convention() -> None:
    """Print the sign convention and interpretation of omega parts."""
    print("=" * 80)
    print("Temporal convention:")
    print("  q'(x,z,t) = q_hat(z) * exp(i * (alpha * x - omega * t))")
    print("Definitions:")
    print("  growth_rate = omega_i = Im(omega)")
    print("  frequency   = omega_r = Re(omega)")
    print("=" * 80)


def chebyshev_matrices(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Chebyshev collocation grid and first/second derivative matrices on [-L, L].

    Returns:
        z  : collocation points (N+1,)
        D  : first derivative matrix d/dz, shape (N+1, N+1)
        D2 : second derivative matrix d2/dz2, shape (N+1, N+1)
    """
    if N < 2:
        raise ValueError("N must be at least 2.")

    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)  # Chebyshev points on [-1, 1]
    z = L * x

    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** k)

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D_x = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D_x = D_x - np.diag(np.sum(D_x, axis=1))

    # Scale derivatives from x in [-1,1] to z in [-L,L]: z = L x
    D = D_x / L
    D2 = D @ D

    return z, D, D2


def get_baseflow_sympy(profile_name: str) -> tuple[sp.Symbol, sp.Expr, sp.Expr, sp.Expr]:
    """Return symbolic z, U(z), U'(z), U''(z) for a selected baseflow profile."""
    z = sp.symbols("z", real=True)

    if profile_name == "tanh_shear":
        U = sp.Rational(1, 2) * (1 + sp.tanh(z))
    elif profile_name == "parabolic":
        U = z**2
    else:
        raise ValueError(f"Unknown profile_name: {profile_name}")

    Up = sp.diff(U, z)
    Upp = sp.diff(Up, z)
    return z, U, Up, Upp


def print_continuous_equation() -> None:
    """Print the continuous Rayleigh equation being discretized."""
    print("Continuous inviscid Rayleigh equation being discretized:")
    print("  (U - c) * (phi'' - alpha^2 * phi) - U'' * phi = 0")
    print("  with c = omega / alpha")


def solve_rayleigh_for_alpha(
    alpha: float,
    profile_name: str,
    N: int,
    L: float,
    mag_threshold: float = EIGENVALUE_MAG_THRESHOLD,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Rayleigh generalized eigenproblem for one alpha.

    Returns:
        c_vals        : filtered phase-speed eigenvalues
        omega_vals    : filtered temporal eigenvalues omega = alpha * c
        z             : full collocation grid on [-L, L]
    """
    z_sym, U_expr, _Up_expr, Upp_expr = get_baseflow_sympy(profile_name)
    U_fun = sp.lambdify(z_sym, U_expr, "numpy")
    Upp_fun = sp.lambdify(z_sym, Upp_expr, "numpy")

    z, D, D2 = chebyshev_matrices(N, L)
    I = np.eye(N + 1)

    U_raw = np.asarray(U_fun(z), dtype=np.complex128)
    Upp_raw = np.asarray(Upp_fun(z), dtype=np.complex128)

    # Lambdified SymPy expressions can return scalars (e.g., U''=2 for z^2).
    # Broadcast scalars to the full collocation grid so diagonal assembly works.
    U_vals = np.broadcast_to(U_raw, z.shape).astype(np.complex128, copy=False)
    Upp_vals = np.broadcast_to(Upp_raw, z.shape).astype(np.complex128, copy=False)

    Lap = D2 - (alpha**2) * I
    A = np.diag(U_vals) @ Lap - np.diag(Upp_vals)
    B = Lap.copy()

    # Enforce phi(-L)=phi(L)=0 by eliminating boundary unknowns.
    interior = slice(1, N)
    A_in = A[interior, interior]
    B_in = B[interior, interior]

    if verbose:
        print("-" * 80)
        print(f"Solving for profile='{profile_name}', alpha={alpha:.4f}, N={N}, L={L}")
        print(f"Grid shape: z -> {z.shape}")
        print(f"D shape: {D.shape}, D2 shape: {D2.shape}")
        print("BC enforcement: removed first and last rows/columns")
        print(f"Interior matrix shapes: A_in={A_in.shape}, B_in={B_in.shape}")

    c_raw, eigvecs = eig(A_in, B_in)
    omega_raw = alpha * c_raw

    finite_mask = np.isfinite(c_raw.real) & np.isfinite(c_raw.imag)
    mag_mask = np.abs(c_raw) < mag_threshold
    keep = finite_mask & mag_mask

    c_vals = c_raw[keep]
    omega_vals = omega_raw[keep]
    eigvecs = eigvecs[:, keep]

    if verbose:
        print(f"Raw eigenvalue count: {c_raw.size}")
        print(f"Filtered eigenvalue count: {c_vals.size}")

    return c_vals, omega_vals, z


def dominant_mode_from_omega(omega_vals: np.ndarray) -> tuple[float, float, complex] | tuple[None, None, None]:
    """Return (omega_i_max, omega_r_at_max, omega_complex_at_max)."""
    if omega_vals.size == 0:
        return None, None, None

    idx = np.argmax(np.imag(omega_vals))
    omega_star = omega_vals[idx]
    return float(np.imag(omega_star)), float(np.real(omega_star)), complex(omega_star)


def scan_alpha(
    alpha_list: np.ndarray,
    profile_name: str,
    N: int,
    L: float,
    mag_threshold: float = EIGENVALUE_MAG_THRESHOLD,
) -> dict[str, np.ndarray | float | complex]:
    """
    Scan real alpha values and track dominant growth and associated frequency.

    Returns dict with arrays and global most-unstable summary.
    """
    growth_list = []
    freq_list = []
    omega_star_list = []

    for i, alpha in enumerate(alpha_list):
        verbose = i == 0
        _c_vals, omega_vals, _z = solve_rayleigh_for_alpha(
            alpha=alpha,
            profile_name=profile_name,
            N=N,
            L=L,
            mag_threshold=mag_threshold,
            verbose=verbose,
        )
        omega_i_max, omega_r_at_max, omega_star = dominant_mode_from_omega(omega_vals)

        if omega_i_max is None:
            growth_list.append(np.nan)
            freq_list.append(np.nan)
            omega_star_list.append(np.nan + 1j * np.nan)
        else:
            growth_list.append(omega_i_max)
            freq_list.append(omega_r_at_max)
            omega_star_list.append(omega_star)

    growth_arr = np.array(growth_list, dtype=float)
    freq_arr = np.array(freq_list, dtype=float)
    omega_arr = np.array(omega_star_list, dtype=np.complex128)

    valid = np.isfinite(growth_arr)
    if not np.any(valid):
        alpha_star = np.nan
        omega_i_star = np.nan
        omega_r_star = np.nan
        omega_star = np.nan + 1j * np.nan
    else:
        valid_idx = np.where(valid)[0]
        j_local = np.argmax(growth_arr[valid])
        j = valid_idx[j_local]
        alpha_star = float(alpha_list[j])
        omega_i_star = float(growth_arr[j])
        omega_r_star = float(freq_arr[j])
        omega_star = complex(omega_arr[j])

    return {
        "alpha": alpha_list,
        "growth": growth_arr,
        "freq": freq_arr,
        "omega_star_alpha": omega_arr,
        "alpha_star": alpha_star,
        "omega_i_star": omega_i_star,
        "omega_r_star": omega_r_star,
        "omega_star": omega_star,
    }


def print_profile_sympy_info(profile_name: str) -> None:
    """Print symbolic baseflow expressions used by the solver."""
    z_sym, U_expr, _Up_expr, Upp_expr = get_baseflow_sympy(profile_name)

    print("=" * 80)
    print(f"Profile: {profile_name}")
    print("SymPy U(z):")
    sp.pprint(U_expr)
    print("SymPy U''(z):")
    sp.pprint(Upp_expr)
    print(f"Symbol used: {z_sym}")
    print("=" * 80)


def print_expected_behavior_notes() -> None:
    """Print study notes about what to expect physically and numerically."""
    print("Expected-behavior notes:")
    print("- tanh_shear has an inflection region; inviscid instability bands may appear.")
    print("- parabolic U=z^2 has no inflection point in the interior; strong inviscid")
    print("  Rayleigh instability is not expected, so positive growth can be a numerical")
    print("  artifact if resolution/domain/filtering are not adequate.")
    print("- Always test sensitivity to N and L before trusting growth rates.")


def print_summary(profile_name: str, result: dict[str, np.ndarray | float | complex]) -> None:
    """Print required grading lines for each profile."""
    alpha_star = result["alpha_star"]
    omega_star = result["omega_star"]
    omega_i_star = result["omega_i_star"]
    omega_r_star = result["omega_r_star"]

    print(f"\nResults for profile: {profile_name}")
    print(f"alpha_star = {alpha_star}")
    print(f"omega_star = {omega_r_star} + 1j*{omega_i_star}")
    print(f"growth_rate = {omega_i_star}")
    print(f"frequency = {omega_r_star}")
    print(f"omega_star (complex) = {omega_star}")


def simple_refinement_check(alpha_list: np.ndarray, L: float, N_base: int) -> None:
    """
    Compare max growth for U=z^2 at N and 2N.

    This is a lightweight convergence sanity check.
    """
    print("\nRefinement check for profile 'parabolic' (U=z^2):")
    res_N = scan_alpha(alpha_list, "parabolic", N_base, L)
    res_2N = scan_alpha(alpha_list, "parabolic", 2 * N_base, L)

    gN = float(res_N["omega_i_star"])
    g2N = float(res_2N["omega_i_star"])
    print(f"  max growth at N={N_base}:   {gN:.6e}")
    print(f"  max growth at N={2*N_base}: {g2N:.6e}")

    print("Interpretation:")
    print("  For U=z^2 (no inflection point), inviscid instability is not expected.")
    print("  If positive growth appears, check if it decreases with refinement and")
    print("  with larger L before treating it as physical.")


def plot_scan_results(results: dict[str, dict[str, np.ndarray | float | complex]]) -> None:
    """Plot dominant growth and corresponding frequency versus alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for profile_name, res in results.items():
        alpha = np.asarray(res["alpha"], dtype=float)
        growth = np.asarray(res["growth"], dtype=float)
        freq = np.asarray(res["freq"], dtype=float)

        axes[0].plot(alpha, growth, marker="o", ms=3, label=profile_name)
        axes[1].plot(alpha, freq, marker="o", ms=3, label=profile_name)

    axes[0].axhline(0.0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("max omega_i(alpha)")
    axes[0].set_title("Dominant temporal growth vs alpha")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("omega_r of dominant mode")
    axes[1].set_title("Dominant-mode frequency vs alpha")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def print_next_steps_and_quiz() -> None:
    """Print required learning prompts and a small quiz."""
    print("\nNext steps to explore:")
    print("1. Double N and compare leading eigenvalues: do they converge?")
    print("2. Increase L (for example 10 -> 15 -> 20): do growth curves change?")
    print("3. Narrow the alpha scan near alpha_star and refine the peak location.")
    print("4. Inspect dominant eigenfunctions phi(z): where are they localized?")
    print("5. Check where U(z) is close to c_r: do you suspect critical-layer behavior?")
    print("6. Try different eigenvalue magnitude filters and assess robustness.")
    print("7. Outline how viscosity would change the equation (Orr-Sommerfeld).")
    print("8. Compare temporal analysis here with what spatial analysis would ask.")

    print("\nQuiz questions:")
    print("Easy: In this convention, which part of omega indicates growth or decay?")
    print("Medium: Why does phi(-L)=phi(L)=0 approximate decay at infinity?")
    print("Hard: For A phi = c B phi, what numerical issues arise if B is nearly singular?")


def main() -> None:
    """Run the full study workflow end-to-end."""
    print_temporal_convention()
    print_continuous_equation()
    print_expected_behavior_notes()

    profile_names = ["tanh_shear", "parabolic"]
    all_results: dict[str, dict[str, np.ndarray | float | complex]] = {}

    for profile_name in profile_names:
        print_profile_sympy_info(profile_name)
        res = scan_alpha(ALPHA_LIST, profile_name, N_DEFAULT, L_DEFAULT)
        all_results[profile_name] = res
        print_summary(profile_name, res)

    # Convergence-oriented sanity hook for U=z^2.
    simple_refinement_check(ALPHA_LIST, L_DEFAULT, N_base=60)

    if PLOT_RESULTS:
        plot_scan_results(all_results)

    print_next_steps_and_quiz()


if __name__ == "__main__":
    main()
