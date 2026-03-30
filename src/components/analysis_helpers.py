from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(slots=True)
class SympyExportBundle:
    enabled: bool
    include_latex: bool
    latex_by_profile: Dict[str, Dict[str, str]] = field(default_factory=dict)


def build_profile_summary(
    profile_name: str,
    profile_payload: Dict[str, Any],
    include_symbolic: bool,
    include_latex: bool,
    sympy_bundle: SympyExportBundle,
) -> Dict[str, Any]:
    summary = profile_payload["scan"]["summary"]
    omega_r = summary["frequency"]
    omega_i = summary["growth_rate"]
    profile_summary: Dict[str, Any] = {
        "alpha_star": summary["alpha_star"],
        "omega_star": f"{omega_r} + 1j*{omega_i}",
        "growth_rate": omega_i,
        "frequency": omega_r,
    }

    if not include_symbolic:
        return profile_summary

    sympy_payload = profile_payload.get("sympy", {})
    symbolic_data: Dict[str, str] = {
        "symbol": str(sympy_payload.get("symbol", "z")),
        "U": str(sympy_payload.get("U", "")),
        "U_prime": str(sympy_payload.get("U_prime", "")),
        "U_double_prime": str(sympy_payload.get("U_double_prime", "")),
        "U_pretty": str(sympy_payload.get("U_pretty", "")),
        "U_prime_pretty": str(sympy_payload.get("U_prime_pretty", "")),
        "U_double_prime_pretty": str(sympy_payload.get("U_double_prime_pretty", "")),
    }
    if include_latex:
        symbolic_data["U_latex"] = str(sympy_payload.get("U_latex", ""))
        symbolic_data["U_prime_latex"] = str(sympy_payload.get("U_prime_latex", ""))
        symbolic_data["U_double_prime_latex"] = str(sympy_payload.get("U_double_prime_latex", ""))

    profile_summary["sympy"] = symbolic_data

    if sympy_bundle.enabled and include_latex:
        sympy_bundle.latex_by_profile[profile_name] = {
            "U_latex": symbolic_data.get("U_latex", ""),
            "U_prime_latex": symbolic_data.get("U_prime_latex", ""),
            "U_double_prime_latex": symbolic_data.get("U_double_prime_latex", ""),
        }

    return profile_summary


def export_sympy_pdf_if_enabled(sympy_bundle: SympyExportBundle, output_path: Path) -> None:
    if not sympy_bundle.enabled or not sympy_bundle.latex_by_profile:
        return
    from components.export import export_sympy_pdf

    export_sympy_pdf(sympy_bundle.latex_by_profile, output_path)


def export_static_vtk(profile_name: str, profile_payload: Dict[str, Any], vtk_path: Path) -> None:
    import numpy as np
    from components.vtk_export import export_profile_to_vtk

    z = np.array(profile_payload["z"])
    if "U" not in profile_payload or "Upp" not in profile_payload:
        return
    U = np.array(profile_payload["U"])
    Upp = np.array(profile_payload["Upp"])
    fields = {"U": U, "Upp": Upp}
    export_profile_to_vtk(z, fields, vtk_path, profile_name=profile_name)


def prepare_time_series_inputs(profile_payload: Dict[str, Any]) -> Tuple[Any, Any]:
    import numpy as np

    z = np.array(profile_payload["z"])
    eigenfunction = None
    if "dominant_eigenfunction" in profile_payload and profile_payload["dominant_eigenfunction"] is not None:
        ef_list = profile_payload["dominant_eigenfunction"]
        eigenfunction = np.array([complex(r, i) for r, i in ef_list])
    return z, eigenfunction


def normalize_eigenfunction(eigenfunction, z_len: int):
    import numpy as np

    n_int = len(eigenfunction)
    if n_int == z_len - 2:
        eigenfunction_padded = np.zeros(z_len, dtype=complex)
        eigenfunction_padded[1:-1] = eigenfunction
    elif n_int == z_len:
        eigenfunction_padded = eigenfunction
    else:
        raise ValueError(
            f"dominant_eigenfunction length {n_int} incompatible with z length {z_len} (expected z or z-2)"
        )
    if np.max(np.abs(eigenfunction_padded)) > 0:
        eigenfunction_padded = eigenfunction_padded / np.max(np.abs(eigenfunction_padded))
    return eigenfunction_padded


def export_time_series_artifacts(
    profile_name: str,
    profile_payload: Dict[str, Any],
    summary: Dict[str, Any],
    artifact_manager,
    numerical_L: float,
    perturbation_amplitude: float,
    overlay_initial_profile: bool,
    initial_profile_label: str,
    export_vtk_enabled: bool,
    export_time_series_mp4_enabled: bool,
) -> None:
    import numpy as np
    from components.spectral import chebyshev_matrices
    from components.time_series_export import (
        reconstruct_time_series,
        reconstruct_velocity_time_series,
        export_time_series_vtk,
    )

    if not export_vtk_enabled:
        return

    z, eigenfunction = prepare_time_series_inputs(profile_payload)
    if eigenfunction is None:
        print(f"[DEBUG] Skipping time series export for {profile_name}: dominant_eigenfunction is None")
        return

    eigenfunction_padded = normalize_eigenfunction(eigenfunction, z_len=len(z))

    omega_r = summary["frequency"]
    omega_i = summary["growth_rate"]
    omega = omega_r + 1j * omega_i
    t_grid = np.linspace(0, 20, 101)

    try:
        _z_ref, D, _D2 = chebyshev_matrices(N=len(z) - 1, L=numerical_L)
    except Exception as e:  # pragma: no cover - safety net
        raise RuntimeError(f"Failed to build derivative matrix for time series export: {e}")

    qzt = reconstruct_time_series(
        z,
        eigenfunction_padded,
        omega,
        t_grid,
        amplitude=perturbation_amplitude,
    )
    uzt = reconstruct_velocity_time_series(
        z,
        eigenfunction_padded,
        omega,
        t_grid,
        amplitude=perturbation_amplitude,
        D=D,
    )

    ts_dir = artifact_manager.time_series_dir(profile_name)
    export_time_series_vtk(
        z,
        {"q": qzt, "u": uzt},
        t_grid,
        ts_dir,
        base_filename="time_series",
    )
    print(f"[DEBUG] Time series VTK export complete for {profile_name} at {ts_dir}")

    if not export_time_series_mp4_enabled:
        return

    try:
        from visualization.baseflow_time_series_plot import plot_baseflow_time_series_to_mp4
        from visualization.vtk_time_series_plot import plot_vtk_time_series_to_mp4
    except Exception as e:  # pragma: no cover - avoids hard failure in headless env
        print(f"[Time Series MP4 Export] Skipped for {profile_name}: {e}")
        return

    try:
        plot_vtk_time_series_to_mp4(
            ts_dir,
            field="q_real",
            output_mp4=artifact_manager.time_series_mp4_filename,
            figsize=(8, 4),
            interval=80,
            dpi=180,
            style="darkgrid",
        )
        plot_vtk_time_series_to_mp4(
            ts_dir,
            field="u_real",
            output_mp4=artifact_manager.velocity_time_series_mp4_filename,
            figsize=(8, 4),
            interval=80,
            dpi=180,
            style="darkgrid",
        )

        initial_profile = None
        if overlay_initial_profile and "U" in profile_payload:
            initial_profile = np.array(profile_payload["U"])
        U_base = np.array(profile_payload.get("U", np.zeros_like(z)))
        U_total = U_base[:, None] + np.real(uzt)
        plot_baseflow_time_series_to_mp4(
            z,
            U_total,
            t_grid,
            output_mp4=artifact_manager.velocity_time_series_mp4_path(profile_name),
            figsize=(8, 4),
            interval=80,
            dpi=180,
            style="darkgrid",
            initial_profile=initial_profile,
            initial_profile_label=initial_profile_label,
        )
    except Exception as e:  # pragma: no cover - runtime visualization guard
        print(f"[Time Series MP4 Export] Failed for {profile_name}: {e}")