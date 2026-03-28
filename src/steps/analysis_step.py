from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config.schema import SimulationConfig
from core.case import Case
from core.context import Context
from pipeline.step import Step


class AnalysisStep(Step):
    name = "analysis"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config


    def run(self, case: Case, context: Context) -> None:
        from components.export import export_sympy_pdf
        from components.vtk_export import export_profile_to_vtk
        from components.time_series_export import reconstruct_time_series, export_time_series_vtk
        import numpy as np
        scan_results = context.get_result("scan_results")
        profiles = scan_results["profiles"]

        include_symbolic = self.config.output.include_symbolic_in_profile_summaries
        include_latex = self.config.output.include_symbolic_latex
        show_symbolic_console = self.config.output.show_symbolic_in_console
        export_sympy_pdf_enabled = getattr(self.config.output, "export_sympy_pdf", False)
        sympy_pdf_filename = getattr(self.config.output, "sympy_pdf_filename", "symbolic_expressions.pdf")
        export_vtk_enabled = getattr(self.config.output, "export_vtk", False)
        vtk_filename_pattern = getattr(self.config.output, "vtk_filename_pattern", "{profile}_field.vtk")
        export_time_series_mp4_enabled = getattr(self.config.output, "export_time_series_mp4", False)
        time_series_mp4_filename = getattr(self.config.output, "time_series_mp4_filename", "time_series.mp4")
        velocity_time_series_mp4_filename = getattr(
            self.config.output, "velocity_time_series_mp4_filename", "velocity_time_series.mp4"
        )
        overlay_initial_profile = getattr(self.config.output, "overlay_initial_profile", False)
        initial_profile_label = getattr(self.config.output, "initial_profile_label", "Initial velocity profile")
        # Import animation function only if needed
        if export_time_series_mp4_enabled:
            from visualization.baseflow_time_series_plot import plot_baseflow_time_series_to_mp4
            from visualization.vtk_time_series_plot import plot_vtk_time_series_to_mp4

        summaries: dict[str, dict[str, float | str | dict[str, str]]] = {}
        sympy_latex_data: dict[str, dict[str, str]] = {}
        for profile_name, profile_payload in profiles.items():
            summary = profile_payload["scan"]["summary"]
            omega_r = summary["frequency"]
            omega_i = summary["growth_rate"]
            profile_summary: dict[str, float | str | dict[str, str]] = {
                "alpha_star": summary["alpha_star"],
                "omega_star": f"{omega_r} + 1j*{omega_i}",
                "growth_rate": omega_i,
                "frequency": omega_r,
            }

            if include_symbolic:
                sympy_payload = profile_payload.get("sympy", {})
                symbolic_data: dict[str, str] = {
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

                # Collect LaTeX for PDF export if enabled
                if export_sympy_pdf_enabled:
                    sympy_latex_data[profile_name] = {
                        "U_latex": symbolic_data.get("U_latex", ""),
                        "U_prime_latex": symbolic_data.get("U_prime_latex", ""),
                        "U_double_prime_latex": symbolic_data.get("U_double_prime_latex", ""),
                    }

            summaries[profile_name] = profile_summary

            # VTK export for each profile (static fields)
            if export_vtk_enabled:
                try:
                    z = np.array(profile_payload["z"])
                    U = None
                    Upp = None
                    sympy_payload = profile_payload.get("sympy", {})
                    if "U" in profile_payload and "Upp" in profile_payload:
                        U = np.array(profile_payload["U"])
                        Upp = np.array(profile_payload["Upp"])
                    if U is not None and Upp is not None:
                        fields = {"U": U, "Upp": Upp}
                        vtk_filename = vtk_filename_pattern.format(profile=profile_name)
                        vtk_path = Path(case.results_dir) / vtk_filename
                        export_profile_to_vtk(z, fields, vtk_path, profile_name=profile_name)
                except Exception as e:
                    print(f"[VTK Export] Failed for {profile_name}: {e}")

            # Time series VTK export for each profile (dominant mode)
            if export_vtk_enabled:
                try:
                    z = np.array(profile_payload["z"])
                    initial_profile = None
                    if overlay_initial_profile:
                        if "U" in profile_payload:
                            initial_profile = np.array(profile_payload["U"])
                        else:
                            print(f"[DEBUG] Initial profile not available for {profile_name}; skipping overlay")
                    eigenfunction = None
                    if "dominant_eigenfunction" in profile_payload and profile_payload["dominant_eigenfunction"] is not None:
                        # Convert from [[real, imag], ...] to complex numpy array
                        ef_list = profile_payload["dominant_eigenfunction"]
                        eigenfunction = np.array([complex(r, i) for r, i in ef_list])
                        # Optionally normalize for visibility
                        if np.max(np.abs(eigenfunction)) > 0:
                            eigenfunction = eigenfunction / np.max(np.abs(eigenfunction))
                    else:
                        print(f"[DEBUG] Skipping time series export for {profile_name}: dominant_eigenfunction is None")
                        continue
                    omega_r = summary["frequency"]
                    omega_i = summary["growth_rate"]
                    omega = omega_r + 1j * omega_i
                    t_grid = np.linspace(0, 20, 101)  # 101 time steps from t=0 to t=20
                    print(f"[DEBUG] Exporting time series for {profile_name}: z.shape={z.shape}, eigenfunction.shape={eigenfunction.shape}, omega={omega}, t_grid.shape={t_grid.shape}")
                    # Pad eigenfunction and qzt to match z length (Chebyshev grid)
                    n_z = len(z)
                    n_int = len(eigenfunction)
                    if n_int == n_z - 2:
                        # Pad with zeros at boundaries
                        eigenfunction_padded = np.zeros(n_z, dtype=complex)
                        eigenfunction_padded[1:-1] = eigenfunction
                    else:
                        eigenfunction_padded = eigenfunction
                    qzt = reconstruct_time_series(z, eigenfunction_padded, omega, t_grid)
                    print(f"[DEBUG] qzt.shape={qzt.shape}, qzt.max={np.max(np.abs(qzt))}, qzt.min={np.min(np.abs(qzt))}")
                    ts_dir = Path(case.results_dir) / f"{profile_name}_time_series"
                    export_time_series_vtk(z, qzt, t_grid, ts_dir, base_filename="q_time_series")
                    print(f"[DEBUG] Time series VTK export complete for {profile_name} at {ts_dir}")

                    # Export .mp4 animation if enabled
                    if export_time_series_mp4_enabled:
                        try:
                            # Disturbance animation q(z,t) (original plot)
                            plot_vtk_time_series_to_mp4(
                                ts_dir,
                                field="q_real",
                                output_mp4=time_series_mp4_filename,
                                figsize=(8, 4),
                                interval=80,
                                dpi=180,
                                style="darkgrid",
                            )
                            # Velocity animation U(z,t) = U_base + Re(q), with optional overlay
                            U_base = np.array(profile_payload.get("U", np.zeros_like(z)))
                            U_total = U_base[:, None] + np.real(qzt)
                            plot_baseflow_time_series_to_mp4(
                                z,
                                U_total,
                                t_grid,
                                output_mp4=Path(ts_dir) / velocity_time_series_mp4_filename,
                                figsize=(8, 4),
                                interval=80,
                                dpi=180,
                                style="darkgrid",
                                initial_profile=initial_profile,
                                initial_profile_label=initial_profile_label,
                            )
                        except Exception as e:
                            print(f"[Time Series MP4 Export] Failed for {profile_name}: {e}")
                except Exception as e:
                    print(f"[Time Series VTK Export] Failed for {profile_name}: {e}")

        # Export PDF if enabled
        if export_sympy_pdf_enabled and sympy_latex_data:
            pdf_path = Path(case.results_dir) / sympy_pdf_filename
            export_sympy_pdf(sympy_latex_data, pdf_path)

        analysis_payload = {
            "temporal_convention": scan_results["temporal_convention"],
            "equation": scan_results["equation"],
            "expected_behavior": scan_results["expected_behavior"],
            "profiles": summaries,
            "refinement": scan_results["refinement"],
            "notes": [
                "tanh_shear may exhibit an unstable band depending on N/L and alpha range.",
                "parabolic has no inflection point, so strong positive growth should be treated with caution.",
                "Always test sensitivity to N and L before drawing physical conclusions.",
            ],
        }

        context.set_result("analysis", analysis_payload)

        summary_path = Path(case.results_dir) / "profile_summaries.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(analysis_payload, handle, indent=2)

        metadata_path_raw = context.data.get("metadata_path")
        if metadata_path_raw:
            metadata_path = Path(metadata_path_raw)
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            metadata["state"] = "COMPLETED"
            metadata["profiles"] = list(self.config.solver.profiles)
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

        for profile_name, summary in summaries.items():
            if show_symbolic_console:
                sympy_payload = summary.get("sympy")
                if isinstance(sympy_payload, dict):
                    print("=" * 80)
                    print(f"Profile: {profile_name}")
                    print("SymPy U(z):")
                    print(sympy_payload.get("U_pretty") or sympy_payload.get("U", ""))
                    print("SymPy U'(z):")
                    print(sympy_payload.get("U_prime_pretty") or sympy_payload.get("U_prime", ""))
                    print("SymPy U''(z):")
                    print(sympy_payload.get("U_double_prime_pretty") or sympy_payload.get("U_double_prime", ""))
                    print(f"Symbol used: {sympy_payload.get('symbol', 'z')}")
                    print("=" * 80)

            print(f"[{profile_name}] alpha_star={summary['alpha_star']}")
            print(f"[{profile_name}] omega_star={summary['omega_star']}")
            print(f"[{profile_name}] growth_rate={summary['growth_rate']}")
            print(f"[{profile_name}] frequency={summary['frequency']}")
