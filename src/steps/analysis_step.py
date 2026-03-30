from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.results import AnalysisSummary, analysis_summary_to_dict, scan_result_to_dict
from components.artifacts import ArtifactManager
from components.analysis_helpers import (
    SympyExportBundle,
    build_profile_summary,
    export_sympy_pdf_if_enabled,
    export_static_vtk,
    export_time_series_artifacts,
)
from config.schema import SimulationConfig
from core.case import Case
from core.context import PipelineContext
from pipeline.step import Step


class AnalysisStep(Step):
    name = "analysis"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config


    def run(self, case: Case, context: PipelineContext) -> None:
        if context.scan_results is None:
            raise RuntimeError("scan_results missing in context")

        scan_payload = scan_result_to_dict(context.scan_results)
        profiles = scan_payload["profiles"]

        artifact_manager = ArtifactManager(
            results_dir=Path(case.results_dir),
            growth_plot_filename=self.config.output.growth_plot_filename,
            frequency_plot_filename=self.config.output.frequency_plot_filename,
            sympy_pdf_filename=getattr(self.config.output, "sympy_pdf_filename", "symbolic_expressions.pdf"),
            vtk_filename_pattern=getattr(self.config.output, "vtk_filename_pattern", "{profile}_field.vtk"),
            time_series_mp4_filename=getattr(self.config.output, "time_series_mp4_filename", "time_series.mp4"),
            velocity_time_series_mp4_filename=getattr(
                self.config.output, "velocity_time_series_mp4_filename", "velocity_time_series.mp4"
            ),
        )

        include_symbolic = self.config.output.include_symbolic_in_profile_summaries
        include_latex = self.config.output.include_symbolic_latex
        show_symbolic_console = self.config.output.show_symbolic_in_console
        export_sympy_pdf_enabled = getattr(self.config.output, "export_sympy_pdf", False)
        export_vtk_enabled = getattr(self.config.output, "export_vtk", False)
        export_time_series_mp4_enabled = getattr(self.config.output, "export_time_series_mp4", False)
        perturbation_amplitude = getattr(self.config.output, "perturbation_amplitude", 1.0e-3)
        overlay_initial_profile = getattr(self.config.output, "overlay_initial_profile", False)
        initial_profile_label = getattr(self.config.output, "initial_profile_label", "Initial velocity profile")

        sympy_bundle = SympyExportBundle(
            enabled=export_sympy_pdf_enabled,
            include_latex=include_latex,
        )

        summaries: dict[str, dict[str, float | str | dict[str, str]]] = {}
        for profile_name, profile_payload in profiles.items():
            profile_summary = build_profile_summary(
                profile_name=profile_name,
                profile_payload=profile_payload,
                include_symbolic=include_symbolic,
                include_latex=include_latex,
                sympy_bundle=sympy_bundle,
            )

            summaries[profile_name] = profile_summary

            if export_vtk_enabled:
                try:
                    export_static_vtk(
                        profile_name=profile_name,
                        profile_payload=profile_payload,
                        vtk_path=artifact_manager.vtk_field_path(profile_name),
                    )
                except Exception as e:
                    print(f"[VTK Export] Failed for {profile_name}: {e}")

                try:
                    export_time_series_artifacts(
                        profile_name=profile_name,
                        profile_payload=profile_payload,
                        summary=profile_summary,
                        artifact_manager=artifact_manager,
                        numerical_L=self.config.numerical.L,
                        perturbation_amplitude=perturbation_amplitude,
                        overlay_initial_profile=overlay_initial_profile,
                        initial_profile_label=initial_profile_label,
                        export_vtk_enabled=export_vtk_enabled,
                        export_time_series_mp4_enabled=export_time_series_mp4_enabled,
                    )
                except Exception as e:
                    print(f"[Time Series Export] Failed for {profile_name}: {e}")

        # Export PDF if enabled
        export_sympy_pdf_if_enabled(sympy_bundle, artifact_manager.sympy_pdf_path())

        analysis_summary = AnalysisSummary(
            temporal_convention=scan_payload["temporal_convention"],
            equation=scan_payload["equation"],
            expected_behavior=scan_payload["expected_behavior"],
            profiles=summaries,
            refinement=scan_payload["refinement"],
            notes=[
                "tanh_shear may exhibit an unstable band depending on N/L and alpha range.",
                "parabolic has no inflection point, so strong positive growth should be treated with caution.",
                "Always test sensitivity to N and L before drawing physical conclusions.",
            ],
        )

        context.analysis = analysis_summary

        summary_path = Path(case.results_dir) / "profile_summaries.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(analysis_summary_to_dict(analysis_summary), handle, indent=2)

        metadata_path = context.metadata_path
        if metadata_path:
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
