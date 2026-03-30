from __future__ import annotations

from typing import Callable, Dict

from analysis.helpers import (
    SympyExportBundle,
    build_profile_summary,
    export_static_vtk,
    export_sympy_pdf_if_enabled,
    export_time_series_artifacts,
)
from fst_io.artifacts import ArtifactManager
from core.results import AnalysisSummary, analysis_summary_to_dict, scan_result_to_dict


AnalysisStrategy = Callable[["Case", "PipelineContext", "SimulationConfig", ArtifactManager], None]


class AnalysisRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, AnalysisStrategy] = {}

    def register(self, name: str, strategy: AnalysisStrategy) -> None:
        self._registry[name] = strategy

    def get(self, name: str = "default") -> AnalysisStrategy:
        if name not in self._registry:
            raise KeyError(f"No analysis strategy registered for '{name}'")
        return self._registry[name]


def _default_analysis_strategy(case, context, config, artifact_manager: ArtifactManager) -> None:
    import json
    from datetime import datetime, timezone

    if context.scan_results is None:
        raise RuntimeError("scan_results missing in context")

    scan_payload = scan_result_to_dict(context.scan_results)
    profiles = scan_payload["profiles"]

    include_symbolic = config.output.include_symbolic_in_profile_summaries
    include_latex = config.output.include_symbolic_latex
    show_symbolic_console = config.output.show_symbolic_in_console
    export_sympy_pdf_enabled = getattr(config.output, "export_sympy_pdf", False)
    export_vtk_enabled = getattr(config.output, "export_vtk", False)
    export_time_series_mp4_enabled = getattr(config.output, "export_time_series_mp4", False)
    perturbation_amplitude = getattr(config.output, "perturbation_amplitude", 1.0e-3)
    overlay_initial_profile = getattr(config.output, "overlay_initial_profile", False)
    initial_profile_label = getattr(config.output, "initial_profile_label", "Initial velocity profile")

    sympy_bundle = SympyExportBundle(
        enabled=export_sympy_pdf_enabled,
        include_latex=include_latex,
    )

    summaries: dict[str, dict] = {}
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
                    numerical_L=config.numerical.L,
                    perturbation_amplitude=perturbation_amplitude,
                    overlay_initial_profile=overlay_initial_profile,
                    initial_profile_label=initial_profile_label,
                    export_vtk_enabled=export_vtk_enabled,
                    export_time_series_mp4_enabled=export_time_series_mp4_enabled,
                )
            except Exception as e:
                print(f"[Time Series Export] Failed for {profile_name}: {e}")

        if show_symbolic_console and "sympy" in profile_summary:
            sympy_payload = profile_summary.get("sympy")
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

        print(f"[{profile_name}] alpha_star={profile_summary['alpha_star']}")
        print(f"[{profile_name}] omega_star={profile_summary['omega_star']}")
        print(f"[{profile_name}] growth_rate={profile_summary['growth_rate']}")
        print(f"[{profile_name}] frequency={profile_summary['frequency']}")

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

    summary_path = artifact_manager.results_dir / "profile_summaries.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis_summary_to_dict(analysis_summary), handle, indent=2)

    export_sympy_pdf_if_enabled(sympy_bundle, artifact_manager.sympy_pdf_path())

    metadata_path = context.metadata_path
    if metadata_path:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["state"] = "COMPLETED"
        metadata["profiles"] = list(config.solver.profiles)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


_DEFAULT_ANALYSIS_REGISTRY = AnalysisRegistry()
_DEFAULT_ANALYSIS_REGISTRY.register("default", _default_analysis_strategy)


def get_analysis_registry() -> AnalysisRegistry:
    return _DEFAULT_ANALYSIS_REGISTRY
