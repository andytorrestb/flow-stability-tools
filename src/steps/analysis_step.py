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
        scan_results = context.get_result("scan_results")
        profiles = scan_results["profiles"]

        include_symbolic = self.config.output.include_symbolic_in_profile_summaries
        include_latex = self.config.output.include_symbolic_latex
        show_symbolic_console = self.config.output.show_symbolic_in_console
        export_sympy_pdf_enabled = getattr(self.config.output, "export_sympy_pdf", False)
        sympy_pdf_filename = getattr(self.config.output, "sympy_pdf_filename", "symbolic_expressions.pdf")

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
