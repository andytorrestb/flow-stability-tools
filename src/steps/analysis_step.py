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
        scan_results = context.get_result("scan_results")
        profiles = scan_results["profiles"]

        summaries: dict[str, dict[str, float | str]] = {}
        for profile_name, profile_payload in profiles.items():
            summary = profile_payload["scan"]["summary"]
            omega_r = summary["frequency"]
            omega_i = summary["growth_rate"]
            summaries[profile_name] = {
                "alpha_star": summary["alpha_star"],
                "omega_star": f"{omega_r} + 1j*{omega_i}",
                "growth_rate": omega_i,
                "frequency": omega_r,
            }

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
            print(f"[{profile_name}] alpha_star={summary['alpha_star']}")
            print(f"[{profile_name}] omega_star={summary['omega_star']}")
            print(f"[{profile_name}] growth_rate={summary['growth_rate']}")
            print(f"[{profile_name}] frequency={summary['frequency']}")
