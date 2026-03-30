from __future__ import annotations

import json
from pathlib import Path

from components.plotting import create_scan_plots
from components.solver import RayleighStudySolver
from core.results import scan_result_to_dict
from config.schema import SimulationConfig
from core.case import Case
from core.context import PipelineContext
from pipeline.step import Step


class SolveStep(Step):
    name = "solve"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self, case: Case, context: PipelineContext) -> None:
        solver = RayleighStudySolver(self.config)
        result = solver.solve()
        context.scan_results = result

        output_path = Path(case.results_dir) / "scan_results.json"
        scan_payload = scan_result_to_dict(result)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(scan_payload, handle, indent=2)

        plot_artifacts: dict[str, str] = {}
        if self.config.output.enable_plots:
            plot_artifacts = create_scan_plots(
                profile_results=scan_payload["profiles"],
                output_dir=Path(case.results_dir),
                growth_filename=self.config.output.growth_plot_filename,
                frequency_filename=self.config.output.frequency_plot_filename,
            )

        context.plot_artifacts = plot_artifacts
