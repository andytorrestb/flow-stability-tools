from __future__ import annotations

import json
from pathlib import Path

from components.plotting import create_scan_plots
from components.solver import RayleighStudySolver
from config.schema import SimulationConfig
from core.case import Case
from core.context import Context
from pipeline.step import Step


class SolveStep(Step):
    name = "solve"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def run(self, case: Case, context: Context) -> None:
        solver = RayleighStudySolver(self.config)
        result = solver.solve()
        context.set_result("scan_results", result)

        output_path = Path(case.results_dir) / "scan_results.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        plot_artifacts: dict[str, str] = {}
        if self.config.output.enable_plots:
            plot_artifacts = create_scan_plots(
                profile_results=result["profiles"],
                output_dir=Path(case.results_dir),
                growth_filename=self.config.output.growth_plot_filename,
                frequency_filename=self.config.output.frequency_plot_filename,
            )

        context.set_result("plot_artifacts", plot_artifacts)
