from __future__ import annotations

from pathlib import Path

from analysis.registry import get_analysis_registry
from fst_io.artifacts import ArtifactManager
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

        strategy = get_analysis_registry().get("default")

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

        strategy(case=case, context=context, config=self.config, artifact_manager=artifact_manager)
