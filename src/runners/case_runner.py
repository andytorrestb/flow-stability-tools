from __future__ import annotations

from pathlib import Path

from config.loader import load_config
from core.case import Case
from core.context import PipelineContext
from pipeline.pipeline import Pipeline
from steps.analysis_step import AnalysisStep
from steps.setup_step import SetupStep
from steps.solve_step import SolveStep


def _build_steps(config) -> list:
    step_map = {
        "setup": lambda: SetupStep(),
        "solve": lambda: SolveStep(config),
        "analysis": lambda: AnalysisStep(config),
    }

    return [step_map[name]() for name in config.pipeline.steps]


def run_case(case_dir: Path) -> PipelineContext:
    config_path = case_dir / "config.yaml"
    config = load_config(config_path)

    case = Case(root_dir=case_dir, name=config.case_name)
    context = PipelineContext()

    pipeline = Pipeline(steps=_build_steps(config))
    pipeline.run(case=case, context=context)

    return context
