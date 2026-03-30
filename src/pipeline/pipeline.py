from __future__ import annotations

from dataclasses import dataclass

from core.case import Case
from core.context import PipelineContext
from core.state import CaseState
from pipeline.step import Step


@dataclass(slots=True)
class Pipeline:
    steps: list[Step]

    def run(self, case: Case, context: PipelineContext) -> None:
        case.set_state(CaseState.RUNNING)
        for step in self.steps:
            step.run(case, context)
        case.set_state(CaseState.COMPLETED)
