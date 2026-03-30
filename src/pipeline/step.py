from __future__ import annotations

from abc import ABC, abstractmethod

from core.case import Case
from core.context import PipelineContext


class Step(ABC):
    name: str

    @abstractmethod
    def run(self, case: Case, context: PipelineContext) -> None:
        raise NotImplementedError
