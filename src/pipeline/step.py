from __future__ import annotations

from abc import ABC, abstractmethod

from core.case import Case
from core.context import Context


class Step(ABC):
    name: str

    @abstractmethod
    def run(self, case: Case, context: Context) -> None:
        raise NotImplementedError
