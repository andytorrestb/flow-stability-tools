from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.state import CaseState


@dataclass(slots=True)
class Case:
    root_dir: Path
    name: str
    state: CaseState = CaseState.INITIALIZED

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    def ensure_directories(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def set_state(self, state: CaseState) -> None:
        self.state = state
