from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Context:
    data: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def set_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get_data(self, key: str) -> Any:
        return self.data[key]

    def set_result(self, key: str, value: Any) -> None:
        self.results[key] = value

    def get_result(self, key: str) -> Any:
        return self.results[key]
