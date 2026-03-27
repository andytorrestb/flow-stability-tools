from __future__ import annotations

from enum import Enum


class CaseState(str, Enum):
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
