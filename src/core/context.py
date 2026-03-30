from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from core.results import AnalysisSummary, ScanResult


@dataclass(slots=True)
class PipelineContext:
    started_at: Optional[str] = None
    metadata_path: Optional[Path] = None
    scan_results: Optional[ScanResult] = None
    analysis: Optional[AnalysisSummary] = None
    plot_artifacts: Dict[str, str] = field(default_factory=dict)
