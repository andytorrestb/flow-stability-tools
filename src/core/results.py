from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ProfileScanResult:
    z: List[float]
    U: List[float]
    Upp: List[float]
    sympy: Dict[str, Any]
    scan: Dict[str, Any]
    dominant_eigenfunction: Optional[List[List[float]]] = None


@dataclass(slots=True)
class ScanResult:
    temporal_convention: str
    equation: str
    expected_behavior: Dict[str, str]
    profiles: Dict[str, ProfileScanResult]
    refinement: Dict[str, Any]


@dataclass(slots=True)
class AnalysisSummary:
    temporal_convention: str
    equation: str
    expected_behavior: Dict[str, str]
    profiles: Dict[str, Dict[str, Any]]
    refinement: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


def scan_result_to_dict(scan_result: ScanResult) -> Dict[str, Any]:
    return asdict(scan_result)


def analysis_summary_to_dict(summary: AnalysisSummary) -> Dict[str, Any]:
    return asdict(summary)