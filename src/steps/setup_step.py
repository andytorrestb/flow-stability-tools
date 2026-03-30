from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.case import Case
from core.context import PipelineContext
from pipeline.step import Step


class SetupStep(Step):
    name = "setup"

    def run(self, case: Case, context: PipelineContext) -> None:
        case.ensure_directories()
        started_at = datetime.now(timezone.utc).isoformat()
        context.started_at = started_at

        metadata = {
            "case_name": case.name,
            "state": "INITIALIZED",
            "started_at": started_at,
        }
        metadata_path = Path(case.results_dir) / "run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        context.metadata_path = metadata_path
