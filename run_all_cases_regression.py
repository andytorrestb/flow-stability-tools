from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runners.case_runner import run_case


REQUIRED_ARTIFACTS = (
    "run_metadata.json",
    "scan_results.json",
    "profile_summaries.json",
)


@dataclass(slots=True)
class CaseRunResult:
    case_name: str
    passed: bool
    elapsed_seconds: float
    reason: str = ""


def discover_case_dirs(cases_root: Path, selected_cases: set[str] | None) -> list[Path]:
    case_dirs: list[Path] = []
    for path in sorted(cases_root.iterdir()):
        if not path.is_dir():
            continue
        if selected_cases is not None and path.name not in selected_cases:
            continue
        if (path / "config.yaml").exists():
            case_dirs.append(path)
    return case_dirs


def validate_case_artifacts(case_dir: Path) -> tuple[bool, str]:
    results_dir = case_dir / "results"
    missing = [name for name in REQUIRED_ARTIFACTS if not (results_dir / name).exists()]
    if missing:
        return False, f"missing artifacts: {', '.join(missing)}"

    metadata_path = results_dir / "run_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    state = metadata.get("state")
    if state != "COMPLETED":
        return False, f"unexpected run state in metadata: {state!r}"

    return True, ""


def run_case_with_checks(case_dir: Path) -> CaseRunResult:
    started = time.perf_counter()
    try:
        run_case(case_dir)
        passed, reason = validate_case_artifacts(case_dir)
    except Exception as exc:
        passed = False
        reason = f"exception: {exc}"

    elapsed = time.perf_counter() - started
    return CaseRunResult(
        case_name=case_dir.name,
        passed=passed,
        elapsed_seconds=elapsed,
        reason=reason,
    )


def print_summary(results: list[CaseRunResult]) -> None:
    print("\n=== Regression Summary ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4s} | {result.case_name:10s} | {result.elapsed_seconds:7.2f}s", end="")
        if result.reason:
            print(f" | {result.reason}")
        else:
            print()

    passed_count = sum(1 for item in results if item.passed)
    print(f"\nPassed {passed_count}/{len(results)} cases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all configured cases as a regression check and validate core artifacts."
    )
    parser.add_argument(
        "--cases-root",
        default="cases",
        help="Path to cases directory (default: cases)",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Optional case folder name to run; repeat flag to run multiple specific cases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_root = (PROJECT_ROOT / args.cases_root).resolve()

    if not cases_root.exists():
        print(f"Cases root does not exist: {cases_root}")
        return 2

    selected_cases = set(args.cases) if args.cases else None
    case_dirs = discover_case_dirs(cases_root=cases_root, selected_cases=selected_cases)

    if not case_dirs:
        print("No case directories with config.yaml were found.")
        return 2

    print(f"Running {len(case_dirs)} case(s) from {cases_root}")
    results = [run_case_with_checks(case_dir) for case_dir in case_dirs]
    print_summary(results)

    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
