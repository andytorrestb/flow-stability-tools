from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runners.case_runner import run_case


if __name__ == "__main__":
    case_dir = Path(__file__).resolve().parent
    context = run_case(case_dir)

    print("Case execution finished.")
    print(f"Results directory: {case_dir / 'results'}")

    analysis = context.get_result("analysis")
    for profile_name, summary in analysis["profiles"].items():
        print(f"{profile_name}: alpha_star={summary['alpha_star']}, omega_star={summary['omega_star']}")

    plot_artifacts = context.results.get("plot_artifacts", {})
    if plot_artifacts:
        print(f"Growth plot: {plot_artifacts.get('growth_plot')}")
        print(f"Frequency plot: {plot_artifacts.get('frequency_plot')}")
