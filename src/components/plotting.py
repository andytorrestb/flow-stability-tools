from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def create_scan_plots(
    profile_results: dict[str, dict[str, list[float] | dict[str, float] | dict[str, str]]],
    output_dir: Path,
    growth_filename: str,
    frequency_filename: str,
) -> dict[str, str]:
    """Create and save growth/frequency versus alpha plots across all profiles."""
    output_dir.mkdir(parents=True, exist_ok=True)

    growth_path = output_dir / growth_filename
    freq_path = output_dir / frequency_filename

    plt.figure(figsize=(8, 5))
    for profile_name, payload in profile_results.items():
        alpha = payload["scan"]["alpha"]
        growth = payload["scan"]["dominant_growth"]
        plt.plot(alpha, growth, marker="o", ms=3, label=profile_name)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("max omega_i(alpha)")
    plt.title("Dominant temporal growth vs alpha")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(growth_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for profile_name, payload in profile_results.items():
        alpha = payload["scan"]["alpha"]
        freq = payload["scan"]["dominant_frequency"]
        plt.plot(alpha, freq, marker="o", ms=3, label=profile_name)
    plt.xlabel("alpha")
    plt.ylabel("omega_r of dominant mode")
    plt.title("Dominant-mode frequency vs alpha")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(freq_path, dpi=150)
    plt.close()

    return {
        "growth_plot": str(growth_path),
        "frequency_plot": str(freq_path),
    }
