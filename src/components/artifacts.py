from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ArtifactManager:
    """Centralizes artifact path construction while preserving existing defaults."""

    results_dir: Path
    growth_plot_filename: str
    frequency_plot_filename: str
    sympy_pdf_filename: str
    vtk_filename_pattern: str
    time_series_mp4_filename: str
    velocity_time_series_mp4_filename: str

    def growth_plot_path(self) -> Path:
        return self.results_dir / self.growth_plot_filename

    def frequency_plot_path(self) -> Path:
        return self.results_dir / self.frequency_plot_filename

    def sympy_pdf_path(self) -> Path:
        return self.results_dir / self.sympy_pdf_filename

    def vtk_field_path(self, profile: str) -> Path:
        return self.results_dir / self.vtk_filename_pattern.format(profile=profile)

    def time_series_dir(self, profile: str) -> Path:
        return self.results_dir / f"{profile}_time_series"

    def time_series_mp4_path(self, profile: str) -> Path:
        return self.time_series_dir(profile) / self.time_series_mp4_filename

    def velocity_time_series_mp4_path(self, profile: str) -> Path:
        return self.time_series_dir(profile) / self.velocity_time_series_mp4_filename