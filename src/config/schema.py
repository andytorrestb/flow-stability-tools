from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BaseflowProfile = Literal["tanh_shear", "parabolic"]


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["rayleigh_inviscid"] = "rayleigh_inviscid"
    profiles: list[BaseflowProfile] = Field(min_length=1)


class NumericalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    L: float = Field(gt=0)
    N: int = Field(ge=8)
    eigenvalue_mag_threshold: float = Field(default=1.0e3, gt=0)


class AlphaScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    alpha_min: float = Field(gt=0)
    alpha_max: float = Field(gt=0)
    alpha_count: int = Field(ge=3)

    @model_validator(mode="after")
    def validate_alpha_window(self) -> "AlphaScanConfig":
        if self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must be greater than alpha_min")
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enable_plots: bool = True
    growth_plot_filename: str = "growth_vs_alpha.png"
    frequency_plot_filename: str = "frequency_vs_alpha.png"
    show_symbolic_in_console: bool = True
    include_symbolic_in_profile_summaries: bool = True
    include_symbolic_latex: bool = False
    export_sympy_pdf: bool = False
    sympy_pdf_filename: str = "symbolic_expressions.pdf"
    export_vtk: bool = False
    vtk_filename_pattern: str = "{profile}_field.vtk"
    export_time_series_mp4: bool = False
    time_series_mp4_filename: str = "time_series.mp4"
    velocity_time_series_mp4_filename: str = "velocity_time_series.mp4"
    perturbation_amplitude: float = 1.0e-3
    overlay_initial_profile: bool = False
    initial_profile_label: str = "Initial velocity profile"


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    profile: BaseflowProfile = "parabolic"
    N_base: int = Field(default=60, ge=8)
    multiplier: int = Field(default=2, ge=2)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    steps: list[Literal["setup", "solve", "analysis"]]


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_name: str = Field(min_length=1)
    solver: SolverConfig
    numerical: NumericalConfig
    alpha_scan: AlphaScanConfig
    output: OutputConfig = OutputConfig()
    refinement: RefinementConfig = RefinementConfig()
    pipeline: PipelineConfig
