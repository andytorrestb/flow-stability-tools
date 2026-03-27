from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from components.baseflow import evaluate_baseflow
from components.scan import alpha_scan
from components.spectral import chebyshev_matrices
from config.schema import SimulationConfig


class Solver(ABC):
    @abstractmethod
    def solve(self) -> dict:
        raise NotImplementedError


class RayleighStudySolver(Solver):
    """Run profile-wise alpha scans for the inviscid Rayleigh equation."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def _alpha_values(self) -> np.ndarray:
        alpha_cfg = self.config.alpha_scan
        return np.linspace(alpha_cfg.alpha_min, alpha_cfg.alpha_max, alpha_cfg.alpha_count)

    def _scan_profile(self, profile_name: str, N: int) -> dict:
        numerical = self.config.numerical
        alpha_values = self._alpha_values()

        z, _D, D2 = chebyshev_matrices(N=N, L=numerical.L)
        baseflow = evaluate_baseflow(profile_name=profile_name, z_grid=z)

        scan_result = alpha_scan(
            alpha_values=alpha_values,
            U_vals=baseflow["U"],
            Upp_vals=baseflow["Upp"],
            D2=D2,
            magnitude_threshold=numerical.eigenvalue_mag_threshold,
        )

        return {
            "z": [float(v) for v in z],
            "sympy": baseflow["symbolic"],
            "scan": scan_result,
        }

    def _refinement_check(self) -> dict:
        refine_cfg = self.config.refinement
        if not refine_cfg.enabled:
            return {"enabled": False}

        base_payload = self._scan_profile(refine_cfg.profile, refine_cfg.N_base)
        finer_n = refine_cfg.N_base * refine_cfg.multiplier
        fine_payload = self._scan_profile(refine_cfg.profile, finer_n)

        g_base = base_payload["scan"]["summary"]["growth_rate"]
        g_fine = fine_payload["scan"]["summary"]["growth_rate"]

        return {
            "enabled": True,
            "profile": refine_cfg.profile,
            "N_base": refine_cfg.N_base,
            "N_refined": finer_n,
            "max_growth_base": g_base,
            "max_growth_refined": g_fine,
            "note": (
                "For parabolic profile, persistent strong positive growth across refinement "
                "is suspect for inviscid Rayleigh and should be checked against N/L sensitivity."
            ),
        }

    def solve(self) -> dict:
        profile_results: dict[str, dict] = {}
        for profile_name in self.config.solver.profiles:
            profile_results[profile_name] = self._scan_profile(profile_name, self.config.numerical.N)

        return {
            "temporal_convention": "q'(x,z,t)=q_hat(z)*exp(i(alpha*x-omega*t)); growth=Im(omega), frequency=Re(omega)",
            "equation": "(U-c)(phi''-alpha^2*phi)-U''*phi=0, c=omega/alpha",
            "expected_behavior": {
                "tanh_shear": "Inflectional profile may show unstable alpha band.",
                "parabolic": "No inflection point; robust inviscid instability is not expected.",
            },
            "profiles": profile_results,
            "refinement": self._refinement_check(),
        }
