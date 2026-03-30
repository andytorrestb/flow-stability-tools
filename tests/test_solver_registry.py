import unittest

from components.solver_registry import SolverRegistry, get_solver
from components.solver import Solver
from config.schema import SimulationConfig


class DummySolver(Solver):
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def solve(self):  # pragma: no cover - minimal stub
        return {"ok": True}


class TestSolverRegistry(unittest.TestCase):
    def test_default_registry_includes_rayleigh(self):
        cfg = SimulationConfig.model_validate(
            {
                "case_name": "test",
                "solver": {"type": "rayleigh_inviscid", "profiles": ["tanh_shear"]},
                "numerical": {"L": 1.0, "N": 8},
                "alpha_scan": {"alpha_min": 0.1, "alpha_max": 0.2, "alpha_count": 3},
                "pipeline": {"steps": ["setup", "solve", "analysis"]},
            }
        )
        solver = get_solver(cfg)
        from components.solver import RayleighStudySolver

        self.assertIsInstance(solver, RayleighStudySolver)

    def test_register_custom_solver(self):
        registry = SolverRegistry()
        registry.register("dummy", DummySolver)

        cfg = SimulationConfig.model_validate(
            {
                "case_name": "test",
                "solver": {"type": "dummy", "profiles": ["tanh_shear"]},
                "numerical": {"L": 1.0, "N": 8},
                "alpha_scan": {"alpha_min": 0.1, "alpha_max": 0.2, "alpha_count": 3},
                "pipeline": {"steps": ["setup", "solve", "analysis"]},
            }
        )

        solver = get_solver(cfg, registry=registry)
        self.assertIsInstance(solver, DummySolver)


if __name__ == "__main__":
    unittest.main()