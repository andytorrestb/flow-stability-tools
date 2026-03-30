from __future__ import annotations

from typing import Callable, Dict

from components.solver import RayleighStudySolver, Solver
from config.schema import SimulationConfig


SolverFactory = Callable[[SimulationConfig], Solver]


class SolverRegistry:
    """Simple registry to allow plug-in solver implementations."""

    def __init__(self) -> None:
        self._registry: Dict[str, SolverFactory] = {}

    def register(self, solver_type: str, factory: SolverFactory) -> None:
        self._registry[solver_type] = factory

    def get(self, solver_type: str) -> SolverFactory:
        if solver_type not in self._registry:
            raise KeyError(f"No solver registered for type '{solver_type}'")
        return self._registry[solver_type]


_DEFAULT_SOLVER_REGISTRY = SolverRegistry()
_DEFAULT_SOLVER_REGISTRY.register("rayleigh_inviscid", RayleighStudySolver)


def get_solver(config: SimulationConfig, registry: SolverRegistry | None = None) -> Solver:
    reg = registry or _DEFAULT_SOLVER_REGISTRY
    factory = reg.get(config.solver.type)
    return factory(config)


def get_default_solver_registry() -> SolverRegistry:
    return _DEFAULT_SOLVER_REGISTRY