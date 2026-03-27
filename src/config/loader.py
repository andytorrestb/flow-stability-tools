from __future__ import annotations

from pathlib import Path

import yaml

from config.schema import SimulationConfig


def load_config(config_path: Path) -> SimulationConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return SimulationConfig.model_validate(raw)
