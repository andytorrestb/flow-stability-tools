from pathlib import Path
from typing import Dict, Any
import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


def export_profile_to_vtk(
    z: np.ndarray,
    fields: Dict[str, np.ndarray],
    output_path: Path,
    profile_name: str = None
) -> None:
    """
    Export 1D field data (z, U(z), U''(z), etc.) to a VTK file for visualization in ParaView.
    fields: dict of {field_name: np.ndarray} (all 1D, same length as z)
    """
    if pv is None:
        raise ImportError("pyvista is required for VTK export. Please install pyvista.")
    z = np.asarray(z)
    n_points = z.size
    grid = pv.StructuredGrid()
    # 1D: treat as (n, 1, 1) grid
    grid.points = np.c_[z, np.zeros_like(z), np.zeros_like(z)]
    grid.dimensions = (n_points, 1, 1)
    for key, arr in fields.items():
        arr = np.asarray(arr)
        if arr.shape != z.shape:
            raise ValueError(f"Field '{key}' shape {arr.shape} does not match z shape {z.shape}")
        grid.point_data[key] = arr.real if np.iscomplexobj(arr) else arr
    grid.save(str(output_path))
