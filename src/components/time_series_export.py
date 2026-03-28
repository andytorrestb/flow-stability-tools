from pathlib import Path
from typing import Dict, Any
import numpy as np
import pyvista as pv

def reconstruct_time_series(z: np.ndarray, eigenfunction: np.ndarray, omega: complex, t_grid: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """
    Reconstruct the time series q(z, t) = amplitude * eigenfunction(z) * exp(-i * omega * t)
    Returns array of shape (len(z), len(t_grid))
    """
    q0 = amplitude * eigenfunction
    # Outer product: (z, t)
    qzt = np.outer(q0, np.exp(-1j * omega * t_grid))
    return qzt

def export_time_series_vtk(z: np.ndarray, qzt: np.ndarray, t_grid: np.ndarray, output_dir: Path, base_filename: str = "time_series"):
    """
    Export a time series as a sequence of VTK files and a .pvd collection for ParaView animation.
    Each file is named base_filename_t{idx:04d}.vtk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_filenames = []
    print(f"[DEBUG][export_time_series_vtk] z.shape={z.shape}, qzt.shape={qzt.shape}, t_grid.shape={t_grid.shape}")
    for i, t in enumerate(t_grid):
        # Use XML StructuredGrid (.vts) for time series compatibility
        grid = pv.StructuredGrid()
        grid.points = np.c_[z, np.zeros_like(z), np.zeros_like(z)]
        grid.dimensions = (len(z), 1, 1)
        grid.point_data["q_real"] = qzt[:, i].real
        grid.point_data["q_imag"] = qzt[:, i].imag
        t_val = float(t) if t is not None else float(i)
        # Ensure TimeValue is a 1-element array, not a scalar
        grid.field_data["TimeValue"] = np.array([t_val], dtype=np.float64)
        fname = f"{base_filename}_t{i:04d}.vts"
        fpath = output_dir / fname
        print(f"[DEBUG][export_time_series_vtk] Writing {fname}: t_val={t_val}, q_real min={np.min(qzt[:,i].real)}, max={np.max(qzt[:,i].real)}")
        grid.save(str(fpath), binary=True)  # XML VTK format
        vtk_filenames.append((fname, t_val))
    print(f"[DEBUG][export_time_series_vtk] Wrote {len(vtk_filenames)} VTK files and .pvd to {output_dir}")
    # Write a .pvd file for ParaView time animation (reference .vts files)
    pvd_path = output_dir / f"{base_filename}.pvd"
    with open(pvd_path, "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write("  <Collection>\n")
        for i, (fname, t) in enumerate(vtk_filenames):
            f.write(f"    <DataSet timestep=\"{t:.6f}\" group=\"\" part=\"0\" file=\"{fname}\"/>\n")
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
