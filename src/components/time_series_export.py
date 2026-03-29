from pathlib import Path
from typing import Dict, Any
import numpy as np
import pyvista as pv

from components.spectral import chebyshev_matrices


def reconstruct_time_series(
    z: np.ndarray,
    eigenfunction: np.ndarray,
    omega: complex,
    t_grid: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Reconstruct q(z, t) = amplitude * phi(z) * exp(-i * omega * t)."""
    if eigenfunction.shape[0] != z.shape[0]:
        raise ValueError(
            f"eigenfunction length {eigenfunction.shape[0]} does not match z length {z.shape[0]}"
        )
    q0 = amplitude * eigenfunction
    qzt = np.outer(q0, np.exp(-1j * omega * t_grid))
    return qzt


def reconstruct_velocity_time_series(
    z: np.ndarray,
    eigenfunction: np.ndarray,
    omega: complex,
    t_grid: np.ndarray,
    amplitude: float = 1.0,
    D: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reconstruct u'(z, t) = amplitude * dphi/dz * exp(-i * omega * t).
    Uses Chebyshev derivative matrix if provided; otherwise infers it from z.
    """
    n = z.shape[0]
    if eigenfunction.shape[0] != n:
        raise ValueError(
            f"eigenfunction length {eigenfunction.shape[0]} does not match z length {n}"
        )

    if D is None:
        # Infer Chebyshev grid parameters and rebuild D for consistent differentiation
        L = float(np.max(np.abs(z)))
        N = n - 1
        z_ref, D_ref, _D2 = chebyshev_matrices(N=N, L=L)
        if z_ref.shape != z.shape or not np.allclose(z_ref, z, atol=1e-12, rtol=1e-10):
            raise ValueError("Provided z does not match inferred Chebyshev grid for differentiation")
        D = D_ref

    phi_z = D @ eigenfunction
    # Enforce boundary consistency with phi(-L)=phi(L)=0 assumption
    phi_z[0] = 0.0
    phi_z[-1] = 0.0

    u0 = amplitude * phi_z
    uzt = np.outer(u0, np.exp(-1j * omega * t_grid))
    return uzt


def export_time_series_vtk(
    z: np.ndarray,
    field_time_series: Dict[str, np.ndarray],
    t_grid: np.ndarray,
    output_dir: Path,
    base_filename: str = "time_series",
):
    """
    Export one or more time series fields as VTK (.vts) plus a .pvd collection.
    field_time_series: mapping name -> array (n_z, n_t). Complex arrays are split into _real/_imag.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_filenames = []

    # Basic shape validation
    n_z = z.shape[0]
    n_t = t_grid.shape[0]
    for name, arr in field_time_series.items():
        if arr.shape != (n_z, n_t):
            raise ValueError(f"Field '{name}' has shape {arr.shape}, expected {(n_z, n_t)}")

    print(
        f"[DEBUG][export_time_series_vtk] z.shape={z.shape}, fields={list(field_time_series.keys())}, t_grid.shape={t_grid.shape}"
    )

    for i, t in enumerate(t_grid):
        grid = pv.StructuredGrid()
        grid.points = np.c_[z, np.zeros_like(z), np.zeros_like(z)]
        grid.dimensions = (n_z, 1, 1)

        for name, arr in field_time_series.items():
            slice_i = arr[:, i]
            if np.iscomplexobj(slice_i):
                grid.point_data[f"{name}_real"] = slice_i.real
                grid.point_data[f"{name}_imag"] = slice_i.imag
            else:
                grid.point_data[name] = slice_i

        t_val = float(t) if t is not None else float(i)
        grid.field_data["TimeValue"] = np.array([t_val], dtype=np.float64)

        fname = f"{base_filename}_t{i:04d}.vts"
        fpath = output_dir / fname
        grid.save(str(fpath), binary=True)
        vtk_filenames.append((fname, t_val))

    print(f"[DEBUG][export_time_series_vtk] Wrote {len(vtk_filenames)} VTK files and .pvd to {output_dir}")

    pvd_path = output_dir / f"{base_filename}.pvd"
    with open(pvd_path, "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        f.write("  <Collection>\n")
        for fname, t in vtk_filenames:
            f.write(f"    <DataSet timestep=\"{t:.6f}\" group=\"\" part=\"0\" file=\"{fname}\"/>\n")
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
