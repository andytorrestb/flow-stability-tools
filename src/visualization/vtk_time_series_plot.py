import os
from pathlib import Path
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, writers

def read_vtk_time_series(time_series_dir, field="q_real"):
    """
    Reads a VTK time series (from .pvd or .vtk files) and returns (z, data, t_grid).
    data: shape (n_z, n_t)
    """
    time_series_dir = Path(time_series_dir)
    # Try to find .pvd file
    pvd_files = list(time_series_dir.glob("*.pvd"))
    if pvd_files:
        pvd_path = pvd_files[0]
        collection = pv.read(pvd_path)
        print(f"[DEBUG][read_vtk_time_series] Reading {pvd_path}, n_blocks={len(collection)}")
        # Print children/blocks for diagnostics
        if hasattr(collection, 'children'):
            print(f"[DEBUG][read_vtk_time_series] Collection children: {collection.children}")
        t_grid = []
        data = []
        for idx, block in enumerate(collection):
            arr = block.point_data[field]
            z = block.points[:, 0]
            data.append(arr)
            # Try to get time from field, else use index
            tval = None
            if 'TimeValue' in block.field_data:
                tval = block.field_data['TimeValue'][0]
            else:
                tval = idx
            print(f"[DEBUG][read_vtk_time_series] Block {idx}: tval={tval}, arr min={np.min(arr)}, max={np.max(arr)}")
            t_grid.append(tval)
        if len(data) > 1:
            data = np.stack(data, axis=1)
            t_grid = np.array(t_grid)
            print(f"[DEBUG][read_vtk_time_series] Final data.shape={data.shape}, t_grid={t_grid}")
            return z, data, t_grid
        # Fallback: if only one or zero blocks found, read all .vts files directly
        print("[DEBUG][read_vtk_time_series] Only one or zero blocks found, falling back to reading all .vts files directly.")
        vts_files = sorted(time_series_dir.glob("*.vts"))
        data = []
        t_grid = []
        for i, f in enumerate(vts_files):
            grid = pv.read(f)
            arr = grid.point_data[field]
            z = grid.points[:, 0]
            data.append(arr)
            tval = None
            if 'TimeValue' in grid.field_data:
                tval = grid.field_data['TimeValue'][0]
            else:
                tval = i
            t_grid.append(tval)
        data = np.stack(data, axis=1)
        t_grid = np.array(t_grid)
        print(f"[DEBUG][read_vtk_time_series] Fallback: data.shape={data.shape}, t_grid={t_grid}")
        return z, data, t_grid
    # Else, look for .vtk files
    vtk_files = sorted(time_series_dir.glob("*.vtk"))
    if not vtk_files:
        raise FileNotFoundError("No .pvd or .vtk files found in directory.")
    data = []
    t_grid = []
    for i, f in enumerate(vtk_files):
        grid = pv.read(f)
        arr = grid.point_data[field]
        z = grid.points[:, 0]
        data.append(arr)
        t_grid.append(i)  # No time info, use index
    data = np.stack(data, axis=1)
    t_grid = np.array(t_grid)
    return z, data, t_grid

def plot_vtk_time_series_to_mp4(time_series_dir, field="q_real", output_mp4="time_series.mp4",
                                figsize=(8, 4), interval=80, dpi=180, style="darkgrid"):
    """
    Reads a VTK time series and creates an .mp4 animation of the field evolution.
    """
    sns.set_style(style)
    z, data, t_grid = read_vtk_time_series(time_series_dir, field=field)
    print(f"[DEBUG] Animation: z.shape={z.shape}, data.shape={data.shape}, t_grid.shape={t_grid.shape}")
    print(f"[DEBUG] Animation: data min={np.min(data)}, max={np.max(data)}, mean={np.mean(data)}")
    print(f"[DEBUG] Animation: data sample (first frame)={data[:,0]}")
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(z.min(), z.max())
    y_min, y_max = np.min(data), np.max(data)
    if np.isclose(y_min, y_max):
        # If data is nearly constant, expand limits for visibility
        y_center = 0.5 * (y_min + y_max)
        y_range = max(1e-3, abs(y_center) * 0.1)
        y_min, y_max = y_center - y_range, y_center + y_range
        print(f"[DEBUG] Animation: y-limits adjusted to ({y_min}, {y_max}) for visibility")
    else:
        print(f"[DEBUG] Animation: y-limits set to ({y_min}, {y_max})")
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("z")
    ax.set_ylabel(field)
    ax.set_title(f"Time series: {field}")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_data(z, data[:, i])
        t_val = t_grid[i] if t_grid is not None else i
        time_text.set_text(f"t = {t_val:.2f}")
        return line, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=data.shape[1], interval=interval, blit=True)
    Writer = writers['ffmpeg']
    writer = Writer(fps=1000//interval, metadata=dict(artist='flow-stability-tools'), bitrate=1800)
    anim.save(str(Path(time_series_dir) / output_mp4), writer=writer, dpi=dpi)
    plt.close(fig)
