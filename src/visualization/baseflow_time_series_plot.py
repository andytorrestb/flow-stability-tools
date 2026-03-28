import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, writers
from pathlib import Path

def plot_baseflow_time_series_to_mp4(
    z,
    Uzt,
    t_grid,
    output_mp4="baseflow_time_series.mp4",
    figsize=(8, 4),
    interval=80,
    dpi=180,
    style="darkgrid",
    initial_profile=None,
    initial_profile_label="Initial velocity profile",
    initial_profile_style="--",
    initial_profile_color="tab:red",
):
    """
    Create an .mp4 animation of the base flow U(z, t) time evolution.
    Uzt: shape (n_z, n_t)
    Optionally overlays an initial profile for instant comparison.
    """
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], lw=2, label="U(z, t)")
    initial_profile_arr = None
    initial_line = None
    if initial_profile is not None:
        initial_profile_arr = np.asarray(initial_profile)
        if initial_profile_arr.shape != z.shape:
            raise ValueError(
                f"initial_profile shape {initial_profile_arr.shape} does not match z shape {z.shape}"
            )
        initial_line, = ax.plot(
            z,
            initial_profile_arr,
            initial_profile_style,
            lw=1.5,
            color=initial_profile_color,
            label=initial_profile_label,
        )
    ax.set_xlim(z.min(), z.max())
    y_min, y_max = np.min(Uzt), np.max(Uzt)
    if initial_profile_arr is not None:
        y_min = min(y_min, np.min(initial_profile_arr))
        y_max = max(y_max, np.max(initial_profile_arr))
    if np.isclose(y_min, y_max):
        y_center = 0.5 * (y_min + y_max)
        y_range = max(1e-3, abs(y_center) * 0.1)
        y_min, y_max = y_center - y_range, y_center + y_range
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("z")
    ax.set_ylabel("U(z, t)")
    ax.set_title("Base flow time series")
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    if initial_line is not None:
        ax.legend(loc="best")

    def init():
        line.set_data([], [])
        time_text.set_text('')
        artists = [line, time_text]
        if initial_line is not None:
            artists.append(initial_line)
        return tuple(artists)

    def animate(i):
        line.set_data(z, Uzt[:, i])
        t_val = t_grid[i] if t_grid is not None else i
        time_text.set_text(f"t = {t_val:.2f}")
        artists = [line, time_text]
        if initial_line is not None:
            artists.append(initial_line)
        return tuple(artists)

    anim = FuncAnimation(fig, animate, init_func=init, frames=Uzt.shape[1], interval=interval, blit=True)
    Writer = writers['ffmpeg']
    writer = Writer(fps=1000//interval, metadata=dict(artist='flow-stability-tools'), bitrate=1800)
    anim.save(str(Path(output_mp4)), writer=writer, dpi=dpi)
    plt.close(fig)
