from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Colorblind-safe palette commonly used in scientific plots (Okabe-Ito inspired).
_SCI_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

_SCI_LINESTYLES = ["-", "--", "-.", ":"]
_SCI_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _prepare_peak_points(profile_results: dict[str, dict[str, list[float] | dict[str, float]]]):
    """Extract peak growth/frequency per profile for annotations."""
    peak_points: dict[str, tuple[float, float, float]] = {}
    for profile_name, payload in profile_results.items():
        alpha = np.asarray(payload["scan"]["alpha"], dtype=float)
        growth = np.asarray(payload["scan"]["dominant_growth"], dtype=float)
        freq = np.asarray(payload["scan"]["dominant_frequency"], dtype=float)
        finite_mask = np.isfinite(alpha) & np.isfinite(growth)
        if not np.any(finite_mask):
            continue
        valid_idx = np.where(finite_mask)[0]
        local_idx = int(np.argmax(growth[finite_mask]))
        idx = int(valid_idx[local_idx])
        alpha_star = float(alpha[idx])
        growth_star = float(growth[idx])
        freq_star = float(freq[idx]) if np.isfinite(freq[idx]) else float("nan")
        peak_points[profile_name] = (alpha_star, growth_star, freq_star)
    return peak_points


def _annotate_growth(ax: plt.Axes, growth_summary_lines: list[str]) -> None:
    if not growth_summary_lines:
        return
    ax.text(
        0.02,
        0.98,
        "\n".join(growth_summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.95, "edgecolor": "#bbbbbb"},
    )


def _annotate_frequency(ax: plt.Axes, freq_summary_lines: list[str]) -> None:
    if not freq_summary_lines:
        return
    ax.text(
        0.98,
        0.98,
        "\n".join(freq_summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.95, "edgecolor": "#bbbbbb"},
    )


def _apply_publication_style() -> None:
    """Set a compact publication-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def _format_publication_axes(ax: plt.Axes) -> None:
    """Apply consistent publication-style axis formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0)


def _series_style(index: int) -> dict[str, str]:
    """Return deterministic style for the i-th series."""
    color = _SCI_COLORS[index % len(_SCI_COLORS)]
    linestyle = _SCI_LINESTYLES[index % len(_SCI_LINESTYLES)]
    marker = _SCI_MARKERS[index % len(_SCI_MARKERS)]
    return {"color": color, "linestyle": linestyle, "marker": marker}


def create_scan_plots(
    profile_results: dict[str, dict[str, list[float] | dict[str, float] | dict[str, str]]],
    output_dir: Path,
    growth_filename: str,
    frequency_filename: str,
) -> dict[str, str]:
    """Create and save growth/frequency versus alpha plots across all profiles."""
    _apply_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    growth_path = output_dir / growth_filename
    freq_path = output_dir / frequency_filename

    peak_points = _prepare_peak_points(profile_results)
    growth_summary_lines: list[str] = []
    ordered_profiles = list(profile_results.keys())
    profile_styles = {name: _series_style(i) for i, name in enumerate(ordered_profiles)}

    fig_g, ax_g = plt.subplots(figsize=(9.2, 5.8))
    ax_g.set_facecolor("#ffffff")

    for profile_name, payload in profile_results.items():
        style = profile_styles[profile_name]
        alpha = np.asarray(payload["scan"]["alpha"], dtype=float)
        growth = np.asarray(payload["scan"]["dominant_growth"], dtype=float)
        freq = np.asarray(payload["scan"]["dominant_frequency"], dtype=float)

        line, = ax_g.plot(
            alpha,
            growth,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            ms=3.6,
            lw=2.0,
            label=profile_name,
        )

        peak = peak_points.get(profile_name)
        if peak is not None:
            alpha_star, growth_star, freq_star = peak
            ax_g.scatter(
                [alpha_star],
                [growth_star],
                marker="*",
                s=130,
                color=line.get_color(),
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            growth_summary_lines.append(
                f"{profile_name}: (alpha*, omega_i*) = ({alpha_star:.3f}, {growth_star:.3e})"
            )

    _format_publication_axes(ax_g)
    ax_g.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax_g.set_xlabel("alpha")
    ax_g.set_ylabel("max omega_i(alpha)")
    ax_g.set_title("Dominant temporal growth vs alpha")
    ax_g.legend(loc="best")
    _annotate_growth(ax_g, growth_summary_lines)
    fig_g.tight_layout()
    fig_g.savefig(growth_path, bbox_inches="tight")
    plt.close(fig_g)

    freq_summary_lines: list[str] = []
    fig_f, ax_f = plt.subplots(figsize=(9.2, 5.8))
    ax_f.set_facecolor("#ffffff")

    for profile_name, payload in profile_results.items():
        style = profile_styles[profile_name]
        alpha = np.asarray(payload["scan"]["alpha"], dtype=float)
        freq = np.asarray(payload["scan"]["dominant_frequency"], dtype=float)
        line, = ax_f.plot(
            alpha,
            freq,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            ms=3.6,
            lw=2.0,
            label=profile_name,
        )

        peak = peak_points.get(profile_name)
        if peak is not None:
            alpha_star, _growth_star, freq_star = peak
            ax_f.axvline(alpha_star, linestyle=":", linewidth=1.2, alpha=0.8, color=line.get_color())
            if np.isfinite(freq_star):
                ax_f.scatter(
                    [alpha_star],
                    [freq_star],
                    marker="D",
                    s=55,
                    color=line.get_color(),
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )
                freq_summary_lines.append(
                    f"{profile_name}: alpha* = {alpha_star:.3f}, omega_r(alpha*) = {freq_star:.3e}"
                )

    _format_publication_axes(ax_f)
    ax_f.set_xlabel("alpha")
    ax_f.set_ylabel("omega_r of dominant mode")
    ax_f.set_title("Dominant-mode frequency vs alpha")
    ax_f.legend(loc="best")
    ax_f.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    _annotate_frequency(ax_f, freq_summary_lines)
    fig_f.tight_layout()
    fig_f.savefig(freq_path, bbox_inches="tight")
    plt.close(fig_f)

    return {
        "growth_plot": str(growth_path),
        "frequency_plot": str(freq_path),
    }
