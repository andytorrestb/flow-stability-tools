from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


_SCI_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

_SCI_LINESTYLES = ["-", "--", "-.", ":"]
_SCI_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "d"]
_PEAK_HIGHLIGHT_COLOR = "#D62728"
_PEAK_HIGHLIGHT_MARKER = "D"

DISPLAY_PROFILE_NAMES = {
    "tanh_shear": "Hyperbolic-tangent shear layer",
    "parabolic": "Parabolic profile",
    "bickley_jet": "Bickley jet",
    "wake_deficit": "Wake-deficit profile",
    "asymmetric_mixing_layer": "Asymmetric mixing layer",
    "double_shear_layer": "Double shear layer",
}


def _display_profile_name(profile_name: str) -> str:
    return DISPLAY_PROFILE_NAMES.get(profile_name, profile_name.replace("_", " ").title())


def _prepare_peak_points(profile_results: dict[str, dict[str, list[float] | dict[str, float]]]):
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


def _annotate_growth(ax: plt.Axes, growth_summary_lines: list[list[str]]) -> None:
    _draw_summary_blocks(ax, growth_summary_lines)


def _annotate_frequency(ax: plt.Axes, freq_summary_lines: list[list[str]]) -> None:
    _draw_summary_blocks(ax, freq_summary_lines)


def _apply_publication_style() -> None:
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0)


def _format_side_panel(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#f8f8f8")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#cccccc")
    ax.text(
        0.03,
        0.98,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )


def _series_style(index: int) -> dict[str, str]:
    color = _SCI_COLORS[index % len(_SCI_COLORS)]
    linestyle = _SCI_LINESTYLES[index % len(_SCI_LINESTYLES)]
    marker = _SCI_MARKERS[index % len(_SCI_MARKERS)]
    return {"color": color, "linestyle": linestyle, "marker": marker}


def _plot_peak_marker(ax: plt.Axes, alpha_star: float, growth_star: float, zorder: int = 6) -> None:
    ax.scatter(
        [alpha_star],
        [growth_star],
        marker="o",
        s=230,
        color="white",
        edgecolors="none",
        zorder=zorder - 1,
    )
    ax.scatter(
        [alpha_star],
        [growth_star],
        marker=_PEAK_HIGHLIGHT_MARKER,
        s=120,
        color=_PEAK_HIGHLIGHT_COLOR,
        edgecolors="black",
        linewidths=0.75,
        zorder=zorder,
    )


def _format_peak_summary_block(display_name: str, alpha_star: float, growth_star: float, freq_star: float) -> list[str]:
    freq_text = f"{freq_star:.3e}" if np.isfinite(freq_star) else "not finite"
    return [
        display_name,
        "Peak coordinate:",
        rf"$\alpha^* = {alpha_star:.3f},\ \Im(\omega^*) = {growth_star:.3e}$",
        "Most amplified wavenumber:",
        rf"$\alpha^* = {alpha_star:.3f}$",
        "Peak temporal growth rate:",
        rf"$\Im(\omega^*) = {growth_star:.3e}$",
        "Temporal frequency:",
        rf"$\Re(\omega^*) = {freq_text}$" if np.isfinite(freq_star) else "not finite",
    ]


def _draw_summary_blocks(ax: plt.Axes, summary_blocks: list[list[str]]) -> None:
    if not summary_blocks:
        ax.text(
            0.03,
            0.84,
            "No peak summary available.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#222222",
        )
        return

    top = 0.84
    block_height = 0.33 if len(summary_blocks) <= 2 else 0.26

    for idx, block in enumerate(summary_blocks):
        y_top = top - idx * block_height
        if y_top < 0.08:
            break

        profile_name = block[0]
        details = "\n".join([
            f"{block[1]}\n  {block[2]}",
            f"{block[3]}\n  {block[4]}",
            f"{block[5]}\n  {block[6]}",
        ])

        ax.text(
            0.5,
            y_top,
            profile_name,
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            0.05,
            y_top - 0.055,
            details,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            linespacing=1.25,
            color="#1d1d1d",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.98, "edgecolor": "#d0d0d0"},
        )


def _add_readable_legend(
    ax: plt.Axes,
    title: str,
    max_inside_entries: int = 4,
    target_ax: plt.Axes | None = None,
    extra_items: list[tuple[object, str]] | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
        seen.add(label)

    if extra_items:
        for handle, label in extra_items:
            if not label or label in seen:
                continue
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)

    if not unique_handles:
        return

    legend_kwargs = {
        "title": title,
        "frameon": True,
        "framealpha": 0.96,
        "facecolor": "white",
        "edgecolor": "#bbbbbb",
        "labelspacing": 0.35,
        "handlelength": 2.0,
        "borderpad": 0.4,
    }

    legend_ax = target_ax if target_ax is not None else ax
    if target_ax is not None:
        legend = legend_ax.legend(unique_handles, unique_labels, loc="lower left", **legend_kwargs)
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_ha("center")
        try:
            legend._legend_box.align = "center"
        except AttributeError:
            pass
        return

    if len(unique_labels) <= max_inside_entries:
        legend_ax.legend(unique_handles, unique_labels, loc="best", **legend_kwargs)
    else:
        legend_ax.legend(
            unique_handles,
            unique_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            **legend_kwargs,
        )


def create_scan_plots(
    profile_results: dict[str, dict[str, list[float] | dict[str, float] | dict[str, str]]],
    output_dir: Path,
    growth_filename: str,
    frequency_filename: str,
) -> dict[str, str]:
    _apply_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    growth_path = output_dir / growth_filename
    freq_path = output_dir / frequency_filename

    peak_points = _prepare_peak_points(profile_results)
    growth_summary_lines: list[list[str]] = []
    ordered_profiles = list(profile_results.keys())
    profile_styles = {name: _series_style(i) for i, name in enumerate(ordered_profiles)}

    fig_g, (ax_g, ax_g_panel) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        gridspec_kw={"width_ratios": [4.6, 2.0]},
    )
    ax_g.set_facecolor("#ffffff")
    _format_side_panel(ax_g_panel, "Peak-Mode Summary")
    peak_growth_proxy = Line2D(
        [0],
        [0],
        marker=_PEAK_HIGHLIGHT_MARKER,
        linestyle="None",
        markerfacecolor=_PEAK_HIGHLIGHT_COLOR,
        markeredgecolor="black",
        markersize=8,
    )

    for profile_name, payload in profile_results.items():
        style = profile_styles[profile_name]
        display_name = _display_profile_name(profile_name)
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
            label=display_name,
        )

        peak = peak_points.get(profile_name)
        if peak is not None:
            alpha_star, growth_star, freq_star = peak
            _plot_peak_marker(ax_g, alpha_star, growth_star)
            growth_summary_lines.append(
                _format_peak_summary_block(display_name, alpha_star, growth_star, freq_star)
            )

    _format_publication_axes(ax_g)
    ax_g.axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=1,
        label="Neutral stability boundary",
    )
    ax_g.set_xlabel(r"Streamwise wavenumber, $\alpha$")
    ax_g.set_ylabel(r"Temporal growth rate, Im($\omega$)")
    ax_g.set_title("Temporal growth rate of the most unstable mode")
    _add_readable_legend(
        ax_g,
        title="Mean-Flow Profiles",
        target_ax=ax_g_panel,
        extra_items=[(peak_growth_proxy, "Peak growth coordinate")],
    )
    _annotate_growth(ax_g_panel, growth_summary_lines)
    fig_g.tight_layout()
    fig_g.savefig(growth_path, bbox_inches="tight")
    plt.close(fig_g)

    freq_summary_lines: list[list[str]] = []
    fig_f, (ax_f, ax_f_panel) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        gridspec_kw={"width_ratios": [4.6, 2.0]},
    )
    ax_f.set_facecolor("#ffffff")
    _format_side_panel(ax_f_panel, "Peak-Mode Summary")
    alpha_star_reference_labeled = False
    peak_frequency_proxy = Line2D(
        [0],
        [0],
        marker=_PEAK_HIGHLIGHT_MARKER,
        linestyle="None",
        markerfacecolor=_PEAK_HIGHLIGHT_COLOR,
        markeredgecolor="black",
        markersize=8,
    )

    for profile_name, payload in profile_results.items():
        style = profile_styles[profile_name]
        display_name = _display_profile_name(profile_name)
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
            label=display_name,
        )

        peak = peak_points.get(profile_name)
        if peak is not None:
            alpha_star, growth_star, freq_star = peak
            vline_label = (
                "Most amplified wavenumber alpha*"
                if not alpha_star_reference_labeled
                else None
            )
            ax_f.axvline(
                alpha_star,
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                color=line.get_color(),
                label=vline_label,
            )
            alpha_star_reference_labeled = True
            if np.isfinite(freq_star):
                _plot_peak_marker(ax_f, alpha_star, freq_star, zorder=6)
            freq_summary_lines.append(
                _format_peak_summary_block(display_name, alpha_star, growth_star, freq_star)
            )

    _format_publication_axes(ax_f)
    ax_f.set_xlabel(r"Streamwise wavenumber, $\alpha$")
    ax_f.set_ylabel(r"Temporal frequency, Re($\omega$)")
    ax_f.set_title("Temporal frequency of the most unstable mode")
    _add_readable_legend(
        ax_f,
        title="Mean-Flow Profiles",
        target_ax=ax_f_panel,
        extra_items=[(peak_frequency_proxy, "Peak growth coordinate")],
    )
    ax_f.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    _annotate_frequency(ax_f_panel, freq_summary_lines)
    fig_f.tight_layout()
    fig_f.savefig(freq_path, bbox_inches="tight")
    plt.close(fig_f)

    return {
        "growth_plot": str(growth_path),
        "frequency_plot": str(freq_path),
    }
