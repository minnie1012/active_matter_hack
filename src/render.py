"""Slide-quality rendering: phase diagram, frame composition, animations.

All figures import styling from `src.style`. No matplotlib defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter

from src import style


# ============================================================================
# Phase diagram (Phase 3 deliverable)
# ============================================================================

def plot_phase_diagram(
    grid_mean: np.ndarray,
    rho_I_values: np.ndarray,
    alpha_values: np.ndarray,
    thumbnails: Optional[list] = None,
    out_path: Optional[Path] = None,
    title: str = "Tumor–immune phase diagram",
) -> plt.Figure:
    """Render the phase diagram heatmap with log color scale and contours.

    Parameters
    ----------
    grid_mean : (n_rho, n_alpha) float array
        Geometric mean of `final_tumor_fraction` over seeds.
    rho_I_values : (n_rho,) array
        T-cell initial count axis (y).
    alpha_values : (n_alpha,) array
        Suppressant chemotaxis coupling axis (x).
    thumbnails : list of dict, optional
        Each dict: {"rho_I": int, "alpha": float, "pos_T": (Nx2), "pos_I": (Mx2),
                    "L": float, "label": str}. Drawn as small insets connected
        to their phase-diagram coordinate via leader lines.
    out_path : Path, optional
        If provided, save the figure to this PNG.
    """
    style.apply_style()
    fig = plt.figure(figsize=style.FIG_PHASE, dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)

    # main heatmap axes covering left ~70% of figure
    main_ax = fig.add_axes([0.10, 0.13, 0.62, 0.78])
    main_ax.set_facecolor(style.BG)

    # clip values so log scale doesn't blow up
    safe = np.clip(grid_mean, 1e-2, 1e2)

    im = main_ax.pcolormesh(
        alpha_values,
        rho_I_values,
        safe,
        cmap=style.PHASE_CMAP,
        norm=LogNorm(vmin=1e-2, vmax=1e2),
        shading="auto",
    )
    main_ax.set_xlabel(r"Immunosuppression strength  $\alpha$  ($\chi_s$)", fontsize=style.LABEL_SIZE)
    main_ax.set_ylabel(r"Initial T-cell count  $\rho_I$", fontsize=style.LABEL_SIZE)
    main_ax.set_yscale("log")
    main_ax.tick_params(colors=style.FG)

    # contours at the two phase boundaries: 0.1 (clearance vs control) and 10 (control vs escape)
    try:
        cs = main_ax.contour(
            alpha_values, rho_I_values, safe,
            levels=[0.1, 10.0],
            colors=[style.FG, style.ACCENT],
            linewidths=[1.6, 1.6],
            linestyles=["--", "-"],
        )
        main_ax.clabel(cs, inline=True, fontsize=style.SMALL_SIZE, fmt="%.1f")
    except Exception:
        pass

    main_ax.set_title(title, fontsize=style.TITLE_SIZE, color=style.FG, pad=14)

    # colorbar
    cax = fig.add_axes([0.74, 0.13, 0.025, 0.78])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(
        r"Final tumor fraction  $N_T(T_f)/N_T(0)$",
        color=style.FG, fontsize=style.LABEL_SIZE,
    )
    cb.ax.tick_params(colors=style.FG)
    cb.outline.set_edgecolor(style.MUTED)

    # phase labels overlaid on the heatmap
    main_ax.text(
        0.05, 0.92, "CLEARANCE", transform=main_ax.transAxes,
        color=style.FG, fontsize=style.ANNOT_SIZE, weight="bold",
        bbox=dict(facecolor=style.BG, edgecolor=style.FG, alpha=0.7,
                  boxstyle="round,pad=0.4"),
    )
    main_ax.text(
        0.55, 0.60, "CONTROL", transform=main_ax.transAxes,
        color=style.ACCENT, fontsize=style.ANNOT_SIZE, weight="bold",
        bbox=dict(facecolor=style.BG, edgecolor=style.ACCENT, alpha=0.7,
                  boxstyle="round,pad=0.4"),
    )
    main_ax.text(
        0.80, 0.08, "ESCAPE", transform=main_ax.transAxes,
        color=style.TUMOR, fontsize=style.ANNOT_SIZE, weight="bold",
        bbox=dict(facecolor=style.BG, edgecolor=style.TUMOR, alpha=0.7,
                  boxstyle="round,pad=0.4"),
    )

    # ---- thumbnails ----
    if thumbnails:
        thumb_w = 0.20
        thumb_h = 0.22
        thumb_xs = [0.80, 0.80, 0.80]
        thumb_ys = [0.72, 0.42, 0.12]
        for k, t in enumerate(thumbnails[:3]):
            ax_t = fig.add_axes([thumb_xs[k], thumb_ys[k], thumb_w, thumb_h])
            ax_t.set_facecolor(style.BG)
            for spine in ax_t.spines.values():
                spine.set_edgecolor(style.MUTED)
            L = t["L"]
            pos_T = t["pos_T"]
            pos_I = t["pos_I"]
            if pos_I is not None and len(pos_I):
                ax_t.add_collection(EllipseCollection(
                    widths=np.full(len(pos_I), 1.4),
                    heights=np.full(len(pos_I), 1.4),
                    angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
                    transOffset=ax_t.transData,
                    facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,
                    linewidths=0.2, alpha=0.85,
                ))
            if pos_T is not None and len(pos_T):
                ax_t.add_collection(EllipseCollection(
                    widths=np.full(len(pos_T), 1.4),
                    heights=np.full(len(pos_T), 1.4),
                    angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
                    transOffset=ax_t.transData,
                    facecolors=style.TUMOR, edgecolors=style.TUMOR_EDGE,
                    linewidths=0.2, alpha=0.9,
                ))
            ax_t.set_xlim(0, L); ax_t.set_ylim(0, L)
            ax_t.set_aspect("equal")
            ax_t.set_xticks([]); ax_t.set_yticks([])
            ax_t.set_title(t.get("label", ""), color=style.FG,
                            fontsize=style.SMALL_SIZE, pad=2)

            # leader line from thumbnail to phase-diagram point
            try:
                xx_data = t["alpha"]
                yy_data = t["rho_I"]
                disp = main_ax.transData.transform((xx_data, yy_data))
                ax_disp = fig.transFigure.inverted().transform(disp)
                thumb_left = (thumb_xs[k], thumb_ys[k] + thumb_h / 2)
                line = mpl.lines.Line2D(
                    [ax_disp[0], thumb_left[0]],
                    [ax_disp[1], thumb_left[1]],
                    transform=fig.transFigure,
                    color=style.MUTED, linewidth=0.8, alpha=0.7,
                )
                fig.add_artist(line)
                main_ax.plot(
                    xx_data, yy_data, "o", markersize=7,
                    markerfacecolor=style.FG, markeredgecolor=style.BG,
                    markeredgewidth=1.0, zorder=10,
                )
            except Exception:
                pass

    if out_path is not None:
        fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    return fig


# ============================================================================
# Single-frame composition for the video (Phase 4 deliverable)
# ============================================================================

def compose_frame(
    fig,
    main_ax,
    side_ax,
    pos_T: np.ndarray,
    pos_I: np.ndarray,
    c_s: np.ndarray,
    L: float,
    t: float,
    rho_I: int,
    alpha: float,
    n_T_traj: np.ndarray,
    n_I_traj: np.ndarray,
    t_traj: np.ndarray,
    phase_grid: Optional[np.ndarray] = None,
    phase_axes: Optional[tuple] = None,
):
    """Populate a frame: main panel + sidebar + header text + phase-inset dot.

    Pass the existing `fig` (with axes already laid out) so animations can
    re-use the canvas. `main_ax` is cleared each call; sidebar is also
    cleared (we re-draw the live trajectory).
    """
    main_ax.clear()
    main_ax.set_facecolor(style.BG)

    # suppressant heatmap
    main_ax.imshow(
        gaussian_filter(c_s, sigma=0.7),
        extent=[0, L, 0, L], origin="lower",
        cmap=style.SUPPRESSANT_CMAP, alpha=style.FIELD_ALPHA,
        interpolation="bilinear",
    )

    # T cells (with glow halo)
    if len(pos_I):
        main_ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_I), style.TCELL_GLOW_DIAM),
            heights=np.full(len(pos_I), style.TCELL_GLOW_DIAM),
            angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
            transOffset=main_ax.transData,
            facecolors=style.TCELL, edgecolors="none",
            alpha=style.TCELL_GLOW_ALPHA,
        ))
        main_ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            heights=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
            transOffset=main_ax.transData,
            facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,
            linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    if len(pos_T):
        main_ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            heights=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
            transOffset=main_ax.transData,
            facecolors=style.TUMOR, edgecolors=style.TUMOR_EDGE,
            linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    main_ax.set_xlim(0, L); main_ax.set_ylim(0, L)
    main_ax.set_aspect("equal")
    main_ax.set_xticks([]); main_ax.set_yticks([])
    for spine in main_ax.spines.values():
        spine.set_edgecolor(style.MUTED)

    # header text (monospace)
    header = f"t = {t:6.2f}    rho_I = {rho_I:3d}    alpha = {alpha:5.2f}    N_T = {len(pos_T):4d}    N_I = {len(pos_I):4d}"
    main_ax.text(
        0.5, 1.02, header, transform=main_ax.transAxes,
        color=style.FG, fontsize=style.SMALL_SIZE,
        family=style.FONT_MONO, ha="center", va="bottom",
    )

    # sidebar trajectory
    side_ax.clear()
    side_ax.set_facecolor(style.BG)
    if len(t_traj) > 1:
        side_ax.plot(t_traj, n_T_traj, color=style.TUMOR,
                     linewidth=2.0, label=r"$N_T$")
        side_ax.plot(t_traj, n_I_traj, color=style.TCELL,
                     linewidth=2.0, label=r"$N_I$")
    side_ax.set_xlabel("t", color=style.FG, fontsize=style.SMALL_SIZE)
    side_ax.set_ylabel("cell count", color=style.FG, fontsize=style.SMALL_SIZE)
    side_ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)
    for spine in side_ax.spines.values():
        spine.set_edgecolor(style.MUTED)
    side_ax.legend(frameon=False, fontsize=style.SMALL_SIZE - 1, loc="upper left")


def make_video(
    pos_T_snapshots: list,
    pos_I_snapshots: list,
    c_s_snapshots: list,
    times: list,
    n_T_traj: list,
    n_I_traj: list,
    L: float,
    rho_I: int,
    alpha: float,
    out_path: Path,
    fps: int = 30,
    dpi: int = 100,
    title: str = "",
) -> None:
    """Render a video of a single run using FFMpegWriter.

    Sidebar trajectory grows with the animation. Header shows live stats.
    Falls back to GIF if MP4 export fails.
    """
    style.apply_style()
    fig = plt.figure(figsize=(12.0, 8.0), dpi=dpi)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.05, 0.06, 0.62, 0.88])
    side_ax = fig.add_axes([0.74, 0.20, 0.23, 0.55])
    main_ax.set_facecolor(style.BG)
    side_ax.set_facecolor(style.BG)
    if title:
        fig.suptitle(title, color=style.FG, fontsize=style.LABEL_SIZE, y=0.98)

    # find ffmpeg via imageio-ffmpeg if needed
    try:
        import imageio_ffmpeg
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    n_frames = len(pos_T_snapshots)
    t_arr = np.asarray(times)
    nT_arr = np.asarray(n_T_traj)
    nI_arr = np.asarray(n_I_traj)

    def draw(k: int):
        compose_frame(
            fig, main_ax, side_ax,
            pos_T_snapshots[k], pos_I_snapshots[k], c_s_snapshots[k],
            L=L, t=times[k],
            rho_I=rho_I, alpha=alpha,
            n_T_traj=nT_arr[: k + 1], n_I_traj=nI_arr[: k + 1],
            t_traj=t_arr[: k + 1],
        )

    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    anim = FuncAnimation(
        fig, draw, frames=n_frames, interval=1000 / fps, blit=False, repeat=False,
    )
    try:
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(out_path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": style.BG})
        print(f"wrote {out_path}")
    except Exception as e:
        print(f"FFMpeg failed ({e}); falling back to GIF.")
        gif_path = Path(out_path).with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": style.BG})
        print(f"wrote {gif_path}")
    plt.close(fig)
