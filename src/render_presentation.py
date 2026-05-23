"""Presentation-set renderer for the biophysical TME simulation.

This module is a thin renderer dedicated to the 5-video presentation series
produced by ``scripts/phase9_presentation_set.py``. It reuses the master
frame composer ``compose_tme_frame`` from ``render_tme`` so the main panel
(O2 background, ECM fibers, vessels, cells, legend strip) matches the look
of ``outputs/videos/tme_full.mp4``. The sidebar is replaced with three
stacked time-series panels: mean EMT, max invasion distance, and macrophage
M1/M2 polarization. A static bottom-trunk parent vessel strip with
pericytes and RBCs is added as an overlay on top of the main panel so the
biophysical-style vasculature is visible even when angiogenesis is off.

DO NOT edit ``render_tme.py`` — this module composes on top of the public
helpers exported there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Ellipse, Rectangle

from src.render_tme import (
    BG, PANEL, INK, SUB, RULE, ACCENT,
    TUMOR, CD8, MAC_M1, MAC_M2, VESSEL, VESSEL_EDGE, CAF, CAF_EDGE,
    FONT_MONO,
    _apply_light_style,
    _make_fiber_points,
    _synthesize_CAFs,
    compose_tme_frame,
)


# ===========================================================================
# Helpers — adapt BiophysicalOut into the per-frame dict compose_tme_frame
# expects, and synthesise the cosmetic missing fields (CAFs, NK/DC/MDSC).
# ===========================================================================

def _build_step_dict_bio(out, k: int, L: float,
                         fallback_caf: np.ndarray) -> dict:
    """Build the per-frame ``step`` dict for ``compose_tme_frame`` from a
    BiophysicalOut.

    The biophysical simulator doesn't track NK / DC / MDSC populations,
    so those are absent (drawn as empty). CAF positions fall back to a
    synthesised ring around the early tumor cloud (same convention as the
    TME renderer).
    """
    pos_T = out.pos_T_snapshots[k]
    pos_I = out.pos_I_snapshots[k]
    pos_M = (out.pos_M_snapshots[k]
             if k < len(out.pos_M_snapshots) else None)
    p_M = (out.p_M_snapshots[k]
           if k < len(out.p_M_snapshots) else None)
    # O2 field — required as background
    if out.O_snapshots and k < len(out.O_snapshots):
        c_O2 = out.O_snapshots[k]
    else:
        # synthesise a near-flat "fully oxygenated" field if save_fields=False
        c_O2 = np.ones((out.params.G, out.params.G), dtype=np.float64)
    vessels = (np.asarray(out.vessel_snapshots[k])
               if k < len(out.vessel_snapshots) else np.zeros((0, 2)))
    vparents_list = getattr(out, "vessel_parent_snapshots", None)
    vparents = (np.asarray(vparents_list[k])
                if vparents_list is not None and k < len(vparents_list)
                else None)

    step = dict(
        pos_T=pos_T,
        pos_I=pos_I,
        pos_M=pos_M,
        p_M=p_M,
        pos_NK=None,
        pos_DC=None,
        pos_MDSC=None,
        vessels=vessels,
        c_O2=c_O2,
        pos_CAF=fallback_caf,
    )
    if vparents is not None:
        step["vessel_parents"] = vparents
    return step


def _hypoxic_fraction_from_H(H: Optional[np.ndarray]) -> float:
    if H is None or H.size == 0:
        return 0.0
    return float(np.mean(H > 0.3))


def _hypoxic_fraction_from_O(c_O2: np.ndarray, O_threshold: float) -> float:
    if c_O2 is None or c_O2.size == 0:
        return 0.0
    if O_threshold <= 0.0:
        return 0.0
    return float(np.mean(c_O2 < O_threshold))


# ===========================================================================
# Bottom-trunk parent vessel + pericytes + RBCs overlay.
# Drawn on top of compose_tme_frame's main panel so the biophysical-style
# trunk is visible in every scenario (even when sprouting is off).
# ===========================================================================

def _draw_parent_trunk_overlay(main_ax, L: float):
    strip_w = L * 0.92
    strip_h = max(0.6, 0.02 * L)
    strip_x0 = 0.5 * L - 0.5 * strip_w
    strip_y0 = 0.5  # near the bottom of the box
    # red parent strip
    main_ax.add_patch(Rectangle(
        (strip_x0, strip_y0), strip_w, strip_h,
        facecolor=VESSEL, edgecolor=VESSEL_EDGE, linewidth=0.8,
        alpha=0.85, zorder=1.55,
    ))
    # ~9 tan pericytes hugging the strip
    n_peri = 9
    peri_xs = np.linspace(strip_x0 + 0.05 * strip_w,
                          strip_x0 + 0.95 * strip_w, n_peri)
    for px in peri_xs:
        py = strip_y0 + 0.5 * strip_h - 0.6
        main_ax.add_patch(Ellipse(
            (px, py), width=1.8, height=1.0,
            facecolor=CAF, edgecolor=CAF_EDGE, linewidth=0.5,
            alpha=0.85, zorder=1.5,
        ))
    # ~18 deterministic RBC ellipses inside the strip
    rbc_rng = np.random.default_rng(0)
    n_rbc = 18
    rbc_xs = rbc_rng.uniform(strip_x0 + 0.02 * strip_w,
                             strip_x0 + 0.98 * strip_w, size=n_rbc)
    rbc_ys = rbc_rng.uniform(strip_y0 + 0.18 * strip_h,
                             strip_y0 + 0.82 * strip_h, size=n_rbc)
    for rx, ry in zip(rbc_xs, rbc_ys):
        main_ax.add_patch(Ellipse(
            (rx, ry), width=0.9, height=0.55,
            facecolor="#C84A4A", edgecolor="#8A1A22", linewidth=0.3,
            alpha=0.9, zorder=1.57,
        ))


# ===========================================================================
# Right-side panel — 3 stacked time-series subplots
# ===========================================================================

def _style_side_axis(ax, title: str, ylabel: str, t_arr,
                     y_arr, color, ylim: Optional[tuple] = None,
                     zero_line: bool = False, hide_xticks: bool = False):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.plot(t_arr, y_arr, color=color, linewidth=1.7)
    if zero_line:
        ax.axhline(0.0, color=SUB, linewidth=0.7, alpha=0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(RULE)
    ax.set_title(title, color=INK, fontsize=9, weight="bold", pad=3)
    if hide_xticks:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)


def make_presentation_figure(figsize=(14.0, 8.0), dpi=110):
    """Allocate the figure for the 5-video presentation set.

    Layout (matches the spec):
      - title strip:  [0.04, 0.92, 0.92, 0.06]
      - legend strip: [0.04, 0.84, 0.92, 0.06]
      - main panel:   [0.04, 0.08, 0.66, 0.74]    (same as tme_full)
      - side top:     [0.74, 0.62, 0.23, 0.20]    (mean EMT)
      - side middle:  [0.74, 0.36, 0.23, 0.20]    (max invasion distance)
      - side bottom:  [0.74, 0.10, 0.23, 0.20]    (macrophage M1/M2)
    """
    _apply_light_style()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)

    title_ax = fig.add_axes([0.04, 0.92, 0.92, 0.06])
    title_ax.set_facecolor(BG); title_ax.axis("off")

    leg_ax = fig.add_axes([0.04, 0.84, 0.92, 0.06])
    leg_ax.set_facecolor(BG); leg_ax.axis("off")

    main_ax = fig.add_axes([0.04, 0.08, 0.66, 0.74])
    main_ax.set_facecolor(BG)

    # 4 stacked time-series subplots on the right
    side_count = fig.add_axes([0.74, 0.65, 0.23, 0.16])
    side_count.set_facecolor(PANEL)

    side_emt = fig.add_axes([0.74, 0.46, 0.23, 0.16])
    side_emt.set_facecolor(PANEL)

    side_inv = fig.add_axes([0.74, 0.27, 0.23, 0.16])
    side_inv.set_facecolor(PANEL)

    side_pM = fig.add_axes([0.74, 0.08, 0.23, 0.16])
    side_pM.set_facecolor(PANEL)

    return fig, {
        "title": title_ax,
        "leg": leg_ax,
        "main": main_ax,
        "side_count": side_count,
        "side_emt": side_emt,
        "side_inv": side_inv,
        "side_pM": side_pM,
    }


def _draw_presentation_frame(fig, axes, out, k: int,
                             fiber_pts: np.ndarray,
                             fallback_caf: np.ndarray,
                             scenario_title: str):
    """Populate the figure for frame ``k``."""
    L = out.params.L
    t_arr = np.asarray(out.times)
    t_now = float(t_arr[k]) if len(t_arr) else 0.0

    # ---- build the per-frame dict for compose_tme_frame ----
    step = _build_step_dict_bio(out, k, L, fallback_caf)

    # hypoxic fraction (used by compose_tme_frame title bar)
    H = (out.H_snapshots[k]
         if out.H_snapshots and k < len(out.H_snapshots) else None)
    if H is not None:
        hyp_frac = _hypoxic_fraction_from_H(H)
    else:
        hyp_frac = _hypoxic_fraction_from_O(
            step["c_O2"], getattr(out.params, "O_threshold", -1.0))

    # cell-count subtitle data (NK/DC/MDSC = 0 — BiophysicalOut has none)
    n_T_now = out.n_T[k] if k < len(out.n_T) else 0
    n_I_now = out.n_I[k] if k < len(out.n_I) else 0
    n_M_now = out.n_M[k] if k < len(out.n_M) else 0
    n_V_now = (out.n_vessels[k]
               if out.n_vessels and k < len(out.n_vessels) else 0)
    trajs = dict(
        t=t_arr[:k + 1],
        # population trajectory series unused (side_pop is None) but pass
        # the scalars compose_tme_frame's title bar uses
        n_T_now=int(n_T_now), n_I_now=int(n_I_now), n_M_now=int(n_M_now),
        n_NK_now=0, n_DC_now=0, n_MDSC_now=0,
        n_V_now=int(n_V_now),
    )

    # ---- compose main panel + legend + default title via the shared composer
    compose_tme_frame(
        fig,
        {
            "main": axes["main"],
            "leg":  axes["leg"],
            "title": axes["title"],
            # IMPORTANT: side_pop=None disables compose_tme_frame's own
            # right-side population trajectory; we draw our 3 series below.
            "side_pop": None,
        },
        step,
        L=L,
        t=t_now,
        hyp_frac=hyp_frac,
        trajs=trajs,
        fiber_pts=fiber_pts,
    )

    # ---- EMT-aware tumor recolor: redraw tumor cells on top, colored by s_T
    # (compose_tme_frame paints them flat pink — we overlay a per-cell color
    # that interpolates EMT_EPI -> EMT_MES based on the EMT state) ----
    pos_T_now = step.get("pos_T", None)
    s_T_arr = (out.s_T_snapshots[k]
               if getattr(out, "s_T_snapshots", None) and k < len(out.s_T_snapshots)
               else None)
    if pos_T_now is not None and len(pos_T_now) > 0 and s_T_arr is not None:
        from matplotlib.collections import EllipseCollection
        from src.render_tme import _emt_colors, TUMOR_EDGE
        n = min(len(pos_T_now), len(s_T_arr))
        face_rgb = _emt_colors(s_T_arr[:n])
        axes["main"].add_collection(EllipseCollection(
            widths=np.full(n, 1.4), heights=np.full(n, 1.4),
            angles=np.zeros(n), units="x", offsets=pos_T_now[:n],
            transOffset=axes["main"].transData,
            facecolors=face_rgb, edgecolors=TUMOR_EDGE,
            linewidths=0.4, alpha=0.95, zorder=10,
        ))

    # ---- bottom-trunk parent vessel + pericytes + RBCs overlay ----
    _draw_parent_trunk_overlay(axes["main"], L)

    # ---- redraw title to include the scenario label + EMT / invasion / p_M
    title_ax = axes["title"]
    title_ax.clear()
    title_ax.set_xlim(0, 1); title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    title_ax.set_facecolor(BG)
    title_ax.text(
        0.5, 0.62,
        f"{scenario_title}   |   t = {t_now:6.2f}   |   hypoxic frac = {hyp_frac*100:.1f}%",
        ha="center", va="center", color=INK, fontsize=13, weight="bold",
    )
    # subtitle: counts + EMT / invasion / p_M scalars at this frame
    mean_emt_now = (out.mean_EMT[k] if k < len(out.mean_EMT) else 0.0)
    inv_now = (out.max_invasion_distance[k]
               if k < len(out.max_invasion_distance) else 0.0)
    pM_arr = getattr(out, "mean_pM", None)
    pM_now = (pM_arr[k] if pM_arr and k < len(pM_arr) else 0.0)
    sub = (f"Tumor: {n_T_now}   CD8: {n_I_now}   Mφ: {n_M_now}   "
           f"vessels: {n_V_now}   |   EMT: {mean_emt_now:.2f}   "
           f"invasion: {inv_now:5.1f}   pM: {pM_now:+.2f}")
    title_ax.text(
        0.5, 0.18, sub, ha="center", va="center",
        color=SUB, fontsize=10, family=FONT_MONO,
    )

    # ---- 4 stacked time series on the right ----
    t_so_far = t_arr[:k + 1]
    nT_arr = np.asarray(out.n_T[:k + 1])
    emt_arr = np.asarray(out.mean_EMT[:k + 1])
    inv_arr = np.asarray(out.max_invasion_distance[:k + 1])
    if pM_arr is None:
        pm_series = np.zeros_like(t_so_far)
    else:
        pm_series = np.asarray(pM_arr[:k + 1])

    _style_side_axis(
        axes["side_count"],
        title="tumor cells (N_T)",
        ylabel="count",
        t_arr=t_so_far, y_arr=nT_arr,
        color=TUMOR,
        ylim=None,
        hide_xticks=True,
    )
    _style_side_axis(
        axes["side_emt"],
        title="EMT level",
        ylabel="mean s_T",
        t_arr=t_so_far, y_arr=emt_arr,
        color="#5B4F8A",
        ylim=(0.0, 1.0),
        hide_xticks=True,
    )
    _style_side_axis(
        axes["side_inv"],
        title="max invasion distance",
        ylabel="distance",
        t_arr=t_so_far, y_arr=inv_arr,
        color=ACCENT,
        ylim=None,
        hide_xticks=True,
    )
    _style_side_axis(
        axes["side_pM"],
        title="macrophage M1/M2 (pM)",
        ylabel="pM   (-1 M2  /  +1 M1)",
        t_arr=t_so_far, y_arr=pm_series,
        color=MAC_M1,
        ylim=(-1.05, 1.05),
        zero_line=True,
    )
    axes["side_pM"].set_xlabel("t", color=INK, fontsize=10)


# ===========================================================================
# Public entry points
# ===========================================================================

def render_presentation_video(out, out_path: Path, title: str,
                              fps: int = 24, dpi: int = 100,
                              bitrate: int = 4500) -> Path:
    """Render the BiophysicalOut to an MP4 using the presentation layout.

    Falls back to GIF if FFmpeg is unavailable.
    """
    L = out.params.L
    fig, axes = make_presentation_figure(figsize=(14.0, 8.0), dpi=dpi)
    fiber_pts = _make_fiber_points(L, n_fiber=1400, seed=12345)
    seed_pos_T = (out.pos_T_snapshots[0]
                  if len(out.pos_T_snapshots) else None)
    fallback_caf = _synthesize_CAFs(L, seed_pos_T, n_caf=25, seed=0)

    # locate ffmpeg
    try:
        import imageio_ffmpeg
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    n_frames = len(out.pos_T_snapshots)

    def draw(k):
        _draw_presentation_frame(fig, axes, out, k, fiber_pts,
                                 fallback_caf, title)

    anim = FuncAnimation(fig, draw, frames=n_frames,
                         interval=1000 / fps, blit=False, repeat=False)

    out_path = Path(out_path)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        anim.save(out_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"wrote {out_path}")
    except Exception as e:
        print(f"FFMpegWriter failed ({e}); falling back to GIF.")
        gif_path = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"wrote {gif_path}")
        out_path = gif_path
    plt.close(fig)
    return out_path


def render_presentation_still(out, frame_idx: int, out_path: Path,
                              title: str, dpi: int = 180) -> Path:
    """Render a single presentation-style still PNG."""
    L = out.params.L
    fig, axes = make_presentation_figure(figsize=(14.0, 8.0), dpi=dpi)
    fiber_pts = _make_fiber_points(L, n_fiber=1400, seed=12345)
    seed_pos_T = (out.pos_T_snapshots[0]
                  if len(out.pos_T_snapshots) else None)
    fallback_caf = _synthesize_CAFs(L, seed_pos_T, n_caf=25, seed=0)
    k = frame_idx
    if k < 0:
        k = len(out.pos_T_snapshots) + k
    _draw_presentation_frame(fig, axes, out, k, fiber_pts,
                             fallback_caf, title)
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return out_path
