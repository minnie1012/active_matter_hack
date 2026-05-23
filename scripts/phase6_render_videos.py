"""Render videos for the extended-physics demos.

Outputs:
  outputs/videos/extended_macrophage.mp4    — 3 species + IL-10 field + polarization colors
  outputs/videos/extended_combo.mp4         — combo therapy (anti-PD-1 + TAM repol)
  outputs/videos/extended_ecm.mp4           — tumor + ECM density field + MMP cavity
  outputs/videos/extended_adhesion.mp4      — cadherin-driven clustering
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from src import style
from src.sim_extended import ExtendedParams, run_extended
from src.sim_ecm import ECMParams, run_extended_ecm
from src.sim_adhesion import AdhesionParams, run_adhesion


# Point matplotlib at the bundled ffmpeg from imageio-ffmpeg
try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Shared composer for a 3-species + field frame
# ---------------------------------------------------------------------------

def _draw_three_species_frame(
    main_ax, side_ax,
    pos_T, pos_I, pos_M, p_M, field, L,
    t, headers, n_T_traj, n_I_traj, n_M_traj, mean_pM_traj, t_traj,
    field_cmap, field_label,
):
    """Compose the standard 3-species frame + sidebar trajectories."""
    main_ax.clear()
    main_ax.set_facecolor(style.BG)
    main_ax.imshow(
        gaussian_filter(field, 0.7), extent=[0, L, 0, L], origin="lower",
        cmap=field_cmap, alpha=style.FIELD_ALPHA, interpolation="bilinear",
    )
    # macrophages first (under T cells / tumor)
    if pos_M is not None and len(pos_M):
        cmap = plt.get_cmap("RdBu_r")
        norm = Normalize(vmin=-1, vmax=1)
        main_ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_M), 1.6), heights=np.full(len(pos_M), 1.6),
            angles=np.zeros(len(pos_M)), units="x", offsets=pos_M,
            transOffset=main_ax.transData,
            facecolors=cmap(norm(p_M)), edgecolors="none", alpha=0.30,
        ))
        main_ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_M), 1.1), heights=np.full(len(pos_M), 1.1),
            angles=np.zeros(len(pos_M)), units="x", offsets=pos_M,
            transOffset=main_ax.transData,
            facecolors=cmap(norm(p_M)), edgecolors=style.MUTED, linewidths=0.3,
            alpha=0.95,
        ))
    if len(pos_I):
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
    main_ax.set_xlim(0, L); main_ax.set_ylim(0, L); main_ax.set_aspect("equal")
    main_ax.set_xticks([]); main_ax.set_yticks([])
    for s in main_ax.spines.values():
        s.set_edgecolor(style.MUTED)
    main_ax.text(0.5, 1.005, headers, transform=main_ax.transAxes,
                 color=style.FG, family=style.FONT_MONO,
                 fontsize=style.SMALL_SIZE, ha="center", va="bottom")
    # tiny label for field
    main_ax.text(0.02, 0.02, f"field: {field_label}",
                 transform=main_ax.transAxes,
                 color=style.MUTED, family=style.FONT_MONO,
                 fontsize=style.SMALL_SIZE - 1, ha="left", va="bottom")

    # sidebar trajectories
    side_ax.clear()
    side_ax.set_facecolor(style.BG)
    n = max(2, len(t_traj))
    side_ax.plot(t_traj, n_T_traj, color=style.TUMOR, lw=1.8, label=r"$N_T$")
    side_ax.plot(t_traj, n_I_traj, color=style.TCELL, lw=1.8, label=r"$N_I$")
    if n_M_traj is not None:
        side_ax.plot(t_traj, n_M_traj, color=style.ACCENT, lw=1.8, label=r"$N_M$")
    side_ax.set_xlabel("t", color=style.FG, fontsize=style.SMALL_SIZE)
    side_ax.set_ylabel("count", color=style.FG, fontsize=style.SMALL_SIZE)
    side_ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)
    for s in side_ax.spines.values():
        s.set_edgecolor(style.MUTED)
    side_ax.legend(frameon=False, fontsize=style.SMALL_SIZE - 1, loc="upper left")

    if mean_pM_traj is not None:
        ax_p = side_ax.twinx()
        ax_p.plot(t_traj, mean_pM_traj, color="#A0A8C8", lw=1.4, ls="--",
                  label=r"$\langle p \rangle$")
        ax_p.set_ylabel(r"$\langle p \rangle$", color="#A0A8C8",
                        fontsize=style.SMALL_SIZE)
        ax_p.tick_params(colors="#A0A8C8", labelsize=style.SMALL_SIZE - 1)
        ax_p.set_ylim(-1, 1)
        ax_p.axhline(0, color="#404868", lw=0.6, ls=":")
        for s in ax_p.spines.values():
            s.set_edgecolor(style.MUTED)


def _save_anim(fig, draw_fn, n_frames, out_path, fps=30, dpi=100):
    style.apply_style()
    anim = FuncAnimation(fig, draw_fn, frames=n_frames,
                          interval=1000 / fps, blit=False, repeat=False)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(out_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": style.BG})
        print(f"  wrote {out_path}")
    except Exception as e:
        print(f"  FFMpeg failed ({e}); writing GIF instead.")
        gif_path = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": style.BG})
        print(f"  wrote {gif_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Macrophage polarization video
# ---------------------------------------------------------------------------

def video_macrophages():
    print("Video 1: macrophage polarization (no treatment)")
    p = ExtendedParams(use_macrophages=True, T_final=160.0,
                       N_I_initial=300, N_M_initial=120, chi_s=5.0)
    t0 = time.perf_counter()
    out = run_extended(params=p, seed=5, snapshot_every=25)
    print(f"  sim: {time.perf_counter()-t0:.1f} s, {len(out.pos_T_snapshots)} frames")

    style.apply_style()
    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax = fig.add_axes([0.74, 0.20, 0.22, 0.55])
    fig.suptitle("Macrophage polarization (3-species sim)",
                 color=style.FG, fontsize=14, y=0.965)

    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I); nM = np.asarray(out.n_M)
    mp = np.asarray(out.mean_pM)

    def draw(k):
        headers = (f"t = {out.times[k]:6.2f}    N_T = {out.n_T[k]:4d}    "
                   f"N_I = {out.n_I[k]:4d}    N_M = {out.n_M[k]:3d}    "
                   f"<p> = {out.mean_pM[k]:+.2f}")
        _draw_three_species_frame(
            main_ax, side_ax,
            out.pos_T_snapshots[k], out.pos_I_snapshots[k],
            out.pos_M_snapshots[k], out.p_M_snapshots[k],
            out.c_IL10_snapshots[k], p.L, out.times[k], headers,
            nT[:k+1], nI[:k+1], nM[:k+1], mp[:k+1], t_arr[:k+1],
            field_cmap=style.SUPPRESSANT_CMAP, field_label="c_IL10 (M2 → CD8 repellent)",
        )

    _save_anim(fig, draw, len(out.pos_T_snapshots),
               ROOT / "outputs" / "videos" / "extended_macrophage.mp4",
               fps=24, dpi=100)


# ---------------------------------------------------------------------------
# 2. Combo therapy video (anti-PD-1 + TAM repolarization)
# ---------------------------------------------------------------------------

def video_combo():
    print("Video 2: combo therapy (anti-PD-1 + TAM repolarization at t=20)")
    p = ExtendedParams(use_macrophages=True, T_final=80.0,
                       N_I_initial=300, N_M_initial=120, chi_s=10.0)
    t0 = time.perf_counter()
    out = run_extended(params=p, seed=11, snapshot_every=25,
                        treat_time=20.0, chi_s_after=0.0, M1_bias_after=15.0)
    print(f"  sim: {time.perf_counter()-t0:.1f} s, {len(out.pos_T_snapshots)} frames")

    style.apply_style()
    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax = fig.add_axes([0.74, 0.20, 0.22, 0.55])
    fig.suptitle("Combination therapy (α → 0, M1 bias → +15 at t = 20)",
                 color=style.FG, fontsize=14, y=0.965)

    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I); nM = np.asarray(out.n_M)
    mp = np.asarray(out.mean_pM)

    def draw(k):
        treat_label = " [TREATMENT]" if out.times[k] >= 20 else ""
        headers = (f"t = {out.times[k]:6.2f}{treat_label}   "
                   f"N_T = {out.n_T[k]:4d}   N_I = {out.n_I[k]:4d}   "
                   f"N_M = {out.n_M[k]:3d}   <p> = {out.mean_pM[k]:+.2f}")
        _draw_three_species_frame(
            main_ax, side_ax,
            out.pos_T_snapshots[k], out.pos_I_snapshots[k],
            out.pos_M_snapshots[k], out.p_M_snapshots[k],
            out.c_IL10_snapshots[k], p.L, out.times[k], headers,
            nT[:k+1], nI[:k+1], nM[:k+1], mp[:k+1], t_arr[:k+1],
            field_cmap=style.SUPPRESSANT_CMAP, field_label="c_IL10",
        )
        # mark the treatment time on the sidebar with a vertical line
        if 20.0 in t_arr[:k+1] or t_arr[:k+1].max() >= 20.0:
            side_ax.axvline(20.0, color=style.ACCENT, lw=1.2, ls="--", alpha=0.8)

    _save_anim(fig, draw, len(out.pos_T_snapshots),
               ROOT / "outputs" / "videos" / "extended_combo.mp4",
               fps=24, dpi=100)


# ---------------------------------------------------------------------------
# 3. ECM density + MMP cavity video
# ---------------------------------------------------------------------------

def video_ecm():
    print("Video 3: ECM density + MMP degradation")
    p = ECMParams(use_macrophages=False, T_final=80.0,
                   N_I_initial=100, chi_s=2.0,
                   rho_E_init=1.6, s_m=1.5)        # dense matrix, MMP active
    t0 = time.perf_counter()
    out = run_extended_ecm(params=p, seed=7, snapshot_every=25)
    print(f"  sim: {time.perf_counter()-t0:.1f} s, {len(out.pos_T_snapshots)} frames")

    # custom earthy colormap for ECM
    from matplotlib.colors import LinearSegmentedColormap
    ecm_cmap = LinearSegmentedColormap.from_list(
        "ecm", [(0.0, "#0A0E27"), (0.25, "#3A2E1E"), (0.55, "#7A5A3A"),
                (0.85, "#C9B27E"), (1.0, "#F4E2B8")])

    style.apply_style()
    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax = fig.add_axes([0.74, 0.20, 0.22, 0.55])
    fig.suptitle("ECM + MMP: tumor carves a depleted cavity",
                 color=style.FG, fontsize=14, y=0.965)

    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I)

    def draw(k):
        headers = (f"t = {out.times[k]:6.2f}   N_T = {out.n_T[k]:4d}   "
                   f"N_I = {out.n_I[k]:4d}   "
                   f"<ρ_E> = {float(np.mean(out.rho_E_snapshots[k])):.2f}")
        _draw_three_species_frame(
            main_ax, side_ax,
            out.pos_T_snapshots[k], out.pos_I_snapshots[k],
            None, None,
            out.rho_E_snapshots[k], p.L, out.times[k], headers,
            nT[:k+1], nI[:k+1], None, None, t_arr[:k+1],
            field_cmap=ecm_cmap, field_label="ρ_E (ECM density)",
        )

    _save_anim(fig, draw, len(out.pos_T_snapshots),
               ROOT / "outputs" / "videos" / "extended_ecm.mp4",
               fps=24, dpi=100)


# ---------------------------------------------------------------------------
# 4. Adhesion (cadherin) video
# ---------------------------------------------------------------------------

def video_adhesion():
    print("Video 4: cell-cell adhesion (cadherin)")
    p = AdhesionParams(use_macrophages=False, T_final=60.0,
                        N_I_initial=0, chi_s=0.0,
                        k_adh=12.0, sigma_adh=1.6, J_align=1.0)
    t0 = time.perf_counter()
    out = run_adhesion(params=p, seed=4, snapshot_every=25)
    print(f"  sim: {time.perf_counter()-t0:.1f} s, {len(out.pos_T_snapshots)} frames")

    style.apply_style()
    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax = fig.add_axes([0.74, 0.20, 0.22, 0.55])
    fig.suptitle("Cadherin adhesion + Vicsek alignment",
                 color=style.FG, fontsize=14, y=0.965)

    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I)
    G = p.G
    zeros_field = np.zeros((G, G), dtype=np.float64)

    def draw(k):
        headers = (f"t = {out.times[k]:6.2f}   N_T = {out.n_T[k]:4d}   "
                   f"k_adh = {p.k_adh:.1f}   J_align = {p.J_align:.1f}")
        _draw_three_species_frame(
            main_ax, side_ax,
            out.pos_T_snapshots[k], out.pos_I_snapshots[k],
            None, None,
            zeros_field, p.L, out.times[k], headers,
            nT[:k+1], nI[:k+1], None, None, t_arr[:k+1],
            field_cmap=style.SUPPRESSANT_CMAP, field_label="(no field — adhesion only)",
        )

    _save_anim(fig, draw, len(out.pos_T_snapshots),
               ROOT / "outputs" / "videos" / "extended_adhesion.mp4",
               fps=24, dpi=100)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    selectors = {
        "macrophage": video_macrophages,
        "combo": video_combo,
        "ecm": video_ecm,
        "adhesion": video_adhesion,
    }
    args = sys.argv[1:]
    targets = args if args else list(selectors)
    for name in targets:
        fn = selectors.get(name)
        if fn is None:
            print(f"  unknown target {name!r}; valid: {list(selectors)}")
            continue
        fn()
    print("\nDone.")
