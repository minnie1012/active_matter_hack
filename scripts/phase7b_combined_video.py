"""Phase 7B: combined macrophages + ECM/MMP + cadherin-adhesion demo.

Renders one video that overlays all three published extensions: a coherent
tumor sheet (cadherin + Vicsek) carving an ECM cavity (MMP) inside an
immunosuppressive M2 halo (macrophages) that excludes CD8 cells.

Outputs:
  outputs/videos/extended_combined.mp4
  outputs/figures/combined_panel.png
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
from matplotlib.colors import LinearSegmentedColormap

from src import style
from src.sim_combined import CombinedParams, run_combined
from scripts.phase6_render_videos import _draw_three_species_frame, _save_anim


# point matplotlib at the bundled ffmpeg from imageio-ffmpeg (same as phase6)
try:
    import imageio_ffmpeg
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


# earthy ECM colormap, matched to phase6_render_videos.video_ecm
ECM_CMAP = LinearSegmentedColormap.from_list(
    "ecm",
    [
        (0.0, "#0A0E27"),
        (0.25, "#3A2E1E"),
        (0.55, "#7A5A3A"),
        (0.85, "#C9B27E"),
        (1.0, "#F4E2B8"),
    ],
)


def _make_params() -> CombinedParams:
    """Demo parameters that exercise all three extensions simultaneously."""
    return CombinedParams(
        use_macrophages=True,
        T_final=60.0,
        N_I_initial=200,
        N_M_initial=120,
        chi_s=8.0,
        rho_E_init=1.6,
        s_m=1.5,
        k_adh=12.0,
        sigma_adh=1.6,
        J_align=1.0,
    )


def _render_video(out, params, mp4_path, png_path, title):
    """Render the combined sim as an mp4 + a single late-time panel PNG."""
    style.apply_style()
    fig = plt.figure(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor(style.BG)
    main_ax = fig.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax = fig.add_axes([0.74, 0.20, 0.22, 0.55])
    fig.suptitle(title, color=style.FG, fontsize=13, y=0.965)

    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T)
    nI = np.asarray(out.n_I)
    nM = np.asarray(out.n_M)
    mp = np.asarray(out.mean_pM)

    def draw(k):
        headers = (
            f"t = {out.times[k]:6.2f}   N_T = {out.n_T[k]:4d}   "
            f"N_I = {out.n_I[k]:4d}   N_M = {out.n_M[k]:3d}   "
            f"<p> = {out.mean_pM[k]:+.2f}   "
            f"<rho_E> = {float(np.mean(out.rho_E_snapshots[k])):.2f}"
        )
        _draw_three_species_frame(
            main_ax, side_ax,
            out.pos_T_snapshots[k], out.pos_I_snapshots[k],
            out.pos_M_snapshots[k], out.p_M_snapshots[k],
            out.rho_E_snapshots[k], params.L, out.times[k], headers,
            nT[: k + 1], nI[: k + 1], nM[: k + 1], mp[: k + 1], t_arr[: k + 1],
            field_cmap=ECM_CMAP, field_label="rho_E (ECM density)",
        )

    n_frames = len(out.pos_T_snapshots)
    print(f"  rendering {n_frames} frames -> {mp4_path.name}")
    _save_anim(fig, draw, n_frames, mp4_path, fps=24, dpi=100)

    # ---- single late-time panel for slides ----
    style.apply_style()
    fig2 = plt.figure(figsize=(12, 7), dpi=120)
    fig2.patch.set_facecolor(style.BG)
    main_ax2 = fig2.add_axes([0.04, 0.06, 0.62, 0.80])
    side_ax2 = fig2.add_axes([0.74, 0.20, 0.22, 0.55])
    fig2.suptitle(title, color=style.FG, fontsize=13, y=0.965)

    k = n_frames - 1
    headers = (
        f"t = {out.times[k]:6.2f}   N_T = {out.n_T[k]:4d}   "
        f"N_I = {out.n_I[k]:4d}   N_M = {out.n_M[k]:3d}   "
        f"<p> = {out.mean_pM[k]:+.2f}   "
        f"<rho_E> = {float(np.mean(out.rho_E_snapshots[k])):.2f}"
    )
    _draw_three_species_frame(
        main_ax2, side_ax2,
        out.pos_T_snapshots[k], out.pos_I_snapshots[k],
        out.pos_M_snapshots[k], out.p_M_snapshots[k],
        out.rho_E_snapshots[k], params.L, out.times[k], headers,
        nT, nI, nM, mp, t_arr,
        field_cmap=ECM_CMAP, field_label="rho_E (ECM density)",
    )
    fig2.savefig(png_path, dpi=120, facecolor=style.BG)
    plt.close(fig2)
    print(f"  wrote {png_path}")


def smoke_test():
    """Quick smoke run to catch crashes before the full render."""
    print("Smoke test: T_final=2.0, snapshot_every=10")
    p = _make_params()
    p.T_final = 2.0
    t0 = time.perf_counter()
    out = run_combined(params=p, seed=7, snapshot_every=10)
    dt_run = time.perf_counter() - t0
    print(
        f"  smoke ok in {dt_run:.1f}s : {len(out.pos_T_snapshots)} frames, "
        f"N_T_final={out.n_T[-1]}, N_I_final={out.n_I[-1]}, "
        f"N_M_final={out.n_M[-1]}, <p>={out.mean_pM[-1]:+.2f}, "
        f"<rho_E>={out.mean_rho_E[-1]:.2f}, "
        f"stuck_cum={out.n_stuck_cum[-1]}"
    )


def full_render():
    print("Full render: T_final=60.0, snapshot_every=25")
    p = _make_params()
    t0 = time.perf_counter()
    out = run_combined(params=p, seed=7, snapshot_every=25)
    print(
        f"  sim done in {time.perf_counter()-t0:.1f}s; "
        f"{len(out.pos_T_snapshots)} frames"
    )

    title = "Macrophages + ECM/MMP + adhesion: immune-excluded cold tumor"

    mp4_path = ROOT / "outputs" / "videos" / "extended_combined.mp4"
    png_path = ROOT / "outputs" / "figures" / "combined_panel.png"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    _render_video(out, p, mp4_path, png_path, title)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke_test()
    else:
        smoke_test()
        full_render()
        print("\nDone.")
