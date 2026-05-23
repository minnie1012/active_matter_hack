"""Phase 7 — TME full-stack demonstration video.

Runs one demo simulation with all the new mechanisms (hypoxia, angiogenesis,
NK / DC / MDSC populations) and produces:

  outputs/videos/tme_full.mp4    — annotated full-frame video (~10s @ 24fps)
  outputs/figures/tme_panel.png  — single annotated frame from late in sim
  outputs/figures/tme_legend.png — standalone cell-type legend strip
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.sim_tme import TMEParams, run_tme
from src.render_tme import (
    render_tme_video,
    render_tme_still,
    save_legend_image,
)


def main():
    out_videos = ROOT / "outputs" / "videos"
    out_figs = ROOT / "outputs" / "figures"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Demo params: modest cell counts so render is brisk
    params = TMEParams(
        T_final=120.0,
        N_T_initial=40,
        N_I_initial=200,
        N_M_initial=80,
        N_NK_initial=60,
        N_DC_initial=40,
        N_MDSC_initial=80,
        # tuned for visible hypoxia + angiogenesis + tumor growth balance
        p_div=0.007,
        p_kill=0.03,
        p_kill_NK=0.02,
        p_phag=0.03,
        chi_s=12.0,
        D_VEGF=6.0,
        # one feeder vessel on the side; angiogenesis builds the branched tree
        n_vessels_init=1,
        n_vessels_max=64,
    )
    print(f"[phase7] starting run_tme, T_final={params.T_final}...")
    t0 = time.perf_counter()
    out = run_tme(params=params, seed=17, snapshot_every=25)
    sim_dt = time.perf_counter() - t0
    print(f"[phase7] sim done in {sim_dt:.1f}s -- {len(out.pos_T_snapshots)} frames")
    print(f"   final n_T={out.n_T[-1]}, n_I={out.n_I[-1]}, n_M={out.n_M[-1]}, "
          f"n_NK={out.n_NK[-1]}, n_DC={out.n_DC[-1]}, n_MDSC={out.n_MDSC[-1]}, "
          f"n_vessels={out.n_vessels[-1]}")
    print(f"   hypoxic fraction range: "
          f"{min(out.hypoxic_fraction):.2f}..{max(out.hypoxic_fraction):.2f}")
    print(f"   cumulative kills (CD8): {out.n_killed_cum[-1]}   "
          f"(NK): {out.n_killed_NK_cum[-1]}   phag: {out.n_phag_cum[-1]}   "
          f"born: {out.n_born_cum[-1]}")

    # ---- video ----
    video_path = out_videos / "tme_full.mp4"
    print(f"[phase7] rendering video → {video_path}")
    t0 = time.perf_counter()
    render_tme_video(out, video_path, fps=24, dpi=100, bitrate=4500)
    print(f"   video rendered in {time.perf_counter()-t0:.1f}s")

    # ---- still: pick a late-time frame (3/4 through) ----
    panel_path = out_figs / "tme_panel.png"
    frame_idx = int(0.75 * len(out.pos_T_snapshots))
    print(f"[phase7] rendering still (frame {frame_idx}) → {panel_path}")
    render_tme_still(out, frame_idx, panel_path, dpi=180)

    # ---- legend strip ----
    legend_path = out_figs / "tme_legend.png"
    print(f"[phase7] rendering legend strip → {legend_path}")
    save_legend_image(legend_path, dpi=180)

    print("[phase7] done.")


if __name__ == "__main__":
    main()
