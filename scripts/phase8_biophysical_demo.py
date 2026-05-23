"""Phase 8 — Biophysical (EMT + fibers + hypoxia) demonstration video.

Runs one demo simulation with all three new mechanisms (EMT, ECM fiber
alignment, hypoxia) and produces:

  outputs/videos/biophysical_demo.mp4    — annotated video (~16s @ 24fps)
  outputs/figures/biophysical_panel.png  — single annotated still frame
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim_biophysical import BiophysicalParams, run_biophysical
from src.render_tme import (
    render_biophysical_video,
    render_biophysical_still,
)


def main():
    out_videos = ROOT / "outputs" / "videos"
    out_figs = ROOT / "outputs" / "figures"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Modest cell counts so render stays brisk.  Turn ECM on so fiber
    # contact-guidance and integrin traction have a substrate to bite.
    params = BiophysicalParams(
        T_final=80.0,
        N_T_initial=50,
        N_I_initial=200,
        N_M_initial=80,
        rho_E_init=0.8,        # ECM substrate (so fibers / integrin matter)
        s_m=1.0,               # baseline (kept for back-compat; the EMT loop
                               # uses per-cell s_m_epi/s_m_mes regardless)
        k_deg=1.0,
        p_div=0.005,
        p_kill=0.04,
        p_phag=0.04,
        chi_s=10.0,
        fiber_alignment_strength=0.6,
    )
    print(f"[phase8] starting run_biophysical, T_final={params.T_final}...")
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=11, snapshot_every=25)
    sim_dt = time.perf_counter() - t0
    print(f"[phase8] sim done in {sim_dt:.1f}s -- {len(out.pos_T_snapshots)} frames")
    print(f"   final n_T={out.n_T[-1]}, n_I={out.n_I[-1]}, n_M={out.n_M[-1]}")
    print(f"   mean_EMT range:  {min(out.mean_EMT):.3f}..{max(out.mean_EMT):.3f}")
    print(f"   frac_mes range:  {min(out.frac_mes):.3f}..{max(out.frac_mes):.3f}")
    print(f"   max invasion:    {max(out.max_invasion_distance):.2f}")
    print(f"   hypoxic area:    {out.hypoxic_area:.3f}")
    print(f"   ECM degraded:    {out.ecm_degraded_area:.3f}")

    # ---- video ----
    video_path = out_videos / "biophysical_demo.mp4"
    print(f"[phase8] rendering video -> {video_path}")
    t0 = time.perf_counter()
    render_biophysical_video(out, video_path, fps=24, dpi=100, bitrate=4500)
    print(f"   video rendered in {time.perf_counter()-t0:.1f}s")

    # ---- still ----
    panel_path = out_figs / "biophysical_panel.png"
    frame_idx = max(0, int(0.75 * len(out.pos_T_snapshots)) - 1)
    print(f"[phase8] rendering still (frame {frame_idx}) -> {panel_path}")
    render_biophysical_still(out, frame_idx, panel_path, dpi=180)

    print("[phase8] done.")


if __name__ == "__main__":
    main()
