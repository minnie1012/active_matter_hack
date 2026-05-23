"""Phase 7c — side-by-side comparison: hypoxia ON vs. hypoxia OFF.

Runs the TME simulator twice with the same seed:
  (1) default params              -> outputs/videos/tme_with_hypoxia.mp4
  (2) hypoxia couplings disabled  -> outputs/videos/tme_no_hypoxia.mp4

The "no hypoxia" run keeps the O2 / VEGF fields and vessels seeded, but
zeroes every behavioral coupling so:
  - tumor division is not slowed by low O2
  - CD8 / NK kill probability is not reduced by low O2
  - hypoxic tumor cells stop secreting VEGF
  - vessels stop drifting up the VEGF gradient and stop sprouting
This is the "no HIF response" baseline: same oxygen plumbing, no signaling.

Also writes a side-by-side comparison figure showing the late-time frame of
each run at ``outputs/figures/tme_hypoxia_compare.png``.
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.sim_tme import TMEParams, run_tme
from src.render_tme import (
    render_tme_video,
    render_tme_still,
)


def default_params() -> TMEParams:
    """Same as phase7_tme_video.py — hypoxia ON."""
    return TMEParams(
        T_final=120.0,
        N_T_initial=40,
        N_I_initial=200,
        N_M_initial=80,
        N_NK_initial=60,
        N_DC_initial=40,
        N_MDSC_initial=80,
        p_div=0.007,
        p_kill=0.03,
        p_kill_NK=0.02,
        p_phag=0.03,
        chi_s=12.0,
        D_VEGF=6.0,
        n_vessels_init=1,
        n_vessels_max=64,
    )


def no_hypoxia_params() -> TMEParams:
    """Disable every hypoxia/angiogenesis coupling. Same seed, same cell counts."""
    p = default_params()
    p.O2_hyp_thresh = -1.0          # nothing ever counts as hypoxic
    p.hypoxia_kill_penalty = 0.0    # CD8 / NK kill prob unchanged in low O2
    p.O2_div_thresh = -10.0         # sigmoid always returns ~1 -> no O2 effect on division
    p.s_VEGF_hyp = 0.0              # no VEGF secretion
    p.chi_vessel = 0.0              # vessels do not drift
    p.sprout_rate = 0.0             # no new vessel sprouts
    return p


def run_and_render(params: TMEParams, label: str, seed: int) -> "TMEOut":
    out_videos = ROOT / "outputs" / "videos"
    out_figs = ROOT / "outputs" / "figures"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {label.upper()} ===")
    t0 = time.perf_counter()
    out = run_tme(params=params, seed=seed, snapshot_every=25)
    print(f"  sim {time.perf_counter()-t0:.1f}s  --  "
          f"{len(out.pos_T_snapshots)} frames")
    print(f"  final n_T={out.n_T[-1]}, n_I={out.n_I[-1]}, "
          f"n_M={out.n_M[-1]}, n_NK={out.n_NK[-1]}, "
          f"n_vessels={out.n_vessels[-1]}")
    print(f"  hypoxic fraction: "
          f"{min(out.hypoxic_fraction):.2f} -> {max(out.hypoxic_fraction):.2f}")
    print(f"  cumulative kills: CD8={out.n_killed_cum[-1]}, "
          f"NK={out.n_killed_NK_cum[-1]}, born={out.n_born_cum[-1]}")

    video_path = out_videos / f"tme_{label}.mp4"
    t0 = time.perf_counter()
    render_tme_video(out, video_path, fps=24, dpi=100, bitrate=4500)
    print(f"  video {time.perf_counter()-t0:.1f}s -> {video_path.name}")
    return out


def make_compare_figure(out_with: "TMEOut", out_without: "TMEOut",
                        path: Path) -> None:
    """Two-panel late-time frame comparison."""
    from src.render_tme import compose_tme_frame, make_tme_figure
    # Easier: just render two stills and tile them
    tmp_a = path.with_name("_tmp_with.png")
    tmp_b = path.with_name("_tmp_no.png")
    k_with = len(out_with.pos_T_snapshots) - 1
    k_no = len(out_without.pos_T_snapshots) - 1
    render_tme_still(out_with, k_with, tmp_a, dpi=140)
    render_tme_still(out_without, k_no, tmp_b, dpi=140)

    img_a = plt.imread(tmp_a)
    img_b = plt.imread(tmp_b)
    fig = plt.figure(figsize=(20, 6.5), dpi=140)
    fig.patch.set_facecolor("white")
    ax1 = fig.add_axes([0.0, 0.05, 0.50, 0.88]); ax1.axis("off")
    ax2 = fig.add_axes([0.50, 0.05, 0.50, 0.88]); ax2.axis("off")
    ax1.imshow(img_a); ax2.imshow(img_b)
    fig.text(0.25, 0.965, "WITH HYPOXIA + ANGIOGENESIS",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.75, 0.965, "NO HYPOXIA (couplings disabled)",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    tmp_a.unlink(missing_ok=True)
    tmp_b.unlink(missing_ok=True)
    print(f"  comparison -> {path}")


def main() -> None:
    seed = 17
    out_with = run_and_render(default_params(),     "with_hypoxia", seed)
    out_no   = run_and_render(no_hypoxia_params(),  "no_hypoxia",   seed)

    fig_path = ROOT / "outputs" / "figures" / "tme_hypoxia_compare.png"
    make_compare_figure(out_with, out_no, fig_path)


if __name__ == "__main__":
    main()
