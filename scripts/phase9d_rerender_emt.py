"""Phase 9d — re-render the two EMT-active videos (tme_full + tme_emt_color_change)
so that tumor cells are colored by their s_T value (pink epithelial -> violet
mesenchymal) instead of the flat-pink default in compose_tme_frame.

The renderer fix lives in src/render_presentation.py: after compose_tme_frame
paints the tumor cells flat pink, we overlay an EllipseCollection of the same
cells coloured per-cell via _emt_colors(s_T).

This driver re-runs the two scenarios that have EMT on:
  - tme_full   (all four mechanisms: EMT + fibers + hypoxia + angiogenesis)
  - tme_emt_color_change   (the EMT showcase from phase 9c)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim_biophysical import BiophysicalParams, run_biophysical
from src.render_presentation import render_presentation_video

# Re-use the helper toggles + scenario params from the phase 9 driver
sys.path.insert(0, str(ROOT / "scripts"))
from phase9_presentation_set import (         # noqa: E402
    SEED, T_FINAL, SNAPSHOT_EVERY,
    build_params, SCENARIOS,
)
from phase9c_emt_color_change import (        # noqa: E402
    make_params as make_emt_showcase_params,
)


OUT_DIR = ROOT / "outputs" / "video_for_presentation"


def rerender_scenario(label: str, toggles, title: str) -> None:
    print(f"\n=== {label.upper()} ===")
    params = build_params(toggles)
    params.T_final = T_FINAL
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=SEED,
                          snapshot_every=SNAPSHOT_EVERY)
    print(f"  sim {time.perf_counter()-t0:.1f}s -- "
          f"{len(out.pos_T_snapshots)} frames")
    print(f"  mean_EMT: {out.mean_EMT[0]:.3f} -> {out.mean_EMT[-1]:.3f}")
    mp4 = OUT_DIR / f"tme_{label}.mp4"
    t0 = time.perf_counter()
    render_presentation_video(out, mp4, title=title,
                              fps=24, dpi=100, bitrate=4500)
    print(f"  video {time.perf_counter()-t0:.1f}s -> {mp4.name}")


def rerender_emt_showcase() -> None:
    print("\n=== EMT_COLOR_CHANGE ===")
    params = make_emt_showcase_params()
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=11, snapshot_every=25)
    print(f"  sim {time.perf_counter()-t0:.1f}s -- "
          f"{len(out.pos_T_snapshots)} frames")
    print(f"  mean_EMT: {out.mean_EMT[0]:.3f} -> {out.mean_EMT[-1]:.3f}")
    mp4 = OUT_DIR / "tme_emt_color_change.mp4"
    t0 = time.perf_counter()
    render_presentation_video(
        out, mp4,
        title="EMT phenotype switch: pink (epithelial) -> violet (mesenchymal)",
        fps=24, dpi=100, bitrate=4500,
    )
    print(f"  video {time.perf_counter()-t0:.1f}s -> {mp4.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) tme_full: all four mechanisms
    label, toggles, title = next(s for s in SCENARIOS if s[0] == "full")
    rerender_scenario(label, toggles, title)
    # 2) tme_emt_color_change: EMT showcase
    rerender_emt_showcase()


if __name__ == "__main__":
    main()
