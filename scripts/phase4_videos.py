"""Render one MP4 per phase + treatment via src.render.make_video.

Outputs:
  outputs/videos/clearance.mp4
  outputs/videos/control.mp4
  outputs/videos/escape.mp4
  outputs/videos/treatment.mp4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.sim import SimParams, run, run_with_treatment
from src.render import make_video


PHASES = [
    # (label, rho_I, alpha, seed)
    ("clearance", 800, 0.0,  1),
    ("control",   229, 3.57, 1),     # the seed that showed dormancy then escape
    ("escape",    50,  10.0, 1),
]


def render_phase(label: str, rho_I: int, alpha: float, seed: int) -> None:
    base = SimParams()
    print(f"-- {label}: rho_I={rho_I}, alpha={alpha}, seed={seed}")
    t0 = time.perf_counter()
    out = run(
        rho_I=rho_I, alpha=alpha, seed=seed,
        params=base,
        snapshot_every=25,            # 400 frames at T_final=100, dt=0.01
        save_fields=True,
    )
    print(f"   sim: {time.perf_counter() - t0:.1f} s, {len(out.pos_T_snapshots)} frames; final N_T={out.n_T[-1]}")
    out_path = ROOT / "outputs" / "videos" / f"{label}.mp4"
    make_video(
        pos_T_snapshots=out.pos_T_snapshots,
        pos_I_snapshots=out.pos_I_snapshots,
        c_s_snapshots=out.c_s_snapshots,
        times=out.times,
        n_T_traj=out.n_T,
        n_I_traj=out.n_I,
        L=base.L,
        rho_I=rho_I,
        alpha=alpha,
        out_path=out_path,
        fps=24,
        title=label.upper(),
    )


def render_treatment() -> None:
    base = SimParams()
    rho_I = 300
    alpha = 10.0
    alpha_after = 0.0
    t_treat = 20.0
    seed = 3
    print(f"-- treatment: rho_I={rho_I}, alpha {alpha} -> {alpha_after} at t={t_treat}")
    t0 = time.perf_counter()
    out = run_with_treatment(
        rho_I=rho_I, alpha=alpha, alpha_after=alpha_after,
        t_treat=t_treat, seed=seed, params=base, snapshot_every=25,
    )
    print(f"   sim: {time.perf_counter() - t0:.1f} s, {len(out.pos_T_snapshots)} frames")
    out_path = ROOT / "outputs" / "videos" / "treatment.mp4"
    make_video(
        pos_T_snapshots=out.pos_T_snapshots,
        pos_I_snapshots=out.pos_I_snapshots,
        c_s_snapshots=out.c_s_snapshots,
        times=out.times,
        n_T_traj=out.n_T,
        n_I_traj=out.n_I,
        L=base.L,
        rho_I=rho_I,
        alpha=alpha,
        out_path=out_path,
        fps=24,
        title=f"TREATMENT  (α: {alpha:.1f} → {alpha_after:.1f} @ t={t_treat:.0f})",
    )


def main():
    for label, rho_I, alpha, seed in PHASES:
        render_phase(label, rho_I, alpha, seed)
    render_treatment()


if __name__ == "__main__":
    main()
