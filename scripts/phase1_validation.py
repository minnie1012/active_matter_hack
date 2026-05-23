"""Phase 1 validation: MIPS sanity check + Fisher–KPP growth front.

Outputs:
  outputs/figures/phase1_mips.png
  outputs/figures/phase1_kpp_front.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from src import style
from src.sim import TumorParams, run_single_species


# ---------------------------------------------------------------------------
# MIPS sanity check: high-density ABPs at elevated Peclet → visible clustering
# ---------------------------------------------------------------------------

def fig_mips():
    style.apply_style()
    # High Peclet by raising v, lowering D_R; high packing fraction
    params = TumorParams(
        L=40.0,         # smaller box → higher packing
        v=1.0,          # 10x default → high Peclet
        D_R=0.05,       # long persistence
        D_T=0.005,
        sigma=1.0,
        k_rep=80.0,     # stiffer to enforce overlap penalty
        p_div=0.0,
        N_max=1200,
    )
    n0 = 600   # phi = 600 * pi * 0.25 / 1600 ~ 0.29
    out = run_single_species(
        params,
        n_initial=n0,
        n_steps=4000,
        init="uniform",
        snapshot_every=4000,
        seed=11,
        enable_proliferation=False,
    )
    final = out.pos_snapshots[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)

    # left: initial snapshot
    ax = axes[0]
    ax.set_facecolor(style.BG)
    init_xy = out.pos_snapshots[0]
    n_init = init_xy.shape[0]
    coll0 = EllipseCollection(
        widths=np.full(n_init, style.TUMOR_DIAM_DATA),
        heights=np.full(n_init, style.TUMOR_DIAM_DATA),
        angles=np.zeros(n_init),
        units="x",
        offsets=init_xy,
        transOffset=ax.transData,
        facecolors=style.TUMOR,
        edgecolors=style.TUMOR_EDGE,
        linewidths=0.4,
        alpha=style.PARTICLE_ALPHA,
    )
    ax.add_collection(coll0)
    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_title(f"t = 0   (N = {n_init})", color=style.FG)
    ax.set_xticks([]); ax.set_yticks([])

    # right: final snapshot showing clustering
    ax = axes[1]
    ax.set_facecolor(style.BG)
    n_fin = final.shape[0]
    coll1 = EllipseCollection(
        widths=np.full(n_fin, style.TUMOR_DIAM_DATA),
        heights=np.full(n_fin, style.TUMOR_DIAM_DATA),
        angles=np.zeros(n_fin),
        units="x",
        offsets=final,
        transOffset=ax.transData,
        facecolors=style.TUMOR,
        edgecolors=style.TUMOR_EDGE,
        linewidths=0.4,
        alpha=style.PARTICLE_ALPHA,
    )
    ax.add_collection(coll1)
    ax.set_xlim(0, params.L)
    ax.set_ylim(0, params.L)
    ax.set_aspect("equal")
    ax.set_title(
        f"t = {out.times[-1]:.0f}   (Pe = v/(D_R·σ) = {params.v/(params.D_R*params.sigma):.0f})",
        color=style.FG,
    )
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Phase 1 sanity: high-Peclet ABPs cluster (motility-induced)",
        fontsize=style.TITLE_SIZE, color=style.FG, y=0.97,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase1_mips.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG)
    print(f"wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fisher–KPP front: small seed grows outward; radial density shows propagating
# front
# ---------------------------------------------------------------------------

def fig_kpp_front():
    style.apply_style()
    params = TumorParams(
        L=100.0,
        v=0.1,          # spec default
        D_R=0.1,
        D_T=0.001,
        sigma=1.0,
        k_rep=30.0,
        p_div=0.005,    # spec default
        nbr_radius=1.5,
        nbr_threshold=6,
        N_max=4000,
    )
    out = run_single_species(
        params,
        n_initial=50,
        n_steps=15000,        # T = 150
        init="disk",
        init_radius=3.0,
        snapshot_every=2500,  # 6 snapshots + final
        seed=3,
    )

    # radial density profile relative to box center
    center = np.array([0.5 * params.L, 0.5 * params.L])
    r_edges = np.linspace(0, 30, 30)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    ring_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)

    # left: final state
    ax = axes[0]
    ax.set_facecolor(style.BG)
    final = out.pos_snapshots[-1]
    coll = EllipseCollection(
        widths=np.full(final.shape[0], style.TUMOR_DIAM_DATA),
        heights=np.full(final.shape[0], style.TUMOR_DIAM_DATA),
        angles=np.zeros(final.shape[0]),
        units="x",
        offsets=final,
        transOffset=ax.transData,
        facecolors=style.TUMOR,
        edgecolors=style.TUMOR_EDGE,
        linewidths=0.3,
        alpha=style.PARTICLE_ALPHA,
    )
    ax.add_collection(coll)
    # 30-unit ROI around center
    ax.set_xlim(center[0] - 30, center[0] + 30)
    ax.set_ylim(center[1] - 30, center[1] + 30)
    ax.set_aspect("equal")
    ax.set_title(f"t = {out.times[-1]:.0f},  N = {final.shape[0]}", color=style.FG)
    ax.set_xticks([]); ax.set_yticks([])

    # right: radial density curves over time
    ax = axes[1]
    ax.set_facecolor(style.BG)
    cmap = plt.get_cmap("magma")
    n_curves = len(out.pos_snapshots)
    for k, (xy, t) in enumerate(zip(out.pos_snapshots, out.times)):
        d = np.linalg.norm(xy - center, axis=1)
        counts, _ = np.histogram(d, bins=r_edges)
        density = counts / ring_areas
        ax.plot(
            r_centers,
            density,
            color=cmap(0.15 + 0.7 * k / max(1, n_curves - 1)),
            linewidth=2,
            label=f"t={t:.0f}",
        )
    ax.set_xlabel("r from center  (cell diameters)")
    ax.set_ylabel(r"local density  $\rho(r)$")
    ax.set_title("Radial density: growing front", color=style.FG)
    ax.legend(loc="upper right", frameon=False, fontsize=style.SMALL_SIZE, ncol=2)
    ax.set_xlim(0, 30)

    fig.suptitle(
        "Phase 1 sanity: tumor seed grows outward with a propagating front",
        fontsize=style.TITLE_SIZE, color=style.FG, y=0.98,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase1_kpp_front.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG)
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating MIPS sanity figure...")
    fig_mips()
    print("Generating Fisher–KPP front figure...")
    fig_kpp_front()
    print("done.")
