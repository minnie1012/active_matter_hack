"""Phase 2 validation: chemotaxis-to-static-source + small-tumor-kill.

Outputs:
  outputs/figures/phase2_chemotaxis_static.png
  outputs/figures/phase2_small_tumor_kill.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.ndimage import gaussian_filter

from src import style
from src.sim import SimParams, init_two_species, run, _step_particles_two_species
from src.fields import deposit_tumor_density, step_fields, static_gaussian_field


# ---------------------------------------------------------------------------
# (a) T cells aggregate at a static Gaussian attractant source.
#
#     We pre-bake c_a as a Gaussian centered at the box, set c_s = 0, freeze
#     the fields (no field update calls), and let T cells move under
#     chemotaxis only (no tumor cells alive). We then measure their radial
#     density relative to the box center after a long time and confirm a
#     clear peak at small r.
# ---------------------------------------------------------------------------

def fig_chemotaxis_static():
    style.apply_style()
    params = SimParams(
        L=60.0, G=64, T_final=80.0,
        v_I=0.5, D_R_I=0.3,
        chi_a=20.0, chi_s=0.0,
        N_T_initial=0, N_I_initial=200,
        N_T_max=8, N_I_max=256,
        p_div=0.0,           # no tumor proliferation either
        D_T_I=0.001,
    )
    np.random.seed(0)
    state = init_two_species(params, seed=0)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]

    # static Gaussian attractant; suppressant stays zero
    c_a = static_gaussian_field(params.L, params.G, params.L / 2, params.L / 2, sig=5.0, amp=20.0)
    c_s = np.zeros((params.G, params.G), dtype=np.float64)

    initial_xy = pos_I[alive_I].copy()

    # iterate particles only (skip step_fields so the source stays static)
    for step in range(params.n_steps):
        _step_particles_two_species(
            pos_T, theta_T, alive_T,
            pos_I, theta_I, alive_I,
            c_a, c_s,
            params.dt, params.v_T, params.v_I,
            params.D_R_T, params.D_R_I, params.D_T_T, params.D_T_I,
            params.sigma_T, params.k_rep_T,
            params.sigma_I, params.k_rep_I,
            params.sigma_TI, params.k_rep_TI, params.L,
            params.chi_a, params.chi_s,
            params.r_kill, 0.0,                 # no killing
            0.0, params.nbr_radius, params.nbr_threshold,
        )

    final_xy = pos_I[alive_I].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=style.DPI,
                              gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.patch.set_facecolor(style.BG)

    # left: initial positions
    ax = axes[0]
    ax.set_facecolor(style.BG)
    ax.imshow(
        gaussian_filter(c_a, sigma=0.6),
        extent=[0, params.L, 0, params.L], origin="lower",
        cmap=style.ATTRACTANT_CMAP, alpha=style.FIELD_ALPHA,
        interpolation="bilinear",
    )
    n0 = initial_xy.shape[0]
    ax.add_collection(EllipseCollection(
        widths=np.full(n0, style.TCELL_DIAM_DATA),
        heights=np.full(n0, style.TCELL_DIAM_DATA),
        angles=np.zeros(n0), units="x", offsets=initial_xy,
        transOffset=ax.transData,
        facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,
        linewidths=0.4, alpha=style.PARTICLE_ALPHA,
    ))
    ax.set_xlim(0, params.L); ax.set_ylim(0, params.L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = 0    (N_T = {n0})", color=style.FG)

    # mid: final positions
    ax = axes[1]
    ax.set_facecolor(style.BG)
    ax.imshow(
        gaussian_filter(c_a, sigma=0.6),
        extent=[0, params.L, 0, params.L], origin="lower",
        cmap=style.ATTRACTANT_CMAP, alpha=style.FIELD_ALPHA,
        interpolation="bilinear",
    )
    n1 = final_xy.shape[0]
    ax.add_collection(EllipseCollection(
        widths=np.full(n1, style.TCELL_DIAM_DATA),
        heights=np.full(n1, style.TCELL_DIAM_DATA),
        angles=np.zeros(n1), units="x", offsets=final_xy,
        transOffset=ax.transData,
        facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,
        linewidths=0.4, alpha=style.PARTICLE_ALPHA,
    ))
    ax.set_xlim(0, params.L); ax.set_ylim(0, params.L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {params.T_final:.0f}    (T cells aggregate)", color=style.FG)

    # right: radial density profile
    ax = axes[2]
    ax.set_facecolor(style.BG)
    center = np.array([params.L / 2, params.L / 2])
    r_edges = np.linspace(0, params.L / 2, 25)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    ring_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    d0 = np.linalg.norm(initial_xy - center, axis=1)
    d1 = np.linalg.norm(final_xy - center, axis=1)
    h0, _ = np.histogram(d0, bins=r_edges)
    h1, _ = np.histogram(d1, bins=r_edges)
    ax.plot(r_centers, h0 / ring_areas, color=style.MUTED, linewidth=2.0, label="t = 0")
    ax.plot(r_centers, h1 / ring_areas, color=style.TCELL, linewidth=2.5, label=f"t = {params.T_final:.0f}")
    ax.fill_between(r_centers, 0, h1 / ring_areas, alpha=0.18, color=style.TCELL)
    ax.set_xlabel("r from source  (cell diameters)")
    ax.set_ylabel(r"T-cell density  $\rho_I(r)$")
    ax.set_title("Chemotactic peak at source", color=style.FG)
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE)

    fig.suptitle(
        "Phase 2 sanity: T cells chemotax up a static attractant gradient",
        fontsize=style.TITLE_SIZE, color=style.FG, y=0.99,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase2_chemotaxis_static.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG)
    print(f"wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) Small-tumor-kill sanity:
#     50 tumor in a tight disk at center, 200 T cells uniformly distributed.
#     Disable proliferation (p_div=0) so we test killing in isolation.
#     Expect N_tumor(t) to decline from 50 toward 0.
# ---------------------------------------------------------------------------

def fig_small_tumor_kill():
    style.apply_style()
    params = SimParams(
        L=60.0, G=64, T_final=80.0,
        chi_a=10.0, chi_s=0.0,
        N_T_initial=50, N_I_initial=200, tumor_disk_radius=4.0,
        p_div=0.0,                # killing-only test
        r_kill=1.5, p_kill=0.1,
        N_T_max=128, N_I_max=256,
    )
    t0 = time.perf_counter()
    out = run(rho_I=200, alpha=0.0, seed=1, params=params)
    print(f"  small-tumor-kill run: {time.perf_counter() - t0:.2f} s")
    times = np.array(out.times)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)

    # left: final snapshot with suppressant absent, attractant shown
    ax = axes[0]
    ax.set_facecolor(style.BG)
    final_ca = out.c_a_snapshots[-1] if out.c_a_snapshots else np.zeros((params.G, params.G))
    ax.imshow(
        gaussian_filter(final_ca, sigma=0.6),
        extent=[0, params.L, 0, params.L], origin="lower",
        cmap=style.ATTRACTANT_CMAP, alpha=style.FIELD_ALPHA, interpolation="bilinear",
    )
    pos_I = out.pos_I_snapshots[-1]
    pos_T = out.pos_T_snapshots[-1]
    if pos_I.size:
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            heights=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
            transOffset=ax.transData, facecolors=style.TCELL,
            edgecolors=style.TCELL_EDGE, linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    if pos_T.size:
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            heights=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
            transOffset=ax.transData, facecolors=style.TUMOR,
            edgecolors=style.TUMOR_EDGE, linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    ax.set_xlim(0, params.L); ax.set_ylim(0, params.L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {params.T_final:.0f}   (N_T = {out.n_T[-1]}, N_I = {out.n_I[-1]})",
                 color=style.FG)

    # right: N_T(t) trajectory
    ax = axes[1]
    ax.set_facecolor(style.BG)
    ax.plot(times, out.n_T, color=style.TUMOR, linewidth=2.5, label=r"$N_{\rm tumor}(t)$")
    ax.plot(times, out.n_I, color=style.TCELL, linewidth=2.5, label=r"$N_{\rm Tcell}(t)$")
    ax.axhline(0, color=style.MUTED, linewidth=0.7, linestyle=":")
    ax.set_xlabel("simulation time")
    ax.set_ylabel("cell count")
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE)
    ax.set_title("T cells eat a small tumor (no proliferation)", color=style.FG)
    ax.set_ylim(bottom=-2)

    fig.suptitle(
        "Phase 2 sanity: killing rule drives tumor extinction",
        fontsize=style.TITLE_SIZE, color=style.FG, y=0.99,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase2_small_tumor_kill.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG)
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Phase 2 validation:")
    print(" (a) chemotaxis to static source")
    fig_chemotaxis_static()
    print(" (b) small-tumor kill (no proliferation)")
    fig_small_tumor_kill()
    print("done.")
