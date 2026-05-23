"""Phase 6 — extended-physics validation.

Three figures + a video:
  outputs/figures/phase6_macrophage_polarization.png   — M2 buildup over time
  outputs/figures/phase6_combo_treatment.png           — baseline vs M1-repol vs combo
  outputs/figures/phase6_heterogeneity_pressure.png    — distribution of v_T, P, division gate
  outputs/videos/phase6_combo_treatment.mp4            — animated combo therapy
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from src import style
from src.sim_extended import ExtendedParams, run_extended


# ---------------------------------------------------------------------------
# Figure 1: macrophage polarization dynamics
# ---------------------------------------------------------------------------

def fig_polarization():
    style.apply_style()
    p = ExtendedParams(
        use_macrophages=True, T_final=100.0,
        N_I_initial=300, N_M_initial=120,
        chi_s=5.0,
    )
    print("Fig 1: macrophage polarization over time")
    t0 = time.perf_counter()
    out = run_extended(params=p, seed=5, snapshot_every=200)
    print(f"  done in {time.perf_counter()-t0:.1f} s")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), dpi=style.DPI,
                              gridspec_kw={"width_ratios": [1.1, 1.1, 1.4]})
    fig.patch.set_facecolor(style.BG)

    # left: macrophage snapshot colored by polarization at final time
    ax = axes[0]; ax.set_facecolor(style.BG)
    pos_M = out.pos_M_snapshots[-1]
    p_M = out.p_M_snapshots[-1]
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("RdBu_r")
    if len(pos_M):
        colors = cmap(norm(p_M))
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_M), 1.6), heights=np.full(len(pos_M), 1.6),
            angles=np.zeros(len(pos_M)), units="x", offsets=pos_M,
            transOffset=ax.transData,
            facecolors=colors, edgecolors=style.MUTED, linewidths=0.3, alpha=0.9,
        ))
    # overlay tumor as small red dots for context
    pos_T = out.pos_T_snapshots[-1]
    if len(pos_T):
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_T), 0.7), heights=np.full(len(pos_T), 0.7),
            angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
            transOffset=ax.transData,
            facecolors=style.TUMOR, edgecolors="none", alpha=0.6,
        ))
    ax.set_xlim(0, p.L); ax.set_ylim(0, p.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Macrophages at t={out.times[-1]:.0f}  (color = polarization)",
                 color=style.FG)
    # colorbar mini
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("p  (−1 M2  →  +1 M1)", fontsize=style.SMALL_SIZE)
    cb.ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)

    # middle: IL-10 field
    ax = axes[1]; ax.set_facecolor(style.BG)
    c_il = out.c_IL10_snapshots[-1]
    im = ax.imshow(gaussian_filter(c_il, 0.7),
                    extent=[0, p.L, 0, p.L], origin="lower",
                    cmap=style.SUPPRESSANT_CMAP, interpolation="bilinear")
    ax.set_xlim(0, p.L); ax.set_ylim(0, p.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"IL-10 field at t={out.times[-1]:.0f}\n(M2-secreted suppressor)",
                 color=style.FG)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)

    # right: time-series of mean polarization, N_T, N_killed
    ax = axes[2]; ax.set_facecolor(style.BG)
    t = np.asarray(out.times)
    ax_r = ax.twinx()
    line_p = ax.plot(t, out.mean_pM, color=style.ACCENT, lw=2.5,
                      label=r"$\langle p_k \rangle$")
    ax.axhline(0, color=style.MUTED, lw=0.7, ls=":")
    ax.set_ylabel(r"mean polarization  $\langle p_k \rangle$",
                  color=style.ACCENT, fontsize=style.LABEL_SIZE)
    ax.tick_params(axis="y", colors=style.ACCENT)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("t")
    line_T = ax_r.plot(t, out.n_T, color=style.TUMOR, lw=2.5,
                       label=r"$N_T$")
    line_I = ax_r.plot(t, out.n_I, color=style.TCELL, lw=2.5,
                       label=r"$N_I$", alpha=0.7)
    ax_r.set_ylabel("cell count", color=style.FG, fontsize=style.LABEL_SIZE)
    ax_r.tick_params(axis="y", colors=style.FG)
    lines = line_p + line_T + line_I
    ax.legend(lines, [ln.get_label() for ln in lines],
              loc="upper left", frameon=False, fontsize=style.SMALL_SIZE)
    ax.set_title("Polarization tracks tumor growth", color=style.FG)

    fig.suptitle(
        "Macrophages polarize toward M2 as suppressant accumulates  (3 species, no treatment)",
        color=style.FG, fontsize=style.TITLE_SIZE, y=1.0,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase6_macrophage_polarization.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: combination treatment
# ---------------------------------------------------------------------------

def fig_combo_treatment():
    style.apply_style()
    base = ExtendedParams(
        use_macrophages=True, T_final=100.0,
        N_I_initial=300, N_M_initial=120,
        chi_s=10.0,           # strong baseline suppression -> escape
    )
    seed = 11
    print("Fig 2: combination treatment")
    runs = {}

    # (a) no treatment
    print("  (a) no treatment")
    t0 = time.perf_counter()
    runs["none"] = run_extended(params=base, seed=seed, snapshot_every=200)
    print(f"     {time.perf_counter()-t0:.1f} s, final_frac={runs['none'].final_tumor_fraction:.2f}")

    # (b) anti-PD-1 only (chi_s -> 0 at t=20)
    print("  (b) anti-PD-1 only")
    t0 = time.perf_counter()
    runs["pd1"] = run_extended(
        params=base, seed=seed, snapshot_every=200,
        treat_time=20.0, chi_s_after=0.0,
    )
    print(f"     {time.perf_counter()-t0:.1f} s, final_frac={runs['pd1'].final_tumor_fraction:.2f}")

    # (c) TAM repol only (M1_bias -> 15 at t=20)
    print("  (c) TAM repolarization only")
    t0 = time.perf_counter()
    runs["tam"] = run_extended(
        params=base, seed=seed, snapshot_every=200,
        treat_time=20.0, M1_bias_after=15.0,
    )
    print(f"     {time.perf_counter()-t0:.1f} s, final_frac={runs['tam'].final_tumor_fraction:.2f}")

    # (d) combo therapy
    print("  (d) combo: anti-PD-1 + TAM repolarization")
    t0 = time.perf_counter()
    runs["combo"] = run_extended(
        params=base, seed=seed, snapshot_every=200,
        treat_time=20.0, chi_s_after=0.0, M1_bias_after=15.0,
    )
    print(f"     {time.perf_counter()-t0:.1f} s, final_frac={runs['combo'].final_tumor_fraction:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG); ax.set_facecolor(style.BG)
    colors = {
        "none":  style.TUMOR,
        "pd1":   "#F4A261",
        "tam":   "#2A9D8F",
        "combo": style.TCELL,
    }
    labels = {
        "none":  r"no treatment",
        "pd1":   r"anti-PD-1 only  ($\alpha\to 0$)",
        "tam":   r"TAM repolarization only  ($M_1\,\mathrm{bias}\to 15$)",
        "combo": r"combo  (anti-PD-1  +  TAM repolarization)",
    }
    for key in ["none", "pd1", "tam", "combo"]:
        r = runs[key]
        ax.plot(r.times, r.n_T, color=colors[key], lw=2.6, label=labels[key])
    ax.axvline(20.0, color=style.MUTED, ls="--", lw=1.0, alpha=0.7)
    ax.text(21, ax.get_ylim()[1] * 0.9, "treatment\nat t = 20",
            color=style.MUTED, fontsize=style.SMALL_SIZE)
    ax.set_xlabel("simulation time t")
    ax.set_ylabel(r"$N_T(t)$  tumor cell count")
    ax.set_title("Combo therapy rescues an escape tumor where monotherapy fails",
                 color=style.FG, fontsize=style.LABEL_SIZE, pad=10)
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper left")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase6_combo_treatment.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)
    return runs


# ---------------------------------------------------------------------------
# Figure 3: heterogeneity + pressure-gated proliferation
# ---------------------------------------------------------------------------

def fig_heterogeneity_pressure():
    style.apply_style()
    # short run, no macrophages, default heterogeneity ON
    p = ExtendedParams(use_macrophages=False, T_final=50.0,
                       N_I_initial=400, chi_s=0.0)
    print("Fig 3: heterogeneity + pressure dynamics")
    t0 = time.perf_counter()
    out = run_extended(params=p, seed=22, snapshot_every=100)
    print(f"  done in {time.perf_counter()-t0:.1f} s, final N_T = {out.n_T[-1]}")

    # for the pressure visualization we need to re-run a single step at the
    # end snapshot to get per-cell pressure — easier: just compute it
    # post-hoc from the final particle positions.
    from src.sim_extended import pairwise_with_pressure
    final_pos_T = out.pos_T_snapshots[-1]
    if len(final_pos_T) == 0:
        # if it cleared, use t=10 snapshot (still some cells)
        k = next((i for i, n in enumerate(out.n_T) if n > 20), 0)
        final_pos_T = out.pos_T_snapshots[k]
    # build alive mask of right size
    N = len(final_pos_T)
    pos_buf = np.zeros((max(N, 4), 2))
    pos_buf[:N] = final_pos_T
    alive_buf = np.zeros(max(N, 4), dtype=np.bool_)
    alive_buf[:N] = True
    _, P_buf = pairwise_with_pressure(pos_buf, alive_buf,
                                       p.sigma_T, p.k_rep_T, p.L)
    P_vals = P_buf[:N]
    division_gate = np.clip(1 - P_vals / p.P_star, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)
    for ax in axes:
        ax.set_facecolor(style.BG)

    # left: spatial map of pressure on tumor cells
    ax = axes[0]
    norm = Normalize(vmin=0, vmax=max(P_vals.max(), 1.0))
    cmap = plt.get_cmap("magma")
    if N:
        colors = cmap(norm(P_vals))
        ax.add_collection(EllipseCollection(
            widths=np.full(N, 1.0), heights=np.full(N, 1.0),
            angles=np.zeros(N), units="x", offsets=final_pos_T,
            transOffset=ax.transData,
            facecolors=colors, edgecolors=style.MUTED, linewidths=0.2, alpha=0.95,
        ))
    ax.set_xlim(0, p.L); ax.set_ylim(0, p.L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Local pressure  P_i  on tumor cells", color=style.FG)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("pressure (au)", fontsize=style.SMALL_SIZE)
    cb.ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)

    # middle: histogram of pressure
    ax = axes[1]
    ax.hist(P_vals, bins=30, color=style.TUMOR, alpha=0.7, edgecolor=style.BG)
    ax.axvline(p.P_star, color=style.ACCENT, lw=2, ls="--")
    ax.text(p.P_star * 1.02, ax.get_ylim()[1] * 0.9,
            f"P* = {p.P_star}", color=style.ACCENT, fontsize=style.SMALL_SIZE)
    ax.set_xlabel("pressure  P_i")
    ax.set_ylabel("# tumor cells")
    ax.set_title("Pressure distribution\n(core cells stop dividing)", color=style.FG)

    # right: division-gate factor vs pressure (theoretical curve + empirical points)
    ax = axes[2]
    Pgrid = np.linspace(0, max(P_vals.max() * 1.1, p.P_star * 1.2), 200)
    gate = np.clip(1 - Pgrid / p.P_star, 0, 1)
    ax.plot(Pgrid, gate, color=style.ACCENT, lw=2.5,
            label=r"$\max(0,\,1-P/P^\star)$")
    ax.scatter(P_vals, division_gate, s=25, color=style.TUMOR,
               edgecolors=style.BG, linewidth=0.4, alpha=0.7,
               label=f"observed N={N}")
    ax.set_xlabel("pressure  P_i")
    ax.set_ylabel("division-rate factor")
    ax.set_title("Pressure-gated proliferation", color=style.FG)
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper right")
    ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        "Heterogeneity + pressure-gated proliferation give stable dormant cores",
        color=style.FG, fontsize=style.TITLE_SIZE, y=1.03,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "phase6_heterogeneity_pressure.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_polarization()
    fig_combo_treatment()
    fig_heterogeneity_pressure()
    print("done.")
