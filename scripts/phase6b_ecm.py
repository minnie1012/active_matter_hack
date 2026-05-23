"""Phase 6b — ECM density (§3) + MMP degradation (§6) validation.

Runs three conditions on the ECM-extended simulation and produces a single
figure:

  outputs/figures/phase6b_ecm.png

Top row:   N_T(t) trajectories for the three conditions on the same axis.
Bottom:    Final rho_E heatmap for each condition (earthy palette).

Conditions
----------
  baseline       : rho_E_init = 0     (ECM disabled)
  dense ECM      : rho_E_init = 2.0,  s_m = 0    (matrix present, no MMP)
  dense + MMP    : rho_E_init = 2.0,  s_m = 2.0  (tumor digests matrix)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import style
from src.sim_ecm import ECMParams, run_extended_ecm


# Earthy palette for the ECM density heatmaps (dark brown -> tan -> cream).
ECM_CMAP = LinearSegmentedColormap.from_list(
    "ecm_density",
    [
        (0.00, "#1A0F08"),  # near-black brown
        (0.35, "#4A2F1A"),  # dark earth
        (0.70, "#A87344"),  # tan
        (1.00, "#F0D9A8"),  # cream
    ],
)


def run_three_conditions(seed: int = 7) -> dict:
    """Run baseline / dense-ECM / dense-ECM-plus-MMP and return ECMRun dicts."""
    # Shared base. Modest CD8 suppression so the baseline doesn't trivially
    # clear or saturate immediately — that way the ECM matters in the
    # observable window. Cap the tumor pool so the y-axis stays informative.
    common = dict(
        T_final=60.0,
        N_T_initial=50,
        N_I_initial=200,
        N_M_initial=80,
        N_T_max=900,            # tighter cap so saturation is visible
        use_macrophages=True,
        chi_s=5.0,
        # ECM grid resolution / pore parameters
        G=64,
        beta_drag=2.0,          # stronger ECM drag so dense ECM measurably slows motion
        r_0=1.0,
        r_p_star=0.75,          # pore radius threshold ≈ sqrt(1/rho_E) crosses this at rho_E≈1.8
        mmp_open_thresh=0.10,
        # MMP defaults (overridden per condition)
        D_m=3.0,
        lam_m=0.15,
        k_deg=2.0,
        k_rep_ECM=0.005,
    )

    runs: dict = {}

    print("Phase 6b: ECM/MMP validation")
    print("---------------------------------")

    # ---- (1) baseline: no ECM ----
    print("  (1) baseline (no ECM)")
    p1 = ECMParams(**common, rho_E_init=0.0, s_m=0.0)
    t0 = time.perf_counter()
    runs["baseline"] = run_extended_ecm(params=p1, seed=seed, snapshot_every=200)
    print(f"       done in {time.perf_counter() - t0:.1f} s, "
          f"final N_T = {runs['baseline'].n_T[-1]}, "
          f"frac = {runs['baseline'].final_tumor_fraction:.2f}")

    # ---- (2) dense ECM, MMP off ----
    print("  (2) dense ECM (rho_E_init = 2.0, MMP off)")
    p2 = ECMParams(**common, rho_E_init=2.0, s_m=0.0)
    t0 = time.perf_counter()
    runs["dense"] = run_extended_ecm(params=p2, seed=seed, snapshot_every=200)
    print(f"       done in {time.perf_counter() - t0:.1f} s, "
          f"final N_T = {runs['dense'].n_T[-1]}, "
          f"frac = {runs['dense'].final_tumor_fraction:.2f}, "
          f"mean rho_E = {runs['dense'].mean_rho_E[-1]:.2f}")

    # ---- (3) dense ECM + MMP on ----
    print("  (3) dense ECM + MMP (s_m = 2.0)")
    p3 = ECMParams(**common, rho_E_init=2.0, s_m=2.0)
    t0 = time.perf_counter()
    runs["mmp"] = run_extended_ecm(params=p3, seed=seed, snapshot_every=200)
    print(f"       done in {time.perf_counter() - t0:.1f} s, "
          f"final N_T = {runs['mmp'].n_T[-1]}, "
          f"frac = {runs['mmp'].final_tumor_fraction:.2f}, "
          f"mean rho_E = {runs['mmp'].mean_rho_E[-1]:.2f}")

    return runs


def make_figure(runs: dict, out_path: Path):
    style.apply_style()
    fig = plt.figure(figsize=(13.0, 8.0), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.1], hspace=0.32, wspace=0.18,
    )

    # ---- top row: N_T(t) for all three on one axis ----
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(style.BG)
    series = [
        ("baseline", "no ECM (baseline)",       style.TUMOR),
        ("dense",    r"dense ECM  ($\rho_{E,0}=2$, MMP off)",   style.ACCENT),
        ("mmp",      r"dense ECM + MMP  ($s_m=2$)",             style.TCELL),
    ]
    for key, lab, color in series:
        r = runs[key]
        ax_top.plot(r.times, r.n_T, color=color, lw=2.6, label=lab)
    ax_top.set_xlabel("simulation time $t$")
    ax_top.set_ylabel(r"$N_T(t)$  tumor cell count")
    ax_top.set_title(
        "Tumor growth under ECM porosity gating and MMP-driven matrix digestion",
        color=style.FG, fontsize=style.LABEL_SIZE + 1, pad=8,
    )
    ax_top.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper left")
    ax_top.grid(False)

    # ---- bottom row: final rho_E heatmaps ----
    rE_max = 0.01
    for _, run in runs.items():
        if run.rho_E_snapshots:
            rE_max = max(rE_max, float(run.rho_E_snapshots[-1].max()), 0.01)

    titles = [
        ("baseline", r"$\rho_E$ at final $t$  (baseline, $\rho_{E,0}=0$)"),
        ("dense",    r"$\rho_E$  (dense ECM, MMP off)"),
        ("mmp",      r"$\rho_E$  (dense ECM + MMP on)"),
    ]
    last_im = None
    for col, (key, ttl) in enumerate(titles):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(style.BG)
        rE = runs[key].rho_E_snapshots[-1] if runs[key].rho_E_snapshots else None
        L = runs[key].params.L
        if rE is None:
            ax.text(0.5, 0.5, "no data", color=style.FG, ha="center", va="center",
                    transform=ax.transAxes)
        else:
            im = ax.imshow(
                rE, extent=[0, L, 0, L], origin="lower",
                cmap=ECM_CMAP, interpolation="bilinear",
                vmin=0.0, vmax=rE_max,
            )
            last_im = im
            # overlay tumor positions as small red dots for spatial context
            pos_T = runs[key].pos_T_snapshots[-1]
            if len(pos_T):
                ax.scatter(
                    pos_T[:, 0], pos_T[:, 1],
                    s=6, color=style.TUMOR, edgecolors="none", alpha=0.85,
                )
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, color=style.FG, fontsize=style.SMALL_SIZE + 1, pad=4)

    # single shared colorbar to the right of the bottom row
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.08, 0.012, 0.35])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label(r"ECM density  $\rho_E$", color=style.FG, fontsize=style.SMALL_SIZE)
        cb.ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)
        for spine in cb.ax.spines.values():
            spine.set_color(style.MUTED)

    fig.suptitle(
        "Phase 6b — ECM density and MMP degradation reshape tumor escape",
        color=style.FG, fontsize=style.TITLE_SIZE, y=0.99,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def main():
    runs = run_three_conditions(seed=7)
    out_path = ROOT / "outputs" / "figures" / "phase6b_ecm.png"
    make_figure(runs, out_path)
    print("done.")


if __name__ == "__main__":
    main()
