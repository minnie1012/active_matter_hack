"""Phase 4 treatment experiment: chi_s halved at t = T_final/2 in an escape run.

Outputs:
  outputs/figures/treatment_panel.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src import style
from src.sim import SimParams, run, run_with_treatment


def main():
    style.apply_style()
    base = SimParams()

    # pick a point clearly in escape: moderate alpha, decent rho_I — so treatment can
    # rescue. Treat early before the tumor saturates the carrying capacity.
    rho_I = 300
    alpha = 10.0
    alpha_after = 0.0              # full checkpoint inhibitor: chi_s -> 0
    seed = 3

    print(f"running no-treatment escape: rho_I={rho_I}, alpha={alpha}")
    base_run = run(rho_I=rho_I, alpha=alpha, seed=seed, params=base)
    print(f"  final fraction = {base_run.final_tumor_fraction:.2f}, "
          f"n_T_traj last={base_run.n_T[-3:]}")

    # Treat early — when tumor is growing but well below cap.
    t_treat = 20.0
    print(f"running treated escape: alpha {alpha} -> {alpha_after} at t = {t_treat}")
    treat_run = run_with_treatment(
        rho_I=rho_I, alpha=alpha, alpha_after=alpha_after,
        seed=seed, t_treat=t_treat, params=base,
    )
    print(f"  final fraction = {treat_run.final_tumor_fraction:.2f}, "
          f"n_T_traj last={treat_run.n_T[-3:]}")

    # ----- two-panel figure -----
    fig, axes = plt.subplots(1, 2, figsize=style.FIG_TWO_PANEL, dpi=style.DPI,
                              sharey=True)
    fig.patch.set_facecolor(style.BG)
    for ax in axes:
        ax.set_facecolor(style.BG)

    t_b = np.asarray(base_run.times)
    t_t = np.asarray(treat_run.times)

    # left: no treatment
    ax = axes[0]
    ax.plot(t_b, base_run.n_T, color=style.TUMOR, linewidth=2.4, label=r"$N_T$")
    ax.plot(t_b, base_run.n_I, color=style.TCELL, linewidth=2.4, label=r"$N_I$")
    ax.set_xlabel("simulation time t")
    ax.set_ylabel("cell count")
    ax.set_title(
        f"No treatment  (α = {alpha:.1f})", color=style.FG, fontsize=style.LABEL_SIZE,
    )
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper left")

    # right: with treatment
    ax = axes[1]
    ax.plot(t_t, treat_run.n_T, color=style.TUMOR, linewidth=2.4, label=r"$N_T$")
    ax.plot(t_t, treat_run.n_I, color=style.TCELL, linewidth=2.4, label=r"$N_I$")
    ax.axvline(t_treat, color=style.ACCENT, linewidth=1.6, linestyle="--", alpha=0.9)
    ymax = max(np.max(treat_run.n_T), np.max(treat_run.n_I), np.max(base_run.n_T))
    ax.text(
        t_treat, 0.95 * ymax,
        f"  treatment: α → {alpha_after:.1f}",
        color=style.ACCENT, fontsize=style.SMALL_SIZE,
        verticalalignment="top",
    )
    ax.set_xlabel("simulation time t")
    ax.set_title(
        f"With treatment at t = {t_treat:.0f}",
        color=style.FG, fontsize=style.LABEL_SIZE,
    )
    ax.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper left")

    fig.suptitle(
        "Simulated checkpoint inhibitor rescues an escape-phase tumor",
        fontsize=style.TITLE_SIZE, color=style.FG, y=1.02,
    )
    fig.tight_layout()
    out_path = ROOT / "outputs" / "figures" / "treatment_panel.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)

    # also save trajectories for the video
    np.savez(
        ROOT / "outputs" / "data" / "treatment_runs.npz",
        no_treat_nT=np.asarray(base_run.n_T),
        no_treat_nI=np.asarray(base_run.n_I),
        no_treat_times=np.asarray(base_run.times),
        treat_nT=np.asarray(treat_run.n_T),
        treat_nI=np.asarray(treat_run.n_I),
        treat_times=np.asarray(treat_run.times),
        rho_I=rho_I, alpha=alpha, alpha_after=alpha_after, t_treat=t_treat,
    )


if __name__ == "__main__":
    main()
