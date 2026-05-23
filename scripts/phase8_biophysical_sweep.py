"""Phase 8 — Biophysical phase diagram sweep.

Sweeps ``J_fiber`` x ``k_hypoxia_EMT`` (5 x 4 grid, 3 seeds) over a short
``T_final=40`` simulation and classifies each cell into one of six
biological phases via metric-based heuristics.

Output:
  outputs/figures/phase8_phase_diagram.png  — color-coded heatmap
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.sim_biophysical import BiophysicalParams, run_biophysical


# ---------------------------------------------------------------------------
# Phases + colours
# ---------------------------------------------------------------------------

PHASE_LIST = [
    "clearance",
    "dormancy",
    "immune_escape",
    "matrix_jammed",
    "collective_invasion",
    "single_cell_invasion",
    "mixed",
]
PHASE_COLOR = {
    "clearance":            "#7FB3D5",   # cool blue
    "dormancy":             "#A8D5BA",   # pale green
    "immune_escape":        "#E85D4A",   # warm red
    "matrix_jammed":        "#8C6E3E",   # collagen brown
    "collective_invasion":  "#E8769A",   # tumor pink
    "single_cell_invasion": "#5B4F8A",   # EMT purple
    "mixed":                "#C7C0B6",   # neutral grey
}


def classify(out, N_T0: int) -> str:
    """Apply the heuristic phase rules in priority order."""
    final_N_T = int(out.n_T[-1]) if out.n_T else 0
    mean_emt = float(out.mean_EMT[-1]) if out.mean_EMT else 0.0
    frac_mes = float(out.frac_mes[-1]) if out.frac_mes else 0.0
    max_inv = float(out.max_invasion_distance[-1]) if out.max_invasion_distance else 0.0
    n_det = int(out.n_detached[-1]) if out.n_detached else 0
    ecm_deg = float(out.ecm_degraded_area)

    if final_N_T < 0.5 * N_T0:
        return "clearance"
    if (0.5 * N_T0 <= final_N_T <= 2 * N_T0) and max_inv < 25:
        return "dormancy"
    if final_N_T > 5 * N_T0 and mean_emt < 0.3 and max_inv < 30:
        return "immune_escape"
    if mean_emt < 0.3 and max_inv < 15 and ecm_deg < 0.05:
        return "matrix_jammed"
    if frac_mes < 0.4 and max_inv > 25:
        return "collective_invasion"
    if frac_mes > 0.5 and n_det > 10:
        return "single_cell_invasion"
    return "mixed"


def main():
    out_figs = ROOT / "outputs" / "figures"
    out_figs.mkdir(parents=True, exist_ok=True)

    J_fiber_vals = [0.0, 0.5, 1.5, 3.0, 6.0]
    k_hyp_vals = [0.0, 0.3, 0.6, 1.2]
    seeds = [0, 1, 2]
    base_T_final = 40.0

    n_x = len(J_fiber_vals)
    n_y = len(k_hyp_vals)

    # store the modal classification across seeds in each cell
    classifications = np.full((n_y, n_x), "mixed", dtype=object)

    for iy, k_hyp in enumerate(k_hyp_vals):
        for ix, J in enumerate(J_fiber_vals):
            phases_here = []
            for seed in seeds:
                params = BiophysicalParams(
                    T_final=base_T_final,
                    N_T_initial=50,
                    N_I_initial=200,
                    N_M_initial=80,
                    rho_E_init=0.8,
                    s_m=1.0,
                    k_deg=1.0,
                    p_div=0.005,
                    p_kill=0.04,
                    p_phag=0.04,
                    chi_s=10.0,
                    J_fiber=float(J),
                    k_hypoxia_EMT=float(k_hyp),
                )
                t0 = time.perf_counter()
                out = run_biophysical(params=params, seed=seed,
                                      snapshot_every=50, save_fields=False)
                dt = time.perf_counter() - t0
                phase = classify(out, N_T0=params.N_T_initial)
                phases_here.append(phase)
                print(f"[sweep] J={J:>4.1f} k_hyp={k_hyp:>4.1f} seed={seed}  "
                      f"phase={phase:<22}  n_T={out.n_T[-1]:>3d}  "
                      f"mEMT={out.mean_EMT[-1]:.2f}  "
                      f"inv={out.max_invasion_distance[-1]:.1f}  "
                      f"({dt:.1f}s)")
            # modal phase (ties → first occurrence)
            vals, counts = np.unique(phases_here, return_counts=True)
            classifications[iy, ix] = str(vals[np.argmax(counts)])

    # ---- plot the heatmap ----
    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=140)
    fig.patch.set_facecolor("#FBF6EF")
    ax.set_facecolor("#FBF6EF")
    for iy in range(n_y):
        for ix in range(n_x):
            phase = classifications[iy, ix]
            ax.add_patch(Rectangle(
                (ix - 0.5, iy - 0.5), 1.0, 1.0,
                facecolor=PHASE_COLOR.get(phase, "#C7C0B6"),
                edgecolor="#D9D2C7", linewidth=1.0,
            ))
            ax.text(ix, iy, phase.replace("_", "\n"),
                    ha="center", va="center", fontsize=7,
                    color="#2B2A2A")
    ax.set_xlim(-0.5, n_x - 0.5); ax.set_ylim(-0.5, n_y - 0.5)
    ax.set_xticks(range(n_x))
    ax.set_xticklabels([f"{v}" for v in J_fiber_vals])
    ax.set_yticks(range(n_y))
    ax.set_yticklabels([f"{v}" for v in k_hyp_vals])
    ax.set_xlabel("J_fiber  (contact-guidance strength)")
    ax.set_ylabel("k_hypoxia_EMT  (hypoxia → EMT rate)")
    ax.set_title("Biophysical phase diagram  (3 seeds, T_final=40)",
                 fontsize=11, weight="bold")
    for s in ax.spines.values():
        s.set_edgecolor("#D9D2C7")
    plt.tight_layout()

    out_path = out_figs / "phase8_phase_diagram.png"
    fig.savefig(out_path, dpi=140, facecolor="#FBF6EF")
    plt.close(fig)
    print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
