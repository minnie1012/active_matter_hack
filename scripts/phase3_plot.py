"""Phase 3 visualization: phase diagram + trajectory inset showing dormancy.

Reads outputs/data/phase_grid.npz, writes outputs/figures/phase_diagram.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import EllipseCollection

from src import style
from src.sim import SimParams, run


def find_dormancy_run(data, exclude_idx=None) -> tuple:
    """Scan all (rho_I, alpha, seed) trajectories for a dormancy-like one.

    Dormancy = tumor lingers below 300 for an extended interval. We prefer
    trajectories that show "delayed escape" (dormancy then ramp up to cap)
    because they are the boundary phase that the spec calls control.
    """
    keys = data["traj_keys"]
    nT = data["traj_nT"]
    nI = data["traj_nI"]
    times = data["traj_t"]
    best = None
    best_score = -1.0
    for ((i, j, k), nT_arr, nI_arr, t_arr) in zip(keys, nT, nI, times):
        if exclude_idx is not None and (int(i), int(j)) == tuple(exclude_idx):
            continue
        nT_arr = np.asarray(nT_arr)
        t_arr_np = np.asarray(t_arr)
        if nT_arr.size < 5:
            continue
        # score = time spent in band [10, 300] weighted toward early dormancy
        in_band = (nT_arr >= 10) & (nT_arr <= 300)
        if in_band.sum() < 3:
            continue
        # bonus if subsequently escapes (ends > 500): that's the "control
        # transitions to escape" story
        ends_high = nT_arr[-1] > 500
        score = float(in_band.sum()) + (5.0 if ends_high else 0.0)
        if score > best_score:
            best_score = score
            best = (int(i), int(j), int(k), nT_arr, np.asarray(nI_arr), t_arr_np)
    return best, best_score


def render(out_path: Path):
    style.apply_style()
    data = np.load(ROOT / "outputs" / "data" / "phase_grid.npz", allow_pickle=True)
    grid = data["grid"]
    mean = data["mean"]
    rho_vals = data["rho_I_values"]
    alpha_vals = data["alpha_values"]
    n_rho, n_alpha = mean.shape

    # locate three trajectories: clear, dormancy/control, escape
    keys = data["traj_keys"]
    nT_all = data["traj_nT"]
    nI_all = data["traj_nI"]
    t_all = data["traj_t"]

    # clearance: lowest final fraction, prefer the highest rho_I+lowest alpha cell
    clear_idx = np.unravel_index(np.argmin(mean), mean.shape)
    # if there are multiple minima, pick the one with largest rho_I (most interior)
    clear_candidates = list(zip(*np.where(mean == mean.min())))
    if clear_candidates:
        clear_idx = max(clear_candidates, key=lambda ij: rho_vals[ij[0]] - alpha_vals[ij[1]])
    clear_traj_idx = np.where((keys[:, 0] == clear_idx[0]) & (keys[:, 1] == clear_idx[1]) & (keys[:, 2] == 0))[0][0]

    # escape: highest fraction at low rho_I (deepest in escape phase)
    esc_idx = (0, n_alpha - 1)
    esc_traj_idx = np.where((keys[:, 0] == esc_idx[0]) & (keys[:, 1] == esc_idx[1]) & (keys[:, 2] == 0))[0][0]

    # control: longest dormancy trajectory; constrain to be different from clearance idx
    best, score = find_dormancy_run(data, exclude_idx=(clear_idx[0], clear_idx[1]))
    if best is None:
        # last-resort: pick any "boundary" cell where seeds split between clearance and escape
        seed_var = grid.std(axis=2)
        v = seed_var.copy()
        v[clear_idx[0], clear_idx[1]] = -1
        ctrl_idx = np.unravel_index(np.argmax(v), v.shape)
        ctrl_seed = int(np.argmax(grid[ctrl_idx[0], ctrl_idx[1]] < 5))
        ctrl_traj_idx = np.where((keys[:, 0] == ctrl_idx[0]) & (keys[:, 1] == ctrl_idx[1]) & (keys[:, 2] == ctrl_seed))[0][0]
    else:
        ctrl_idx = (best[0], best[1])
        ctrl_seed = best[2]
        ctrl_traj_idx = np.where((keys[:, 0] == ctrl_idx[0]) & (keys[:, 1] == ctrl_idx[1]) & (keys[:, 2] == ctrl_seed))[0][0]

    print(f"clearance: rho_I={rho_vals[clear_idx[0]]}, alpha={alpha_vals[clear_idx[1]]:.2f}")
    print(f"control:   rho_I={rho_vals[ctrl_idx[0]]}, alpha={alpha_vals[ctrl_idx[1]]:.2f}  (dormancy score = {score})")
    print(f"escape:    rho_I={rho_vals[esc_idx[0]]}, alpha={alpha_vals[esc_idx[1]]:.2f}")

    # re-run thumbnails (single seed, just final snapshot)
    base = SimParams()
    thumbs = {}
    for name, (i, j) in [("clear", clear_idx), ("ctrl", ctrl_idx), ("esc", esc_idx)]:
        seed = ctrl_seed + 1 if name == "ctrl" else 1
        out = run(rho_I=int(rho_vals[i]), alpha=float(alpha_vals[j]), seed=seed,
                  params=base, snapshot_every=base.n_steps - 1)
        thumbs[name] = {"pos_T": out.pos_T_snapshots[-1], "pos_I": out.pos_I_snapshots[-1]}

    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=(15.0, 8.0), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)

    # main heatmap (left half)
    main_ax = fig.add_axes([0.06, 0.13, 0.40, 0.76])
    main_ax.set_facecolor(style.BG)
    safe = np.clip(mean, 1e-2, 1e2)
    im = main_ax.pcolormesh(
        alpha_vals, rho_vals, safe,
        cmap=style.PHASE_CMAP, norm=LogNorm(vmin=1e-2, vmax=1e2),
        shading="auto",
    )
    main_ax.set_xlabel(r"Immunosuppression strength  $\alpha$  ($\chi_s$)")
    main_ax.set_ylabel(r"Initial T-cell count  $\rho_I$")
    main_ax.set_yscale("log")
    main_ax.set_title("Tumor–immune phase diagram", fontsize=style.TITLE_SIZE, pad=14)

    # colorbar attached to main_ax
    cax = fig.add_axes([0.47, 0.13, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Final tumor fraction  $N_T(T_f) / N_T(0)$",
                 fontsize=style.SMALL_SIZE, labelpad=8)
    cb.ax.tick_params(colors=style.FG, labelsize=style.SMALL_SIZE - 1)
    cb.outline.set_edgecolor(style.MUTED)

    # markers + phase labels
    for (i, j), label, marker_color, txt_color, xy_offset in [
        (clear_idx, "Clearance", style.FG, style.FG, (10, 10)),
        (ctrl_idx,  "Control",   style.ACCENT, style.ACCENT, (10, -16)),
        (esc_idx,   "Escape",    style.TUMOR, style.TUMOR, (-72, 8)),
    ]:
        main_ax.plot(alpha_vals[j], rho_vals[i], "o", markersize=10,
                     markerfacecolor=marker_color, markeredgecolor=style.BG,
                     markeredgewidth=1.5, zorder=10)
        main_ax.annotate(
            label, (alpha_vals[j], rho_vals[i]),
            textcoords="offset points", xytext=xy_offset,
            fontsize=style.SMALL_SIZE, color=txt_color, weight="bold",
            bbox=dict(facecolor=style.BG, edgecolor=marker_color, alpha=0.8,
                      boxstyle="round,pad=0.3"),
        )

    # ---- thumbnails column (middle) ----
    L = base.L
    thumb_cfg = [
        ("clear",  0.54, 0.66, f"Clearance  ρ={rho_vals[clear_idx[0]]}, α={alpha_vals[clear_idx[1]]:.1f}", style.FG),
        ("ctrl",   0.54, 0.385, f"Control (dormancy)  ρ={rho_vals[ctrl_idx[0]]}, α={alpha_vals[ctrl_idx[1]]:.1f}", style.ACCENT),
        ("esc",    0.54, 0.11, f"Escape  ρ={rho_vals[esc_idx[0]]}, α={alpha_vals[esc_idx[1]]:.1f}", style.TUMOR),
    ]
    for name, x, y, label, color in thumb_cfg:
        ax_t = fig.add_axes([x, y, 0.16, 0.235])
        ax_t.set_facecolor(style.BG)
        for s in ax_t.spines.values():
            s.set_edgecolor(color); s.set_linewidth(1.4)
        pos_T = thumbs[name]["pos_T"]; pos_I = thumbs[name]["pos_I"]
        if len(pos_I):
            ax_t.add_collection(EllipseCollection(
                widths=np.full(len(pos_I), 1.3), heights=np.full(len(pos_I), 1.3),
                angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
                transOffset=ax_t.transData, facecolors=style.TCELL,
                edgecolors=style.TCELL_EDGE, linewidths=0.2, alpha=0.85,
            ))
        if len(pos_T):
            ax_t.add_collection(EllipseCollection(
                widths=np.full(len(pos_T), 1.3), heights=np.full(len(pos_T), 1.3),
                angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
                transOffset=ax_t.transData, facecolors=style.TUMOR,
                edgecolors=style.TUMOR_EDGE, linewidths=0.2, alpha=0.9,
            ))
        ax_t.set_xlim(0, L); ax_t.set_ylim(0, L); ax_t.set_aspect("equal")
        ax_t.set_xticks([]); ax_t.set_yticks([])
        ax_t.set_title(label, color=color, fontsize=style.SMALL_SIZE - 1, pad=3, weight="bold")

    # ---- trajectory inset (right) ----
    traj_ax = fig.add_axes([0.78, 0.13, 0.19, 0.76])
    traj_ax.set_facecolor(style.BG)
    for s in traj_ax.spines.values():
        s.set_edgecolor(style.MUTED)
    for traj_idx, color, lab, lw in [
        (clear_traj_idx, style.FG,      "Clearance", 2.0),
        (ctrl_traj_idx,  style.ACCENT,  "Control",   2.4),
        (esc_traj_idx,   style.TUMOR,   "Escape",    2.0),
    ]:
        traj_ax.plot(t_all[traj_idx], nT_all[traj_idx],
                     color=color, linewidth=lw, label=lab)
    traj_ax.set_xlabel("t", fontsize=style.LABEL_SIZE)
    traj_ax.set_ylabel(r"$N_T(t)$  tumor count", fontsize=style.LABEL_SIZE)
    traj_ax.legend(frameon=False, fontsize=style.SMALL_SIZE, loc="upper left")
    traj_ax.tick_params(labelsize=style.SMALL_SIZE)
    traj_ax.set_title("Order parameter vs. time",
                      color=style.FG, fontsize=style.LABEL_SIZE, pad=8, weight="bold")

    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "outputs" / "figures" / "phase_diagram.png"
    render(out)
