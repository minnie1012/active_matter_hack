"""Phase 6c — cell-cell adhesion (cadherin) demo.

Three-panel comparison of tumor morphology after T = 80 simulation units:
  panel 1: k_adh = 0   (no adhesion, expect fragmented/diffuse tumor)
  panel 2: k_adh = 4   (weak adhesion, cohesive cluster)
  panel 3: k_adh = 12, J_align = 1.0   (strong adhesion + Vicsek alignment,
                                        streaming sheet)

For each run we save:
  * a tumor-position snapshot,
  * a histogram of nearest-neighbor distances,
  * a radial density profile measured around the tumor centroid.

Output: outputs/figures/phase6c_adhesion.png
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

from src import style
from src.sim_adhesion import AdhesionParams, run_adhesion


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _periodic_dist2(p, q, L):
    """Squared minimum-image distance between p and q in a periodic box."""
    dx = p[0] - q[0]; dy = p[1] - q[1]
    half_L = 0.5 * L
    if dx > half_L: dx -= L
    elif dx < -half_L: dx += L
    if dy > half_L: dy -= L
    elif dy < -half_L: dy += L
    return dx * dx + dy * dy


def nearest_neighbor_distances(pos, L):
    """Brute-force minimum-image nearest-neighbor distance for each point."""
    N = pos.shape[0]
    if N < 2:
        return np.zeros(0)
    out = np.empty(N)
    for i in range(N):
        best = np.inf
        for j in range(N):
            if j == i:
                continue
            d2 = _periodic_dist2(pos[i], pos[j], L)
            if d2 < best:
                best = d2
        out[i] = np.sqrt(best)
    return out


def periodic_centroid(pos, L):
    """Circular-mean centroid for a periodic box."""
    if len(pos) == 0:
        return np.array([0.5 * L, 0.5 * L])
    ang_x = pos[:, 0] * (2 * np.pi / L)
    ang_y = pos[:, 1] * (2 * np.pi / L)
    cx = (np.arctan2(np.sin(ang_x).mean(), np.cos(ang_x).mean()) % (2 * np.pi)) * L / (2 * np.pi)
    cy = (np.arctan2(np.sin(ang_y).mean(), np.cos(ang_y).mean()) % (2 * np.pi)) * L / (2 * np.pi)
    return np.array([cx, cy])


def radial_density_profile(pos, L, n_bins=30, r_max=None):
    """Annular cell density around the periodic centroid."""
    if len(pos) == 0:
        return np.zeros(n_bins), np.linspace(0, 1, n_bins + 1)
    if r_max is None:
        r_max = 0.45 * L
    centroid = periodic_centroid(pos, L)
    d = np.array([np.sqrt(_periodic_dist2(p, centroid, L)) for p in pos])
    edges = np.linspace(0, r_max, n_bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    annulus_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = counts / np.maximum(annulus_area, 1e-9)
    return density, edges


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def _make_params(k_adh: float, J_align: float) -> AdhesionParams:
    """Common parameters for all three runs; only adhesion knobs differ."""
    return AdhesionParams(
        T_final=80.0,
        L=80.0,
        # focus on adhesion: turn off macrophages and immune pressure to keep
        # the visual driven by adhesion alone
        use_macrophages=False,
        N_T_initial=80,
        N_I_initial=0,
        chi_s=0.0,
        chi_a=0.0,
        # tumor: a little more motile so the no-adhesion case visibly disperses
        v_T_mean=0.4,
        v_T_cv=0.10,
        D_R_T_mean=0.3,
        D_R_T_cv=0.10,
        sigma_T=1.0,
        k_rep_T=30.0,
        # mechanical / proliferation
        p_div=0.0,           # disable proliferation so the visual is purely mechanical
        P_star=8.0,
        # adhesion knobs (these vary across panels)
        k_adh=k_adh,
        sigma_adh=1.6,
        J_align=J_align,
        apply_adhesion_to_T_only=True,
        # pool size big enough but proliferation is off, so fine
        N_T_max=300,
        tumor_disk_radius=5.0,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def main():
    style.apply_style()

    cfgs = [
        ("(1) no adhesion",         dict(k_adh=0.0,  J_align=0.0)),
        ("(2) weak cadherin",       dict(k_adh=4.0,  J_align=0.0)),
        ("(3) strong cadherin\n    + Vicsek alignment", dict(k_adh=12.0, J_align=1.0)),
    ]

    seed = 7
    runs = []
    for title, kw in cfgs:
        p = _make_params(**kw)
        print(f"running {title.strip()!s}: k_adh={p.k_adh}, J_align={p.J_align} ...", flush=True)
        t0 = time.perf_counter()
        out = run_adhesion(params=p, seed=seed, snapshot_every=400, save_fields=False)
        dt_s = time.perf_counter() - t0
        n_final = out.n_T[-1] if out.n_T else 0
        print(f"  done in {dt_s:.1f} s, N_T={n_final}")
        runs.append((title, p, out))

    # 3 panels wide x 3 rows (snapshot, NN hist, radial density)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 13.0), dpi=style.DPI,
                              gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    fig.patch.set_facecolor(style.BG)

    # consistent histogram x-range
    nn_max = 0.0
    nn_arrays = []
    radial_arrays = []
    L_box = runs[0][1].L
    for _, p, out in runs:
        pos = out.pos_T_snapshots[-1]
        nn = nearest_neighbor_distances(pos, p.L)
        nn_arrays.append(nn)
        if len(nn):
            nn_max = max(nn_max, float(np.percentile(nn, 98)))
        dens, edges = radial_density_profile(pos, p.L)
        radial_arrays.append((dens, edges))

    # global density max for shared y-scale across radial plots
    dens_max = max((d.max() if len(d) else 0.0) for d, _ in radial_arrays)
    if dens_max == 0:
        dens_max = 1.0

    for col, ((title, p, out), nn, (dens, edges)) in enumerate(
        zip(runs, nn_arrays, radial_arrays)
    ):
        # ---- row 0: snapshot ----
        ax = axes[0, col]; ax.set_facecolor(style.BG)
        pos = out.pos_T_snapshots[-1]
        N = len(pos)
        if N:
            ax.add_collection(EllipseCollection(
                widths=np.full(N, p.sigma_T), heights=np.full(N, p.sigma_T),
                angles=np.zeros(N), units="x", offsets=pos,
                transOffset=ax.transData,
                facecolors=style.TUMOR, edgecolors=style.BG,
                linewidths=0.4, alpha=style.PARTICLE_ALPHA,
            ))
        ax.set_xlim(0, p.L); ax.set_ylim(0, p.L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        sub = f"k_adh={p.k_adh:g}, J_align={p.J_align:g}, N_T={N}"
        ax.set_title(f"{title}\n{sub}", color=style.FG,
                     fontsize=style.LABEL_SIZE, pad=8)

        # ---- row 1: NN-distance histogram ----
        ax = axes[1, col]; ax.set_facecolor(style.BG)
        if len(nn):
            ax.hist(nn, bins=24, range=(0, max(nn_max, p.sigma_adh * 1.5)),
                    color=style.TUMOR, alpha=0.75, edgecolor=style.BG)
        ax.axvline(p.sigma_T, color=style.MUTED, ls=":", lw=1.2)
        ax.axvline(p.sigma_adh, color=style.ACCENT, ls="--", lw=1.2)
        if col == 0:
            ax.set_ylabel("count", color=style.FG)
        ax.set_xlabel(r"nearest-neighbor distance  $d_{\mathrm{NN}}$",
                      color=style.FG, fontsize=style.SMALL_SIZE + 1)
        if len(nn):
            med = float(np.median(nn))
            ax.set_title(f"median $d_{{NN}}$ = {med:.2f}",
                         color=style.FG, fontsize=style.SMALL_SIZE + 1, pad=4)
        # legend-style annotation only once
        if col == 0:
            ymax = ax.get_ylim()[1]
            ax.text(p.sigma_T, ymax * 0.95, r" $\sigma_T$",
                    color=style.MUTED, fontsize=style.SMALL_SIZE)
            ax.text(p.sigma_adh, ymax * 0.85, r" $\sigma_{adh}$",
                    color=style.ACCENT, fontsize=style.SMALL_SIZE)

        # ---- row 2: radial density profile ----
        ax = axes[2, col]; ax.set_facecolor(style.BG)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(r_centers, dens, color=style.TUMOR, lw=2.4, marker="o", ms=3.5,
                markeredgecolor=style.BG, markerfacecolor=style.TUMOR)
        ax.set_xlabel(r"radius $r$ from tumor centroid", color=style.FG,
                      fontsize=style.SMALL_SIZE + 1)
        if col == 0:
            ax.set_ylabel("density (cells / area)", color=style.FG)
        ax.set_ylim(0, dens_max * 1.1)

    fig.suptitle(
        "Cadherin adhesion converts a diffuse tumor into a cohesive cluster",
        color=style.FG, fontsize=style.TITLE_SIZE, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase6c_adhesion.png"
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    print(f"wrote {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
