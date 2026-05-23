"""Build (and execute) the four shareable notebooks in notebooks/.

Each notebook re-imports from src/, so it stays in sync with the simulation
code. Cells with simulations run at modest sizes so a teammate opening the
notebook fresh sees real plots within a minute; the big sweep figure is
loaded from the cached .npz.

Run:
    python scripts/build_notebooks.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat as nbf
from nbclient import NotebookClient


NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


# ===========================================================================
# 01 — single-species validation
# ===========================================================================

def nb01_single_species():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    nb.cells = [
        md(
            "# Phase 1 — Single-species ABP validation\n\n"
            "**Project:** Tumor–Immune Active Matter (UCSD Vibe Hackathon, 2026-05-23).\n\n"
            "This notebook sanity-checks the core particle dynamics in isolation, before we add the second\n"
            "species, the reaction–diffusion fields, or the killing rule. Two checks:\n\n"
            "1. **Motility-induced clustering** at high Péclet number — confirms the active Brownian particle\n"
            "   (ABP) loop is propelling and interacting correctly.\n"
            "2. **Fisher–KPP front** from a small seed — confirms proliferation + density gating produce a\n"
            "   propagating tumor mass with a visible advancing edge.\n\n"
            "All code lives under `src/`. This notebook only orchestrates."
        ),
        code(
            "import sys, pathlib\n"
            "# robust root resolution: walk up until we find src/\n"
            "ROOT = pathlib.Path().resolve()\n"
            "while not (ROOT / 'src').is_dir() and ROOT.parent != ROOT:\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
            "\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.collections import EllipseCollection\n"
            "\n"
            "from src import style\n"
            "from src.sim import TumorParams, run_single_species, step_single_species, init_tumor_uniform\n"
            "\n"
            "style.apply_style()\n"
            "print('imports OK')"
        ),
        md(
            "## 1.1  Motility-induced clustering\n\n"
            "Place 400 ABPs uniformly in a 35×35 box at packing fraction ~0.25. Run with elevated speed\n"
            "(`v = 1.0` vs. the tumor default 0.1) and modest rotational diffusion so the persistence length\n"
            "exceeds the cell diameter. Repulsive collisions trap fast-movers and you should see clusters /\n"
            "voids by t ≈ 30."
        ),
        code(
            "params = TumorParams(L=35.0, v=1.0, D_R=0.05, D_T=0.005,\n"
            "                     sigma=1.0, k_rep=80.0, p_div=0.0, N_max=600)\n"
            "out = run_single_species(params, n_initial=400, n_steps=3000,\n"
            "                          init='uniform', snapshot_every=3000,\n"
            "                          seed=11, enable_proliferation=False)\n"
            "print(f'snapshots: {len(out.pos_snapshots)}; final N={out.pos_snapshots[-1].shape[0]}')"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG)\n"
            "for ax, xy, title in zip(axes, [out.pos_snapshots[0], out.pos_snapshots[-1]],\n"
            "                         ['t = 0',\n"
            "                          f't = {out.times[-1]:.0f}  (clusters form)']):\n"
            "    ax.set_facecolor(style.BG)\n"
            "    ax.add_collection(EllipseCollection(\n"
            "        widths=np.full(len(xy), 1.0), heights=np.full(len(xy), 1.0),\n"
            "        angles=np.zeros(len(xy)), units='x', offsets=xy,\n"
            "        transOffset=ax.transData,\n"
            "        facecolors=style.TUMOR, edgecolors=style.TUMOR_EDGE,\n"
            "        linewidths=0.3, alpha=style.PARTICLE_ALPHA,\n"
            "    ))\n"
            "    ax.set_xlim(0, params.L); ax.set_ylim(0, params.L)\n"
            "    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])\n"
            "    ax.set_title(title, color=style.FG)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure:** at t = 0 the particles are uniformly random. At late time, even though\n"
            "every particle is the same speed, you see non-uniform density — a hallmark of motility-induced\n"
            "phase separation (MIPS). It's not a textbook MIPS slab (need a stiffer wall + longer run for\n"
            "that), but it confirms the inner loop is working: particles propel, repel, persist."
        ),
        md(
            "## 1.2  Fisher–KPP growth front\n\n"
            "Now switch proliferation **on** (`p_div = 0.005` per step) and start with a small disk of 50\n"
            "cells at the box center. A Fisher–KPP equation predicts a propagating front whose radial\n"
            "density profile saturates inside the colony and falls off ahead of the front. With local-density\n"
            "gating preventing overpacking, that's what we should see."
        ),
        code(
            "params = TumorParams(L=100.0, v=0.1, D_R=0.1, D_T=0.001,\n"
            "                     sigma=1.0, k_rep=30.0,\n"
            "                     p_div=0.005, nbr_radius=1.5, nbr_threshold=6,\n"
            "                     N_max=4000)\n"
            "out = run_single_species(params, n_initial=50, n_steps=10000,\n"
            "                          init='disk', init_radius=3.0,\n"
            "                          snapshot_every=2000, seed=3)\n"
            "print(f'snapshots: {len(out.pos_snapshots)}, final N = {out.n_alive[-1]}')"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG)\n"
            "\n"
            "center = np.array([params.L/2, params.L/2])\n"
            "\n"
            "# left: final snapshot\n"
            "ax = axes[0]; ax.set_facecolor(style.BG)\n"
            "xy = out.pos_snapshots[-1]\n"
            "ax.add_collection(EllipseCollection(\n"
            "    widths=np.full(len(xy), 1.0), heights=np.full(len(xy), 1.0),\n"
            "    angles=np.zeros(len(xy)), units='x', offsets=xy,\n"
            "    transOffset=ax.transData,\n"
            "    facecolors=style.TUMOR, edgecolors=style.TUMOR_EDGE,\n"
            "    linewidths=0.2, alpha=style.PARTICLE_ALPHA))\n"
            "ax.set_xlim(center[0]-25, center[0]+25); ax.set_ylim(center[1]-25, center[1]+25)\n"
            "ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])\n"
            "ax.set_title(f't = {out.times[-1]:.0f},  N = {len(xy)}', color=style.FG)\n"
            "\n"
            "# right: radial density curves\n"
            "ax = axes[1]; ax.set_facecolor(style.BG)\n"
            "r_edges = np.linspace(0, 25, 25)\n"
            "r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])\n"
            "ring_areas = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)\n"
            "import matplotlib as mpl\n"
            "cmap = plt.get_cmap('magma')\n"
            "for k, (xy, t) in enumerate(zip(out.pos_snapshots, out.times)):\n"
            "    d = np.linalg.norm(xy - center, axis=1)\n"
            "    counts, _ = np.histogram(d, bins=r_edges)\n"
            "    ax.plot(r_centers, counts/ring_areas,\n"
            "             color=cmap(0.15 + 0.75 * k / max(1, len(out.pos_snapshots)-1)),\n"
            "             lw=2, label=f't={t:.0f}')\n"
            "ax.set_xlabel('r from center  (cell diameters)')\n"
            "ax.set_ylabel(r'local density  $\\rho(r)$')\n"
            "ax.set_title('Growing front', color=style.FG)\n"
            "ax.legend(fontsize=8, frameon=False, ncol=2)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure:** the radial density profile (right) shows a flat 'plateau' inside the\n"
            "colony (saturated by the local-density gate) and a falling tail at increasing radius. The edge\n"
            "of the tail moves outward over time — the propagating Fisher–KPP front. The left snapshot is the\n"
            "physical picture of that: a roughly circular colony filling the central region of the box."
        ),
        md(
            "## 1.3  Profiling gate\n\n"
            "Quick benchmark: 500 ABPs × 1000 steps must stay under 5 s post-JIT, or we'd need a cell-list.\n"
            "This is the gate that we passed before moving to Phase 2."
        ),
        code(
            "params = TumorParams(N_max=512)\n"
            "pos, theta, alive = init_tumor_uniform(500, params.L, params.N_max, seed=7)\n"
            "param_arr = TumorParams(**{**params.__dict__, 'p_div': 0.0}).to_array()\n"
            "# warm up\n"
            "for _ in range(3):\n"
            "    step_single_species(pos, theta, alive, param_arr)\n"
            "\n"
            "import time\n"
            "t0 = time.perf_counter()\n"
            "for _ in range(1000):\n"
            "    step_single_species(pos, theta, alive, param_arr)\n"
            "elapsed = time.perf_counter() - t0\n"
            "print(f'500 ABPs x 1000 steps : {elapsed:.2f} s   ({elapsed*1e3/1000:.2f} ms/step)')\n"
            "print(f'GATE (<5 s):  {\"PASS\" if elapsed < 5 else \"FAIL\"}')"
        ),
        md(
            "✅ With this passing, the inner loop is fast enough to run a 192-point sweep on 16 cores in ~10\n"
            "minutes — see `notebooks/03_phase_diagram.ipynb`."
        ),
    ]
    return nb


# ===========================================================================
# 02 — two-species demo
# ===========================================================================

def nb02_two_species():
    nb = nbf.v4.new_notebook()
    nb.metadata = nb01_single_species().metadata
    nb.cells = [
        md(
            "# Phase 2 — Two species + RD fields + killing\n\n"
            "Three checks that each ingredient works in isolation, then a single full-model run.\n\n"
            "* **2.1** T cells chemotax up a *frozen* attractant gradient.\n"
            "* **2.2** Many T cells eat a small tumor (no proliferation).\n"
            "* **2.3** A short full-physics run with both species, fields, killing, and proliferation."
        ),
        code(
            "import sys, pathlib\n"
            "# robust root resolution: walk up until we find src/\n"
            "ROOT = pathlib.Path().resolve()\n"
            "while not (ROOT / 'src').is_dir() and ROOT.parent != ROOT:\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
            "\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.collections import EllipseCollection\n"
            "from scipy.ndimage import gaussian_filter\n"
            "\n"
            "from src import style\n"
            "from src.sim import SimParams, run, init_two_species, _step_particles_two_species\n"
            "from src.fields import static_gaussian_field\n"
            "\n"
            "style.apply_style()\n"
            "print('imports OK')"
        ),
        md(
            "## 2.1  Chemotaxis to a static source\n\n"
            "Pre-bake a Gaussian attractant centered in a 60×60 box and let 200 T cells (no tumor present)\n"
            "evolve under chemotaxis + self-propulsion + repulsion. The attractant doesn't get updated, so\n"
            "any aggregation must be driven by the chemotactic gradient alone."
        ),
        code(
            "p = SimParams(L=60.0, G=64, T_final=80.0,\n"
            "              v_I=0.5, D_R_I=0.3,\n"
            "              chi_a=20.0, chi_s=0.0,\n"
            "              N_T_initial=0, N_I_initial=200,\n"
            "              N_T_max=8, N_I_max=256, p_div=0.0)\n"
            "np.random.seed(0)\n"
            "state = init_two_species(p, seed=0)\n"
            "pos_T, theta_T, alive_T = state['pos_T'], state['theta_T'], state['alive_T']\n"
            "pos_I, theta_I, alive_I = state['pos_I'], state['theta_I'], state['alive_I']\n"
            "c_a = static_gaussian_field(p.L, p.G, p.L/2, p.L/2, sig=5.0, amp=20.0)\n"
            "c_s = np.zeros((p.G, p.G))\n"
            "initial_xy = pos_I[alive_I].copy()\n"
            "for _ in range(p.n_steps):\n"
            "    _step_particles_two_species(\n"
            "        pos_T, theta_T, alive_T, pos_I, theta_I, alive_I, c_a, c_s,\n"
            "        p.dt, p.v_T, p.v_I, p.D_R_T, p.D_R_I, p.D_T_T, p.D_T_I,\n"
            "        p.sigma_T, p.k_rep_T, p.sigma_I, p.k_rep_I, p.sigma_TI, p.k_rep_TI, p.L,\n"
            "        p.chi_a, p.chi_s, p.r_kill, 0.0, 0.0, p.nbr_radius, p.nbr_threshold)\n"
            "final_xy = pos_I[alive_I].copy()\n"
            "print(f't = 0:  median r from center = {np.median(np.linalg.norm(initial_xy - p.L/2, axis=1)):.1f}')\n"
            "print(f't = {p.T_final:.0f}: median r from center = {np.median(np.linalg.norm(final_xy - p.L/2, axis=1)):.1f}')"
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=110,\n"
            "                          gridspec_kw={'width_ratios':[1,1,1.1]})\n"
            "fig.patch.set_facecolor(style.BG)\n"
            "for ax, xy, title in zip(\n"
            "        axes[:2], [initial_xy, final_xy],\n"
            "        ['t = 0  (uniform)', f't = {p.T_final:.0f}  (aggregated)']):\n"
            "    ax.set_facecolor(style.BG)\n"
            "    ax.imshow(gaussian_filter(c_a, 0.6), extent=[0, p.L, 0, p.L], origin='lower',\n"
            "               cmap=style.ATTRACTANT_CMAP, alpha=style.FIELD_ALPHA, interpolation='bilinear')\n"
            "    ax.add_collection(EllipseCollection(\n"
            "        widths=np.full(len(xy), 1.0), heights=np.full(len(xy), 1.0),\n"
            "        angles=np.zeros(len(xy)), units='x', offsets=xy,\n"
            "        transOffset=ax.transData,\n"
            "        facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,\n"
            "        linewidths=0.4, alpha=style.PARTICLE_ALPHA))\n"
            "    ax.set_xlim(0, p.L); ax.set_ylim(0, p.L)\n"
            "    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])\n"
            "    ax.set_title(title, color=style.FG)\n"
            "\n"
            "ax = axes[2]; ax.set_facecolor(style.BG)\n"
            "center = np.array([p.L/2, p.L/2])\n"
            "r_edges = np.linspace(0, p.L/2, 25)\n"
            "r_c = 0.5*(r_edges[:-1] + r_edges[1:])\n"
            "areas = np.pi*(r_edges[1:]**2 - r_edges[:-1]**2)\n"
            "for xy, lab, col, lw in [\n"
            "        (initial_xy, 't = 0', style.MUTED, 2.0),\n"
            "        (final_xy,   f't = {p.T_final:.0f}', style.TCELL, 2.5)]:\n"
            "    d = np.linalg.norm(xy - center, axis=1)\n"
            "    h, _ = np.histogram(d, bins=r_edges)\n"
            "    ax.plot(r_c, h/areas, color=col, lw=lw, label=lab)\n"
            "ax.fill_between(r_c, 0, np.histogram(np.linalg.norm(final_xy - center, axis=1), bins=r_edges)[0]/areas,\n"
            "                 color=style.TCELL, alpha=0.2)\n"
            "ax.set_xlabel('r from source'); ax.set_ylabel(r'T-cell density  $\\rho_I(r)$')\n"
            "ax.legend(frameon=False); ax.set_title('Chemotactic peak at source', color=style.FG)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure:** 200 T cells start uniformly across the box and end up packed at the\n"
            "Gaussian source. The radial density jumps from a flat ~0.1 to a spike ~5–6 near r = 0. ✓"
        ),
        md(
            "## 2.2  Small tumor + many T cells — killing in isolation\n\n"
            "Place 50 tumor cells in a tight disk, surround with 200 T cells, **disable proliferation** so\n"
            "the only dynamics are: T cells find the tumor, T cells kill on contact. Tumor count should\n"
            "drop to zero in tens of time units."
        ),
        code(
            "p = SimParams(L=60.0, G=64, T_final=40.0,\n"
            "              chi_a=10.0, chi_s=0.0,\n"
            "              N_T_initial=50, N_I_initial=200, tumor_disk_radius=4.0,\n"
            "              p_div=0.0,                    # killing-only test\n"
            "              r_kill=1.5, p_kill=0.1,\n"
            "              N_T_max=128, N_I_max=256)\n"
            "out = run(rho_I=200, alpha=0.0, seed=1, params=p)\n"
            "print(f'final N_T = {out.n_T[-1]} (started 50), T_final fraction = {out.final_tumor_fraction:.3f}')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG); ax.set_facecolor(style.BG)\n"
            "t = np.asarray(out.times)\n"
            "ax.plot(t, out.n_T, color=style.TUMOR, lw=2.5, label=r'$N_T(t)$')\n"
            "ax.plot(t, out.n_I, color=style.TCELL, lw=2.5, label=r'$N_I(t)$')\n"
            "ax.axhline(0, color=style.MUTED, lw=0.7, ls=':')\n"
            "ax.set_xlabel('t'); ax.set_ylabel('count')\n"
            "ax.set_title('Killing rule drives tumor extinction', color=style.FG)\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure:** the red curve is the tumor count; it drops to 0 within ~10 time units.\n"
            "T cells stay at 200 throughout — no proliferation, no death. ✓"
        ),
        md(
            "## 2.3  Full model: one representative run\n\n"
            "Now everything is on at once: tumor proliferation, T-cell chemotaxis up the attractant, away\n"
            "from the suppressant, killing, repulsion. We're sampling a control / boundary point\n"
            "(ρ_I=229, α=3.57) where the dynamics are interesting — the tumor and immune response are\n"
            "roughly balanced. Reusing the same seed that produced the slide-3 thumbnail."
        ),
        code(
            "p = SimParams()   # defaults\n"
            "out = run(rho_I=229, alpha=3.57, seed=1, params=p, snapshot_every=200)\n"
            "print(f'final N_T = {out.n_T[-1]},  final N_I = {out.n_I[-1]}')\n"
            "print(f'final tumor fraction (capped) = {out.final_tumor_fraction:.2f}')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG); ax.set_facecolor(style.BG)\n"
            "t = np.asarray(out.times)\n"
            "ax.plot(t, out.n_T, color=style.TUMOR, lw=2.5, label=r'$N_T(t)$')\n"
            "ax.plot(t, out.n_I, color=style.TCELL, lw=2.5, label=r'$N_I(t)$')\n"
            "ax.set_xlabel('t'); ax.set_ylabel('cell count')\n"
            "ax.set_title('Full model:  ρ_I = 229,  α = 3.57  (control / dormancy)', color=style.FG)\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure:** the tumor curve sits in a dormant band of ~50–100 cells for most of\n"
            "the run. This is the 'control' phase — what the spec called dormancy. Phase 3 sweeps this\n"
            "two-parameter space to map where each phase lives."
        ),
    ]
    return nb


# ===========================================================================
# 03 — phase diagram
# ===========================================================================

def nb03_phase_diagram():
    nb = nbf.v4.new_notebook()
    nb.metadata = nb01_single_species().metadata
    nb.cells = [
        md(
            "# Phase 3 — Tumor–immune phase diagram\n\n"
            "Sweep over `ρ_I` (initial T-cell count) and `α = χ_s` (immunosuppression strength).  \n"
            "8 × 8 grid × 3 seeds = **192 runs**, executed in parallel via `joblib`.\n\n"
            "Cached result loaded from `outputs/data/phase_grid.npz` so this notebook re-renders in seconds.\n"
            "To re-run the full sweep:\n\n"
            "```\n"
            "python -m src.sweep full --grid-size 8 --n-seeds 3\n"
            "```"
        ),
        code(
            "import sys, pathlib\n"
            "# robust root resolution: walk up until we find src/\n"
            "ROOT = pathlib.Path().resolve()\n"
            "while not (ROOT / 'src').is_dir() and ROOT.parent != ROOT:\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
            "\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import LogNorm\n"
            "from src import style\n"
            "style.apply_style()\n"
            "\n"
            "data = np.load(ROOT / 'outputs' / 'data' / 'phase_grid.npz', allow_pickle=True)\n"
            "grid = data['grid']           # (n_rho, n_alpha, n_seeds)\n"
            "mean = data['mean']           # geometric mean over seeds\n"
            "rho_vals = data['rho_I_values']\n"
            "alpha_vals = data['alpha_values']\n"
            "print(f'grid shape {grid.shape}')\n"
            "print(f'ρ_I values: {rho_vals}')\n"
            "print(f'α   values: {np.round(alpha_vals, 2)}')"
        ),
        md(
            "## 3.1  The phase diagram\n\n"
            "Cells colored by final tumor fraction `N_T(T_final) / N_T(0)`, on a log color scale, geometric\n"
            "mean over 3 seeds. Floor at 0.01 (clearance), cap at 100 (escape)."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG); ax.set_facecolor(style.BG)\n"
            "safe = np.clip(mean, 1e-2, 1e2)\n"
            "im = ax.pcolormesh(alpha_vals, rho_vals, safe,\n"
            "                   cmap=style.PHASE_CMAP, norm=LogNorm(vmin=1e-2, vmax=1e2),\n"
            "                   shading='auto')\n"
            "ax.set_xlabel(r'Immunosuppression strength  $\\alpha = \\chi_s$')\n"
            "ax.set_ylabel(r'Initial T-cell count  $\\rho_I$')\n"
            "ax.set_yscale('log')\n"
            "ax.set_title('Tumor–immune phase diagram', color=style.FG)\n"
            "cb = fig.colorbar(im, ax=ax, label=r'Final tumor fraction (log)', pad=0.02)\n"
            "cb.ax.tick_params(colors=style.FG)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure.** Clearance (dark) occupies a small wedge at high ρ_I and low α. Almost\n"
            "everywhere else, the tumor saturates the N_T_max cap (escape). The transition between the two\n"
            "is sharp — that boundary is the 'control / dormancy' region.\n\n"
            "This is biologically meaningful: it predicts that adoptive T-cell therapies or checkpoint\n"
            "inhibitors should have a **threshold** rather than a smooth dose-response. Either you cross the\n"
            "boundary into clearance or you don't."
        ),
        md(
            "## 3.2  Trajectories per phase\n\n"
            "Three trajectories, one per phase, plotted side-by-side. The control trajectory is the diagnostic\n"
            "one — it shows the dormant period before the eventual outcome."
        ),
        code(
            "keys = data['traj_keys']\n"
            "nT_all = data['traj_nT']\n"
            "t_all = data['traj_t']\n"
            "\n"
            "def get_traj(rho_idx, alpha_idx, seed_idx):\n"
            "    idx = np.where((keys[:,0]==rho_idx) & (keys[:,1]==alpha_idx) & (keys[:,2]==seed_idx))[0][0]\n"
            "    return t_all[idx], nT_all[idx]\n"
            "\n"
            "# pick: clearance @ rho=800 alpha=0, control @ rho=229 alpha=3.57 seed=1, escape @ rho=10 alpha=25\n"
            "rho_to_idx = {int(v): i for i, v in enumerate(rho_vals)}\n"
            "alpha_to_idx = {float(np.round(a, 2)): i for i, a in enumerate(alpha_vals)}\n"
            "alpha_keys = list(alpha_to_idx.keys())\n"
            "\n"
            "t_clear, nT_clear = get_traj(rho_to_idx[800], alpha_to_idx[min(alpha_keys)], 0)\n"
            "t_ctrl,  nT_ctrl  = get_traj(rho_to_idx[229], 1, 0)   # alpha index 1 = 3.57\n"
            "t_esc,   nT_esc   = get_traj(rho_to_idx[10],  len(alpha_vals)-1, 0)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.8), dpi=110)\n"
            "fig.patch.set_facecolor(style.BG); ax.set_facecolor(style.BG)\n"
            "ax.plot(t_clear, nT_clear, color=style.FG,     lw=2.4, label=r'Clearance  $\\rho_I=800, \\alpha=0$')\n"
            "ax.plot(t_ctrl,  nT_ctrl,  color=style.ACCENT, lw=2.6, label=r'Control     $\\rho_I=229, \\alpha=3.57$')\n"
            "ax.plot(t_esc,   nT_esc,   color=style.TUMOR,  lw=2.4, label=r'Escape       $\\rho_I=10,  \\alpha=25$')\n"
            "ax.set_xlabel('t'); ax.set_ylabel(r'$N_T(t)$')\n"
            "ax.set_title('Tumor count vs. time, one trajectory per phase', color=style.FG)\n"
            "ax.legend(frameon=False, fontsize=10)\n"
            "plt.tight_layout()"
        ),
        md(
            "## 3.3  The pre-rendered slide figure\n\n"
            "The composed slide-quality version with thumbnails per phase + the trajectory inset is\n"
            "cached on disk. Embedding it here for ease of review:"
        ),
        code(
            "from IPython.display import Image\n"
            "Image(filename=str(ROOT / 'outputs' / 'figures' / 'phase_diagram.png'))"
        ),
        md(
            "## 3.4  Caveats — read me before sharing the result\n\n"
            "* The 'control' phase is **bistable** at the boundary, not a stable steady-state — different\n"
            "  seeds in the same gridcell can go either way.\n"
            "* The escape value in this heatmap is capped by the array allocation `N_T_max = 800` (set in\n"
            "  `SimParams`). Real escape runs probably want larger arrays; that wasn't necessary for\n"
            "  *distinguishing* the phases.\n"
            "* Five tuning choices diverged from the spec defaults to make the phases visible in the time\n"
            "  budget — `chi_a 5 → 20`, `p_kill 0.05 → 0.12`, `p_div 0.005 → 0.004`, `T_final 200 → 100`,\n"
            "  `N_T_max 4000 → 800`. All logged in `docs/DECISIONS.md`."
        ),
    ]
    return nb


# ===========================================================================
# 04 — treatment experiment
# ===========================================================================

def nb04_treatment():
    nb = nbf.v4.new_notebook()
    nb.metadata = nb01_single_species().metadata
    nb.cells = [
        md(
            "# Phase 4 — Simulated checkpoint inhibitor\n\n"
            "Pick a point clearly in the **escape** phase (ρ_I = 300, α = 10). Run two simulations from the\n"
            "same seed:\n\n"
            "* **No treatment** — α stays at 10.0 for the whole run. Tumor saturates.\n"
            "* **With treatment** — at `t = 20`, set α to 0 (full checkpoint inhibition). Same seed,\n"
            "  same initial condition, identical noise stream up to that moment.\n\n"
            "If the model captures the biology, the second trajectory should pivot from escape to clearance."
        ),
        code(
            "import sys, pathlib\n"
            "# robust root resolution: walk up until we find src/\n"
            "ROOT = pathlib.Path().resolve()\n"
            "while not (ROOT / 'src').is_dir() and ROOT.parent != ROOT:\n"
            "    ROOT = ROOT.parent\n"
            "if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\n"
            "\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src import style\n"
            "style.apply_style()\n"
            "\n"
            "# load cached treatment trajectories\n"
            "d = np.load(ROOT / 'outputs' / 'data' / 'treatment_runs.npz', allow_pickle=True)\n"
            "print('cached fields:', sorted(d.files))\n"
            "rho_I = int(d['rho_I']); alpha = float(d['alpha']); alpha_after = float(d['alpha_after']); t_treat = float(d['t_treat'])\n"
            "print(f'ρ_I = {rho_I}   α = {alpha} → {alpha_after} at t = {t_treat}')"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=110, sharey=True)\n"
            "fig.patch.set_facecolor(style.BG)\n"
            "for ax in axes: ax.set_facecolor(style.BG)\n"
            "\n"
            "ax = axes[0]\n"
            "ax.plot(d['no_treat_times'], d['no_treat_nT'], color=style.TUMOR, lw=2.6, label=r'$N_T$')\n"
            "ax.plot(d['no_treat_times'], d['no_treat_nI'], color=style.TCELL, lw=2.6, label=r'$N_I$')\n"
            "ax.set_xlabel('t'); ax.set_ylabel('cell count')\n"
            "ax.set_title(f'No treatment  (α = {alpha:.1f})', color=style.FG)\n"
            "ax.legend(frameon=False)\n"
            "\n"
            "ax = axes[1]\n"
            "ax.plot(d['treat_times'], d['treat_nT'], color=style.TUMOR, lw=2.6, label=r'$N_T$')\n"
            "ax.plot(d['treat_times'], d['treat_nI'], color=style.TCELL, lw=2.6, label=r'$N_I$')\n"
            "ax.axvline(t_treat, color=style.ACCENT, lw=1.8, ls='--', alpha=0.9)\n"
            "ax.text(t_treat + 1, ax.get_ylim()[1] * 0.95,\n"
            "        f'  treatment: α → {alpha_after:.1f}',\n"
            "        color=style.ACCENT, fontsize=10, verticalalignment='top')\n"
            "ax.set_xlabel('t')\n"
            "ax.set_title(f'With treatment at t = {t_treat:.0f}', color=style.FG)\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()"
        ),
        md(
            "**Reading the figure.** Identical trajectory up to t = 20 (350 tumor cells, escape-phase\n"
            "behavior). Then α drops to 0 and the orange dashed line marks the moment. T cells flood\n"
            "into the tumor mass and the tumor curve plunges 350 → 130 in ~5 time units, then continues\n"
            "to decline to extinction by t ≈ 90. **The same starting state goes to two opposite outcomes\n"
            "depending on whether or not you flip a single parameter mid-run.**\n\n"
            "Mechanistically, this is exactly what anti-PD-1 / anti-PD-L1 antibodies do clinically:\n"
            "they reduce the effective immunosuppression around tumors, letting infiltrating CD8+ T\n"
            "cells re-engage."
        ),
        md(
            "## 4.1  Watch the videos\n\n"
            "Each phase has an MP4 in `outputs/videos/`. The treatment one is the most dramatic — you\n"
            "see the moment T cells stop being repelled and start swarming."
        ),
        code(
            "from IPython.display import Video\n"
            "Video(str(ROOT / 'outputs' / 'videos' / 'treatment.mp4'), embed=False, width=720)"
        ),
        md(
            "Other videos to watch:\n\n"
            "* `outputs/videos/clearance.mp4` — overwhelming T-cell infiltration (the 'hot tumor').\n"
            "* `outputs/videos/control.mp4`   — tumor pinned in a T-cell ring (the 'immune-excluded' tumor).\n"
            "* `outputs/videos/escape.mp4`    — tumor mass with T cells repelled to the periphery (the 'cold tumor').\n\n"
            "## 4.2  Biological mapping\n\n"
            "| Simulation phase | Clinical phenotype | Histology signature |\n"
            "|---|---|---|\n"
            "| Clearance (low α)         | 'Hot' tumor               | Infiltrating CD8+ T cells |\n"
            "| Control / dormancy        | Immune-excluded tumor     | T cells in a ring around the core |\n"
            "| Escape (high α)           | 'Cold' tumor              | Few intratumoral T cells |\n\n"
            "Treatment ≈ checkpoint inhibitor: lower α, push the system across the boundary."
        ),
    ]
    return nb


# ===========================================================================

NB_BUILDERS = [
    ("01_single_species_validation.ipynb", nb01_single_species),
    ("02_two_species_demo.ipynb",          nb02_two_species),
    ("03_phase_diagram.ipynb",             nb03_phase_diagram),
    ("04_treatment_experiment.ipynb",      nb04_treatment),
]


def build_and_execute(execute: bool = True):
    for fname, builder in NB_BUILDERS:
        nb = builder()
        path = NB_DIR / fname
        nbf.write(nb, path)
        print(f"wrote {path}")
    if not execute:
        return

    for fname, _ in NB_BUILDERS:
        path = NB_DIR / fname
        print(f"\nexecuting {fname}...")
        nb = nbf.read(path, as_version=4)
        t0 = time.perf_counter()
        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                 resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbf.write(nb, path)
        print(f"  done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    build_and_execute(execute=True)
