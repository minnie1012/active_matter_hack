"""Aesthetic light-theme renderer for the TME extended simulation.

NOT shared with ``src.render`` / ``src.style`` — defines its own light
palette internally to avoid touching the dark-theme constants used elsewhere.

The look targets the BioRender TME schematic style: cream background, soft
pastel cell glyphs, smooth radial O2 colormap, with an ECM fiber network
drawn as collagen-like strokes tangent to the rho_E contours and
red sprouting vessel branches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

# ===========================================================================
# Light palette — local to this module
# ===========================================================================

BG = "#FBF6EF"          # cream background
PANEL = "#FFFCF6"       # near-white card
INK = "#2B2A2A"         # body text
SUB = "#8A857F"         # secondary text / faint dividers
RULE = "#D9D2C7"        # rule lines / panel borders
ACCENT = "#B6273A"      # vessel red

# species colors (from the spec)
TUMOR = "#E8769A"
TUMOR_EDGE = "#A03A60"
CD8 = "#3A5FCD"
CD8_EDGE = "#1F3D9B"
MAC_M1 = "#7FB3D5"
MAC_M2 = "#A578B9"
MAC_EDGE = "#4A5B70"
NK = "#E68A2E"
NK_EDGE = "#9A571A"
DC = "#8B5BA6"
DC_EDGE = "#553070"
MDSC = "#9BB5C9"
MDSC_EDGE = "#5A7088"
VESSEL = "#B6273A"
VESSEL_EDGE = "#7A1A26"
CAF = "#D9A36B"
CAF_EDGE = "#7A5A3A"
FIBER = "#8C6E3E"

# colormaps
# O2 field: blue (oxygenated edges) → cream → warm red (hypoxic core)
# we plot 1 - c_O2 so low O2 = high "hypoxia map" value = warm red
O2_CMAP = LinearSegmentedColormap.from_list(
    "o2_hypoxia",
    [
        (0.00, "#E5EEF6"),  # near-white blue (well oxygenated)
        (0.35, "#F6E5DC"),  # pale beige
        (0.65, "#E8A48E"),  # warm peach
        (1.00, "#B83A33"),  # deep hypoxic red
    ],
)
PH_CMAP = LinearSegmentedColormap.from_list(
    "ph",
    [(0.0, "#7FB3D5"), (0.5, "#F4D35E"), (1.0, "#E5867E")],
)
ROS_CMAP = LinearSegmentedColormap.from_list(
    "ros",
    [(0.0, "#F3F0E7"), (0.5, "#F5C46B"), (1.0, "#C8462B")],
)

# typography
FONT = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"


def _apply_light_style():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "font.family": FONT,
        "font.size": 10,
        "legend.facecolor": PANEL,
        "legend.edgecolor": RULE,
        "legend.fontsize": 9,
    })


# ===========================================================================
# Heuristic derived fields
# ===========================================================================

def compute_derived_fields(rho_T: np.ndarray, c_O2: np.ndarray):
    """Compute pH and ROS heuristic fields from tumor density and O2."""
    rT = rho_T.copy()
    rT_max = max(rT.max(), 1e-6)
    rho_norm = rT / rT_max
    c_O2_max = max(c_O2.max(), 1e-6)
    o2_norm = np.clip(c_O2 / c_O2_max, 0.0, 1.0)
    pH = 7.4 - (7.4 - 6.5) * rho_norm
    ROS = 0.05 + 1.0 * rho_norm * (1.0 - o2_norm)
    return pH, ROS


def deposit_density(positions: np.ndarray, L: float, G: int) -> np.ndarray:
    """Light wrapper: bilinear deposit of points → density array (for the renderer
    only; doesn't need numba)."""
    rho = np.zeros((G, G), dtype=np.float64)
    if positions is None or len(positions) == 0:
        return rho
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    for i in range(len(positions)):
        gx = positions[i, 0] * inv_dx
        gy = positions[i, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        rho[iy0, ix0] += (1 - fx) * (1 - fy) * inv_cell_area
        rho[iy0, ix1] += fx * (1 - fy) * inv_cell_area
        rho[iy1, ix0] += (1 - fx) * fy * inv_cell_area
        rho[iy1, ix1] += fx * fy * inv_cell_area
    return rho


# ===========================================================================
# Cell-type legend strip
# ===========================================================================

CELL_GLYPHS = [
    ("Tumor",      TUMOR, TUMOR_EDGE, 70, "o"),
    ("CD8 T",      CD8,   CD8_EDGE,   55, "o"),
    ("NK",         NK,    NK_EDGE,    55, "o"),
    ("DC",         DC,    DC_EDGE,    65, "D"),
    ("MDSC",       MDSC,  MDSC_EDGE,  55, "o"),
    (u"Mφ (M1)",   MAC_M1, MAC_EDGE,  75, "o"),
    (u"Mφ (M2)",   MAC_M2, MAC_EDGE,  75, "o"),
    ("CAF",        CAF,    CAF_EDGE,  75, "o"),
    ("Vessel",     VESSEL, VESSEL_EDGE, 50, "s"),
]


def draw_legend_strip(ax, fontsize=9):
    """Draw a horizontal legend strip of cell types."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(CELL_GLYPHS)
    pad = 0.02
    cell_w = (1 - 2 * pad) / n
    for i, (label, fc, ec, sz, marker) in enumerate(CELL_GLYPHS):
        xc = pad + (i + 0.5) * cell_w
        ax.scatter(
            [xc], [0.72], s=sz, marker=marker,
            facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=3,
        )
        ax.text(
            xc, 0.25, label, ha="center", va="center",
            color=INK, fontsize=fontsize,
        )


def save_legend_image(out_path: Path, dpi=150):
    """Standalone legend image for the slide deck."""
    _apply_light_style()
    fig, ax = plt.subplots(figsize=(10.5, 1.2), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw_legend_strip(ax, fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)


# ===========================================================================
# Gradient bar (kept available for external callers; no longer used in
# the main TME figure layout)
# ===========================================================================

def draw_gradient_bar(ax, cmap, vmin_label, vmax_label, title,
                      reverse=False, fontsize=9):
    """Vertical gradient bar with title at top, min at bottom, max at top."""
    ax.set_facecolor(BG)
    n = 256
    arr = np.linspace(0, 1, n).reshape(-1, 1)
    if reverse:
        arr = arr[::-1]
    ax.imshow(arr, cmap=cmap, aspect="auto", extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RULE)
        spine.set_linewidth(0.8)
    ax.text(0.5, 1.06, title, transform=ax.transAxes,
            ha="center", va="bottom", color=INK, fontsize=fontsize,
            weight="bold")
    ax.text(1.25, 1.0, str(vmax_label), transform=ax.transAxes,
            ha="left", va="top", color=INK, fontsize=fontsize - 1)
    ax.text(1.25, 0.0, str(vmin_label), transform=ax.transAxes,
            ha="left", va="bottom", color=INK, fontsize=fontsize - 1)


# ===========================================================================
# ECM fiber helpers
# ===========================================================================

def _make_fiber_points(L: float, n_fiber: int = 800, seed: int = 12345):
    """Sample ``n_fiber`` quasi-uniform points across [0,L]^2 (low-discrepancy
    via fixed seed so segment positions are stable across frames)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, L, size=(n_fiber, 2))
    return pts


def _proxy_rho_E_from_state(out_step: dict, L: float, G: int) -> np.ndarray:
    """Derive an ECM-density proxy field for fiber visualization.

    The simulation does not snapshot rho_E (ECM is disabled in the TME demo).
    We approximate the collagen / stromal density field with a smoothed
    tumor-density field plus a faint constant baseline — collagen is
    deposited around the tumor / by CAFs, so this matches the desired visual:
    fibers densest at the invasive edge.
    """
    pos_T = out_step.get("pos_T")
    if pos_T is None or len(pos_T) == 0:
        return np.full((G, G), 0.1, dtype=np.float64)
    rho_T = deposit_density(pos_T, L, G)
    # smooth — invasive edge ends up as the high-gradient zone
    rho_E = gaussian_filter(rho_T, 2.2)
    # normalise to [0, 1]-ish
    m = max(rho_E.max(), 1e-6)
    rho_E = rho_E / m
    # add a faint baseline so fibers exist (faintly) far from the tumor
    rho_E = 0.10 + 0.90 * rho_E
    return rho_E


def _build_fiber_lines(
    fiber_pts: np.ndarray, rho_E: np.ndarray, L: float,
    pos_CAF: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (segments[N,2,2], alphas[N]) for the ECM fiber LineCollection.

    Fibers are oriented PARALLEL to the rho_E gradient — i.e. radial, aligned
    with the direction of tumor invasion (the TACS-3 collagen signature of
    Provenzano et al. 2006). Length and alpha both scale with local density;
    alpha is boosted at the invasive edge (high |grad rho_E|) and near CAFs.
    """
    G = rho_E.shape[0]
    dx_grid = L / G
    # central-difference gradient
    gy, gx = np.gradient(rho_E, dx_grid)
    ix = np.clip((fiber_pts[:, 0] / dx_grid).astype(int), 0, G - 1)
    iy = np.clip((fiber_pts[:, 1] / dx_grid).astype(int), 0, G - 1)
    gx_at = gx[iy, ix]
    gy_at = gy[iy, ix]
    rho_at = rho_E[iy, ix]
    grad_mag = np.hypot(gx_at, gy_at)

    # direction parallel to gradient — fibers radiate outward from the tumor
    norm = np.hypot(gx_at, gy_at)
    tx = gx_at.copy()
    ty = gy_at.copy()
    flat = norm < 1e-8
    if flat.any():
        idx = np.flatnonzero(flat)
        ang = (idx * 0.6180339887) * 2 * np.pi
        tx[flat] = np.cos(ang)
        ty[flat] = np.sin(ang)
        norm[flat] = 1.0
    tx /= norm
    ty /= norm

    # length: bigger strokes, scales with sqrt(rho_local)
    half_len = 0.5 * L * 0.020 * np.sqrt(np.clip(rho_at, 0.0, None))
    x0 = fiber_pts[:, 0] - half_len * tx
    y0 = fiber_pts[:, 1] - half_len * ty
    x1 = fiber_pts[:, 0] + half_len * tx
    y1 = fiber_pts[:, 1] + half_len * ty
    segs = np.empty((len(fiber_pts), 2, 2), dtype=np.float64)
    segs[:, 0, 0] = x0; segs[:, 0, 1] = y0
    segs[:, 1, 0] = x1; segs[:, 1, 1] = y1

    # alpha = body density + invasive-edge boost + CAF-proximity boost
    body = np.clip(rho_at, 0.0, 1.0)
    g_max = max(grad_mag.max(), 1e-9)
    edge = grad_mag / g_max
    caf_boost = np.zeros_like(body)
    if pos_CAF is not None and len(pos_CAF) > 0:
        # for each fiber point, distance to nearest CAF; closer → boost
        # use squared-distance to avoid sqrt in inner loop
        dx_c = fiber_pts[:, 0][:, None] - pos_CAF[:, 0][None, :]
        dy_c = fiber_pts[:, 1][:, None] - pos_CAF[:, 1][None, :]
        d2 = dx_c * dx_c + dy_c * dy_c
        d_nearest = np.sqrt(d2.min(axis=1))
        caf_boost = np.exp(-d_nearest / (0.05 * L))  # decay scale ~5% of box
    alphas = np.clip(0.30 * body + 0.55 * edge + 0.55 * caf_boost, 0.0, 0.95)
    return segs, alphas


# ===========================================================================
# Vessel sprouting tree helpers
# ===========================================================================

def _vessel_tree_edges_nn(vess: np.ndarray) -> list:
    """Fallback when vessel_parent_snapshots is unavailable.

    Build a minimum-spanning-tree (Prim's, O(N^2)) over the vessel set so the
    vasculature visually appears as a single connected branched network.
    Returns a list of (i, j) index pairs.
    """
    n = len(vess)
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    # cheapest edge cost from any tree node to each non-tree node
    best_cost = np.full(n, np.inf)
    best_from = np.full(n, -1, dtype=int)
    in_tree[0] = True
    # init from node 0
    d0 = np.linalg.norm(vess - vess[0], axis=1)
    best_cost = d0
    best_from[:] = 0
    best_cost[0] = np.inf  # already in tree
    edges = []
    for _ in range(n - 1):
        # pick non-tree node with smallest cost
        j = int(np.argmin(best_cost))
        if not np.isfinite(best_cost[j]):
            break
        edges.append((int(best_from[j]), j))
        in_tree[j] = True
        best_cost[j] = np.inf
        # update neighbours
        dj = np.linalg.norm(vess - vess[j], axis=1)
        improved = dj < best_cost
        improved &= ~in_tree
        best_cost = np.where(improved, dj, best_cost)
        best_from = np.where(improved, j, best_from)
    return edges


# ===========================================================================
# CAF synthesis (fallback when pos_CAF is missing)
# ===========================================================================

def _synthesize_CAFs(L: float, pos_T: Optional[np.ndarray], n_caf: int = 25,
                     seed: int = 0) -> np.ndarray:
    """Deterministic CAF placement on an annular ring at the tumor edge.

    If ``pos_T`` is empty we just place them on a ring around the box center.
    """
    rng = np.random.default_rng(seed)
    if pos_T is not None and len(pos_T) > 0:
        cx = float(pos_T[:, 0].mean())
        cy = float(pos_T[:, 1].mean())
        # tumor radius ~ std of positions; expand a bit
        rx = float(pos_T[:, 0].std())
        ry = float(pos_T[:, 1].std())
        R = max(np.hypot(rx, ry) * 1.4, 0.06 * L)
    else:
        cx = 0.5 * L; cy = 0.5 * L
        R = 0.20 * L
    angles = np.linspace(0, 2 * np.pi, n_caf, endpoint=False)
    angles = angles + rng.uniform(0, 0.4, size=n_caf)  # jitter
    radii = R * (0.85 + 0.25 * rng.random(n_caf))
    xs = (cx + radii * np.cos(angles)) % L
    ys = (cy + radii * np.sin(angles)) % L
    return np.column_stack([xs, ys])


# ===========================================================================
# Master frame composer
# ===========================================================================

def compose_tme_frame(
    fig,
    axes: dict,
    out_step: dict,
    L: float,
    t: float,
    hyp_frac: float,
    trajs: dict,
    fiber_pts: Optional[np.ndarray] = None,
):
    """Populate the multi-axes figure for one frame.

    ``axes`` keys: main, leg, side_pop, title.
    ``out_step`` keys: pos_T, pos_I, pos_M, p_M, pos_NK, pos_DC, pos_MDSC,
                       pos_CAF, pos_CAF (optional), vessels, vessel_parents
                       (optional), c_O2 (G,G).
    ``trajs`` keys: t, n_T, n_I, n_M, n_NK, n_DC, n_MDSC, hyp_frac.
    """
    # ---- main panel ----
    main = axes["main"]
    main.clear()
    main.set_facecolor(BG)
    # background: 1 - O2 (hypoxia map)
    c_O2 = out_step["c_O2"]
    hyp = 1.0 - np.clip(c_O2 / max(c_O2.max(), 1e-6), 0.0, 1.0)
    hyp = gaussian_filter(hyp, 1.4)
    main.imshow(
        hyp, extent=[0, L, 0, L], origin="lower", cmap=O2_CMAP,
        alpha=0.85, interpolation="bicubic", zorder=0,
        vmin=0.0, vmax=1.0,
    )

    # ---- ECM fiber network (below cells, above O2 background) ----
    G = c_O2.shape[0]
    rho_E = _proxy_rho_E_from_state(out_step, L, G)
    if fiber_pts is None:
        fiber_pts = _make_fiber_points(L, n_fiber=1400)
    pos_CAF_field = out_step.get("pos_CAF", None)
    segs, alphas = _build_fiber_lines(fiber_pts, rho_E, L, pos_CAF=pos_CAF_field)
    from matplotlib.colors import to_rgb
    rgb = to_rgb(FIBER)
    colors = np.column_stack([
        np.full(len(segs), rgb[0]),
        np.full(len(segs), rgb[1]),
        np.full(len(segs), rgb[2]),
        alphas,
    ])
    lc = LineCollection(segs, colors=colors, linewidths=0.9, zorder=1,
                        capstyle="round")
    main.add_collection(lc)

    # ---- vessels (under cells but above field) ----
    vess = out_step.get("vessels", None)
    vessel_parents = out_step.get("vessel_parents", None)
    if vess is not None and len(vess):
        # connecting branches: prefer explicit parent index, else MST
        if vessel_parents is not None and len(vessel_parents) == len(vess):
            edges = []
            for k in range(len(vess)):
                p = int(vessel_parents[k])
                if 0 <= p < len(vess):
                    edges.append((p, k))
        else:
            edges = _vessel_tree_edges_nn(np.asarray(vess))
        if edges:
            branch_segs = np.array(
                [[[vess[i, 0], vess[i, 1]], [vess[j, 0], vess[j, 1]]]
                 for (i, j) in edges],
                dtype=np.float64,
            )
            # only draw branches that don't wrap across the periodic box
            dx_b = np.abs(branch_segs[:, 0, 0] - branch_segs[:, 1, 0])
            dy_b = np.abs(branch_segs[:, 0, 1] - branch_segs[:, 1, 1])
            keep = (dx_b < 0.5 * L) & (dy_b < 0.5 * L)
            if keep.any():
                lc_v = LineCollection(
                    branch_segs[keep], colors=VESSEL, linewidths=0.9,
                    alpha=0.85, zorder=2,
                )
                main.add_collection(lc_v)
        # tip cells: most-recent 5 vessels get a tiny "stub" toward tumor centre
        if out_step.get("pos_T") is not None and len(out_step["pos_T"]) > 0:
            cx = float(np.mean(out_step["pos_T"][:, 0]))
            cy = float(np.mean(out_step["pos_T"][:, 1]))
            n_tip = min(5, len(vess))
            tip_segs = []
            for k in range(len(vess) - n_tip, len(vess)):
                dx_v = cx - vess[k, 0]
                dy_v = cy - vess[k, 1]
                # handle PBC wrap for tip stub direction
                if dx_v > 0.5 * L: dx_v -= L
                elif dx_v < -0.5 * L: dx_v += L
                if dy_v > 0.5 * L: dy_v -= L
                elif dy_v < -0.5 * L: dy_v += L
                r = np.hypot(dx_v, dy_v)
                if r < 1e-6:
                    continue
                # 2σ stub ~ ~3.0 units toward tumor centre
                stub_len = min(3.0, 0.5 * r)
                x1 = vess[k, 0] + stub_len * dx_v / r
                y1 = vess[k, 1] + stub_len * dy_v / r
                tip_segs.append([[vess[k, 0], vess[k, 1]], [x1, y1]])
            if tip_segs:
                lc_tip = LineCollection(
                    tip_segs, colors=VESSEL, linewidths=0.7,
                    alpha=0.7, zorder=2,
                )
                main.add_collection(lc_tip)
        # vessel dots — slightly larger
        main.scatter(
            vess[:, 0], vess[:, 1], s=82, marker="s",
            facecolor=VESSEL, edgecolor=VESSEL_EDGE, linewidth=1.0,
            zorder=3,
        )

    # ---- helper to draw a species as filled circles ----
    def _draw_species(pos, diam, fc, ec, lw=0.4, alpha=0.95, zorder=4):
        if pos is None or len(pos) == 0:
            return
        if isinstance(fc, np.ndarray):  # per-particle facecolors (macrophage)
            main.add_collection(EllipseCollection(
                widths=np.full(len(pos), diam),
                heights=np.full(len(pos), diam),
                angles=np.zeros(len(pos)), units="x", offsets=pos,
                transOffset=main.transData,
                facecolors=fc, edgecolors=ec, linewidths=lw, alpha=alpha,
                zorder=zorder,
            ))
        else:
            main.add_collection(EllipseCollection(
                widths=np.full(len(pos), diam),
                heights=np.full(len(pos), diam),
                angles=np.zeros(len(pos)), units="x", offsets=pos,
                transOffset=main.transData,
                facecolors=fc, edgecolors=ec, linewidths=lw, alpha=alpha,
                zorder=zorder,
            ))

    # ---- CAFs (drawn early so other cells can sit on top) ----
    pos_CAF = out_step.get("pos_CAF", None)
    if pos_CAF is not None and len(pos_CAF) > 0:
        # tan-yellow ellipses, ~1.6× tumor size
        main.add_collection(EllipseCollection(
            widths=np.full(len(pos_CAF), 2.6),
            heights=np.full(len(pos_CAF), 1.6),
            angles=np.zeros(len(pos_CAF)), units="x", offsets=pos_CAF,
            transOffset=main.transData,
            facecolors=CAF, edgecolors=CAF_EDGE, linewidths=0.6,
            alpha=0.95, zorder=3,
        ))

    # MDSC bottom-most (largest pool)
    _draw_species(out_step.get("pos_MDSC"), 1.6, MDSC, MDSC_EDGE, lw=0.4, alpha=0.85, zorder=4)
    # Macrophages: per-cell color based on polarization
    pos_M = out_step.get("pos_M")
    p_M = out_step.get("p_M")
    if pos_M is not None and len(pos_M):
        from matplotlib.colors import to_rgb
        c_m2 = np.array(to_rgb(MAC_M2))
        c_m1 = np.array(to_rgb(MAC_M1))
        if p_M is None:
            p_M = np.zeros(len(pos_M))
        t_arr = (np.asarray(p_M) + 1.0) * 0.5
        colors = (1 - t_arr[:, None]) * c_m2 + t_arr[:, None] * c_m1
        _draw_species(pos_M, 1.8, colors, MAC_EDGE, lw=0.4, alpha=0.92, zorder=5)
    # NK
    _draw_species(out_step.get("pos_NK"), 1.5, NK, NK_EDGE, lw=0.4, alpha=0.95, zorder=6)
    # DC — use a diamond by drawing scatter on top
    pos_DC = out_step.get("pos_DC")
    if pos_DC is not None and len(pos_DC):
        main.scatter(pos_DC[:, 0], pos_DC[:, 1],
                     s=50, marker="D",
                     facecolor=DC, edgecolor=DC_EDGE, linewidth=0.6,
                     alpha=0.95, zorder=7)
    # CD8
    _draw_species(out_step.get("pos_I"), 1.3, CD8, CD8_EDGE, lw=0.4, alpha=0.95, zorder=8)
    # Tumor (on top because they're the focus)
    _draw_species(out_step.get("pos_T"), 1.4, TUMOR, TUMOR_EDGE, lw=0.4, alpha=0.95, zorder=9)

    main.set_xlim(0, L); main.set_ylim(0, L)
    main.set_aspect("equal")
    main.set_xticks([]); main.set_yticks([])
    for s in main.spines.values():
        s.set_edgecolor(RULE)
        s.set_linewidth(1.0)

    # ---- title bar ----
    title_ax = axes.get("title", None)
    if title_ax is not None:
        title_ax.clear()
        title_ax.set_xlim(0, 1); title_ax.set_ylim(0, 1)
        title_ax.axis("off")
        title_ax.set_facecolor(BG)
        title_ax.text(
            0.5, 0.6,
            f"Tumor Microenvironment   |   t = {t:6.2f}   |   hypoxic fraction = {hyp_frac*100:.1f}%",
            ha="center", va="center",
            color=INK, fontsize=14, weight="bold",
        )
        n_T = trajs.get("n_T_now", 0); n_I = trajs.get("n_I_now", 0)
        n_M = trajs.get("n_M_now", 0); n_NK = trajs.get("n_NK_now", 0)
        n_DC = trajs.get("n_DC_now", 0); n_MDSC = trajs.get("n_MDSC_now", 0)
        n_V = trajs.get("n_V_now", 0)
        sub = (f"Tumor: {n_T}    CD8: {n_I}    Mφ: {n_M}    NK: {n_NK}    "
               f"DC: {n_DC}    MDSC: {n_MDSC}    Vessels: {n_V}")
        title_ax.text(
            0.5, 0.18, sub, ha="center", va="center",
            color=SUB, fontsize=10, family=FONT_MONO,
        )

    # ---- legend strip ----
    if axes.get("leg") is not None:
        axes["leg"].clear()
        draw_legend_strip(axes["leg"], fontsize=9)

    # ---- side population trajectory ----
    if axes.get("side_pop") is not None:
        side = axes["side_pop"]
        side.clear()
        side.set_facecolor(PANEL)
        t_arr = trajs["t"]
        for key, color, label in (
            ("n_T", TUMOR, r"$N_T$ tumor"),
            ("n_I", CD8, r"$N_{CD8}$"),
            ("n_M", MAC_M2, r"$N_M$ macro"),
            ("n_NK", NK, r"$N_{NK}$"),
            ("n_DC", DC, r"$N_{DC}$"),
            ("n_MDSC", MDSC, r"$N_{MDSC}$"),
        ):
            arr = trajs.get(key, None)
            if arr is None:
                continue
            side.plot(t_arr, arr, color=color, linewidth=1.6, alpha=0.9,
                      label=label)
        side.set_xlabel("t", color=INK, fontsize=10)
        side.set_ylabel("cell count", color=INK, fontsize=10)
        side.tick_params(colors=INK, labelsize=9)
        for s in side.spines.values():
            s.set_edgecolor(RULE)
        side.legend(frameon=False, fontsize=8, loc="upper left", ncol=2,
                    handlelength=1.6, columnspacing=0.8, labelspacing=0.3)


# ===========================================================================
# Figure layout
# ===========================================================================

def make_tme_figure(figsize=(14.0, 8.0), dpi=110):
    """Allocate the multi-axes figure used for both still frames and videos.

    Layout:
      - title strip:  [0.04, 0.92, 0.92, 0.06]
      - legend strip: [0.04, 0.84, 0.92, 0.06]
      - main panel:   [0.04, 0.08, 0.66, 0.74]
      - side pop:     [0.74, 0.20, 0.23, 0.55]   (large trajectory sidebar)
    """
    _apply_light_style()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)

    title_ax = fig.add_axes([0.04, 0.92, 0.92, 0.06])
    title_ax.set_facecolor(BG); title_ax.axis("off")

    leg_ax = fig.add_axes([0.04, 0.84, 0.92, 0.06])
    leg_ax.set_facecolor(BG); leg_ax.axis("off")

    main_ax = fig.add_axes([0.04, 0.08, 0.66, 0.74])
    main_ax.set_facecolor(BG)

    side_pop = fig.add_axes([0.74, 0.20, 0.23, 0.55])
    side_pop.set_facecolor(PANEL)

    return fig, {
        "title": title_ax,
        "leg": leg_ax,
        "main": main_ax,
        "side_pop": side_pop,
    }


# ===========================================================================
# Public renderer entry points
# ===========================================================================

def _build_step_dict(out, k: int, L: float, fallback_caf: np.ndarray):
    """Assemble the per-frame ``step`` dict shared by video / still renderers.

    Includes optional ``pos_CAF`` and ``vessel_parents`` if the sister-agent
    fields are present on the ``TMEOut`` object.
    """
    step = dict(
        pos_T=out.pos_T_snapshots[k],
        pos_I=out.pos_I_snapshots[k],
        pos_M=out.pos_M_snapshots[k] if k < len(out.pos_M_snapshots) else None,
        p_M=out.p_M_snapshots[k] if k < len(out.p_M_snapshots) else None,
        pos_NK=out.pos_NK_snapshots[k],
        pos_DC=out.pos_DC_snapshots[k],
        pos_MDSC=out.pos_MDSC_snapshots[k],
        vessels=out.vessel_snapshots[k],
        c_O2=out.c_O2_snapshots[k],
    )
    # pos_CAF: static across frames if simulation provides it
    pos_CAF = getattr(out, "pos_CAF", None)
    if pos_CAF is not None and len(pos_CAF) > 0:
        step["pos_CAF"] = np.asarray(pos_CAF)
    else:
        step["pos_CAF"] = fallback_caf
    # vessel_parents: per-frame list parallel to vessel_snapshots
    vps = getattr(out, "vessel_parent_snapshots", None)
    if vps is not None and k < len(vps):
        step["vessel_parents"] = np.asarray(vps[k])
    return step


def render_tme_video(out, out_path: Path, fps: int = 24, dpi: int = 100,
                     bitrate: int = 5000) -> Path:
    """Render the full TME simulation to MP4 (or GIF fallback)."""
    L = out.params.L
    fig, axes = make_tme_figure(figsize=(14.0, 8.0), dpi=dpi)
    # pre-compute fiber sampling once: stable visuals across frames
    fiber_pts = _make_fiber_points(L, n_fiber=1400, seed=12345)

    # ffmpeg path
    try:
        import imageio_ffmpeg
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    n_frames = len(out.pos_T_snapshots)
    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I)
    nM = np.asarray(out.n_M); nNK = np.asarray(out.n_NK)
    nDC = np.asarray(out.n_DC); nMDSC = np.asarray(out.n_MDSC)
    nV = np.asarray(out.n_vessels) if out.n_vessels else np.zeros(n_frames, dtype=int)
    hyp_arr = np.asarray(out.hypoxic_fraction)

    # fallback CAF positions: synthesize once relative to the early tumor cloud
    # so they look anchored to the stroma; if the sim provides pos_CAF later
    # _build_step_dict will prefer the real ones.
    seed_pos_T = out.pos_T_snapshots[0] if len(out.pos_T_snapshots) else None
    fallback_caf = _synthesize_CAFs(L, seed_pos_T, n_caf=25, seed=0)

    def draw(k):
        step = _build_step_dict(out, k, L, fallback_caf)
        trajs = dict(
            t=t_arr[:k+1],
            n_T=nT[:k+1], n_I=nI[:k+1], n_M=nM[:k+1],
            n_NK=nNK[:k+1], n_DC=nDC[:k+1], n_MDSC=nMDSC[:k+1],
            n_T_now=int(nT[k]), n_I_now=int(nI[k]), n_M_now=int(nM[k]),
            n_NK_now=int(nNK[k]), n_DC_now=int(nDC[k]),
            n_MDSC_now=int(nMDSC[k]), n_V_now=int(nV[k]) if k < len(nV) else 0,
        )
        compose_tme_frame(
            fig, axes, step,
            L=L, t=float(t_arr[k]),
            hyp_frac=float(hyp_arr[k]),
            trajs=trajs,
            fiber_pts=fiber_pts,
        )

    anim = FuncAnimation(fig, draw, frames=n_frames,
                         interval=1000 / fps, blit=False, repeat=False)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        anim.save(out_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"wrote {out_path}")
    except Exception as e:
        print(f"FFMpegWriter failed ({e}); falling back to GIF.")
        gif_path = Path(out_path).with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=dpi,
                  savefig_kwargs={"facecolor": BG})
        print(f"wrote {gif_path}")
        out_path = gif_path
    plt.close(fig)
    return out_path


def render_tme_still(out, frame_index: int, out_path: Path, dpi: int = 150):
    """Render a single annotated still frame to PNG."""
    L = out.params.L
    fig, axes = make_tme_figure(figsize=(14.0, 8.0), dpi=dpi)
    fiber_pts = _make_fiber_points(L, n_fiber=1400, seed=12345)
    k = frame_index
    if k < 0:
        k = len(out.pos_T_snapshots) + k
    seed_pos_T = out.pos_T_snapshots[0] if len(out.pos_T_snapshots) else None
    fallback_caf = _synthesize_CAFs(L, seed_pos_T, n_caf=25, seed=0)
    step = _build_step_dict(out, k, L, fallback_caf)
    t_arr = np.asarray(out.times)
    nT = np.asarray(out.n_T); nI = np.asarray(out.n_I)
    nM = np.asarray(out.n_M); nNK = np.asarray(out.n_NK)
    nDC = np.asarray(out.n_DC); nMDSC = np.asarray(out.n_MDSC)
    nV = np.asarray(out.n_vessels) if out.n_vessels else np.zeros(len(t_arr), dtype=int)
    hyp_arr = np.asarray(out.hypoxic_fraction)
    trajs = dict(
        t=t_arr[:k+1],
        n_T=nT[:k+1], n_I=nI[:k+1], n_M=nM[:k+1],
        n_NK=nNK[:k+1], n_DC=nDC[:k+1], n_MDSC=nMDSC[:k+1],
        n_T_now=int(nT[k]), n_I_now=int(nI[k]), n_M_now=int(nM[k]),
        n_NK_now=int(nNK[k]), n_DC_now=int(nDC[k]),
        n_MDSC_now=int(nMDSC[k]), n_V_now=int(nV[k]) if k < len(nV) else 0,
    )
    compose_tme_frame(
        fig, axes, step,
        L=L, t=float(t_arr[k]),
        hyp_frac=float(hyp_arr[k]),
        trajs=trajs,
        fiber_pts=fiber_pts,
    )
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return out_path
