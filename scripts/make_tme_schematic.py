"""
make_tme_schematic.py
=====================

Static, slide-ready cartoon of the Tumor Microenvironment (TME) — pure matplotlib.
Mimics a BioRender-style schematic with:

    * top row of labeled "cell-type" glyphs + functional captions,
    * central tumor mass with scattered cells, hypoxic core, ECM strands,
    * branching red vasculature with RBCs + pericytes,
    * right-side O2 / pH / ROS gradient bars,
    * left-side annotated callouts.

Outputs:
    outputs/figures/tme_schematic.png      (~14x9 in, 200 dpi)
    outputs/figures/tme_cell_legend.png    (~14x2.5 in, 200 dpi)
"""

from __future__ import annotations

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    Circle, Ellipse, FancyBboxPatch, PathPatch, FancyArrowPatch, Polygon, Rectangle,
)
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
#   Palette
# ---------------------------------------------------------------------------
BG          = "#FBF6EF"          # cream background
INK         = "#2A2A2A"          # text / outlines
OUTLINE     = "#3A2E2A"          # dark outline for glyphs
VESSEL      = "#B6273A"          # red vasculature
VESSEL_DK   = "#7E1828"
RBC         = "#D04A55"
PERICYTE    = "#E9C36B"
TUMOR_PINK  = "#E8769A"
TUMOR_PINK2 = "#F2A4BC"
TCELL       = "#5B4F8A"
TCELL_LT    = "#8A7FBA"
BCELL       = "#2E64AE"
DENDRITIC   = "#3A5FCD"
NK          = "#E68A2E"
MDSC        = "#9BB5C9"
MAC         = "#7A5DA8"          # macrophage purple
NEUTRO      = "#5C8FD0"
CAF         = "#D88AA8"
ECM         = "#C9B79A"          # tan ECM strands
HYPOXIA_BG  = "#F4E7C7"
GRAD_O2_HI  = "#F4C9CF"          # light pink (low O2 at top)
GRAD_O2_LO  = "#7E1828"          # dark red  (high O2 at bottom)
GRAD_PH_HI  = "#F4C9CF"
GRAD_PH_LO  = "#7E1828"
GRAD_ROS_HI = "#7E1828"          # ROS inverted: dark on top (high)
GRAD_ROS_LO = "#F4C9CF"

RNG = np.random.default_rng(7)


# ===========================================================================
#   Cell-glyph helpers
# ===========================================================================
def _add(ax, patch):
    ax.add_patch(patch)
    return patch


def macrophage(ax, cx, cy, r=0.32, color=MAC):
    """Multi-lobed amoeboid macrophage with spiky protrusions + dark nucleus."""
    # spiky outline using a star-like polygon
    n_spikes = 14
    angles = np.linspace(0, 2 * np.pi, n_spikes, endpoint=False)
    outer = r * (1.0 + 0.18 * RNG.uniform(-1, 1, n_spikes))
    inner = r * 0.62
    pts = []
    for i, a in enumerate(angles):
        pts.append((cx + outer[i] * np.cos(a), cy + outer[i] * np.sin(a)))
        a2 = a + np.pi / n_spikes
        pts.append((cx + inner * np.cos(a2), cy + inner * np.sin(a2)))
    _add(ax, Polygon(pts, closed=True, facecolor=color, edgecolor=OUTLINE,
                     linewidth=1.1, alpha=0.95, zorder=3))
    # dark inner blob (nucleus)
    _add(ax, Circle((cx - 0.05 * r, cy + 0.04 * r), r * 0.42,
                    facecolor="#3D2C5A", edgecolor=OUTLINE, linewidth=0.8, zorder=4))
    _add(ax, Circle((cx + 0.15 * r, cy - 0.1 * r), r * 0.22,
                    facecolor="#3D2C5A", edgecolor=OUTLINE, linewidth=0.6, zorder=4))


def dendritic(ax, cx, cy, r=0.30, color=DENDRITIC):
    """Star-shaped dendritic cell with long projections."""
    n_arms = 9
    pts = []
    for i in range(n_arms * 2):
        a = i * np.pi / n_arms
        rad = r * 1.55 if i % 2 == 0 else r * 0.55
        pts.append((cx + rad * np.cos(a), cy + rad * np.sin(a)))
    _add(ax, Polygon(pts, closed=True, facecolor=color, edgecolor=OUTLINE,
                     linewidth=1.1, alpha=0.92, zorder=3))
    _add(ax, Circle((cx, cy), r * 0.45,
                    facecolor="#26408B", edgecolor=OUTLINE, linewidth=0.7, zorder=4))


def mdsc(ax, cx, cy, r=0.30, color=MDSC):
    """Smooth round MDSC, pale blue-grey, faint nucleus."""
    _add(ax, Circle((cx, cy), r, facecolor=color, edgecolor=OUTLINE,
                    linewidth=1.0, alpha=0.95, zorder=3))
    _add(ax, Circle((cx - 0.1 * r, cy + 0.1 * r), r * 0.42,
                    facecolor="#7E97A8", edgecolor=OUTLINE, linewidth=0.6, alpha=0.85, zorder=4))


def tcell(ax, cx, cy, r=0.26, color=TCELL):
    """Compact round T-lymphocyte, indigo, large nucleus."""
    _add(ax, Circle((cx, cy), r, facecolor=color, edgecolor=OUTLINE,
                    linewidth=1.0, alpha=0.95, zorder=3))
    _add(ax, Circle((cx - 0.08 * r, cy + 0.08 * r), r * 0.6,
                    facecolor="#3E3566", edgecolor=OUTLINE, linewidth=0.6, zorder=4))


def bcell(ax, cx, cy, r=0.26, color=BCELL):
    tcell(ax, cx, cy, r=r, color=color)


def nk_cell(ax, cx, cy, r=0.30, color=NK):
    """Bi-lobed NK cell — two overlapping circles with granules."""
    _add(ax, Circle((cx - 0.18 * r, cy), r * 0.85,
                    facecolor=color, edgecolor=OUTLINE, linewidth=1.0, alpha=0.95, zorder=3))
    _add(ax, Circle((cx + 0.4 * r, cy + 0.05 * r), r * 0.7,
                    facecolor=color, edgecolor=OUTLINE, linewidth=1.0, alpha=0.95, zorder=3))
    # granules
    for gx, gy in [(-0.1, 0.1), (0.05, -0.12), (0.35, 0.15), (0.2, -0.05)]:
        _add(ax, Circle((cx + gx * r * 2.4, cy + gy * r * 2.4), r * 0.10,
                        facecolor="#9C4A0B", edgecolor=None, alpha=0.7, zorder=4))


def cancer_cell(ax, cx, cy, r=0.30, color=TUMOR_PINK):
    """Mitotic-looking dumbbell cancer cell."""
    _add(ax, Circle((cx - 0.45 * r, cy), r * 0.85,
                    facecolor=color, edgecolor=OUTLINE, linewidth=1.0, alpha=0.95, zorder=3))
    _add(ax, Circle((cx + 0.45 * r, cy), r * 0.85,
                    facecolor=color, edgecolor=OUTLINE, linewidth=1.0, alpha=0.95, zorder=3))
    # darker nuclei
    _add(ax, Circle((cx - 0.45 * r, cy), r * 0.32,
                    facecolor="#B4365E", edgecolor=OUTLINE, linewidth=0.5, zorder=4))
    _add(ax, Circle((cx + 0.45 * r, cy), r * 0.32,
                    facecolor="#B4365E", edgecolor=OUTLINE, linewidth=0.5, zorder=4))


def neutrophil(ax, cx, cy, r=0.26, color=NEUTRO):
    """Multi-lobed nucleus inside a pale circle."""
    _add(ax, Circle((cx, cy), r, facecolor="#CFE0F2", edgecolor=OUTLINE,
                    linewidth=1.0, alpha=0.95, zorder=3))
    for ox, oy in [(-0.25, 0.0), (0.05, 0.25), (0.25, -0.05), (-0.05, -0.25)]:
        _add(ax, Circle((cx + ox * r, cy + oy * r), r * 0.3,
                        facecolor=color, edgecolor=OUTLINE, linewidth=0.5, zorder=4))


def caf_cell(ax, cx, cy, w=0.85, h=0.30, angle=0, color=CAF):
    """Elongated spindle CAF."""
    _add(ax, Ellipse((cx, cy), w, h, angle=angle, facecolor=color,
                     edgecolor=OUTLINE, linewidth=1.0, alpha=0.92, zorder=3))
    # central darker nucleus
    rad = math.radians(angle)
    _add(ax, Ellipse((cx, cy), w * 0.25, h * 0.55, angle=angle,
                     facecolor="#A35070", edgecolor=OUTLINE, linewidth=0.5, zorder=4))


# ===========================================================================
#   Top-row legend (cell types + captions)
# ===========================================================================
def draw_legend_row(ax, y_glyph=0.55, y_caption_box=1.35, y_label=0.05,
                    x_positions=None):
    """Draw the top row of cell-type glyphs with their function-caption boxes."""
    entries = [
        ("Macrophage",                 macrophage,    MAC,
         "M2 polarization\n\u2191 VEGF production"),
        ("Dendritic cell",             dendritic,     DENDRITIC,
         "\u2193 Tumor antigen\ncross-presentation"),
        ("Myeloid-derived\nsuppressor cell", mdsc,    MDSC,
         "\u2191 Population expansion\n\u2191 Immunosuppressive effects"),
        ("T lymphocyte",               tcell,         TCELL,
         "\u2191 TH1 cells\n\u2191 Treg cells"),
        ("Natural killer cell",        nk_cell,       NK,
         "Inhibition of\ntumor cell lysis"),
        ("Cancer cell",                cancer_cell,   TUMOR_PINK,
         "Cancer cell\nproliferation"),
    ]

    if x_positions is None:
        x_positions = np.linspace(1.5, 12.5, len(entries))

    for x, (label, drawfn, color, caption) in zip(x_positions, entries):
        # caption box (rounded, light pink)
        box = FancyBboxPatch(
            (x - 1.0, y_caption_box - 0.32), 2.0, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#F7D7DF", edgecolor="#C8869A", linewidth=1.0, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y_caption_box + 0.03, caption,
                ha="center", va="center", fontsize=9, color=INK,
                family="DejaVu Sans", zorder=3)

        # glyph
        drawfn(ax, x, y_glyph, r=0.32 if drawfn is not nk_cell else 0.30,
               color=color)

        # cell-type label below
        ax.text(x, y_label, label, ha="center", va="top",
                fontsize=10.5, color=INK, family="DejaVu Sans",
                fontweight="medium")


# ===========================================================================
#   Tumor body
# ===========================================================================
def draw_tumor_body(ax, cx=7.2, cy=4.3, w=8.0, h=4.6):
    """Big lobed tumor blob made of clustered pink "lobes" (cancer cell mass)."""
    # outer lobed silhouette (multiple ellipses overlapping)
    lobes = [
        (cx - 2.7, cy + 0.2, 2.6, 1.9, 5),
        (cx - 1.0, cy + 1.0, 3.0, 1.7, -3),
        (cx + 1.4, cy + 0.6, 2.8, 2.0, 7),
        (cx + 2.6, cy - 0.4, 2.2, 1.6, -8),
        (cx - 2.2, cy - 0.8, 2.4, 1.6, -10),
        (cx - 0.2, cy - 0.6, 3.2, 1.6, 4),
        (cx + 0.4, cy + 1.3, 2.0, 1.2, 12),
    ]
    for (lx, ly, lw, lh, ang) in lobes:
        ax.add_patch(Ellipse((lx, ly), lw, lh, angle=ang,
                             facecolor=TUMOR_PINK2, edgecolor="#C04C72",
                             linewidth=1.1, alpha=0.95, zorder=1.2))

    # inner darker pink lobes — gives the mottled texture
    inner_lobes = [
        (cx - 1.6, cy + 0.6, 1.5, 1.0, 0),
        (cx + 0.6, cy + 0.4, 1.6, 1.0, 8),
        (cx + 1.8, cy - 0.2, 1.3, 0.9, -5),
        (cx - 0.8, cy - 0.5, 1.4, 0.9, 4),
        (cx + 1.2, cy + 1.2, 1.2, 0.8, 10),
        (cx - 2.0, cy - 0.2, 1.2, 0.9, -6),
    ]
    for (lx, ly, lw, lh, ang) in inner_lobes:
        ax.add_patch(Ellipse((lx, ly), lw, lh, angle=ang,
                             facecolor=TUMOR_PINK, edgecolor="#A8345D",
                             linewidth=0.8, alpha=0.85, zorder=1.4))


# ===========================================================================
#   ECM strands
# ===========================================================================
def draw_ecm(ax, cx=7.2, cy=4.3, n=14, seed=11):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x0 = cx + rng.uniform(-3.6, 3.6)
        y0 = cy + rng.uniform(-1.8, 1.8)
        length = rng.uniform(1.0, 2.4)
        angle = rng.uniform(-25, 25)
        # sinusoidal wavy strand
        t = np.linspace(0, 1, 40)
        xs = x0 + length * t
        ys = y0 + 0.12 * np.sin(t * 2 * np.pi * 1.5) + 0.0
        # rotate
        a = math.radians(angle)
        xr = cx + (xs - cx) * math.cos(a) - (ys - cy) * math.sin(a) + (x0 - cx) * 0
        yr = cy + (xs - cx) * math.sin(a) + (ys - cy) * math.cos(a)
        # offset so the strand starts near (x0,y0)
        xr += (x0 - xr[0])
        yr += (y0 - yr[0])
        ax.plot(xr, yr, color=ECM, linewidth=1.4, alpha=0.55, zorder=1.1)


# ===========================================================================
#   Vasculature (branching red vessel) + RBCs + pericytes
# ===========================================================================
def draw_vasculature(ax, base_y=1.35, top_y=4.7, cx=7.2):
    """Branching tree-like red vessel with thinner branches reaching into tumor."""
    # Main horizontal bottom vessel
    bottom = FancyBboxPatch((0.4, base_y - 0.35), 10.6, 0.7,
                            boxstyle="round,pad=0.0,rounding_size=0.18",
                            facecolor=VESSEL, edgecolor=VESSEL_DK, linewidth=1.2,
                            zorder=2.0)
    ax.add_patch(bottom)

    # Pericytes (yellow blobs above/below vessel)
    for px in [1.4, 3.2, 5.2, 7.6, 9.4, 10.6]:
        ax.add_patch(Ellipse((px, base_y + 0.45), 0.85, 0.3, angle=0,
                             facecolor=PERICYTE, edgecolor="#B8932E",
                             linewidth=0.8, alpha=0.95, zorder=2.05))
    for px in [2.3, 4.2, 6.3, 8.5, 10.0]:
        ax.add_patch(Ellipse((px, base_y - 0.45), 0.85, 0.3, angle=0,
                             facecolor=PERICYTE, edgecolor="#B8932E",
                             linewidth=0.8, alpha=0.95, zorder=2.05))

    # RBCs along bottom vessel
    rng = np.random.default_rng(3)
    for x in np.linspace(0.8, 10.7, 16):
        x += rng.uniform(-0.1, 0.1)
        y = base_y + rng.uniform(-0.18, 0.18)
        ax.add_patch(Ellipse((x, y), 0.30, 0.20, angle=rng.uniform(-15, 15),
                             facecolor=RBC, edgecolor="#7E1828",
                             linewidth=0.5, alpha=0.95, zorder=2.2))

    # Vertical/branching vessel pieces going up into the tumor
    # Each branch defined by a tapered "stem" polygon
    def stem(p_start, p_end, w_start, w_end):
        x0, y0 = p_start
        x1, y1 = p_end
        # perpendicular direction
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        ux, uy = -dy / L, dx / L
        poly = [
            (x0 + ux * w_start / 2, y0 + uy * w_start / 2),
            (x1 + ux * w_end / 2,   y1 + uy * w_end / 2),
            (x1 - ux * w_end / 2,   y1 - uy * w_end / 2),
            (x0 - ux * w_start / 2, y0 - uy * w_start / 2),
        ]
        ax.add_patch(Polygon(poly, closed=True, facecolor=VESSEL,
                             edgecolor=VESSEL_DK, linewidth=1.0, zorder=1.9))

    # Main trunk up from center
    stem((cx, base_y + 0.1), (cx, 3.6), 1.2, 0.6)
    # Left branches
    stem((cx, 3.6),        (cx - 1.7, 4.5), 0.6, 0.35)
    stem((cx - 1.7, 4.5),  (cx - 2.8, 4.9), 0.35, 0.22)
    stem((cx - 1.7, 4.5),  (cx - 1.4, 5.3), 0.30, 0.18)
    # Right branches
    stem((cx, 3.6),        (cx + 1.8, 4.4), 0.6, 0.35)
    stem((cx + 1.8, 4.4),  (cx + 2.9, 4.9), 0.35, 0.22)
    stem((cx + 1.8, 4.4),  (cx + 1.5, 5.3), 0.30, 0.18)
    # Mid small branch
    stem((cx, 3.6),        (cx - 0.4, 5.0), 0.4, 0.20)
    stem((cx, 3.6),        (cx + 0.6, 5.1), 0.4, 0.20)

    # A few inline RBCs in trunk
    for y in np.linspace(2.0, 3.4, 4):
        ax.add_patch(Ellipse((cx + rng.uniform(-0.2, 0.2), y), 0.28, 0.18,
                             angle=rng.uniform(-25, 25),
                             facecolor=RBC, edgecolor="#7E1828",
                             linewidth=0.5, alpha=0.95, zorder=2.2))


# ===========================================================================
#   Scatter immune + tumor cells inside the mass
# ===========================================================================
def scatter_inside_tumor(ax, cx=7.2, cy=4.7):
    """Place cancer cells, T cells, B cells, macrophages, DCs, NKs, neutrophils."""
    rng = np.random.default_rng(42)

    # cancer cells (pink dumbbells) scattered broadly
    cancer_pts = [
        (cx - 2.6, cy + 0.7), (cx - 2.0, cy + 1.3), (cx - 1.4, cy - 1.0),
        (cx + 2.5, cy + 0.7), (cx + 2.3, cy - 0.7), (cx + 1.6, cy + 1.5),
        (cx - 2.8, cy - 0.4), (cx + 3.0, cy + 0.0), (cx - 0.4, cy + 1.6),
        (cx + 0.8, cy - 1.1), (cx - 1.8, cy + 0.0),
    ]
    for (x, y) in cancer_pts:
        cancer_cell(ax, x, y, r=0.30)

    # macrophages (purple spiky) — tumor-associated
    for (x, y) in [(cx - 1.0, cy + 1.4), (cx + 1.5, cy + 1.3),
                   (cx - 2.4, cy + 1.5), (cx + 2.8, cy - 0.9),
                   (cx - 0.6, cy - 1.0)]:
        macrophage(ax, x, y, r=0.34)

    # T cells (indigo small) — clusters around mid + edges
    for (x, y) in [(cx - 1.6, cy + 1.0), (cx + 0.4, cy + 1.2),
                   (cx + 2.0, cy + 1.1), (cx - 0.2, cy - 0.8),
                   (cx + 1.0, cy - 0.6), (cx - 2.4, cy + 0.0),
                   (cx + 2.6, cy + 0.0)]:
        tcell(ax, x, y, r=0.22, color=TCELL)

    # B cells (blue)
    for (x, y) in [(cx - 2.7, cy + 1.1), (cx + 2.7, cy + 1.2)]:
        bcell(ax, x, y, r=0.24, color=BCELL)

    # NK cells (orange bilobed)
    for (x, y) in [(cx + 0.0, cy + 1.4), (cx - 1.9, cy - 0.8),
                   (cx + 1.9, cy - 0.4)]:
        nk_cell(ax, x, y, r=0.30)

    # Dendritic cells (star, blue)
    for (x, y) in [(cx + 1.2, cy + 1.7), (cx - 2.1, cy + 0.6),
                   (cx + 2.4, cy - 0.2)]:
        dendritic(ax, x, y, r=0.30)

    # Neutrophils (pale w/ multilobed nucleus)
    for (x, y) in [(cx + 0.9, cy + 0.5), (cx - 0.5, cy + 0.4)]:
        neutrophil(ax, x, y, r=0.24)

    # MDSCs
    for (x, y) in [(cx + 2.6, cy + 1.5), (cx - 2.9, cy + 0.6)]:
        mdsc(ax, x, y, r=0.22)


# ===========================================================================
#   Hypoxia label
# ===========================================================================
def draw_hypoxia(ax, x=7.0, y=4.3):
    box = FancyBboxPatch((x - 0.7, y - 0.18), 1.4, 0.45,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor=HYPOXIA_BG, edgecolor="#B8923A",
                         linewidth=1.0, zorder=5)
    ax.add_patch(box)
    ax.text(x, y + 0.045, "Hypoxia", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#5B3A12", zorder=6)


# ===========================================================================
#   CAFs around the periphery
# ===========================================================================
def draw_cafs(ax, cx=7.2, cy=4.7):
    spots = [
        (cx - 3.6, cy + 0.6,  0.9, 0.30,  18),
        (cx - 3.8, cy - 0.3,  0.9, 0.28, -22),
        (cx + 3.7, cy + 0.4,  1.0, 0.30, -18),
        (cx + 3.8, cy - 0.5,  0.9, 0.28,  20),
        (cx - 0.6, cy + 2.0,  0.9, 0.26,  6),
        (cx + 1.5, cy + 2.0,  0.9, 0.26, -6),
    ]
    for (x, y, w, h, ang) in spots:
        caf_cell(ax, x, y, w=w, h=h, angle=ang, color=CAF)


# ===========================================================================
#   Right-side gradient bars (O2 / pH / ROS)
# ===========================================================================
def draw_gradient_bar(ax, x, y0, y1, width, top_color, bottom_color,
                      label, top_text, bottom_text,
                      inverted_top_label=False):
    """Draw a thin vertical gradient strip (triangle-like) with top/bottom callouts."""
    # Sample gradient as horizontal slices
    n = 120
    ys = np.linspace(y0, y1, n)
    top_rgb = np.array(mcolors.to_rgb(top_color))
    bot_rgb = np.array(mcolors.to_rgb(bottom_color))
    # We render a thin tapered triangle: thinner at top, wider at bottom
    for i, yy in enumerate(ys):
        t = i / (n - 1)
        col = (1 - t) * top_rgb + t * bot_rgb
        # taper: 30% width at top, 100% at bottom
        w = width * (0.30 + 0.70 * t)
        ax.add_patch(Rectangle((x - w / 2, yy), w,
                               (y1 - y0) / (n - 1) + 0.01,
                               facecolor=col, edgecolor=None, zorder=2))

    # Outline
    outline = Polygon(
        [(x - width * 0.30 / 2, y1),
         (x + width * 0.30 / 2, y1),
         (x + width / 2, y0),
         (x - width / 2, y0)],
        closed=True, facecolor="none", edgecolor=VESSEL_DK,
        linewidth=1.0, zorder=3,
    )
    ax.add_patch(outline)

    # Top callout box
    tb = FancyBboxPatch((x - 0.32, y1 + 0.05), 0.64, 0.45,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor="#FBE9EC", edgecolor="#C8869A",
                        linewidth=0.9, zorder=3)
    ax.add_patch(tb)
    ax.text(x, y1 + 0.28, top_text, ha="center", va="center",
            fontsize=7.0, color=INK, zorder=4)

    # Bottom callout box
    bb = FancyBboxPatch((x - 0.32, y0 - 0.55), 0.64, 0.45,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor="#FBE9EC", edgecolor="#C8869A",
                        linewidth=0.9, zorder=3)
    ax.add_patch(bb)
    ax.text(x, y0 - 0.32, bottom_text, ha="center", va="center",
            fontsize=7.0, color=INK, zorder=4)

    # Axis-name label
    ax.text(x, y0 - 0.85, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK, zorder=4)


def draw_gradients(ax, x_start=13.3, dx=0.85, y0=2.3, y1=4.9):
    draw_gradient_bar(ax, x_start, y0, y1, width=0.40,
                      top_color=GRAD_O2_HI, bottom_color=GRAD_O2_LO,
                      label="O$_2$", top_text="0\u20132.5\nmmHg",
                      bottom_text="30\u201340\nmm Hg")
    draw_gradient_bar(ax, x_start + dx, y0, y1, width=0.40,
                      top_color=GRAD_PH_HI, bottom_color=GRAD_PH_LO,
                      label="pH", top_text="\u223c6.5",
                      bottom_text="\u223c7.4")
    draw_gradient_bar(ax, x_start + 2 * dx, y0, y1, width=0.40,
                      top_color=GRAD_ROS_HI, bottom_color=GRAD_ROS_LO,
                      label="ROS", top_text="high",
                      bottom_text="low")


# ===========================================================================
#   Left-side annotation callouts
# ===========================================================================
def draw_callouts(ax):
    """Annotation arrows on the left + top labels."""
    callouts_left = [
        ("B lymphocyte",            (0.10, 5.65), (2.80, 5.30)),
        ("T lymphocyte",            (0.10, 5.15), (3.90, 5.30)),
        ("Extra-cellular matrix",   (0.10, 4.65), (3.40, 4.55)),
        ("Cancer-associated\nfibroblast", (0.10, 4.10), (3.05, 4.65)),
        ("Tumor vasculature\nAntiangiogenic therapies\nVessel normalization",
                                    (0.10, 3.15), (4.60, 3.20)),
        ("Pericyte",                (0.20, 2.35), (2.30, 1.95)),
        ("Red blood cell",          (0.20, 0.70), (3.40, 1.20)),
    ]
    for text, xy_text, xy_target in callouts_left:
        ax.annotate(
            text, xy=xy_target, xytext=xy_text,
            fontsize=9.5, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.9,
                            shrinkA=0, shrinkB=2),
            zorder=10,
        )

    # Top labels with leader lines (kept BELOW divider at y=6.85)
    callouts_top = [
        ("Fibroblasts",                 (3.10, 6.55), (4.30, 6.05)),
        ("Tumor-associated\nmacrophages", (5.55, 6.70), (5.70, 5.95)),
        ("Natural killer cell",         (8.50, 6.65), (7.85, 5.95)),
        ("Dendritic cell",              (9.85, 6.45), (8.60, 5.70)),
        ("Neutrophils",                 (10.15, 6.10), (7.50, 5.25)),
    ]
    for text, xy_text, xy_target in callouts_top:
        ax.annotate(
            text, xy=xy_target, xytext=xy_text,
            fontsize=9.5, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.9,
                            shrinkA=0, shrinkB=2),
            zorder=10,
        )

    # Little red oval marker for "Tumor vasculature" callout
    ax.add_patch(Ellipse((2.55, 3.10), 0.42, 0.15, facecolor=VESSEL,
                         edgecolor=VESSEL_DK, linewidth=0.8, zorder=11))


# ===========================================================================
#   FULL FIGURE
# ===========================================================================
def make_main_figure(outpath):
    # data range matches figure aspect ratio (14/9 ~= 1.5556) so equal-aspect
    # leaves no extra letterboxing. 14.0 wide / 9.0 tall <=> 14:9
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 14.0)
    ax.set_ylim(0.0, 9.0)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.axis("off")

    # ---- top legend row ----
    # legend across the top band (caption box at y ~8.50, glyph at 7.85,
    # type-label at 7.20, divider just below at 6.85)
    draw_legend_row_in_axes(ax, y_glyph=7.85, y_caption_box=8.50, y_label=7.30,
                            x_positions=np.linspace(1.55, 12.45, 6))

    # divider under legend
    ax.plot([0.4, 13.6], [6.85, 6.85], color="#D5C8B6", linewidth=1.0, zorder=1)

    # ---- tumor body, slightly lower and centered ----
    cx, cy = 6.8, 4.45
    draw_tumor_body(ax, cx=cx, cy=cy, w=8.0, h=4.6)
    draw_ecm(ax, cx=cx, cy=cy, n=16)
    draw_vasculature(ax, base_y=1.45, top_y=4.8, cx=cx)
    scatter_inside_tumor(ax, cx=cx, cy=cy)
    draw_cafs(ax, cx=cx, cy=cy)
    draw_hypoxia(ax, x=cx, y=cy)

    # ---- callouts (left + top) ----
    draw_callouts(ax)

    # ---- right gradient bars ----
    draw_gradients(ax, x_start=12.2, dx=0.65, y0=2.6, y1=4.95)

    # ---- title bar at very top? leave blank for slide composition ----

    fig.savefig(outpath, dpi=200, facecolor=BG, bbox_inches=None)
    plt.close(fig)


def draw_legend_row_in_axes(ax, y_glyph, y_caption_box, y_label, x_positions):
    """Same as draw_legend_row but inside the main ax with explicit y coords."""
    entries = [
        ("Macrophage",                 macrophage,    MAC,
         "M2 polarization\n\u2191 VEGF production"),
        ("Dendritic cell",             dendritic,     DENDRITIC,
         "\u2193 Tumor antigen\ncross-presentation"),
        ("Myeloid-derived\nsuppressor cell", mdsc,    MDSC,
         "\u2191 Population expansion\n\u2191 Immunosuppressive effects"),
        ("T lymphocyte",               tcell,         TCELL,
         "\u2191 TH1 cells\n\u2191 Treg cells"),
        ("Natural killer cell",        nk_cell,       NK,
         "Inhibition of\ntumor cell lysis"),
        ("Cancer cell",                cancer_cell,   TUMOR_PINK,
         "Cancer cell\nproliferation"),
    ]

    for x, (label, drawfn, color, caption) in zip(x_positions, entries):
        # caption box
        box = FancyBboxPatch(
            (x - 0.95, y_caption_box - 0.30), 1.90, 0.60,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#F7D7DF", edgecolor="#C8869A", linewidth=1.0, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y_caption_box + 0.0, caption,
                ha="center", va="center", fontsize=8.0, color=INK,
                family="DejaVu Sans", zorder=3)

        # glyph
        drawfn(ax, x, y_glyph, r=0.30 if drawfn is not nk_cell else 0.27,
               color=color)

        # label
        ax.text(x, y_label, label, ha="center", va="top",
                fontsize=9.0, color=INK, family="DejaVu Sans",
                fontweight="bold")


# ===========================================================================
#   LEGEND-ONLY FIGURE
# ===========================================================================
def make_legend_strip(outpath):
    fig = plt.figure(figsize=(14, 2.5), facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 2.5)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.axis("off")

    # caption row near top, glyphs middle, labels near bottom
    draw_legend_row_in_axes(
        ax,
        y_glyph=1.30,
        y_caption_box=2.10,
        y_label=0.55,
        x_positions=np.linspace(1.7, 13.5, 6),
    )

    fig.savefig(outpath, dpi=200, facecolor=BG, bbox_inches=None)
    plt.close(fig)


# ===========================================================================
#   ENTRY POINT
# ===========================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))
    out_dir = os.path.join(project_root, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    main_png  = os.path.join(out_dir, "tme_schematic.png")
    legend_png = os.path.join(out_dir, "tme_cell_legend.png")

    print(f"[tme_schematic] rendering main figure -> {main_png}")
    make_main_figure(main_png)
    print(f"[tme_schematic] rendering legend strip -> {legend_png}")
    make_legend_strip(legend_png)

    # quick sanity report
    for p in (main_png, legend_png):
        size_kb = os.path.getsize(p) / 1024
        print(f"  wrote {p}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
