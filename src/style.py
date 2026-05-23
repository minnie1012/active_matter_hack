"""Visual constants for the tumor–immune active matter project.

All figures import from here. No matplotlib defaults anywhere in the
final outputs — colors, fonts, sizes, and colormaps all live in this file.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----- core palette --------------------------------------------------------

BG = "#0A0E27"          # very dark navy background
FG = "#E5E9F0"          # near-white foreground text
MUTED = "#7A8AAF"       # secondary text / grid lines
ACCENT = "#F4A261"      # warm accent for annotations / treatment line

TUMOR = "#E63946"       # warm desaturated red — tumor cells
TCELL = "#457B9D"       # cool slate blue — T cells
TUMOR_EDGE = "#7A1D24"  # darker tumor edge
TCELL_EDGE = "#1D3557"  # darker T cell edge

# ----- alpha levels --------------------------------------------------------

PARTICLE_ALPHA = 0.85
FIELD_ALPHA = 0.7
TRAJ_ALPHA = 0.45

# ----- typography ----------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"
TITLE_SIZE = 18
LABEL_SIZE = 13
TICK_SIZE = 11
ANNOT_SIZE = 11
SMALL_SIZE = 9

# ----- figure sizing -------------------------------------------------------

FIG_FRAME = (10.8, 10.8)        # square video frame canvas in inches at 100 dpi → 1080
FIG_PHASE = (10.0, 8.5)         # phase diagram slide figure
FIG_TWO_PANEL = (12.0, 5.0)     # treatment two-panel
DPI = 100                        # 100 dpi × 10.8 in = 1080 px

# ----- colormaps -----------------------------------------------------------

# Suppressant: dark purple → dark teal → warm amber. Diverging-ish, perceptually
# monotone in luminance so heatmap reads as "low → high".
SUPPRESSANT_CMAP = LinearSegmentedColormap.from_list(
    "suppressant",
    [
        (0.00, "#2D1B4E"),  # deep purple (low)
        (0.40, "#1F3A52"),  # dark teal
        (0.75, "#C97B36"),  # burnt orange
        (1.00, "#F4D35E"),  # warm yellow (high)
    ],
)

# Attractant: dark teal → mint → soft cyan. Cooler so it doesn't compete with
# tumor red when both are toggled on.
ATTRACTANT_CMAP = LinearSegmentedColormap.from_list(
    "attractant",
    [
        (0.00, "#0E2A2B"),
        (0.50, "#2E7773"),
        (1.00, "#A8E6CF"),
    ],
)

# Phase-diagram log heatmap: black → teal → magenta. Perceptually monotone.
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase",
    [
        (0.00, "#0A0E27"),
        (0.30, "#1D3557"),
        (0.55, "#457B9D"),
        (0.78, "#E63946"),
        (1.00, "#F4D35E"),
    ],
)

# ----- one-call style application -----------------------------------------

def apply_style() -> None:
    """Apply the dark style to matplotlib's global rcParams.

    Call this once at the top of every notebook / script that produces a
    figure for the slide deck.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "figure.edgecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "savefig.dpi": DPI,
            "axes.facecolor": BG,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": FG,
            "ytick.color": FG,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "text.color": FG,
            "font.family": FONT_FAMILY,
            "font.size": LABEL_SIZE,
            "legend.facecolor": BG,
            "legend.edgecolor": MUTED,
            "legend.labelcolor": FG,
            "legend.fontsize": ANNOT_SIZE,
            "lines.linewidth": 2.0,
            "image.interpolation": "bilinear",
        }
    )


def new_dark_fig(figsize=FIG_FRAME):
    """Return (fig, ax) styled for the dark palette."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


# ----- particle visual sizing (used by render.py) -------------------------

# Visual scale: how big each particle's circle is, in *data* units (cell
# diameters). The simulation has sigma_T = 1.0 so a particle diameter of 1.0
# in data units matches its physical interaction range.
TUMOR_DIAM_DATA = 1.0
TCELL_DIAM_DATA = 1.0

# T-cell glow: outer halo radius (data units) and alpha
TCELL_GLOW_DIAM = 2.0
TCELL_GLOW_ALPHA = 0.18
