"""Phase 5: generate hero stills + assemble 5-slide deck.

Outputs:
  outputs/figures/hero_clearance.png
  outputs/figures/hero_control.png
  outputs/figures/hero_escape.png
  slides/deck.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import EllipseCollection
from scipy.ndimage import gaussian_filter

from src import style
from src.sim import SimParams, run, run_with_treatment


# ---------------------------------------------------------------------------
# Hero frames
# ---------------------------------------------------------------------------

def make_hero(label: str, rho_I: int, alpha: float, seed: int, frame_t: float,
              out_path: Path):
    """Re-run the phase at modest resolution; render a single frame at t≈frame_t."""
    style.apply_style()
    base = SimParams()
    out = run(rho_I=rho_I, alpha=alpha, seed=seed,
              params=base, snapshot_every=50, save_fields=True)
    # find snapshot closest to frame_t
    times = np.asarray(out.times)
    k = int(np.argmin(np.abs(times - frame_t)))
    pos_T = out.pos_T_snapshots[k]
    pos_I = out.pos_I_snapshots[k]
    c_s = out.c_s_snapshots[k]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=style.DPI)
    fig.patch.set_facecolor(style.BG)
    ax.set_facecolor(style.BG)
    ax.imshow(gaussian_filter(c_s, sigma=0.7),
              extent=[0, base.L, 0, base.L], origin="lower",
              cmap=style.SUPPRESSANT_CMAP, alpha=style.FIELD_ALPHA,
              interpolation="bilinear")
    if len(pos_I):
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_I), style.TCELL_GLOW_DIAM),
            heights=np.full(len(pos_I), style.TCELL_GLOW_DIAM),
            angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
            transOffset=ax.transData,
            facecolors=style.TCELL, edgecolors="none",
            alpha=style.TCELL_GLOW_ALPHA,
        ))
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            heights=np.full(len(pos_I), style.TCELL_DIAM_DATA),
            angles=np.zeros(len(pos_I)), units="x", offsets=pos_I,
            transOffset=ax.transData,
            facecolors=style.TCELL, edgecolors=style.TCELL_EDGE,
            linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    if len(pos_T):
        ax.add_collection(EllipseCollection(
            widths=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            heights=np.full(len(pos_T), style.TUMOR_DIAM_DATA),
            angles=np.zeros(len(pos_T)), units="x", offsets=pos_T,
            transOffset=ax.transData,
            facecolors=style.TUMOR, edgecolors=style.TUMOR_EDGE,
            linewidths=0.3, alpha=style.PARTICLE_ALPHA,
        ))
    ax.set_xlim(0, base.L); ax.set_ylim(0, base.L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(style.MUTED)
    ax.set_title(
        f"{label.upper()}\nρ_I = {rho_I}   α = {alpha:.1f}   t = {times[k]:.0f}",
        color=style.FG, fontsize=style.LABEL_SIZE, pad=8, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.DPI, facecolor=style.BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def make_all_heroes():
    out_dir = ROOT / "outputs" / "figures"
    return [
        make_hero("clearance", 800, 0.0,  1, frame_t=15, out_path=out_dir / "hero_clearance.png"),
        make_hero("control",   229, 3.57, 1, frame_t=50, out_path=out_dir / "hero_control.png"),
        make_hero("escape",     50, 10.0, 1, frame_t=60, out_path=out_dir / "hero_escape.png"),
    ]


# ---------------------------------------------------------------------------
# Slide assembly
# ---------------------------------------------------------------------------

SLIDE_W = 16
SLIDE_H = 9


def _new_slide(pdf: PdfPages):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=120)
    fig.patch.set_facecolor(style.BG)
    return fig


def _close_slide(pdf: PdfPages, fig):
    pdf.savefig(fig, facecolor=style.BG)
    plt.close(fig)


def _add_image(fig, img_path: Path, rect, title: str | None = None,
               title_color=None):
    """Embed an image at fig coords rect=[x,y,w,h]; optional title above it."""
    title_color = title_color or style.FG
    ax = fig.add_axes(rect)
    ax.set_facecolor(style.BG)
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(style.MUTED)
    if title:
        ax.set_title(title, color=title_color,
                     fontsize=style.SMALL_SIZE, pad=4, weight="bold")
    return ax


def build_deck(heroes: list[Path], out_pdf: Path):
    style.apply_style()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # ===== slide 1: title + three hero frames =====
        fig = _new_slide(pdf)
        fig.text(0.5, 0.88,
                 "Tumor–Immune Active Matter:\nthree phases from minimal rules",
                 color=style.FG, fontsize=30, ha="center", va="center",
                 weight="bold")
        fig.text(0.5, 0.66,
                 "Slow proliferating tumor cells secrete an attractant (recruits T cells) and a\n"
                 "short-range suppressant (repels them). The balance produces three regimes\n"
                 "matching the clinical 'hot / excluded / cold' classification of solid tumors.",
                 color=style.MUTED, fontsize=15, ha="center", va="center")
        labels = ["CLEARANCE", "CONTROL (dormancy)", "ESCAPE"]
        colors = [style.FG, style.ACCENT, style.TUMOR]
        for i, (h, lab, col) in enumerate(zip(heroes, labels, colors)):
            _add_image(fig, h, [0.05 + 0.32 * i, 0.10, 0.27, 0.45], title=lab,
                       title_color=col)
        fig.text(0.5, 0.04,
                 "Vibe Coding Active Matter & Biophysics Hackathon  •  UCSD  •  2026-05-23",
                 color=style.MUTED, fontsize=style.SMALL_SIZE, ha="center")
        _close_slide(pdf, fig)

        # ===== slide 2: phase diagram =====
        fig = _new_slide(pdf)
        fig.text(0.5, 0.94, "Phase diagram in (ρ_I, α) space",
                 color=style.FG, fontsize=24, ha="center", va="center",
                 weight="bold")
        fig.text(0.5, 0.89,
                 "Sweep over initial T-cell count (ρ_I) and immunosuppression strength (α = χ_s).\n"
                 "Each cell: geometric mean of 3 seeds. Final tumor fraction on log color scale.",
                 color=style.MUTED, fontsize=13, ha="center", va="center")
        _add_image(fig, ROOT / "outputs" / "figures" / "phase_diagram.png",
                   [0.04, 0.05, 0.92, 0.78])
        _close_slide(pdf, fig)

        # ===== slide 3: mechanism =====
        fig = _new_slide(pdf)
        fig.text(0.5, 0.93, "Mechanism: an activator–inhibitor race",
                 color=style.FG, fontsize=24, ha="center", va="center", weight="bold")
        bullets = [
            "•  Tumor cells secrete BOTH long-range attractant c_a (D_a = 5) and short-range\n"
            "   suppressant c_s (D_s = 0.5). T cells feel  +χ_a∇c_a − α∇c_s.",
            "",
            "•  Three competing rates set the outcome at any (ρ_I, α):",
            "      —  proliferation rate    p_div · N_T",
            "      —  T-cell recruitment    χ_a × tumor c_a flux  (grows with N_T)",
            "      —  T-cell exclusion       α × tumor c_s flux  (also grows with N_T)",
            "",
            "•  In CLEARANCE: recruitment dominates exclusion, T cells reach tumor surface\n"
            "   fast enough that kill rate > birth rate; tumor extinct in ~10 time units.",
            "",
            "•  In CONTROL (dormancy): the two field couplings are nearly balanced. Tumor\n"
            "   is pinned 50–100 cells; small fluctuations either clear it or trigger escape.",
            "",
            "•  In ESCAPE: suppressant repels T cells faster than attractant pulls them in.\n"
            "   Tumor mass grows unimpeded; T cells form a frustrated ring outside.",
        ]
        fig.text(0.07, 0.80, "\n".join(bullets), color=style.FG,
                 fontsize=14, va="top", ha="left",
                 family=style.FONT_FAMILY)
        # small inset showing control thumbnail
        _add_image(fig, ROOT / "outputs" / "figures" / "hero_control.png",
                   [0.68, 0.10, 0.27, 0.55],
                   title="CONTROL: T-cell ring + persistent core",
                   title_color=style.ACCENT)
        _close_slide(pdf, fig)

        # ===== slide 4: biology =====
        fig = _new_slide(pdf)
        fig.text(0.5, 0.93, "Biological mapping",
                 color=style.FG, fontsize=24, ha="center", va="center", weight="bold")
        rows = [
            ("Simulation phase",      "Clinical phenotype",        "Histology signature"),
            ("Clearance  (low α)",    "'Hot' tumor",               "Infiltrating CD8+ T cells"),
            ("Control / dormancy",    "Immune-excluded tumor",     "T cells in a ring around the core"),
            ("Escape  (high α)",      "'Cold' tumor",              "Few intratumoral T cells; immunosuppressive stroma"),
        ]
        # render as a simple table
        x0 = 0.06; col_w = [0.30, 0.30, 0.34]
        row_h = 0.09; y0 = 0.78
        for r, row in enumerate(rows):
            for c, txt in enumerate(row):
                x = x0 + sum(col_w[:c])
                y = y0 - r * row_h
                if r == 0:
                    fig.text(x, y, txt, color=style.ACCENT,
                             fontsize=15, ha="left", va="center", weight="bold")
                else:
                    fig.text(x, y, txt, color=style.FG,
                             fontsize=14, ha="left", va="center")
            if r == 0:
                fig.add_artist(plt.Line2D([x0, x0 + sum(col_w)], [y0 - 0.04, y0 - 0.04],
                                            transform=fig.transFigure,
                                            color=style.MUTED, linewidth=1))

        fig.text(0.06, 0.34,
                 "Mechanistic interpretation of immunotherapy:",
                 color=style.ACCENT, fontsize=17, weight="bold")
        fig.text(0.06, 0.26,
                 "Anti-PD-1 / anti-PD-L1 / anti-CTLA-4 antibodies reduce effective\n"
                 "immunosuppression strength α. In our model: lowering α moves a\n"
                 "tumor from the escape phase into the control or clearance phase.\n"
                 "The treatment experiment (next slide) demonstrates this rescue.",
                 color=style.FG, fontsize=14, va="top")
        _close_slide(pdf, fig)

        # ===== slide 5: treatment =====
        fig = _new_slide(pdf)
        fig.text(0.5, 0.94, "Treatment experiment: α-knockdown rescues an escape tumor",
                 color=style.FG, fontsize=22, ha="center", va="center", weight="bold")
        fig.text(0.5, 0.89,
                 "Identical run, identical seed. At t = 20, suppressant chemotaxis coupling α\n"
                 "is set from 10 → 0 (full checkpoint inhibition). Trajectory pivots immediately.",
                 color=style.MUTED, fontsize=13, ha="center", va="center")
        _add_image(fig, ROOT / "outputs" / "figures" / "treatment_panel.png",
                   [0.06, 0.10, 0.88, 0.70])
        fig.text(0.5, 0.05,
                 "MP4 versions of every panel are in outputs/videos/. "
                 "Code: src/sim.py + src/fields.py + src/interactions.py + src/render.py.",
                 color=style.MUTED, fontsize=11, ha="center")
        _close_slide(pdf, fig)
    print(f"wrote {out_pdf}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    heroes = make_all_heroes()
    out = ROOT / "slides" / "deck.pdf"
    build_deck(heroes, out)
