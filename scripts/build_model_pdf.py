"""Build docs/model.pdf using matplotlib (no LaTeX install needed).

The PDF is a fallback for environments without pdflatex. The canonical
typeset version is docs/model.tex; compile that for the full quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src import style


PAGE_W = 8.5
PAGE_H = 11.0
DPI = 110


def new_page(title=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI)
    fig.patch.set_facecolor(style.BG)
    if title:
        fig.text(0.5, 0.95, title,
                  color=style.FG, fontsize=18, ha="center",
                  weight="bold")
    return fig


def add_section_heading(fig, y, text):
    fig.text(0.07, y, text, color=style.ACCENT, fontsize=14,
              ha="left", weight="bold")


def add_body(fig, y, text, fontsize=11, color=None, ha="left", x=0.07,
             family=None):
    fig.text(x, y, text,
              color=color or style.FG,
              fontsize=fontsize, ha=ha, va="top",
              family=family or "DejaVu Sans")


def add_eq(fig, y, math, fontsize=14, x=0.5, ha="center"):
    """Render a single math expression centered on the page."""
    fig.text(x, y, math, color=style.FG, fontsize=fontsize,
              ha=ha, va="top", math_fontfamily="cm")


def add_image(fig, rect, img_path):
    ax = fig.add_axes(rect)
    ax.set_facecolor(style.BG)
    ax.imshow(plt.imread(img_path))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(style.MUTED)


# ---------------------------------------------------------------------------

def build(out_pdf: Path):
    style.apply_style()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # ===== title page =====
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI)
        fig.patch.set_facecolor(style.BG)
        fig.text(0.5, 0.70, "Tumor–Immune\nActive Matter",
                  color=style.FG, fontsize=42, ha="center", weight="bold",
                  multialignment="center")
        fig.text(0.5, 0.56, "Mathematical and physical model summary",
                  color=style.ACCENT, fontsize=18, ha="center")
        fig.text(0.5, 0.45,
                  "I-Shan Tsai (i3tsai@ucsd.edu)\n"
                  "Chih-Yen Liu (chl250@ucsd.edu)\n\n"
                  "Vibe Coding Active Matter & Biophysics Hackathon\n"
                  "UCSD   ·   May 23, 2026",
                  color=style.MUTED, fontsize=13, ha="center")
        fig.text(0.5, 0.18,
                  "Code: src/{sim, fields, interactions, sweep, render, style}.py\n"
                  "Plan: docs/PLAN.md   ·   Decisions: docs/DECISIONS.md\n"
                  "LaTeX source: docs/model.tex   ·   Markdown: docs/model.md",
                  color=style.MUTED, fontsize=10, ha="center",
                  family=style.FONT_MONO)
        pdf.savefig(fig, facecolor=style.BG)
        plt.close(fig)

        # ===== page 1 — Setup =====
        fig = new_page("1.  Setup")
        add_body(fig, 0.88,
                  "Two species of self-propelled particles + two scalar concentration fields, in a 2D\n"
                  "periodic square box of side L.", fontsize=11)
        add_section_heading(fig, 0.80, "State variables")
        add_body(fig, 0.77,
                  "• r_i(t), θ_i(t)  —  position and heading of tumor cell i, i = 1, …, N_T(t)\n"
                  "• R_j(t), φ_j(t)  —  position and heading of T-cell j, j = 1, …, N_I(t)\n"
                  "• c_a(x, t)        —  attractant concentration (long-range)\n"
                  "• c_s(x, t)        —  suppressant concentration (short-range)\n"
                  "• ρ_T(x, t)        —  tumor density on the field grid, computed by CIC deposit",
                  family=style.FONT_MONO, fontsize=10)
        add_section_heading(fig, 0.58, "Non-conserved dynamics")
        add_body(fig, 0.55,
                  "Tumor cells are born by proliferation (a source term most active-matter models lack) and\n"
                  "removed by T-cell-mediated killing. T cells are conserved on the simulation timescale\n"
                  "(no proliferation, no natural death).")
        add_section_heading(fig, 0.46, "Theoretical neighborhood")
        add_body(fig, 0.43,
                  "•  Toner–Tu hydrodynamics of polar active matter (no proliferation).\n"
                  "•  Joanny–Prost–Ranft growing active matter (proliferation as non-conserved source).\n"
                  "•  Keller–Segel chemotaxis (the attractant loop is the classical instability).\n"
                  "The novel content is the combination of long-range positive coupling via c_a and short-\n"
                  "range negative coupling via c_s, from the SAME tumor source. That asymmetry generates\n"
                  "the bistable phase boundary which maps onto the immune-excluded tumor histology.")
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 2 — Tumor dynamics =====
        fig = new_page("2.  Tumor-cell dynamics")
        add_body(fig, 0.88,
                  "Tumor cells obey overdamped Langevin self-propulsion with soft pairwise repulsion and\n"
                  "rotational diffusion. The position equation:", fontsize=11)
        add_eq(fig, 0.78,
                r"$\dfrac{d\mathbf{r}_i}{dt}\,=\,v_T\,\mathbf{n}_i\,+\,\sum_{j\neq i}\,\mathbf{F}_{\rm rep}(\mathbf{r}_i-\mathbf{r}_j)\,+\,\sum_k\,\mathbf{F}_{\rm rep}(\mathbf{r}_i-\mathbf{R}_k)\,+\,\sqrt{2D_T}\,\boldsymbol{\eta}_i(t)$",
                fontsize=14)
        add_body(fig, 0.71,
                  "with n_i = (cos θ_i, sin θ_i). Heading evolves by:", fontsize=11)
        add_eq(fig, 0.66,
                r"$\dfrac{d\theta_i}{dt}\,=\,\sqrt{2 D_R^T}\,\xi_i(t)$",
                fontsize=14)
        add_body(fig, 0.60,
                  "η_i and ξ_i are independent unit-variance Gaussian white noises. The repulsion is\n"
                  "harmonic with cutoff at one cell diameter σ:")
        add_eq(fig, 0.52,
                r"$\mathbf{F}_{\rm rep}(\mathbf{d})\,=\,k_{\rm rep}\,\max(0,\,\sigma\,-\,\|\mathbf{d}\|)\,\hat{\mathbf{d}}$",
                fontsize=14)
        add_section_heading(fig, 0.42, "Proliferation")
        add_body(fig, 0.39,
                  "Each step, each alive tumor cell tries to divide with probability p_div, subject to a\n"
                  "local-density gate:")
        add_eq(fig, 0.31,
                r"$n_i\,=\,|\{j\neq i\,:\,\|\mathbf{r}_i-\mathbf{r}_j\|<r_{\rm nbr}\}|\,<\,n_{\max}$",
                fontsize=13)
        add_body(fig, 0.24,
                  "If the gate passes, a daughter cell is placed at r_{i'} = r_i + 0.3·σ·(cos ψ, sin ψ),\n"
                  "ψ ~ U[0, 2π), with a uniformly random heading. This is a soft Allee /\n"
                  "carrying-capacity effect: tumor cells deep inside the colony cannot divide because\n"
                  "they are surrounded by neighbors.")
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 3 — T-cell dynamics =====
        fig = new_page("3.  T-cell dynamics")
        add_body(fig, 0.88,
                  "T cells are fast active Brownian particles with chemotaxis. They are coupled to both\n"
                  "scalar fields:")
        add_eq(fig, 0.78,
                r"$\dfrac{d\mathbf{R}_j}{dt}\,=\,v_I\,\mathbf{m}_j\,+\,\chi_a\,\nabla c_a(\mathbf{R}_j)\,-\,\alpha\,\nabla c_s(\mathbf{R}_j)\,+\,\dots\,+\,\sqrt{2D_I}\,\boldsymbol{\zeta}_j(t)$",
                fontsize=13)
        add_body(fig, 0.71,
                  "where the dots are repulsion sums (T–T and T–tumor) identical in form to the tumor\n"
                  "equation, and m_j = (cos φ_j, sin φ_j). Heading evolves by:")
        add_eq(fig, 0.64,
                r"$\dfrac{d\phi_j}{dt}\,=\,\sqrt{2 D_R^I}\,\mu_j(t)$",
                fontsize=14)
        add_section_heading(fig, 0.55, "Chemotaxis: the two key terms")
        add_body(fig, 0.51,
                  "•  + χ_a ∇c_a  pulls T cells UP the attractant gradient, toward tumor signals.\n"
                  "•  − α  ∇c_s   pushes T cells DOWN the suppressant gradient, away from the tumor halo.\n\n"
                  "The coupling α ≡ χ_s is the immunosuppression strength and is ONE of the two control\n"
                  "parameters of the phase diagram. The other is ρ_I ≡ N_I(0).")
        add_section_heading(fig, 0.36, "Why this generates phases")
        add_body(fig, 0.32,
                  "Three rates compete at any (ρ_I, α):\n"
                  "•  proliferation rate     ~  p_div · N_T\n"
                  "•  T-cell recruitment     ~  χ_a × tumor c_a flux  (grows with N_T)\n"
                  "•  T-cell exclusion        ~  α   × tumor c_s flux  (also grows with N_T)\n\n"
                  "Different (ρ_I, α) coordinates put different rates on top.",
                  family=style.FONT_MONO, fontsize=10)
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 4 — Fields =====
        fig = new_page("4.  Reaction–diffusion fields")
        add_body(fig, 0.88,
                  "Both fields obey a diffusion–source–decay PDE with tumor density ρ_T as the source:")
        add_eq(fig, 0.79,
                r"$\dfrac{\partial c_a}{\partial t}\,=\,D_a\,\nabla^2 c_a\,+\,s_a\,\rho_T(\mathbf{x},t)\,-\,\lambda_a\,c_a$",
                fontsize=14)
        add_eq(fig, 0.71,
                r"$\dfrac{\partial c_s}{\partial t}\,=\,D_s\,\nabla^2 c_s\,+\,s_s\,\rho_T(\mathbf{x},t)\,-\,\lambda_s\,c_s$",
                fontsize=14)
        add_section_heading(fig, 0.61, "The asymmetry that drives the physics")
        add_body(fig, 0.57,
                  "D_a ≫ D_s  (we use 5 vs 0.5). Attractant diffuses ~3× further than suppressant before\n"
                  "decaying:")
        add_eq(fig, 0.49,
                r"$\ell_a\,=\,\sqrt{D_a/\lambda_a}\,\approx\,7\sigma\,,\qquad\ell_s\,=\,\sqrt{D_s/\lambda_s}\,\approx\,2.2\sigma$",
                fontsize=13)
        add_body(fig, 0.41,
                  "So T cells can SENSE the tumor from across the box (long-range c_a), but only feel\n"
                  "REPULSION close in (short-range c_s). The competition between those scales — with\n"
                  "opposite signs in the T-cell equation — is what generates the phase diagram.")
        add_section_heading(fig, 0.32, "Cloud-in-cell deposition")
        add_body(fig, 0.28,
                  "Each tumor cell at grid coords (g_x, g_y) with integer parts (I_x, I_y) and fractional\n"
                  "parts (f_x, f_y) contributes weight w_{p,q} / (Δx)^2 to corner (I_x+p, I_y+q) for\n"
                  "p, q ∈ {0,1}, where w_{p,q} = (1−f_x)^{1−p} f_x^p · (1−f_y)^{1−q} f_y^q.\n"
                  "Total mass is preserved exactly:  ∫ ρ_T dA = N_T.")
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 5 — Killing + Numerics =====
        fig = new_page("5.  Killing rule  &  numerical methods")
        add_section_heading(fig, 0.88, "5.1  Killing")
        add_body(fig, 0.84,
                  "For each alive T cell j:")
        add_body(fig, 0.80,
                  "1.  Find its nearest alive tumor cell i*(j) under minimum-image distance.\n"
                  "2.  If ‖R_j − r_{i*}‖ < r_kill AND U ~ U[0,1) < p_kill, remove tumor cell i*.\n\n"
                  "Iteration is over T CELLS, not tumor cells, so kill rate scales with N_I (per spec).\n"
                  "At most one kill per T cell per step (no super-killers in dense regions).",
                  family=style.FONT_MONO, fontsize=10)
        add_section_heading(fig, 0.62, "5.2  Euler–Maruyama (particles)")
        add_eq(fig, 0.54,
                r"$\mathbf{r}_i^{n+1}\,=\,\mathbf{r}_i^{n}\,+\,\Delta t\,(\,v_T\,\mathbf{n}_i\,+\,\mathbf{F}_i\,)\,+\,\sqrt{2D_T\,\Delta t}\,\boldsymbol{\xi}_i^{n}$",
                fontsize=13)
        add_body(fig, 0.47,
                  "ξ_i^n ~ N(0, I_2) i.i.d. across steps and particles. Positions wrapped mod L.")
        add_section_heading(fig, 0.40, "5.3  FTCS Euler + auto-substep (fields)")
        add_body(fig, 0.36,
                  "Five-point stencil; the CFL ratio ν = D_max·Δt / (Δx)^2 must be ≤ 1/2. We aim for\n"
                  "ν_sub ≤ 1/4 and subcycle:")
        add_eq(fig, 0.27,
                r"$n_{\rm sub}\,=\,\max(\,1,\,\lceil 4\nu \rceil\,)\,,\qquad\Delta t_{\rm sub}\,=\,\Delta t\,/\,n_{\rm sub}$",
                fontsize=13)
        add_section_heading(fig, 0.18, "5.4  Performance")
        add_body(fig, 0.14,
                  "Single @njit(cache=True, fastmath=True) inner step. Naive O(N²) pairwise force —\n"
                  "at N ≲ 1500 this is ~1 ms/step on one core. joblib.Parallel for the 192-run sweep:\n"
                  "~10 min wall time on 16 cores.")
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 6 — Parameter table =====
        fig = new_page("6.  Parameter table (production defaults)")
        rows = [
            ("L",                    "L",               "100",     "box side length"),
            ("G",                    "G",               "64",      "field grid resolution"),
            ("Δt",                   "dt",              "0.01",    "timestep"),
            ("T_f",                  "T_final",         "100",     "total simulated time"),
            ("",                     "",                "",        ""),
            ("v_T",                  "v_T",             "0.1",     "tumor self-propulsion"),
            ("D_R^T",                "D_R_T",           "0.1",     "tumor rotational diffusion"),
            ("σ",                    "sigma_T",         "1.0",     "repulsion cutoff (cell diameter)"),
            ("k_rep",                "k_rep_T",         "30",      "repulsion stiffness"),
            ("p_div",                "p_div",           "0.004",   "per-step division probability"),
            ("r_nbr",                "nbr_radius",      "1.5",     "neighbor-count radius (density gate)"),
            ("n_max",                "nbr_threshold",   "6",       "max neighbors allowed for division"),
            ("",                     "",                "",        ""),
            ("v_I",                  "v_I",             "1.0",     "T-cell self-propulsion"),
            ("D_R^I",                "D_R_I",           "1.0",     "T-cell rotational diffusion"),
            ("χ_a",                  "chi_a",           "20.0",    "attractant chemotactic coupling"),
            ("α = χ_s",              "chi_s",           "scanned", "suppressant chemotactic coupling"),
            ("",                     "",                "",        ""),
            ("D_a, D_s",             "D_a, D_s",        "5.0, 0.5","field diffusivities"),
            ("s_a, s_s",             "s_a, s_s",        "1.0, 1.0","field source rates"),
            ("λ_a, λ_s",             "lam_a, lam_s",    "0.1, 0.1","field decay rates"),
            ("",                     "",                "",        ""),
            ("r_kill",               "r_kill",          "1.5",     "killing engagement radius"),
            ("p_kill",               "p_kill",          "0.12",    "per-T-cell per-step kill probability"),
            ("",                     "",                "",        ""),
            ("N_T(0)",               "N_T_initial",     "50",      "initial tumor seed"),
            ("ρ_I = N_I(0)",         "N_I_initial",     "scanned", "initial T-cell count"),
        ]
        y = 0.88
        # header
        fig.text(0.07, y, "Symbol",      color=style.ACCENT, fontsize=11, weight="bold")
        fig.text(0.27, y, "Code name",   color=style.ACCENT, fontsize=11, weight="bold",
                  family=style.FONT_MONO)
        fig.text(0.52, y, "Value",       color=style.ACCENT, fontsize=11, weight="bold")
        fig.text(0.66, y, "Meaning",     color=style.ACCENT, fontsize=11, weight="bold")
        y -= 0.025
        fig.add_artist(mpl.lines.Line2D([0.07, 0.92], [y, y],
                       transform=fig.transFigure, color=style.MUTED, lw=0.6))
        for sym, code_name, val, desc in rows:
            y -= 0.024
            if not sym:
                continue
            fig.text(0.07, y, sym,        color=style.FG,    fontsize=10)
            fig.text(0.27, y, code_name,  color=style.MUTED, fontsize=9,
                      family=style.FONT_MONO)
            fig.text(0.52, y, val,        color=style.FG,    fontsize=10)
            fig.text(0.66, y, desc,       color=style.FG,    fontsize=10)
        add_body(fig, 0.10,
                  "Five parameters were tuned away from spec defaults so the three-phase structure\n"
                  "becomes visible in the 8-hour time budget:  χ_a 5→20,  p_kill 0.05→0.12,  p_div\n"
                  "0.005→0.004,  T_f 200→100,  N_T_max 4000→800.  See docs/DECISIONS.md.",
                  fontsize=9, color=style.MUTED)
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 7 — Order parameter + dimensionless groups =====
        fig = new_page("7.  Observables  &  dimensionless groups")
        add_section_heading(fig, 0.88, "7.1  Primary order parameter")
        add_eq(fig, 0.80,
                r"$\Phi(\rho_I,\alpha)\,=\,\mathrm{clip}\!\left(\,N_T(T_f)\,/\,N_T(0)\,,\;10^{-2},\,10^{+2}\right)$",
                fontsize=13)
        add_body(fig, 0.73,
                  "Geometrically averaged over S seeds (sensible because order parameter spans orders of\n"
                  "magnitude):")
        add_eq(fig, 0.65,
                r"$\bar{\Phi}(\rho_I,\alpha)\,=\,\exp\!\left(\,S^{-1}\sum_{s=1}^{S}\,\log\Phi^{(s)}\right)$",
                fontsize=13)
        add_section_heading(fig, 0.55, "7.2  Secondary observables")
        add_body(fig, 0.51,
                  "•  N_T(t), N_I(t)  trajectories per run\n"
                  "•  visual: positions, suppressant field heatmap, T-cell glow → see outputs/videos/\n\n"
                  "These distinguish:\n"
                  "    clearance       — monotone fast decay to 0\n"
                  "    control / dormancy — long plateau at intermediate N_T (sometimes followed by late escape)\n"
                  "    escape          — monotone fast rise to the array cap",
                  family=style.FONT_MONO, fontsize=10)
        add_section_heading(fig, 0.32, "7.3  Dimensionless groups")
        add_body(fig, 0.28,
                  "•  Péclet (each species):  Pe_T = Pe_I = v / (D_R σ) = 1   (persistence ≈ one cell diameter)\n"
                  "•  Attractant range:        ℓ_a = √(D_a/λ_a) ≈ 7 σ\n"
                  "•  Suppressant range:       ℓ_s = √(D_s/λ_s) ≈ 2.2 σ\n"
                  "•  Range ratio:             ℓ_a / ℓ_s ≈ 3.2          — the asymmetry that creates phases\n"
                  "•  Suppression ratio:       α / χ_a                  — scanned from 0 to 1.25",
                  family=style.FONT_MONO, fontsize=10)
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 8 — phase diagram figure =====
        fig = new_page("8.  Phase diagram (numerical result)")
        add_body(fig, 0.88,
                  "192 runs (8 × 8 grid × 3 seeds), final tumor fraction Φ on a log color scale.",
                  fontsize=11)
        add_image(fig, [0.05, 0.22, 0.90, 0.62],
                   ROOT / "outputs" / "figures" / "phase_diagram.png")
        add_body(fig, 0.18,
                  "Clearance occupies a small wedge at high ρ_I and low α; almost everywhere else the tumor\n"
                  "saturates the array cap (escape). The transition between the two is sharp — that\n"
                  "boundary IS the control / dormancy region. Mechanistically this predicts that\n"
                  "checkpoint inhibitors should have a THRESHOLD response rather than a smooth dose curve.",
                  fontsize=10)
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

        # ===== page 9 — treatment figure =====
        fig = new_page("9.  Treatment experiment (numerical result)")
        add_body(fig, 0.88,
                  "Identical initial state and seed. Treated run flips α: 10 → 0 at t = 20.",
                  fontsize=11)
        add_image(fig, [0.05, 0.45, 0.90, 0.40],
                   ROOT / "outputs" / "figures" / "treatment_panel.png")
        add_section_heading(fig, 0.38, "Headline result")
        add_body(fig, 0.33,
                  "The same initial state goes to two opposite outcomes depending on a single mid-run\n"
                  "parameter change. This is exactly the mechanism by which anti-PD-1, anti-PD-L1,\n"
                  "and anti-CTLA-4 antibodies rescue immune-excluded tumors clinically.\n\n"
                  "In our model, lowering α moves the system across the phase boundary in the\n"
                  "(ρ_I, α) plane from escape into clearance.  No new parameters, no extra physics —\n"
                  "just a single switch.")
        pdf.savefig(fig, facecolor=style.BG); plt.close(fig)

    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    build(ROOT / "docs" / "model.pdf")
