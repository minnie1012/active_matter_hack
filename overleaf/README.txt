Tumor-Immune Active Matter — Overleaf-ready bundle
====================================================

To use:
  1. Upload this whole folder (or the .zip) to Overleaf as a new project.
  2. Make sure model.tex is set as the main document.
  3. Compile with pdflatex twice (so the table of contents resolves).

Contents:
  model.tex             — the LaTeX source
  figures/              — 11 PNG figures referenced by model.tex

  Model 1 figures (minimal three-species + extensions):
    phase_diagram.png                    — Model 1 phase diagram
    phase6_combo_treatment.png           — combo therapy trajectories
    phase6_macrophage_polarization.png   — M2 polarization dynamics
    phase6_heterogeneity_pressure.png    — pressure-gated proliferation
    phase6b_ecm.png                      — ECM density + MMP cavity
    phase6c_adhesion.png                 — cadherin-driven morphology

  Model 2 figures (biophysical TME + EMT + fibers + hypoxia):
    combined_panel.png                   — sim_combined.py snapshot
    tme_panel.png                        — full TME, 8 cell species
    tme_hypoxia_compare.png              — hypoxia ON vs OFF
    biophysical_panel.png                — EMT dynamics over time
    biophysical_mechanism_compare.png    — 2x2 ablation of ingredients

Authors:
  I-Shan Tsai (i3tsai@ucsd.edu)
  Chih-Yen Liu (chl250@ucsd.edu)

Vibe Coding Active Matter & Biophysics Hackathon, UCSD, 2026-05-23.
