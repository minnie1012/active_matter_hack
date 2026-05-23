# CLAUDE.md — Tumor–Immune Two-Species Active Matter

## Project context

This is a one-day hackathon project for **Vibe Coding: Active Matter & Biophysics Hackathon (UCSD, May 23, 2026)**. The hackathon awards scientific exploration of emergent behavior from minimal rules, not feature completeness. We are NOT trying to clone PhysiCell or CompuCell3D. We are building a minimal active-matter model that produces a clean phase diagram with three biologically meaningful phases.

Deliverable: 5-slide presentation including simulation videos, phase diagrams, in-depth analysis of emergent behavior, and direct biological connections. Total time budget: roughly 8 hours.

## Scientific framing

The tumor microenvironment is a two-species active mixture:

- **Tumor cells** are slow self-propelled particles that proliferate (a source term most active matter models lack), repel each other at contact, and secrete two diffusing chemical fields: a long-range attractant and a short-range immunosuppressant.
- **T cells** are fast self-propelled particles that chemotax up the attractant gradient and are slowed or deflected by the immunosuppressant gradient. They kill tumor cells on contact.

The system lives in the same theoretical neighborhood as Toner-Tu hydrodynamics and growing active matter (Joanny, Prost, Ranft). The novel physics here is the long-range-attraction / short-range-repulsion structure combined with non-conserved dynamics (proliferation + killing).

We expect three phases as we sweep the two control parameters:

1. **Clearance**: T cells penetrate, kill faster than tumor proliferates, final tumor mass approaches zero.
2. **Control / dormancy**: dynamic equilibrium, tumor pinned at some steady-state size.
3. **Escape**: T cells excluded or repelled, tumor grows unbounded.

These map directly onto the clinical phenotypes "hot tumor," "immune-excluded tumor," and "cold tumor," and to the logic of checkpoint inhibitors (which effectively reduce the immunosuppression parameter).

## Minimal viable model

### Particles

Two species in a 2D square box with periodic or reflecting boundaries (start with periodic; reflecting is a creative-hacking variant).

**Tumor cell `i`** at position `r_i`. Overdamped Langevin dynamics:

```
dr_i/dt = v_T * n_i + sum_j F_rep(r_i - r_j) + sqrt(2 D_T) * eta_i(t)
dn_i/dt: rotational diffusion with rate D_R_T
```

- `v_T`: tumor self-propulsion speed (small, e.g. 0.1)
- `F_rep`: soft repulsion (harmonic or WCA) with cutoff `sigma_T`
- Proliferation: each tumor cell divides with rate `p_div` per timestep when local density is below carrying capacity `rho_max`; daughter placed at small random offset
- Death: removed when killed by a T cell (see below)

**T cell `j`** at position `R_j`. Similar Langevin but with chemotaxis and suppression:

```
dR_j/dt = v_I * m_j + chi_a * grad(c_a) - chi_s * grad(c_s) + F_rep_TI + sqrt(2 D_I) * xi_j(t)
```

- `v_I`: T cell self-propulsion speed (large, e.g. 5–10x `v_T`)
- `chi_a`: chemotactic coupling to attractant
- `chi_s`: chemotactic coupling AWAY from suppressant (this is the key tunable for "immunosuppression strength alpha")
- `F_rep_TI`: weak repulsion between T cells and between species

**Killing rule**: when a T cell is within `r_kill` of a tumor cell, the tumor cell is removed with probability `p_kill` per timestep.

### Chemical fields

Two scalar fields on a grid coarser than the particle resolution (e.g. 128x128 for a 256x256 simulation box):

```
d c_a/dt = D_a * laplacian(c_a) + s_a * rho_tumor(x,y) - lambda_a * c_a
d c_s/dt = D_s * laplacian(c_s) + s_s * rho_tumor(x,y) - lambda_s * c_s
```

- `D_a >> D_s` (attractant diffuses far, suppressant stays local)
- Source proportional to local tumor density, computed by binning particles into the grid
- Linear decay so fields reach steady state

Use explicit finite-difference Euler stepping with subcycling if needed for stability. Substep the field update if the diffusion CFL condition is tight.

### Suggested default parameters

These are sensible starting points; the user should treat them as a baseline to be tuned, not gospel.

```
box: L = 100 (units of cell diameter)
grid: 128 x 128
dt = 0.01
T_final = 200 (i.e. 20000 steps)
N_tumor_initial = 50 (placed in a disk at center, radius ~5)
N_Tcell_initial = scanned (this is rho_I, axis 1 of phase diagram)
v_T = 0.1, v_I = 1.0
D_R_T = 0.1, D_R_I = 1.0
sigma_T = 1.0 (cell diameter)
p_div = 0.005 per step (doubling time ~ 200 in nondimensional units)
r_kill = 1.5, p_kill = 0.05
chi_a = 5.0 (fixed)
chi_s = scanned (this is alpha, axis 2 of phase diagram)
D_a = 5.0, D_s = 0.5
s_a = s_s = 1.0
lambda_a = lambda_s = 0.1
```

### Order parameter

Primary: `N_tumor(T_final) / N_tumor(0)`. Cap it at some large number (e.g. 100) for visualization so escape phase doesn't dominate the color scale. Use log color scale.

Secondary: time-resolved `N_tumor(t)`, `N_Tcell(t)`. These distinguish "control" (flat curve) from "delayed clearance" (slow decline) from "escape" (exponential growth).

## Implementation plan

### Phase 1: scaffolding (target: 90 minutes)

1. Set up Python project with `numpy`, `numba` or `scipy`, `matplotlib`. Numba is strongly recommended for the particle loop — without it, 1000 particles for 20000 steps will be painfully slow.
2. Implement single-species ABP with repulsion. Verify motility-induced phase separation appears at high Peclet number as a sanity check.
3. Add proliferation. Verify a small seed grows into an expanding disk with a Fisher-KPP-like front.

### Phase 2: two species and fields (target: 120 minutes)

4. Add T cell species with chemotaxis to a static Gaussian "tumor source" for testing. Verify T cells aggregate at the source.
5. Replace static source with the actual tumor field (deposit tumor positions onto grid). Implement both `c_a` and `c_s` fields with the reaction-diffusion equations above.
6. Add killing rule. Verify T cells eat a small tumor.

### Phase 3: phase diagram sweep (target: 150 minutes)

7. Wrap the simulation in a function `run(rho_I, alpha, seed) -> N_tumor_final, trajectory`.
8. Run an 8x8 or 10x10 grid in `(rho_I, alpha)` space with 3 seeds per point, in parallel using `multiprocessing` or `joblib`. Each run should be under 60 seconds; budget the grid to fit.
9. Plot the phase diagram as a heatmap with a log-scaled colormap. Overlay phase boundary contours (e.g. final tumor fraction = 0.1 and = 10).

### Phase 4: treatment experiment and polish (target: 60 minutes)

10. Pick a point clearly in the escape phase. Run for `T_final/2`, then suddenly halve `chi_s` (simulate checkpoint inhibitor). Confirm the system transitions toward control or clearance. This is the headline result for slide 5.
11. Render videos for one representative run in each phase.

### Phase 5: slides (target: 30 minutes)

Slide structure should follow the rubric exactly: title + simulation videos, phase diagram, mechanism analysis, biological connection, treatment experiment.

## Visualization requirements — this is critical

The hackathon will be judged in part on visual quality. Default matplotlib output is unacceptable. The standard to hit is "looks like a Nature methods supplementary video," not "looks like a homework plot."

### Required visual qualities

**Color palette.** Do not use matplotlib defaults. Use a perceptually uniform palette. Suggestion:

- Tumor cells: warm tone, e.g. `#E63946` (a desaturated red) with slight transparency
- T cells: cool tone, e.g. `#1D3557` or `#457B9D`
- Suppressant field: muted purple-to-yellow diverging, e.g. a custom colormap from `#2D1B4E` (low) to `#F4A261` (high) via dark teal
- Attractant field: not shown by default; available as a toggle for the methods slide
- Background: very dark, e.g. `#0A0E27` or pure `#000000`. Light backgrounds wash out the field colors and look amateur.

**Particle rendering.** Use `matplotlib.collections.CircleCollection` or `EllipseCollection` for speed, not scatter. Particles should have:

- A radius corresponding to physical size (not pixels)
- Soft edges (use a Gaussian sprite or `alpha` falloff via radial gradient) so they look like cells, not pixels
- A subtle drop shadow or glow for T cells in particular — this reads as "active" and distinguishes them visually

**Field rendering.** The suppressant field is rendered as a smooth heatmap behind the particles. Use `imshow` with `interpolation='bilinear'` and `alpha` around 0.7 so particles read clearly on top. Apply a slight Gaussian blur to the field before plotting if the grid is coarse.

**Composition.** Each frame should have:

- Main panel: 80% of the canvas, showing the simulation
- Right sidebar: live order parameter plot (`N_tumor(t)` and `N_Tcell(t)` on twin axes), updating in real time as the video plays
- Top: simulation time and current parameter values, in a clean monospace font
- Bottom: a small phase-diagram inset with a dot marking where the current run sits

**Video output.** Use `matplotlib.animation.FFMpegWriter` at 30 fps and at least 1080p. Frame every 50 simulation steps so a `T_final = 200` run becomes a 400-frame video (~13 seconds). Save as both MP4 (for slides) and GIF (for the README).

**Phase diagram figure.** Should be standalone slide-worthy:

- 10x10 grid as a heatmap with log colormap
- Phase boundary contours overlaid in white
- Three representative trajectories (one per phase) shown as small inset thumbnails connected by leader lines to their (`rho_I`, `alpha`) coordinate
- Title in large weight, axes labeled with units, colorbar labeled "Final tumor fraction (log)"

**Treatment experiment figure.** Two-panel:

- Left: order parameter vs. time for the no-treatment run (escape phase, grows exponentially)
- Right: same starting condition with treatment applied at `t = T_final / 2`, showing inflection
- Both panels share y-axis, treatment time marked with a vertical dashed line and a small annotation

**Code organization for visualization.** Keep all styling in a `style.py` module — colors, font sizes, figure sizes, colormap definitions. Every figure imports from it. This is non-negotiable for visual consistency across the deck.

### Recommended visualization stack

- `matplotlib` for static figures and animation
- `numpy` for everything numerical
- `scipy.ndimage.gaussian_filter` for field smoothing in renders
- Optional: `cmasher` or `cmocean` for higher-quality perceptual colormaps than matplotlib defaults
- Optional but recommended: do one "hero" figure in `datashader` or matplotlib with custom per-particle gradient sprites for the slide cover

## Code structure

```
project/
  src/
    sim.py          # core simulation loop, numba-accelerated
    fields.py       # reaction-diffusion update
    interactions.py # forces, chemotaxis, killing
    sweep.py        # phase diagram parallel runner
    style.py        # all visual constants
    render.py       # frame rendering, animation, figure functions
  notebooks/
    01_single_species_validation.ipynb
    02_two_species_demo.ipynb
    03_phase_diagram.ipynb
    04_treatment_experiment.ipynb
  outputs/
    figures/
    videos/
    data/
  slides/
    deck.pdf
  README.md
```

## Engineering notes for the LLM

- **Performance first.** Numba-jit the inner particle loop or this will not finish in time. A naive O(N^2) interaction with N=2000 particles for 20000 steps is too slow in pure Python; with numba it is fine. If we exceed a few thousand particles, switch to cell lists.
- **Stability.** If the simulation explodes (NaN positions, particles flying off), almost always the cause is timestep too large for the chosen force stiffness. Halve `dt` before increasing force constants.
- **Field substepping.** If `D_a * dt / dx^2 > 0.25` the explicit diffusion update is unstable. Either substep the field update or reduce `D_a`.
- **Killing fairness.** Iterate over T cells, not tumor cells, when applying the kill rule, so that the kill rate scales with T cell density, not tumor density.
- **Periodic boundaries.** Both the particle distances and the field laplacian must respect the boundary condition. Use `np.roll` for the laplacian if periodic.
- **Seeding.** Set `numpy` and `numba` seeds explicitly. Phase diagram needs reproducibility.

## What "done" looks like

- A phase diagram figure showing three clearly distinguishable phases with smooth boundaries
- At least one animation per phase, 1080p, with the composition described above
- The treatment experiment figure showing a clear phase transition under intervention
- All five slides drafted, with the headline finding stated in one sentence on slide 1

## What to avoid

- Do not implement features that don't directly produce one of the deliverables above
- Do not over-parameterize. Each new parameter doubles the sweep cost. Fix everything except `rho_I` and `alpha` for the main result.
- Do not use matplotlib defaults for any figure that will go in the slides
- Do not try to add a third cell type, vasculature, or 3D. Out of scope.
- Do not narrate every step in the slides. Show the result, then the mechanism, then the biology. Three sentences per slide maximum.

## Biological connections to land in the slides

- The three phases correspond to "hot," "excluded," and "cold" tumor phenotypes — terms the audience will recognize from immunotherapy literature
- The escape phase is rescued by reducing `alpha`, which is mechanistically what anti-PD-1 / anti-PD-L1 / anti-CTLA-4 drugs do
- The "T cell ring around the tumor" pattern that emerges in intermediate parameters is a known histopathological feature of immune-excluded tumors
- Heterogeneity in `chi_s` across tumor cells (a stretch goal if time permits) maps onto clonal heterogeneity in immunoediting
