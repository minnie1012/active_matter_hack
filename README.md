# active_matter_hack

Tumor–immune active-matter modeling for the UCSD **Vibe Coding Active Matter & Biophysics Hackathon** (2026-05-23).

**Authors:** I-Shan Tsai (i3tsai@ucsd.edu), Chih-Yen Liu (chl250@ucsd.edu).

## Background

Solid tumors are clinically classified by CD8⁺ T-cell distribution:

| Phenotype | Histology | Anti-PD-1 response |
|---|---|---|
| **Hot** | infiltrating CD8⁺ T cells | responds |
| **Excluded** | T cells form a ring at the margin, do not enter | combo therapy needed |
| **Cold** | few or no T cells anywhere | unresponsive |

We treat this trichotomy as an **active-matter phase diagram** and ask which physical ingredients (motility, secretion, mechanics) move the boundary. The project is organized as two nested models:

- **Model 1** (`src/sim.py`, `src/sim_extended.py`, `src/sim_ecm.py`, `src/sim_adhesion.py`) — three species of active Brownian particles (tumor, CD8 T cells, macrophages) coupled by three RD fields ($c_a$, $c_s$, $c_{\text{IL10}}$), plus four cell-level extensions (per-cell heterogeneity, mechanical-pressure-gated proliferation, ECM density + MMP, cadherin adhesion).
- **Model 2** (`src/sim_biophysical.py`, `src/sim_tme.py`, `src/sim_combined.py`) — adds the mechanobiology Model 1 omits: continuous EMT state $s_{T,i}$, ECM fiber alignment + integrin biphasic traction, hypoxia/VEGF + dynamic vasculature, and NK / DC / MDSC / CAF species.

Reference: `docs/model.md`, `overleaf/model.tex` / `docs/model.pdf`, `docs/simulation_summary.md`, `docs/extensions.md`.

## Equations

### Model 1 — minimal three-species system

**Tumor cells** are slow soft-repulsive ABPs:

$$
\frac{d\mathbf{r}_i}{dt} = v_{T,i}\,\mathbf{n}_i + \sum_{j\ne i}\mathbf{F}_{\text{rep}}^{T\text{-}T} + \sum_j \mathbf{F}_{\text{rep}}^{T\text{-}I} + \sum_k \mathbf{F}_{\text{rep}}^{T\text{-}M} + \sqrt{2D_T}\,\boldsymbol{\eta}_i, \qquad \frac{d\theta_i}{dt} = \sqrt{2D_{R,i}^T}\,\xi_i
$$

with harmonic repulsion (gentler than WCA at $\Delta t = 0.01$):

$$
\mathbf{F}_{\text{rep}}(\mathbf{d}) = k_{\text{rep}}\,\max(0,\,\sigma - \lVert\mathbf{d}\rVert)\,\hat{\mathbf{d}}.
$$

Per-cell heterogeneity: $v_{T,i} \sim \mathcal{LN}(\mu_v,\sigma_v^2)$, $D_{R,i}^T \sim \Gamma(k,\theta)$, daughters inherit with drift $v_{T,i'} = v_{T,i} + \varepsilon$.

**Mechanical-pressure-gated proliferation** (Byrne–Drasdo) uses the local virial computed inside the existing force loop:

$$
P_i = \frac{1}{\pi\sigma^2}\sum_{j\in\mathcal{N}_i}\mathbf{r}_{ij}\cdot\mathbf{F}_{ij}, \qquad p_{\text{div},i} = p_{\text{div}}^0\,\max\!\big(0,\,1 - P_i/P^\star\big)
$$

Passed cells place a daughter at offset $0.3\sigma$ in a uniformly random direction.

**CD8 T cells** are fast ABPs with two-channel chemotaxis — the $c_{\text{IL10}}$ term is critical, since IL-10 from M2 macrophages is no longer slaved to tumor density:

$$
\boxed{\;\frac{d\mathbf{R}_j}{dt} = v_{I,j}\,\mathbf{m}_j + \chi_a\,\nabla c_a(\mathbf{R}_j) - \alpha\,\nabla\!\left[c_s + c_{\text{IL10}}\right]\!(\mathbf{R}_j) + \mathbf{F}^{\text{rep}}_j + \sqrt{2D_I}\,\boldsymbol{\zeta}_j\;}
$$

$\alpha \equiv \chi_s$ is the immunosuppression strength — one of the two control parameters of the phase diagram, the other being $\rho_I = N_I(0)$.

**Macrophages** carry a continuous polarization scalar $p_k \in [-1,+1]$ (M2 to M1). Position dynamics mirror T cells with weaker chemotaxis ($v_M \approx 0.2\,v_I$, $D_R^M \approx 3\,D_R^I$, $\chi_a^{(M)} = 8$). Polarization relaxes toward a local equilibrium set by suppressant and IL-10, with $b_{M_1}$ as the treatment knob (CSF1R inhibitor / TLR agonist / anti-CD47):

$$
\boxed{\;\frac{dp_k}{dt} = \frac{p_{\text{eq}}(\mathbf{R}_k^{(M)}) - p_k}{\tau_p} + \sqrt{2D_p}\,\zeta_k, \qquad p_{\text{eq}}(\mathbf{x}) = \tanh\!\big(b_{M_1} - \kappa_s c_s(\mathbf{x}) - \kappa_{\text{IL}} c_{\text{IL10}}(\mathbf{x})\big)\;}
$$

The $-\kappa_{\text{IL}}c_{\text{IL10}}$ term is **positive feedback**: once IL-10 builds up, M2 self-reinforces — the mechanism behind clinical TAM resistance.

**M1 phagocytosis.** For each macrophage with $p_k > 0$ within $r_{\text{phag}}$ of an alive tumor cell, the closest tumor is removed with probability

$$
p_{\text{phag,eff}} = p_{\text{phag}}\,\frac{1+p_k}{2}.
$$

**Reaction–diffusion fields.** Three scalars on a $G\times G$ grid. Range asymmetry $D_a \gg D_s$ ($\ell_a \approx 7\sigma$, $\ell_s \approx 2.2\sigma$) is the defining physics — long-range sensing, short-range exclusion:

$$
\partial_t c_a = D_a\nabla^2 c_a + s_a\,\rho_T - \lambda_a c_a
$$

$$
\partial_t c_s = D_s\nabla^2 c_s + s_s\,\rho_T - \lambda_s c_s
$$

$$
\partial_t c_{\text{IL10}} = D_{\text{IL10}}\nabla^2 c_{\text{IL10}} + s_{\text{IL10}}\,\rho_{M_2} - \lambda_{\text{IL10}} c_{\text{IL10}}, \qquad \rho_{M_2}(\mathbf{x}) = \sum_k \max(0,-p_k)\,\delta(\mathbf{x}-\mathbf{R}_k^{(M)})
$$

Only M2-skewed macrophages contribute to the IL-10 source. Tumor density $\rho_T$ is deposited via bilinear (cloud-in-cell) weights, mass-exact: $w_{p,q} = (1-f_x)^{1-p}f_x^p(1-f_y)^{1-q}f_y^q$.

**Killing rule.** For each alive T cell $j$: find the nearest alive tumor $i^\star(j)$ under minimum-image distance; if $\lVert\mathbf{R}_j - \mathbf{r}_{i^\star}\rVert < r_{\text{kill}}$ and $U\sim\mathcal{U}[0,1) < p_{\text{kill}}$, remove $i^\star$. Iteration is over T cells, so kill rate $\propto N_I$ not $N_T$. At most one kill per T cell per step. $p_{\text{kill}} \approx 0.10$–$0.12$ matches Weigelin (2021) "additive cytotoxicity" — ~3 sub-lethal hits at ~10% apoptosis each.

### Model 1 modular extensions

**ECM density + MMP** (`sim_ecm.py`). Two new fields with tumor-sourced MMP degrading and recovering ECM:

$$
\partial_t m = D_m\nabla^2 m + s_m\,\rho_T - \lambda_m m, \qquad \partial_t \rho_E = -k_{\text{deg}}\,m\,\rho_E + k_{\text{rec}}\,\rho_E(\rho_{E,0} - \rho_E)
$$

ECM acts back on tumor motility as a drag + pore-size cutoff (Wolf–Friedl):

$$
\dot{\mathbf{r}}_i = \frac{1}{1 + \beta\rho_E(\mathbf{r}_i)}\big[v_T\mathbf{n}_i + \mathbf{F}_i\big] + \text{noise}, \qquad v_{\text{eff}} = 0 \text{ if } r_p = r_0/\sqrt{\rho_E} < r_p^\star \text{ and } m < m^\star
$$

**Cadherin adhesion** (`sim_adhesion.py`). Morse-like attractive shoulder out to $\sigma_{\text{adh}} > \sigma$, tumor–tumor only:

$$
\mathbf{F}_{cc}(\mathbf{d}) = k_{\text{rep}}\,\max(0,\sigma-\lVert\mathbf{d}\rVert)\,\hat{\mathbf{d}} \;-\; k_{\text{adh}}\,\max(0,\lVert\mathbf{d}\rVert-\sigma)\,\max(0,\sigma_{\text{adh}}-\lVert\mathbf{d}\rVert)\,\hat{\mathbf{d}}
$$

Optional Vicsek torque on cadherin neighbors:

$$
\dot\theta_i \mathrel{+}\!= \frac{J_{\text{align}}}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i}\sin(\theta_j-\theta_i), \qquad \mathcal{N}_i = \{j:\sigma < |\mathbf{r}_{ij}| < \sigma_{\text{adh}}\}
$$

### Model 2 — biophysical TME

**Continuous EMT state** $s_{T,i} \in [0,1]$ per tumor cell (overdamped limit of Lu–Jolly–Levine–Onuchic bistable circuit):

$$
\boxed{\;\frac{ds_{T,i}}{dt} = \frac{s_{\text{eq}}(\mathbf{r}_i) - s_{T,i}}{\tau_s} + \sqrt{2D_{\text{EMT}}}\,\eta_i(t)\;}
$$

$$
s_{\text{eq}}(\mathbf{x}) = \text{clip}\!\big(k_{\text{hyp},s}\,H(\mathbf{x}) + k_{\text{ECM},s}\,\rho_E(\mathbf{x}) + k_{\text{supp},s}\,c_s(\mathbf{x}) - k_{\text{MET}},\; 0,\; 1\big)
$$

Every relevant cell-level parameter interpolates linearly in $s_{T,i}$:

$$
v_T(s) = (1-s)\,v_{\text{epi}} + s\,v_{\text{mes}}
$$

$$
k_{\text{adh}}(s) = (1-s)\,k_{\text{adh}}^0 \qquad \text{(cadherin lost as }s\to 1\text{)}
$$

$$
s_m(s,H) = (1-s)\,s_m^{\text{epi}} + s\,s_m^{\text{mes}} + k_{\text{MMP,hyp}}\,H
$$

$$
p_{\text{div}}(s,H) = p_{\text{div}}^0\,(1-s)\,(1 - k_{\text{div,hyp}}\,H) \qquad \text{(growth arrest for mesenchymal/hypoxic)}
$$

Daughters inherit $s_{T,i'} = s_{T,i} + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,\sigma_{d,\text{EMT}}^2)$.

**ECM fiber alignment + integrin biphasic traction**. A pre-existing single-angle fiber field $\theta_E(\mathbf{x})$ (smoothed random field at $t=0$) drives a Vicsek torque on tumor headings:

$$
\boxed{\;\frac{d\theta_i}{dt} \mathrel{+}\!= J_{\text{fiber}}\,\sin\!\big(\theta_E(\mathbf{r}_i) - \theta_i\big) + \sqrt{2D_R^T}\,\xi_i\;}
$$

Tumor speed is gated biphasically by ECM density (DiMilla–Lauffenburger–Quinn), normalized so the peak is unity at $\rho_E = K_d$:

$$
\boxed{\;v_{\text{eff}}(s,\mathbf{r}_i) = v_T(s)\cdot 4\,\frac{\rho_E(\mathbf{r}_i)/K_d}{(1 + \rho_E(\mathbf{r}_i)/K_d)^2}\;}
$$

Too little ligand → no traction; too much → mechanically pinned. Combined with MMP degradation, cells stuck in dense matrix locally lower $\rho_E$ toward the optimum and escape.

**Hypoxia, VEGF, dynamic vasculature**. Oxygen is supplied by discrete vessel points $\{\mathbf{x}_v\}$ that the simulation grows:

$$
\boxed{\;\frac{\partial c_{O_2}}{\partial t} = D_O\nabla^2 c_{O_2} - q_T\,\rho_T - \lambda_O\,c_{O_2} + s_{O_2}^{\text{ves}}\sum_v \delta(\mathbf{x}-\mathbf{x}_v)\;}
$$

Hypoxia is a step indicator (avoiding a per-cell ODE):

$$
H(\mathbf{x}) = \mathbb{1}\!\big[c_{O_2}(\mathbf{x}) < O^\star\big] \in \{0,1\}
$$

Hypoxic regions soften killing — the "hypoxia rescues escape" mechanism:

$$
p_{\text{kill}}^{\text{eff}}(\mathbf{r}_i) = p_{\text{kill}}\,\big(1 - k_{\text{kill,hyp}}\,H(\mathbf{r}_i)\big)
$$

Hypoxia also boosts suppressant secretion ($c_s^{\text{secretion}} = s_s(1 + k_{\text{supp,hyp}}H)$), enters $s_{\text{eq}}$ in the EMT equation, and arrests division.

Hypoxic tumor cells secrete VEGF:

$$
\frac{\partial c_{\text{VEGF}}}{\partial t} = D_V\nabla^2 c_{\text{VEGF}} + s_V^{\text{hyp}}\sum_i H(\mathbf{r}_i)\,\delta(\mathbf{x}-\mathbf{r}_i) - \lambda_V c_{\text{VEGF}}
$$

Existing vessels sprout daughters with probability $r_{\text{sprout}}$ per step when $c_{\text{VEGF}} > c_V^\star$; vessels drift up the VEGF gradient:

$$
\boxed{\;\frac{d\mathbf{x}_v}{dt} = \chi_V\,\nabla c_{\text{VEGF}}(\mathbf{x}_v), \qquad \mathbf{x}_{v'} := \mathbf{x}_v + \epsilon\,\hat{\mathbf{e}}_\perp \text{ at sprouting}\;}
$$

Closing the **angiogenic feedback loop**:

$$
\text{tumor growth} \to \text{hypoxic core} \to \text{VEGF} \to \text{vessel sprouting} \to \text{O}_2 \text{ supply restored} \to \text{tumor regrows}
$$

**Multi-species TME** (`sim_tme.py`):

- **NK cells** — fast, persistent; chemotax up $c_a$ but **not** suppressed by $\alpha$; separate kill kernel with $r_{\text{kill}}^{\text{NK}}, p_{\text{kill}}^{\text{NK}}$; hypoxia softens NK kills too.
- **Dendritic cells (DC)** — engulf tumor at rate $p_{\text{eat}}^{\text{DC}}$ and add a CD8-recruit term biasing local T-cell influx (proxy for lymph-node priming).
- **MDSCs** — slow, weakly motile; source $c_s$ directly via per-cell rate $s_{\text{MDSC}}$.
- **CAFs** — *stationary*, initialized in an annulus $[r_{\text{caf}}^{\text{in}}, r_{\text{caf}}^{\text{out}}]$ around the tumor seed; each deposits ECM into $\rho_E$ at rate $s_E^{\text{caf}}$, producing the desmoplastic ring that **structurally** excludes T cells regardless of chemotaxis strength.

### Coupling graph

The implemented Model 2 ties seven fields and four cell-level state variables into one feedback graph:

$$
\rho_T \to c_a, c_s \to \text{chemotaxis on CD8, NK, M}\phi \to \text{kills, M2 drive} \to c_{\text{IL10}}, c_s^{\text{eff}}
$$

$$
\rho_T \to O_2 \text{ depletion} \to H \to s_T, c_{\text{VEGF}}, p_{\text{kill}}^{\text{eff}} \to \text{vessels}, m, \rho_E, \theta_E \to v_{\text{eff}}
$$

## Order parameter

Primary observable is the final tumor fraction, geometric-mean over $S=3$ seeds:

$$
\Phi(\rho_I,\alpha) = \text{clip}\!\left(\frac{N_T(T_f)}{N_T(0)},\,10^{-2},\,10^{+2}\right), \qquad \bar\Phi = \exp\!\left(\tfrac{1}{S}\sum_s \log\Phi^{(s)}\right)
$$

Time traces $N_T(t), N_I(t), N_M(t), \langle p_k\rangle(t)$ separate the three phases — clearance (monotone decay), control/dormancy (long plateau), escape (saturating growth).

## Numerical methods

- **Particle integration (Euler–Maruyama)** with $\Delta t = 0.01$, PBC via minimum-image wrap:
  $$\mathbf{r}_i^{n+1} = \mathbf{r}_i^n + \Delta t\,[v_{T,i}\mathbf{n}_i + \mathbf{F}_i] + \sqrt{2D_T\Delta t}\,\boldsymbol{\xi}_i^n$$
- **Field integration (FTCS + auto-substep).** Five-point Laplacian; CFL $\nu = D_{\max}\Delta t/(\Delta x)^2$ targeted at $\le 1/4$ via $n_{\text{sub}} = \max(1,\lceil 4\nu\rceil)$.
- **Particle ↔ grid:** bilinear cloud-in-cell deposit for $\rho_T$; bilinear sampling for $\nabla c$ — exactly mass-consistent dual.
- **Minimum-image PBC** $d_x \leftarrow d_x - L\cdot\text{round}(d_x/L)$ inside every pairwise loop.
- **Performance.** Single `@njit(cache=True, fastmath=True)` inner step. Naive $O(N^2)$ pair loop is ~1 ms/step at $N \lesssim 1500$ on one core. 192-run sweep via `joblib.Parallel(n_jobs=-1, backend="loky")` in ~10 min on 16 cores.

## Production parameter defaults (Model 1)

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $L, G$ | `L, G` | 100, 64 | box side, field grid |
| $\Delta t, T_f$ | `dt, T_final` | 0.01, 100 | timestep, simulated time |
| $\langle v_T\rangle$, $\text{CV}_{v_T}$ | `v_T_mean, v_T_cv` | 0.1, 0.30 | tumor speed mean, log-normal CV |
| $\langle D_R^T\rangle$ | `D_R_T_mean` | 0.1 | tumor rotational diffusion |
| $\sigma, k_{\text{rep}}$ | `sigma_T, k_rep_T` | 1.0, 30 | cell diameter, repulsion stiffness |
| $p_{\text{div}}^0, P^\star$ | `p_div, P_star` | 0.004, 8.0 | division rate, pressure-gate threshold |
| $\langle v_I\rangle$ | `v_I_mean` | 1.0 | T-cell speed |
| $\chi_a$ | `chi_a` | 20.0 | attractant chemotaxis |
| $\alpha=\chi_s$ | `chi_s` | scanned | suppressant chemotaxis |
| $v_M, \chi_a^{(M)}$ | `v_M, chi_a_M` | 0.2, 8.0 | macrophage speed, chemotaxis |
| $\tau_p, \kappa_s, \kappa_{\text{IL}}$ | `tau_p, kappa_s, kappa_il` | 15.0, 8.0, 4.0 | polarization timescale + drives |
| $p_{\text{phag}}$ | `p_phag` | 0.10 | M1 max phagocytosis rate |
| $D_a, D_s, D_{\text{IL10}}$ | — | 5.0, 0.5, 4.0 | field diffusivities |
| $s_a, s_s, s_{\text{IL10}}$ | — | 1.0, 1.0, 1.0 | source rates |
| $\lambda_{\text{all}}$ | — | 0.1 | field decay |
| $r_{\text{kill}}, p_{\text{kill}}$ | `r_kill, p_kill` | 1.5, 0.12 | kill geometry, rate |
| $N_T(0)$ | `N_T_initial` | 50 | seed tumor |
| $\rho_I = N_I(0)$ | `N_I_initial` | scanned | seed T cells |
| $N_M(0)$ | `N_M_initial` | 80–120 | seed macrophages |

**Dimensionless groups:** $\text{Pe}_T = \text{Pe}_I = 1$ (persistence ≈ one cell diameter); $\ell_a/\ell_s \approx 3.2$ (the asymmetry that drives phase separation); $\alpha/\chi_a$ scans 0 → 1.25.

Five parameters were tuned away from spec defaults for the 8-hour budget: $\chi_a$ 5 → 20, $p_{\text{kill}}$ 0.05 → 0.12, $p_{\text{div}}$ 0.005 → 0.004, $T_f$ 200 → 100, $N_T^{\max}$ 4000 → 800. See `docs/DECISIONS.md`; literature calibration in `docs/calibration_research.md`.

## Featured videos

- [TME full](outputs/video_for_presentation/tme_full.mp4) — full-stack demo with every mechanism on: EMT, ECM fiber alignment + integrin biphasic traction, hypoxia / HIF response, angiogenesis, and the NK / DC / MDSC immune populations layered on top of the CD8 + tumor base.
- [TME baseline](outputs/video_for_presentation/tme_baseline.mp4) — biophysical control with EMT, fiber/integrin, and hypoxia all switched OFF (same seed, same ECM and immune setup), so any structure in the other runs can be attributed to those three mechanisms.
- [TME only EMT](outputs/videos/tme_only_emt.mp4) — only the EMT switch is on. Tumor cells can transition between epithelial and mesenchymal states (different motility, secretion, and daughter drift), while fiber/integrin coupling and hypoxia signaling stay off.
- [TME with hypoxia](outputs/videos/tme_with_hypoxia.mp4) — full HIF response engaged: low O₂ slows tumor division, reduces CD8 / NK kill probability, makes hypoxic tumor cells secrete VEGF, and lets vessels drift up the VEGF gradient and sprout. Paired with `tme_no_hypoxia.mp4` as the "same plumbing, no signaling" control.

## Headline results

| Figure | What is varied | Key observation |
|---|---|---|
| `phase_diagram.png` | $(\rho_I,\alpha)$ sweep, 192 runs | **Sharp, bistable** transition — narrow clearance wedge, thin dormancy band, escape dominates. Predicts checkpoint response as a **threshold**, not a dose curve |
| `phase6_combo_treatment.png` | Anti-PD-1 ($\alpha\to 0$) ± TAM repol ($b_{M_1}\to 15$) at $t=20$ | TAM repol alone **fails** — IL-10 has built up and the $-\kappa_{\text{IL}}c_{\text{IL10}}$ feedback locks M2. Only **combo** reliably crosses the boundary |
| `phase6_macrophage_polarization.png` | Untreated 3-species, $\langle p_k\rangle$ vs $t$ | $\langle p_k\rangle$ swings from 0 to $\approx-0.3$ as tumor grows — polarization is **spatially slaved** to the tumor's chemical halo |
| `phase6_heterogeneity_pressure.png` | Pressure-gated $p_{\text{div}}(P_i)$ | Core cells reach $P_i \approx P^\star$ and stop dividing; surface cells continue. Dormancy becomes a genuine steady state |
| `phase6b_ecm.png` | $\rho_{E,0}\times s_m$ (3 cases) | Dense ECM, MMP off ⇒ suppressed to ~660 cells. Dense ECM, MMP on ⇒ full escape with depleted-ECM **cavity** around the tumor (Friedl trail-blazing) |
| `phase6c_adhesion.png` | $k_{\text{adh}}\times J_{\text{align}}$ | No adhesion ⇒ diffuse cloud (NN $1.21\sigma$). Weak cadherin ⇒ cohesive blob ($0.99\sigma$). Strong + alignment ⇒ hexagonal sheet |
| `tme_panel.png` | Full 8-species TME with CAFs and vessels | CAFs form an annular ring; T cells blocked at the ring → **structural** immune exclusion. 83.8% hypoxic at $t=90$ |
| `tme_hypoxia_compare.png` | Hypoxia couplings ON vs OFF (same plumbing) | ON: 81.6% hypoxic, vessels sprout into hypoxic regions. OFF: 0% hypoxic. Hypoxia is the **resistance reservoir** |
| `biophysical_panel.png` | $s_{T,i}$ + fibers + hypoxia together | $\langle s_T\rangle$ saturates near 1 by $t \approx 40$; mesenchymal fraction → 100%; max invasion distance grows near-linearly |
| `biophysical_mechanism_compare.png` | 2×2 ablation of EMT / fibers / hypoxia | Each is necessary, none alone is sufficient — **EMT enables, fibers guide, hypoxia hardens** |

Full table with biological interpretation in `docs/simulation_summary.md`.

## Testable predictions

- **Sharp $\rho_I$ threshold for anti-PD-1 response.** Above the threshold, monotherapy works; below, combo is needed. Matches Tumeh (2014) baseline-TIL data.
- **Timing of combo therapy matters.** If TAM repol fires *after* significant $c_{\text{IL10}}$ has built up, the IL-10 self-feedback fights the M1 bias and the combo may fail. Early combo > sequential.
- **Non-monotone invasion in $\rho_E$.** Stromal density should have an optimum (consequence of biphasic integrin traction).
- **Hypoxic-fraction-dependent combo failure.** Tumors with high HIF should be relatively combo-resistant — the hypoxia kill penalty and MMP boost are not addressed by either drug class.

## Project layout

```
src/                 simulation kernels (numba-jitted)
  sim.py             minimal 2-species + 2-field
  sim_extended.py    + macrophages, pressure-gated proliferation, heterogeneity
  sim_ecm.py         + MMP and ECM density
  sim_adhesion.py    + cadherin + Vicsek alignment
  sim_biophysical.py + EMT scalar, integrin traction, fiber alignment
  sim_tme.py         + NK, DC, MDSC, CAFs, vessels, O2, VEGF
  sim_combined.py    everything stacked
  fields.py          RD field stepping (FTCS + auto-substep)
  interactions.py    pairwise force / killing kernels
  render*.py         matplotlib + ffmpeg rendering
  sweep.py           joblib-parallel parameter sweeps
  style.py           plotting style
scripts/             phase 1–9 driver scripts (one per experiment)
notebooks/           01 validation → 04 treatment experiment
docs/                model.md/.tex/.pdf, simulation_summary.md, extensions.md, DECISIONS.md, PLAN.md, calibration_research.md
outputs/             figures/, videos/, video_for_presentation/, data/
overleaf/            LaTeX sources for the writeup
slides/              presentation assets
```

## Reproducing

```bash
python scripts/smoke_test.py            # ~10s sanity check
python scripts/phase1_validation.py     # single-species validation
python scripts/phase3_plot.py           # phase diagram (192 runs)
python scripts/phase4_treatment.py      # treatment experiment
python scripts/phase7_tme_video.py      # full TME demo video
python scripts/phase7c_hypoxia_compare.py    # hypoxia ON vs OFF
python scripts/phase8b_mechanism_compare.py  # EMT / fiber / hypoxia ablation
python scripts/phase9_presentation_set.py    # presentation render set
```

All scripts dump to `outputs/`. See `docs/PLAN.md` for the phase-by-phase narrative.

## Limitations

2D, periodic, minimal. Model 2 captures vasculature dynamically but still abstracts away: lymph-node trafficking, antigen heterogeneity / immunoediting, dendritic-cell priming as a true adaptive-immunity loop (we use a local recruit term), clonal evolution, 3D tumor geometry. The full nematic $\mathbf{Q}$ remodeling (Maxwell-relaxation form) is implemented in `sim_biophysical.py` but disabled by default ($k_a = 0$) — the pre-existing fiber field already gives the contact-guidance phenomenology.
