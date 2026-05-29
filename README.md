# active_matter_hack

Tumor–immune active-matter modeling for the UCSD **Vibe Coding Active Matter & Biophysics Hackathon** (2026-05-23).

**Authors:** I-Shan Tsai (i3tsai@ucsd.edu), Chih-Yen Liu (chl250@ucsd.edu).

## Background

A solid tumor is a dense, self-driven population of cells embedded in a structured extracellular matrix (ECM) and harassed by an immune infiltrate. Clinically, tumors fall into a famous "hot / excluded / cold" trichotomy: some are cleared by T cells, some grow despite a heavy infiltrate, and some never get infiltrated at all. We treat this as an **active-matter phase diagram** and ask which physical ingredients (motility, secretion, mechanics) actually move the boundary between phases.

The project builds two nested models:

1. **Minimal model** (`src/sim.py`) — two species of active Brownian particles (tumor, CD8 T cells) coupled to two reaction–diffusion fields (a long-range tumor-secreted attractant $c_a$ and a short-range suppressant $c_s$). This is enough to recover the clearance / dormancy / escape trichotomy as a sharp bistable phase diagram in $(\rho_I, \alpha)$ — initial T-cell count vs. suppressant strength.
2. **Full TME model** (`src/sim_tme.py`, `src/sim_biophysical.py`, `src/sim_combined.py`) — adds NK, DC, MDSC, M1/M2 macrophages, CAFs, and microvessels, plus four new fields (ECM density $\rho_E$, MMP $m$, oxygen $c_{O_2}$, VEGF $c_{\text{VEGF}}$) and per-cell EMT / integrin / hypoxia state. This lets the same code reproduce CAF-driven immune exclusion, hypoxia-driven EMT at the invasive front, and angiogenic rescue.

Every parameter and equation lives in `docs/model.md` (`docs/model.tex` / `.pdf` for typeset versions).

## Equations

### Minimal model

Tumor cells $i$ are slow soft-repulsive ABPs with density-gated birth; T cells $j$ are fast ABPs with two-channel chemotaxis:

$$
\frac{d\mathbf{r}_i}{dt} = v_T\,\mathbf{n}_i + \sum_{k\ne i}\mathbf{F}_{\text{rep}}(\mathbf{r}_i - \mathbf{r}_k) + \sqrt{2D_T}\,\boldsymbol{\eta}_i, \qquad \frac{d\theta_i}{dt} = \sqrt{2D_R^T}\,\xi_i
$$

$$
\frac{d\mathbf{R}_j}{dt} = v_I\,\mathbf{m}_j + \chi_a\,\nabla c_a - \alpha\,\nabla c_s + \sum_{k\ne j}\mathbf{F}_{\text{rep}} + \sqrt{2D_I}\,\boldsymbol{\zeta}_j
$$

Both fields obey diffusion–source–decay PDEs with tumor density as the source:

$$
\partial_t c_a = D_a\nabla^2 c_a + s_a\,\rho_T - \lambda_a c_a, \qquad \partial_t c_s = D_s\nabla^2 c_s + s_s\,\rho_T - \lambda_s c_s
$$

The **crucial asymmetry** is $D_a \gg D_s$ ($D_a=5$, $D_s=0.5$): attractant diffuses far, suppressant stays local. The competition between these two ranges is what generates the phase diagram. Killing: at each step, each T cell tests its nearest tumor neighbor and removes it if within $r_{\text{kill}}$ with probability $p_{\text{kill}}$. Tumor cells divide with probability $p_{\text{div}}$ per step, gated by a local-neighbor cap $n_{\max}$ (soft carrying capacity).

**Repulsion** is harmonic (gentler than WCA at $\Delta t = 0.01$):

$$
\mathbf{F}_{\text{rep}}(\mathbf{d}) = k_{\text{rep}}\,\max(0,\, \sigma - \lVert\mathbf{d}\rVert)\,\hat{\mathbf{d}}
$$

### Full TME model — additional ingredients

| Mechanism | Equation / coupling | Code |
|---|---|---|
| **EMT** | Per-cell scalar $s_{T,i}\in[0,1]$ driven by hypoxia + ECM + suppressant; modulates speed $v_T(s_T)$, secretion, cadherin loss | `sim_biophysical.py` |
| **Integrin biphasic traction** | $v_{\text{eff}} = v_{\max}\,e\rho_E/(K_d + e\rho_E)$, with engaged fraction $\dot e = k_{\text{on}}(1-e)\rho_E - k_{\text{off}}(F)\,e$ (slip-bond) | `sim_biophysical.py` |
| **ECM fiber alignment** | Nematic field $Q$, aligning torque $\dot\theta_i \mathrel{+}= \gamma_{\text{align}} S\sin(2(\varphi_Q - \theta_i))$, anisotropic mobility tensor | `sim_biophysical.py` |
| **Hypoxia + HIF response** | $\partial_t c_{O_2} = D_O\nabla^2 c_{O_2} - \text{consumption}_T\,\rho_T + \text{source from vessels}$; low O₂ slows division, reduces kill probability, drives EMT, boosts suppressant secretion | `sim_tme.py` |
| **Angiogenesis** | Hypoxic tumor secretes VEGF; vessels drift up $\nabla c_{\text{VEGF}}$ and sprout | `sim_tme.py` |
| **MMP / ECM remodeling** | $\partial_t m = D_m\nabla^2 m + s_m\rho_T - \lambda_m m$, $\partial_t\rho_E = -k_{\text{deg}}\,m\,\rho_E + \text{recovery}$ | `sim_ecm.py` |
| **Cadherin adhesion** | Attractive shoulder $-k_{\text{adh}}\max(0,\lVert\mathbf{d}\rVert-\sigma)\max(0,\sigma_{\text{adh}}-\lVert\mathbf{d}\rVert)\hat{\mathbf{d}}$ + optional Vicsek alignment | `sim_adhesion.py` |
| **Macrophage polarization** | $\dot p_k = -p_k/\tau_p + g[c_{\text{IFN}} - \kappa c_s - \kappa' c_{\text{IL10}}] + \sqrt{2D_p}\zeta_k$; M2 ($p<0$) secretes IL-10 ⇒ second suppressant | `sim_combined.py` |
| **Pressure-gated proliferation** | $p_{\text{div},i} = p_{\text{div}}^0\max(0, 1 - P_i/P^\star)$ (Byrne–Drasdo) | `sim_extended.py` |
| **CAFs** | Stationary stromal ring; deposit ECM, secrete CXCL12, physically block T cells | `sim_tme.py` |

## Order parameter and phase diagram

Primary observable is the final tumor fraction, geometric-mean over seeds:

$$
\bar\Phi(\rho_I, \alpha) = \exp\!\left(\tfrac{1}{S}\sum_s \log\,\text{clip}\!\left(\tfrac{N_T(T_f)}{N_T(0)},\;10^{-2},\;10^{+2}\right)\right)
$$

Time traces $N_T(t)$ separate the three phases: **clearance** (monotone decay), **dormancy / control** (long plateau), **escape** (saturating growth). The $(\rho_I, \alpha)$ sweep is $8\times 8\times 3$ seeds = 192 runs, parallelized with `joblib`.

## Numerical methods

- **Particle integration:** Euler–Maruyama with $\Delta t = 0.01$. PBC via minimum-image wrap.
- **Field integration:** FTCS five-point Laplacian with auto-subcycling at CFL $\nu = D_{\max}\Delta t/(\Delta x)^2 \le 1/4$.
- **Particle↔grid:** bilinear ("cloud-in-cell") deposit for $\rho_T$; bilinear sampling for $\nabla c$. The two are exactly mass-consistent.
- **Performance:** single `@njit(cache=True, fastmath=True)` inner loop; naive $O(N^2)$ pair loop is ~1 ms/step at $N\lesssim 1500$.

## Production parameter defaults (minimal model)

| Symbol | Code | Value | Meaning |
|---|---|---|---|
| $L$ | `L` | 100 | box side |
| $G$ | `G` | 64 | field grid |
| $\Delta t$ | `dt` | 0.01 | timestep |
| $T_f$ | `T_final` | 100 | total time |
| $v_T,\ v_I$ | `v_T, v_I` | 0.1, 1.0 | self-propulsion speeds |
| $D_R^T,\ D_R^I$ | `D_R_T, D_R_I` | 0.1, 1.0 | rotational diffusion |
| $\sigma$ | `sigma_T` | 1.0 | repulsion cutoff |
| $k_{\text{rep}}$ | `k_rep_T` | 30 | repulsion stiffness |
| $\chi_a$ | `chi_a` | 20.0 | attractant chemotaxis |
| $\alpha$ | `chi_s` | swept | suppressant chemotaxis |
| $D_a,\ D_s$ | `D_a, D_s` | 5.0, 0.5 | field diffusion |
| $\lambda_a,\ \lambda_s$ | `lam_a, lam_s` | 0.1, 0.1 | field decay |
| $p_{\text{div}}$ | `p_div` | 0.004 | per-step division prob |
| $n_{\max}$ | `nbr_threshold` | 6 | crowd cap |
| $r_{\text{kill}},\ p_{\text{kill}}$ | `r_kill, p_kill` | 1.5, 0.12 | kill geometry / rate |
| $N_T(0)$ | `N_T_initial` | 50 | seed tumor |
| $\rho_I = N_I(0)$ | `N_I_initial` | swept | seed T cells |

Dimensionless groups: $\mathrm{Pe}_T = \mathrm{Pe}_I = 1$ (persistence ≈ one cell diameter), attractant range $\ell_a = \sqrt{D_a/\lambda_a} \approx 7$, suppressant range $\ell_s \approx 2.2$, ratio $\ell_a/\ell_s \approx 3.2$ — the asymmetry that drives phase separation.

## Featured videos

- [TME full](outputs/video_for_presentation/tme_full.mp4) — full-stack demo with every mechanism on: EMT, ECM fiber alignment + integrin biphasic traction, hypoxia / HIF response, angiogenesis, and the NK / DC / MDSC immune populations layered on top of the CD8 + tumor base.
- [TME baseline](outputs/video_for_presentation/tme_baseline.mp4) — biophysical control with EMT, fiber/integrin, and hypoxia all switched OFF (same seed, same ECM and immune setup), so any structure in the other runs can be attributed to those three mechanisms.
- [TME only EMT](outputs/videos/tme_only_emt.mp4) — only the EMT switch is on. Tumor cells can transition between epithelial and mesenchymal states (different motility, secretion, and daughter drift), while fiber/integrin coupling and hypoxia signaling stay off.
- [TME with hypoxia](outputs/videos/tme_with_hypoxia.mp4) — full HIF response engaged: low O₂ slows tumor division, reduces CD8 / NK kill probability, makes hypoxic tumor cells secrete VEGF, and lets vessels drift up the VEGF gradient and sprout. Paired with `tme_no_hypoxia.mp4` as the "same plumbing, no signaling" control.

## Headline results

| Figure | Variable / mechanism | Key observation |
|---|---|---|
| `phase_diagram.png` | $(\rho_I, \alpha)$ sweep, 192 runs | Sharp bistable transition — narrow clearance wedge, thin dormancy band, escape dominates the rest |
| `phase6_combo_treatment.png` | Anti-PD-1 ($\alpha\to 0$) ± TAM repolarization ($b_{M1}\to 15$) at $t=20$ | Only the **combination** crosses the boundary in the dormancy region — TAM repol alone gets locked back by IL-10 self-feedback |
| `tme_panel.png` | Full 8-species TME with CAFs and vessels | CAFs form an annular ring; T cells are blocked at the ring → structural immune exclusion |
| `tme_hypoxia_compare.png` | Hypoxia couplings ON vs. OFF (same plumbing) | Hypoxia drives 80%+ hypoxic fraction and an angiogenic rescue loop; without it, oxygen stays uniform and EMT never fires |
| `biophysical_mechanism_compare.png` | 2×2 ablation of EMT / fibers / hypoxia | Each is necessary, none alone is sufficient — EMT *enables*, fibers *guide*, hypoxia *hardens* |

Full table with biological interpretation in `docs/simulation_summary.md`.

## Project layout

```
src/                 simulation kernels (numba-jitted)
  sim.py             minimal 2-species + 2-field
  sim_extended.py    + macrophages, pressure-gated growth
  sim_ecm.py         + MMP and ECM density
  sim_adhesion.py    + cadherin + Vicsek alignment
  sim_biophysical.py + EMT, integrin traction, fiber alignment
  sim_tme.py         + NK, DC, MDSC, CAFs, vessels, O2, VEGF
  sim_combined.py    everything stacked
  fields.py          RD field stepping (FTCS + auto-substep)
  interactions.py    pairwise force / killing kernels
  render.py / render_tme.py / render_presentation.py    matplotlib + ffmpeg
  sweep.py           joblib-parallel parameter sweeps
  style.py           plotting style
scripts/             phase 1–9 driver scripts (one per experiment)
notebooks/           01 validation → 04 treatment experiment
docs/                model.md/.tex/.pdf, simulation_summary.md, extensions.md
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
python scripts/phase8b_mechanism_compare.py  # EMT / fiber / hypoxia ablation
```

All scripts dump to `outputs/`. See `docs/PLAN.md` for the phase-by-phase narrative and `docs/DECISIONS.md` for parameter choices that were tuned away from spec defaults.
