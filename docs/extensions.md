# Extension roadmap — toward a more realistic tumor–immune model

**Companion to** `docs/model.md` / `docs/model.tex`. The baseline is two ABP species + two RD fields + killing. Below: how each of the 10 factors in the user's table maps to a concrete modification of `src/sim.py` + `src/fields.py` + `src/interactions.py`, plus a macrophage extension, plus a prioritized roadmap.

Notation: subscripts $i$ over tumor, $j$ over T cells, $k$ generic; $\sigma$ = cell diameter; positions $\mathbf{r}_i$; existing fields $c_a$, $c_s$.

---

## Part A — the user's 10 factors, made concrete

### 1.  Cell–cell adhesion (cadherins)

**Equation change.** Augment the harmonic repulsion with an attractive shoulder out to $\sigma_{\text{adh}} > \sigma$:
$$
\mathbf{F}_{cc}(\mathbf{d}) \;=\; \underbrace{k_{\text{rep}}\,\max(0,\sigma-|\mathbf{d}|)\,\hat{\mathbf{d}}}_{\text{baseline}} \;-\; k_{\text{adh}}\,\max(0,|\mathbf{d}|-\sigma)\cdot\max(0,\sigma_{\text{adh}}-|\mathbf{d}|)\,\hat{\mathbf{d}}.
$$
Optional Vicsek-type aligning torque coupling on cadherin neighbors: $d\theta_i/dt \mathrel{+}= (J/N_i)\sum_{j\in\mathcal{N}_i}\sin(\theta_j-\theta_i)$.

**Code change.** Modify `pairwise_harmonic_force` in `src/interactions.py` to add the second piecewise term. One new parameter `k_adh`, one new cutoff `sigma_adh`.

**Phase-diagram impact.** Cohesive sheets resist T-cell infiltration mechanically → escape phase grows; the dormancy phase narrows. Cites the Bi/Manning shape-index jamming picture (Ilina et al., Nat Cell Biol 2020).

---

### 2.  Cell–ECM adhesion (integrins)

**Equation change.** Introduce a per-cell engaged-integrin variable $e_i \in [0,1]$ and an ECM density field $\rho_E(\mathbf{x},t)$ (initially uniform). Speed becomes biphasic in $e_i\rho_E$ (DiMilla–Lauffenburger–Quinn):
$$
v_{\text{eff}}(e_i,\rho_E) = v_{\max}\,\frac{e_i\,\rho_E}{K_d + e_i\,\rho_E},
\qquad
\dot e_i = k_{\text{on}}(1-e_i)\rho_E(\mathbf{r}_i) - k_{\text{off}}(\|\mathbf{F}_i\|)\,e_i.
$$
$k_{\text{off}}$ grows with traction (slip-bond); a catch-bond variant has $k_{\text{off}}$ initially decreasing with force.

**Code change.** New `e_T[i]` array per cell; modify the self-propulsion term in `_step_particles_two_species` to use `v_eff(e_T[i], rho_E[grid(pos_T[i])])`.

**Phase-diagram impact.** Self-propulsion becomes substrate-dependent. Escape is only possible inside an intermediate $\rho_E$ band — the escape *region* becomes a *band* in any $(\rho_E,\cdot)$ slice.

---

### 3.  ECM density / porosity

**Equation change.** $\rho_E$ acts as a friction-like drag and a hard pore-size cutoff:
$$
\gamma(\rho_E) = \gamma_0\,(1+\beta\rho_E),\qquad \dot{\mathbf{r}}_i = \frac{1}{\gamma(\rho_E)}\!\left[\,v_T\,\mathbf{n}_i + \mathbf{F}_i\,\right] + \boldsymbol\eta_i,
\qquad
v_{\text{eff}}=0 \text{ if } r_p(\mathbf{x}) < r_p^\star.
$$
Pore radius $r_p \propto 1/\sqrt{\rho_E}$; $r_p^\star$ is the nuclear cross-section.

**Code change.** New scalar field `rho_E` on the same grid as `c_a, c_s`. Drag factor applied inside the particle loop. No new equations to integrate if $\rho_E$ is static.

**Phase-diagram impact.** High $\rho_E$ shrinks escape and grows dormancy: a new "ECM-jammed" axis appears.

---

### 4.  ECM stiffness — durotaxis

**Equation change.** Young's-modulus field $E_m(\mathbf{x})$ couples in two places:
$$
\mathbf{F}_{\text{duro},i} \;=\; \chi_E\,\nabla E_m(\mathbf{r}_i),
\qquad
v_T(E_m) \;=\; v_0\,\frac{E_m}{E_\star+E_m}.
$$
Optionally $D_R^T(E_m) = D_{R,0}/(1+\kappa E_m)$ — stiffer substrate → longer persistence (Maiuri et al. 2015 universal coupling).

**Code change.** New field array `E_m`; new force term computed via `grad_field_at` (already implemented for $c_a, c_s$).

**Phase-diagram impact.** Durotactic drift adds an effective outward force that competes with $-\alpha\nabla c_s$. The escape phase grows along the stiffness-gradient axis (mirrors stromal stiffening at the tumor margin).

---

### 5.  ECM fiber alignment / contact guidance

**Equation change.** Nematic director field $Q(\mathbf{x})$ (traceless symmetric 2×2) + scalar order $S(\mathbf{x})$. Aligning torque + anisotropic mobility:
$$
\dot\theta_i = \gamma_{\text{align}}\,S(\mathbf{r}_i)\,\sin\!\big(2(\varphi_Q(\mathbf{r}_i)-\theta_i)\big) + \sqrt{2D_R^T}\,\xi_i,
$$
$$
\dot{\mathbf{r}}_i = \mathbf{M}(\mathbf{r}_i)\cdot(v_T\,\mathbf{n}_i + \mathbf{F}_i), \quad \mathbf{M} = M_\parallel\,\hat{\mathbf{e}}_\parallel\hat{\mathbf{e}}_\parallel + M_\perp\,\hat{\mathbf{e}}_\perp\hat{\mathbf{e}}_\perp,\ M_\parallel > M_\perp.
$$
Cells also deposit/align fibers: $\partial_t Q = -k_d Q + k_a\langle\mathbf{n}_i\otimes\mathbf{n}_i - \tfrac{1}{2}\mathbf{I}\rangle_{\text{local}}$ (slow Maxwell remodeling).

**Code change.** Two new grid fields (`Q_xx`, `Q_xy`). Anisotropic mobility makes the particle update tensor-valued — moderate refactor in `step_particles_two_species`.

**Phase-diagram impact.** Aligned ECM creates 1-D channels; finger/stream morphologies emerge. Compatible with the TACS-3 (Provenzano et al. 2006) radial-fiber motif. Escape boundary moves to higher $\alpha$ because guidance gives coherent escape routes.

---

### 6.  ECM degradation (MMPs)

**Equation change.** Add a third RD field $m(\mathbf{x},t)$ sourced by tumor:
$$
\partial_t m = D_m\nabla^2 m + s_m\,\rho_T - \lambda_m m,
\qquad
\partial_t \rho_E = -k_{\text{deg}}\,m\,\rho_E + k_{\text{rep}}\,\rho_E(\rho_{E,0}-\rho_E).
$$
Membrane-bound MT1-MMP variant: local sink at each cell, $\partial_t\rho_E|_{r_i} = -k_{\text{mt1}}\rho_E$ within $r_{\text{cut}}$.

**Code change.** Add `m` field to `step_fields`; coupled update of `rho_E` in the same routine.

**Phase-diagram impact.** Tumor bypasses the porosity gate of §3 → ECM-jammed phase contracts. Combined with §5, generates "trail-blazing" leader cells, matching Friedl single→collective plasticity.

---

### 7.  Self-propulsion speed heterogeneity

**Equation change.** Per-cell $v_{T,i}$ drawn from $\mathcal{LN}(\mu,\sigma_v^2)$ at birth, with daughter drift $v_{T,i'} = v_{T,i} + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,\sigma_{\text{daughter}}^2)$. EOM: replace $v_T\mathbf{n}_i$ with $v_{T,i}\mathbf{n}_i$.

**Code change.** Pass `v_T_arr` instead of scalar `v_T` to the inner loop. Trivial.

**Phase-diagram impact.** Broadens MIPS boundaries (Cates–Tailleur). The control/dormancy phase enlarges at the expense of clean escape: the population fragments into a slow core + fast invasive tail.

---

### 8.  Persistence / noise heterogeneity

**Equation change.** Per-cell $D_{R,i}\sim\Gamma(k,\theta)$. Optionally couple via the universal Maiuri relation: $D_{R,i} = D_0\exp(-v_{T,i}/v_\star)$.

**Code change.** `D_R_arr` array, otherwise identical to §7.

**Phase-diagram impact.** Long-tailed persistence distribution → rare highly-persistent cells dominate escape. Sharp phase boundary becomes a fuzzy crossover; critical $\alpha^\star$ shifts toward smaller values because escape is now governed by the upper quantile, not the mean.

---

### 9.  Chemotaxis to additional cytokines

**Equation change.** Promote $(c_a, c_s)$ to a vector field $\mathbf{c} = (c_1,\dots,c_K)$ with per-species per-receptor coupling matrix $\chi_{jk}$:
$$
\partial_t c_k = D_k\nabla^2 c_k + \sum_\alpha s_{k,\alpha}\,\rho_\alpha - \lambda_k c_k,
\qquad
\dot{\mathbf{R}}_j = v_I\mathbf{m}_j + \sum_k \chi_{jk}\nabla c_k + \mathbf{F}_j + \text{noise}.
$$
Tumor-immunology realization: $c_1$ = CXCL9/10 (CD8-recruiting), $c_2$ = CCL22 (Treg-recruiting), $c_3$ = TGF-β (suppressant), $c_4$ = IL-10 (M2-suppressant). Receptor saturation: $\chi_{jk}(c_k) = \chi_0/(1+c_k/K_k)$.

**Code change.** Generalize `step_fields` to a list of fields. The bilinear `grad_field_at` is unchanged per field.

**Phase-diagram impact.** $(\rho_I, \alpha)$ becomes a 2D slice of a higher-dim control manifold. Independent attractants compress the escape phase; antagonistic ones (CCL22, TGF-β) create new exclusion phases.

---

### 10.  Proliferation / crowding pressure

**Equation change.** Replace the discrete neighbor-count gate by a continuous mechanical-pressure gate (Byrne–Drasdo):
$$
P_i = \frac{1}{A_i}\sum_{j\in\mathcal{N}_i}\mathbf{r}_{ij}\cdot\mathbf{F}_{ij},
\qquad
p_{\text{div},i} = p_{\text{div},0}\,\max\!\big(0,\,1 - P_i/P^\star\big) + \gamma_{\text{apop}}\,\Theta(P_i - P_{\text{apop}}).
$$
Daughter placement biased against $\nabla P_i$ (newborns push into low-pressure regions, recovering Greenspan outward growth).

**Code change.** Compute $P_i$ inside the existing pairwise loop in `pairwise_harmonic_force`. Modify the proliferation gate in `_step_particles_two_species`.

**Phase-diagram impact.** Dormancy becomes genuinely stable (cells don't divide at the homeostatic core pressure). Escape boundary moves to higher $\alpha$ — cells must additionally relax mechanical pressure to expand.

---

## Part B — adding macrophages (top recommendation)

The single most-impactful third species. Adds a **second secreting population** so the suppressant is no longer slaved to tumor density — IL-10 can move and adapt with the immune infiltrate.

### State variables

New species $\mathbf{M}$ with positions $\mathbf{R}^{(M)}_k$, headings $\phi^{(M)}_k$, and a polarization scalar $p_k \in [-1, +1]$ ($-1$ = M2, $+1$ = M1). Two new fields:
- $c_{\text{TNF}}(\mathbf{x},t)$ — M1-secreted, short-range, anti-tumor.
- $c_{\text{IL10}}(\mathbf{x},t)$ — M2-secreted, long-range, anti-CD8 (acts like a second $c_s$).

### Dynamics

**Position and heading** — same overdamped Langevin form as T cells but with slower speed ($v_M \approx 0.2\,v_I$) and lower persistence ($D_R^M \approx 3\,D_R^I$). Weak chemotaxis up $c_a$ for recruitment.

**Polarization** —
$$
\dot p_k \;=\; -\frac{p_k}{\tau_p} + g\!\left[\,c_{\text{IFN}}(\mathbf{R}^{(M)}_k) - \kappa\,c_s(\mathbf{R}^{(M)}_k) - \kappa'\,c_{\text{IL10}}(\mathbf{R}^{(M)}_k)\,\right] + \sqrt{2D_p}\,\zeta_k(t).
$$
The local TME drives polarization: tumor-secreted suppressant pushes macrophages toward M2 (negative feedback that locks in the immunosuppressive niche).

**Cytokine secretion** —
$$
\partial_t c_{\text{TNF}}  = D_{\text{TNF}}\nabla^2 c_{\text{TNF}}  + s_{\text{TNF}}\,\rho_{M^+} - \lambda_{\text{TNF}}\,c_{\text{TNF}},
\qquad
\rho_{M^+}(\mathbf{x}) = \sum_k \max(0,\,p_k)\,\delta(\mathbf{x}-\mathbf{R}^{(M)}_k),
$$
and analogously for $c_{\text{IL10}}$ with source $\propto\max(0,-p_k)$.

**Phagocytosis** — for each M1-skewed macrophage ($p_k > 0$) within $r_{\text{phag}} = 2\sigma$ of an alive tumor cell: kill with rate $k_{\text{phag}}\cdot(1+p_k)/2$.

**CD8 coupling** — replace the baseline $-\alpha\nabla c_s$ in the T-cell equation by $-\alpha\nabla(c_s + c_{\text{IL10}})$.

### New control parameter and treatment knob

The natural new control is $s_{\text{IL10}}$ (macrophage M2-skewing strength). The treatment experiment becomes **macrophage repolarization**: at $t = T_f/2$, set $g \to -g$ so $p_k$ relaxes toward $+1$. CD8 should reinfiltrate as the IL-10 field collapses. This mirrors CSF1R inhibitors and TLR-agonist clinical drugs targeting TAMs.

### Estimated implementation cost

≈ 90 minutes (one new species mirror of the T-cell scaffolding, two new fields tacked onto `step_fields`, one new force term in the CD8 update, one new polarization update). All numba-jittable.

---

## Part C — alternative additional species (lower priority)

| Species | Main mechanism | New fields | New per-particle state | Why it's lower priority |
|---|---|---|---|---|
| **Tregs** | Contact + cytokine suppression of CD8 | $c_{\text{TGF}}$ | exhaustion counter on CD8 | Mostly duplicates $c_s$ effects |
| **NK cells** | Missing-self killing | (reuses $c_a$) | MHC-I level $m_k$ per tumor | Requires immunoediting dynamics; slow convergence |
| **MDSCs** | Arginine depletion → slows CD8/NK | $c_{\text{Arg}}$ | — | Mostly redundant with TAM-M2 |
| **CAFs** | ECM deposition + CXCL12 sequestration | $\rho_E$, $c_{\text{CXCL12}}$ | — | High-effort because needs the tensor-mobility ECM stack |
| **Dendritic cells** | Antigen presentation, CD8 recruitment | (reuses $c_a$ + boundary CD8 source) | mature/immature flag | Only useful with a lymph-node compartment |

Equations for each are in the research notes; happy to expand any of these on request.

---

## Part D — prioritized roadmap

Estimating wall-clock under the same hackathon constraints (numba inner loop, ~10 min sweeps).

| Order | Addition | Hours | Why now |
|---|---|---|---|
| 1 | **M2-polarized macrophages** (Part B) | 1.5 | Largest qualitative change per code delta; gives a second treatment knob (repolarization vs. checkpoint). |
| 2 | **Heterogeneous $v_T$ and $D_R$** (§7+8) | 0.5 | Trivial code change, broadens phases, makes the bimodal sweep more nuanced. |
| 3 | **Pressure-gated proliferation** (§10) | 1.0 | Replaces the cutoff artefact $n_{\max}$ with smooth Byrne–Drasdo; makes dormancy genuinely stable. |
| 4 | **MMP field + ECM density** (§3+§6) | 2.0 | Adds two related new fields together; cleanly handled by extending `step_fields`. |
| 5 | **Cadherin adhesion** (§1) | 0.5 | Modifies one existing function; unlocks collective-invasion morphology. |
| 6 | **Multi-cytokine vector field** (§9) | 1.5 | Refactor `step_fields` to take a list. Enables future Tregs, MDSCs cleanly. |
| 7 | Tregs, NK, CAFs, ECM stiffness, fiber alignment, EMT switch | many | Phase-2 work — each is publication-scale by itself. |

A reasonable next 4-hour sprint: items 1, 2, 3 — gives you (i) a moving second suppressor, (ii) realistic heterogeneity, (iii) a clean dormant fixed point. The combination should transform the bimodal phase diagram into a real three-band figure with a stable, broad control phase.

---

## Part E — selected references

**ECM / mechanobiology**
- Friedl & Gilmour, *Classifying collective cancer cell invasion*, Nat Cell Biol 2012.
- Ilina et al., *Cell-cell adhesion and 3D matrix confinement determine jamming*, Nat Cell Biol 2020.
- Wolf & Friedl, *Physical limits of cell migration*, J Cell Biol 2013.
- Sunyer et al., *Collective cell durotaxis*, Science 2016.
- Provenzano et al., *Collagen reorganization at the tumor-stromal interface (TACS-3)*, BMC Med 2006.
- Macklin et al., *PhysiCell*, PLOS Comp Biol 2018; PhysiCell-ECM (Metzcar et al., bioRxiv 2022).
- Anderson & Chaplain, *Continuous and discrete models of tumour invasion*, Bull Math Biol 1998.
- Maiuri et al., *Actin flows mediate a universal speed-persistence coupling*, Cell 2015.

**Pressure-gated growth & active matter**
- Byrne & Drasdo, *Individual-based and continuum models of growing cell populations*, J Math Biol 2009.
- Ranft, Prost, Jülicher et al., *Fluidization of tissues by cell division and apoptosis*, PNAS 2010.
- Cates & Tailleur, *Motility-induced phase separation*, Annu Rev Condens Matter Phys 2015.

**Tumor immunology — multi-species**
- Knudson et al., *Multi-scale ABM of macrophage polarization*, Front Immunol 2024.
- Italiani & Boraschi, *Continuous polarization framework*, Front Immunol 2014.
- Tanaka & Sakaguchi, *Tregs in solid tumor immunotherapy*, Cell Death Dis 2025.
- Vivier et al., *NK cell innate killing*, Science 2011.
- Gabrilovich, *MDSCs (Arg1)*, Cancer Res 2021.
- Kalluri, *CAFs*, Nat Rev Cancer 2016; Mariathasan et al., *TGF-β / CAF / immune exclusion*, Nature 2018.
- Feig et al., *FAP+ CAFs and CXCL12 exclusion*, PNAS 2013.

**EMT**
- Lu, Jolly, Levine, Onuchic, *Hybrid E/M state*, PNAS 2013.
- Pastushenko & Blanpain, *Hybrid E/M drives collective invasion*, Trends Cell Biol 2019.

---

*See also* `docs/PLAN.md` (project plan) · `docs/DECISIONS.md` (current parameter choices) · `docs/model.tex` (typeset equations).
