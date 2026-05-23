# Model summary — Tumor–Immune Active Matter

**Project:** Vibe Coding Active Matter & Biophysics Hackathon, UCSD, 2026-05-23.
**Author:** I-Tsen Tsai (i3tsai@ucsd.edu).
**Code:** [`src/sim.py`](../src/sim.py), [`src/fields.py`](../src/fields.py), [`src/interactions.py`](../src/interactions.py).

This document collects every equation, parameter, numerical scheme, and assumption used in the simulation, in one place. It is intended to be read alongside the code — every symbol below has a name in `SimParams` you can grep for.

---

## 1.  Physical setup

A 2D periodic square box of side $L$ contains two species of self-propelled particles plus two scalar concentration fields.

| Symbol | Role | Code name |
|---|---|---|
| $\mathbf{r}_i(t)$ | tumor cell positions, $i = 1, \dots, N_T(t)$ | `pos_T` |
| $\theta_i(t)$ | tumor heading angles | `theta_T` |
| $\mathbf{R}_j(t)$ | T-cell positions, $j = 1, \dots, N_I(t)$ | `pos_I` |
| $\phi_j(t)$ | T-cell heading angles | `theta_I` |
| $c_a(\mathbf{x}, t)$ | attractant field (long range) | `c_a` |
| $c_s(\mathbf{x}, t)$ | suppressant field (short range) | `c_s` |
| $\rho_T(\mathbf{x}, t)$ | tumor density on the field grid | `rho` |

Both species are *non-conserved*: tumor cells are born by proliferation; tumor cells die when killed by T cells. T cells are conserved (no proliferation or natural death over the simulation timescale).

---

## 2.  Tumor-cell dynamics

Tumor cells are slow, soft-repulsive active Brownian particles with rotational diffusion and a local-density-gated birth rule. The overdamped Langevin equation of motion is

$$
\frac{d\mathbf{r}_i}{dt} \;=\; v_T\, \mathbf{n}_i \;+\; \sum_{j \ne i} \mathbf{F}_{\text{rep}}(\mathbf{r}_i - \mathbf{r}_j) \;+\; \sum_{k} \mathbf{F}_{\text{rep}}(\mathbf{r}_i - \mathbf{R}_k) \;+\; \sqrt{2 D_T}\;\boldsymbol{\eta}_i(t),
$$

where $\mathbf{n}_i = (\cos\theta_i, \sin\theta_i)$ is the unit heading vector, and

$$
\frac{d\theta_i}{dt} \;=\; \sqrt{2 D_R^T}\,\xi_i(t).
$$

$\boldsymbol{\eta}_i$ and $\xi_i$ are independent unit-variance Gaussian white noises.

### 2.1  Harmonic repulsion

$$
\mathbf{F}_{\text{rep}}(\mathbf{d}) \;=\; k_{\text{rep}}\,\max(0,\, \sigma - \lVert \mathbf{d} \rVert)\,\hat{\mathbf{d}}.
$$

The force vanishes outside the contact distance $\sigma$ and grows linearly with overlap inside. Harmonic (not WCA) is chosen because it is gentler at small overlaps and stable at our integration step $\Delta t = 0.01$ — see [`docs/DECISIONS.md`](DECISIONS.md).

### 2.2  Proliferation rule

At every step, each alive tumor cell $i$ attempts to divide independently with probability $p_{\text{div}}$ per step. The attempt succeeds only if the local density gate is satisfied:

$$
n_i \;\equiv\; \big|\{ j \ne i \;:\; \lVert \mathbf{r}_i - \mathbf{r}_j \rVert < r_{\text{nbr}} \}\big| \;<\; n_{\max}.
$$

If the gate passes, a daughter cell is placed at

$$
\mathbf{r}_{i'} \;=\; \mathbf{r}_i + 0.3\,\sigma\,(\cos\psi, \sin\psi), \qquad \psi \sim \mathcal{U}[0, 2\pi),
$$

with a fresh random heading. This represents a soft Allee/carrying-capacity effect: tumor cells deep inside the colony cannot divide because they are surrounded.

---

## 3.  T-cell dynamics

T cells are fast active Brownian particles with chemotaxis. They are coupled to both scalar fields:

$$
\boxed{\;\frac{d\mathbf{R}_j}{dt} \;=\; v_I\,\mathbf{m}_j \;+\; \chi_a\, \nabla c_a\big|_{\mathbf{R}_j} \;-\; \alpha\, \nabla c_s\big|_{\mathbf{R}_j} \;+\; \sum_{k \ne j} \mathbf{F}_{\text{rep}}(\mathbf{R}_j - \mathbf{R}_k) \;+\; \sum_{i} \mathbf{F}_{\text{rep}}(\mathbf{R}_j - \mathbf{r}_i) \;+\; \sqrt{2 D_I}\,\boldsymbol{\zeta}_j(t)\;}
$$

where $\mathbf{m}_j = (\cos\phi_j, \sin\phi_j)$ and the heading evolves as

$$
\frac{d\phi_j}{dt} \;=\; \sqrt{2 D_R^I}\, \mu_j(t).
$$

The key physical content is in the chemotactic terms:

* $+\chi_a \nabla c_a$ pulls T cells **up** the attractant gradient — toward tumor-secreted signals.
* $-\alpha\, \nabla c_s$ pushes T cells **down** the suppressant gradient — away from the tumor's immunosuppressive halo.

$\alpha \equiv \chi_s$ is the immunosuppression strength and is one of the two control parameters of the phase diagram. The other is $\rho_I \equiv N_I(0)$.

---

## 4.  Reaction–diffusion fields

Both fields obey a diffusion–source–decay PDE on a $G \times G$ grid covering the periodic box. The tumor density $\rho_T$ acts as the source for both:

$$
\frac{\partial c_a}{\partial t} \;=\; D_a\,\nabla^2 c_a \;+\; s_a\,\rho_T(\mathbf{x}, t) \;-\; \lambda_a\, c_a,
$$

$$
\frac{\partial c_s}{\partial t} \;=\; D_s\,\nabla^2 c_s \;+\; s_s\,\rho_T(\mathbf{x}, t) \;-\; \lambda_s\, c_s.
$$

The crucial asymmetry is $D_a \gg D_s$ ($D_a = 5$, $D_s = 0.5$): attractant diffuses far, suppressant stays local to the tumor mass. This separates spatial scales — T cells can sense the tumor from across the box (long-range $c_a$), but only feel repulsion close in (short-range $c_s$). The competition between these two ranges is what generates the phase diagram.

The tumor density on the grid is computed by bilinear ("cloud-in-cell") deposition. Each tumor cell $i$ contributes weight

$$
\rho_T(g_x, g_y) \mathrel{+}= \frac{(1 - f_x)(1 - f_y)}{(\Delta x)^2}, \quad \dots
$$

to its four surrounding grid corners, where $f_x, f_y$ are the fractional offsets and $\Delta x = L/G$ is the grid spacing. Total tumor mass is conserved exactly: $\int \rho_T\,dA = N_T$.

---

## 5.  Killing rule

For each alive T cell $j$, find its nearest alive tumor cell $i^\star(j)$ under minimum-image distance. If

$$
\lVert \mathbf{R}_j - \mathbf{r}_{i^\star} \rVert \;<\; r_{\text{kill}}
\qquad \text{and} \qquad U \sim \mathcal{U}[0,1) \;<\; p_{\text{kill}},
$$

then tumor cell $i^\star$ is removed (its `alive` flag set to false). The iteration is over T cells, **not** over tumor cells — per the spec engineering note — so the kill rate scales with $N_I$, not $N_T$. At most one kill per T cell per step (no super-killers in dense regions).

---

## 6.  Numerical methods

### 6.1  Time integration (Euler–Maruyama)

For every particle position, one step of the Euler–Maruyama scheme reads

$$
\mathbf{r}_i^{n+1} \;=\; \mathbf{r}_i^n + \Delta t\,\big[\,v_T\, \mathbf{n}_i + \mathbf{F}_i\big] + \sqrt{2 D_T \Delta t}\,\boldsymbol{\xi}_i^{n},
$$

with $\boldsymbol{\xi}_i^n \sim \mathcal{N}(0, I)$ i.i.d. across steps and particles. Positions are wrapped modulo $L$ to enforce periodicity.

### 6.2  Field integration (FTCS + auto-substep)

The five-point Laplacian stencil on the periodic grid:

$$
\nabla^2 c\,\big|_{i,j} \;=\; \frac{c_{i+1,j} + c_{i-1,j} + c_{i,j+1} + c_{i,j-1} - 4\, c_{i,j}}{(\Delta x)^2}.
$$

We use explicit FTCS Euler with automatic subcycling whenever the diffusion CFL ratio threatens stability:

$$
\nu \;\equiv\; \frac{D_{\max}\, \Delta t}{(\Delta x)^2}, \qquad n_{\text{sub}} \;=\; \max\!\left(1,\; \left\lceil 4\,\nu \right\rceil\right), \qquad \Delta t_{\text{sub}} = \Delta t / n_{\text{sub}}.
$$

This keeps the per-substep ratio at or below $0.25$, well inside the stability region $\nu \le 1/2$ of explicit FTCS.

### 6.3  Bilinear gradient sampling

To evaluate $\nabla c$ at the off-grid position of a T cell, we (i) compute the central-difference gradient at every grid node, (ii) bilinearly interpolate from the four surrounding nodes. This is the dual of the CIC deposit and is exactly consistent with it.

### 6.4  Minimum-image distance under PBC

$$
d_x \leftarrow d_x - L\cdot\text{round}(d_x / L), \qquad d_y \leftarrow d_y - L\cdot\text{round}(d_y / L).
$$

Applied inside every pairwise force loop.

### 6.5  Performance

* Single `@njit(cache=True, fastmath=True)` inner step.
* Naive $O(N^2)$ pairwise force; at our $N \lesssim 1500$ this is ~1 ms / step on one core.
* Cell list ready to swap in if N grows; not needed at current parameters.
* `joblib.Parallel(n_jobs=-1, backend="loky")` for the 192-run sweep.

---

## 7.  Parameter table (current production defaults)

| Symbol | Code name | Value | Meaning |
|---|---|---|---|
| $L$ | `L` | 100 | box side length |
| $G$ | `G` | 64 | field grid resolution |
| $\Delta t$ | `dt` | 0.01 | timestep |
| $T_f$ | `T_final` | 100 | total simulated time |
| $v_T$ | `v_T` | 0.1 | tumor self-propulsion speed |
| $D_R^T$ | `D_R_T` | 0.1 | tumor rotational diffusion |
| $D_T^T$ | `D_T_T` | 0.001 | tumor translational diffusion |
| $\sigma$ | `sigma_T` | 1.0 | repulsion cutoff (cell diameter) |
| $k_{\text{rep}}$ | `k_rep_T` | 30 | repulsion stiffness |
| $p_{\text{div}}$ | `p_div` | 0.004 | per-step division probability |
| $r_{\text{nbr}}$ | `nbr_radius` | 1.5 | neighbor-count radius for density gate |
| $n_{\max}$ | `nbr_threshold` | 6 | max neighbors allowed for division |
| $v_I$ | `v_I` | 1.0 | T-cell self-propulsion speed |
| $D_R^I$ | `D_R_I` | 1.0 | T-cell rotational diffusion |
| $D_T^I$ | `D_T_I` | 0.001 | T-cell translational diffusion |
| $\chi_a$ | `chi_a` | 20.0 | attractant chemotactic coupling |
| $\alpha = \chi_s$ | `chi_s` | scanned | suppressant chemotactic coupling |
| $D_a$ | `D_a` | 5.0 | attractant diffusion |
| $D_s$ | `D_s` | 0.5 | suppressant diffusion |
| $s_a$ | `s_a` | 1.0 | attractant secretion rate |
| $s_s$ | `s_s` | 1.0 | suppressant secretion rate |
| $\lambda_a$ | `lam_a` | 0.1 | attractant decay rate |
| $\lambda_s$ | `lam_s` | 0.1 | suppressant decay rate |
| $r_{\text{kill}}$ | `r_kill` | 1.5 | killing engagement radius |
| $p_{\text{kill}}$ | `p_kill` | 0.12 | per-T-cell per-step kill probability |
| $N_T(0)$ | `N_T_initial` | 50 | initial tumor seed |
| $\rho_I = N_I(0)$ | `N_I_initial` | scanned | initial T-cell count |

Five parameters were tuned **away** from the spec defaults to make the three-phase structure visible in the 8-hour time budget: $\chi_a$ 5 → 20, $p_{\text{kill}}$ 0.05 → 0.12, $p_{\text{div}}$ 0.005 → 0.004, $T_f$ 200 → 100, $N_T^{\max}$ 4000 → 800. See `docs/DECISIONS.md`.

---

## 8.  Order parameter

The primary phase-diagram observable is the final tumor fraction, clipped for log plotting:

$$
\Phi(\rho_I, \alpha) \;\equiv\; \operatorname{clip}\!\left(\,\frac{N_T(T_f)}{N_T(0)},\; 10^{-2},\; 10^{+2}\right),
$$

geometrically averaged over $S$ seeds:

$$
\bar{\Phi}(\rho_I, \alpha) \;=\; \exp\!\left(\frac{1}{S}\sum_{s=1}^{S} \log \Phi^{(s)}(\rho_I, \alpha)\right).
$$

Secondary observables: time-resolved $N_T(t), N_I(t)$ trajectories. These distinguish *clearance* (monotone fast decay), *control / dormancy* (long plateau at intermediate $N_T$), and *escape* (monotone fast rise to the array cap).

---

## 9.  Key dimensionless groups

A useful way to organize the parameters:

* **Péclet number for tumor:** $\mathrm{Pe}_T = v_T / (D_R^T \sigma) = 1$. Tumor persistence length is one cell diameter.
* **Péclet number for T cell:** $\mathrm{Pe}_I = v_I / (D_R^I \sigma) = 1$. T cells also have ~one-cell persistence; chemotaxis dominates their effective motion.
* **Attractant range:** $\ell_a = \sqrt{D_a / \lambda_a} \approx 7$ (≈ 7 cell diameters). Long compared to $\sigma$.
* **Suppressant range:** $\ell_s = \sqrt{D_s / \lambda_s} \approx 2.2$. Just a few cells.
* **Ratio:** $\ell_a / \ell_s \approx 3.2$. The asymmetry that enables phase separation.
* **Suppression-to-attraction ratio:** $\alpha / \chi_a$. At our $\chi_a = 20$, $\alpha = 0$ to $25$ scans this ratio from 0 to 1.25 — bracketing the regime where suppression starts to overwhelm attraction.

---

## 10.  Relation to known active-matter theory

The model lives in the same neighborhood as:

* **Toner–Tu** hydrodynamics of polar active matter (no proliferation).
* **Joanny–Prost–Ranft** *growing* active matter (proliferation as a non-conserved source term).
* **Keller–Segel** chemotaxis (the attractant feedback loop is the classical instability).

The novel physical content here is the *combination* of (i) long-range positive coupling via $c_a$ and (ii) short-range negative coupling via $c_s$ from the *same source*. That asymmetry is what generates the bistable phase boundary that maps onto the immune-excluded tumor histology.
