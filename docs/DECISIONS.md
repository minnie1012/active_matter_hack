# Physics & implementation decisions

Low-stakes defaults chosen without asking the user, per their standing
preference. Each entry: the choice + one sentence on why.

## Forces & dynamics

- **Harmonic repulsion** `F = k·max(0, σ−r)·r̂` rather than WCA. Cheaper to evaluate, gentler at small overlaps, less risk of NaN at `dt = 0.01`.
- **Euler–Maruyama** integrator. Overdamped Langevin doesn't need higher order, and matches the form in the spec.
- **Periodic boundaries** with minimum-image distances. Spec lists periodic as the default; reflecting is a creative-hacking variant for later.

## Particles

- **Pre-allocated arrays + `alive` boolean mask** of size `N_max` (5000 tumor, 1000 T cell) rather than growable arrays. Numba can't resize arrays inside `@njit`; division writes into the first dead slot found by linear scan, which is fine at this N.
- **Rotational diffusion** updates the heading angle `θ` directly: `θ ← θ + sqrt(2·D_R·dt)·N(0,1)`, with `n = (cos θ, sin θ)`. Simpler than evolving `n` as a unit vector with projection.
- **Daughter cell placement** at offset `0.3·σ_T` in a uniformly random direction from the parent. Small enough that it overlaps the parent (forces resolve overlap in subsequent steps), big enough that they don't sit on top of each other and trigger a numerical singularity.
- **Carrying-capacity gate** on proliferation: a tumor cell only divides if the count of tumor neighbors within `1.5·σ_T` is below a threshold (default 6, ~hexagonal packing). This is the spec's `rho_max` rendered locally.

## Fields

- **Explicit FTCS Euler** with automatic substepping when CFL ratio `D·dt/dx² > 0.25`.
- **Bilinear (CIC)** deposit of tumor positions to the grid for source terms — anti-aliases the source field, keeps the gradient smooth.
- **Bilinear sampling** of the field gradient at T-cell positions for chemotaxis force — same reason.
- **Linear decay** `−λ·c` rather than Hill or Michaelis–Menten. Spec calls for `lambda_a = lambda_s = 0.1`; nonlinear decay adds parameters we don't need to sweep.

## Killing

- **Iterate over T cells** when applying the kill rule (per spec engineering note) so the rate scales with `N_Tcell`, not `N_tumor`.
- **One kill per T cell per step** at most: after a T cell kills a tumor cell, it doesn't get a second roll the same step. Prevents super-killers in dense regions.

## Numerics

- **Random seeding** via `numpy.random.default_rng(seed)`; we pre-draw the rotational noise increments for a chunk of steps to keep the JIT loop tight (no `rng` object inside `@njit`). For the inner loop we use `numba`'s built-in `np.random.normal()` which respects `np.random.seed` set outside.
- **Float dtype**: float64 throughout. Memory is not the bottleneck at our N.

## Visualization

- **Dark background** `#0A0E27`. Spec mandates this.
- **`EllipseCollection`** for particle rendering rather than `scatter`. Spec mandates and it's much faster for thousands of particles.
- **`imshow` + `gaussian_filter(σ=0.7 grid cells)`** for the suppressant field. Smooths visible pixelation from the coarse grid.
- **Right sidebar** plot of `N_tumor(t)` and `N_Tcell(t)` on twin axes per spec composition rules.
