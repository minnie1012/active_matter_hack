# Implementation Plan — Tumor–Immune Active Matter Hackathon

Spec source: `CLAUDE_tumor_immune.md` (the file is named with the `_tumor_immune` suffix; treating it as the canonical CLAUDE.md). Total budget: ~8 hrs on 2026-05-23.

---

## 0. Environment assumptions (FLAGGED — please verify)

- **Python is 3.14.0, not 3.11.** You wrote "assume Python 3.11" — your actual interpreter is 3.14.0. I verified numba 0.65 JITs cleanly on it (smoke test passed: a small `@njit` sum compiled and ran). I'll proceed on 3.14 unless you say otherwise.
- **`ffmpeg` is not on PATH.** Without it `matplotlib.animation.FFMpegWriter` fails. Fix: `pip install imageio-ffmpeg` (ships a vendored binary, no admin install) and point matplotlib at it via `mpl.rcParams['animation.ffmpeg_path']`. GIF fallback via Pillow is available if MP4 export still fails.
- **Project directory is not a git repo.** You asked for commits after each phase. I'll `git init` at the start of Phase 1, but will pause and confirm before doing so since it touches global state.
- **Windows shell.** I'll use forward slashes and Unix-style paths in code; multiprocessing on Windows requires the `if __name__ == "__main__":` guard everywhere — I'll honor that in `sweep.py`.
- **No GPU assumed.** Pure CPU numba; parallelizing across cores with joblib for the sweep.

---

## 1. Packages and minimum versions

Already installed and adequate (versions in your env shown for reference):

| Package | Installed | Minimum needed | Notes |
|---|---|---|---|
| numpy | 2.4.1 | ≥ 1.26 | 2.x is fine; numba 0.65 supports it. |
| numba | 0.65.0 | ≥ 0.60 | Required for the JITed inner loop. |
| scipy | 1.17.0 | ≥ 1.11 | For `ndimage.gaussian_filter`. |
| matplotlib | 3.10.8 | ≥ 3.8 | Static figs + animation. |
| joblib | 1.5.3 | ≥ 1.3 | Parallel sweep. |
| pillow | 12.1.0 | ≥ 10 | GIF fallback writer. |

**Needs install (you should run before I start Phase 1):**

```
pip install imageio-ffmpeg tqdm cmasher
```

- `imageio-ffmpeg` — vendored ffmpeg binary for MP4 export (required for slide-quality video; no admin needed).
- `tqdm` — progress bar for the sweep so we know if it's stuck.
- `cmasher` — perceptual colormaps better than matplotlib defaults; spec calls them out as "optional but recommended."

Tell me when those are installed and I'll start Phase 1.

---

## 2. Architectural decisions (my recommendations)

### 2.1 Numba vs pure NumPy → **numba, no fallback**
Spec mandates it and you confirmed it as a standing preference. Single `@njit(cache=True)` inner step that takes flat float64 arrays for both species. No Python objects crossing the boundary. `parallel=True` only if I can prove it helps on the benchmark — race-condition risk on the proliferation step is real.

### 2.2 Neighbor search: **naive O(N²) inside numba** for the hackathon, cell-list as a hot-swap if needed
Reasoning: with `N_tumor ≲ 2000` and `N_Tcell ≲ 500`, O(N²) inside `@njit` is ~5M ops/step. On modern CPU that's well under 5 ms/step → 100 s for 20 000 steps. The 5 s / 1000-step / 500-particle profiling gate you set should pass comfortably. If it doesn't, I have a uniform-grid cell-list ready as a drop-in.

### 2.3 Repulsion form: **harmonic** (`F = k * max(0, σ − r) * r̂`)
Cheaper, gentler, less stiff than WCA — won't blow up the integrator at default `dt=0.01`. Will log this in `docs/DECISIONS.md`.

### 2.4 Integrator: **Euler–Maruyama** at fixed `dt=0.01`
Spec already prescribes this; overdamped Langevin doesn't need anything fancier.

### 2.5 Particle storage with proliferation/death: **pre-allocated arrays + boolean `alive` mask**
Spec needs growth. Resizing arrays inside numba is awkward; growing Python lists kills perf. Plan: allocate `pos[N_max, 2]` with `N_max = 5000` for tumor, `1000` for T cells, and use `alive` masks. Division writes into the first dead slot found (single linear scan). If we hit `N_max` we cap proliferation (logged event, not a crash).

### 2.6 Periodic boundaries: **minimum-image distances + `np.roll`-based Laplacian**
Spec mandates periodic. I'll wrap positions with `mod L` each step.

### 2.7 Field stepping: **explicit FTCS Euler with automatic substep**
At the start of `step_fields`, compute the CFL ratio `D_max * dt / dx²` and substep by `ceil(4 * ratio)` if > 0.25. Cheap and safe.

### 2.8 Animation backend: **matplotlib `FFMpegWriter` via imageio-ffmpeg**
GIF via Pillow as a fallback. Frames built from `EllipseCollection` (spec mandates this over `scatter` for speed). Field as `imshow` with bilinear interp + `gaussian_filter` blur.

### 2.9 Sweep parallelism: **joblib `Parallel(n_jobs=-1, backend='loky')`**
Multiprocessing on Windows; `loky` survives the spawn-vs-fork issue. Each worker imports the numba module fresh and pays JIT cost once. Caching via `@njit(cache=True)` so subsequent workers reuse the AOT cache.

### 2.10 Output format: **`.npz` per sweep point + a single aggregated `phase_grid.npz`**
Easy to re-plot without re-running. Plus trajectory snapshots every 50 steps for one representative run per phase (for video).

---

## 3. Phase → files → time

### Phase 1 — Scaffolding & single-species ABP (target 90 min)

| File | Function(s) | Purpose |
|---|---|---|
| `src/style.py` | constants + `apply_style()`, `tumor_cmap`, `suppressant_cmap`, `attractant_cmap` | All visual constants. Dark bg `#0A0E27`, tumor `#E63946`, T-cell `#457B9D`. |
| `src/sim.py` | `@njit step_particles(...)`, `init_state(...)`, `run_single_species(...)` | Core JIT inner loop. Single-species ABP first. |
| `src/interactions.py` | `@njit pairwise_repulsion(...)` (called from `step_particles`) | Harmonic repulsion w/ minimum-image. |
| `notebooks/01_single_species_validation.ipynb` | — | MIPS sanity check + Fisher–KPP front from a seed. |
| `docs/DECISIONS.md` | — | Log harmonic vs WCA choice + others. |

**Profiling gate:** 500 ABPs × 1000 steps must finish in <5 s after JIT warm-up. If not, stop and optimize. **Validation artifact for me to show you:** a still frame of MIPS clustering at high Peclet (one PNG) and a 1-D radial density profile from a growing tumor seed (one PNG).

### Phase 2 — Two species, fields, killing (target 120 min)

| File | Function(s) | Purpose |
|---|---|---|
| `src/fields.py` | `@njit step_fields(c_a, c_s, rho_tumor, params)`, `deposit_to_grid(positions, alive, grid_shape, L)` | RD update for `c_a`, `c_s`. Bilinear deposit. |
| `src/interactions.py` | `@njit chemotaxis_force(grad_c, pos, L, grid_shape)`, `@njit apply_killing(...)` | Bilinear gradient sampling + Bernoulli kill. |
| `src/sim.py` | `run(rho_I, alpha, seed, T_final, snapshot_every) -> dict` | The function we'll call 100× later. |
| `notebooks/02_two_species_demo.ipynb` | — | (a) T cells aggregate at static Gaussian source. (b) T cells eat a small tumor seed. |

**Validation artifact:** two PNGs — (a) T-cell radial density vs distance from static source showing peak at the source; (b) `N_tumor(t)` curve declining from 50 to 0 in the static-T-cell-vs-small-tumor sanity run.

### Phase 3 — Phase diagram sweep (target 150 min)

| File | Function(s) | Purpose |
|---|---|---|
| `src/sweep.py` | `sweep(rho_I_values, alpha_values, n_seeds, T_final) -> np.ndarray`, `main()` guarded | joblib `Parallel`. Saves `outputs/data/phase_grid.npz`. |
| `notebooks/03_phase_diagram.ipynb` | — | Render heatmap + contours + 3 trajectory thumbnails. |
| `src/render.py` | `plot_phase_diagram(grid, ...)`, `phase_thumbnail(snapshot, ...)` | Slide-quality phase diagram. |

**Sweep budget math:** plan for **8×8 grid × 3 seeds = 192 runs**. If a single run is 60 s, that's 192 min on 1 core, ~24 min on 8 cores. If runs are faster (likely 20–30 s), bump to 10×10 × 3. I'll measure with 4 pilot runs before launching the full grid.

**Validation artifact:** the phase diagram PNG itself.

### Phase 4 — Treatment experiment + videos (target 60 min)

| File | Function(s) | Purpose |
|---|---|---|
| `src/sim.py` | `run_with_treatment(..., t_treat, chi_s_after)` | Modify `chi_s` mid-run. |
| `src/render.py` | `make_video(snapshots, params, out_path)`, `treatment_panel(...)` | Composite frame: main + sidebar + header + phase-inset. |
| `notebooks/04_treatment_experiment.ipynb` | — | Two-panel figure. |

**Validation artifact:** the 13-second video for one phase (MP4) + the treatment two-panel figure.

### Phase 5 — Slides (target 30 min)

Five slides per spec rubric, assembled in `slides/deck.pdf`. I will not write code in this phase — only export figures with consistent styling.

---

## 4. Top three risks + mitigations

### Risk 1: **Numba × Python 3.14 × NumPy 2.4 type-system surprises**
Numba historically lags new Python / NumPy releases. The smoke test compiled, but the proliferation/killing code uses dynamic indexing patterns that have hit edge cases on numpy 2.x.
- **Mitigation:** keep all JIT arguments as `np.float64`/`np.int64` ndarrays only — no Python lists, no dict, no None. Build a 60-second smoke test as the first thing in Phase 1 that exercises every JIT-decorated function with realistic shapes. If a function fails to compile, I peel it back to pure-numpy *for that function only* and keep numba on the hot inner loop.

### Risk 2: **Phase diagram doesn't show three clean phases**
The whole deliverable hinges on this. Parameter defaults in the spec are educated guesses; the actual phase boundaries depend sensitively on `p_div`, `r_kill`, `p_kill`, field decay rates. We could easily sweep `(rho_I, alpha)` and see only "always clears" or "always escapes."
- **Mitigation:** before launching the 192-run grid, do a 9-point pilot (corners + center + edges) at the spec defaults. If we don't see at least two distinct outcomes, tune `p_div` and `p_kill` until the pilot shows separation, then sweep. Time-box the tuning to 20 minutes — if no separation by then, ask you to pick which parameter to widen.

### Risk 3: **Video export pipeline fails on Windows under matplotlib animation**
ffmpeg missing, `FuncAnimation` + `EllipseCollection` + dark background + `imshow` + sidebar + inset is a lot to ask of matplotlib, and Windows matplotlib animation has historically been the buggiest combo.
- **Mitigation:** install `imageio-ffmpeg` *now* (before code). Render frames one-by-one to PNGs first (no `FuncAnimation`) and stitch with `imageio-ffmpeg` directly — bypasses `matplotlib.animation` entirely if it misbehaves. GIF via Pillow as a third fallback so we never block on this for slides.

---

## 5. Scientific choices I'm NOT asking you about

Per your standing preference, I'm taking these as defaults and noting them in `docs/DECISIONS.md`:

- Harmonic (not WCA) repulsion
- Periodic (not reflecting) boundaries — spec already says start here
- Bilinear field interpolation for chemotaxis gradient sampling
- Linear (not Hill-function) field decay
- Daughter cell placed at random direction `0.3σ_T` from parent
- Order parameter: `N_tumor(T_final)/N_tumor(0)`, capped at 100, log color scale — already in spec, no decision to make

## 6. Scientific choices I AM asking you about

None right now. The spec is unambiguous about:
- **Axes** of the phase diagram: `rho_I = N_Tcell_initial` and `alpha = chi_s`
- **Order parameter**: `N_tumor(T_final) / N_tumor(0)`

If during the sweep one of these turns out to give a degenerate phase diagram, I'll pause and ask before changing axes.

---

## 7. What I need from you before Phase 1

1. Confirm you're OK with Python 3.14 (you said 3.11 — your machine is 3.14; numba works).
2. Run `pip install imageio-ffmpeg tqdm cmasher`.
3. Confirm I can `git init` the project directory.
4. Tell me to start Phase 1.
