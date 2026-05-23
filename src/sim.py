"""Core simulation.

Phase 1: single-species ABP with proliferation  (run_single_species)
Phase 2+: two-species tumor-immune with reaction-diffusion fields and
killing  (run, run_with_treatment)

The hot loops are `@njit(cache=True)` functions operating on flat float64
arrays. Outer Python drivers allocate state, call the step N times, dump
snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numba import njit

from src.interactions import (
    pairwise_harmonic_force,
    count_neighbors_at,
    first_dead_slot,
    cross_species_repulsion,
    apply_killing,
)
from src.fields import (
    deposit_tumor_density,
    step_fields,
    grad_field_at,
)


# ===========================================================================
# Phase 1 — single-species ABP (kept for backward compatibility)
# ===========================================================================

@dataclass
class TumorParams:
    """Single-species tumor ABP parameters (Phase 1 only)."""
    L: float = 100.0
    dt: float = 0.01
    v: float = 0.1
    D_R: float = 0.1
    D_T: float = 0.001
    sigma: float = 1.0
    k_rep: float = 30.0
    p_div: float = 0.005
    nbr_radius: float = 1.5
    nbr_threshold: int = 6
    N_max: int = 5000

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.dt, self.v, self.D_R, self.D_T,
                self.sigma, self.k_rep, self.L,
                self.p_div, float(self.nbr_threshold), self.nbr_radius,
            ],
            dtype=np.float64,
        )


def init_tumor_disk(n, L, radius, N_max, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.zeros((N_max, 2), dtype=np.float64)
    theta = np.zeros(N_max, dtype=np.float64)
    alive = np.zeros(N_max, dtype=np.bool_)
    cx = cy = 0.5 * L
    placed = 0
    attempts = 0
    while placed < n and attempts < 10000:
        x = rng.uniform(cx - radius, cx + radius)
        y = rng.uniform(cy - radius, cy + radius)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            pos[placed] = (x, y)
            theta[placed] = rng.uniform(0.0, 2.0 * np.pi)
            alive[placed] = True
            placed += 1
        attempts += 1
    return pos, theta, alive


def init_tumor_uniform(n, L, N_max, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.zeros((N_max, 2), dtype=np.float64)
    theta = np.zeros(N_max, dtype=np.float64)
    alive = np.zeros(N_max, dtype=np.bool_)
    pos[:n, 0] = rng.uniform(0.0, L, size=n)
    pos[:n, 1] = rng.uniform(0.0, L, size=n)
    theta[:n] = rng.uniform(0.0, 2.0 * np.pi, size=n)
    alive[:n] = True
    return pos, theta, alive


@njit(cache=True, fastmath=True)
def step_single_species(pos, theta, alive, params) -> int:
    """Single-species ABP step. params layout in TumorParams.to_array."""
    dt = params[0]; v = params[1]; D_R = params[2]; D_T = params[3]
    sigma = params[4]; k_rep = params[5]; L = params[6]
    p_div = params[7]; nbr_thresh = int(params[8]); nbr_radius = params[9]

    forces = pairwise_harmonic_force(pos, alive, sigma, k_rep, L)
    sqrt_2DT_dt = np.sqrt(2.0 * D_T * dt)
    sqrt_2DR_dt = np.sqrt(2.0 * D_R * dt)
    r2_nbr = nbr_radius * nbr_radius
    N_max = pos.shape[0]

    for i in range(N_max):
        if not alive[i]:
            continue
        cs = np.cos(theta[i]); sn = np.sin(theta[i])
        pos[i, 0] += dt * (v * cs + forces[i, 0]) + sqrt_2DT_dt * np.random.normal()
        pos[i, 1] += dt * (v * sn + forces[i, 1]) + sqrt_2DT_dt * np.random.normal()
        theta[i] += sqrt_2DR_dt * np.random.normal()
        pos[i, 0] = pos[i, 0] % L
        pos[i, 1] = pos[i, 1] % L

    n_born = 0
    write_cursor = 0
    for i in range(N_max):
        if not alive[i]:
            continue
        if np.random.random() < p_div:
            cnt = count_neighbors_at(pos, alive, i, r2_nbr, L)
            if cnt < nbr_thresh:
                slot = first_dead_slot(alive, write_cursor)
                if slot < 0:
                    break
                write_cursor = slot + 1
                angle = 2.0 * np.pi * np.random.random()
                pos[slot, 0] = (pos[i, 0] + 0.3 * sigma * np.cos(angle)) % L
                pos[slot, 1] = (pos[i, 1] + 0.3 * sigma * np.sin(angle)) % L
                theta[slot] = 2.0 * np.pi * np.random.random()
                alive[slot] = True
                n_born += 1
    return n_born


@dataclass
class SingleSpeciesRun:
    pos_snapshots: list = field(default_factory=list)
    n_alive: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: Optional[TumorParams] = None


def run_single_species(
    params: TumorParams,
    n_initial: int,
    n_steps: int,
    init: str = "disk",
    init_radius: float = 5.0,
    snapshot_every: int = 50,
    seed: int = 0,
    enable_proliferation: bool = True,
) -> SingleSpeciesRun:
    np.random.seed(seed)
    if init == "disk":
        pos, theta, alive = init_tumor_disk(n_initial, params.L, init_radius, params.N_max, seed=seed)
    elif init == "uniform":
        pos, theta, alive = init_tumor_uniform(n_initial, params.L, params.N_max, seed=seed)
    else:
        raise ValueError(f"unknown init mode {init!r}")
    p = params if enable_proliferation else TumorParams(**{**params.__dict__, "p_div": 0.0})
    param_arr = p.to_array()
    out = SingleSpeciesRun(params=params)
    for step in range(n_steps):
        step_single_species(pos, theta, alive, param_arr)
        if step % snapshot_every == 0 or step == n_steps - 1:
            out.pos_snapshots.append(pos[alive].copy())
            out.n_alive.append(int(alive.sum()))
            out.times.append(step * p.dt)
    return out


# ===========================================================================
# Phase 2 — two-species (tumor + T cell) with RD fields and killing
# ===========================================================================

@dataclass
class SimParams:
    """All parameters for the two-species tumor-immune simulation.

    Defaults: spec baseline with three deviations tuned during Phase 3 pilot
    so all three phases (clearance/control/escape) are actually reachable:
      * chi_a 5 -> 20  (T cells find the tumor faster)
      * p_kill 0.05 -> 0.12  (more lethal per contact, widens clearance)
      * p_div 0.005 -> 0.004 (slightly slower tumor growth)
      * T_final 200 -> 100  (each run cheaper; phases still emerge by ~t=60)
      * N_T_max 4000 -> 800  (caps the worst-case O(N^2) cost in escape)
      * N_I_max 2000 -> 1200
      * G 128 -> 64  (4x faster field updates)

    Sweep varies `N_I_initial` (rho_I) and `chi_s` (alpha).
    """
    L: float = 100.0
    dt: float = 0.01
    T_final: float = 100.0
    G: int = 64                  # field grid resolution

    # tumor
    v_T: float = 0.1
    D_R_T: float = 0.1
    D_T_T: float = 0.001
    sigma_T: float = 1.0
    k_rep_T: float = 30.0
    p_div: float = 0.004
    nbr_radius: float = 1.5
    nbr_threshold: int = 6

    # T cell
    v_I: float = 1.0
    D_R_I: float = 1.0
    D_T_I: float = 0.001
    sigma_I: float = 1.0
    k_rep_I: float = 30.0
    sigma_TI: float = 1.0
    k_rep_TI: float = 30.0

    # chemotaxis
    chi_a: float = 20.0
    chi_s: float = 5.0           # = alpha (scanned)

    # fields
    D_a: float = 5.0
    D_s: float = 0.5
    s_a: float = 1.0
    s_s: float = 1.0
    lam_a: float = 0.1
    lam_s: float = 0.1

    # killing
    r_kill: float = 1.5
    p_kill: float = 0.12

    # particle pool sizes
    N_T_max: int = 800
    N_I_max: int = 1200

    # initial conditions
    N_T_initial: int = 50
    N_I_initial: int = 100      # = rho_I (scanned)
    tumor_disk_radius: float = 5.0

    @property
    def n_steps(self) -> int:
        return int(round(self.T_final / self.dt))


def init_two_species(
    params: SimParams,
    seed: int = 0,
) -> dict:
    """Allocate and initialize particle + field state for a fresh run."""
    rng = np.random.default_rng(seed)

    # tumor: disk at center
    pos_T = np.zeros((params.N_T_max, 2), dtype=np.float64)
    theta_T = np.zeros(params.N_T_max, dtype=np.float64)
    alive_T = np.zeros(params.N_T_max, dtype=np.bool_)
    cx = cy = 0.5 * params.L
    placed = 0
    attempts = 0
    while placed < params.N_T_initial and attempts < 100000:
        x = rng.uniform(cx - params.tumor_disk_radius, cx + params.tumor_disk_radius)
        y = rng.uniform(cy - params.tumor_disk_radius, cy + params.tumor_disk_radius)
        if (x - cx) ** 2 + (y - cy) ** 2 <= params.tumor_disk_radius ** 2:
            pos_T[placed] = (x, y)
            theta_T[placed] = rng.uniform(0, 2 * np.pi)
            alive_T[placed] = True
            placed += 1
        attempts += 1

    # T cells: uniform in box
    pos_I = np.zeros((params.N_I_max, 2), dtype=np.float64)
    theta_I = np.zeros(params.N_I_max, dtype=np.float64)
    alive_I = np.zeros(params.N_I_max, dtype=np.bool_)
    n_I = params.N_I_initial
    pos_I[:n_I, 0] = rng.uniform(0.0, params.L, size=n_I)
    pos_I[:n_I, 1] = rng.uniform(0.0, params.L, size=n_I)
    theta_I[:n_I] = rng.uniform(0.0, 2 * np.pi, size=n_I)
    alive_I[:n_I] = True

    # fields: zero (will grow as tumor secretes)
    c_a = np.zeros((params.G, params.G), dtype=np.float64)
    c_s = np.zeros((params.G, params.G), dtype=np.float64)

    return {
        "pos_T": pos_T, "theta_T": theta_T, "alive_T": alive_T,
        "pos_I": pos_I, "theta_I": theta_I, "alive_I": alive_I,
        "c_a": c_a, "c_s": c_s,
    }


# ---------------------------------------------------------------------------
# JIT inner loop (two species)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _step_particles_two_species(
    pos_T, theta_T, alive_T,
    pos_I, theta_I, alive_I,
    c_a, c_s,
    dt, v_T, v_I, D_R_T, D_R_I, D_T_T, D_T_I,
    sigma_T, k_rep_T, sigma_I, k_rep_I, sigma_TI, k_rep_TI, L,
    chi_a, chi_s,
    r_kill, p_kill,
    p_div, nbr_radius, nbr_thresh,
):
    """Particle update + killing + proliferation. Fields stepped separately."""
    # forces
    f_TT = pairwise_harmonic_force(pos_T, alive_T, sigma_T, k_rep_T, L)
    f_II = pairwise_harmonic_force(pos_I, alive_I, sigma_I, k_rep_I, L)
    f_TI_T, f_TI_I = cross_species_repulsion(
        pos_T, alive_T, pos_I, alive_I, sigma_TI, k_rep_TI, L
    )

    sqrt_2DT_T = np.sqrt(2.0 * D_T_T * dt)
    sqrt_2DT_I = np.sqrt(2.0 * D_T_I * dt)
    sqrt_2DR_T = np.sqrt(2.0 * D_R_T * dt)
    sqrt_2DR_I = np.sqrt(2.0 * D_R_I * dt)

    # tumor update
    N_Tmax = pos_T.shape[0]
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        cs = np.cos(theta_T[i]); sn = np.sin(theta_T[i])
        fx = f_TT[i, 0] + f_TI_T[i, 0]
        fy = f_TT[i, 1] + f_TI_T[i, 1]
        pos_T[i, 0] += dt * (v_T * cs + fx) + sqrt_2DT_T * np.random.normal()
        pos_T[i, 1] += dt * (v_T * sn + fy) + sqrt_2DT_T * np.random.normal()
        theta_T[i] += sqrt_2DR_T * np.random.normal()
        pos_T[i, 0] = pos_T[i, 0] % L
        pos_T[i, 1] = pos_T[i, 1] % L

    # T cell update — includes chemotaxis force
    N_Imax = pos_I.shape[0]
    for j in range(N_Imax):
        if not alive_I[j]:
            continue
        cs = np.cos(theta_I[j]); sn = np.sin(theta_I[j])
        # chemotaxis: chi_a * grad(c_a) - chi_s * grad(c_s)
        gax, gay = grad_field_at(c_a, pos_I[j, 0], pos_I[j, 1], L)
        gsx, gsy = grad_field_at(c_s, pos_I[j, 0], pos_I[j, 1], L)
        cx_force = chi_a * gax - chi_s * gsx
        cy_force = chi_a * gay - chi_s * gsy
        fx = f_II[j, 0] + f_TI_I[j, 0] + cx_force
        fy = f_II[j, 1] + f_TI_I[j, 1] + cy_force
        pos_I[j, 0] += dt * (v_I * cs + fx) + sqrt_2DT_I * np.random.normal()
        pos_I[j, 1] += dt * (v_I * sn + fy) + sqrt_2DT_I * np.random.normal()
        theta_I[j] += sqrt_2DR_I * np.random.normal()
        pos_I[j, 0] = pos_I[j, 0] % L
        pos_I[j, 1] = pos_I[j, 1] % L

    # killing
    n_killed = apply_killing(pos_T, alive_T, pos_I, alive_I, r_kill, p_kill, L)

    # proliferation
    r2_nbr = nbr_radius * nbr_radius
    n_born = 0
    write_cursor = 0
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        if np.random.random() < p_div:
            cnt = count_neighbors_at(pos_T, alive_T, i, r2_nbr, L)
            if cnt < nbr_thresh:
                slot = first_dead_slot(alive_T, write_cursor)
                if slot < 0:
                    break
                write_cursor = slot + 1
                angle = 2.0 * np.pi * np.random.random()
                pos_T[slot, 0] = (pos_T[i, 0] + 0.3 * sigma_T * np.cos(angle)) % L
                pos_T[slot, 1] = (pos_T[i, 1] + 0.3 * sigma_T * np.sin(angle)) % L
                theta_T[slot] = 2.0 * np.pi * np.random.random()
                alive_T[slot] = True
                n_born += 1
    return n_born, n_killed


@dataclass
class TwoSpeciesRun:
    """Container for trajectory snapshots from a two-species run."""
    pos_T_snapshots: list = field(default_factory=list)
    pos_I_snapshots: list = field(default_factory=list)
    c_a_snapshots: list = field(default_factory=list)
    c_s_snapshots: list = field(default_factory=list)
    n_T: list = field(default_factory=list)
    n_I: list = field(default_factory=list)
    n_killed_cum: list = field(default_factory=list)
    n_born_cum: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: Optional[SimParams] = None
    final_tumor_fraction: float = 0.0


def run(
    rho_I: int,
    alpha: float,
    seed: int = 0,
    T_final: Optional[float] = None,
    snapshot_every: int = 50,
    params: Optional[SimParams] = None,
    save_fields: bool = True,
) -> TwoSpeciesRun:
    """Run the two-species sim with rho_I (initial T-cell count) and alpha (chi_s).

    Returns a TwoSpeciesRun with trajectory snapshots and the final tumor
    fraction (capped at 100, with a floor at 1e-2 for log scaling).
    """
    if params is None:
        params = SimParams()
    if T_final is not None:
        params = SimParams(**{**params.__dict__, "T_final": T_final})
    params = SimParams(**{**params.__dict__, "N_I_initial": int(rho_I), "chi_s": float(alpha)})

    np.random.seed(seed)
    state = init_two_species(params, seed=seed)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    c_a = state["c_a"]; c_s = state["c_s"]

    n_steps = params.n_steps
    out = TwoSpeciesRun(params=params)
    cum_born = 0
    cum_killed = 0
    N_T_initial = int(alive_T.sum())

    for step in range(n_steps):
        # particles + killing + proliferation
        n_born, n_killed = _step_particles_two_species(
            pos_T, theta_T, alive_T,
            pos_I, theta_I, alive_I,
            c_a, c_s,
            params.dt, params.v_T, params.v_I,
            params.D_R_T, params.D_R_I, params.D_T_T, params.D_T_I,
            params.sigma_T, params.k_rep_T,
            params.sigma_I, params.k_rep_I,
            params.sigma_TI, params.k_rep_TI, params.L,
            params.chi_a, params.chi_s,
            params.r_kill, params.p_kill,
            params.p_div, params.nbr_radius, params.nbr_threshold,
        )
        cum_born += n_born
        cum_killed += n_killed

        # field update
        rho = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        c_a, c_s = step_fields(
            c_a, c_s, rho,
            params.D_a, params.D_s,
            params.s_a, params.s_s,
            params.lam_a, params.lam_s,
            params.dt, params.L,
        )

        # tumor extinction: short-circuit (saves time on clearance runs)
        if step > 100 and not alive_T.any():
            for _remaining in range(step, n_steps, snapshot_every):
                out.pos_T_snapshots.append(np.zeros((0, 2)))
                out.pos_I_snapshots.append(pos_I[alive_I].copy())
                if save_fields:
                    out.c_a_snapshots.append(c_a.copy())
                    out.c_s_snapshots.append(c_s.copy())
                out.n_T.append(0)
                out.n_I.append(int(alive_I.sum()))
                out.n_killed_cum.append(cum_killed)
                out.n_born_cum.append(cum_born)
                out.times.append(_remaining * params.dt)
            break

        if step % snapshot_every == 0 or step == n_steps - 1:
            out.pos_T_snapshots.append(pos_T[alive_T].copy())
            out.pos_I_snapshots.append(pos_I[alive_I].copy())
            if save_fields:
                out.c_a_snapshots.append(c_a.copy())
                out.c_s_snapshots.append(c_s.copy())
            out.n_T.append(int(alive_T.sum()))
            out.n_I.append(int(alive_I.sum()))
            out.n_killed_cum.append(cum_killed)
            out.n_born_cum.append(cum_born)
            out.times.append(step * params.dt)

    # cap at 100, floor at 1e-2 for log-scale plotting
    final_n_T = out.n_T[-1] if out.n_T else 0
    frac = final_n_T / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 100.0))
    return out


def run_with_treatment(
    rho_I: int,
    alpha: float,
    alpha_after: float,
    t_treat: Optional[float] = None,
    seed: int = 0,
    T_final: Optional[float] = None,
    snapshot_every: int = 50,
    params: Optional[SimParams] = None,
) -> TwoSpeciesRun:
    """Same as `run` but flips chi_s -> alpha_after at t = t_treat.

    Default t_treat = T_final / 2.
    """
    if params is None:
        params = SimParams()
    if T_final is not None:
        params = SimParams(**{**params.__dict__, "T_final": T_final})
    if t_treat is None:
        t_treat = params.T_final / 2.0
    treat_step = int(round(t_treat / params.dt))
    params = SimParams(**{**params.__dict__, "N_I_initial": int(rho_I), "chi_s": float(alpha)})

    np.random.seed(seed)
    state = init_two_species(params, seed=seed)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    c_a = state["c_a"]; c_s = state["c_s"]
    n_steps = params.n_steps
    out = TwoSpeciesRun(params=params)
    cum_born = 0; cum_killed = 0
    N_T_initial = int(alive_T.sum())
    chi_s_current = params.chi_s

    for step in range(n_steps):
        if step == treat_step:
            chi_s_current = float(alpha_after)

        n_born, n_killed = _step_particles_two_species(
            pos_T, theta_T, alive_T,
            pos_I, theta_I, alive_I,
            c_a, c_s,
            params.dt, params.v_T, params.v_I,
            params.D_R_T, params.D_R_I, params.D_T_T, params.D_T_I,
            params.sigma_T, params.k_rep_T,
            params.sigma_I, params.k_rep_I,
            params.sigma_TI, params.k_rep_TI, params.L,
            params.chi_a, chi_s_current,
            params.r_kill, params.p_kill,
            params.p_div, params.nbr_radius, params.nbr_threshold,
        )
        cum_born += n_born; cum_killed += n_killed
        rho = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        c_a, c_s = step_fields(
            c_a, c_s, rho,
            params.D_a, params.D_s, params.s_a, params.s_s,
            params.lam_a, params.lam_s, params.dt, params.L,
        )
        if step % snapshot_every == 0 or step == n_steps - 1:
            out.pos_T_snapshots.append(pos_T[alive_T].copy())
            out.pos_I_snapshots.append(pos_I[alive_I].copy())
            out.c_a_snapshots.append(c_a.copy())
            out.c_s_snapshots.append(c_s.copy())
            out.n_T.append(int(alive_T.sum()))
            out.n_I.append(int(alive_I.sum()))
            out.n_killed_cum.append(cum_killed)
            out.n_born_cum.append(cum_born)
            out.times.append(step * params.dt)
    frac = out.n_T[-1] / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 100.0))
    return out
