"""Extended tumor-immune simulation with biological-realism upgrades.

Three additions over `src.sim.run`:
1. Per-cell heterogeneity in self-propulsion (v_T_i, v_I_j) and rotational
   diffusion (D_R_T_i, D_R_I_j). Tumor daughters inherit + small drift.
2. Pressure-gated proliferation (Byrne–Drasdo). The division gate is now
   `p_div_local = p_div_0 * max(0, 1 - P_i / P_star)` where P_i is the local
   mechanical pressure from pairwise repulsion virials.
3. Macrophages (TAMs): a third self-propelled species carrying a polarization
   scalar p_k ∈ [-1, +1] (M2 ↔ M1). M2-skewed macrophages secrete a long-range
   immunosuppressant `c_IL10`; M1-skewed macrophages phagocytose nearby tumor.
   The CD8 equation gains a `-α∇(c_s + c_IL10)` chemotactic-repulsion term.

The baseline `src.sim.run` is untouched. Use `run_extended()` from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from numba import njit

from src.interactions import (
    count_neighbors_at,
    first_dead_slot,
    cross_species_repulsion,
    apply_killing,
)
from src.fields import (
    deposit_tumor_density,
    step_fields,                # used for c_a, c_s
    step_field_one_substep,
    n_substeps_for_cfl,
    grad_field_at,
)


# ===========================================================================
# Parameter container
# ===========================================================================

@dataclass
class ExtendedParams:
    """Parameters for the extended (3-species + heterogeneity + pressure) sim."""
    # --- box / time -----------------------------------------------------
    L: float = 100.0
    dt: float = 0.01
    T_final: float = 100.0
    G: int = 64

    # --- tumor (means; per-cell arrays carry heterogeneity) -------------
    v_T_mean: float = 0.1
    v_T_cv: float = 0.30                # log-normal CV at birth
    D_R_T_mean: float = 0.1
    D_R_T_cv: float = 0.40              # gamma CV at birth
    D_T_T: float = 0.001
    sigma_T: float = 1.0
    k_rep_T: float = 30.0

    # --- tumor proliferation (pressure-gated) ---------------------------
    p_div: float = 0.004
    P_star: float = 8.0                 # pressure where p_div → 0
    nbr_radius: float = 1.5             # kept as fall-back density gate
    nbr_threshold: int = 12             # relaxed; pressure is primary gate
    daughter_v_drift: float = 0.02
    daughter_DR_drift: float = 0.02

    # --- T cell ---------------------------------------------------------
    v_I_mean: float = 1.0
    v_I_cv: float = 0.20
    D_R_I_mean: float = 1.0
    D_R_I_cv: float = 0.30
    D_T_I: float = 0.001
    sigma_I: float = 1.0
    k_rep_I: float = 30.0
    sigma_TI: float = 1.0
    k_rep_TI: float = 30.0

    # --- chemotaxis -----------------------------------------------------
    chi_a: float = 20.0
    chi_s: float = 5.0                  # scanned (α) — coupling to c_s + c_IL10

    # --- attractant / suppressant fields (as baseline) ------------------
    D_a: float = 5.0
    D_s: float = 0.5
    s_a: float = 1.0
    s_s: float = 1.0
    lam_a: float = 0.1
    lam_s: float = 0.1

    # --- macrophages ----------------------------------------------------
    use_macrophages: bool = True
    v_M_mean: float = 0.2               # slow
    D_R_M_mean: float = 3.0             # low persistence
    D_T_M: float = 0.001
    sigma_M: float = 1.0
    k_rep_M: float = 30.0
    sigma_TM: float = 1.0               # tumor-macrophage repulsion cutoff
    k_rep_TM: float = 30.0
    sigma_IM: float = 1.0
    k_rep_IM: float = 30.0
    chi_a_M: float = 8.0                # chemotaxis to attractant (recruits to tumor)
    tau_p: float = 15.0                 # polarization relaxation timescale
    D_p: float = 0.03                   # polarization noise
    kappa_s: float = 8.0                # suppressant → M2 drive
    kappa_il: float = 4.0               # IL-10 → M2 (positive feedback)
    M1_bias: float = 0.0                # external drive toward M1 (treatment)
    r_phag: float = 1.8                 # M1 phagocytosis radius
    p_phag: float = 0.10                # base phagocytosis probability per step

    # --- IL-10 field (M2-secreted, anti-CD8) ---------------------------
    D_IL10: float = 4.0                 # long-range like attractant
    s_IL10: float = 1.0
    lam_IL10: float = 0.1

    # --- killing ---------------------------------------------------------
    r_kill: float = 1.5
    p_kill: float = 0.12

    # --- pool sizes -----------------------------------------------------
    N_T_max: int = 1500
    N_I_max: int = 1200
    N_M_max: int = 600

    # --- initial conditions --------------------------------------------
    N_T_initial: int = 50
    N_I_initial: int = 100
    N_M_initial: int = 80
    tumor_disk_radius: float = 5.0

    @property
    def n_steps(self) -> int:
        return int(round(self.T_final / self.dt))


# ===========================================================================
# Heterogeneity draws
# ===========================================================================

def _draw_lognormal(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from a log-normal with target mean and CV."""
    if cv <= 1e-6:
        return np.full(n, mean, dtype=np.float64)
    sig2 = np.log(1.0 + cv * cv)
    mu = np.log(mean) - 0.5 * sig2
    return rng.lognormal(mu, np.sqrt(sig2), size=n).astype(np.float64)


def _draw_gamma(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if cv <= 1e-6:
        return np.full(n, mean, dtype=np.float64)
    k = 1.0 / (cv * cv)
    theta = mean / k
    return rng.gamma(k, theta, size=n).astype(np.float64)


# ===========================================================================
# State initialization
# ===========================================================================

def init_extended_state(params: ExtendedParams, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    L = params.L

    # ------ tumor: small disk at center ------
    pos_T = np.zeros((params.N_T_max, 2), dtype=np.float64)
    theta_T = np.zeros(params.N_T_max, dtype=np.float64)
    alive_T = np.zeros(params.N_T_max, dtype=np.bool_)
    v_T_arr = np.zeros(params.N_T_max, dtype=np.float64)
    DR_T_arr = np.zeros(params.N_T_max, dtype=np.float64)
    cx = cy = 0.5 * L
    R0 = params.tumor_disk_radius
    placed = 0; attempts = 0
    while placed < params.N_T_initial and attempts < 200_000:
        x = rng.uniform(cx - R0, cx + R0); y = rng.uniform(cy - R0, cy + R0)
        if (x - cx) ** 2 + (y - cy) ** 2 <= R0 ** 2:
            pos_T[placed] = (x, y)
            theta_T[placed] = rng.uniform(0, 2 * np.pi)
            alive_T[placed] = True
            placed += 1
        attempts += 1
    # heterogeneity for the alive ones
    vT_draws = _draw_lognormal(params.v_T_mean, params.v_T_cv, params.N_T_initial, rng)
    dR_draws = _draw_gamma(params.D_R_T_mean, params.D_R_T_cv, params.N_T_initial, rng)
    v_T_arr[:params.N_T_initial] = vT_draws
    DR_T_arr[:params.N_T_initial] = dR_draws

    # ------ T cells: uniform in box ------
    pos_I = np.zeros((params.N_I_max, 2), dtype=np.float64)
    theta_I = np.zeros(params.N_I_max, dtype=np.float64)
    alive_I = np.zeros(params.N_I_max, dtype=np.bool_)
    v_I_arr = np.zeros(params.N_I_max, dtype=np.float64)
    DR_I_arr = np.zeros(params.N_I_max, dtype=np.float64)
    n_I = params.N_I_initial
    pos_I[:n_I, 0] = rng.uniform(0, L, size=n_I)
    pos_I[:n_I, 1] = rng.uniform(0, L, size=n_I)
    theta_I[:n_I] = rng.uniform(0, 2 * np.pi, size=n_I)
    alive_I[:n_I] = True
    v_I_arr[:n_I] = _draw_lognormal(params.v_I_mean, params.v_I_cv, n_I, rng)
    DR_I_arr[:n_I] = _draw_gamma(params.D_R_I_mean, params.D_R_I_cv, n_I, rng)

    # ------ macrophages: uniform in box, slight bias toward outer region ------
    pos_M = np.zeros((params.N_M_max, 2), dtype=np.float64)
    theta_M = np.zeros(params.N_M_max, dtype=np.float64)
    alive_M = np.zeros(params.N_M_max, dtype=np.bool_)
    p_M = np.zeros(params.N_M_max, dtype=np.float64)
    n_M = params.N_M_initial if params.use_macrophages else 0
    if n_M > 0:
        pos_M[:n_M, 0] = rng.uniform(0, L, size=n_M)
        pos_M[:n_M, 1] = rng.uniform(0, L, size=n_M)
        theta_M[:n_M] = rng.uniform(0, 2 * np.pi, size=n_M)
        alive_M[:n_M] = True
        # initial polarization centered around 0 (uncommitted)
        p_M[:n_M] = rng.normal(0.0, 0.2, size=n_M)

    # ------ fields ------
    c_a = np.zeros((params.G, params.G), dtype=np.float64)
    c_s = np.zeros((params.G, params.G), dtype=np.float64)
    c_IL10 = np.zeros((params.G, params.G), dtype=np.float64)

    return {
        "pos_T": pos_T, "theta_T": theta_T, "alive_T": alive_T,
        "v_T_arr": v_T_arr, "DR_T_arr": DR_T_arr,
        "pos_I": pos_I, "theta_I": theta_I, "alive_I": alive_I,
        "v_I_arr": v_I_arr, "DR_I_arr": DR_I_arr,
        "pos_M": pos_M, "theta_M": theta_M, "alive_M": alive_M, "p_M": p_M,
        "c_a": c_a, "c_s": c_s, "c_IL10": c_IL10,
    }


# ===========================================================================
# JIT helpers
# ===========================================================================

@njit(cache=True, fastmath=True)
def pairwise_with_pressure(
    pos: np.ndarray, alive: np.ndarray,
    sigma: float, k_rep: float, L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise harmonic repulsion + per-cell pressure (scalar virial sum).

    Returns (forces[N_max, 2], pressure[N_max]).

    Pressure is computed as
        P_i = (1 / A_i) * sum_j (r_ij · F_ij)
    with A_i = π σ² (a per-cell "cell area"); divided so P is intensive.
    """
    N_max = pos.shape[0]
    forces = np.zeros((N_max, 2), dtype=np.float64)
    pressure = np.zeros(N_max, dtype=np.float64)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
    A_inv = 1.0 / (np.pi * sigma2)
    for i in range(N_max):
        if not alive[i]:
            continue
        xi = pos[i, 0]; yi = pos[i, 1]
        for j in range(i + 1, N_max):
            if not alive[j]:
                continue
            dx = xi - pos[j, 0]; dy = yi - pos[j, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < sigma2 and r2 > 1e-12:
                r = np.sqrt(r2)
                fmag_over_r = k_rep * (sigma - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                forces[i, 0] += fx; forces[i, 1] += fy
                forces[j, 0] -= fx; forces[j, 1] -= fy
                virial = (dx * fx + dy * fy) * A_inv
                pressure[i] += virial
                pressure[j] += virial
    return forces, pressure


@njit(cache=True, fastmath=True)
def cross_repulsion_with_pressure_a(
    pos_a: np.ndarray, alive_a: np.ndarray,
    pos_b: np.ndarray, alive_b: np.ndarray,
    sigma: float, k_rep: float, L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-species harmonic repulsion + pressure contribution accumulated
    on species A (used to add inter-species pressure to tumor cells)."""
    N_a = pos_a.shape[0]; N_b = pos_b.shape[0]
    fa = np.zeros_like(pos_a); fb = np.zeros_like(pos_b)
    pa = np.zeros(N_a, dtype=np.float64)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
    A_inv = 1.0 / (np.pi * sigma2)
    for i in range(N_a):
        if not alive_a[i]:
            continue
        xi = pos_a[i, 0]; yi = pos_a[i, 1]
        for j in range(N_b):
            if not alive_b[j]:
                continue
            dx = xi - pos_b[j, 0]; dy = yi - pos_b[j, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < sigma2 and r2 > 1e-12:
                r = np.sqrt(r2)
                fmag_over_r = k_rep * (sigma - r) / r
                fx = fmag_over_r * dx; fy = fmag_over_r * dy
                fa[i, 0] += fx; fa[i, 1] += fy
                fb[j, 0] -= fx; fb[j, 1] -= fy
                pa[i] += (dx * fx + dy * fy) * A_inv
    return fa, fb, pa


@njit(cache=True)
def apply_phagocytosis(
    pos_tumor: np.ndarray, alive_tumor: np.ndarray,
    pos_M: np.ndarray, alive_M: np.ndarray, p_M: np.ndarray,
    r_phag: float, p_phag: float, L: float,
) -> int:
    """For each M1-skewed macrophage (p_k > 0), at most one phagocytosis event.

    Per-step probability scales linearly with (1 + p_k) / 2 — fully M1 phag at
    rate p_phag, fully M2 phag at 0.
    """
    half_L = 0.5 * L
    r2_phag = r_phag * r_phag
    n_killed = 0
    N_T = pos_tumor.shape[0]; N_M = pos_M.shape[0]
    for k in range(N_M):
        if not alive_M[k] or p_M[k] <= 0.0:
            continue
        xk = pos_M[k, 0]; yk = pos_M[k, 1]
        best_i = -1; best_r2 = r2_phag
        for i in range(N_T):
            if not alive_tumor[i]:
                continue
            dx = xk - pos_tumor[i, 0]; dy = yk - pos_tumor[i, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < best_r2:
                best_r2 = r2; best_i = i
        if best_i >= 0:
            p_eff = p_phag * 0.5 * (1.0 + p_M[k])
            if np.random.random() < p_eff:
                alive_tumor[best_i] = False
                n_killed += 1
    return n_killed


# ===========================================================================
# JIT inner step
# ===========================================================================

@njit(cache=True, fastmath=True)
def _step_extended(
    # tumor
    pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
    # T cell
    pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
    # macrophage
    pos_M, theta_M, alive_M, p_M,
    # fields
    c_a, c_s, c_IL10,
    # scalars (lots; positional)
    dt,
    sigma_T, k_rep_T, sigma_I, k_rep_I, sigma_TI, k_rep_TI,
    sigma_M, k_rep_M, sigma_TM, k_rep_TM, sigma_IM, k_rep_IM,
    L,
    D_T_T, D_T_I, D_T_M,
    chi_a, chi_s, chi_a_M,
    r_kill, p_kill, r_phag, p_phag,
    p_div0, P_star, nbr_radius, nbr_thresh,
    tau_p, D_p, kappa_s, kappa_il, M1_bias,
    daughter_v_drift, daughter_DR_drift,
    use_macrophages,
):
    """One step of the extended dynamics. Returns (n_born, n_killed_cd8, n_phag)."""
    # ---- pairwise forces + pressure on tumor ----
    f_TT, P_TT = pairwise_with_pressure(pos_T, alive_T, sigma_T, k_rep_T, L)
    f_II = pairwise_with_pressure(pos_I, alive_I, sigma_I, k_rep_I, L)[0]
    f_TI_T, f_TI_I, P_TI = cross_repulsion_with_pressure_a(
        pos_T, alive_T, pos_I, alive_I, sigma_TI, k_rep_TI, L,
    )
    # tumor-macrophage and immune-macrophage repulsion
    if use_macrophages:
        f_MM = pairwise_with_pressure(pos_M, alive_M, sigma_M, k_rep_M, L)[0]
        f_TM_T, f_TM_M, P_TM = cross_repulsion_with_pressure_a(
            pos_T, alive_T, pos_M, alive_M, sigma_TM, k_rep_TM, L,
        )
        f_IM_I, f_IM_M, _ = cross_repulsion_with_pressure_a(
            pos_I, alive_I, pos_M, alive_M, sigma_IM, k_rep_IM, L,
        )
    else:
        f_MM = np.zeros_like(pos_M)
        f_TM_T = np.zeros_like(pos_T)
        f_TM_M = np.zeros_like(pos_M)
        P_TM = np.zeros(pos_T.shape[0], dtype=np.float64)
        f_IM_I = np.zeros_like(pos_I)
        f_IM_M = np.zeros_like(pos_M)

    # total tumor pressure (intra + cross-species)
    P_tumor = P_TT + P_TI + P_TM

    # ---- tumor update ----
    N_Tmax = pos_T.shape[0]
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        v_i = v_T_arr[i]
        D_R_i = DR_T_arr[i]
        sqrt_2DT = np.sqrt(2.0 * D_T_T * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_i * dt)
        cs = np.cos(theta_T[i]); sn = np.sin(theta_T[i])
        fx = f_TT[i, 0] + f_TI_T[i, 0] + f_TM_T[i, 0]
        fy = f_TT[i, 1] + f_TI_T[i, 1] + f_TM_T[i, 1]
        pos_T[i, 0] += dt * (v_i * cs + fx) + sqrt_2DT * np.random.normal()
        pos_T[i, 1] += dt * (v_i * sn + fy) + sqrt_2DT * np.random.normal()
        theta_T[i] += sqrt_2DR * np.random.normal()
        pos_T[i, 0] = pos_T[i, 0] % L
        pos_T[i, 1] = pos_T[i, 1] % L

    # ---- T-cell update (chemotaxis: c_s + c_IL10) ----
    N_Imax = pos_I.shape[0]
    for j in range(N_Imax):
        if not alive_I[j]:
            continue
        v_j = v_I_arr[j]
        D_R_j = DR_I_arr[j]
        sqrt_2DT = np.sqrt(2.0 * D_T_I * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_j * dt)
        cs = np.cos(theta_I[j]); sn = np.sin(theta_I[j])
        gax, gay = grad_field_at(c_a, pos_I[j, 0], pos_I[j, 1], L)
        gsx, gsy = grad_field_at(c_s, pos_I[j, 0], pos_I[j, 1], L)
        gilx, gily = grad_field_at(c_IL10, pos_I[j, 0], pos_I[j, 1], L)
        # Effective suppressant gradient = c_s + c_IL10
        chemo_x = chi_a * gax - chi_s * (gsx + gilx)
        chemo_y = chi_a * gay - chi_s * (gsy + gily)
        fx = f_II[j, 0] + f_TI_I[j, 0] + f_IM_I[j, 0] + chemo_x
        fy = f_II[j, 1] + f_TI_I[j, 1] + f_IM_I[j, 1] + chemo_y
        pos_I[j, 0] += dt * (v_j * cs + fx) + sqrt_2DT * np.random.normal()
        pos_I[j, 1] += dt * (v_j * sn + fy) + sqrt_2DT * np.random.normal()
        theta_I[j] += sqrt_2DR * np.random.normal()
        pos_I[j, 0] = pos_I[j, 0] % L
        pos_I[j, 1] = pos_I[j, 1] % L

    # ---- macrophage update ----
    if use_macrophages:
        N_Mmax = pos_M.shape[0]
        sqrt_2Dp_dt = np.sqrt(2.0 * D_p * dt)
        for k in range(N_Mmax):
            if not alive_M[k]:
                continue
            v_k = 0.2          # we keep v_M scalar to limit param sprawl
            D_R_k = 3.0
            sqrt_2DT = np.sqrt(2.0 * D_T_M * dt)
            sqrt_2DR = np.sqrt(2.0 * D_R_k * dt)
            cs = np.cos(theta_M[k]); sn = np.sin(theta_M[k])
            # weak chemotaxis up c_a (TAMs are recruited like T cells but weaker)
            gax, gay = grad_field_at(c_a, pos_M[k, 0], pos_M[k, 1], L)
            chemo_x = chi_a_M * gax
            chemo_y = chi_a_M * gay
            fx = f_MM[k, 0] + f_TM_M[k, 0] + f_IM_M[k, 0] + chemo_x
            fy = f_MM[k, 1] + f_TM_M[k, 1] + f_IM_M[k, 1] + chemo_y
            pos_M[k, 0] += dt * (v_k * cs + fx) + sqrt_2DT * np.random.normal()
            pos_M[k, 1] += dt * (v_k * sn + fy) + sqrt_2DT * np.random.normal()
            theta_M[k] += sqrt_2DR * np.random.normal()
            pos_M[k, 0] = pos_M[k, 0] % L
            pos_M[k, 1] = pos_M[k, 1] % L

            # polarization update — relax toward p_eq driven by local fields
            cs_local_x = pos_M[k, 0]; cs_local_y = pos_M[k, 1]
            # sample c_s and c_IL10 by bilinear (reuse grad function not needed; just need value)
            G = c_s.shape[0]
            dxg = L / G
            ix = int(cs_local_x / dxg) % G
            iy = int(cs_local_y / dxg) % G
            c_s_here = c_s[iy, ix]
            c_il_here = c_IL10[iy, ix]
            # equilibrium polarization: tanh(M1_bias - kappa_s c_s - kappa_il c_il)
            drive = M1_bias - kappa_s * c_s_here - kappa_il * c_il_here
            p_eq = np.tanh(drive)
            p_M[k] += dt * (p_eq - p_M[k]) / tau_p + sqrt_2Dp_dt * np.random.normal()
            # clamp polarization
            if p_M[k] > 1.0:
                p_M[k] = 1.0
            elif p_M[k] < -1.0:
                p_M[k] = -1.0

    # ---- killing (CD8 → tumor) ----
    n_killed = apply_killing(pos_T, alive_T, pos_I, alive_I, r_kill, p_kill, L)

    # ---- phagocytosis (M1 → tumor) ----
    n_phag = 0
    if use_macrophages:
        n_phag = apply_phagocytosis(
            pos_T, alive_T, pos_M, alive_M, p_M, r_phag, p_phag, L,
        )

    # ---- pressure-gated proliferation ----
    n_born = 0
    write_cursor = 0
    r2_nbr = nbr_radius * nbr_radius
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        # pressure gate (Byrne–Drasdo)
        pressure_factor = 1.0 - P_tumor[i] / P_star
        if pressure_factor <= 0.0:
            continue
        # combine with a soft random division roll
        if np.random.random() >= p_div0 * pressure_factor:
            continue
        # density gate (safety net at very high local crowding)
        cnt = count_neighbors_at(pos_T, alive_T, i, r2_nbr, L)
        if cnt >= nbr_thresh:
            continue
        # find empty slot
        slot = first_dead_slot(alive_T, write_cursor)
        if slot < 0:
            break
        write_cursor = slot + 1
        angle = 2.0 * np.pi * np.random.random()
        pos_T[slot, 0] = (pos_T[i, 0] + 0.3 * sigma_T * np.cos(angle)) % L
        pos_T[slot, 1] = (pos_T[i, 1] + 0.3 * sigma_T * np.sin(angle)) % L
        theta_T[slot] = 2.0 * np.pi * np.random.random()
        alive_T[slot] = True
        # inherit + drift
        v_T_arr[slot] = max(0.001, v_T_arr[i] + daughter_v_drift * np.random.normal())
        DR_T_arr[slot] = max(0.001, DR_T_arr[i] + daughter_DR_drift * np.random.normal())
        n_born += 1
    return n_born, n_killed, n_phag


# ===========================================================================
# IL-10 deposition + field stepping
# ===========================================================================

@njit(cache=True)
def deposit_M2_density(
    pos_M: np.ndarray, alive_M: np.ndarray, p_M: np.ndarray, L: float, G: int,
) -> np.ndarray:
    """Cloud-in-cell deposit weighted by max(0, -p_k) → only M2-skewed cells
    contribute to IL-10 source.
    """
    rho = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N_max = pos_M.shape[0]
    for k in range(N_max):
        if not alive_M[k]:
            continue
        w_M2 = -p_M[k] if p_M[k] < 0.0 else 0.0
        if w_M2 <= 0.0:
            continue
        gx = pos_M[k, 0] * inv_dx
        gy = pos_M[k, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        rho[iy0, ix0] += w00 * w_M2 * inv_cell_area
        rho[iy0, ix1] += w10 * w_M2 * inv_cell_area
        rho[iy1, ix0] += w01 * w_M2 * inv_cell_area
        rho[iy1, ix1] += w11 * w_M2 * inv_cell_area
    return rho


# ===========================================================================
# Run driver
# ===========================================================================

@dataclass
class ExtendedRun:
    pos_T_snapshots: list = field(default_factory=list)
    pos_I_snapshots: list = field(default_factory=list)
    pos_M_snapshots: list = field(default_factory=list)
    p_M_snapshots: list = field(default_factory=list)
    c_a_snapshots: list = field(default_factory=list)
    c_s_snapshots: list = field(default_factory=list)
    c_IL10_snapshots: list = field(default_factory=list)
    n_T: list = field(default_factory=list)
    n_I: list = field(default_factory=list)
    n_M: list = field(default_factory=list)
    mean_pM: list = field(default_factory=list)
    n_killed_cum: list = field(default_factory=list)
    n_phag_cum: list = field(default_factory=list)
    n_born_cum: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: Optional[ExtendedParams] = None
    final_tumor_fraction: float = 0.0


def run_extended(
    params: Optional[ExtendedParams] = None,
    seed: int = 0,
    snapshot_every: int = 50,
    save_fields: bool = True,
    # mid-run treatment knobs (default: none)
    treat_time: Optional[float] = None,
    chi_s_after: Optional[float] = None,
    M1_bias_after: Optional[float] = None,
) -> ExtendedRun:
    """Run the extended sim. Optionally apply a checkpoint-inhibitor-style
    treatment at `treat_time` by overriding chi_s and/or M1_bias.
    """
    if params is None:
        params = ExtendedParams()
    np.random.seed(seed)
    state = init_extended_state(params, seed=seed)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    v_T_arr = state["v_T_arr"]; DR_T_arr = state["DR_T_arr"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    v_I_arr = state["v_I_arr"]; DR_I_arr = state["DR_I_arr"]
    pos_M = state["pos_M"]; theta_M = state["theta_M"]; alive_M = state["alive_M"]
    p_M = state["p_M"]
    c_a = state["c_a"]; c_s = state["c_s"]; c_IL10 = state["c_IL10"]

    n_steps = params.n_steps
    out = ExtendedRun(params=params)
    cum_born = 0; cum_killed = 0; cum_phag = 0
    N_T_initial = int(alive_T.sum())
    chi_s_cur = params.chi_s
    M1_bias_cur = params.M1_bias
    if treat_time is not None:
        treat_step = int(round(treat_time / params.dt))
    else:
        treat_step = -1

    for step in range(n_steps):
        if step == treat_step:
            if chi_s_after is not None:
                chi_s_cur = float(chi_s_after)
            if M1_bias_after is not None:
                M1_bias_cur = float(M1_bias_after)

        n_born, n_killed, n_phag = _step_extended(
            pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
            pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
            pos_M, theta_M, alive_M, p_M,
            c_a, c_s, c_IL10,
            params.dt,
            params.sigma_T, params.k_rep_T,
            params.sigma_I, params.k_rep_I,
            params.sigma_TI, params.k_rep_TI,
            params.sigma_M, params.k_rep_M,
            params.sigma_TM, params.k_rep_TM,
            params.sigma_IM, params.k_rep_IM,
            params.L,
            params.D_T_T, params.D_T_I, params.D_T_M,
            params.chi_a, chi_s_cur, params.chi_a_M,
            params.r_kill, params.p_kill,
            params.r_phag, params.p_phag,
            params.p_div, params.P_star,
            params.nbr_radius, params.nbr_threshold,
            params.tau_p, params.D_p,
            params.kappa_s, params.kappa_il, M1_bias_cur,
            params.daughter_v_drift, params.daughter_DR_drift,
            params.use_macrophages,
        )
        cum_born += n_born; cum_killed += n_killed; cum_phag += n_phag

        # ---- field updates: c_a, c_s, c_IL10 ----
        rho_T = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        c_a, c_s = step_fields(
            c_a, c_s, rho_T,
            params.D_a, params.D_s, params.s_a, params.s_s,
            params.lam_a, params.lam_s, params.dt, params.L,
        )
        # IL-10 from M2-skewed macrophages
        rho_M2 = deposit_M2_density(pos_M, alive_M, p_M, params.L, params.G)
        dx_grid = params.L / params.G
        n_sub = n_substeps_for_cfl(params.D_IL10, params.dt, dx_grid, 0.25)
        dt_sub = params.dt / n_sub
        for _ in range(n_sub):
            c_IL10 = step_field_one_substep(
                c_IL10, rho_M2,
                params.D_IL10, params.s_IL10, params.lam_IL10,
                dt_sub, dx_grid,
            )

        # short-circuit extinction
        if step > 100 and not alive_T.any():
            break

        if step % snapshot_every == 0 or step == n_steps - 1:
            out.pos_T_snapshots.append(pos_T[alive_T].copy())
            out.pos_I_snapshots.append(pos_I[alive_I].copy())
            if params.use_macrophages:
                mask = alive_M
                out.pos_M_snapshots.append(pos_M[mask].copy())
                out.p_M_snapshots.append(p_M[mask].copy())
            else:
                out.pos_M_snapshots.append(np.zeros((0, 2)))
                out.p_M_snapshots.append(np.zeros(0))
            if save_fields:
                out.c_a_snapshots.append(c_a.copy())
                out.c_s_snapshots.append(c_s.copy())
                out.c_IL10_snapshots.append(c_IL10.copy())
            out.n_T.append(int(alive_T.sum()))
            out.n_I.append(int(alive_I.sum()))
            out.n_M.append(int(alive_M.sum()) if params.use_macrophages else 0)
            if params.use_macrophages and alive_M.any():
                out.mean_pM.append(float(p_M[alive_M].mean()))
            else:
                out.mean_pM.append(0.0)
            out.n_killed_cum.append(cum_killed)
            out.n_phag_cum.append(cum_phag)
            out.n_born_cum.append(cum_born)
            out.times.append(step * params.dt)

    frac = (out.n_T[-1] if out.n_T else 0) / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 1e2))
    return out
