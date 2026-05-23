"""Cell-cell adhesion extension (cadherins, factor #1 in extensions.md).

This module *augments* `src.sim_extended` with an attractive harmonic shoulder
in the tumor-tumor pairwise force plus an optional Vicsek-style alignment
torque on cadherin neighbors. Cadherins are an epithelial marker, so by
default the adhesion only operates between tumor cells; T-tumor and
M-tumor cross interactions fall back to the existing cross-species
repulsion.

Force on tumor i from neighbor tumor j (distance r, unit r_hat = (r_i-r_j)/r):

    F_ij = +k_rep * max(0, sigma - r) * r_hat                            (existing repulsion)
         - k_adh * max(0, r - sigma) * max(0, sigma_adh - r) * r_hat     (NEW attractive shoulder)

Alignment torque on tumor i over its cadherin neighbors (sigma < r_ij < sigma_adh):

    dtheta_i/dt += (J_align / N_i) * sum_j sin(theta_j - theta_i)

Public API:
    AdhesionParams        — dataclass extending ExtendedParams
    pairwise_with_adhesion_and_pressure(...)  — @njit force+pressure+torque
    run_adhesion(params, seed, ...)            — driver mirroring run_extended
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    step_fields,
    step_field_one_substep,
    n_substeps_for_cfl,
    grad_field_at,
)
from src.sim_extended import (
    ExtendedParams,
    ExtendedRun,
    init_extended_state,
    pairwise_with_pressure,
    cross_repulsion_with_pressure_a,
    apply_phagocytosis,
    deposit_M2_density,
)


# ===========================================================================
# Parameter container
# ===========================================================================

@dataclass
class AdhesionParams(ExtendedParams):
    """ExtendedParams + cadherin adhesion + optional Vicsek alignment.

    New fields:
        k_adh       attractive shoulder stiffness
        sigma_adh   outer cutoff of attractive region (must be > sigma_T)
        J_align     Vicsek coupling on cadherin neighbors (per-step torque)
        apply_adhesion_to_T_only   if True (default), adhesion is tumor-tumor only
    """
    k_adh: float = 8.0
    sigma_adh: float = 1.6
    J_align: float = 0.5
    apply_adhesion_to_T_only: bool = True


# ===========================================================================
# JIT helpers
# ===========================================================================

@njit(cache=True)
def pairwise_with_adhesion_and_pressure(
    pos: np.ndarray,
    alive: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    sigma_adh: float,
    k_rep: float,
    k_adh: float,
    J_align: float,
    L: float,
):
    """O(N^2) repulsion + attractive shoulder + pressure + Vicsek torque.

    Returns
    -------
    forces[N_max, 2]        : net pairwise force vector on each cell
    pressure[N_max]         : per-cell mechanical pressure (virial / pi sigma^2)
    align_torque[N_max]     : Vicsek torque   (J/N_i) * sum_j sin(theta_j-theta_i)

    Notes
    -----
    * Newton 3rd law applied to both species' force terms.
    * The pressure virial is computed from the *signed* radial force component,
      following the same convention as `pairwise_with_pressure` in
      sim_extended.py — repulsion contributes positive pressure, adhesion
      contributes negative pressure (cells under tension).
    * The alignment sum runs only over j in the attractive shoulder
      (sigma < r < sigma_adh), divided by N_i = number of such neighbors.
    """
    N_max = pos.shape[0]
    forces = np.zeros((N_max, 2), dtype=np.float64)
    pressure = np.zeros(N_max, dtype=np.float64)
    align_sum = np.zeros(N_max, dtype=np.float64)
    align_cnt = np.zeros(N_max, dtype=np.int64)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
    sigma_adh2 = sigma_adh * sigma_adh
    A_inv = 1.0 / (np.pi * sigma2)

    for i in range(N_max):
        if not alive[i]:
            continue
        xi = pos[i, 0]; yi = pos[i, 1]
        ti = theta[i]
        for j in range(i + 1, N_max):
            if not alive[j]:
                continue
            dx = xi - pos[j, 0]; dy = yi - pos[j, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < 1e-12:
                continue
            if r2 < sigma2:
                # ---- repulsive core ----
                r = np.sqrt(r2)
                fmag_over_r = k_rep * (sigma - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                forces[i, 0] += fx; forces[i, 1] += fy
                forces[j, 0] -= fx; forces[j, 1] -= fy
                virial = (dx * fx + dy * fy) * A_inv
                pressure[i] += virial
                pressure[j] += virial
            elif r2 < sigma_adh2:
                # ---- attractive shoulder (cadherin) ----
                r = np.sqrt(r2)
                # F on i = -k_adh * (r - sigma)*(sigma_adh - r) * r_hat_ij
                # r_hat_ij = (r_i - r_j) / r  = (dx, dy) / r
                # so contribution along (dx, dy): coef * dx/r, coef * dy/r
                # with coef = -k_adh * (r - sigma)*(sigma_adh - r)
                # we write it as fmag_over_r * dx with fmag_over_r negative.
                fmag_over_r = -k_adh * (r - sigma) * (sigma_adh - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                forces[i, 0] += fx; forces[i, 1] += fy
                forces[j, 0] -= fx; forces[j, 1] -= fy
                virial = (dx * fx + dy * fy) * A_inv
                pressure[i] += virial
                pressure[j] += virial
                # ---- Vicsek alignment over cadherin neighbors ----
                dth = theta[j] - ti
                s_ij = np.sin(dth)
                align_sum[i] += s_ij
                align_sum[j] -= s_ij          # sin(t_i - t_j) = -sin(t_j - t_i)
                align_cnt[i] += 1
                align_cnt[j] += 1

    # finalize torque: (J / N_i) * sum
    align_torque = np.zeros(N_max, dtype=np.float64)
    if J_align != 0.0:
        for i in range(N_max):
            if alive[i] and align_cnt[i] > 0:
                align_torque[i] = J_align * align_sum[i] / align_cnt[i]

    return forces, pressure, align_torque


# ===========================================================================
# JIT inner step  (mirrors `_step_extended` but with adhesion on tumor-tumor)
# ===========================================================================

@njit(cache=True)
def _step_adhesion(
    # tumor
    pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
    # T cell
    pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
    # macrophage
    pos_M, theta_M, alive_M, p_M,
    # fields
    c_a, c_s, c_IL10,
    # core scalars
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
    # adhesion scalars (new)
    k_adh, sigma_adh, J_align,
):
    """One step of the extended dynamics with cadherin adhesion on tumor-tumor.

    Returns (n_born, n_killed_cd8, n_phag).
    """
    # ---- tumor-tumor: cadherin adhesion + repulsion + pressure + torque ----
    f_TT, P_TT, align_T = pairwise_with_adhesion_and_pressure(
        pos_T, alive_T, theta_T,
        sigma_T, sigma_adh, k_rep_T, k_adh, J_align, L,
    )
    # ---- other pairwise forces (no adhesion: cadherins are within-species) ----
    f_II = pairwise_with_pressure(pos_I, alive_I, sigma_I, k_rep_I, L)[0]
    f_TI_T, f_TI_I, P_TI = cross_repulsion_with_pressure_a(
        pos_T, alive_T, pos_I, alive_I, sigma_TI, k_rep_TI, L,
    )
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

    P_tumor = P_TT + P_TI + P_TM

    # ---- tumor update (+ Vicsek torque applied additively to theta) ----
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
        # alignment torque (deterministic) + rotational diffusion
        theta_T[i] += dt * align_T[i] + sqrt_2DR * np.random.normal()
        pos_T[i, 0] = pos_T[i, 0] % L
        pos_T[i, 1] = pos_T[i, 1] % L

    # ---- T-cell update ----
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
            v_k = 0.2
            D_R_k = 3.0
            sqrt_2DT = np.sqrt(2.0 * D_T_M * dt)
            sqrt_2DR = np.sqrt(2.0 * D_R_k * dt)
            cs = np.cos(theta_M[k]); sn = np.sin(theta_M[k])
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

            G = c_s.shape[0]
            dxg = L / G
            ix = int(pos_M[k, 0] / dxg) % G
            iy = int(pos_M[k, 1] / dxg) % G
            c_s_here = c_s[iy, ix]
            c_il_here = c_IL10[iy, ix]
            drive = M1_bias - kappa_s * c_s_here - kappa_il * c_il_here
            p_eq = np.tanh(drive)
            p_M[k] += dt * (p_eq - p_M[k]) / tau_p + sqrt_2Dp_dt * np.random.normal()
            if p_M[k] > 1.0:
                p_M[k] = 1.0
            elif p_M[k] < -1.0:
                p_M[k] = -1.0

    # ---- killing & phagocytosis ----
    n_killed = apply_killing(pos_T, alive_T, pos_I, alive_I, r_kill, p_kill, L)
    n_phag = 0
    if use_macrophages:
        n_phag = apply_phagocytosis(
            pos_T, alive_T, pos_M, alive_M, p_M, r_phag, p_phag, L,
        )

    # ---- pressure-gated proliferation (uses tumor-tumor pressure) ----
    n_born = 0
    write_cursor = 0
    r2_nbr = nbr_radius * nbr_radius
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        pressure_factor = 1.0 - P_tumor[i] / P_star
        if pressure_factor <= 0.0:
            continue
        if np.random.random() >= p_div0 * pressure_factor:
            continue
        cnt = count_neighbors_at(pos_T, alive_T, i, r2_nbr, L)
        if cnt >= nbr_thresh:
            continue
        slot = first_dead_slot(alive_T, write_cursor)
        if slot < 0:
            break
        write_cursor = slot + 1
        angle = 2.0 * np.pi * np.random.random()
        pos_T[slot, 0] = (pos_T[i, 0] + 0.3 * sigma_T * np.cos(angle)) % L
        pos_T[slot, 1] = (pos_T[i, 1] + 0.3 * sigma_T * np.sin(angle)) % L
        theta_T[slot] = 2.0 * np.pi * np.random.random()
        alive_T[slot] = True
        v_T_arr[slot] = max(0.001, v_T_arr[i] + daughter_v_drift * np.random.normal())
        DR_T_arr[slot] = max(0.001, DR_T_arr[i] + daughter_DR_drift * np.random.normal())
        n_born += 1
    return n_born, n_killed, n_phag


# ===========================================================================
# Run driver
# ===========================================================================

def run_adhesion(
    params: Optional[AdhesionParams] = None,
    seed: int = 0,
    snapshot_every: int = 50,
    save_fields: bool = True,
    treat_time: Optional[float] = None,
    chi_s_after: Optional[float] = None,
    M1_bias_after: Optional[float] = None,
) -> ExtendedRun:
    """Run the extended sim with cadherin adhesion. Mirrors `run_extended`.

    Returns an `ExtendedRun` (same schema; macrophage fields may be empty
    if use_macrophages is False).
    """
    if params is None:
        params = AdhesionParams()
    if params.sigma_adh <= params.sigma_T:
        raise ValueError(
            f"sigma_adh ({params.sigma_adh}) must be > sigma_T ({params.sigma_T})"
        )

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
    treat_step = int(round(treat_time / params.dt)) if treat_time is not None else -1

    for step in range(n_steps):
        if step == treat_step:
            if chi_s_after is not None:
                chi_s_cur = float(chi_s_after)
            if M1_bias_after is not None:
                M1_bias_cur = float(M1_bias_after)

        n_born, n_killed, n_phag = _step_adhesion(
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
            params.k_adh, params.sigma_adh, params.J_align,
        )
        cum_born += n_born; cum_killed += n_killed; cum_phag += n_phag

        # ---- field updates ----
        rho_T = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        c_a, c_s = step_fields(
            c_a, c_s, rho_T,
            params.D_a, params.D_s, params.s_a, params.s_s,
            params.lam_a, params.lam_s, params.dt, params.L,
        )
        if params.use_macrophages:
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
