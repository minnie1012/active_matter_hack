"""Extended tumor-immune simulation with an ECM density field and MMP.

Implements the equations in `docs/extensions.md` §3 (ECM density / porosity)
and §6 (MMP-driven ECM degradation), layered on top of `src.sim_extended`.

Two new fields live on the same (G, G) grid as `c_a`, `c_s`, `c_IL10`:

  rho_E(x, t) — extracellular matrix density.  Local sink from MMP and slow
                 logistic recovery toward rho_E_init.
  m(x, t)     — MMP concentration.  Tumor-density-sourced, diffusive, decays
                 with rate lam_m.

Coupling to tumor motion (§3):
  - Effective drag: v_T -> v_T / (1 + beta_drag * rho_E_local).
  - Pore-size gate: if r_p = r_0 / sqrt(rho_E_local) < r_p_star, the cell's
    self-propulsion is zeroed (it's stuck) UNLESS the local MMP exceeds a
    threshold (which "opens" the pore by digestion).

ECM/MMP update (§6):
  d rho_E / dt = -k_deg * m * rho_E + k_rep_ECM * rho_E * (rho_E_init - rho_E)
  d m / dt     = D_m * lap(m) + s_m * rho_T - lam_m * m

The baseline `src.sim_extended.run_extended` is untouched; use
`run_extended_ecm()` from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numba import njit

from src.interactions import (
    count_neighbors_at,
    first_dead_slot,
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
class ECMParams(ExtendedParams):
    """Extended params + ECM density (§3) and MMP (§6) parameters.

    Inherits everything from ExtendedParams.  Defaults below are chosen so
    that with rho_E_init=0 the model reduces exactly to the macrophage sim.
    """
    # --- MMP field (sourced by tumor density) -----------------------------
    D_m: float = 3.0
    s_m: float = 0.0                 # 0 → MMP off; raise to enable degradation
    lam_m: float = 0.15

    # --- ECM density field ------------------------------------------------
    rho_E_init: float = 0.0          # 0 → ECM disabled (drag = 1, no gate)
    k_deg: float = 1.0               # MMP × ECM degradation rate
    k_rep_ECM: float = 0.005         # logistic recovery rate toward rho_E_init

    # --- Drag + pore-size gate (§3) ---------------------------------------
    beta_drag: float = 0.8           # γ(ρ_E) = γ0 (1 + β ρ_E)  → v_eff /= (1+β ρ_E)
    r_0: float = 1.0                 # baseline pore radius constant
    r_p_star: float = 0.55           # nuclear cross-section pore threshold
    mmp_open_thresh: float = 0.10    # MMP level above which pores are "opened"

    # --- numeric ----------------------------------------------------------
    ecm_init_jitter: float = 0.0     # optional small noise on the initial ECM


# ===========================================================================
# Bilinear sample of a scalar field (numba-friendly)
# ===========================================================================

@njit(cache=True, fastmath=True)
def sample_field_at(c: np.ndarray, x: float, y: float, L: float) -> float:
    """Bilinear sample of a (G, G) periodic field at world coord (x, y)."""
    G = c.shape[0]
    dx = L / G
    inv_dx = 1.0 / dx
    gx = x * inv_dx
    gy = y * inv_dx
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    ix0 = ix % G
    iy0 = iy % G
    ix1 = (ix + 1) % G
    iy1 = (iy + 1) % G
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    return (w00 * c[iy0, ix0] + w10 * c[iy0, ix1]
            + w01 * c[iy1, ix0] + w11 * c[iy1, ix1])


# ===========================================================================
# JIT step — extended dynamics with ECM drag / pore gate on tumor motion
# ===========================================================================

@njit(cache=True, fastmath=True)
def _step_extended_ecm(
    # tumor
    pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
    # T cell
    pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
    # macrophage
    pos_M, theta_M, alive_M, p_M,
    # fields
    c_a, c_s, c_IL10,
    rho_E, m_field,
    # scalars
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
    # ECM / MMP scalars
    beta_drag, r_0, r_p_star, mmp_open_thresh,
):
    """One step of the ECM-extended dynamics.

    Returns (n_born, n_killed_cd8, n_phag, n_stuck) where n_stuck counts
    tumor cells whose self-propulsion was zeroed by the pore gate this step
    (diagnostic; doesn't affect dynamics).
    """
    # ---- pairwise forces + pressure on tumor ----
    f_TT, P_TT = pairwise_with_pressure(pos_T, alive_T, sigma_T, k_rep_T, L)
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

    # ---- tumor update — with ECM drag and pore-size gate ----
    N_Tmax = pos_T.shape[0]
    n_stuck = 0
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        v_i = v_T_arr[i]
        D_R_i = DR_T_arr[i]

        # local ECM and MMP at the cell position
        rE_local = sample_field_at(rho_E, pos_T[i, 0], pos_T[i, 1], L)
        if rE_local < 0.0:
            rE_local = 0.0
        m_local = sample_field_at(m_field, pos_T[i, 0], pos_T[i, 1], L)

        # drag factor:  1 / (1 + beta * rho_E)
        drag_factor = 1.0 / (1.0 + beta_drag * rE_local)

        # pore-size gate:  r_p = r_0 / sqrt(rho_E) below threshold and no MMP
        v_eff = v_i * drag_factor
        if rE_local > 1e-9:
            r_pore = r_0 / np.sqrt(rE_local)
            if r_pore < r_p_star and m_local < mmp_open_thresh:
                v_eff = 0.0
                n_stuck += 1

        sqrt_2DT = np.sqrt(2.0 * D_T_T * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_i * dt)
        cs = np.cos(theta_T[i])
        sn = np.sin(theta_T[i])
        # forces scale by the same drag factor (overdamped: r-dot = F/γ)
        fx = (f_TT[i, 0] + f_TI_T[i, 0] + f_TM_T[i, 0]) * drag_factor
        fy = (f_TT[i, 1] + f_TI_T[i, 1] + f_TM_T[i, 1]) * drag_factor
        pos_T[i, 0] += dt * (v_eff * cs + fx) + sqrt_2DT * np.random.normal()
        pos_T[i, 1] += dt * (v_eff * sn + fy) + sqrt_2DT * np.random.normal()
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
        cs = np.cos(theta_I[j])
        sn = np.sin(theta_I[j])
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
            cs = np.cos(theta_M[k])
            sn = np.sin(theta_M[k])
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

            # polarization update
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

    # ---- killing (CD8 → tumor) ----
    n_killed = apply_killing(pos_T, alive_T, pos_I, alive_I, r_kill, p_kill, L)

    # ---- phagocytosis (M1 → tumor) ----
    n_phag = 0
    if use_macrophages:
        n_phag = apply_phagocytosis(
            pos_T, alive_T, pos_M, alive_M, p_M, r_phag, p_phag, L,
        )

    # ---- pressure-gated proliferation (unchanged from sim_extended) ----
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
    return n_born, n_killed, n_phag, n_stuck


# ===========================================================================
# ECM in-place update (logistic recovery + MMP-driven degradation)
# ===========================================================================

@njit(cache=True, fastmath=True)
def update_rho_E_inplace(
    rho_E: np.ndarray, m_field: np.ndarray,
    k_deg: float, k_rep_ECM: float, rho_E_init: float, dt: float,
):
    """Local ODE step (no diffusion) on rho_E:

        d rho_E / dt = -k_deg * m * rho_E + k_rep_ECM * rho_E * (rho_E_init - rho_E)

    The logistic recovery term re-fills degraded matrix toward the steady
    baseline.  Step in-place with an explicit Euler since timescales are
    slow.
    """
    G = rho_E.shape[0]
    for i in range(G):
        for j in range(G):
            r = rho_E[i, j]
            m = m_field[i, j]
            dr = (-k_deg * m * r + k_rep_ECM * r * (rho_E_init - r))
            r_new = r + dt * dr
            if r_new < 0.0:
                r_new = 0.0
            rho_E[i, j] = r_new


# ===========================================================================
# State init / run driver
# ===========================================================================

def init_ecm_state(params: ECMParams, seed: int = 0) -> dict:
    """Initialize the base extended state plus rho_E and m fields."""
    state = init_extended_state(params, seed=seed)
    G = params.G
    # rho_E: uniform baseline + optional jitter
    rng = np.random.default_rng(seed + 991)
    rho_E = np.full((G, G), float(params.rho_E_init), dtype=np.float64)
    if params.ecm_init_jitter > 0.0:
        rho_E += params.ecm_init_jitter * rng.standard_normal((G, G))
        rho_E = np.clip(rho_E, 0.0, None)
    m_field = np.zeros((G, G), dtype=np.float64)
    state["rho_E"] = rho_E
    state["m_field"] = m_field
    return state


@dataclass
class ECMRun:
    pos_T_snapshots: list = field(default_factory=list)
    pos_I_snapshots: list = field(default_factory=list)
    pos_M_snapshots: list = field(default_factory=list)
    p_M_snapshots: list = field(default_factory=list)
    c_a_snapshots: list = field(default_factory=list)
    c_s_snapshots: list = field(default_factory=list)
    c_IL10_snapshots: list = field(default_factory=list)
    rho_E_snapshots: list = field(default_factory=list)
    m_snapshots: list = field(default_factory=list)
    n_T: list = field(default_factory=list)
    n_I: list = field(default_factory=list)
    n_M: list = field(default_factory=list)
    mean_pM: list = field(default_factory=list)
    n_killed_cum: list = field(default_factory=list)
    n_phag_cum: list = field(default_factory=list)
    n_born_cum: list = field(default_factory=list)
    n_stuck_cum: list = field(default_factory=list)
    mean_rho_E: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: Optional[ECMParams] = None
    final_tumor_fraction: float = 0.0


def run_extended_ecm(
    params: Optional[ECMParams] = None,
    seed: int = 0,
    snapshot_every: int = 50,
    save_fields: bool = True,
    treat_time: Optional[float] = None,
    chi_s_after: Optional[float] = None,
    M1_bias_after: Optional[float] = None,
    s_m_after: Optional[float] = None,
) -> ECMRun:
    """Run the extended simulation with ECM + MMP fields.

    Mirrors `src.sim_extended.run_extended` plus an extra `s_m_after`
    treatment knob (turning MMP on/off mid-run, e.g. to model an MMP
    inhibitor).
    """
    if params is None:
        params = ECMParams()
    np.random.seed(seed)
    state = init_ecm_state(params, seed=seed)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    v_T_arr = state["v_T_arr"]; DR_T_arr = state["DR_T_arr"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    v_I_arr = state["v_I_arr"]; DR_I_arr = state["DR_I_arr"]
    pos_M = state["pos_M"]; theta_M = state["theta_M"]; alive_M = state["alive_M"]
    p_M = state["p_M"]
    c_a = state["c_a"]; c_s = state["c_s"]; c_IL10 = state["c_IL10"]
    rho_E = state["rho_E"]; m_field = state["m_field"]

    n_steps = params.n_steps
    out = ECMRun(params=params)
    cum_born = cum_killed = cum_phag = cum_stuck = 0
    N_T_initial = int(alive_T.sum())
    chi_s_cur = params.chi_s
    M1_bias_cur = params.M1_bias
    s_m_cur = params.s_m
    treat_step = int(round(treat_time / params.dt)) if treat_time is not None else -1
    dx_grid = params.L / params.G

    for step in range(n_steps):
        if step == treat_step:
            if chi_s_after is not None:
                chi_s_cur = float(chi_s_after)
            if M1_bias_after is not None:
                M1_bias_cur = float(M1_bias_after)
            if s_m_after is not None:
                s_m_cur = float(s_m_after)

        n_born, n_killed, n_phag, n_stuck = _step_extended_ecm(
            pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
            pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
            pos_M, theta_M, alive_M, p_M,
            c_a, c_s, c_IL10,
            rho_E, m_field,
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
            params.beta_drag, params.r_0, params.r_p_star, params.mmp_open_thresh,
        )
        cum_born += n_born; cum_killed += n_killed
        cum_phag += n_phag; cum_stuck += n_stuck

        # ---- field updates: c_a, c_s, c_IL10 (as before) ----
        rho_T = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        c_a, c_s = step_fields(
            c_a, c_s, rho_T,
            params.D_a, params.D_s, params.s_a, params.s_s,
            params.lam_a, params.lam_s, params.dt, params.L,
        )
        rho_M2 = deposit_M2_density(pos_M, alive_M, p_M, params.L, params.G)
        n_sub_il = n_substeps_for_cfl(params.D_IL10, params.dt, dx_grid, 0.25)
        dt_sub_il = params.dt / n_sub_il
        for _ in range(n_sub_il):
            c_IL10 = step_field_one_substep(
                c_IL10, rho_M2,
                params.D_IL10, params.s_IL10, params.lam_IL10,
                dt_sub_il, dx_grid,
            )

        # ---- ECM and MMP update ----
        # MMP: tumor-sourced, diffusive, decays. Re-use the standard substep.
        n_sub_m = n_substeps_for_cfl(params.D_m, params.dt, dx_grid, 0.25)
        dt_sub_m = params.dt / n_sub_m
        for _ in range(n_sub_m):
            m_field = step_field_one_substep(
                m_field, rho_T,
                params.D_m, s_m_cur, params.lam_m,
                dt_sub_m, dx_grid,
            )
        # rho_E: pure local ODE (degradation by MMP + logistic recovery)
        if params.rho_E_init > 0.0 or params.k_deg > 0.0:
            update_rho_E_inplace(
                rho_E, m_field,
                params.k_deg, params.k_rep_ECM, params.rho_E_init, params.dt,
            )

        if step > 100 and not alive_T.any():
            break

        if step % snapshot_every == 0 or step == n_steps - 1:
            out.pos_T_snapshots.append(pos_T[alive_T].copy())
            out.pos_I_snapshots.append(pos_I[alive_I].copy())
            if params.use_macrophages:
                out.pos_M_snapshots.append(pos_M[alive_M].copy())
                out.p_M_snapshots.append(p_M[alive_M].copy())
            else:
                out.pos_M_snapshots.append(np.zeros((0, 2)))
                out.p_M_snapshots.append(np.zeros(0))
            if save_fields:
                out.c_a_snapshots.append(c_a.copy())
                out.c_s_snapshots.append(c_s.copy())
                out.c_IL10_snapshots.append(c_IL10.copy())
                out.rho_E_snapshots.append(rho_E.copy())
                out.m_snapshots.append(m_field.copy())
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
            out.n_stuck_cum.append(cum_stuck)
            out.mean_rho_E.append(float(rho_E.mean()))
            out.times.append(step * params.dt)

    frac = (out.n_T[-1] if out.n_T else 0) / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 1e2))
    return out
