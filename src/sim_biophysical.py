"""Biophysical tumor-invasion simulator with EMT, ECM fiber alignment, and hypoxia.

This module extends ``sim_combined`` (macrophages + ECM/MMP + cadherin adhesion)
with three additional coarse-grained biophysical mechanisms aimed at tumor
invasion dynamics:

1. **EMT state per tumor cell** (``s_T`` ∈ [0, 1]). EMT couples adhesion,
   motility, and ECM degradation into one tumor-cell phenotype: mesenchymal
   cells lose cadherins, gain speed and secrete more MMP, while epithelial
   cells stick together and barely degrade matrix.
2. **ECM fiber orientation field** (``theta_E``) + integrin biphasic traction.
   Collagen fibers provide contact guidance, acting like tracks for invasive
   migration. Tumor cells migrate best at intermediate cell-ECM adhesion
   because too little adhesion gives poor traction, while too much adhesion
   causes sticking.
3. **Hypoxia / oxygen field** (``c_O2``). Low oxygen promotes EMT, MMP
   secretion, and immune suppression in the tumor microenvironment.

Setting ``J_fiber=0``, ``D_EMT=0``, ``s_T0=0``, ``k_hypoxia_EMT=0``,
``k_ECM_EMT=0``, ``k_supp_EMT=0``, ``D_O=0`` should reduce the dynamics to
``sim_combined`` (with O2 still tracked but inert).
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
    pairwise_with_pressure,
    cross_repulsion_with_pressure_a,
    apply_phagocytosis,
    deposit_M2_density,
)
from src.sim_ecm import (
    ECMRun,
    init_ecm_state,
    sample_field_at,
    update_rho_E_inplace,
)
from src.sim_combined import CombinedParams
from src.sim_tme import deposit_vessel_O2_source


# ===========================================================================
# Parameter container
# ===========================================================================

@dataclass
class BiophysicalParams(CombinedParams):
    """CombinedParams + EMT state, ECM fiber alignment, and hypoxia.

    All new mechanisms can be cleanly disabled to recover ``sim_combined``
    behaviour (see module docstring for the zero-out recipe).
    """
    # ---- EMT state knobs -------------------------------------------------
    s_T0: float = 0.15
    D_EMT: float = 0.002
    daughter_EMT_drift: float = 0.03
    k_hypoxia_EMT: float = 0.6
    k_ECM_EMT: float = 0.10
    k_supp_EMT: float = 0.05
    k_MET: float = 0.05
    v_epi: float = 0.06
    v_mes: float = 0.30
    s_m_epi: float = 0.0       # epithelial cells secrete no MMP
    s_m_mes: float = 1.5       # mesenchymal cells fully secrete

    # ---- ECM fibers + integrin -------------------------------------------
    J_fiber: float = 1.5
    fiber_alignment_strength: float = 0.5
    A_base: float = 1.5

    # ---- Hypoxia / O2 field ----------------------------------------------
    D_O: float = 4.0
    supply_O: float = 1.0
    consumption_T: float = 0.5
    lambda_O: float = 0.05
    O_threshold: float = 8.0   # = 0.4 * supply_O / lambda_O ; user can tune
    k_MMP_hyp: float = 2.0
    k_supp_hyp: float = 1.0
    k_div_hyp: float = 0.5

    # ---- VEGF field (sourced by hypoxic tumor cells) ---------------------
    D_VEGF: float = 6.0
    lam_VEGF: float = 0.10
    s_VEGF_hyp: float = 1.0

    # ---- Vessels / angiogenesis ------------------------------------------
    n_vessels_init: int = 1
    n_vessels_max: int = 64
    vessel_edge_margin: float = 2.0
    chi_vessel: float = 80.0
    D_vessel: float = 0.02
    sprout_rate: float = 0.03
    sprout_VEGF_thresh: float = 0.005
    s_O2_vessel: float = 1.0   # how much each vessel pumps into c_O2 per step


# ===========================================================================
# BiophysicalOut — extends the ECMRun data container with EMT / O2 fields
# ===========================================================================

@dataclass
class BiophysicalOut(ECMRun):
    s_T_snapshots: list = field(default_factory=list)
    O_snapshots: list = field(default_factory=list)
    H_snapshots: list = field(default_factory=list)
    theta_E: Optional[np.ndarray] = None       # static; saved once
    mean_EMT: list = field(default_factory=list)
    frac_mes: list = field(default_factory=list)
    max_invasion_distance: list = field(default_factory=list)
    n_detached: list = field(default_factory=list)
    # final-summary scalars
    final_tumor_size: int = 0
    ecm_degraded_area: float = 0.0
    hypoxic_area: float = 0.0
    # vessel snapshots (angiogenesis tree)
    vessel_snapshots: list = field(default_factory=list)
    vessel_parent_snapshots: list = field(default_factory=list)
    n_vessels: list = field(default_factory=list)
    c_VEGF_snapshots: list = field(default_factory=list)


# ===========================================================================
# Numba-friendly helper: pairwise tumor-tumor with per-cell adhesion / align
# ===========================================================================

@njit(cache=True, fastmath=True)
def pairwise_adhesion_per_cell(
    pos: np.ndarray,
    alive: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    sigma_adh: float,
    k_rep: float,
    k_adh_per_cell: np.ndarray,
    J_align_per_cell: np.ndarray,
    L: float,
):
    """Tumor-tumor pairwise with per-cell cadherin / Vicsek strengths.

    Mirrors ``sim_adhesion.pairwise_with_adhesion_and_pressure`` but uses an
    averaged per-cell ``k_adh`` and per-cell ``J_align``. This way an EMT
    transition reduces the cadherin / alignment a cell experiences without
    requiring a per-pair table.

    Symmetric force splitting: ``k_eff = 0.5 * (k_adh_per_cell[i] +
    k_adh_per_cell[j])``. For the torque, cell i's update uses
    ``J_align_per_cell[i]`` (i.e. the cell whose theta we update sets the
    strength of its own alignment).
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
        k_adh_i = k_adh_per_cell[i]
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
                # ---- cadherin attractive shoulder, EMT-averaged stiffness ----
                r = np.sqrt(r2)
                k_eff = 0.5 * (k_adh_i + k_adh_per_cell[j])
                fmag_over_r = -k_eff * (r - sigma) * (sigma_adh - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                forces[i, 0] += fx; forces[i, 1] += fy
                forces[j, 0] -= fx; forces[j, 1] -= fy
                virial = (dx * fx + dy * fy) * A_inv
                pressure[i] += virial
                pressure[j] += virial
                # alignment counter (only over the cadherin shoulder)
                dth = theta[j] - ti
                s_ij = np.sin(dth)
                align_sum[i] += s_ij
                align_sum[j] -= s_ij
                align_cnt[i] += 1
                align_cnt[j] += 1

    # finalize torque with PER-CELL J
    align_torque = np.zeros(N_max, dtype=np.float64)
    for i in range(N_max):
        if alive[i] and align_cnt[i] > 0:
            align_torque[i] = J_align_per_cell[i] * align_sum[i] / align_cnt[i]

    return forces, pressure, align_torque


# ===========================================================================
# CIC deposit of a per-cell scalar onto a (G, G) grid (MMP source map)
# ===========================================================================

@njit(cache=True, fastmath=True)
def deposit_tumor_scalar(
    pos: np.ndarray, alive: np.ndarray, vals: np.ndarray,
    L: float, G: int,
) -> np.ndarray:
    """Cloud-in-cell deposit of a per-cell scalar weight (e.g. EMT-modulated
    MMP secretion rate) onto a (G, G) periodic grid. Returns the source map
    in the same units that ``s_m * rho`` would normally have.
    """
    out = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N_max = pos.shape[0]
    for i in range(N_max):
        if not alive[i]:
            continue
        w = vals[i]
        gx = pos[i, 0] * inv_dx
        gy = pos[i, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        out[iy0, ix0] += w * (1.0 - fx) * (1.0 - fy) * inv_cell_area
        out[iy0, ix1] += w * fx * (1.0 - fy) * inv_cell_area
        out[iy1, ix0] += w * (1.0 - fx) * fy * inv_cell_area
        out[iy1, ix1] += w * fx * fy * inv_cell_area
    return out


# ===========================================================================
# Oxygen field one explicit-Euler substep — uses per-cell tumor density
# ===========================================================================

@njit(cache=True, fastmath=True)
def step_O2_one_substep(
    c_O: np.ndarray, rho_T: np.ndarray,
    D_O: float, supply_O: float, consumption_T: float, lambda_O: float,
    dt_sub: float, dx: float,
) -> np.ndarray:
    """One explicit FTCS Euler step for the oxygen field with a tumor-coupled
    sink:

        dO/dt = D_O * lap(O) + supply_O - consumption_T * rho_T * O - lambda_O * O
    """
    G = c_O.shape[0]
    inv_dx2 = 1.0 / (dx * dx)
    c_new = np.empty_like(c_O)
    for i in range(G):
        im = (i - 1) % G
        ip = (i + 1) % G
        for j in range(G):
            jm = (j - 1) % G
            jp = (j + 1) % G
            lap = (c_O[ip, j] + c_O[im, j]
                   + c_O[i, jp] + c_O[i, jm] - 4.0 * c_O[i, j]) * inv_dx2
            sink = (consumption_T * rho_T[i, j] + lambda_O) * c_O[i, j]
            c_new[i, j] = c_O[i, j] + dt_sub * (D_O * lap + supply_O - sink)
            if c_new[i, j] < 0.0:
                c_new[i, j] = 0.0
    return c_new


# ===========================================================================
# Hypoxia level field H = max(0, O_thr - O) / O_thr
# ===========================================================================

@njit(cache=True, fastmath=True)
def hypoxia_field(c_O: np.ndarray, O_threshold: float) -> np.ndarray:
    """Coarse hypoxia level in [0, 1] across the grid.

    If ``O_threshold <= 0`` the field is zeroed (hypoxia disabled), allowing
    the user to disable the coupling without changing other knobs.
    """
    G = c_O.shape[0]
    H = np.zeros((G, G), dtype=np.float64)
    if O_threshold <= 0.0:
        return H
    inv_thr = 1.0 / O_threshold
    for i in range(G):
        for j in range(G):
            v = O_threshold - c_O[i, j]
            if v < 0.0:
                v = 0.0
            H[i, j] = v * inv_thr
    return H


# ===========================================================================
# Build the static fiber orientation field
# ===========================================================================

def make_fiber_field(L: float, G: int, alignment_strength: float,
                     seed: int = 0) -> np.ndarray:
    """Initialise the (G, G) fiber orientation field ``theta_E`` (radians).

    Collagen fibers provide contact guidance, acting like tracks for invasive
    migration. We interpolate between a fully-random nematic and a fully-radial
    orientation aimed away from the box center, controlled by
    ``alignment_strength`` (0 = random, 1 = radial).
    """
    rng = np.random.default_rng(int(seed) + 2027)
    theta_E = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    cx = cy = 0.5 * L
    for i in range(G):
        for j in range(G):
            x = (j + 0.5) * dx
            y = (i + 0.5) * dx
            dxc = x - cx
            dyc = y - cy
            # wrap to nearest image
            if dxc > 0.5 * L: dxc -= L
            elif dxc < -0.5 * L: dxc += L
            if dyc > 0.5 * L: dyc -= L
            elif dyc < -0.5 * L: dyc += L
            theta_radial = np.arctan2(dyc, dxc)
            theta_rand = rng.uniform(0.0, np.pi)
            # nematic blend on the circle (mod pi)
            a = float(alignment_strength)
            # use a "tangent-style" mix that respects nematic symmetry
            c2_blend = (1.0 - a) * np.cos(2.0 * theta_rand) + a * np.cos(2.0 * theta_radial)
            s2_blend = (1.0 - a) * np.sin(2.0 * theta_rand) + a * np.sin(2.0 * theta_radial)
            theta_E[i, j] = 0.5 * np.arctan2(s2_blend, c2_blend)
    return theta_E


# ===========================================================================
# JIT step — full biophysical kernel
# ===========================================================================

@njit(cache=True, fastmath=True)
def _step_biophysical(
    # tumor
    pos_T, theta_T, alive_T, v_T_arr, DR_T_arr, s_T,
    # T cell
    pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
    # macrophage
    pos_M, theta_M, alive_M, p_M,
    # fields
    c_a, c_s, c_IL10,
    rho_E, m_field, c_O2, theta_E, H_field,
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
    daughter_v_drift, daughter_DR_drift, daughter_EMT_drift,
    use_macrophages,
    # ECM / MMP scalars
    beta_drag, r_0, r_p_star, mmp_open_thresh,
    # adhesion scalars
    k_adh, sigma_adh, J_align,
    # EMT scalars
    D_EMT, k_hypoxia_EMT, k_ECM_EMT, k_supp_EMT, k_MET,
    v_epi, v_mes,
    # ECM fiber + integrin
    J_fiber, A_base,
    # hypoxia couplings on tumor proliferation
    k_div_hyp,
):
    """One step of the full biophysical dynamics.

    Returns (n_born, n_killed_cd8, n_phag, n_stuck).
    """
    N_Tmax = pos_T.shape[0]

    # ---- per-cell EMT update + derived per-cell phenotype arrays ----
    # EMT couples adhesion, motility, and ECM degradation into one phenotype.
    k_adh_per_cell = np.zeros(N_Tmax, dtype=np.float64)
    J_align_per_cell = np.zeros(N_Tmax, dtype=np.float64)
    v_eff_arr = np.zeros(N_Tmax, dtype=np.float64)

    sqrt_2DEMT = np.sqrt(2.0 * D_EMT * dt)
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        H_local = sample_field_at(H_field, pos_T[i, 0], pos_T[i, 1], L)
        if H_local < 0.0:
            H_local = 0.0
        rE_local = sample_field_at(rho_E, pos_T[i, 0], pos_T[i, 1], L)
        if rE_local < 0.0:
            rE_local = 0.0
        cs_local = sample_field_at(c_s, pos_T[i, 0], pos_T[i, 1], L)
        drift = (
            k_hypoxia_EMT * H_local
            + k_ECM_EMT * rE_local
            + k_supp_EMT * cs_local
            - k_MET * s_T[i]
        )
        s_new = s_T[i] + dt * drift + sqrt_2DEMT * np.random.normal()
        if s_new < 0.0:
            s_new = 0.0
        elif s_new > 1.0:
            s_new = 1.0
        s_T[i] = s_new
        k_adh_per_cell[i] = k_adh * (1.0 - s_new)
        J_align_per_cell[i] = J_align * (1.0 - s_new)
        v_eff_arr[i] = v_epi + (v_mes - v_epi) * s_new

    # ---- tumor-tumor: per-cell cadherin + alignment ----
    f_TT, P_TT, align_T = pairwise_adhesion_per_cell(
        pos_T, alive_T, theta_T,
        sigma_T, sigma_adh, k_rep_T,
        k_adh_per_cell, J_align_per_cell, L,
    )
    # ---- other pairwise forces (cross-species: plain repulsion) ----
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

    # ---- tumor update — ECM drag + pore gate + EMT + fiber + integrin ----
    n_stuck = 0
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        D_R_i = DR_T_arr[i]

        # local field samples
        rE_local = sample_field_at(rho_E, pos_T[i, 0], pos_T[i, 1], L)
        if rE_local < 0.0:
            rE_local = 0.0
        m_local = sample_field_at(m_field, pos_T[i, 0], pos_T[i, 1], L)
        # fiber orientation at the cell
        thE = sample_field_at(theta_E, pos_T[i, 0], pos_T[i, 1], L)

        # integrin biphasic traction:
        # "Tumor cells migrate best at intermediate cell-ECM adhesion
        #  because too little adhesion gives poor traction, while too
        #  much adhesion causes sticking."
        A_i = A_base * rE_local
        if A_i < 0.0:
            A_i = 0.0
        elif A_i > 1.0:
            A_i = 1.0
        traction = 4.0 * A_i * (1.0 - A_i)

        # combined drag + traction:  v_eff_i = v(s_T) * traction / (1 + beta * rho_E)
        drag_factor = 1.0 / (1.0 + beta_drag * rE_local)
        v_eff_i = v_eff_arr[i] * traction * drag_factor

        # pore-size gate: r_p = r_0 / sqrt(rho_E) below threshold and no MMP
        if rE_local > 1e-9:
            r_pore = r_0 / np.sqrt(rE_local)
            if r_pore < r_p_star and m_local < mmp_open_thresh:
                v_eff_i = 0.0
                n_stuck += 1

        sqrt_2DT = np.sqrt(2.0 * D_T_T * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_i * dt)
        cs_ = np.cos(theta_T[i])
        sn_ = np.sin(theta_T[i])
        # cross-species forces also scaled by drag (overdamped)
        fx = (f_TT[i, 0] + f_TI_T[i, 0] + f_TM_T[i, 0]) * drag_factor
        fy = (f_TT[i, 1] + f_TI_T[i, 1] + f_TM_T[i, 1]) * drag_factor
        pos_T[i, 0] += dt * (v_eff_i * cs_ + fx) + sqrt_2DT * np.random.normal()
        pos_T[i, 1] += dt * (v_eff_i * sn_ + fy) + sqrt_2DT * np.random.normal()

        # ---- heading update: Vicsek cadherin torque + nematic contact guidance ----
        # "Collagen fibers provide contact guidance, acting like tracks
        #  for invasive migration."
        # sin(2 Δθ) because fibers are nematic (line-symmetric).
        fiber_torque = J_fiber * np.sin(2.0 * (thE - theta_T[i]))
        theta_T[i] += dt * (align_T[i] + fiber_torque) + sqrt_2DR * np.random.normal()
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
        cs_ = np.cos(theta_I[j])
        sn_ = np.sin(theta_I[j])
        gax, gay = grad_field_at(c_a, pos_I[j, 0], pos_I[j, 1], L)
        gsx, gsy = grad_field_at(c_s, pos_I[j, 0], pos_I[j, 1], L)
        gilx, gily = grad_field_at(c_IL10, pos_I[j, 0], pos_I[j, 1], L)
        chemo_x = chi_a * gax - chi_s * (gsx + gilx)
        chemo_y = chi_a * gay - chi_s * (gsy + gily)
        fx = f_II[j, 0] + f_TI_I[j, 0] + f_IM_I[j, 0] + chemo_x
        fy = f_II[j, 1] + f_TI_I[j, 1] + f_IM_I[j, 1] + chemo_y
        pos_I[j, 0] += dt * (v_j * cs_ + fx) + sqrt_2DT * np.random.normal()
        pos_I[j, 1] += dt * (v_j * sn_ + fy) + sqrt_2DT * np.random.normal()
        theta_I[j] += sqrt_2DR * np.random.normal()
        pos_I[j, 0] = pos_I[j, 0] % L
        pos_I[j, 1] = pos_I[j, 1] % L

    # ---- macrophage update (same as combined) ----
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
            cs_ = np.cos(theta_M[k])
            sn_ = np.sin(theta_M[k])
            gax, gay = grad_field_at(c_a, pos_M[k, 0], pos_M[k, 1], L)
            chemo_x = chi_a_M * gax
            chemo_y = chi_a_M * gay
            fx = f_MM[k, 0] + f_TM_M[k, 0] + f_IM_M[k, 0] + chemo_x
            fy = f_MM[k, 1] + f_TM_M[k, 1] + f_IM_M[k, 1] + chemo_y
            pos_M[k, 0] += dt * (v_k * cs_ + fx) + sqrt_2DT * np.random.normal()
            pos_M[k, 1] += dt * (v_k * sn_ + fy) + sqrt_2DT * np.random.normal()
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

    # ---- killing (CD8 → tumor) ----
    n_killed = apply_killing(pos_T, alive_T, pos_I, alive_I, r_kill, p_kill, L)

    # ---- phagocytosis (M1 → tumor) ----
    n_phag = 0
    if use_macrophages:
        n_phag = apply_phagocytosis(
            pos_T, alive_T, pos_M, alive_M, p_M, r_phag, p_phag, L,
        )

    # ---- pressure-gated proliferation, with mild hypoxia slowdown ----
    n_born = 0
    write_cursor = 0
    r2_nbr = nbr_radius * nbr_radius
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        pressure_factor = 1.0 - P_tumor[i] / P_star
        if pressure_factor <= 0.0:
            continue
        # hypoxia slowdown on division
        H_loc = sample_field_at(H_field, pos_T[i, 0], pos_T[i, 1], L)
        hyp_factor = 1.0 - k_div_hyp * H_loc
        if hyp_factor < 0.0:
            hyp_factor = 0.0
        if np.random.random() >= p_div0 * pressure_factor * hyp_factor:
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
        # daughter inherits EMT plus a small drift
        sd = s_T[i] + daughter_EMT_drift * np.random.normal()
        if sd < 0.0:
            sd = 0.0
        elif sd > 1.0:
            sd = 1.0
        s_T[slot] = sd
        n_born += 1
    return n_born, n_killed, n_phag, n_stuck


# ===========================================================================
# Helpers used during the outer per-step Python loop
# ===========================================================================

def _build_emt_secretion_map(
    pos_T, alive_T, s_T, s_m_epi, s_m_mes, k_MMP_hyp,
    H_field, L, G,
):
    """Per-tumor-cell MMP secretion rate, hypoxia-amplified, deposited via CIC.

    s_m_eff_i = s_m_epi + (s_m_mes - s_m_epi) * s_T[i],
    then multiplied by (1 + k_MMP_hyp * H_local).
    """
    N = pos_T.shape[0]
    vals = np.zeros(N, dtype=np.float64)
    H_max = H_field.max() if H_field.size > 0 else 0.0
    if H_max > 0.0:
        for i in range(N):
            if not alive_T[i]:
                continue
            base = s_m_epi + (s_m_mes - s_m_epi) * s_T[i]
            # sample H at cell
            H_loc = _sample_field_py(H_field, pos_T[i, 0], pos_T[i, 1], L)
            vals[i] = base * (1.0 + k_MMP_hyp * H_loc)
    else:
        for i in range(N):
            if not alive_T[i]:
                continue
            vals[i] = s_m_epi + (s_m_mes - s_m_epi) * s_T[i]
    return deposit_tumor_scalar(pos_T, alive_T, vals, L, G)


def _sample_field_py(field_arr: np.ndarray, x: float, y: float, L: float) -> float:
    """Python wrapper around numba sample for use in plain helpers."""
    return float(sample_field_at(field_arr, x, y, L))


def _max_invasion_distance(pos_T_alive: np.ndarray, L: float) -> float:
    """Max distance of any alive tumor cell from box center under PBC."""
    if len(pos_T_alive) == 0:
        return 0.0
    cx = cy = 0.5 * L
    dx = pos_T_alive[:, 0] - cx
    dy = pos_T_alive[:, 1] - cy
    half = 0.5 * L
    dx = np.where(dx > half, dx - L, dx)
    dx = np.where(dx < -half, dx + L, dx)
    dy = np.where(dy > half, dy - L, dy)
    dy = np.where(dy < -half, dy + L, dy)
    return float(np.max(np.hypot(dx, dy)))


@njit(cache=True, fastmath=True)
def _count_detached(pos_T: np.ndarray, alive_T: np.ndarray,
                    radius: float, L: float) -> int:
    """Count alive tumor cells with fewer than 3 same-species neighbors
    inside ``radius`` under PBC."""
    N = pos_T.shape[0]
    n_det = 0
    r2 = radius * radius
    half_L = 0.5 * L
    for i in range(N):
        if not alive_T[i]:
            continue
        cnt = 0
        xi = pos_T[i, 0]; yi = pos_T[i, 1]
        for j in range(N):
            if i == j or not alive_T[j]:
                continue
            dx = xi - pos_T[j, 0]; dy = yi - pos_T[j, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            if dx * dx + dy * dy < r2:
                cnt += 1
                if cnt >= 3:
                    break
        if cnt < 3:
            n_det += 1
    return n_det


# ===========================================================================
# Run driver
# ===========================================================================

def run_biophysical(
    params: Optional[BiophysicalParams] = None,
    seed: int = 0,
    snapshot_every: int = 25,
    save_fields: bool = True,
) -> BiophysicalOut:
    """Run the biophysical extension of ``run_combined``.

    Adds three coarse-grained mechanisms on top of ``sim_combined``:
    per-cell EMT, ECM fiber alignment / integrin biphasic traction,
    and a hypoxia / oxygen field.

    Reduces exactly to ``run_combined`` behaviour when ``J_fiber=0``,
    ``D_EMT=0``, ``s_T0=0``, ``k_hypoxia_EMT=0``, ``k_ECM_EMT=0``,
    ``k_supp_EMT=0``, ``D_O=0`` and ``O_threshold=-1`` (i.e. hypoxia
    field clamped to zero).
    """
    if params is None:
        params = BiophysicalParams()
    if params.sigma_adh <= params.sigma_T:
        raise ValueError(
            f"sigma_adh ({params.sigma_adh}) must be > sigma_T ({params.sigma_T})"
        )

    np.random.seed(seed)
    rng = np.random.default_rng(seed + 311)
    state = init_ecm_state(params, seed=seed)
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    v_T_arr = state["v_T_arr"]; DR_T_arr = state["DR_T_arr"]
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    v_I_arr = state["v_I_arr"]; DR_I_arr = state["DR_I_arr"]
    pos_M = state["pos_M"]; theta_M = state["theta_M"]; alive_M = state["alive_M"]
    p_M = state["p_M"]
    c_a = state["c_a"]; c_s = state["c_s"]; c_IL10 = state["c_IL10"]
    rho_E = state["rho_E"]; m_field = state["m_field"]

    # ---- new state: per-cell EMT, O2, hypoxia, theta_E ----
    N_Tmax = pos_T.shape[0]
    s_T = np.zeros(N_Tmax, dtype=np.float64)
    # init alive cells with s_T0 + N(0, 0.05) clipped to [0,1]
    for i in range(N_Tmax):
        if alive_T[i]:
            v = params.s_T0 + 0.05 * rng.standard_normal()
            s_T[i] = float(np.clip(v, 0.0, 1.0))

    G = params.G
    L = params.L
    # initial oxygen at steady-state (supply / lambda)
    if params.lambda_O > 0.0:
        O_steady = params.supply_O / params.lambda_O
    else:
        O_steady = params.supply_O
    c_O2 = np.full((G, G), float(O_steady), dtype=np.float64)
    theta_E = make_fiber_field(L, G, params.fiber_alignment_strength, seed=seed)

    # ---- VEGF field + vessel state ---------------------------------------
    c_VEGF = np.zeros((G, G), dtype=np.float64)
    nv0 = max(0, int(params.n_vessels_init))
    nvmax = max(nv0, int(params.n_vessels_max))
    vessels = np.zeros((nvmax, 2), dtype=np.float64)
    margin = float(params.vessel_edge_margin)
    if nv0 > 0:
        # seed vessels at the BOTTOM trunk (mirror sim_tme but rotated 90°
        # so the tree grows UPWARD into the tumor mass at the box center).
        x_centers = np.linspace(0.35 * L, 0.65 * L, nv0) if nv0 > 1 else np.array([0.5 * L])
        y_trunk = margin + 1.5
        for k in range(nv0):
            x = x_centers[k] + rng.normal(0.0, 0.4)
            y = y_trunk + rng.normal(0.0, 0.4)
            vessels[k, 0] = float(np.clip(x, 1.0, L - 1.0))
            vessels[k, 1] = float(np.clip(y, 1.0, L - 1.0))
    n_vessels = int(nv0)
    vessel_parent = np.full(nvmax, -1, dtype=np.int32)

    n_steps = params.n_steps
    out = BiophysicalOut(params=params)
    out.theta_E = theta_E.copy()
    cum_born = cum_killed = cum_phag = cum_stuck = 0
    N_T_initial = int(alive_T.sum())
    dx_grid = params.L / params.G
    # snapshot initial ECM mean so the final summary can compute degraded area
    rho_E_init_field_mean = float(rho_E.mean())
    # dedicated Python RNG for vessel noise + sprouting (decoupled from numba seed)
    rng_v = np.random.default_rng(seed + 4242)

    for step in range(n_steps):
        # ---- hypoxia field (sampled inside the kernel as well) ----
        H_field = hypoxia_field(c_O2, params.O_threshold)

        # ---- main JIT inner step ----
        n_born, n_killed, n_phag, n_stuck = _step_biophysical(
            pos_T, theta_T, alive_T, v_T_arr, DR_T_arr, s_T,
            pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
            pos_M, theta_M, alive_M, p_M,
            c_a, c_s, c_IL10,
            rho_E, m_field, c_O2, theta_E, H_field,
            params.dt,
            params.sigma_T, params.k_rep_T,
            params.sigma_I, params.k_rep_I,
            params.sigma_TI, params.k_rep_TI,
            params.sigma_M, params.k_rep_M,
            params.sigma_TM, params.k_rep_TM,
            params.sigma_IM, params.k_rep_IM,
            params.L,
            params.D_T_T, params.D_T_I, params.D_T_M,
            params.chi_a, params.chi_s, params.chi_a_M,
            params.r_kill, params.p_kill,
            params.r_phag, params.p_phag,
            params.p_div, params.P_star,
            params.nbr_radius, params.nbr_threshold,
            params.tau_p, params.D_p,
            params.kappa_s, params.kappa_il, params.M1_bias,
            params.daughter_v_drift, params.daughter_DR_drift,
            params.daughter_EMT_drift,
            params.use_macrophages,
            params.beta_drag, params.r_0, params.r_p_star, params.mmp_open_thresh,
            params.k_adh, params.sigma_adh, params.J_align,
            params.D_EMT, params.k_hypoxia_EMT, params.k_ECM_EMT,
            params.k_supp_EMT, params.k_MET,
            params.v_epi, params.v_mes,
            params.J_fiber, params.A_base,
            params.k_div_hyp,
        )
        cum_born += n_born; cum_killed += n_killed
        cum_phag += n_phag; cum_stuck += n_stuck

        # ---- field updates: c_a, c_s, c_IL10 ----
        rho_T = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        # c_s gets a hypoxia bonus on its (tumor) source rate ("immune
        # suppression amplified in hypoxic zones"). Implement via spatially
        # multiplied effective rho.
        H_now = hypoxia_field(c_O2, params.O_threshold)
        c_a, _ = step_fields(
            c_a, c_s, rho_T,
            params.D_a, params.D_s, params.s_a, params.s_s,
            params.lam_a, params.lam_s, params.dt, params.L,
        )
        # we still want the c_a returned by step_fields; redo c_s with
        # spatially-varying source rho_T * (1 + k_supp_hyp * H)
        n_sub_s = n_substeps_for_cfl(params.D_s, params.dt, dx_grid, 0.25)
        dt_sub_s = params.dt / n_sub_s
        rho_eff_s = rho_T * (1.0 + params.k_supp_hyp * H_now)
        for _ in range(n_sub_s):
            c_s = step_field_one_substep(
                c_s, rho_eff_s,
                params.D_s, params.s_s, params.lam_s,
                dt_sub_s, dx_grid,
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

        # ---- MMP source: EMT- and hypoxia-modulated per-cell secretion ----
        n_sub_m = n_substeps_for_cfl(params.D_m, params.dt, dx_grid, 0.25)
        dt_sub_m = params.dt / n_sub_m
        mmp_src = _build_emt_secretion_map(
            pos_T, alive_T, s_T,
            params.s_m_epi, params.s_m_mes, params.k_MMP_hyp,
            H_now, params.L, params.G,
        )
        # use this map as a pre-multiplied source with effective s=1.0
        for _ in range(n_sub_m):
            m_field = step_field_one_substep(
                m_field, mmp_src,
                params.D_m, 1.0, params.lam_m,
                dt_sub_m, dx_grid,
            )
        if params.rho_E_init > 0.0 or params.k_deg > 0.0:
            update_rho_E_inplace(
                rho_E, m_field,
                params.k_deg, params.k_rep_ECM, params.rho_E_init, params.dt,
            )

        # ---- vessel O2 deposition (vessels pump O2 into c_O2) ----
        if n_vessels > 0 and params.s_O2_vessel > 0.0:
            vessel_src = deposit_vessel_O2_source(
                vessels, n_vessels, params.L, params.G,
            )
            # one explicit additive bump, scaled by per-step rate
            c_O2 = c_O2 + params.dt * params.s_O2_vessel * vessel_src

        # ---- O2 field step ----
        if params.D_O > 0.0:
            n_sub_o = n_substeps_for_cfl(params.D_O, params.dt, dx_grid, 0.25)
            dt_sub_o = params.dt / n_sub_o
            for _ in range(n_sub_o):
                c_O2 = step_O2_one_substep(
                    c_O2, rho_T,
                    params.D_O, params.supply_O,
                    params.consumption_T, params.lambda_O,
                    dt_sub_o, dx_grid,
                )

        # ---- VEGF field: hypoxic-tumor source, diffuse + decay ----
        # H_now is in [0,1]; hypoxic tumor cells (H_local > 0) source VEGF.
        # Implement by weighting tumor density rho_T by local hypoxia level.
        if params.D_VEGF > 0.0 or params.s_VEGF_hyp > 0.0:
            # hypoxic tumor density: rho_T * H (both on the same grid)
            rho_T_hyp = rho_T * H_now
            n_sub_v = n_substeps_for_cfl(params.D_VEGF, params.dt, dx_grid, 0.25)
            dt_sub_v = params.dt / n_sub_v
            for _ in range(n_sub_v):
                c_VEGF = step_field_one_substep(
                    c_VEGF, rho_T_hyp,
                    params.D_VEGF, params.s_VEGF_hyp, params.lam_VEGF,
                    dt_sub_v, dx_grid,
                )

        # ---- vessel dynamics: drift up VEGF + noise; sprout when VEGF high ----
        if n_vessels > 0:
            for k in range(n_vessels):
                gvx, gvy = grad_field_at(
                    c_VEGF, vessels[k, 0], vessels[k, 1], params.L,
                )
                drift_x = params.chi_vessel * gvx * params.dt
                drift_y = params.chi_vessel * gvy * params.dt
                noise_x = np.sqrt(2.0 * params.D_vessel * params.dt) * rng_v.standard_normal()
                noise_y = np.sqrt(2.0 * params.D_vessel * params.dt) * rng_v.standard_normal()
                vessels[k, 0] = (vessels[k, 0] + drift_x + noise_x) % params.L
                vessels[k, 1] = (vessels[k, 1] + drift_y + noise_y) % params.L
            # sprouting
            if params.sprout_rate > 0.0 and n_vessels < params.n_vessels_max:
                cur = n_vessels  # don't iterate over freshly spawned vessels
                for k in range(cur):
                    if n_vessels >= params.n_vessels_max:
                        break
                    vegf_local = float(sample_field_at(
                        c_VEGF, vessels[k, 0], vessels[k, 1], params.L,
                    ))
                    if vegf_local < params.sprout_VEGF_thresh:
                        continue
                    if rng_v.random() < params.sprout_rate:
                        ang = rng_v.uniform(0.0, 2.0 * np.pi)
                        nx = (vessels[k, 0] + 1.5 * np.cos(ang)) % params.L
                        ny = (vessels[k, 1] + 1.5 * np.sin(ang)) % params.L
                        vessels[n_vessels, 0] = nx
                        vessels[n_vessels, 1] = ny
                        vessel_parent[n_vessels] = k
                        n_vessels += 1

        # extinction shortcut
        if step > 100 and not alive_T.any():
            break

        if step % snapshot_every == 0 or step == n_steps - 1:
            pos_T_alive = pos_T[alive_T].copy()
            s_T_alive = s_T[alive_T].copy()
            out.pos_T_snapshots.append(pos_T_alive)
            out.pos_I_snapshots.append(pos_I[alive_I].copy())
            if params.use_macrophages:
                out.pos_M_snapshots.append(pos_M[alive_M].copy())
                out.p_M_snapshots.append(p_M[alive_M].copy())
            else:
                out.pos_M_snapshots.append(np.zeros((0, 2)))
                out.p_M_snapshots.append(np.zeros(0))
            out.s_T_snapshots.append(s_T_alive)
            if save_fields:
                out.c_a_snapshots.append(c_a.copy())
                out.c_s_snapshots.append(c_s.copy())
                out.c_IL10_snapshots.append(c_IL10.copy())
                out.rho_E_snapshots.append(rho_E.copy())
                out.m_snapshots.append(m_field.copy())
                out.O_snapshots.append(c_O2.copy())
                out.H_snapshots.append(H_now.copy())
                out.c_VEGF_snapshots.append(c_VEGF.copy())
            # vessel snapshots (always saved; small lists)
            out.vessel_snapshots.append(vessels[:n_vessels].copy())
            out.vessel_parent_snapshots.append(vessel_parent[:n_vessels].copy())
            out.n_vessels.append(int(n_vessels))
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

            # ---- EMT / invasion / detachment metrics ----
            if len(s_T_alive) > 0:
                out.mean_EMT.append(float(np.mean(s_T_alive)))
                out.frac_mes.append(float(np.mean(s_T_alive > 0.6)))
            else:
                out.mean_EMT.append(0.0)
                out.frac_mes.append(0.0)
            out.max_invasion_distance.append(
                _max_invasion_distance(pos_T_alive, params.L)
            )
            out.n_detached.append(
                int(_count_detached(pos_T, alive_T, 1.5 * params.sigma_T, params.L))
            )

    # ---- final summary scalars ----
    out.final_tumor_size = int(out.n_T[-1]) if out.n_T else 0
    if params.rho_E_init > 0.0:
        out.ecm_degraded_area = float(
            np.mean(rho_E < 0.5 * params.rho_E_init)
        )
    else:
        out.ecm_degraded_area = 0.0
    H_now = hypoxia_field(c_O2, params.O_threshold)
    out.hypoxic_area = float(np.mean(H_now > 0.3))
    frac = (out.n_T[-1] if out.n_T else 0) / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 1e2))
    return out
