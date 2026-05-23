"""TME-extended simulation: hypoxia, angiogenesis, and extra immune cell types.

Layered on top of ``src.sim_ecm.ECMParams`` (which inherits ``ExtendedParams``).

New mechanisms (all can be disabled cleanly by zeroing a single knob):

* ``c_O2(x, t)``: oxygen field — sourced by discrete vessel nodes, consumed by
  every alive cell, diffusion + small decay.  Initial uniform normoxia.
* ``c_VEGF(x, t)``: VEGF field — secreted only by hypoxic tumor cells
  (where local ``c_O2 < O2_hyp_thresh``), small decay.
* Vessel nodes ``vessels[Nv, 2]``: deposit oxygen, drift up the VEGF gradient
  (angiogenesis), optionally sprout new vessels with probability proportional
  to local VEGF.
* Three new cell species (using the existing overdamped Langevin / repulsion
  machinery):
    - **NK** cells: fast innate killers; weak attractant chemotaxis; not
      suppressed by ``c_s``/``c_IL10``; smaller per-step kill probability
      ``p_kill_NK``.
    - **DC** cells: dendritic cells; strong attractant chemotaxis; deposit a
      short-range buff field ``c_DC`` that locally amplifies CD8 kill prob.
    - **MDSC** cells: myeloid-derived suppressor cells; act like macrophages
      but pure suppressors (always M2-style); contribute to ``c_s`` source.

Hypoxia couplings:

* Tumor proliferation gated by oxygen via a smooth sigmoid factor.
* Hypoxic tumor secrete VEGF; normoxic tumor doesn't.
* CD8 / NK kill probability reduced in hypoxic regions.

A single driver function ``run_tme`` returns a ``TMEOut`` dataclass.
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
    init_extended_state,
    pairwise_with_pressure,
    cross_repulsion_with_pressure_a,
    apply_phagocytosis,
    deposit_M2_density,
    _draw_lognormal,
    _draw_gamma,
)
from src.sim_ecm import ECMParams, sample_field_at, update_rho_E_inplace


# ===========================================================================
# Parameter container
# ===========================================================================

@dataclass
class TMEParams(ECMParams):
    """ECMParams + hypoxia/VEGF/vasculature + NK/DC/MDSC populations.

    Defaults are picked so that with ``s_O2_vessel=0`` and the new pool sizes
    all set to 0 the model degenerates to ``run_extended_ecm``.
    """
    # --- oxygen field -----------------------------------------------------
    D_O2: float = 8.0           # fast diffusion
    lam_O2: float = 0.02        # small natural decay
    k_O2_cons: float = 0.35     # consumption per unit (rho_T+rho_I+rho_M)
    O2_init: float = 0.85       # near-normoxia init
    O2_hyp_thresh: float = 0.45 # below this: cell considered hypoxic
    O2_div_thresh: float = 0.5  # midpoint of sigmoid gate on division
    O2_div_scale: float = 0.10  # sigmoid sharpness
    hypoxia_kill_penalty: float = 0.6   # kill prob reduced by this factor when hypoxic

    # --- VEGF field -------------------------------------------------------
    D_VEGF: float = 3.0
    lam_VEGF: float = 0.10
    s_VEGF_hyp: float = 1.0     # source rate from hypoxic tumor cells

    # --- vasculature ------------------------------------------------------
    n_vessels_init: int = 6
    n_vessels_max: int = 32
    s_O2_vessel: float = 3.0    # O2 deposit rate per vessel into its grid cell
    chi_vessel: float = 80.0    # vessel drift speed up VEGF grad
    D_vessel: float = 0.02      # vessel positional noise
    sprout_rate: float = 0.03   # per-step probability of sprouting at vessel
    sprout_VEGF_thresh: float = 0.005
    vessel_edge_margin: float = 14.0   # initial seed margin from box edges

    # --- NK cells (innate, fast, not checkpoint-suppressed) ---------------
    N_NK_max: int = 300
    N_NK_initial: int = 60
    v_NK_mean: float = 1.2
    v_NK_cv: float = 0.25
    D_R_NK_mean: float = 0.8
    D_R_NK_cv: float = 0.30
    D_T_NK: float = 0.001
    sigma_NK: float = 1.0
    k_rep_NK: float = 30.0
    chi_a_NK: float = 8.0       # weak attractant chemotaxis
    r_kill_NK: float = 1.4
    p_kill_NK: float = 0.07

    # --- DC cells (antigen presenting) ------------------------------------
    N_DC_max: int = 150
    N_DC_initial: int = 40
    v_DC_mean: float = 0.6
    v_DC_cv: float = 0.20
    D_R_DC_mean: float = 1.5
    D_R_DC_cv: float = 0.30
    D_T_DC: float = 0.001
    sigma_DC: float = 1.0
    k_rep_DC: float = 30.0
    chi_a_DC: float = 30.0      # strong attractant chemotaxis
    # buff field c_DC: DC-deposited, short range; multiplies CD8 kill prob
    D_DC: float = 1.0           # short-range
    s_DC: float = 1.0
    lam_DC: float = 0.3
    cd8_buff_scale: float = 4.0  # p_kill_eff = p_kill * (1 + scale * c_DC)

    # --- MDSC cells (suppressor myeloids) ---------------------------------
    N_MDSC_max: int = 300
    N_MDSC_initial: int = 80
    v_MDSC_mean: float = 0.2
    v_MDSC_cv: float = 0.25
    D_R_MDSC_mean: float = 3.0
    D_R_MDSC_cv: float = 0.30
    D_T_MDSC: float = 0.001
    sigma_MDSC: float = 1.0
    k_rep_MDSC: float = 30.0
    chi_a_MDSC: float = 6.0
    s_s_MDSC: float = 0.6        # extra source contributed to c_s

    # --- CAFs (cancer-associated fibroblasts; stationary stromal cells) ---
    # Biology refs in docs/calibration_research.md (Kalluri 2016;
    # Mariathasan et al. 2018; Feig et al. 2013): peritumoral CAFs deposit
    # collagen and drive ECM densification, especially at the invasive edge.
    N_CAF_init: int = 30
    sigma_CAF: float = 1.2
    caf_ecm_source: float = 0.06     # per-step rho_E source per CAF (CIC deposit)
    caf_ring_inner: float = 18.0     # annulus inner radius around tumor center
    caf_ring_outer: float = 30.0     # annulus outer radius around tumor center
    # invasive-edge ECM densification (CAF/tumor cross-talk at the margin)
    edge_ecm_boost: float = 0.04     # per-step ECM boost amplitude at high |grad rho_T|


# ===========================================================================
# Vessel deposition helper (Python; called once per step at low N_v)
# ===========================================================================

def deposit_vessel_O2_source(vessels: np.ndarray, n_vessels: int, L: float, G: int) -> np.ndarray:
    """Source map for vessel-deposited oxygen.

    Returns a (G, G) array whose [iy, ix] cell holds the count of vessels in
    that cell; the field-stepper multiplies by ``s_O2_vessel`` and treats it
    as the source rate.
    """
    src = np.zeros((G, G), dtype=np.float64)
    if n_vessels <= 0:
        return src
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    for k in range(n_vessels):
        x = vessels[k, 0]
        y = vessels[k, 1]
        ix = int(x * inv_dx) % G
        iy = int(y * inv_dx) % G
        src[iy, ix] += inv_cell_area
    return src


@njit(cache=True)
def deposit_density_one_species(
    pos: np.ndarray, alive: np.ndarray, L: float, G: int,
) -> np.ndarray:
    """Cloud-in-cell density deposit (same shape as deposit_tumor_density)."""
    rho = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N_max = pos.shape[0]
    for i in range(N_max):
        if not alive[i]:
            continue
        gx = pos[i, 0] * inv_dx
        gy = pos[i, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        rho[iy0, ix0] += w00 * inv_cell_area
        rho[iy0, ix1] += w10 * inv_cell_area
        rho[iy1, ix0] += w01 * inv_cell_area
        rho[iy1, ix1] += w11 * inv_cell_area
    return rho


@njit(cache=True, fastmath=True)
def deposit_hypoxic_tumor_density(
    pos_T: np.ndarray, alive_T: np.ndarray,
    c_O2: np.ndarray, O2_hyp_thresh: float, L: float, G: int,
) -> np.ndarray:
    """Density of only those tumor cells whose local O2 < threshold."""
    rho = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N_max = pos_T.shape[0]
    for i in range(N_max):
        if not alive_T[i]:
            continue
        o2 = sample_field_at(c_O2, pos_T[i, 0], pos_T[i, 1], L)
        if o2 >= O2_hyp_thresh:
            continue
        gx = pos_T[i, 0] * inv_dx
        gy = pos_T[i, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        rho[iy0, ix0] += w00 * inv_cell_area
        rho[iy0, ix1] += w10 * inv_cell_area
        rho[iy1, ix0] += w01 * inv_cell_area
        rho[iy1, ix1] += w11 * inv_cell_area
    return rho


# ===========================================================================
# Specialised kill kernels for the new species
# ===========================================================================

@njit(cache=True)
def apply_killing_with_O2_buff(
    pos_tumor: np.ndarray, alive_tumor: np.ndarray,
    pos_eff: np.ndarray, alive_eff: np.ndarray,
    c_O2: np.ndarray, c_DC: np.ndarray,
    O2_hyp_thresh: float, hypoxia_penalty: float,
    cd8_buff_scale: float,
    r_kill: float, p_kill: float, L: float,
) -> int:
    """CD8 killing with O2 (hypoxia penalty) and c_DC (DC buff) modulating
    the effective per-step kill probability.

    p_kill_eff = p_kill * (1 - hypoxia_penalty * [O2<thr]) * (1 + buff*c_DC)
    """
    half_L = 0.5 * L
    r2_kill = r_kill * r_kill
    n_killed = 0
    N_t = pos_tumor.shape[0]
    N_e = pos_eff.shape[0]
    for j in range(N_e):
        if not alive_eff[j]:
            continue
        xj = pos_eff[j, 0]; yj = pos_eff[j, 1]
        best_i = -1; best_r2 = r2_kill
        for i in range(N_t):
            if not alive_tumor[i]:
                continue
            dx = xj - pos_tumor[i, 0]; dy = yj - pos_tumor[i, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < best_r2:
                best_r2 = r2; best_i = i
        if best_i >= 0:
            o2_local = sample_field_at(c_O2, xj, yj, L)
            hyp_factor = 1.0
            if o2_local < O2_hyp_thresh:
                hyp_factor = 1.0 - hypoxia_penalty
                if hyp_factor < 0.0:
                    hyp_factor = 0.0
            dc_local = sample_field_at(c_DC, xj, yj, L)
            p_eff = p_kill * hyp_factor * (1.0 + cd8_buff_scale * dc_local)
            if p_eff > 1.0:
                p_eff = 1.0
            if np.random.random() < p_eff:
                alive_tumor[best_i] = False
                n_killed += 1
    return n_killed


@njit(cache=True)
def apply_killing_NK(
    pos_tumor: np.ndarray, alive_tumor: np.ndarray,
    pos_nk: np.ndarray, alive_nk: np.ndarray,
    c_O2: np.ndarray, O2_hyp_thresh: float, hypoxia_penalty: float,
    r_kill: float, p_kill: float, L: float,
) -> int:
    half_L = 0.5 * L
    r2_kill = r_kill * r_kill
    n_killed = 0
    N_t = pos_tumor.shape[0]
    N_e = pos_nk.shape[0]
    for j in range(N_e):
        if not alive_nk[j]:
            continue
        xj = pos_nk[j, 0]; yj = pos_nk[j, 1]
        best_i = -1; best_r2 = r2_kill
        for i in range(N_t):
            if not alive_tumor[i]:
                continue
            dx = xj - pos_tumor[i, 0]; dy = yj - pos_tumor[i, 1]
            if dx > half_L: dx -= L
            elif dx < -half_L: dx += L
            if dy > half_L: dy -= L
            elif dy < -half_L: dy += L
            r2 = dx * dx + dy * dy
            if r2 < best_r2:
                best_r2 = r2; best_i = i
        if best_i >= 0:
            o2_local = sample_field_at(c_O2, xj, yj, L)
            hyp_factor = 1.0
            if o2_local < O2_hyp_thresh:
                hyp_factor = 1.0 - hypoxia_penalty
                if hyp_factor < 0.0:
                    hyp_factor = 0.0
            p_eff = p_kill * hyp_factor
            if np.random.random() < p_eff:
                alive_tumor[best_i] = False
                n_killed += 1
    return n_killed


# ===========================================================================
# Generic single-species Langevin update (numba)
# ===========================================================================

@njit(cache=True, fastmath=True)
def _step_chemotactic_species(
    pos, theta, alive,
    v_arr, DR_arr,
    f_self, f_cross_T, f_cross_I, f_cross_M,
    c_a, chi_a,
    D_T_trans, dt, L,
):
    """Generic update for a chemotactic, self-repelling, attractant-chasing
    species (used for NK, DC, MDSC).  ``f_cross_*`` arrays are zeros where
    that interaction is disabled.
    """
    N = pos.shape[0]
    for j in range(N):
        if not alive[j]:
            continue
        v_j = v_arr[j]
        D_R_j = DR_arr[j]
        sqrt_2DT = np.sqrt(2.0 * D_T_trans * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_j * dt)
        cs = np.cos(theta[j]); sn = np.sin(theta[j])
        gax, gay = grad_field_at(c_a, pos[j, 0], pos[j, 1], L)
        fx = f_self[j, 0] + f_cross_T[j, 0] + f_cross_I[j, 0] + f_cross_M[j, 0] + chi_a * gax
        fy = f_self[j, 1] + f_cross_T[j, 1] + f_cross_I[j, 1] + f_cross_M[j, 1] + chi_a * gay
        pos[j, 0] += dt * (v_j * cs + fx) + sqrt_2DT * np.random.normal()
        pos[j, 1] += dt * (v_j * sn + fy) + sqrt_2DT * np.random.normal()
        theta[j] += sqrt_2DR * np.random.normal()
        pos[j, 0] = pos[j, 0] % L
        pos[j, 1] = pos[j, 1] % L


# ===========================================================================
# Cross-repulsion that returns forces on species A only
# ===========================================================================

@njit(cache=True, fastmath=True)
def cross_repulsion_A(
    pos_a: np.ndarray, alive_a: np.ndarray,
    pos_b: np.ndarray, alive_b: np.ndarray,
    sigma: float, k_rep: float, L: float,
) -> np.ndarray:
    """Like ``cross_species_repulsion`` but returns only forces on species A.

    Useful when we only need one half of a Newton-3 pair.
    """
    N_a = pos_a.shape[0]; N_b = pos_b.shape[0]
    fa = np.zeros_like(pos_a)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
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
                fa[i, 0] += fmag_over_r * dx
                fa[i, 1] += fmag_over_r * dy
    return fa


# ===========================================================================
# CAF helpers: stationary stromal ECM source + invasive-edge densification
# ===========================================================================

@njit(cache=True, fastmath=True)
def deposit_caf_ecm(
    pos_CAF: np.ndarray, rho_E: np.ndarray, source_per_step: float, L: float,
):
    """Add ``source_per_step`` of ECM density into the four grid cells
    surrounding each CAF (cloud-in-cell), in-place on ``rho_E``.

    ``source_per_step`` should already include the ``dt`` factor.  Each CAF
    contributes a total mass of ``source_per_step / cell_area`` distributed
    across its four enclosing cells with bilinear weights — same convention
    as ``deposit_tumor_density``.
    """
    G = rho_E.shape[0]
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N = pos_CAF.shape[0]
    for k in range(N):
        gx = pos_CAF[k, 0] * inv_dx
        gy = pos_CAF[k, 1] * inv_dx
        ix = int(gx); iy = int(gy)
        fx = gx - ix; fy = gy - iy
        ix0 = ix % G; iy0 = iy % G
        ix1 = (ix + 1) % G; iy1 = (iy + 1) % G
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        amp = source_per_step * inv_cell_area
        rho_E[iy0, ix0] += w00 * amp
        rho_E[iy0, ix1] += w10 * amp
        rho_E[iy1, ix0] += w01 * amp
        rho_E[iy1, ix1] += w11 * amp


@njit(cache=True, fastmath=True)
def boost_ecm_at_edge(
    rho_T: np.ndarray, rho_E: np.ndarray, boost_per_step: float, dx: float,
):
    """In-place ECM densification at the invasive front.

    Computes |grad rho_T| via central differences with periodic BCs, then
    adds ``boost_per_step * |grad rho_T| / max(|grad rho_T|)`` to ``rho_E``.
    ``boost_per_step`` already includes the ``dt`` factor.

    If the gradient is uniformly zero (no tumor yet), nothing is added.
    """
    G = rho_T.shape[0]
    inv_2dx = 1.0 / (2.0 * dx)
    # first pass: |grad rho_T| and its max
    gmag = np.zeros((G, G), dtype=np.float64)
    gmax = 0.0
    for i in range(G):
        ip1 = (i + 1) % G
        im1 = (i - 1) % G
        for j in range(G):
            jp1 = (j + 1) % G
            jm1 = (j - 1) % G
            gx = (rho_T[i, jp1] - rho_T[i, jm1]) * inv_2dx
            gy = (rho_T[ip1, j] - rho_T[im1, j]) * inv_2dx
            g = np.sqrt(gx * gx + gy * gy)
            gmag[i, j] = g
            if g > gmax:
                gmax = g
    if gmax <= 1e-12:
        return
    inv_gmax = 1.0 / gmax
    for i in range(G):
        for j in range(G):
            rho_E[i, j] += boost_per_step * gmag[i, j] * inv_gmax


# ===========================================================================
# State init
# ===========================================================================

def init_tme_state(params: TMEParams, seed: int = 0) -> dict:
    """Initial state: inherit extended/ECM state + add fields, vessels, NK/DC/MDSC."""
    rng = np.random.default_rng(seed)
    state = init_extended_state(params, seed=seed)
    G = params.G; L = params.L

    # ECM (likely off for this demo but kept for compatibility)
    rho_E = np.full((G, G), float(params.rho_E_init), dtype=np.float64)
    m_field = np.zeros((G, G), dtype=np.float64)
    state["rho_E"] = rho_E
    state["m_field"] = m_field

    # ---- new fields ----
    c_O2 = np.full((G, G), float(params.O2_init), dtype=np.float64)
    c_VEGF = np.zeros((G, G), dtype=np.float64)
    c_DC = np.zeros((G, G), dtype=np.float64)
    state["c_O2"] = c_O2
    state["c_VEGF"] = c_VEGF
    state["c_DC"] = c_DC

    # ---- vessels: seed as one or more sprout origins along the BOTTOM of
    # the box; the renderer also draws a static horizontal "parent vessel"
    # strip at y ≈ 0 to provide the visual baseline.  Sprouts grow upward
    # toward the tumor as VEGF accumulates. ----
    nv0 = max(0, int(params.n_vessels_init))
    nvmax = max(nv0, int(params.n_vessels_max))
    vessels = np.zeros((nvmax, 2), dtype=np.float64)
    margin = float(params.vessel_edge_margin)
    if nv0 > 0:
        x_centers = np.linspace(0.35 * L, 0.65 * L, nv0)
        y_trunk = margin + 1.5
        for k in range(nv0):
            x = x_centers[k] + rng.normal(0.0, 0.4)
            y = y_trunk + rng.normal(0.0, 0.4)
            vessels[k, 0] = np.clip(x, 1.0, L - 1.0)
            vessels[k, 1] = np.clip(y, 1.0, L - 1.0)
    state["vessels"] = vessels
    state["n_vessels"] = int(nv0)
    # parent index for each vessel slot (initial vessels: -1).  Same shape
    # as ``vessels`` so we can index in parallel.
    vessel_parent = np.full(nvmax, -1, dtype=np.int32)
    state["vessel_parent"] = vessel_parent

    # ---- CAFs: stationary stromal cells in an annulus around tumor seed ----
    n_caf = max(0, int(params.N_CAF_init))
    pos_CAF = np.zeros((n_caf, 2), dtype=np.float64)
    if n_caf > 0:
        r_in = float(params.caf_ring_inner)
        r_out = float(params.caf_ring_outer)
        cx = cy = 0.5 * L
        # uniform sampling in the annulus: r ~ sqrt(U(r_in^2, r_out^2)),
        # theta ~ U(0, 2 pi).  Clip to inside the box.
        u = rng.uniform(r_in * r_in, r_out * r_out, size=n_caf)
        r = np.sqrt(u)
        ang = rng.uniform(0.0, 2.0 * np.pi, size=n_caf)
        xs = np.clip(cx + r * np.cos(ang), 0.0, L)
        ys = np.clip(cy + r * np.sin(ang), 0.0, L)
        # wrap into [0, L) just in case
        pos_CAF[:, 0] = xs % L
        pos_CAF[:, 1] = ys % L
    state["pos_CAF"] = pos_CAF

    # ---- NK cells: uniform in box ----
    def _alloc(N_max, N_init, vmean, vcv, dRmean, dRcv):
        pos = np.zeros((N_max, 2), dtype=np.float64)
        theta = np.zeros(N_max, dtype=np.float64)
        alive = np.zeros(N_max, dtype=np.bool_)
        varr = np.zeros(N_max, dtype=np.float64)
        dRarr = np.zeros(N_max, dtype=np.float64)
        n = min(N_init, N_max)
        if n > 0:
            pos[:n, 0] = rng.uniform(0, L, size=n)
            pos[:n, 1] = rng.uniform(0, L, size=n)
            theta[:n] = rng.uniform(0, 2 * np.pi, size=n)
            alive[:n] = True
            varr[:n] = _draw_lognormal(vmean, vcv, n, rng)
            dRarr[:n] = _draw_gamma(dRmean, dRcv, n, rng)
        return pos, theta, alive, varr, dRarr

    (state["pos_NK"], state["theta_NK"], state["alive_NK"],
     state["v_NK_arr"], state["DR_NK_arr"]) = _alloc(
        params.N_NK_max, params.N_NK_initial,
        params.v_NK_mean, params.v_NK_cv,
        params.D_R_NK_mean, params.D_R_NK_cv,
    )
    (state["pos_DC"], state["theta_DC"], state["alive_DC"],
     state["v_DC_arr"], state["DR_DC_arr"]) = _alloc(
        params.N_DC_max, params.N_DC_initial,
        params.v_DC_mean, params.v_DC_cv,
        params.D_R_DC_mean, params.D_R_DC_cv,
    )
    (state["pos_MDSC"], state["theta_MDSC"], state["alive_MDSC"],
     state["v_MDSC_arr"], state["DR_MDSC_arr"]) = _alloc(
        params.N_MDSC_max, params.N_MDSC_initial,
        params.v_MDSC_mean, params.v_MDSC_cv,
        params.D_R_MDSC_mean, params.D_R_MDSC_cv,
    )
    return state


# ===========================================================================
# Tumor inner-step variant: O2-gated division + hypoxia-aware CD8 killing
# ===========================================================================

@njit(cache=True, fastmath=True)
def _step_tumor_immune_core(
    pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
    pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
    pos_M, theta_M, alive_M, p_M,
    c_a, c_s, c_IL10,
    c_O2, c_DC,
    rho_E, m_field,
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
    beta_drag, r_0, r_p_star, mmp_open_thresh,
    # hypoxia params
    O2_hyp_thresh, hypoxia_kill_penalty,
    O2_div_thresh, O2_div_scale,
    cd8_buff_scale,
):
    """Core tumor+T+M update with O2-gated division and hypoxia-modulated CD8 killing.
    Returns (n_born, n_killed_cd8, n_phag).
    """
    # ---- pairwise forces ----
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

    # ---- tumor update with ECM drag/gate (same as sim_ecm) ----
    N_Tmax = pos_T.shape[0]
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        v_i = v_T_arr[i]; D_R_i = DR_T_arr[i]
        rE_local = sample_field_at(rho_E, pos_T[i, 0], pos_T[i, 1], L)
        if rE_local < 0.0:
            rE_local = 0.0
        m_local = sample_field_at(m_field, pos_T[i, 0], pos_T[i, 1], L)
        drag_factor = 1.0 / (1.0 + beta_drag * rE_local)
        v_eff = v_i * drag_factor
        if rE_local > 1e-9:
            r_pore = r_0 / np.sqrt(rE_local)
            if r_pore < r_p_star and m_local < mmp_open_thresh:
                v_eff = 0.0
        sqrt_2DT = np.sqrt(2.0 * D_T_T * dt)
        sqrt_2DR = np.sqrt(2.0 * D_R_i * dt)
        cs = np.cos(theta_T[i]); sn = np.sin(theta_T[i])
        fx = (f_TT[i, 0] + f_TI_T[i, 0] + f_TM_T[i, 0]) * drag_factor
        fy = (f_TT[i, 1] + f_TI_T[i, 1] + f_TM_T[i, 1]) * drag_factor
        pos_T[i, 0] += dt * (v_eff * cs + fx) + sqrt_2DT * np.random.normal()
        pos_T[i, 1] += dt * (v_eff * sn + fy) + sqrt_2DT * np.random.normal()
        theta_T[i] += sqrt_2DR * np.random.normal()
        pos_T[i, 0] = pos_T[i, 0] % L
        pos_T[i, 1] = pos_T[i, 1] % L

    # ---- T cell update ----
    N_Imax = pos_I.shape[0]
    for j in range(N_Imax):
        if not alive_I[j]:
            continue
        v_j = v_I_arr[j]; D_R_j = DR_I_arr[j]
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
            v_k = 0.2; D_R_k = 3.0
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

    # ---- CD8 killing with O2/DC modulation ----
    n_killed = apply_killing_with_O2_buff(
        pos_T, alive_T, pos_I, alive_I,
        c_O2, c_DC,
        O2_hyp_thresh, hypoxia_kill_penalty,
        cd8_buff_scale,
        r_kill, p_kill, L,
    )

    # ---- phagocytosis (M1 → tumor) ----
    n_phag = 0
    if use_macrophages:
        n_phag = apply_phagocytosis(
            pos_T, alive_T, pos_M, alive_M, p_M, r_phag, p_phag, L,
        )

    # ---- O2-gated proliferation ----
    n_born = 0
    write_cursor = 0
    r2_nbr = nbr_radius * nbr_radius
    for i in range(N_Tmax):
        if not alive_T[i]:
            continue
        pressure_factor = 1.0 - P_tumor[i] / P_star
        if pressure_factor <= 0.0:
            continue
        o2_local = sample_field_at(c_O2, pos_T[i, 0], pos_T[i, 1], L)
        # sigmoid gate on O2
        z = (o2_local - O2_div_thresh) / O2_div_scale
        o2_factor = 1.0 / (1.0 + np.exp(-z))
        if np.random.random() >= p_div0 * pressure_factor * o2_factor:
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
# TME dataclass + driver
# ===========================================================================

@dataclass
class TMEOut:
    pos_T_snapshots: list = field(default_factory=list)
    pos_I_snapshots: list = field(default_factory=list)
    pos_M_snapshots: list = field(default_factory=list)
    p_M_snapshots: list = field(default_factory=list)
    pos_NK_snapshots: list = field(default_factory=list)
    pos_DC_snapshots: list = field(default_factory=list)
    pos_MDSC_snapshots: list = field(default_factory=list)
    vessel_snapshots: list = field(default_factory=list)
    vessel_parent_snapshots: list = field(default_factory=list)
    pos_CAF: Optional[np.ndarray] = None
    c_a_snapshots: list = field(default_factory=list)
    c_s_snapshots: list = field(default_factory=list)
    c_IL10_snapshots: list = field(default_factory=list)
    c_O2_snapshots: list = field(default_factory=list)
    c_VEGF_snapshots: list = field(default_factory=list)
    c_DC_snapshots: list = field(default_factory=list)
    n_T: list = field(default_factory=list)
    n_I: list = field(default_factory=list)
    n_M: list = field(default_factory=list)
    n_NK: list = field(default_factory=list)
    n_DC: list = field(default_factory=list)
    n_MDSC: list = field(default_factory=list)
    n_vessels: list = field(default_factory=list)
    mean_pM: list = field(default_factory=list)
    hypoxic_fraction: list = field(default_factory=list)
    n_killed_cum: list = field(default_factory=list)
    n_killed_NK_cum: list = field(default_factory=list)
    n_phag_cum: list = field(default_factory=list)
    n_born_cum: list = field(default_factory=list)
    times: list = field(default_factory=list)
    params: Optional[TMEParams] = None
    final_tumor_fraction: float = 0.0


def run_tme(
    params: Optional[TMEParams] = None,
    seed: int = 0,
    snapshot_every: int = 25,
    save_fields: bool = True,
) -> TMEOut:
    """Run the TME-extended sim with hypoxia, angiogenesis, and 6 cell species."""
    if params is None:
        params = TMEParams()
    np.random.seed(seed)
    state = init_tme_state(params, seed=seed)

    # tumor
    pos_T = state["pos_T"]; theta_T = state["theta_T"]; alive_T = state["alive_T"]
    v_T_arr = state["v_T_arr"]; DR_T_arr = state["DR_T_arr"]
    # CD8 T
    pos_I = state["pos_I"]; theta_I = state["theta_I"]; alive_I = state["alive_I"]
    v_I_arr = state["v_I_arr"]; DR_I_arr = state["DR_I_arr"]
    # macrophage
    pos_M = state["pos_M"]; theta_M = state["theta_M"]; alive_M = state["alive_M"]
    p_M = state["p_M"]
    # NK / DC / MDSC
    pos_NK = state["pos_NK"]; theta_NK = state["theta_NK"]; alive_NK = state["alive_NK"]
    v_NK_arr = state["v_NK_arr"]; DR_NK_arr = state["DR_NK_arr"]
    pos_DC = state["pos_DC"]; theta_DC = state["theta_DC"]; alive_DC = state["alive_DC"]
    v_DC_arr = state["v_DC_arr"]; DR_DC_arr = state["DR_DC_arr"]
    pos_MDSC = state["pos_MDSC"]; theta_MDSC = state["theta_MDSC"]; alive_MDSC = state["alive_MDSC"]
    v_MDSC_arr = state["v_MDSC_arr"]; DR_MDSC_arr = state["DR_MDSC_arr"]
    # fields
    c_a = state["c_a"]; c_s = state["c_s"]; c_IL10 = state["c_IL10"]
    rho_E = state["rho_E"]; m_field = state["m_field"]
    c_O2 = state["c_O2"]; c_VEGF = state["c_VEGF"]; c_DC = state["c_DC"]
    # vessels
    vessels = state["vessels"]
    n_vessels = state["n_vessels"]
    vessel_parent = state["vessel_parent"]
    # CAFs (stationary; one shared array, no per-frame snapshot)
    pos_CAF = state["pos_CAF"]

    n_steps = params.n_steps
    out = TMEOut(params=params)
    out.pos_CAF = pos_CAF.copy()
    cum_born = cum_killed = cum_killed_NK = cum_phag = 0
    N_T_initial = int(alive_T.sum())
    dx_grid = params.L / params.G
    G = params.G

    rng_py = np.random.default_rng(seed + 4242)

    for step in range(n_steps):
        # ----- core tumor/CD8/macrophage step (with hypoxia couplings) -----
        n_born, n_killed, n_phag = _step_tumor_immune_core(
            pos_T, theta_T, alive_T, v_T_arr, DR_T_arr,
            pos_I, theta_I, alive_I, v_I_arr, DR_I_arr,
            pos_M, theta_M, alive_M, p_M,
            c_a, c_s, c_IL10,
            c_O2, c_DC,
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
            params.chi_a, params.chi_s, params.chi_a_M,
            params.r_kill, params.p_kill,
            params.r_phag, params.p_phag,
            params.p_div, params.P_star,
            params.nbr_radius, params.nbr_threshold,
            params.tau_p, params.D_p,
            params.kappa_s, params.kappa_il, params.M1_bias,
            params.daughter_v_drift, params.daughter_DR_drift,
            params.use_macrophages,
            params.beta_drag, params.r_0, params.r_p_star, params.mmp_open_thresh,
            params.O2_hyp_thresh, params.hypoxia_kill_penalty,
            params.O2_div_thresh, params.O2_div_scale,
            params.cd8_buff_scale,
        )
        cum_born += n_born; cum_killed += n_killed; cum_phag += n_phag

        # ----- NK / DC / MDSC updates: build sparse cross-repulsions ------
        # We do tumor-X cross repulsion only (X-X self) — keeps signature manageable.
        # Self-repulsion within each species:
        f_NK_self = pairwise_with_pressure(pos_NK, alive_NK, params.sigma_NK, params.k_rep_NK, params.L)[0]
        f_DC_self = pairwise_with_pressure(pos_DC, alive_DC, params.sigma_DC, params.k_rep_DC, params.L)[0]
        f_MDSC_self = pairwise_with_pressure(pos_MDSC, alive_MDSC, params.sigma_MDSC, params.k_rep_MDSC, params.L)[0]
        # Tumor repels NK/DC/MDSC (forces on the immune cell only):
        f_TNK = cross_repulsion_A(pos_NK, alive_NK, pos_T, alive_T, params.sigma_TI, params.k_rep_TI, params.L)
        f_TDC = cross_repulsion_A(pos_DC, alive_DC, pos_T, alive_T, params.sigma_TI, params.k_rep_TI, params.L)
        f_TMDSC = cross_repulsion_A(pos_MDSC, alive_MDSC, pos_T, alive_T, params.sigma_TI, params.k_rep_TI, params.L)
        zero_NK = np.zeros_like(pos_NK)
        zero_DC = np.zeros_like(pos_DC)
        zero_MDSC = np.zeros_like(pos_MDSC)
        _step_chemotactic_species(
            pos_NK, theta_NK, alive_NK, v_NK_arr, DR_NK_arr,
            f_NK_self, f_TNK, zero_NK, zero_NK,
            c_a, params.chi_a_NK,
            params.D_T_NK, params.dt, params.L,
        )
        _step_chemotactic_species(
            pos_DC, theta_DC, alive_DC, v_DC_arr, DR_DC_arr,
            f_DC_self, f_TDC, zero_DC, zero_DC,
            c_a, params.chi_a_DC,
            params.D_T_DC, params.dt, params.L,
        )
        _step_chemotactic_species(
            pos_MDSC, theta_MDSC, alive_MDSC, v_MDSC_arr, DR_MDSC_arr,
            f_MDSC_self, f_TMDSC, zero_MDSC, zero_MDSC,
            c_a, params.chi_a_MDSC,
            params.D_T_MDSC, params.dt, params.L,
        )

        # ----- NK killing (hypoxia-modulated, smaller p_kill) -----
        n_killed_nk = apply_killing_NK(
            pos_T, alive_T, pos_NK, alive_NK,
            c_O2, params.O2_hyp_thresh, params.hypoxia_kill_penalty,
            params.r_kill_NK, params.p_kill_NK, params.L,
        )
        cum_killed_NK += n_killed_nk

        # ----- field updates -----
        rho_T = deposit_tumor_density(pos_T, alive_T, params.L, params.G)
        rho_I = deposit_density_one_species(pos_I, alive_I, params.L, params.G)
        rho_M = deposit_density_one_species(pos_M, alive_M, params.L, params.G) if params.use_macrophages else np.zeros_like(rho_T)
        rho_MDSC = deposit_density_one_species(pos_MDSC, alive_MDSC, params.L, params.G)
        rho_DC = deposit_density_one_species(pos_DC, alive_DC, params.L, params.G)
        rho_hyp = deposit_hypoxic_tumor_density(
            pos_T, alive_T, c_O2, params.O2_hyp_thresh, params.L, params.G,
        )

        # c_a — tumor-sourced; c_s — tumor + MDSC sourced
        n_sub_a = n_substeps_for_cfl(params.D_a, params.dt, dx_grid, 0.25)
        dt_sub_a = params.dt / n_sub_a
        for _ in range(n_sub_a):
            c_a = step_field_one_substep(
                c_a, rho_T,
                params.D_a, params.s_a, params.lam_a,
                dt_sub_a, dx_grid,
            )
        rho_eff_for_s = rho_T + (params.s_s_MDSC / max(params.s_s, 1e-9)) * rho_MDSC
        n_sub_s = n_substeps_for_cfl(params.D_s, params.dt, dx_grid, 0.25)
        dt_sub_s = params.dt / n_sub_s
        for _ in range(n_sub_s):
            c_s = step_field_one_substep(
                c_s, rho_eff_for_s,
                params.D_s, params.s_s, params.lam_s,
                dt_sub_s, dx_grid,
            )

        # IL-10 from M2 macrophages
        rho_M2 = deposit_M2_density(pos_M, alive_M, p_M, params.L, params.G)
        n_sub_il = n_substeps_for_cfl(params.D_IL10, params.dt, dx_grid, 0.25)
        dt_sub_il = params.dt / n_sub_il
        for _ in range(n_sub_il):
            c_IL10 = step_field_one_substep(
                c_IL10, rho_M2,
                params.D_IL10, params.s_IL10, params.lam_IL10,
                dt_sub_il, dx_grid,
            )

        # c_DC: DC-deposited buff field
        n_sub_dc = n_substeps_for_cfl(params.D_DC, params.dt, dx_grid, 0.25)
        dt_sub_dc = params.dt / n_sub_dc
        for _ in range(n_sub_dc):
            c_DC = step_field_one_substep(
                c_DC, rho_DC,
                params.D_DC, params.s_DC, params.lam_DC,
                dt_sub_dc, dx_grid,
            )

        # O2: lumped consumption term -k_cons * (rho_T + rho_I + rho_M) * c_O2.
        # We treat the linear-in-c_O2 sink as an effective decay = lam_O2 + k_cons * total_rho.
        # That's a per-cell sink; bake it into the FTCS substep by using a spatially varying lam.
        # Quick approach: compute a local effective decay array, then use a custom step.
        total_rho = rho_T + rho_I + rho_M + (rho_MDSC if params.use_macrophages else rho_MDSC)
        # add NK and DC too — they consume oxygen too
        rho_NK = deposit_density_one_species(pos_NK, alive_NK, params.L, params.G)
        total_rho = total_rho + rho_NK + rho_DC
        # vessel source
        vessel_src_grid = deposit_vessel_O2_source(vessels, n_vessels, params.L, params.G)
        # Effective: dc/dt = D lap c + s_O2_vessel * src - (lam_O2 + k_cons*total_rho)*c
        n_sub_o2 = n_substeps_for_cfl(params.D_O2, params.dt, dx_grid, 0.25)
        dt_sub_o2 = params.dt / n_sub_o2
        # Build a "sink array" and do explicit Euler manually
        sink_per_step = params.lam_O2 + params.k_O2_cons * total_rho  # (G,G)
        inv_dx2 = 1.0 / (dx_grid * dx_grid)
        for _ in range(n_sub_o2):
            # step c_O2: D lap + vessel source - sink*c_O2 elementwise
            lap = _laplacian_periodic(c_O2) * inv_dx2
            c_O2 = c_O2 + dt_sub_o2 * (
                params.D_O2 * lap
                + params.s_O2_vessel * vessel_src_grid
                - sink_per_step * c_O2
            )
            np.clip(c_O2, 0.0, None, out=c_O2)

        # VEGF: hypoxic tumor source
        n_sub_v = n_substeps_for_cfl(params.D_VEGF, params.dt, dx_grid, 0.25)
        dt_sub_v = params.dt / n_sub_v
        for _ in range(n_sub_v):
            c_VEGF = step_field_one_substep(
                c_VEGF, rho_hyp,
                params.D_VEGF, params.s_VEGF_hyp, params.lam_VEGF,
                dt_sub_v, dx_grid,
            )

        # ECM update (degenerate when rho_E_init=0)
        if params.rho_E_init > 0.0 or params.k_deg > 0.0:
            update_rho_E_inplace(
                rho_E, m_field,
                params.k_deg, params.k_rep_ECM, params.rho_E_init, params.dt,
            )
            n_sub_m = n_substeps_for_cfl(params.D_m, params.dt, dx_grid, 0.25)
            dt_sub_m = params.dt / n_sub_m
            for _ in range(n_sub_m):
                m_field = step_field_one_substep(
                    m_field, rho_T,
                    params.D_m, params.s_m, params.lam_m,
                    dt_sub_m, dx_grid,
                )

        # ----- CAF-driven ECM source (stationary stromal deposition) -----
        # CAFs continuously add collagen to their immediate neighborhood,
        # keeping rho_E elevated near the tumor margin even when MMP would
        # otherwise clear it (Kalluri 2016; Mariathasan et al. 2018).
        if pos_CAF.shape[0] > 0 and params.caf_ecm_source > 0.0:
            deposit_caf_ecm(
                pos_CAF, rho_E,
                params.caf_ecm_source * params.dt, params.L,
            )

        # ----- Invasive-edge ECM densification (CAF/tumor cross-talk) -----
        # Boost rho_E where the tumor density gradient is steep (the
        # invasive front), modelling co-recruitment of collagen at the edge.
        if params.edge_ecm_boost > 0.0:
            boost_ecm_at_edge(
                rho_T, rho_E,
                params.edge_ecm_boost * params.dt, dx_grid,
            )

        # ----- vessel dynamics: drift up VEGF gradient + small noise -----
        if n_vessels > 0:
            for k in range(n_vessels):
                gvx, gvy = grad_field_at(c_VEGF, vessels[k, 0], vessels[k, 1], params.L)
                drift_x = params.chi_vessel * gvx * params.dt
                drift_y = params.chi_vessel * gvy * params.dt
                noise_x = np.sqrt(2.0 * params.D_vessel * params.dt) * rng_py.standard_normal()
                noise_y = np.sqrt(2.0 * params.D_vessel * params.dt) * rng_py.standard_normal()
                vessels[k, 0] = (vessels[k, 0] + drift_x + noise_x) % params.L
                vessels[k, 1] = (vessels[k, 1] + drift_y + noise_y) % params.L
            # sprouting (cheap): with small probability per existing vessel per step,
            # spawn a new vessel near the parent if local VEGF exceeds threshold.
            if params.sprout_rate > 0.0 and n_vessels < params.n_vessels_max:
                for k in range(n_vessels):
                    if n_vessels >= params.n_vessels_max:
                        break
                    vegf_local = sample_field_at(c_VEGF, vessels[k, 0], vessels[k, 1], params.L)
                    if vegf_local < params.sprout_VEGF_thresh:
                        continue
                    if rng_py.random() < params.sprout_rate:
                        # offset by ~one cell diameter
                        ang = rng_py.uniform(0, 2 * np.pi)
                        nx = (vessels[k, 0] + 1.5 * np.cos(ang)) % params.L
                        ny = (vessels[k, 1] + 1.5 * np.sin(ang)) % params.L
                        vessels[n_vessels, 0] = nx
                        vessels[n_vessels, 1] = ny
                        # record the parent vessel index for later viz
                        vessel_parent[n_vessels] = k
                        n_vessels += 1

        # extinction shortcut
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
            out.pos_NK_snapshots.append(pos_NK[alive_NK].copy())
            out.pos_DC_snapshots.append(pos_DC[alive_DC].copy())
            out.pos_MDSC_snapshots.append(pos_MDSC[alive_MDSC].copy())
            out.vessel_snapshots.append(vessels[:n_vessels].copy())
            out.vessel_parent_snapshots.append(vessel_parent[:n_vessels].copy())
            if save_fields:
                out.c_a_snapshots.append(c_a.copy())
                out.c_s_snapshots.append(c_s.copy())
                out.c_IL10_snapshots.append(c_IL10.copy())
                out.c_O2_snapshots.append(c_O2.copy())
                out.c_VEGF_snapshots.append(c_VEGF.copy())
                out.c_DC_snapshots.append(c_DC.copy())
            out.n_T.append(int(alive_T.sum()))
            out.n_I.append(int(alive_I.sum()))
            out.n_M.append(int(alive_M.sum()) if params.use_macrophages else 0)
            out.n_NK.append(int(alive_NK.sum()))
            out.n_DC.append(int(alive_DC.sum()))
            out.n_MDSC.append(int(alive_MDSC.sum()))
            out.n_vessels.append(int(n_vessels))
            if params.use_macrophages and alive_M.any():
                out.mean_pM.append(float(p_M[alive_M].mean()))
            else:
                out.mean_pM.append(0.0)
            # hypoxic fraction: fraction of alive tumor cells below O2_hyp_thresh
            n_alive = int(alive_T.sum())
            if n_alive > 0:
                hyp_count = 0
                for i in range(pos_T.shape[0]):
                    if not alive_T[i]:
                        continue
                    o2i = sample_field_at(c_O2, pos_T[i, 0], pos_T[i, 1], params.L)
                    if o2i < params.O2_hyp_thresh:
                        hyp_count += 1
                out.hypoxic_fraction.append(hyp_count / n_alive)
            else:
                out.hypoxic_fraction.append(0.0)
            out.n_killed_cum.append(cum_killed)
            out.n_killed_NK_cum.append(cum_killed_NK)
            out.n_phag_cum.append(cum_phag)
            out.n_born_cum.append(cum_born)
            out.times.append(step * params.dt)

    frac = (out.n_T[-1] if out.n_T else 0) / max(1, N_T_initial)
    out.final_tumor_fraction = float(np.clip(frac, 1e-2, 1e2))
    return out


# ===========================================================================
# Periodic Laplacian (numpy, vectorised)
# ===========================================================================

def _laplacian_periodic(c: np.ndarray) -> np.ndarray:
    """Vectorised 5-point Laplacian with PBC.  dx is folded into the caller."""
    G = c.shape[0]
    # Note: caller multiplies by D and dt; we do not divide by dx^2 here so that
    # the explicit-Euler step matches the conventions in step_field_one_substep.
    # Actually step_field_one_substep DOES divide by dx^2.  To keep the same
    # CFL convention, we divide by dx^2 inside the step; but dx is fixed across
    # caller invocations so we precompute once in the calling site.
    return (
        np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0)
        + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1)
        - 4.0 * c
    )
