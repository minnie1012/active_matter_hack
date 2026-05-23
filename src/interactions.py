"""Pairwise interactions, chemotaxis forces, and killing rule.

All functions are `@njit` so they can be called from inside the hot loop in
`sim.py`. Numba can inline across modules when both are decorated, so this
split is purely organizational.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def pairwise_harmonic_force(
    pos: np.ndarray,
    alive: np.ndarray,
    sigma: float,
    k_rep: float,
    L: float,
) -> np.ndarray:
    """O(N^2) pairwise harmonic repulsion with minimum-image PBC.

    Force on i from j: F_ij = k_rep * max(0, sigma - r) * r_hat_ij
    Newton 3rd law: equal & opposite applied to j.

    Returns
    -------
    forces : (N_max, 2) float64 array, zero for dead slots.
    """
    N_max = pos.shape[0]
    forces = np.zeros((N_max, 2), dtype=np.float64)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
    for i in range(N_max):
        if not alive[i]:
            continue
        xi = pos[i, 0]
        yi = pos[i, 1]
        for j in range(i + 1, N_max):
            if not alive[j]:
                continue
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            if dx > half_L:
                dx -= L
            elif dx < -half_L:
                dx += L
            if dy > half_L:
                dy -= L
            elif dy < -half_L:
                dy += L
            r2 = dx * dx + dy * dy
            if r2 < sigma2 and r2 > 1e-12:
                r = np.sqrt(r2)
                # F = k * (sigma - r) along r_hat, decomposed via dx/r, dy/r
                fmag_over_r = k_rep * (sigma - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
    return forces


@njit(cache=True)
def count_neighbors_at(
    pos: np.ndarray,
    alive: np.ndarray,
    i: int,
    r2_max: float,
    L: float,
) -> int:
    """Count alive neighbors of particle `i` within sqrt(r2_max) under PBC."""
    half_L = 0.5 * L
    N_max = pos.shape[0]
    cnt = 0
    xi = pos[i, 0]
    yi = pos[i, 1]
    for j in range(N_max):
        if j == i or not alive[j]:
            continue
        dx = xi - pos[j, 0]
        dy = yi - pos[j, 1]
        if dx > half_L:
            dx -= L
        elif dx < -half_L:
            dx += L
        if dy > half_L:
            dy -= L
        elif dy < -half_L:
            dy += L
        if dx * dx + dy * dy < r2_max:
            cnt += 1
    return cnt


@njit(cache=True)
def first_dead_slot(alive: np.ndarray, start: int) -> int:
    """Return the first index >= start where alive is False, or -1."""
    for k in range(start, alive.shape[0]):
        if not alive[k]:
            return k
    return -1


@njit(cache=True, fastmath=True)
def cross_species_repulsion(
    pos_a: np.ndarray,
    alive_a: np.ndarray,
    pos_b: np.ndarray,
    alive_b: np.ndarray,
    sigma: float,
    k_rep: float,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic repulsion between two distinct particle populations under PBC.

    Returns (forces_a, forces_b), each shaped like its species' pos array.
    """
    N_a = pos_a.shape[0]
    N_b = pos_b.shape[0]
    fa = np.zeros_like(pos_a)
    fb = np.zeros_like(pos_b)
    half_L = 0.5 * L
    sigma2 = sigma * sigma
    for i in range(N_a):
        if not alive_a[i]:
            continue
        xi = pos_a[i, 0]
        yi = pos_a[i, 1]
        for j in range(N_b):
            if not alive_b[j]:
                continue
            dx = xi - pos_b[j, 0]
            dy = yi - pos_b[j, 1]
            if dx > half_L:
                dx -= L
            elif dx < -half_L:
                dx += L
            if dy > half_L:
                dy -= L
            elif dy < -half_L:
                dy += L
            r2 = dx * dx + dy * dy
            if r2 < sigma2 and r2 > 1e-12:
                r = np.sqrt(r2)
                fmag_over_r = k_rep * (sigma - r) / r
                fx = fmag_over_r * dx
                fy = fmag_over_r * dy
                fa[i, 0] += fx
                fa[i, 1] += fy
                fb[j, 0] -= fx
                fb[j, 1] -= fy
    return fa, fb


@njit(cache=True)
def apply_killing(
    pos_tumor: np.ndarray,
    alive_tumor: np.ndarray,
    pos_tcell: np.ndarray,
    alive_tcell: np.ndarray,
    r_kill: float,
    p_kill: float,
    L: float,
) -> int:
    """For each alive T cell, attempt to kill at most one tumor neighbor.

    Per spec engineering note: iterate over T cells (not tumor cells) so the
    rate scales with N_Tcell, not N_tumor. Per our decisions log: at most one
    kill per T cell per step.

    Returns the number of tumor cells killed this step.
    """
    half_L = 0.5 * L
    r2_kill = r_kill * r_kill
    n_killed = 0
    N_t = pos_tumor.shape[0]
    N_i = pos_tcell.shape[0]
    for j in range(N_i):
        if not alive_tcell[j]:
            continue
        xj = pos_tcell[j, 0]
        yj = pos_tcell[j, 1]
        # find the closest alive tumor cell within r_kill
        best_i = -1
        best_r2 = r2_kill
        for i in range(N_t):
            if not alive_tumor[i]:
                continue
            dx = xj - pos_tumor[i, 0]
            dy = yj - pos_tumor[i, 1]
            if dx > half_L:
                dx -= L
            elif dx < -half_L:
                dx += L
            if dy > half_L:
                dy -= L
            elif dy < -half_L:
                dy += L
            r2 = dx * dx + dy * dy
            if r2 < best_r2:
                best_r2 = r2
                best_i = i
        if best_i >= 0 and np.random.random() < p_kill:
            alive_tumor[best_i] = False
            n_killed += 1
    return n_killed
