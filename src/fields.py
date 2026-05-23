"""Reaction-diffusion scalar fields on a periodic grid.

Two scalar fields live on a (G, G) grid covering the (L, L) physical box:
- c_a: long-range attractant secreted by tumor cells
- c_s: short-range immunosuppressant secreted by tumor cells

Both obey  dc/dt = D * laplacian(c) + s * rho(x,y) - lambda * c
where rho is the local tumor density (cells per unit area) computed by
bilinear ("cloud-in-cell") deposition of tumor particle positions.

Stepping: explicit FTCS Euler with automatic subcycling to keep the
diffusion CFL ratio  D*dt/dx^2 <= 0.25 .

All functions are `@njit(cache=True)` and operate on contiguous float64
arrays so the inner loop in sim.py can call them without leaving JIT.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def deposit_tumor_density(
    pos: np.ndarray,
    alive: np.ndarray,
    L: float,
    G: int,
) -> np.ndarray:
    """Bilinear (CIC) deposit of alive tumor positions onto a (G, G) grid.

    The returned grid holds *density* (cells per unit area), so that the
    source term `s * rho` has the right units regardless of grid resolution.

    Periodic wrap on the four corners of each cell.
    """
    rho = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    inv_dx = 1.0 / dx
    inv_cell_area = inv_dx * inv_dx
    N_max = pos.shape[0]
    for i in range(N_max):
        if not alive[i]:
            continue
        # grid-coordinates of particle, in [0, G)
        gx = pos[i, 0] * inv_dx
        gy = pos[i, 1] * inv_dx
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
        rho[iy0, ix0] += w00 * inv_cell_area
        rho[iy0, ix1] += w10 * inv_cell_area
        rho[iy1, ix0] += w01 * inv_cell_area
        rho[iy1, ix1] += w11 * inv_cell_area
    return rho


@njit(cache=True, fastmath=True)
def step_field_one_substep(
    c: np.ndarray,
    rho: np.ndarray,
    D: float,
    s: float,
    lam: float,
    dt_sub: float,
    dx: float,
) -> np.ndarray:
    """One explicit FTCS Euler step for the periodic RD field.

    c_new[i,j] = c[i,j] + dt*(D*lap(c) + s*rho - lam*c)
    """
    G = c.shape[0]
    inv_dx2 = 1.0 / (dx * dx)
    c_new = np.empty_like(c)
    for i in range(G):
        im = (i - 1) % G
        ip = (i + 1) % G
        for j in range(G):
            jm = (j - 1) % G
            jp = (j + 1) % G
            lap = (c[ip, j] + c[im, j] + c[i, jp] + c[i, jm] - 4.0 * c[i, j]) * inv_dx2
            c_new[i, j] = c[i, j] + dt_sub * (D * lap + s * rho[i, j] - lam * c[i, j])
    return c_new


@njit(cache=True)
def n_substeps_for_cfl(D_max: float, dt: float, dx: float, cfl_limit: float = 0.25) -> int:
    """Number of substeps so that effective dt keeps D*dt/dx^2 <= cfl_limit."""
    ratio = D_max * dt / (dx * dx)
    if ratio <= cfl_limit:
        return 1
    # need ratio / n <= cfl_limit  =>  n >= ratio / cfl_limit
    n = int(ratio / cfl_limit) + 1
    return n


@njit(cache=True)
def step_fields(
    c_a: np.ndarray,
    c_s: np.ndarray,
    rho: np.ndarray,
    D_a: float,
    D_s: float,
    s_a: float,
    s_s: float,
    lam_a: float,
    lam_s: float,
    dt: float,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance both fields by dt with automatic substepping.

    Returns updated copies (not in-place — Euler needs the old `c` array
    while computing each new cell).
    """
    G = c_a.shape[0]
    dx = L / G
    D_max = D_a if D_a > D_s else D_s
    n_sub = n_substeps_for_cfl(D_max, dt, dx, 0.25)
    dt_sub = dt / n_sub
    for _ in range(n_sub):
        c_a = step_field_one_substep(c_a, rho, D_a, s_a, lam_a, dt_sub, dx)
        c_s = step_field_one_substep(c_s, rho, D_s, s_s, lam_s, dt_sub, dx)
    return c_a, c_s


@njit(cache=True, fastmath=True)
def grad_field_at(
    c: np.ndarray,
    x: float,
    y: float,
    L: float,
) -> tuple[float, float]:
    """Bilinear sample of the gradient of `c` at world coord (x, y).

    Uses central differences on the grid, then bilinear interpolates the
    gradient field at the requested position. Periodic.
    """
    G = c.shape[0]
    dx = L / G
    inv_dx = 1.0 / dx
    inv_2dx = 0.5 * inv_dx

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

    # central-difference gradient at the four surrounding nodes
    def gradnode(ii, jj):
        ip = (ii + 1) % G
        im = (ii - 1) % G
        jp = (jj + 1) % G
        jm = (jj - 1) % G
        gxn = (c[jj, ip] - c[jj, im]) * inv_2dx
        gyn = (c[jp, ii] - c[jm, ii]) * inv_2dx
        return gxn, gyn

    g00x, g00y = gradnode(ix0, iy0)
    g10x, g10y = gradnode(ix1, iy0)
    g01x, g01y = gradnode(ix0, iy1)
    g11x, g11y = gradnode(ix1, iy1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    gx_out = w00 * g00x + w10 * g10x + w01 * g01x + w11 * g11x
    gy_out = w00 * g00y + w10 * g10y + w01 * g01y + w11 * g11y
    return gx_out, gy_out


@njit(cache=True)
def static_gaussian_field(L: float, G: int, x0: float, y0: float, sig: float, amp: float) -> np.ndarray:
    """Pre-baked Gaussian field for the static-source chemotaxis sanity test."""
    c = np.zeros((G, G), dtype=np.float64)
    dx = L / G
    half_L = 0.5 * L
    inv_2sig2 = 1.0 / (2.0 * sig * sig)
    for i in range(G):
        y = (i + 0.5) * dx
        dy = y - y0
        if dy > half_L:
            dy -= L
        elif dy < -half_L:
            dy += L
        for j in range(G):
            x = (j + 0.5) * dx
            dxw = x - x0
            if dxw > half_L:
                dxw -= L
            elif dxw < -half_L:
                dxw += L
            c[i, j] = amp * np.exp(-(dxw * dxw + dy * dy) * inv_2sig2)
    return c
