"""Phase-diagram sweep.

Runs `src.sim.run` over a 2D grid of (rho_I, alpha) with multiple seeds per
point, in parallel. Saves to outputs/data/phase_grid.npz.

Usage:
    python -m src.sweep
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim import SimParams, run


@dataclass
class SweepSpec:
    rho_I_values: np.ndarray
    alpha_values: np.ndarray
    seeds: list
    T_final: float = 200.0
    snapshot_every: int = 100      # coarse snapshots during sweep to save memory
    save_fields: bool = False
    base_params: Optional[SimParams] = None


def _one_run(rho_I: int, alpha: float, seed: int, spec: SweepSpec) -> tuple:
    """Worker: returns (rho_I, alpha, seed, final_tumor_fraction, n_T_traj, n_I_traj, times)."""
    p = spec.base_params if spec.base_params is not None else SimParams()
    out = run(
        rho_I=int(rho_I),
        alpha=float(alpha),
        seed=int(seed),
        T_final=spec.T_final,
        snapshot_every=spec.snapshot_every,
        params=p,
        save_fields=spec.save_fields,
    )
    return (
        int(rho_I),
        float(alpha),
        int(seed),
        float(out.final_tumor_fraction),
        np.asarray(out.n_T, dtype=np.float32),
        np.asarray(out.n_I, dtype=np.float32),
        np.asarray(out.times, dtype=np.float32),
    )


def run_sweep(spec: SweepSpec, n_jobs: int = -1, verbose: bool = True) -> dict:
    """Run all (rho_I, alpha, seed) combinations in parallel.

    Returns a dict with:
      - grid: (n_rho, n_alpha, n_seeds) array of final_tumor_fraction
      - mean: (n_rho, n_alpha) mean over seeds (log-space friendly)
      - rho_I_values, alpha_values, seeds
      - trajectories: dict[(ri, ai, si)] -> (n_T, n_I, times)
    """
    rho_vals = np.asarray(spec.rho_I_values)
    alpha_vals = np.asarray(spec.alpha_values)
    seeds = list(spec.seeds)
    n_r, n_a, n_s = len(rho_vals), len(alpha_vals), len(seeds)

    work = [
        (rho_vals[ri], alpha_vals[ai], seeds[si])
        for ri in range(n_r) for ai in range(n_a) for si in range(n_s)
    ]
    if verbose:
        print(f"sweep: {n_r} x {n_a} x {n_s} = {len(work)} runs on {n_jobs} jobs")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)(
        delayed(_one_run)(rho_I, alpha, seed, spec) for (rho_I, alpha, seed) in work
    )
    if verbose:
        print(f"sweep wall time: {time.perf_counter() - t0:.1f} s")

    grid = np.zeros((n_r, n_a, n_s), dtype=np.float64)
    trajectories = {}
    rho_index = {int(v): i for i, v in enumerate(rho_vals)}
    alpha_index = {float(v): i for i, v in enumerate(alpha_vals)}
    seed_index = {int(s): i for i, s in enumerate(seeds)}
    for (ri, ai, si, frac, nT, nI, tt) in results:
        i = rho_index[int(ri)]; j = alpha_index[float(ai)]; k = seed_index[int(si)]
        grid[i, j, k] = frac
        trajectories[(i, j, k)] = (nT, nI, tt)

    # Geometric mean across seeds (sensible for log-scaled order param).
    # Clip to avoid log(0).
    safe = np.clip(grid, 1e-2, 1e2)
    log_mean = np.exp(np.mean(np.log(safe), axis=2))

    return {
        "grid": grid,
        "mean": log_mean,
        "rho_I_values": rho_vals,
        "alpha_values": alpha_vals,
        "seeds": np.asarray(seeds),
        "trajectories": trajectories,
    }


def save_sweep(out_dict: dict, out_path: Path) -> None:
    """Persist the sweep grid + trajectories to a .npz file."""
    # flatten trajectories into ragged arrays via object dtype.
    keys = sorted(out_dict["trajectories"].keys())
    traj_keys = np.array(keys, dtype=np.int64)
    traj_nT = np.array(
        [out_dict["trajectories"][k][0] for k in keys], dtype=object
    )
    traj_nI = np.array(
        [out_dict["trajectories"][k][1] for k in keys], dtype=object
    )
    traj_t = np.array(
        [out_dict["trajectories"][k][2] for k in keys], dtype=object
    )
    np.savez(
        out_path,
        grid=out_dict["grid"],
        mean=out_dict["mean"],
        rho_I_values=out_dict["rho_I_values"],
        alpha_values=out_dict["alpha_values"],
        seeds=out_dict["seeds"],
        traj_keys=traj_keys,
        traj_nT=traj_nT,
        traj_nI=traj_nI,
        traj_t=traj_t,
        allow_pickle=True,
    )


# ---------------------------------------------------------------------------

def main_pilot():
    """9-point pilot: corners + edges + center to verify three phases exist."""
    spec = SweepSpec(
        rho_I_values=np.array([20, 100, 500]),
        alpha_values=np.array([0.0, 5.0, 20.0]),
        seeds=[1],
        T_final=200.0,
        snapshot_every=200,
    )
    out = run_sweep(spec, n_jobs=-1)
    print("\npilot grid (rows=rho_I, cols=alpha):")
    print(f"  rho_I:  {spec.rho_I_values}")
    print(f"  alpha:  {spec.alpha_values}")
    print(out["mean"])
    out_path = ROOT / "outputs" / "data" / "pilot.npz"
    save_sweep(out, out_path)
    print(f"saved -> {out_path}")
    return out


def main_full(grid_size: int = 8, n_seeds: int = 3):
    """Full sweep on a log-spaced grid."""
    rho_vals = np.unique(np.round(np.logspace(np.log10(10), np.log10(800), grid_size)).astype(int))
    alpha_vals = np.linspace(0.0, 25.0, grid_size)
    spec = SweepSpec(
        rho_I_values=rho_vals,
        alpha_values=alpha_vals,
        seeds=list(range(1, n_seeds + 1)),
        T_final=200.0,
        snapshot_every=200,
    )
    out = run_sweep(spec, n_jobs=-1)
    out_path = ROOT / "outputs" / "data" / "phase_grid.npz"
    save_sweep(out, out_path)
    print(f"saved -> {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    if args.mode == "pilot":
        main_pilot()
    else:
        main_full(grid_size=args.grid_size, n_seeds=args.n_seeds)
