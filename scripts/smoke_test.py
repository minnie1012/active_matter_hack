"""Phase 1 smoke + profiling gate.

Run from project root:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.sim import TumorParams, run_single_species, step_single_species
from src.sim import init_tumor_uniform


def warmup_jit():
    """Trigger JIT compile on small input so subsequent timed runs are fair."""
    params = TumorParams(N_max=64)
    pos, theta, alive = init_tumor_uniform(20, params.L, params.N_max, seed=0)
    pa = params.to_array()
    for _ in range(3):
        step_single_species(pos, theta, alive, pa)


def smoke():
    print("=== smoke test ===")
    params = TumorParams(N_max=2048)
    out = run_single_species(
        params,
        n_initial=50,
        n_steps=200,
        init="disk",
        init_radius=5.0,
        snapshot_every=50,
        seed=42,
    )
    print(f"snapshots: {len(out.pos_snapshots)}")
    print(f"n_alive trajectory: {out.n_alive}")
    print(f"final positions shape: {out.pos_snapshots[-1].shape}")
    assert out.pos_snapshots[-1].shape[1] == 2
    assert not np.isnan(out.pos_snapshots[-1]).any()
    print("smoke OK\n")


def profiling_gate():
    print("=== profiling gate (500 ABPs x 1000 steps, no proliferation) ===")
    params = TumorParams(N_max=512)
    np.random.seed(7)
    pos, theta, alive = init_tumor_uniform(500, params.L, params.N_max, seed=7)
    pa = TumorParams(**{**params.__dict__, "p_div": 0.0}).to_array()

    t0 = time.perf_counter()
    for _ in range(1000):
        step_single_species(pos, theta, alive, pa)
    elapsed = time.perf_counter() - t0
    print(f"elapsed: {elapsed:.3f} s for 1000 steps -> {elapsed*1e3/1000:.3f} ms/step")
    gate_ok = elapsed < 5.0
    print(f"GATE: {'PASS' if gate_ok else 'FAIL'} (threshold 5.0 s)")
    return elapsed, gate_ok


def profiling_with_proliferation():
    print("=== same gate but WITH proliferation ===")
    params = TumorParams(N_max=2048)
    np.random.seed(11)
    pos, theta, alive = init_tumor_uniform(500, params.L, params.N_max, seed=11)
    pa = params.to_array()

    t0 = time.perf_counter()
    for _ in range(1000):
        step_single_species(pos, theta, alive, pa)
    elapsed = time.perf_counter() - t0
    n_after = int(alive.sum())
    print(f"elapsed: {elapsed:.3f} s for 1000 steps -> {n_after} alive")
    return elapsed, n_after


if __name__ == "__main__":
    print("warming up JIT...")
    warmup_jit()
    print("warm.\n")
    smoke()
    profiling_gate()
    profiling_with_proliferation()
