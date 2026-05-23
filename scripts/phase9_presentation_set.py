"""Phase 9 — standardized 5-video presentation set.

Runs the biophysical simulator five times with the SAME seed and the same
ECM / immune / division / killing baseline, varying only which of EMT,
ECM-fiber alignment + integrin biphasic traction, and hypoxia + angiogenesis
are switched on:

  (1) baseline       — all off; macrophages + ECM substrate only
  (2) with_hypoxia   — hypoxia + angiogenesis on
  (3) only_emt       — EMT on
  (4) only_fiber     — fibers + integrin on
  (5) full           — EMT + fibers + hypoxia + angiogenesis

Outputs to ``outputs/video_for_presentation/tme_{label}.mp4``.

All scenarios seed a single bottom-trunk vessel. In the "hypoxia off"
scenarios the vessel is frozen (``chi_vessel=0, sprout_rate=0``) so the
trunk just sits at the bottom of the box without sprouting; this keeps the
static parent strip visible in every video and the comparison fair.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.sim_biophysical import BiophysicalParams, run_biophysical
from src.render_presentation import render_presentation_video


# ===========================================================================
# Shared knobs
# ===========================================================================

SEED = 11
T_FINAL = 80.0
SNAPSHOT_EVERY = 25
RHO_E_INIT = 0.8

OUT_DIR = ROOT / "outputs" / "video_for_presentation"


def _shared_kwargs() -> dict:
    """Cell counts, ECM substrate, baseline rates — identical across runs."""
    return dict(
        T_final=T_FINAL,
        N_T_initial=50,
        N_I_initial=200,
        N_M_initial=80,
        rho_E_init=RHO_E_INIT,
        s_m=1.0,
        k_deg=1.0,
        p_div=0.005,
        p_kill=0.04,
        p_phag=0.04,
        chi_s=10.0,
        # always start with one bottom-trunk vessel so the parent strip is
        # visible in every video (we zero out vessel motion / sprouting in
        # the no-hypoxia scenarios via _hypoxia_off).
        n_vessels_init=1,
    )


# ===========================================================================
# Toggle helpers (copied verbatim from phase8b_mechanism_compare.py)
# ===========================================================================

def _emt_on(p: BiophysicalParams) -> None:
    p.s_T0 = 0.15
    p.D_EMT = 0.002
    p.daughter_EMT_drift = 0.03
    p.k_hypoxia_EMT = 0.6
    p.k_ECM_EMT = 0.10
    p.k_supp_EMT = 0.05
    p.k_MET = 0.05
    p.v_epi = 0.06
    p.v_mes = 0.30
    p.s_m_epi = 0.0
    p.s_m_mes = 1.5


def _emt_off(p: BiophysicalParams) -> None:
    p.s_T0 = 0.0
    p.D_EMT = 0.0
    p.daughter_EMT_drift = 0.0
    p.k_hypoxia_EMT = 0.0
    p.k_ECM_EMT = 0.0
    p.k_supp_EMT = 0.0
    p.k_MET = 0.0
    p.v_epi = 0.10
    p.v_mes = 0.10
    p.s_m_epi = 1.0
    p.s_m_mes = 1.0


def _fiber_on(p: BiophysicalParams) -> None:
    p.J_fiber = 1.5
    p.fiber_alignment_strength = 0.6
    p.A_base = 1.5


def _fiber_off(p: BiophysicalParams) -> None:
    p.J_fiber = 0.0
    p.fiber_alignment_strength = 0.0
    p.A_base = 0.5 / RHO_E_INIT


def _hypoxia_on(p: BiophysicalParams) -> None:
    p.D_O = 4.0
    p.supply_O = 1.0
    p.consumption_T = 0.5
    p.lambda_O = 0.05
    p.O_threshold = 8.0
    p.k_MMP_hyp = 2.0
    p.k_supp_hyp = 1.0
    p.k_div_hyp = 0.5
    # keep default vessel chemotaxis + sprouting on


def _hypoxia_on_early_with_immune(p: BiophysicalParams) -> None:
    """Variant used for the `with_hypoxia` showcase scenario.

    Goal: a clear narrative arc — strong initial CD8 kills tumor early, then
    hypoxia builds, vessels sprout aggressively, hypoxia-amplified suppression
    blunts CD8, tumor regrows.
    """
    _hypoxia_on(p)
    # boost CD8 so the tumor visibly decreases in the first few sim units
    p.N_I_initial = 350
    p.p_kill = 0.07
    # speed up O2 depletion -> earlier hypoxia
    p.consumption_T = 0.9
    # stronger / earlier angiogenesis
    p.s_VEGF_hyp = 2.0
    p.sprout_VEGF_thresh = 0.002
    p.chi_vessel = 120.0
    # crank hypoxic suppression so CD8 falters once hypoxia builds
    p.k_supp_hyp = 1.8


def _hypoxia_off(p: BiophysicalParams) -> None:
    p.D_O = 0.0
    p.consumption_T = 0.0
    p.O_threshold = -1.0
    p.k_MMP_hyp = 0.0
    p.k_supp_hyp = 0.0
    p.k_div_hyp = 0.0
    # freeze the seeded bottom-trunk vessel so the static parent strip
    # stays in place and no sprouting happens
    p.chi_vessel = 0.0
    p.sprout_rate = 0.0


# ===========================================================================
# Scenario table
# ===========================================================================

SCENARIOS = [
    (
        "baseline",
        [_emt_off, _fiber_off, _hypoxia_off],
        "Baseline: no EMT, no fibers, no hypoxia",
    ),
    (
        "with_hypoxia",
        [_emt_off, _fiber_off, _hypoxia_on_early_with_immune],
        "Hypoxia + angiogenesis only",
    ),
    (
        "only_emt",
        [_emt_on, _fiber_off, _hypoxia_off],
        "EMT only",
    ),
    (
        "only_fiber",
        [_emt_off, _fiber_on, _hypoxia_off],
        "ECM fibers + integrin only",
    ),
    (
        "full",
        [_emt_on, _fiber_on, _hypoxia_on],
        "Full model: EMT + fibers + hypoxia + angiogenesis",
    ),
]


# ===========================================================================
# Driver helpers
# ===========================================================================

def build_params(toggles) -> BiophysicalParams:
    p = BiophysicalParams(**_shared_kwargs())
    for t in toggles:
        t(p)
    return p


def run_and_render(label: str, toggles, title: str,
                   t_final: float = T_FINAL,
                   snapshot_every: int = SNAPSHOT_EVERY) -> "BiophysicalOut":
    print(f"\n=== {label.upper()} ===")
    params = build_params(toggles)
    # honour caller-specified t_final / snapshot_every (used by the smoke
    # test below) without touching the BiophysicalParams default elsewhere
    params.T_final = t_final

    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=SEED,
                          snapshot_every=snapshot_every)
    sim_dt = time.perf_counter() - t0
    n_frames = len(out.pos_T_snapshots)
    n_v_last = int(out.n_vessels[-1]) if out.n_vessels else 0
    print(f"  sim {sim_dt:.1f}s -- {n_frames} frames")
    print(f"  final n_T={out.n_T[-1]}, n_I={out.n_I[-1]}, "
          f"n_M={out.n_M[-1]}, n_vessels={n_v_last}")
    print(f"  mean_EMT  : {min(out.mean_EMT):.3f} -> {max(out.mean_EMT):.3f}")
    print(f"  max invasion: {max(out.max_invasion_distance):.2f}")
    pm = getattr(out, "mean_pM", None)
    if pm:
        print(f"  mean_pM   : {min(pm):+.3f} -> {max(pm):+.3f}")

    video_path = OUT_DIR / f"tme_{label}.mp4"
    t0 = time.perf_counter()
    render_presentation_video(out, video_path, title=title,
                              fps=24, dpi=100, bitrate=4500)
    print(f"  video {time.perf_counter()-t0:.1f}s -> {video_path.name}")
    return out


def smoke_test():
    """Quick verification of the renderer on a tiny run."""
    print("\n=== smoke test: full scenario, T_final=4.0, snapshot_every=10 ===")
    label, toggles, title = SCENARIOS[-1]  # "full"
    out = run_and_render(
        label="smoke_" + label,
        toggles=toggles,
        title="[SMOKE] " + title,
        t_final=4.0,
        snapshot_every=10,
    )
    # remove the smoke artefact so it doesn't end up in the deliverable
    smoke_path = OUT_DIR / f"tme_smoke_{label}.mp4"
    if smoke_path.exists():
        smoke_path.unlink()
    return out


def print_summary(outs: dict):
    print("\n=== summary ===")
    header = (f"{'scenario':14s}  {'N_T':>5s}  {'EMT[-1]':>8s}  "
              f"{'invasion':>9s}  {'pM[-1]':>7s}  {'vess[-1]':>8s}  "
              f"{'hyp_area':>9s}")
    print(header)
    for label, out in outs.items():
        n_T = out.n_T[-1] if out.n_T else 0
        emt_last = out.mean_EMT[-1] if out.mean_EMT else 0.0
        inv_max = max(out.max_invasion_distance) if out.max_invasion_distance else 0.0
        pm = getattr(out, "mean_pM", None)
        pm_last = pm[-1] if pm else 0.0
        n_v_last = out.n_vessels[-1] if out.n_vessels else 0
        hyp_area = getattr(out, "hypoxic_area", 0.0)
        print(f"{label:14s}  {n_T:5d}  {emt_last:8.3f}  "
              f"{inv_max:9.2f}  {pm_last:+7.3f}  {n_v_last:8d}  "
              f"{hyp_area:9.3f}")


def main(do_smoke: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if do_smoke:
        smoke_test()

    outs = {}
    for label, toggles, title in SCENARIOS:
        outs[label] = run_and_render(label, toggles, title)

    print_summary(outs)
    print(f"\nAll 5 videos saved to {OUT_DIR}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the T_final=4.0 smoke test and go straight "
                         "to the full set")
    ap.add_argument("--smoke-only", action="store_true",
                    help="just run the smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        smoke_test()
    else:
        main(do_smoke=not args.no_smoke)
