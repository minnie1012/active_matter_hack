"""Phase 9b — single demo video with EARLIER, more dramatic angiogenesis.

Produces ``outputs/video_for_presentation/tme_hypoxia_earlier.mp4`` showing
the textbook tumor-immune-angiogenesis arc:

    1) strong initial CD8 wave → tumor shrinks visibly
    2) tumor mass consumes O2 → hypoxic core forms early
    3) hypoxic core secretes VEGF → vessels sprout AGGRESSIVELY from the
       bottom trunk and reach the tumor quickly
    4) hypoxia amplifies suppressant action on CD8 → tumor regrows under
       angiogenic support

Uses the same renderer as the phase 9 presentation set (4-plot sidebar:
tumor count, EMT level, max invasion distance, M1/M2 polarization).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim_biophysical import BiophysicalParams, run_biophysical
from src.render_presentation import render_presentation_video


SEED = 11
RHO_E_INIT = 0.8


def make_params() -> BiophysicalParams:
    p = BiophysicalParams(
        T_final=100.0,                # longer arc: shrink + regrow
        N_T_initial=50,
        N_I_initial=400,              # heavy CD8 wave → tumor decreases fast
        N_M_initial=80,
        rho_E_init=RHO_E_INIT,
        s_m=1.0,
        k_deg=1.0,
        p_div=0.005,
        p_kill=0.08,                  # very effective initial killing
        p_phag=0.04,
        chi_s=10.0,
    )

    # --- EMT off (this scenario isolates the hypoxia/angio narrative) ---
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

    # --- fibers + integrin off (traction neutral via A_i = 0.5) ---
    p.J_fiber = 0.0
    p.fiber_alignment_strength = 0.0
    p.A_base = 0.5 / RHO_E_INIT

    # --- hypoxia ON with aggressive / early dynamics ---
    p.D_O = 4.0
    p.supply_O = 1.0
    p.consumption_T = 1.2             # very fast O2 depletion
    p.lambda_O = 0.05
    p.O_threshold = 12.0              # more area registers as hypoxic
    p.k_MMP_hyp = 2.0
    p.k_supp_hyp = 2.5                # strong hypoxia-amplified suppression
    p.k_div_hyp = 0.5

    # --- angiogenesis: faster + earlier sprouting ---
    p.n_vessels_init = 1
    p.s_VEGF_hyp = 3.0                # 3x VEGF source from hypoxic tumor
    p.sprout_VEGF_thresh = 0.001      # sprout at very low VEGF
    p.chi_vessel = 150.0              # vessels drift fast
    p.sprout_rate = 0.05              # higher per-step sprouting probability

    return p


def main() -> None:
    out_dir = ROOT / "outputs" / "video_for_presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== HYPOXIA_EARLIER ===")
    params = make_params()
    print(f"  T_final={params.T_final}, N_I_initial={params.N_I_initial}, "
          f"p_kill={params.p_kill}")
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=SEED, snapshot_every=25)
    sim_dt = time.perf_counter() - t0
    print(f"  sim {sim_dt:.1f}s -- {len(out.pos_T_snapshots)} frames")
    print(f"  N_T trajectory: start={out.n_T[0]}, "
          f"min={min(out.n_T)}, end={out.n_T[-1]}")
    print(f"  n_vessels: 1 -> {out.n_vessels[-1] if out.n_vessels else 0}")
    if hasattr(out, "mean_pM") and out.mean_pM:
        print(f"  mean_pM: {min(out.mean_pM):+.3f} -> {max(out.mean_pM):+.3f}")
    print(f"  hypoxic area (final): {out.hypoxic_area:.3f}")

    mp4 = out_dir / "tme_hypoxia_earlier.mp4"
    t0 = time.perf_counter()
    render_presentation_video(
        out, mp4,
        title="Hypoxia + angiogenesis (early onset): tumor decreases, then regrows under angiogenic support",
        fps=24, dpi=100, bitrate=4500,
    )
    print(f"  video {time.perf_counter()-t0:.1f}s -> {mp4}")


if __name__ == "__main__":
    main()
