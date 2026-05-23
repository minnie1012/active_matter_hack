"""Phase 9c — single demo video showcasing the EMT phenotype switch.

Produces ``outputs/video_for_presentation/tme_emt_color_change.mp4`` where
tumor cells visibly transition in color from pink (epithelial, s_T ≈ 0)
to violet (mesenchymal, s_T ≈ 1) under combined EMT drivers:

  • a hypoxic core forming inside the tumor (k_hypoxia_EMT)
  • a dense ECM substrate around it (k_ECM_EMT)
  • suppressant accumulation at the rim (k_supp_EMT)

Angiogenesis is OFF (frozen bottom-trunk vessel) so the hypoxic core
persists and the EMT pump runs without VEGF-driven O2 rescue.

Uses the standard presentation renderer (4-plot sidebar: tumor count,
EMT level, max invasion distance, M1/M2 polarization).
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
RHO_E_INIT = 1.0       # heavier ECM → stronger k_ECM_EMT contribution


def make_params() -> BiophysicalParams:
    p = BiophysicalParams(
        T_final=100.0,
        N_T_initial=50,
        N_I_initial=150,           # moderate CD8 → tumor survives + EMT pumps
        N_M_initial=80,
        rho_E_init=RHO_E_INIT,
        s_m=1.0,
        k_deg=1.0,
        p_div=0.005,
        p_kill=0.03,
        p_phag=0.04,
        chi_s=12.0,                # stronger suppressant rim
    )

    # --- EMT ON, with aggressive drivers so pink → violet is dramatic ---
    p.s_T0 = 0.05                  # start nearly fully epithelial (pink)
    p.D_EMT = 0.003
    p.daughter_EMT_drift = 0.05    # daughters inherit + drift faster
    p.k_hypoxia_EMT = 0.9          # strong hypoxia → EMT coupling
    p.k_ECM_EMT = 0.15
    p.k_supp_EMT = 0.10
    p.k_MET = 0.03                 # slow reverse so transition is monotone
    p.v_epi = 0.06
    p.v_mes = 0.30
    p.s_m_epi = 0.0
    p.s_m_mes = 1.5

    # --- fibers + integrin OFF (cleaner pure-EMT showcase) ---
    p.J_fiber = 0.0
    p.fiber_alignment_strength = 0.0
    p.A_base = 0.5 / RHO_E_INIT    # traction neutral

    # --- hypoxia ON so the EMT pump has fuel ---
    p.D_O = 4.0
    p.supply_O = 1.0
    p.consumption_T = 0.7
    p.lambda_O = 0.05
    p.O_threshold = 10.0
    p.k_MMP_hyp = 2.0
    p.k_supp_hyp = 1.0
    p.k_div_hyp = 0.5

    # --- angiogenesis OFF: vessel sits at the bottom, no sprouting,
    # so the hypoxic core persists and drives the EMT switch ---
    p.n_vessels_init = 1
    p.chi_vessel = 0.0
    p.sprout_rate = 0.0

    return p


def main() -> None:
    out_dir = ROOT / "outputs" / "video_for_presentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== EMT_COLOR_CHANGE ===")
    params = make_params()
    print(f"  T_final={params.T_final}, "
          f"s_T0={params.s_T0}, k_hypoxia_EMT={params.k_hypoxia_EMT}")
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=SEED, snapshot_every=25)
    sim_dt = time.perf_counter() - t0
    print(f"  sim {sim_dt:.1f}s -- {len(out.pos_T_snapshots)} frames")
    print(f"  mean_EMT: {out.mean_EMT[0]:.3f} -> {out.mean_EMT[-1]:.3f}")
    if hasattr(out, "frac_mes") and out.frac_mes:
        print(f"  frac_mes (s_T > 0.6): "
              f"{out.frac_mes[0]:.3f} -> {out.frac_mes[-1]:.3f}")
    print(f"  N_T: start={out.n_T[0]}, end={out.n_T[-1]}, max={max(out.n_T)}")
    print(f"  hypoxic area (final): {out.hypoxic_area:.3f}")

    mp4 = out_dir / "tme_emt_color_change.mp4"
    t0 = time.perf_counter()
    render_presentation_video(
        out, mp4,
        title="EMT phenotype switch: tumor cells transition pink (epithelial) -> violet (mesenchymal)",
        fps=24, dpi=100, bitrate=4500,
    )
    print(f"  video {time.perf_counter()-t0:.1f}s -> {mp4}")


if __name__ == "__main__":
    main()
