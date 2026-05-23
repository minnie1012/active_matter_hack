"""Phase 8b — isolate each of the three biophysical mechanisms.

Runs the biophysical simulator four times with the SAME seed and the same
ECM substrate / immune populations, varying only which of EMT, ECM-fiber
alignment + integrin biphasic traction, and hypoxia are switched on:

  (1) baseline            — all three OFF        -> tme_baseline.mp4
  (2) only EMT            — EMT ON               -> tme_only_emt.mp4
  (3) only fiber+integrin — fiber/integrin ON    -> tme_only_fiber.mp4
  (4) all three           — EMT + fiber + hypoxia ON  -> tme_all_three.mp4

Also writes a 2x2 panel figure with the late-time frame of each run at
``outputs/figures/biophysical_mechanism_compare.png``.

How the toggles map to BiophysicalParams:
  - EMT off:   s_T0=0, all EMT drivers zeroed, daughter drift 0,
               v_epi=v_mes (so s_T value doesn't matter), s_m_epi=s_m_mes.
  - fiber/integrin off: J_fiber=0, A_base chosen so A_i=0.5 at
               rho_E_init -> traction = 4*A*(1-A) = 1 (neutral).
  - hypoxia off: D_O=0, consumption_T=0, O_threshold=-1 so H=0 everywhere
               and all hypoxia-coupled multipliers are zero.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.sim_biophysical import BiophysicalParams, run_biophysical
from src.render_tme import (
    render_biophysical_video,
    render_biophysical_still,
)


SEED = 11
T_FINAL = 60.0          # a bit shorter than the demo so 4 runs fit in time
SNAPSHOT_EVERY = 25
RHO_E_INIT = 0.8        # shared ECM substrate level


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
    )


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
    # Make phenotype EMT-state-independent: epi == mes == baseline.
    p.v_epi = 0.10
    p.v_mes = 0.10
    p.s_m_epi = 1.0
    p.s_m_mes = 1.0


def _fiber_on(p: BiophysicalParams) -> None:
    p.J_fiber = 1.5
    p.fiber_alignment_strength = 0.6
    p.A_base = 1.5                          # biphasic integrin engaged


def _fiber_off(p: BiophysicalParams) -> None:
    p.J_fiber = 0.0
    p.fiber_alignment_strength = 0.0
    # Set A_base so A_i = A_base * rho_E_init = 0.5  ->  traction = 1.0
    p.A_base = 0.5 / RHO_E_INIT             # ~= 0.625
    # (clip to [0,1] inside the kernel makes this exact.)


def _hypoxia_on(p: BiophysicalParams) -> None:
    p.D_O = 4.0
    p.supply_O = 1.0
    p.consumption_T = 0.5
    p.lambda_O = 0.05
    p.O_threshold = 8.0
    p.k_MMP_hyp = 2.0
    p.k_supp_hyp = 1.0
    p.k_div_hyp = 0.5


def _hypoxia_off(p: BiophysicalParams) -> None:
    p.D_O = 0.0
    p.consumption_T = 0.0
    p.O_threshold = -1.0
    p.k_MMP_hyp = 0.0
    p.k_supp_hyp = 0.0
    p.k_div_hyp = 0.0


def baseline_params() -> BiophysicalParams:
    p = BiophysicalParams(**_shared_kwargs())
    _emt_off(p); _fiber_off(p); _hypoxia_off(p)
    return p


def only_emt_params() -> BiophysicalParams:
    p = BiophysicalParams(**_shared_kwargs())
    _emt_on(p); _fiber_off(p); _hypoxia_off(p)
    return p


def only_fiber_params() -> BiophysicalParams:
    p = BiophysicalParams(**_shared_kwargs())
    _emt_off(p); _fiber_on(p); _hypoxia_off(p)
    return p


def all_three_params() -> BiophysicalParams:
    p = BiophysicalParams(**_shared_kwargs())
    _emt_on(p); _fiber_on(p); _hypoxia_on(p)
    return p


def run_and_render(label: str, params: BiophysicalParams) -> "BiophysicalOut":
    print(f"\n=== {label.upper()} ===")
    t0 = time.perf_counter()
    out = run_biophysical(params=params, seed=SEED, snapshot_every=SNAPSHOT_EVERY)
    print(f"  sim {time.perf_counter()-t0:.1f}s -- {len(out.pos_T_snapshots)} frames")
    print(f"  final n_T={out.n_T[-1]}, n_I={out.n_I[-1]}, n_M={out.n_M[-1]}")
    print(f"  mean_EMT  : {min(out.mean_EMT):.3f} -> {max(out.mean_EMT):.3f}")
    print(f"  frac_mes  : {min(out.frac_mes):.3f} -> {max(out.frac_mes):.3f}")
    print(f"  max invasion: {max(out.max_invasion_distance):.2f}")
    print(f"  hypoxic area: {out.hypoxic_area:.3f}, "
          f"ECM degraded: {out.ecm_degraded_area:.3f}")
    video_path = ROOT / "outputs" / "videos" / f"tme_{label}.mp4"
    t0 = time.perf_counter()
    render_biophysical_video(out, video_path, fps=24, dpi=100, bitrate=4500)
    print(f"  video {time.perf_counter()-t0:.1f}s -> {video_path.name}")
    return out


def compare_figure(outs: dict, path: Path) -> None:
    """2x2 grid of late-time stills."""
    tmps = {}
    for label, out in outs.items():
        k = len(out.pos_T_snapshots) - 1
        tmp = path.with_name(f"_tmp_{label}.png")
        render_biophysical_still(out, k, tmp, dpi=130)
        tmps[label] = tmp

    fig = plt.figure(figsize=(20, 12), dpi=130)
    fig.patch.set_facecolor("white")
    titles = {
        "baseline":   "BASELINE — all three OFF (reduces to sim_combined)",
        "only_emt":   "ONLY EMT (fibers + hypoxia off)",
        "only_fiber": "ONLY FIBERS + INTEGRIN (EMT + hypoxia off)",
        "all_three":  "ALL THREE — EMT + fibers + hypoxia",
    }
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    keys = ["baseline", "only_emt", "only_fiber", "all_three"]
    for (row, col), key in zip(positions, keys):
        x = 0.0 + col * 0.50
        y = 0.50 - row * 0.50
        ax = fig.add_axes([x + 0.005, y + 0.005, 0.49, 0.46]); ax.axis("off")
        img = plt.imread(tmps[key])
        ax.imshow(img)
        fig.text(x + 0.25, y + 0.475, titles[key], ha="center", va="top",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    for tmp in tmps.values():
        tmp.unlink(missing_ok=True)
    print(f"  comparison -> {path}")


def main():
    outs = {
        "baseline":   run_and_render("baseline",   baseline_params()),
        "only_emt":   run_and_render("only_emt",   only_emt_params()),
        "only_fiber": run_and_render("only_fiber", only_fiber_params()),
        "all_three":  run_and_render("all_three",  all_three_params()),
    }
    fig_path = ROOT / "outputs" / "figures" / "biophysical_mechanism_compare.png"
    compare_figure(outs, fig_path)

    # ---- summary table ----
    print("\n=== summary ===")
    print(f"{'scenario':14s}  {'N_T_end':>7s}  {'mean_EMT':>9s}  "
          f"{'frac_mes':>9s}  {'invasion':>9s}  {'hyp_area':>9s}  {'ecm_deg':>9s}")
    for label, out in outs.items():
        print(f"{label:14s}  {out.n_T[-1]:7d}  "
              f"{out.mean_EMT[-1]:9.3f}  {out.frac_mes[-1]:9.3f}  "
              f"{max(out.max_invasion_distance):9.2f}  "
              f"{out.hypoxic_area:9.3f}  {out.ecm_degraded_area:9.3f}")


if __name__ == "__main__":
    main()
