# Simulation summary — what each result shows

Every simulation in the project, what was varied, what was observed, and the biological interpretation.

## Model 1 — minimal three-species (`sim.py`, `sim_extended.py`)

| # | Figure | Variable swept / mechanism tested | Key observation | Biological meaning |
|---|---|---|---|---|
| **1** | `phase_diagram.png` | Initial T-cell count `ρ_I` × suppressant strength `α` over 8 × 8 × 3 seeds (192 runs) | Sharp, bistable transition: tiny clearance wedge (high `ρ_I`, low `α`); narrow control / dormancy band at the boundary; escape fills the rest of the plane | Clinical "hot / excluded / cold" trichotomy; predicts checkpoint-inhibitor response is a **threshold**, not a smooth dose curve |
| **2** | `phase6_combo_treatment.png` | Treatment knobs at `t = 20`: (i) none, (ii) `α → 0` (anti-PD-1), (iii) `b_M1 → 15` (TAM repol), (iv) both | Anti-PD-1 alone rescues at moderate `α`. TAM repolarization alone **fails** (the `−κ_IL c_IL10` self-feedback locks M2). Combo rescues reliably | Combination therapy (CSF1R inhibitor + anti-PD-1) is the only protocol that crosses the phase boundary in the boundary region |
| **3** | `phase6_macrophage_polarization.png` | Untreated 3-species run, tracking `⟨p_k⟩` over time | As tumor grows and `c_s` builds up, the macrophage population swings from neutral `⟨p⟩ ≈ 0` toward M2 `⟨p⟩ ≈ −0.3`; IL-10 field develops around the core | M2-skewing is **spatially slaved** to the tumor's chemical halo. Polarization is a continuum, not a switch |
| **4** | `phase6_heterogeneity_pressure.png` | Pressure-gated proliferation `p_div(P_i) = p_div^0 · max(0, 1 − P_i / P*)` instead of neighbor-count gate | Core tumor cells reach `P_i ≈ P*` and stop dividing; surface cells continue. Visible homeostatic-pressure plateau | Mechanism by which the "control / dormancy" phase becomes genuinely stable (Byrne–Drasdo): the tumor self-limits at high local stress |
| **5** | `phase6b_ecm.png` | ECM density `ρ_E,0` × MMP secretion `s_m` (3 conditions) | Baseline (no ECM) saturates; dense ECM + MMP off suppresses tumor to ~660 cells (porosity gate); dense ECM + MMP on recovers full escape, with visible depleted-ECM **cavity** around the tumor | Friedl's "trail-blazing" invasion signature: tumor cells digest their own matrix to escape mechanical jamming |
| **6** | `phase6c_adhesion.png` | Cadherin attractive shoulder `k_adh` × Vicsek alignment `J_align` (3 conditions) | No adhesion → diffuse cell cloud (NN distance 1.21σ). Weak cadherin → cohesive blob (NN = 0.99σ). Strong cadherin + alignment → hexagonally packed sheet | Collective vs. single-cell invasion mode (Friedl & Gilmour); cadherin loss is the morphological signature of EMT |

## Model 2 — biophysical TME (`sim_combined.py`, `sim_tme.py`, `sim_biophysical.py`)

| # | Figure | Variable swept / mechanism tested | Key observation | Biological meaning |
|---|---|---|---|---|
| **7** | `combined_panel.png` | All Model-1 extensions enabled together (macrophages + heterogeneity + pressure + ECM/MMP + cadherin) | Cohesive cadherin-glued tumor digests its way into the matrix; M2-skewed macrophages form a peri-tumoral ring; CD8s excluded | Confirms the Model-1 components compose coherently before adding mechanobiology |
| **8** | `tme_panel.png` | Full TME with 8 cell species: tumor, CD8, NK, DC, MDSC, M1, M2, CAFs, plus vessels | At `t = 90`, hypoxic fraction = 83.8%. **CAFs form an annular ring around the tumor**; T cells are blocked at the CAF ring | The **structural** mechanism for immune exclusion — desmoplastic stroma physically blocks T-cell access regardless of chemotaxis strength |
| **9** | `tme_hypoxia_compare.png` | Hypoxia + angiogenesis ON vs. all hypoxia couplings disabled | With hypoxia: 81.6% hypoxic fraction develops; vessels sprout into hypoxic regions. Without hypoxia: 0% hypoxic fraction | Hypoxia is the resistance reservoir — kills are softened in low-O₂ regions, EMT is driven, suppressant secretion boosted. Angiogenesis (VEGF → vessel sprouting) is the rescue loop |
| **10** | `biophysical_panel.png` | EMT scalar `s_T,i` per cell driven by hypoxia + ECM + suppressant | Mean `⟨s_T⟩` saturates near 1 by t ≈ 40; mesenchymal fraction (`s_T > 0.6`) reaches 100%; max invasion distance grows near-linearly | Hypoxic core drives EMT at the margin: cells lose cadherin, gain motility, secrete more MMP — the classical "invasive front" pattern in carcinoma histology |
| **11** | `biophysical_mechanism_compare.png` | 2×2 ablation: baseline / only EMT+fibers / only fibers+integrin / all three | Baseline: cohesive epithelial tumor, no invasion. Fibers alone: still cohesive. EMT alone: partial streaming but no metabolic resistance. **All three on**: fully invasive phenotype, 100% mesenchymal, fiber-aligned streamers, hypoxic resistant core | Each mechanobiology ingredient contributes a **distinct** piece of the invasive phenotype. EMT enables, fibers guide, hypoxia hardens. None alone is sufficient |

## Headline by-the-numbers

| Quantity | Model 1 | Model 2 |
|---|---|---|
| Cell species | 3 (tumor + T + M) | 8 (+ NK, DC, MDSC, CAFs, vessels) |
| Scalar fields | 3 (`c_a`, `c_s`, `c_IL10`) | 6 (+ `ρ_E`, `m` MMP, `c_O2`, `c_VEGF`) |
| Per-cell state | position, heading | + EMT `s_T`, integrin `e`, hypoxia `H` |
| Treatment knobs | `α` (anti-PD-1), `b_M1` (TAM repol) | + angiogenesis on/off, ECM density |
| Phase diagram | 3 phases in (`ρ_I`, `α`) | – (single-run regime study) |
| Headline figure | `phase_diagram.png` | `biophysical_mechanism_compare.png` |
