# Calibration & Validation of the Tumor–Immune Active-Matter Model

*Research notes for the Vibe Coding Active Matter & Biophysics Hackathon, UCSD, 2026-05.*
*Author: I-Tsen Tsai (i3tsai@ucsd.edu).*

Audience: computational physicist with no biology background. **Jargon defined on first use.**
Companion to `docs/model.md`. The model has parameters listed in §7 of that document;
this report grounds each one in measurement.

---

## 1. Parameter calibration

Cross-reference column 1 to the symbols in `docs/model.md` §7. All numerical values are reported
in their natural biological units (μm, min, s); converting to the model's dimensionless units
requires choosing a length scale (set by tumor cell diameter σ ≈ 10–20 μm) and a time scale
(set by either v_T/σ or D_R^{−1}).

### 1.1 Tumor cell migration speed (v_T)

"Intravital" = imaging inside a living animal, typically through a surgical window with a
multiphoton microscope.

* **Mesenchymal mode** (single cells crawling through ECM via proteolysis): **< 1 μm/min**, i.e. < 60 μm/h.
  Reviewed by Friedl & Wolf, *Nat. Rev. Cancer* (2003) and many follow-ups; the **0.1–0.5 μm/min**
  band is typical for breast-carcinoma cells in collagen.
* **Amoeboid / collective streaming** in breast tumors: **≈ 2–4 μm/min** for cells in
  "streams" near vessels — Patsialou *et al.*, *IntraVital* (2013, PMC3908591), intravital MPM of
  human MDA-MB-231 xenografts.
* **Glioblastoma**: up to **100 μm/h ≈ 1.7 μm/min** along white-matter tracts
  (Strobl *et al., Sci. Rep.* 2019, PMC6375955; reviewed in *Front. Oncol.* 2018).

**Plug value for v_T:** ≈ 0.2 μm/min ≈ **0.02 cell-diameters/min**. Our model currently runs
at v_T = 0.1 (model units, ≈ 1 σ per 10 model-time units), within an order of magnitude.

### 1.2 T-cell migration speed (v_I)

* **Lymph node naive T cells:** **10–15 μm/min** is the textbook number from the foundational
  two-photon studies — Miller *et al., Science* (2002); Bousso & Robey, *Nat. Immunol.* (2003).
* **Antigen-positive solid tumors (slow patrolling state):** **4 ± 2 μm/min**, with frequent
  stops — Boissonnas *et al., J. Exp. Med.* (2007); same study reports
  **10 ± 4 μm/min** in antigen-negative controls.
* **Human lung and ovarian carcinomas (resident T cells imaged in fresh slices):** mean speeds
  **2–6 μm/min**, two-state random-walk behavior — Salmon *et al., Front. Immunol.* (2015,
  3389/fimmu.2015.00500).

**Plug value for v_I:** ≈ 4 μm/min ≈ **0.3 cell-diameters/min**. The ratio v_I / v_T ≈ 10–20
in the data; our model uses v_I/v_T = 10, consistent.

### 1.3 Rotational diffusion / persistence time (D_R^{T,I})

Persistence time τ_p relates to rotational diffusion as **D_R ≈ 1/(2τ_p)** in 2D.

* **Naive T cells in lymph nodes:** persistence ≈ **2 min** ⇒ D_R ≈ 0.25 min⁻¹
  (Beltman *et al., Phys. Rev. E* 2005; Mrass *et al., PNAS* 2018, PMC4865483, which
  documents the two-state fast/slow model).
* **Tumor-infiltrating T cells:** velocity-state persistence **0.25–0.5 min**, i.e. D_R^I in
  **1–2 min⁻¹** (Mrass *et al.*, *Immunity*-adjacent literature). Significantly more turning
  than in lymph node — the tumor stroma "traps" cells.
* **Tumor cells:** persistence on hour scale (Friedl reviews), D_R^T ≈ 0.01–0.05 min⁻¹.

**Implication for the model.** Our model uses D_R^T = 0.1, D_R^I = 1.0 in model time units;
the **10× ratio is right**, but the absolute scale depends on what we declare "1 model
time unit" to be. If 1 model time = 1 min, our numbers are within a factor of 2 of measurement.

### 1.4 Cell division rate (p_div)

Ki-67 is a protein expressed in cycling cells; the **Ki-67 index** is the % of cells
positive on immunohistochemistry. The "potential doubling time" T_pot (from BrdU labelling,
which marks DNA-synthesizing cells) is the gold standard.

* **Breast cancer median Ki-67 = 20 %**, range 1–99 % across subtypes; luminal A ≈ 17 %,
  triple-negative ≈ 50 %. Inwald *et al., Breast Cancer Res. Treat.* (2013).
* **T_pot across solid tumors:** 3.2 days (oat-cell lung) → 23.2 days (lymphoma); ~38 %
  of tumors have T_pot ≤ 5 days. Rew & Wilson, *Eur. J. Cancer* (2000), and the Begg
  multicenter survey (*Radiother. Oncol.* 1999).
* Converting: doubling rate **k_div = ln 2 / T_pot ≈ 0.05–0.25 day⁻¹**, or
  **3 × 10⁻⁵ to 1.7 × 10⁻⁴ min⁻¹**.

**Plug value for p_div:** if 1 model time unit = 1 min, p_div per step (Δt = 0.01) should be
**3 × 10⁻⁷ to 2 × 10⁻⁶**. Our current value 0.004 is **3–4 orders of magnitude too
high** — this is a known compression knob to make dynamics visible in 100 model-time units;
flagged in `docs/DECISIONS.md`.

### 1.5 T-cell killing rate per contact (p_kill)

The "per capita killing rate" PCKR is the standard observable.

* **Bulk in vitro & in vivo estimates:** **1–20 tumor kills / CTL / day** ("CTL" =
  cytotoxic T lymphocyte, the CD8⁺ killing population). Halle *et al., Immunity* (2016) gives
  **1.24 ± 0.11 kills/CTL/day** in liver, **3.18 ± 0.26 kills/CTL/day** in spleen using
  intravital imaging.
* **Per-contact probability is low.** Weigelin *et al., Nat. Commun.* (2021) and Wiedemann
  *et al., PMC9166723* (microfluidic) show that for solid-tumor targets, a single 1:1
  conjugation event has **< 10 % chance** of inducing apoptosis; lethality requires
  **≥ 3 serial sublethal hits with intervals < 50 min** — "additive cytotoxicity."

**Plug value for p_kill.** With contact lifetimes ~ 10 min and ~ 3 hits needed for
a 50 % kill probability, a per-contact-event p_kill ≈ **0.1–0.3** is in the right band.
Our model's p_kill = 0.12 is **directly defensible**.

### 1.6 Chemokine diffusion and decay (D_a, D_s, λ_a, λ_s)

* **Chemokine diffusion in tissue, single-molecule tracking:**
  - CCL19: **D = 8.4 ± 0.2 μm²/s** in collagen matrix.
  - CXCL13: **D = 6.2 ± 0.3 μm²/s** in collagen; **6.6 ± 0.4 μm²/s** in murine lymph node
    sections. Thelen *et al., bioRxiv* (2017, "Ultra-fast super-resolution imaging") and
    Cosgrove *et al., PNAS* (2020, PMC5972203).
  - These are the **free-solution-like values**; effective diffusion is **5–10× lower** once
    binding to glycosaminoglycans is included, putting **D_effective ≈ 1 μm²/s** for tissue
    transport.
* **Chemokine decay / half-life:**
  - Surface half-life of CCL3, CCL5 ≈ **30 min** on endothelium (Pruenster *et al., Nat.
    Immunol.* 2009, PMID 16033532-context).
  - Plasma half-lives are minutes, but tissue dwell can be much longer because of GAG binding.
  - For modeling: **λ ≈ 10⁻³–10⁻² s⁻¹** (half-life 1–10 min) is a common assumption.
* **Resulting decay length √(D/λ): 10–100 μm** in vivo — about **5–50 cell diameters**.
  This is the range our model uses for ℓ_a ≈ 7σ.

### 1.7 Chemotactic coupling χ

Direct measurement of the Keller–Segel χ for T cells is rare; bacterial χ is the
benchmark literature (Berg & Brown; Ahmed & Stocker, *Nat. Comm.* 2019).

* **Bacterial χ ≈ 10⁻³–10⁻² cm²/s ≈ 10⁵–10⁶ μm²/s** per unit log-gradient.
* **Mammalian neutrophil chemotactic coefficient (under-agarose assay):**
  Lauffenburger framework gives χ_0 ≈ **10–100 × random-motility coefficient**;
  in dimensionless terms, the **gradient-following bias** is typically 0.1–0.3 of v_I in
  experimentally realistic gradients.

**Take-away.** χ is the most parameter-poor input — only order-of-magnitude bounds exist
for tumor-relevant chemokines. **Flag it as a free knob** in our sensitivity analysis.

---

## 2. Validation targets

These are the observable phenotypes the model should reproduce *before* we publish a phase
diagram.

### 2.1 Hot / excluded / cold tumor classification

* **Canonical reference: Tumeh *et al., Nature* 515:568 (2014).** First demonstration that
  **pre-existing CD8⁺ infiltration at the invasive margin predicts response to PD-1
  blockade** (pembrolizumab) in melanoma. Responder vs. non-responder pre-treatment CD8
  density differs by **≈ 3–5× at the invasive margin**, ~2× in the tumor centre.
* **Galon immunoscore framework**: Galon *et al., Science* (2006) and *Immunity* (2013);
  formalized as the **Immunoscore** with 4 levels (I0–I4) based on CD3⁺/CD8⁺ density at
  core and margin, validated across 14 cancer types in *Lancet* (2018) by the SITC consortium.
* **The three-phenotype framework (inflamed / excluded / desert)**: Chen & Mellman, *Nature*
  (2017, "Elements of cancer immunity") is the most cited single reference.

**What our model must show.** A 2D simulation snapshot at low α should show CD8 inside the
tumor mass (**hot**); at high α, CD8 at the margin only (**excluded**); at very high α or
low χ_a, CD8 absent entirely (**cold**).

### 2.2 T-cell ring around excluded tumors

* **Joyce & Fearon, *Science* (2015):** the original conceptual review on T-cell exclusion
  by stromal mechanisms.
* **Salmon *et al.*, *J. Clin. Invest.* (2012):** histology of human lung tumors showing T cells
  accumulating in stroma but not in tumor nests — the classic "T-cell ring" image.
* **Grout *et al.*, *Cancer Discov.* (2022):** CAF (cancer-associated fibroblast) **αSMA⁺ FAP⁺**
  barriers correlate quantitatively with reduced intratumoral CD3⁺ density.

### 2.3 Macrophages and anti-PD-1 resistance

* **Arlauckas *et al., Sci. Transl. Med.* (2017):** intravital imaging shows TAMs
  ("tumor-associated macrophages") **physically strip anti-PD-1 antibodies off CD8⁺ T cells**
  within minutes — a Fcγ-receptor-mediated capture mechanism.
* **Peranzoni *et al., PNAS* (2018):** TAMs **trap T cells in the stroma**, limiting anti-PD-1
  efficacy; macrophage depletion (anti-CSF1R) restores CD8 contact with tumor cells.
* These are the references to cite if we extend the model to include macrophages (Extension #3).

### 2.4 Survival statistics

* **KEYNOTE-006 (Schachter *et al., Lancet* 2017):** pembrolizumab vs. ipilimumab in advanced
  melanoma — 5-year overall survival **38.7 %** (pembro) vs. **31.0 %** (ipi). The
  reference Kaplan-Meier curve for "treated" arms.
* **Pooled untreated metastatic melanoma historical control:** median OS ~ **6–10 months**;
  5-year OS < 10 %. Source: Korn *et al., J. Clin. Oncol.* (2008) meta-analysis.

Our model could in principle generate analogue survival curves by tracking the time when
N_T(t) re-crosses some threshold.

### 2.5 In vitro 3D spheroid co-culture data

* **Pawlowski / Bracci framework** — Charles River and several academic groups publish
  spheroid-killing time courses with E:T (effector-to-target) ratios 1:1 to 10:1.
* **Reference dataset:** Courau *et al., J. Immunother. Cancer* (2019) — colorectal organoid
  + autologous TIL killing curves with timepoints at 24, 48, 72 h. Open data accompany the
  paper.
* **Microfluidic / lab-on-chip:** Boussommier-Calleja *et al., Biomaterials* (2019);
  Wiedemann *et al., PMC9166723*.

These are the most directly fit-able datasets: same observable as our model (number of live
tumor cells vs. time) at known E:T ratios.

---

## 3. Existing benchmark frameworks

* **PhysiCell** (Ghaffarizadeh *et al., PLOS Comp. Biol.* 2018) is the dominant agent-based
  multicellular framework. The **PhysiCell + EMEWS workflow** (Ozik *et al., BMC Bioinform.*
  2018, PMC6349239) implements **approximate Bayesian computation (ABC)** for ABM calibration
  — directly applicable to our problem. No formal challenge dataset, but the PhysiCell
  community has a published library of immune-tumor models that we can cross-compare with.
* **DREAM Anti-PD1 Response Prediction Challenge** (Mason *et al., J. Transl. Med.* 2024,
  PMC10880244) — 59 teams, 417 models, used protected phase-III RCT data from two anti-PD-1
  trials in NSCLC. **Mostly statistical / genomic** rather than mechanistic, but the held-out
  validation cohort is the standard benchmark.
* **IMI / IMI2 challenges** (Innovative Medicines Initiative): the **OPTIMA** and **CANCER-ID**
  projects publish multi-modal datasets (imaging + RNA-seq + outcomes) but access is
  consortium-restricted.
* **Cancer Systems Biology Consortium (CSBC)** of the NCI publishes open ABM benchmark
  pipelines; their tumor-immune working group has periodic model-comparison reports
  (e.g., Norton *et al., Cells* 2019).
* **ABM-specific calibration toolkit:** Lima *et al., J. Theor. Biol.* (2024, PMC10869399) —
  ABC for monophasic / biphasic tumor growth, with reusable code and synthetic benchmark
  data.

There is **no single "MNIST-of-tumor-immune-modeling"**; the field has reusable tools but
not a canonical benchmark.

---

## 4. One-page calibration plan (one-week budget)

**Day 1 — Fix what's well-measured.** Set v_T, v_I, σ, D_R^I to literature midpoints
(§1.1–1.3); freeze them. They span < 1 order of magnitude in the literature, so further
sweeping is wasted compute. Time normalization: define **1 model time unit = 1 min**, so
v_I = 4 → v_I = 4 σ/min in our units. Re-derive all dimensionless groups in §9 of `model.md`.

**Day 2 — Tighten p_kill and the killing model.** Re-implement killing in the
**Halle/Weigelin "serial-sublethal-hits" form**: each contact event deposits damage, kill
occurs at cumulative damage threshold. Calibrate the damage-per-hit so that the bulk PCKR
matches **2 kills/CTL/day** at E:T = 1:1. This is the most-measured single number we have.

**Day 3 — Latin-Hypercube sample the unknowns.** Sweep χ_a, χ_s (≡ α), D_a/D_s, λ_a/λ_s,
p_div jointly in **N = 256 runs** using `joblib`. Use the existing phase-diagram code
(`src/sim.py`) as the engine. Sample p_div across **3 orders of magnitude** —
literature spans 1–25 % per day.

**Day 4 — Sensitivity analysis (Sobol indices).** Compute first-order and total Sobol
indices for the final tumor fraction Φ against all 7 swept parameters. We expect α and
χ_a/χ_s ratio to dominate (this is what the phase diagram already says); confirm and
**flag the others as low-sensitivity**, hence safe to leave at literature midpoints.

**Day 5 — Validation against Courau or Boussommier-Calleja data.** Pick **one** open in vitro
spheroid+TIL killing time course. Fit the residual 2–3 high-sensitivity parameters by
**ABC-rejection** (the Lima 2024 recipe) against the observed N_T(t) decay. Quote the
posterior credible intervals.

**Day 6 — Cross-check against histology.** Generate **simulation snapshots** at the
posterior-mode parameters; classify them by the same Tumeh-style rule (CD8 density at
core vs. margin). Confirm that the (α, ρ_I) axes of the model phase diagram map onto
the (excluded, hot, cold) classification — this is the **internal-consistency validation**.

**Day 7 — Document uncertainties.** Produce a single-page table: each parameter, its
literature range, the value used, its Sobol total index, whether it was inferred or fixed.
This becomes the "calibration honesty" page of the final report. Anything with Sobol > 0.2
and no direct measurement is **explicitly labeled a knob**.

**Outcome.** A calibrated model where ≈ 5 parameters are pinned to data within an order of
magnitude, 2 are inferred against an open dataset, and the remaining 2–3 are honestly
flagged as scan axes.
