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

---

## Hypoxia, angiogenesis, and extended cell repertoire — literature anchors

Notes assembled for the planned extensions in `docs/extensions.md` (hypoxia/angiogenesis stack, NK, DC, MDSC, and a CAF stub). Same convention as §1: concrete numbers in parentheses with the canonical citation; "needs check" where I am not confident enough to attribute an author.

### Hypoxia

Normoxic interstitium sits at **30–40 mmHg pO₂** (skin, muscle); arterial blood is ≈ 95 mmHg. Solid-tumor cores routinely measure **median pO₂ < 10 mmHg** with substantial fractions of voxels **< 2.5 mmHg** ("severe hypoxia") — the foundational Eppendorf-electrode survey is **Vaupel, Kallinowski & Okunieff, *Cancer Res.* 49:6449 (1989)** and the follow-up review **Vaupel, *Semin. Radiat. Oncol.* 14:198 (2004)**. Cervical and head-and-neck tumors are the classic worst offenders (median ≈ 3–10 mmHg).

**HIF-1α** (hypoxia-inducible factor α) is the master transcriptional sensor. Under normoxia, prolyl-hydroxylase-domain enzymes (PHD2 dominant) hydroxylate two prolines, VHL ubiquitinates, and the protein is destroyed in minutes. Stabilization is graded but effectively saturates **below ≈ 5 % O₂ (≈ 35 mmHg)**, and is strongly amplified **below 1–2 % O₂ (≈ 7–15 mmHg)** — see **Semenza, *Cell* 148:399 (2012)** and **Jaakkola et al., *Science* 292:468 (2001)**. Once stabilized, HIF-1α transactivates **VEGF-A, GLUT1, CA9, LDHA, EPO** and dozens of others; VEGF transcription rises roughly **10–30-fold** within hours of switching to 1 % O₂ (**Forsythe et al., *Mol. Cell. Biol.* 16:4604 (1996)**).

Hypoxia arrests proliferation at the **G1/S checkpoint** under sustained **< 1 % O₂**, via p27, hypophosphorylated Rb, and reduced dNTP supply (**Gardner et al., *J. Biol. Chem.* 276:7919 (2001)**). At the population level, doubling-time prolongation by **2–5×** is typical.

CTL and NK function collapses well before proliferation does. **Noman et al., *Cancer Res.* 71:5976 (2011)** showed HIF-1α-dependent upregulation of PD-L1 on tumor cells under hypoxia, plus reduced CTL lysis. **Vuillefroy de Silly, Dietrich & Walker, *OncoImmunology* 5:e1232236 (2016)** review the field and report **30–70 % reductions** in granzyme/perforin release and IFN-γ production at **1 % O₂** versus 21 %; NK cells are if anything more sensitive (**Balsamo et al., *Eur. J. Immunol.* 43:2756 (2013)**).

**Parameter mapping.** Add a scalar oxygen field $c_{O_2}\in[0,1]$ on the existing grid (1 = arterial, 0 = anoxic). A defensible hypoxia threshold is `O2_hyp_thresh ≈ 0.1` (≈ 10 % of arterial, i.e. ≈ 10 mmHg, near the HIF saturation knee). For the killing penalty, parameterize as $p_{\text{kill}} \to p_{\text{kill}}\,(1 - h\,\Theta(\text{thresh}-c_{O_2}))$ with `hypoxia_penalty` $h \approx 0.5$ — i.e. cut $p_{\text{kill}}$ in half inside hypoxic voxels, consistent with the 30–70 % cytolysis loss above. Proliferation should get the same gate: $p_{\text{div}} \to p_{\text{div}}\,(1 - h_{\text{div}})$ with $h_{\text{div}} \approx 0.7$ in hypoxia.

### Angiogenesis

**VEGF-A** is the dominant pro-angiogenic ligand; **Ferrara, *Endocr. Rev.* 25:581 (2004)** is the canonical review, and **Carmeliet, *Nat. Med.* 9:653 (2003)** is the standard companion. Tumor cells secrete VEGF-A constitutively at low rates and at much higher rates under HIF-1α stabilization (§ above) — so VEGF in our model is naturally sourced by tumor cells in hypoxic voxels.

Sprouting angiogenesis is the canonical mechanism: a quiescent capillary endothelium "wakes up," a **tip cell** specifies (Dll4/Notch lateral inhibition; **Hellström et al., *Nature* 445:776 (2007)**), and that tip cell migrates up the local VEGF gradient at **≈ 10–50 µm/h** in vitro and on the same order in vivo (**Gerhardt et al., *J. Cell Biol.* 161:1163 (2003)**). Tip-cell migration in our active-matter language is exactly a $+\chi\nabla c_{\text{VEGF}}$ term on a third particle species (endothelial sprouts) or, in the simpler continuum version, $\partial_t \rho_V = D_V\nabla^2 \rho_V + \chi_{\text{vessel}}\nabla\!\cdot(\rho_V \nabla c_{\text{VEGF}})$. Translating 10–50 µm/h to our units (1 σ ≈ 15 µm, 1 model time = 1 min) gives a tip-cell drift of **0.01–0.06 σ/min**, comparable to $v_T$. A defensible $\chi_{\text{vessel}}$ produces that drift in the steady-state VEGF gradient of the simulation.

Tumor vessels are notoriously dysfunctional: **leaky, dilated, tortuous, chaotically branched, with poor pericyte coverage** — see **Folkman, *N. Engl. J. Med.* 285:1182 (1971)** for the original conceptual paper and **Carmeliet & Jain, *Nature* 407:249 (2000)** for the structural review. The clinical consequence is heterogeneous perfusion: vessel density can be high while *functional* perfusion is low. This is exactly what permits the persistent hypoxia of the previous section despite a vessel-rich stroma.

**Jain, *Science* 307:58 (2005)** introduced the **vessel-normalization** paradigm: low-dose anti-VEGF (bevacizumab, etc.) prunes immature sprouts and transiently restores a more regular vasculature with better O₂ delivery and drug penetration — a 1–2 week window. In our model, this maps to turning OFF the sprouting source term and `chi_vessel` partway through a simulation: existing chaotic vessels regress (decay term on $\rho_V$), oxygen normalizes, and the hypoxia-dependent suppression terms should relax. That is the cleanest qualitative target for an anti-angiogenic treatment experiment.

### NK cells

**Natural killer cells** are innate lymphocytes that kill **without prior antigen sensitization and without MHC-restricted TCR recognition**. The dominant detection logic is **"missing self"** (**Kärre, *Scand. J. Immunol.* 55:221 (2002)**; **Vivier et al., *Science* 331:44 (2011)**): inhibitory receptors (KIR in humans, Ly49 in mice) read self-MHC-I; loss of MHC-I — common in tumors escaping CTL pressure — releases the inhibitory brake, while activating receptors (NKG2D reading stress ligands MICA/B, ULBP1–6) provide the "go" signal.

Chemotactic repertoire is partially overlapping with CD8 but distinct enough to model separately. NK cells respond strongly to **CX3CL1 (fractalkine)**, **CXCL10 / CXCR3**, **CCL5 / CCR5**, **S1P / S1PR5**; they are **less responsive to CCL19/CCL21** (lymph-node homing), which is a hallmark of CD8 trafficking. See **Bernardini, Antonangeli, Bonanni & Santoni, *Front. Immunol.* 7:402 (2016)** for a focused review.

The model architectural reason to give NK their own equation rather than reusing the CD8 one is the **absence of a checkpoint-style $-\alpha\nabla c_s$ term**. NK cells are not the primary targets of PD-1/PD-L1 in the way exhausted CD8s are — they express PD-1 at lower levels and the dominant suppressive axis on NK cells in tumors is **TGF-β (canonical: needs check for a quantitative NK-specific number)** and adenosine (A2AR). Coarse-graining: in v1 we omit the suppressant coupling for NK and let them roam more freely, then optionally add a separate TGF-β–driven slowdown later.

Per-encounter killing efficacy is lower than CD8: **5–15 %** per conjugate is the typical microfluidic number, with **2–3× higher motility** than CD8 in tissue (**Deguine, Breart, Lemaître & Bousso, *Immunity* 33:632 (2010)** in lymph nodes; numbers in solid tumors are scarcer — needs check). Implementation: copy the T-cell scaffolding, set $v_{NK} \approx 2 v_I$, $p_{\text{kill}}^{NK} \approx 0.05$, drop the $-\alpha\nabla c_s$ term, optionally bias chemotaxis toward a new $c_{\text{CXCL10}}$ field.

### Dendritic cells

**Conventional DCs (cDC1 in particular)** sample tumor antigens, mature, and migrate via **CCR7 / CCL19,21** to the **tumor-draining lymph node**, where they prime naive CD8 T cells. **Gardner & Ruffell, *Trends Immunol.* 37:855 (2016)** is the standard review of intratumoral DC biology; **Roberts et al., *Cancer Cell* 30:324 (2016)** establishes the dominance of cDC1 (Batf3-dependent) for cross-presentation in tumors.

Our 2D periodic-box model has **no lymph-node compartment**. The defensible coarse-graining is to skip the round trip entirely and let DCs act locally as **antigen-presentation hotspots**: each mature DC deposits a short-range field $c_{DC}(\mathbf{x},t)$ via the same RD machinery as $c_a, c_s$, and CD8 effective $p_{\text{kill}}$ is multiplicatively boosted in regions where $c_{DC}$ is high. This collapses two real biological steps (DC → LN → primed CD8 → return to tumor) into one local "the CD8 here has seen the antigen recently" buff. It is correct in steady state when the lymph-node transit is fast compared to the simulation horizon (real DC-to-LN transit ≈ 12–24 h; our $T_f = 100$ min, so technically the steady-state assumption is **a stretch** — flag this).

Functional dichotomy: **mature, immunostimulatory DCs** vs. **tolerogenic / "regulatory" DCs** induced by tumor-derived IL-10, TGF-β, VEGF (**Gardner & Ruffell 2016**; **Veglia & Gabrilovich, *Curr. Opin. Immunol.* 45:43 (2017)**). One could model a DC polarization scalar analogous to the macrophage $p_k$ in `extensions.md` Part B; for v1 we assume all DCs are mature.

Suggested numbers: $v_{DC} \approx 0.3\,v_I$ (slower, intermediate between TAM and CD8), short-range $D_{DC} \approx D_s$, secretion rate tuned so peak $c_{DC}$ at $\sigma$ from a DC matches the magnitude of $c_a$ from a tumor cell. The CD8 $p_{\text{kill}}$ multiplier could be $(1 + \beta_{DC}\,c_{DC})$ with $\beta_{DC}$ scanned in $[0, 5]$.

### MDSCs

**Myeloid-derived suppressor cells** are a heterogeneous population of immature myeloid cells (granulocytic / PMN-MDSC and monocytic M-MDSC subsets in humans, Ly6G⁺/Ly6C⁺ respectively in mice) **expanded under chronic tumor-derived inflammatory signals** — GM-CSF, G-CSF, IL-6, S100A8/9 (**Gabrilovich & Nagaraj, *Nat. Rev. Immunol.* 9:162 (2009)**; **Veglia, Sanseviero & Gabrilovich, *Nat. Immunol.* 22:108 (2021)**).

Their suppression mechanisms are biochemically diverse but converge on T-cell dysfunction: **Arginase-1** (depletes L-arginine, downregulates CD3ζ), **iNOS** (NO and peroxynitrite that nitrate the TCR), **ROS**, **TGF-β**, **IL-10**, **adenosine via CD39/CD73**, and direct contact-dependent Treg induction. Coarse-graining all of this into "MDSCs add to the suppressant field $c_s$" is justifiable for v1: every one of those mechanisms reduces CTL effector function at short range from the MDSC, which is exactly the operational definition of $c_s$ in our model.

Clinical correlation: **higher circulating and intratumoral MDSC counts predict worse OS and resistance to checkpoint blockade** across melanoma, RCC, NSCLC, GBM (**Diaz-Montero et al., *Cancer Immunol. Immunother.* 58:49 (2009)**; **Weide et al., *Clin. Cancer Res.* 20:1601 (2014)**).

Abundance: highly tumor- and patient-dependent. In aggressive solid tumors MDSCs can reach **5–20 % of CD45⁺ infiltrate**, **comparable to or slightly less than TAMs** (canonical numbers vary; needs check for a single representative figure). Implementation: identical scaffolding to the macrophage extension, but only the M2-equivalent $c_s$ source term — no phagocytosis, no polarization switch. $v_{MDSC} \approx v_M \approx 0.2\,v_I$.

### CAFs (placeholder — not implemented)

**Cancer-associated fibroblasts** are a stromal cell type that deposits and remodels ECM, secretes growth factors (TGF-β, HGF, FGF, CXCL12), and is the dominant non-immune driver of T-cell exclusion in many carcinomas (**Kalluri, *Nat. Rev. Cancer* 16:582 (2016)**; **Mariathasan et al., *Nature* 554:544 (2018)** for the TGF-β / CAF / exclusion axis; **Feig et al., *PNAS* 110:20212 (2013)** for FAP⁺ CAFs and CXCL12 sequestration of CD8).

Future-extension sketch (do not implement in the hackathon scope): a **stationary or slowly-moving** CAF species concentrated near the tumor margin, with a local source term on a new ECM field $\rho_E(\mathbf{x},t)$ — i.e. $\partial_t \rho_E = s_E \rho_{CAF} - k_{\text{deg}} m \rho_E$, where $\rho_E$ then acts as the drag / pore-size gate of `extensions.md` §3. This cleanly composes with the MMP field of §6 and is the natural backbone for a quantitative T-cell-exclusion phenotype.

---

## References

Canonical anchors for the section above. Author–year–journal only; full DOIs in `docs/extensions.md` where they overlap.

* Vaupel P, Kallinowski F, Okunieff P. *Cancer Res.* 49:6449 (1989) — Eppendorf pO₂ survey.
* Vaupel P. *Semin. Radiat. Oncol.* 14:198 (2004) — hypoxia in tumors review.
* Semenza GL. *Cell* 148:399 (2012) — HIF biology master review.
* Jaakkola P et al. *Science* 292:468 (2001) — PHD/VHL O₂-sensing mechanism.
* Forsythe JA et al. *Mol. Cell. Biol.* 16:4604 (1996) — HIF-1 transactivates VEGF.
* Gardner LB et al. *J. Biol. Chem.* 276:7919 (2001) — hypoxic G1/S arrest.
* Noman MZ et al. *Cancer Res.* 71:5976 (2011) — hypoxia, HIF, PD-L1, CTL suppression.
* Vuillefroy de Silly R, Dietrich P-Y, Walker PR. *OncoImmunology* 5:e1232236 (2016) — hypoxia and CTL/NK dysfunction.
* Balsamo M et al. *Eur. J. Immunol.* 43:2756 (2013) — hypoxia and NK function.
* Ferrara N. *Endocr. Rev.* 25:581 (2004) — VEGF biology.
* Carmeliet P. *Nat. Med.* 9:653 (2003) — angiogenesis review.
* Folkman J. *N. Engl. J. Med.* 285:1182 (1971) — tumor angiogenesis original.
* Carmeliet P, Jain RK. *Nature* 407:249 (2000) — vascular abnormalities.
* Jain RK. *Science* 307:58 (2005) — vessel normalization.
* Hellström M et al. *Nature* 445:776 (2007) — Dll4/Notch tip-cell selection.
* Gerhardt H et al. *J. Cell Biol.* 161:1163 (2003) — tip-cell VEGF gradient migration.
* Kärre K. *Scand. J. Immunol.* 55:221 (2002) — missing-self hypothesis.
* Vivier E et al. *Science* 331:44 (2011) — NK cell innate immunity review.
* Bernardini G, Antonangeli F, Bonanni V, Santoni A. *Front. Immunol.* 7:402 (2016) — NK chemokine receptors.
* Deguine J, Breart B, Lemaître F, Bousso P. *Immunity* 33:632 (2010) — NK killing imaging.
* Gardner A, Ruffell B. *Trends Immunol.* 37:855 (2016) — tumor DC biology.
* Roberts EW et al. *Cancer Cell* 30:324 (2016) — cDC1 cross-presentation in tumors.
* Veglia F, Gabrilovich DI. *Curr. Opin. Immunol.* 45:43 (2017) — DC dysfunction in cancer.
* Gabrilovich DI, Nagaraj S. *Nat. Rev. Immunol.* 9:162 (2009) — MDSC review.
* Veglia F, Sanseviero E, Gabrilovich DI. *Nat. Immunol.* 22:108 (2021) — MDSC update.
* Diaz-Montero CM et al. *Cancer Immunol. Immunother.* 58:49 (2009) — MDSCs and prognosis.
* Weide B et al. *Clin. Cancer Res.* 20:1601 (2014) — MDSCs and melanoma OS.
* Kalluri R. *Nat. Rev. Cancer* 16:582 (2016) — CAF biology.
* Mariathasan S et al. *Nature* 554:544 (2018) — TGF-β / CAF / immune exclusion.
* Feig C et al. *PNAS* 110:20212 (2013) — FAP⁺ CAFs and CXCL12.
