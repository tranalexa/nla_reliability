# Analysis findings: NLA reliability as measurement science

This document explains **what we measured**, **what the numbers mean**, and **how the analyses fit together**. It is written for readers who did not run the pipeline themselves. Numeric summaries come from local runs on artifacts in `data/data/` (PRISM, Bias in Bios, MMLU; 400 items × 12 AV samples each).

For pipeline setup and Modal commands, see the [main README](../README.md).

---

## 1. The core question

**Natural Language Autoencoders (NLA)** map an activation vector to text (activation verbalizer, AV) and back to a vector (activation reconstructor, AR). We ask:

> If we fix one Gemma layer-32 activation and sample the verbalization **stochastically** (temperature 1.0), how **dependable** is the round-trip—and does it **preserve** what the activation encoded?

That is a **measurement** question, not only an engineering one:

| Concept | Question in this repo | Example metric |
|--------|------------------------|----------------|
| **Reliability** | Do repeated AV→AR samples **agree**? | Within-activation recon cosine; G-theory on fidelity; MPNet text consistency |
| **Validity** | Does the result still **encode the label** we care about? | Linear probes on original vs mean-reconstructed vectors |
| **Bias / method effects** | Is the metric dominated by artifact? | Cosine inflation; raw vs mean-centered gaps |

**Reliability** and **validity** can diverge: reconstructions can be highly consistent with each other while losing attribute-specific information (especially on MMLU).

---

## 2. Experimental design

```text
For each dataset:
  400 items  →  Step 1: one activation vector each (Gemma-3-12B, layer 32, L2-normalized)
            →  Step 2: K = 12 stochastic AV descriptions per activation
            →  Step 3: 12 AR reconstructions + cosine scores per activation

Row alignment:  item_idx / activation_idx i  ↔  CSV row i  ↔  12 description rows
```

| Dataset | Content | Labels used in probes / grouping |
|---------|---------|----------------------------------|
| **PRISM** | Multi-turn conversations (flattened to last user token) | `gender` (model-predicted) |
| **Bias in Bios** | Gender-scrubbed biographies | `profession` (26 classes), `gender` (M/F) |
| **MMLU** | 4 subjects × 100 MCQ items | `subject` (4 balanced subjects) |

**Facets in measurement terms:**

- **Objects of measurement (p):** the 400 activations (one per prompt).
- **Occasions / instrumentation (i):** the 12 AV samples (stochastic verbalizations).
- **Universe of interpretation:** “this activation, as operationalized through AV→AR.”

We use **public** NLA checkpoints (`kitft/nla-gemma3-12b-L32-{av,ar}`); this repo does not train them.

---

## 3. AV outputs (what Step 2 looks like)

AV does **not** paraphrase the input prompt. It only sees the **injected activation**. Typical descriptions are meta-commentary (“blog format established…”, “expects elaboration like…”) with invented quasi-quotes. That is why:

- **Text-space** similarity across 12 samples is moderate (~0.82–0.87 with MPNet), not ~1.0.
- **Reconstruction-space** similarity is much higher (~0.999 raw pairwise), because AR maps diverse wordings to nearly the same direction.

Preview examples: `scripts/preview_descriptions.py` → `scripts/description_preview.txt`.

---

## 4. Cosine similarity: raw vs mean-centered

### 4.1 The inflation problem

Step 1 vectors are **L2-normalized**. They still share a **dominant mean direction** (‖mean of all activations‖ ≈ **0.99** on all three datasets). Consequence:

- **Any two** activations have raw cosine ≈ **0.98–0.99** even when they are different items.
- **Matched fidelity** (original vs its own recon) is also ≈ **0.99**, barely above a **random other** activation’s cosine.

So **raw cosine near 1.0 is not strong evidence** of good reconstruction; it often reflects shared background geometry.

### 4.2 What mean-centering shows

Subtract the global mean activation **μ** from every vector, re-normalize, then compute cosine. **Gaps** (matched − mismatched, within − between) separate activation-specific structure from the shared direction.

**Summary from `reports/summary_stats.csv` (centered gaps are the headline):**

| Dataset | Centered fidelity gap (matched − mismatched) | Centered consistency gap (within − between) |
|---------|-----------------------------------------------|-----------------------------------------------|
| PRISM | ~0.67 | ~0.81 |
| Bias in Bios | ~0.65 | ~0.79 |
| MMLU | large & positive (re-run `make_report_tables.py --data-dir data/data` for exact values) | ~0.27 (much smaller than PRISM/BiasBios) |

**Interpretation:**

- **Large centered fidelity gap:** each reconstruction aligns to **its** activation more than to others—specificity is real after removing μ.
- **Large centered consistency gap (PRISM, BiasBios):** 12 recons for the **same** activation agree with each other much more than recons from **different** activations—**reliability in vector space** is strong.
- **MMLU** has a **smaller** centered consistency gap (~0.27 vs ~0.81): cross-item recons are still relatively similar after centering; subject-level structure is less cleanly separated in recon space (consistent with probe results below).

Scripts: `diagnose_cosine_inflation.py`, `diagnose_reconstruction_consistency.py`, `make_report_tables.py`.

---

## 5. Text-space consistency (sentence embeddings)

**Question:** Do the 12 AV **wordings** agree semantically, even when recon vectors agree more?

**Method:** Embed each description with `sentence-transformers/all-mpnet-base-v2` (one batched pass over 4,800 strings). For each activation, compute all C(12,2) pairwise cosines and **average** → one score per activation (`scripts/compute_text_consistency.py`).

**Typical per-activation means (400 activations each):**

| Dataset | Mean pairwise text cosine (MPNet) | Std (across activations) |
|---------|----------------------------------|---------------------------|
| PRISM | ~0.86 | ~0.03 |
| Bias in Bios | ~0.87 | ~0.03 |
| MMLU | ~0.83 | ~0.04 |

**Interpretation:**

- Stochastic AV samples are **moderately** consistent in language space—much lower than recon cosine (~0.999).
- **Compression:** recon space is more “collapsed” than text space; AR agrees more than the verbalizations differ semantically.
- Text and recon means correlate only **weakly** per activation (Pearson r roughly 0.4 on PRISM, near 0 on BiasBios/MMLU)—high recon reliability does **not** imply high text paraphrase similarity.

Outputs: `data/text_consistency_mpnet_{dataset}.parquet` (columns: `activation_idx`, `mean_pairwise_text_cosine`).

---

## 6. Vector-space reliability (Step 3 parquets)

From `pairwise_consistency_*.parquet` and `fidelity_scores_*.parquet`:

| Metric | Typical raw value | Role |
|--------|-------------------|------|
| Within-activation recon–recon cosine | ~0.999 | Reliability across 12 verbalizations |
| Matched fidelity (orig vs recon) | ~0.994 | Headline “return to activation” (inflated raw) |
| Between-activation baselines | ~0.98 raw | Shows inflation; ~0.15 centered (PRISM) |

These support: **the NLA round-trip is stable** for a fixed activation, in direction space, after accounting for μ.

---

## 7. Linear probes: validity (original vs reconstructed)

### 7.1 What we did

Train a **logistic regression** on activation vectors to predict dataset metadata labels. Compare:

- **Original:** Step 1 `activation_vector` (400 × 3840).
- **Mean recon:** mean of 12 `recon_vector`s per `activation_idx`, then L2-normalized (`--vector-source compare` in `scripts/train_linear_probes_multi_dataset.py`).

Same train/test split (20% holdout, seed 42). This tests **linear decodability** of attributes—not full semantic equality.

### 7.2 Results (`reports/linear_probe_compare.csv`)

| Dataset | Target | Majority baseline | Probe (original) | Probe (mean recon) | Δ (recon − orig) |
|---------|--------|---------------------|------------------|--------------------|------------------|
| PRISM | gender | 0.575 | 0.575 | 0.575 | 0.000 |
| Bias in Bios | profession | 0.263 | 0.300 | 0.263 | −0.037 |
| Bias in Bios | gender | 0.500 | 0.550 | 0.588 | +0.037 |
| MMLU | subject | 0.250 | **0.963** | **0.763** | **−0.200** |

### 7.3 Interpretation

**Original probes**

- **MMLU subject ~0.96:** layer-32 activations strongly encode which of the four subjects the item came from (easy 4-way problem, 100 per class).
- **BiasBios profession ~0.30:** 26 classes, many rare—probe is weak but above majority (~0.26).
- **PRISM gender ~0.58:** modest; class imbalance (`non_binary` n=1) limits stable probing.

**Recon vs original**

- **PRISM / BiasBios (net):** round-trip **preserves** coarse linear structure about gender/profession about as well as originals on these metrics.
- **MMLU:** large drop (**0.96 → 0.76**). Recon still well above chance (4-way, majority 0.25) but **loses subject-specific decodability**. This is the clearest **validity** warning: vector reliability (high cosine) does not mean subject information is fully preserved.

**Takeaway:** For “does AV→AR keep what the activation encoded?”, MMLU is the stress test; PRISM/BiasBios look fine on linear probes **and** on fidelity-based G-theory.

---

## 8. Generalizability theory (G-study and D-study)

### 8.1 Design

We treat:

- **p** = 400 activations (random),
- **i** = 12 AV samples (random),
- **Score** = `fidelity_cos` per (activation, sample)—one value per cell.

This is a **p × i** crossed design with one observation per cell. ANOVA yields variance components:

| Component | Meaning |
|-----------|---------|
| **σ²_p** | Differences **between activations** (signal for comparing items) |
| **σ²_i** | Differences **between sample occasions** (global AV bias across all items) |
| **σ²_pi** | Activation × sample interaction (**instability** of verbalization for a given item) |

Script: `scripts/g_theory_study.py` → `nla/g_theory.py`.

### 8.2 G-study results (`reports/g_theory_variance_components.csv`)

| Dataset | σ²_p (% of comp.) | σ²_pi (% of comp.) | Cronbach's α (12 samples as “items”) |
|---------|-------------------|---------------------|--------------------------------------|
| PRISM | 90.6% | 9.4% | 0.991 |
| Bias in Bios | 98.6% | 1.4% | 0.999 |
| MMLU | 75.5% | 24.5% | 0.974 |

**σ²_i** estimated as ~0 (clamped): little global shift across sample index that isn’t already in σ²_pi.

**Reading:**

- Most variance is **between activations**, as desired.
- **MMLU** has the largest **sample instability** (σ²_pi)—stochastic AV matters more there.
- High **α** matches “12 verbalizations measure the same activation reasonably well” in fidelity space.

### 8.3 D-study: what are n′ = 1 and n′ = 12?

**n′** is not a different benchmark. It is a **projection**:

> “Suppose each activation’s score were the mean of **n′** (out of 12) AV→AR samples—how dependable would that score be?”

| n′ | Meaning | Your pipeline |
|----|---------|----------------|
| **1** | Single stochastic verbalization | Hypothetical cheaper design |
| **12** | Mean over all 12 samples | **What you actually ran** |

**G_rel(n′)** = dependability for **ranking or comparing activations** (relative decision).  
**Φ_abs(n′)** ≈ 1.0 here because fidelity scores sit in a tiny band (~0.99); focus on **G_rel**.

### 8.4 D-study table (`reports/g_theory_d_study.csv`)

| Dataset | G_rel (n′=1) | G_rel (n′=6) | G_rel (n′=12) |
|---------|--------------|--------------|---------------|
| PRISM | 0.906 | 0.983 | **0.991** |
| Bias in Bios | 0.986 | 0.998 | **0.999** |
| MMLU | 0.755 | 0.949 | **0.974** |

**How to use this:**

- You do **not** “always need 1” sample—you need **at least 1**.
- You **chose 12** to reduce sample noise; D-study quantifies the gain.
- **MMLU:** one sample is relatively unreliable for comparing activations (G ≈ 0.76); **6–12** samples bring G into the mid–high 0.9s.
- **PRISM / BiasBios:** even **one** fidelity sample is already fairly dependable for **relative** decisions (G > 0.9), but probes/text still flag other issues on MMLU.

**Decision rule example:** If you require G_rel ≥ 0.90, MMLU needs roughly **n′ ≥ 3–4** samples (by interpolation from the table), whereas PRISM might tolerate n′=1 for fidelity-only claims.

---

## 9. Synthesis: one story across analyses

```text
                    RELIABILITY                         VALIDITY
                    (agreement / noise)                 (right label?)
                           │                                  │
   Text (MPNet)     moderate ~0.85                    (not probed directly)
                           │
   Recon cosine      very high ~0.999 raw              probes: MMLU −0.20
   + centered gaps   strong within > between          PRISM/BiasBios ~0
                           │
   G-theory          σ²_pi small except MMLU          fidelity-focused
   on fidelity       D-study: n′=12 helps MMLU most
```

**Coherent claims you can make:**

1. **Shared mean direction** inflates raw cosines; always report **centered gaps** (or probes) for discrimination.
2. **NLA is reliable** for repeated verbalization of the **same** activation (high within-item recon agreement; high G at n′=12).
3. **NLA is not uniformly valid** for all attributes: **MMLU subject** degrades after recon despite high cosine; **PRISM gender** and **BiasBios** coarse attributes are more stable.
4. **Sample count (K=12)** is most justified where **σ²_pi** is large (MMLU); D-study makes that quantitative.
5. **Text and vector reliability diverge**—wording varies more than recon direction; do not infer semantic agreement from recon cosine alone.

---

## 10. Per-dataset cheat sheet

### PRISM

- Centered consistency gap ~0.81; fidelity specificity strong.
- G_rel: 0.91 (1 sample) → 0.99 (12 samples).
- Gender probe: no loss orig → recon.
- Text MPNet ~0.86.

### Bias in Bios

- Similar to PRISM on cosines and G-theory (σ²_pi ~1.4%).
- Profession probe noisy; gender probe slight gain on recon.
- Text MPNet ~0.87.

### MMLU

- Smaller centered consistency gap (~0.27)—recons from different items less separated.
- Largest σ²_pi (~25%); G_rel 0.76 → 0.97 (n′ 1 → 12).
- Subject probe **0.96 → 0.76**—main validity concern.
- Text MPNet ~0.83.

---

## 11. How to reproduce

Artifacts under `data/data/` (see `pull_from_modal.py`). Then:

```bash
# Cosine tables + figures
uv run python scripts/make_report_tables.py --data-dir data/data

# G-study + D-study
uv run python scripts/g_theory_study.py --dataset all --data-dir data/data

# Text embedding consistency (MPNet)
uv run python scripts/compute_text_consistency.py --dataset prism \
  --descriptions data/data/descriptions_prism.parquet \
  --output data/text_consistency_mpnet_prism.parquet

# Linear probes: original vs mean recon
uv run python scripts/train_linear_probes_multi_dataset.py \
  --vector-source compare --dataset all --data-dir data/data \
  --output reports/linear_probe_compare.csv

# Inflation + consistency deep dives
uv run python scripts/diagnose_cosine_inflation.py \
  --activations data/data/activations_layer32_prism_gemma-3-12b-pt.parquet \
  --pairwise data/data/pairwise_consistency_prism.parquet \
  --fidelity data/data/fidelity_scores_prism.parquet
```

Key outputs:

| Path | Content |
|------|---------|
| `reports/summary_stats.csv` | Fidelity & consistency distributions |
| `reports/g_theory_variance_components.csv` | σ²_p, σ²_i, σ²_pi, α |
| `reports/g_theory_d_study.csv` | G_rel, Φ_abs vs n′ |
| `reports/linear_probe_compare.csv` | Orig vs recon probe accuracy |
| `figures/*.png` | Distributions and inflation plots |

---

## 12. Limitations and extensions

- **Linear probes** only test linearly decodable structure; nonlinear preservation is possible.
- **Mean recon** aggregation ignores which of the 12 samples might be outliers; median or best-of-K could differ.
- **G-study on fidelity_cos** is one scalar outcome; σ²_pi would change for text-based or probe-based scores.
- **Class imbalance** (PRISM `non_binary`, rare professions) makes some probe accuracies unstable.
- **Φ_abs ≈ 1** on fidelity is uninformative because scores cluster near 0.99; prefer G_rel and centered gaps.
- **IRT** is not used: items are not a calibrated bank measuring a single latent θ; G-theory matches the nested sampling design better.
- **Classical test theory:** Cronbach's α is reported alongside G; α at n′=12 matches G_rel(12) by design.

---

## 13. Suggested wording for a methods / results section

> We evaluated NLA as a measurement procedure over fixed layer-32 activations, with stochastic activation verbalization (K=12, temperature 1.0) as a source of occasion variance. Reliability was assessed via within-activation reconstruction consistency (mean-centered cosine gaps), generalizability theory on per-sample fidelity (G-study/D-study), and semantic consistency of descriptions (MPNet embeddings). Validity was assessed via linear probes predicting dataset attributes from original versus mean-reconstructed activations. Raw cosines were near unity across conditions; mean-centering and probe transfer revealed that reconstruction reliability was high in vector space but attribute preservation depended on benchmark, with the largest degradation on MMLU subject classification.

---

*Generated to accompany the `nla_reliability` analysis scripts. Update numbers by re-running the commands in §11 if artifacts change.*
