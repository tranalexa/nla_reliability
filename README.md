# nla_reliability

This repo evaluates NLA reliability on two datasets: **PRISM** and **Bias in Bios**. For each dataset, we sample 400 items, extract Gemma-3-12B layer-32 activations, generate 12 AV descriptions per activation, reconstruct each description with AR, and evaluate consistency and fidelity using raw and mean-centered cosine diagnostics.

Everything heavy runs on [Modal](https://modal.com). Artifacts live on a persistent volume named **`nla-cache`**, which appears as the folder **`/cache`** inside GPU jobs (not a path on your laptop until you `modal volume get`).

## Project layout

```text
extract_activations.py   # Modal Step 1
sample_descriptions.py   # Modal Step 2
reconstruct_scores.py    # Modal Step 3 (AR pairwise + fidelity)
nla/                     # shared library
  datasets.py            # dataset loading: PRISM, Bias in Bios
  paths.py               # dataset-specific artifact path helpers
  nla_inference.py       # NLAClient / NLACritic (vendored from kitft; see Attribution)
  activation_utils.py    # activation loading utilities
scripts/                 # local tools (pull, preview, diagnostics)
data/                    # downloaded volume artifacts (gitignored)
```

### Attribution (`nla/nla_inference.py`)

Vendored from [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) (`nla_inference.py` on main). Implements the **activation verbalizer (AV)** and **activation reconstructor (AR)** sides of Natural Language Autoencoders (NLAs) from Anthropic / [Transformer Circuits (2026)](https://transformer-circuits.pub/2026/nla/index.html). This reliability repo does not retrain NLAs; it uses the published [Gemma-3 AV checkpoint](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) and [AR checkpoint](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar) on Modal.

---

## Datasets

### PRISM

**Source:** `Transluce/PRISM-gender-Llama-3.1-8B-Instruct`

Each PRISM item is a multi-turn conversation. The pipeline flattens all turns up to the last user turn into a single string (`User: … \n Assistant: … \n User: …`), dropping any trailing assistant turns so the final Gemma token is the last user token. Gemma reads this flattened conversation text directly.

Metadata kept in the CSV: `gender` (model-predicted), `gt_gender` (survey ground truth). These are not used by the core pipeline; they are available for diagnostic grouping.

### Bias in Bios

**Source:** `LabHC/bias_in_bios`

Each item is a biography (`hard_text` field, which already omits the opening name sentence). Before extraction, explicit gender indicators (pronouns, honorifics, gendered nouns) are scrubbed and replaced with `[PERSON]`. Sampling is stratified 50/50 by gender (200 male / 200 female per 400-item run).

Metadata kept in the CSV: `profession`, `gender`. These are not used by the core pipeline.

---

## Pipeline overview

```text
PRISM (400 items)        → Step 1: Gemma forward pass → prism activations
Bias in Bios (400 items) → Step 1: Gemma forward pass → biosbias activations
    ↓
Step 2: AV × 12 stochastic descriptions per activation
    ↓
Step 3: AR reconstruction → pairwise consistency + fidelity scores
    ↓
Local diagnostics: raw + centered cosine analysis
```

Row index `i` is aligned within each dataset: item `i` in the CSV ↔ activation `i` ↔ description rows with `activation_idx == i`.

### Expected outputs

| Dataset | Items | Samples/item | Description rows | Pairwise rows | Fidelity rows |
|---------|-------|-------------|-----------------|---------------|---------------|
| PRISM | 400 | 12 | 4,800 | 26,400 | 4,800 |
| Bias in Bios | 400 | 12 | 4,800 | 26,400 | 4,800 |
| **Total** | **800** | 12 | **9,600** | **52,800** | **9,600** |

Pairwise rows = 400 × C(12, 2) = 400 × 66 = 26,400 per dataset.

---

## Evaluation metrics

Raw cosine similarity between an original activation and its reconstruction is typically near 0.99 — but this is not meaningful evidence of reconstruction quality. The original activation vectors share a dominant background direction (the global mean vector has norm ≈ 0.99), which inflates cosine similarity for *any* pair of vectors. The meaningful question is whether reconstructions carry **activation-specific** directional information beyond that shared background.

Three complementary metrics are reported, each at the raw level and after mean-centering. Let **μ** be the mean of all original activation vectors and let **r̂** = r / ‖r‖ denote a unit-normalised reconstruction.

### 1 · Raw cosine similarity

`cos(original_i, reconstruction_i)` — reported for reference, but should not be interpreted in isolation. High raw cosine is expected regardless of reconstruction quality due to the shared mean direction.

### 2 · Fidelity (specificity)

Does a reconstruction point toward the *correct* original activation more than toward wrong ones?

| Metric | Definition |
|--------|-----------|
| Raw matched fidelity | `cos(original_i, reconstruction_i)` |
| Raw mismatched fidelity | `cos(original_j, reconstruction_i)`,  j ≠ i |
| **Raw fidelity gap** | matched − mismatched |
| Centered matched fidelity | `cos(original_i − μ,  r̂_i − μ)` |
| Centered mismatched fidelity | `cos(original_j − μ,  r̂_i − μ)`,  j ≠ i |
| **Centered fidelity gap** | matched − mismatched |

Fidelity is supported when the **centered gap** is large and positive. Script: `scripts/diagnose_reconstruction_fidelity.py`

### 3 · Consistency (reliability)

Do 12 reconstructions from different descriptions of the *same* activation agree with each other more than reconstructions from *different* activations do?

| Metric | Definition |
|--------|-----------|
| Raw within-item | `cos(recon_{i,a}, recon_{i,b})`,  a ≠ b, same activation |
| Raw between-item baseline | `cos(recon_{i,a}, recon_{j,b})`,  i ≠ j |
| **Raw consistency gap** | within-item − between-item |
| Centered within-item | `cos(r̂_{i,a} − μ,  r̂_{i,b} − μ)` |
| Centered between-item baseline | `cos(r̂_{i,a} − μ,  r̂_{j,b} − μ)`,  i ≠ j |
| **Centered consistency gap** | within-item − between-item |

Consistency is supported when the **centered gap** is large and positive. Script: `scripts/diagnose_reconstruction_consistency.py`

> Raw cosine scores may be high because of shared activation-space structure. The more meaningful results are baseline-adjusted gaps after centering. Fidelity is supported when centered matched similarity is much higher than centered mismatched similarity. Consistency is supported when centered within-item similarity is much higher than centered between-item similarity.

---

## Setup (once per machine)

You need [uv](https://docs.astral.sh/uv/), a Modal account, and Hugging Face access to gated Gemma + AV/AR weights.

```bash
uv sync
uv run modal setup
modal secret create --force huggingface HF_TOKEN=<your_hf_token>
```

Accept licenses: [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt), [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [nla-gemma3-12b-L32-ar](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar).

`LabHC/bias_in_bios` is publicly accessible (no license gate needed).

### Smoke test (no GPU, no Modal)

Verify both dataset loaders work before running any Modal jobs:

```bash
HF_TOKEN=<your_hf_token> uv run python scripts/smoke_test_loaders.py
```

This loads 5 items from each dataset, checks that `item_idx`, `dataset`, and `prompt_text` are present and non-empty, and prints prompt previews. Expected output ends with:

```
  PASS  prism
  PASS  biosbias

All dataset loaders OK.
```

---

## Run on Modal

Scripts accept `--dataset prism|biosbias` for single-dataset runs, or `--all-datasets` to run both in sequence with one command. Each dataset runs through the same stage independently; artifacts remain separate.

> **Note — `--all-datasets` volume conflict:** When running both datasets sequentially with `--all-datasets`, the second dataset's container may fail with `RuntimeError: there are open files preventing the operation` if the first container's HuggingFace xet transfer logs are still open on the volume. This is a transient Modal platform race condition. If it happens, re-run the failed dataset individually (e.g. `--dataset biosbias`).

### Step 1 — Extract activations (~5–15 min per dataset, 1× A100 each)

```bash
# Single dataset
uv run modal run extract_activations.py --dataset prism --n-items 400
uv run modal run extract_activations.py --dataset biosbias --n-items 400

# Both datasets in one command (sequential)
uv run modal run extract_activations.py --all-datasets --n-items 400 --seed 42
```

### Step 2 — AV descriptions (~30–90 min per dataset, 12× A100 each)

```bash
uv run modal run sample_descriptions.py --dataset prism --n-samples 12
uv run modal run sample_descriptions.py --dataset biosbias --n-samples 12

# Both datasets in one command
uv run modal run sample_descriptions.py --all-datasets --n-samples 12
```

Fewer GPUs: add `--n-shards 1`

### Step 3 — AR reconstruction + scores (~15–45 min per dataset, 12× A100 each)

Pass `--save-vectors` to save raw reconstructed vectors; required for the centered-cosine diagnostics.

```bash
uv run modal run reconstruct_scores.py --dataset prism --save-vectors
uv run modal run reconstruct_scores.py --dataset biosbias --save-vectors

# Both datasets in one command
uv run modal run reconstruct_scores.py --all-datasets --save-vectors
```

Fewer GPUs: add `--n-shards 1`

Raw cosine scores (pairwise and fidelity) will be near 0.99. This is expected. See [Evaluation metrics](#evaluation-metrics) for how to interpret them using baseline-adjusted centered gaps.

---

## Pull results locally (`data/`)

```bash
# Single dataset
uv run python scripts/pull_from_modal.py --dataset prism
uv run python scripts/pull_from_modal.py --dataset biosbias

# Both datasets
uv run python scripts/pull_from_modal.py --all-datasets

# Subset of artifact types
uv run python scripts/pull_from_modal.py --dataset prism --only csv,activations,descriptions
```

Artifact types available with `--only`: `csv`, `activations`, `descriptions`, `pairwise`, `fidelity`, `vectors`

If a pull fails: `modal volume ls nla-cache` to inspect the volume.

---

## Diagnostics (local)

After pulling artifacts to `data/`, run any of the following scripts. The diagnostic scripts accept `--activations` and `--recon-vectors` paths; defaults shown below assume the standard artifact names.

```bash
# Full pairwise cosine similarity matrix: raw and mean-centered
uv run python scripts/activation_cosine_matrix.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet

uv run python scripts/activation_cosine_matrix.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet

# Inflation diagnostic: are ~0.99 raw cosines artefacts of the shared mean direction?
uv run python scripts/diagnose_cosine_inflation.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet

uv run python scripts/diagnose_cosine_inflation.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet

# Fidelity: does each reconstruction match its own activation better than other activations?
# Requires recon_vectors_*.parquet (Step 3 --save-vectors)
uv run python scripts/diagnose_reconstruction_fidelity.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet

uv run python scripts/diagnose_reconstruction_fidelity.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_biosbias.parquet

# Consistency: do the 12 reconstructions per activation agree more than cross-activation ones?
# Requires recon_vectors_*.parquet (Step 3 --save-vectors)
uv run python scripts/diagnose_reconstruction_consistency.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet

uv run python scripts/diagnose_reconstruction_consistency.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_biosbias.parquet
```

---

## Report tables and figures (local)

After pulling artifacts to `data/` (Step F), generate summary tables and figures:

```bash
uv run python scripts/make_report_tables.py
```

Outputs:

| Path | What it is |
|------|------------|
| `reports/summary_stats.csv` | Per-dataset mean/std/median/p5/p95 for all fidelity and consistency metrics |
| `reports/results_table.tex` | LaTeX table ready for the paper |
| `figures/fidelity_dist.png` | Violin plots of centered matched and mismatched fidelity by dataset |
| `figures/consistency_dist.png` | Violin plots of centered within-item and between-item consistency by dataset |
| `figures/raw_vs_centered.png` | Bar chart comparing raw vs centered fidelity and consistency gaps |
| `figures/cosine_inflation.png` | Bar chart illustrating cosine inflation — raw vs centered fidelity metrics |

Requires `recon_vectors_*.parquet` (Step 3 `--save-vectors`) for centered metrics. If absent, falls back to pre-computed raw scores from `fidelity_scores_*.parquet` and `pairwise_consistency_*.parquet`.

Custom paths:

```bash
uv run python scripts/make_report_tables.py \
  --data-dir data --out-dir reports --fig-dir figures
```

---

## Volume artifacts (`nla-cache` → `/cache` in jobs)

### PRISM

| Path | What it is |
|------|------------|
| `prism_400.csv` | 400 PRISM items with `item_idx`, `dataset`, `prompt_text`, `gender`, `gt_gender` |
| `activations_layer32_prism_gemma-3-12b-pt.parquet` | PRISM activation vectors (layer 32, last user-turn token, L2-normalized) |
| `descriptions_prism.parquet` | `activation_idx`, `sample_idx`, `description` (4,800 rows) |
| `pairwise_consistency_prism.parquet` | Within-activation recon-recon cosine scores (26,400 rows) |
| `fidelity_scores_prism.parquet` | Per-reconstruction fidelity cosine scores (4,800 rows) |
| `recon_vectors_prism.parquet` | Raw reconstructed vectors (4,800 rows × 3840-d); written only with `--save-vectors` |

### Bias in Bios

| Path | What it is |
|------|------------|
| `biosbias_400.csv` | 400 BiasBios items with `item_idx`, `dataset`, `prompt_text`, `profession`, `gender` |
| `activations_layer32_biosbias_gemma-3-12b-pt.parquet` | BiasBios activation vectors (layer 32, last token, L2-normalized) |
| `descriptions_biosbias.parquet` | `activation_idx`, `sample_idx`, `description` (4,800 rows) |
| `pairwise_consistency_biosbias.parquet` | Within-activation recon-recon cosine scores (26,400 rows) |
| `fidelity_scores_biosbias.parquet` | Per-reconstruction fidelity cosine scores (4,800 rows) |
| `recon_vectors_biosbias.parquet` | Raw reconstructed vectors (4,800 rows × 3840-d); written only with `--save-vectors` |

| `hf/` | Cached model weights (reuse across runs) |

---

## End to end pipeline

```bash
uv sync && uv run modal setup
# Set HF token once:
modal secret create --force huggingface HF_TOKEN=<your_hf_token>

# Run all three steps for both datasets
uv run modal run extract_activations.py --all-datasets --n-items 400 --seed 42
uv run modal run sample_descriptions.py --all-datasets --n-samples 12
uv run modal run reconstruct_scores.py --all-datasets --save-vectors

# Pull all artifacts locally
uv run python scripts/pull_from_modal.py --all-datasets

# Run diagnostics for each dataset
uv run python scripts/diagnose_cosine_inflation.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet
uv run python scripts/diagnose_reconstruction_fidelity.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet
uv run python scripts/diagnose_reconstruction_consistency.py \
  --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet

uv run python scripts/diagnose_cosine_inflation.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet
uv run python scripts/diagnose_reconstruction_fidelity.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_biosbias.parquet
uv run python scripts/diagnose_reconstruction_consistency.py \
  --activations data/activations_layer32_biosbias_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_biosbias.parquet
```

**Check you're done — per dataset:**

| File | Expected |
|------|----------|
| `data/prism_400.csv` | 400 rows |
| `data/activations_layer32_prism_gemma-3-12b-pt.parquet` | 400 rows |
| `data/descriptions_prism.parquet` | 4,800 rows (400 × 12) |
| `data/pairwise_consistency_prism.parquet` | 26,400 rows (400 × 66), no NaNs |
| `data/fidelity_scores_prism.parquet` | 4,800 rows, no NaNs |
| `data/recon_vectors_prism.parquet` | 4,800 rows (if `--save-vectors`) |
| `data/biosbias_400.csv` | 400 rows |
| `data/activations_layer32_biosbias_gemma-3-12b-pt.parquet` | 400 rows |
| `data/descriptions_biosbias.parquet` | 4,800 rows (400 × 12) |
| `data/pairwise_consistency_biosbias.parquet` | 26,400 rows (400 × 66), no NaNs |
| `data/fidelity_scores_biosbias.parquet` | 4,800 rows, no NaNs |
| `data/recon_vectors_biosbias.parquet` | 4,800 rows (if `--save-vectors`) |

Raw cosine scores near 0.99 are expected — see [Evaluation metrics](#evaluation-metrics) for how to interpret them using baseline-adjusted centered gaps.

**Cross-dataset totals:**
- 9,600 description rows (4,800 per dataset)
- 52,800 pairwise consistency rows (26,400 per dataset)
- 9,600 fidelity rows (4,800 per dataset)
