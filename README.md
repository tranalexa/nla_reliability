# nla_reliability

SelfDescribe prompts describe a fictional person. We run **Gemma-3-12B** once per prompt and save the **layer-32 last-token** hidden state (a 3840-d vector). A separate **activation verbalizer (AV)** never sees the prompt text—it only gets that vector and writes an explanation. We generate **12 stochastic** explanations per vector, then analyze consistency.

Everything heavy runs on [Modal](https://modal.com). Artifacts live on a persistent volume named **`nla-cache`**, which appears as the folder **`/cache`** inside GPU jobs (not a path on your laptop until you `modal volume get`).

## Project layout

```text
extract_activations.py   # Modal Step 1 (run from repo root)
sample_descriptions.py   # Modal Step 2
reconstruct_scores.py    # Modal Step 3 (AR pairwise + fidelity)
nla/                     # shared library
  prompt_modes.py        # persona vs full extraction
  paths.py               # data/ paths + Modal volume names
  activation_utils.py    # CSV + parquet loading, optional live Gemma
  nla_inference.py       # NLAClient / SGLang AV (vendored from kitft; see Attribution)
scripts/                 # local tools (pull, preview, probes)
data/                    # downloaded volume artifacts (gitignored)
```

### Attribution (`nla/nla_inference.py`)

Vendored from [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) (`nla_inference.py` on main). Implements the **activation verbalizer** side of Natural Language Autoencoders (NLAs) from Anthropic / [Transformer Circuits (2026)](https://transformer-circuits.pub/2026/nla/index.html). This reliability repo does not retrain NLAs; it uses the published [Gemma-3 AV checkpoint](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) on Modal.

---

## Pipeline overview

```text
SelfDescribe CSV (400 rows)
    → Step 1: Gemma forward → activations parquet
    → Step 2: AV (SGLang) × 12 samples → descriptions parquet
    → Step 3: AR (NLACritic) → pairwise_consistency + fidelity_scores parquets
    → Pull to data/ → preview, linear probes (optional)
```

Row index `i` is aligned everywhere: same persona in CSV, activation `i`, and description rows with `activation_idx == i`.

---

## Step 1 — Extract activations

**Concept:** An “activation” here is Gemma’s internal representation **right after reading** the prompt (last non-padding token, layer 32), L2-normalized. That snapshot is what we later feed to the AV.

**Why persona-only (default)?** Every SelfDescribe prompt ends with the same line asking for a Wikipedia infobox. If we include that in the forward pass, the last token encodes “write an infobox,” not the persona—and AV outputs get generic/meta. Default mode strips that suffix and runs Gemma on the persona sentence only.

| | |
|--|--|
| Script | `extract_activations.py` |
| Model | `google/gemma-3-12b-pt` |
| **Default** | `persona-only` |
| Full prompt | `--full` |

| Mode | What Gemma reads | Output on volume |
|------|------------------|------------------|
| `persona-only` **(default)** | Persona text only | `activations_layer32_persona-only_gemma-3-12b-pt.parquet` |
| `full` | Entire `user_prompt` | `activations_layer32_full_…parquet` + legacy `activations_layer32.parquet` |

Exact suffix removed in persona mode:

```text
Write a hypothetical but realistic Wikipedia biography infobox for me.
```

---

## Step 2 — Sample descriptions

**Concept:** The AV is a fine-tuned Gemma that takes your vector (injected at a special token in a fixed investigator template) and autoregresses an `<explanation>…</explanation>` block. Temperature 1.0 → different wording each time. **Reliability** = how stable those wordings are for the **same** vector.

**Default:** Uses persona-only activations from Step 1. Pass `--full` only if Step 1 was run with `--full`.

| | |
|--|--|
| Script | `sample_descriptions.py` |
| AV model | `kitft/nla-gemma3-12b-L32-av` |
| Output | `descriptions.parquet` (400 activations × 12 samples = 4800 rows) |

Expect AV text in a structured “investigator” style (often mentions tone, a paraphrased phrase, and what the model might say next)—not a clean copy of the CSV sentence.

---

## Step 3 — AR reconstruction + scores

**Concept:** The AR (activation reconstructor) maps each AV description back to a 3840-d vector. **Pairwise consistency** = cosine similarity between reconstructions from different samples of the same activation (66 pairs × 400 activations = 26,400 rows). **Fidelity** = cosine similarity between each reconstruction and the original activation (4,800 rows).

Pass `--save-vectors` to also save the raw reconstructed vectors. The raw vectors are required by the fidelity and consistency diagnostics; without them only cosine scores are stored.

**Default:** Uses persona-only activations from Step 1 (same file Step 2 read).

| | |
|--|--|
| Script | `reconstruct_scores.py` |
| AR model | `kitft/nla-gemma3-12b-L32-ar` |
| Inputs | persona activations parquet + `descriptions.parquet` |
| Outputs | `pairwise_consistency.parquet` (26,400 rows), `fidelity_scores.parquet` (4,800 rows) |
| `--save-vectors` | additionally writes `recon_vectors.parquet` (4,800 rows, one 3840-d vector per row) |

Raw cosine scores (pairwise and fidelity) will be near 0.99. This is expected and largely reflects a shared background direction present in all activation vectors, not reconstruction quality alone. See [Evaluation metrics](#evaluation-metrics) below for how to interpret these numbers.

---

## Evaluation metrics

Raw cosine similarity between an original activation and its reconstruction is typically near 0.99 — but this alone is not meaningful evidence of reconstruction quality. The original activation vectors share a dominant background direction (the global mean vector has norm ≈ 0.99), which inflates cosine similarity for *any* pair of vectors drawn from this space. The meaningful question is whether reconstructions carry **activation-specific** directional information beyond that shared background.

Three complementary metrics are reported, each at the raw level and after mean-centering. Let **μ** be the mean of all original activation vectors and let **r̂** = r / ‖r‖ denote a unit-normalised reconstruction.

### 1 · Raw cosine similarity

`cos(original_i, reconstruction_i)` — reported for reference and continuity with prior work, but should not be interpreted in isolation. High raw cosine is expected regardless of reconstruction quality due to the shared mean direction.

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

You need [uv](https://docs.astral.sh/uv/), a Modal account, and Hugging Face access to gated Gemma + AV weights.

```bash
uv sync
uv run modal setup
modal secret create --force huggingface HF_TOKEN=<your_hf_token>
```

Accept licenses: [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt), [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av), [nla-gemma3-12b-L32-ar](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar).

---

## Run on Modal

### Step 1 — activations (~5–15 min, 1× A100)

```bash
uv run modal run extract_activations.py

# Optional: full prompt (for comparison / ablations)
uv run modal run extract_activations.py --full
```

### Step 2 — AV descriptions (~30–90 min, 12× A100 by default)

```bash
uv run modal run sample_descriptions.py

# Only if Step 1 used --full:
uv run modal run sample_descriptions.py --full
```

Fewer GPUs: `uv run modal run sample_descriptions.py --n-shards 1`

### Step 3 — AR scores (~15–45 min, 12× A100 by default)

```bash
# Scores only (pairwise consistency + fidelity cosines):
uv run modal run reconstruct_scores.py

# With raw reconstructed vectors saved (required for diagnostics):
uv run modal run reconstruct_scores.py \
  --activations activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \
  --descriptions descriptions_prism.parquet \
  --pairwise-out pairwise_consistency_prism.parquet \
  --fidelity-out fidelity_scores_prism.parquet \
  --save-vectors
```

Fewer GPUs: `uv run modal run reconstruct_scores.py --n-shards 1`

---

## Pull results locally (`data/`)

**Concept:** Modal keeps files on the cloud volume. We copy the pieces you need into **`data/`** so local scripts have one place to look (not scattered at repo root).

```bash
# Default: CSV + persona activations + descriptions
uv run python scripts/pull_from_modal.py

# Also pull full-prompt activation parquets (after Step 1 --full)
uv run python scripts/pull_from_modal.py --full

# Subset only
uv run python scripts/pull_from_modal.py --only csv,descriptions
```

Manual equivalent (same destinations):

```bash
modal volume get nla-cache /cache/selfdescribe_400.csv data/selfdescribe_400.csv
modal volume get nla-cache /cache/descriptions.parquet data/descriptions.parquet
modal volume get nla-cache /cache/activations_layer32_persona-only_gemma-3-12b-pt.parquet \
  data/activations_layer32_persona-only_gemma-3-12b-pt.parquet
modal volume get nla-cache pairwise_consistency.parquet data/pairwise_consistency.parquet
modal volume get nla-cache fidelity_scores.parquet data/fidelity_scores.parquet

# After Step 3 --save-vectors:
modal volume get nla-cache recon_vectors_prism.parquet data/recon_vectors_prism.parquet
```

Preview (reads from `data/` by default):

```bash
uv run python scripts/preview_descriptions.py
# → scripts/description_preview.txt
```

If a pull fails: `modal volume ls nla-cache`, or match the path from Step 1 logs (`output parquet -> ...`).

---

## Diagnostics (local)

After pulling artifacts to `data/`, run any of the following scripts. All read from `data/` by default; pass explicit paths with `--activations` / `--recon-vectors` if needed.

```bash
# Full pairwise cosine similarity matrix among the 400 original activations
# (raw and mean-centered). Shows how dominated by a shared direction they are.
uv run python scripts/activation_cosine_matrix.py

# Fidelity: does each reconstruction match its own original better than other originals?
# Requires recon_vectors_*.parquet (Step 3 --save-vectors).
uv run python scripts/diagnose_reconstruction_fidelity.py \
  --activations data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet

# Consistency: do the 12 reconstructions per activation agree with each other
# more than reconstructions from different activations do?
# Requires recon_vectors_*.parquet (Step 3 --save-vectors).
uv run python scripts/diagnose_reconstruction_consistency.py \
  --activations data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \
  --recon-vectors data/recon_vectors_prism.parquet

# Earlier inflation diagnostic (uses only cosine scores, no raw vectors needed):
uv run python scripts/diagnose_cosine_inflation.py
```

---

## Volume artifacts (`nla-cache` → `/cache` in jobs)

| Path | What it is |
|------|------------|
| `selfdescribe_400.csv` | 400 prompts + `attr_class` / `attr` labels |
| `activations_layer32_persona-only_gemma-3-12b-pt.parquet` | Default vectors (persona last-token) |
| `activations_layer32.parquet` | Legacy alias; only written when Step 1 uses `--full` |
| `descriptions.parquet` | `activation_idx`, `sample_idx`, `description` |
| `pairwise_consistency*.parquet` | Within-activation recon-recon cosine scores (26,400 rows) |
| `fidelity_scores*.parquet` | Per-reconstruction fidelity cosine scores (4,800 rows) |
| `recon_vectors*.parquet` | Raw reconstructed vectors — `activation_idx`, `sample_idx`, `recon_vector` (4,800 rows × 3840-d); written only with `--save-vectors` |
| `hf/` | Cached model weights (reuse across runs) |

---

## Linear probes (optional, local CPU)

**Concept:** A quick sanity check—not the main reliability metric. We train a simple logistic regression on activations to predict dataset labels within each `attr_class` (Gender, Religion, Occupation, Country). If persona-only activations beat full-prompt ones on probe accuracy (especially where full-prompt was near the majority baseline), extraction is carrying more persona signal.

```bash
uv run python scripts/pull_from_modal.py
uv run python scripts/train_linear_probes.py

# Compare to full-prompt activations (pull after Step 1 --full):
uv run python scripts/pull_from_modal.py --full
uv run python scripts/train_linear_probes.py \
  --compare-activations data/activations_layer32_full_gemma-3-12b-pt.parquet
```

| Flag | Meaning |
|------|---------|
| `--activations` | Which parquet to probe (default: `data/…persona-only…parquet`) |
| `--compare-activations` | Second parquet; prints side-by-side accuracies |
| `--csv` | Prompts/labels (default: `data/selfdescribe_400.csv`) |

---

## End to end pipeline

```bash
uv sync && uv run modal setup
# set HF token once: modal secret create --force huggingface HF_TOKEN=...

uv run modal run extract_activations.py
uv run modal run sample_descriptions.py
uv run modal run reconstruct_scores.py

uv run python scripts/pull_from_modal.py

uv run python scripts/preview_descriptions.py
uv run python scripts/train_linear_probes.py
```

**Check you’re done:** `data/descriptions.parquet` has 4800 rows; preview shows persona text (no infobox line) and AV text about the persona theme, not only “Wikipedia generator.” After Step 3: `pairwise_consistency.parquet` has 26,400 rows, `fidelity_scores.parquet` has 4,800 rows, no NaNs. Raw cosine scores near 0.99 are expected — see [Evaluation metrics](#evaluation-metrics) for how to interpret them using baseline-adjusted centered gaps.
