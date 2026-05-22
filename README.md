# nla_reliability

SelfDescribe prompts describe a fictional person. We run **Gemma-3-12B** once per prompt and save the **layer-32 last-token** hidden state (a 3840-d vector). A separate **activation verbalizer (AV)** never sees the prompt text—it only gets that vector and writes an explanation. We generate **8 stochastic** explanations per vector, then analyze consistency.

Everything heavy runs on [Modal](https://modal.com). Artifacts live on a persistent volume named **`nla-cache`**, which appears as the folder **`/cache`** inside GPU jobs (not a path on your laptop until you `modal volume get`).

---

## Pipeline overview

```text
SelfDescribe CSV (400 rows)
    → Step 1: Gemma forward → activations parquet
    → Step 2: AV (SGLang) × 8 samples → descriptions parquet
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
| Output | `descriptions.parquet` (400 activations × 8 samples = 3200 rows) |

Expect AV text in a structured “investigator” style (often mentions tone, a paraphrased phrase, and what the model might say next)—not a clean copy of the CSV sentence.

---

## Setup (once per machine)

You need [uv](https://docs.astral.sh/uv/), a Modal account, and Hugging Face access to gated Gemma + AV weights.

```bash
uv sync
uv run modal setup
modal secret create --force huggingface HF_TOKEN=<your_hf_token>
```

Accept licenses: [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt), [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av).

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
```

Preview (reads from `data/` by default):

```bash
uv run python scripts/preview_descriptions.py
# → scripts/description_preview.txt
```

If a pull fails: `modal volume ls nla-cache`, or match the path from Step 1 logs (`output parquet -> ...`).

---

## Volume artifacts (`nla-cache` → `/cache` in jobs)

| Path | What it is |
|------|------------|
| `selfdescribe_400.csv` | 400 prompts + `attr_class` / `attr` labels |
| `activations_layer32_persona-only_gemma-3-12b-pt.parquet` | Default vectors (persona last-token) |
| `activations_layer32.parquet` | Legacy alias; only written when Step 1 uses `--full` |
| `descriptions.parquet` | `activation_idx`, `sample_idx`, `description` |
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

uv run python scripts/pull_from_modal.py

uv run python scripts/preview_descriptions.py
uv run python scripts/train_linear_probes.py
```

**Check you’re done:** `data/descriptions.parquet` has 3200 rows; preview shows persona text (no infobox line) and AV text about the persona theme, not only “Wikipedia generator.”
