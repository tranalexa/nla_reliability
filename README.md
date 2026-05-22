# nla_reliability

Measure **reliability** of Natural Language Autoencoders (NLAs): how consistently an activation verbalizer (AV) describes the same internal vector under stochastic sampling.

Pipeline runs on [Modal](https://modal.com) with a shared volume `nla-cache` mounted at `/cache`.

## Pipeline

### Step 1 — Extract activations

We turn 400 SelfDescribe prompts (shuffled sample, seed 42) into fixed numerical “snapshots” of what Gemma-3-12B’s internal state looks at the **last token** after reading each prompt. Those layer-32 vectors are what we later ask the NLA verbalizer to describe.

| | |
|--|--|
| Script | `extract_activations.py` |
| Model | `google/gemma-3-12b-pt` (override with `--model-id`) |
| Default output | `activations_layer32.parquet` (N × 3840, unit L2 norm) — written when `--prompt-mode full` |

**Prompt modes** (`--prompt-mode`):

| Mode | Forward text | Output file pattern |
|------|----------------|---------------------|
| `full` (default) | Entire `user_prompt` | Tagged `activations_layer32_full_gemma3-12b-pt.parquet` **and** legacy `activations_layer32.parquet` |
| `persona-only` | Persona text only (suffix stripped) | `activations_layer32_persona-only_gemma3-12b-pt.parquet` |
| `last-persona-token` | Same transform as `persona-only` | `activations_layer32_last-persona-token_gemma3-12b-pt.parquet` (separate file for A/B probe runs) |

Persona modes remove one **exact** suffix string (no regex), split on the first occurrence, keep text before it:

```text
Write a hypothetical but realistic Wikipedia biography infobox for me.
```

If a row lacks that suffix, Step 1 logs a warning and uses the full prompt for that row (row count stays aligned with CSV). The AV in Step 2 still never sees prompt text—only the injected activation + investigator template; persona modes change **extraction geometry**, not AV inputs.

### Step 2 — Sample descriptions

We run the trained activation verbalizer on each saved vector many times with stochastic decoding, to see how much the wording drifts when the input activation is identical. That produces the raw data for NLA reliability: 8 natural-language descriptions per activation.

| | |
|--|--|
| Script | `sample_descriptions.py` |
| Model | `kitft/nla-gemma3-12b-L32-av` (SGLang) |
| Output | `descriptions.parquet` (N × 8 rows) |

```text
SelfDescribe (400) → Gemma-3-12B forward → L2-normalized activations
                              ↓
                    NLA AV (inject vector → generate)
                              ↓
                    descriptions.parquet (reliability analysis)
```

Row `i` in activations, CSV, and descriptions all refer to the **same** prompt (shuffle seed 42, first 400 rows).

## Compute (rough)

| Step | Hardware | Scale | Wall clock (order of magnitude) |
|------|----------|-------|----------------------------------|
| Step 1 | 1× A100 | 400 forwards | ~5–15 min |
| Step 2 | 12× A100-80GB | ~3.2k gens (400×8), `max_new_tokens=500` | ~30–90 min |

Step 2 dominates cost (12 GPUs × SGLang AV + long generations). If a shard times out, re-run with fewer shards (`--n-shards 6`) or lower `MAX_NEW_TOKENS` in `sample_descriptions.py`.

**Before a re-run:** delete stale volume CSV/parquet if row count ≠ 400 so Step 1 regenerates `selfdescribe_400.csv` and matching activations.

## Files

**Entrypoints:** `extract_activations.py` (Step 1) and `sample_descriptions.py` (Step 2). Everything else supports those runs.

### `extract_activations.py` (Step 1)

Builds the activation dataset: one last-token residual per SelfDescribe prompt, normalized for downstream use.

Modal app `nla-extract`. On one A100:

- Samples **400** rows from [Transluce/SelfDescribe](https://huggingface.co/datasets/Transluce/SelfDescribe-Llama-3.1-8B-Instruct) (shuffle seed 42) → `selfdescribe_400.csv`
- Loads `google/gemma-3-12b-pt`, runs a forward pass per prompt
- Applies `--prompt-mode` (see `prompt_modes.py`)
- Takes the **last-token** residual at **layer 32** (3840-dim)
- **L2-normalizes** each vector; writes tagged parquet + `.npy`; on `full`, also copies to legacy `activations_layer32.parquet` / `.npy`

### `sample_descriptions.py` (Step 2)

Runs the verbalizer at scale: many stochastic descriptions per activation so you can quantify inconsistency.

Modal app `nla-sample`. Orchestration only:

- Spawns **12 parallel** GPU workers (configurable `--n-shards`), each with its own SGLang subprocess + AV weights
- Reads `activations_layer32.parquet`, assigns activations `i` where `i % n_shards == shard_id`
- For each activation: builds injected `input_embeds` (via `nla_inference`), calls SGLang **8 times** at `temperature=1.0`
- Writes `descriptions_shard{N}.parquet`, then **`merge_shards`** → `descriptions.parquet`

Does not implement injection math; bundles `nla_inference.py` into the container image.

### `nla_inference.py` (library)

Vendored from [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders). Core pieces:

- **`NLAClient`** (actor / verbalizer): load `nla_meta.yaml` + embedding table; tokenize the AV prompt; **inject** an activation at a special token; POST `input_embeds` to SGLang; strip `<explanation>` tags
- **`NLACritic`** (not used in this repo yet): text → reconstructed vector for round-trip / MSE

Use alone when you already have a SGLang server and want `client.generate(vector)` in a notebook. Step 2 imports this file on Modal.

## Setup (once)

Install [uv](https://docs.astral.sh/uv/), then sync the project env (Python ≥3.12):

```bash
uv sync
```

Run Modal from the synced env:

```bash
uv run modal setup

# Step 1 — activations (400-row sample)
uv run modal run extract_activations.py
# Persona-only last token (better attribute signal for probes)
uv run modal run extract_activations.py --prompt-mode persona-only

# Step 2 — AV descriptions (requires Step 1 on nla-cache)
uv run modal run sample_descriptions.py
```

Optional: `source .venv/bin/activate` after `uv sync` if you prefer an activated shell without `uv run`.

Hugging Face:

1. Accept licenses for [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) and [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av).
2. Create a token with gated-repo read access.
3. Store on Modal:

```bash
modal secret create --force huggingface HF_TOKEN=<token>
```

## Run

### Step 1

Populates `nla-cache` with prompts, activations, and cached Gemma weights.

```bash
uv run modal run extract_activations.py
uv run modal run extract_activations.py --prompt-mode persona-only
uv run modal run extract_activations.py --model-id google/gemma-3-12b-pt --prompt-mode full
```

Regenerates `selfdescribe_400.csv` automatically if row count ≠ 400.

Tagged outputs on the volume (pull with matching paths):

```bash
modal volume get nla-cache /cache/activations_layer32_persona-only_gemma3-12b-pt.parquet .
```

### Step 2

Reads Step 1 outputs and writes `descriptions.parquet`. Default: 12 parallel A100-80GB shards.

```bash
uv run modal run sample_descriptions.py
```

Single GPU (slower):

```bash
uv run modal run sample_descriptions.py --n-shards 1
```

## Volume artifacts (`nla-cache` → `/cache`)

| Path | Contents |
|------|----------|
| `selfdescribe_400.csv` | `user_prompt`, `attr_class`, `attr` (400 rows) |
| `activations_layer32.parquet` | Legacy full-prompt activations (`--prompt-mode full`) |
| `activations_layer32.npy` | Same matrix as NumPy |
| `activations_layer32_{mode}_{model_slug}.parquet` | Tagged Step 1 outputs (e.g. `persona-only`, `full`) |
| `descriptions.parquet` | `activation_idx`, `sample_idx`, `description` (N × 8 rows) |
| `descriptions_shard*.parquet` | Per-shard outputs before merge (optional) |
| `hf/` | Cached HF weights |

Download locally:

```bash
modal volume get nla-cache /cache/descriptions.parquet descriptions.parquet
modal volume get nla-cache /cache/selfdescribe_400.csv selfdescribe_400.csv
modal volume get nla-cache /cache/activations_layer32.parquet activations_layer32.parquet
```

Preview ~10 random AV descriptions side-by-side with the matching SelfDescribe prompt:

```bash
uv run python scripts/preview_descriptions.py
# -> scripts/description_preview.txt
```

## Step 2 internals (short)

Each shard:

1. Starts **SGLang** with the AV checkpoint (`--disable-radix-cache` required for `input_embeds`).
2. For each activation: build prompt embeddings once (cached), inject vector, POST to SGLang at `temperature=1.0`.
3. Merges shards into `descriptions.parquet`.

**Normalization:** Step 1 saves **unit-norm** vectors. At inference, `nla_inference` rescales to `injection_scale` from `nla_meta.yaml` (80000 for this AV) — not double unit-normalization.

**Tunables** in `sample_descriptions.py`: `N_SHARDS_DEFAULT`, `CONCURRENCY`, `MAX_NEW_TOKENS` (500; `nla_inference` CLI default is 200).

## Linear probes (local)

Train sklearn logistic regression on layer-32 **last-token** Gemma activations to predict SelfDescribe `attr` labels within each `attr_class` (Gender, Religion, Occupation, **Country** = nationality). Compares probe accuracy to a majority-class baseline on an 80/20 split (stratified when possible). Row order must match Step 1 CSV (seed 42).

```bash
modal volume get nla-cache /cache/selfdescribe_400.csv selfdescribe_400.csv
modal volume get nla-cache /cache/activations_layer32.parquet activations_layer32.parquet

uv sync
# Full-prompt activations (default)
uv run python scripts/train_linear_probes.py

# Persona-only parquet from Step 1
modal volume get nla-cache /cache/activations_layer32_persona-only_gemma3-12b-pt.parquet .
uv run python scripts/train_linear_probes.py \
  --activations activations_layer32_persona-only_gemma3-12b-pt.parquet

# A/B: persona vs full (same seed/split)
uv run python scripts/train_linear_probes.py \
  --activations activations_layer32_persona-only_gemma3-12b-pt.parquet \
  --compare-activations activations_layer32.parquet
```

| Flag | Meaning |
|------|---------|
| `--activations PATH` | Parquet from Step 1 (default: `activations_layer32.parquet`) |
| `--compare-activations PATH` | Second parquet; prints side-by-side probe accuracies |
| `--csv PATH` | SelfDescribe CSV (default: `selfdescribe_400.csv`) |
