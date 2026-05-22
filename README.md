# nla_reliability

Measure **reliability** of Natural Language Autoencoders (NLAs): how consistently an activation verbalizer (AV) describes the same internal vector under stochastic sampling.

Pipeline runs on [Modal](https://modal.com) with a shared volume `nla-cache` mounted at `/cache`.

## Pipeline

### Step 1 — Extract activations

We turn 400 real user prompts into fixed numerical “snapshots” of what Gemma-3-12B’s internal state looks like after reading each one. Those layer-32 vectors are the objects we will later ask the NLA verbalizer to describe, so every reliability measurement is grounded in the same base model and prompt set.

| | |
|--|--|
| Script | `extract_activations.py` |
| Model | `google/gemma-3-12b-pt` |
| Output | `activations_layer32.parquet` (400 × 3840, unit L2 norm) |

### Step 2 — Sample descriptions

We run the trained activation verbalizer on each saved vector many times with stochastic decoding, to see how much the wording drifts when the input activation is identical. That produces the raw data for NLA reliability: 8 natural-language descriptions per activation, not a single deterministic caption.

| | |
|--|--|
| Script | `sample_descriptions.py` |
| Model | `kitft/nla-gemma3-12b-L32-av` (SGLang) |
| Output | `descriptions.parquet` (3200 rows) |

```text
SelfDescribe prompts → Gemma-3-12B forward → L2-normalized activations
                              ↓
                    NLA AV (inject vector → generate)
                              ↓
                    descriptions.parquet (reliability analysis)
```

Row `i` in activations, CSV, and descriptions all refer to the **same** prompt (shuffle seed 42, first 400 rows).

## Files

**Entrypoints:** `extract_activations.py` (Step 1) and `sample_descriptions.py` (Step 2). Everything else supports those runs.

### `extract_activations.py` (Step 1)

Builds the activation dataset: one last-token residual per SelfDescribe prompt, normalized for downstream use. See **Step 1** above for intent; details below.

Modal app `nla-extract`. On one A100:

- Downloads 400 prompts from [Transluce/SelfDescribe](https://huggingface.co/datasets/Transluce/SelfDescribe-Llama-3.1-8B-Instruct) (shuffle seed 42) → `selfdescribe_400.csv`
- Loads `google/gemma-3-12b-pt`, runs a forward pass per prompt
- Takes the **last-token** residual at **layer 32** (3840-dim)
- **L2-normalizes** each vector and writes `activations_layer32.parquet` + `.npy`

### `sample_descriptions.py` (Step 2)

Runs the verbalizer at scale: many stochastic descriptions per activation so you can quantify inconsistency. See **Step 2** above for intent; details below.

Modal app `nla-sample`. Orchestration only:

- Spawns **12 parallel** GPU workers (configurable `--n-shards`), each with its own SGLang subprocess + AV weights
- Reads `activations_layer32.parquet`, assigns activations `i` where `i % n_shards == shard_id`
- For each activation: builds injected `input_embeds` (via `nla_inference`), calls SGLang **8 times** at `temperature=1.0`
- Writes `descriptions_shard{N}.parquet`, then **`merge_shards`** → `descriptions.parquet` (3200 rows)

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

# Step 1 — activations
uv run modal run extract_activations.py

# Step 2 — AV descriptions (requires Step 1 on nla-cache)
uv run modal run sample_descriptions.py
```

Optional: `source .venv/bin/activate` after `uv sync` if you prefer an activated shell without `uv run`.

```bash
modal setup
```

Hugging Face:

1. Accept licenses for [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) and [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av).
2. Create a token with gated-repo read access.
3. Store on Modal:

```bash
modal secret create --force huggingface HF_TOKEN=<token>
```

## Run

### Step 1

Populates `nla-cache` with prompts, activations, and cached Gemma weights. Run once before Step 2.

```bash
uv run modal run extract_activations.py
```

### Step 2

Reads Step 1 outputs and writes `descriptions.parquet`. Default: 12 parallel A100-80GB shards (~5–15 min wall clock).

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
| `selfdescribe_400.csv` | `user_prompt`, `attr_class`, `attr` |
| `activations_layer32.parquet` | `activation_vector` (400 rows, dim 3840, **unit L2 norm**) |
| `activations_layer32.npy` | Same matrix as NumPy |
| `descriptions.parquet` | `activation_idx`, `sample_idx`, `description` (3200 rows) |
| `descriptions_shard*.parquet` | Per-shard outputs before merge (optional) |
| `hf/` | Cached HF weights |

Download locally:

```bash
modal volume get nla-cache /cache/descriptions.parquet descriptions.parquet
modal volume get nla-cache /cache/selfdescribe_400.csv selfdescribe_400.csv
```

## Step 2 internals (short)

Each shard:

1. Starts **SGLang** with the AV checkpoint (`--disable-radix-cache` required for `input_embeds`).
2. For each activation: build prompt embeddings once (cached), inject vector, POST to SGLang at `temperature=1.0`.
3. Merges shards into `descriptions.parquet`.

**Normalization:** Step 1 saves **unit-norm** vectors. At inference, `nla_inference` rescales to `injection_scale` from `nla_meta.yaml` (80000 for this AV) — not double unit-normalization.

**Tunables** in `sample_descriptions.py`: `N_SHARDS_DEFAULT`, `CONCURRENCY`, `MAX_NEW_TOKENS` (96 by default; raise if descriptions truncate or keep `<explanation>` tags).