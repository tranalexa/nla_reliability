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

| Mode | Forward text | Output file on volume |
|------|----------------|------------------------|
| `full` (default) | Entire `user_prompt` | `/cache/activations_layer32_full_gemma-3-12b-pt.parquet` **and** legacy `/cache/activations_layer32.parquet` |
| `persona-only` | Persona text only (suffix stripped) | `/cache/activations_layer32_persona-only_gemma-3-12b-pt.parquet` |
| `last-persona-token` | Same transform as `persona-only` | `/cache/activations_layer32_last-persona-token_gemma-3-12b-pt.parquet` |

Model slug in filenames is `gemma-3-12b-pt` (from `google/gemma-3-12b-pt`).

Persona modes remove one **exact** suffix string (no regex), split on the first occurrence, keep text before it:

```text
Write a hypothetical but realistic Wikipedia biography infobox for me.
```

If a row lacks that suffix, Step 1 logs a warning and uses the full prompt for that row. The AV in Step 2 still never sees prompt text—only the injected activation + investigator template.

### Step 2 — Sample descriptions

We run the trained activation verbalizer on each saved vector many times with stochastic decoding.

| | |
|--|--|
| Script | `sample_descriptions.py` |
| Model | `kitft/nla-gemma3-12b-L32-av` (SGLang) |
| Output | `descriptions.parquet` (N × 8 rows) by default |

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

## Setup (once)

```bash
uv sync
uv run modal setup
modal secret create --force huggingface HF_TOKEN=<token>
```

Accept licenses for [gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt) and [nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av).

## Run

### Step 1

```bash
# Full-prompt activations (+ legacy activations_layer32.parquet)
uv run modal run extract_activations.py --prompt-mode full

# Persona-only last token (recommended for attribute signal)
uv run modal run extract_activations.py --prompt-mode persona-only
```

### Step 2

Default (legacy full-prompt activations on volume):

```bash
uv run modal run sample_descriptions.py
```

**Persona activations** (use vectors from persona Step 1):

```bash
uv run modal run sample_descriptions.py --prompt-mode persona-only
```

Equivalent explicit path:

```bash
uv run modal run sample_descriptions.py \
  --activations /cache/activations_layer32_persona-only_gemma-3-12b-pt.parquet
```

Optional separate output file (keeps old `descriptions.parquet`):

```bash
uv run modal run sample_descriptions.py --prompt-mode persona-only \
  --output descriptions_persona.parquet
```

Fewer shards:

```bash
uv run modal run sample_descriptions.py --prompt-mode persona-only --n-shards 1
```

### Download & preview

```bash
modal volume get nla-cache /cache/selfdescribe_400.csv selfdescribe_400.csv
modal volume get nla-cache /cache/descriptions.parquet descriptions.parquet
modal volume get nla-cache /cache/activations_layer32_persona-only_gemma-3-12b-pt.parquet .

uv run python scripts/preview_descriptions.py --activations-source persona-only
```

Preview shows **persona text** (suffix stripped), not the raw CSV infobox line. AV descriptions come from whatever activations Step 2 used—re-run Step 2 with `--prompt-mode persona-only` after persona extraction.

## Volume artifacts (`nla-cache` → `/cache`)

| Path | Contents |
|------|----------|
| `selfdescribe_400.csv` | `user_prompt`, `attr_class`, `attr` (400 rows) |
| `activations_layer32.parquet` | Full-prompt activations (`--prompt-mode full`) |
| `activations_layer32_{mode}_gemma-3-12b-pt.parquet` | Tagged Step 1 outputs |
| `descriptions.parquet` | AV samples (default Step 2 output) |
| `hf/` | Cached HF weights |

## Linear probes (local)

```bash
modal volume get nla-cache /cache/selfdescribe_400.csv selfdescribe_400.csv
modal volume get nla-cache /cache/activations_layer32.parquet activations_layer32.parquet
modal volume get nla-cache /cache/activations_layer32_persona-only_gemma-3-12b-pt.parquet .

uv run python scripts/train_linear_probes.py
uv run python scripts/train_linear_probes.py \
  --activations activations_layer32_persona-only_gemma-3-12b-pt.parquet \
  --compare-activations activations_layer32.parquet
```

| Flag | Meaning |
|------|---------|
| `--activations PATH` | Parquet from Step 1 (default: `activations_layer32.parquet`) |
| `--compare-activations PATH` | Second parquet; side-by-side probe table |
| `--csv PATH` | SelfDescribe CSV (default: `selfdescribe_400.csv`) |

## End-to-end persona workflow

```bash
uv run modal run extract_activations.py --prompt-mode persona-only
uv run modal run sample_descriptions.py --prompt-mode persona-only
modal volume get nla-cache /cache/descriptions.parquet descriptions.parquet
uv run python scripts/preview_descriptions.py --activations-source persona-only
uv run python scripts/train_linear_probes.py \
  --activations activations_layer32_persona-only_gemma-3-12b-pt.parquet \
  --compare-activations activations_layer32.parquet
```
