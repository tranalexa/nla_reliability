# nla_reliability

Reliability evaluation for Anthropic's **Natural Language Autoencoders (NLA)**.
We sample 400 items from four defined runs (PRISM, Bias-in-Bios, MMLU + choices,
MMLU question-only), extract Gemma-3-12B layer-32 activations, generate 12
stochastic AV (Activation Verbalizer) descriptions per activation with the
public NLA AV checkpoint, reconstruct each description back to an activation with
the AR (Activation Reconstructor) checkpoint, and quantify reliability using
mean-centered cosine fidelity/consistency, MPNet text-consistency, linear
probes, and generalizability-theory variance decomposition.

> **Built on Anthropic's Natural Language Autoencoders.**
>
> NLA is Anthropic / Transformer Circuits' framework for unsupervised
> verbalization and reconstruction of LLM activations. This repo does **not**
> retrain NLAs — it consumes Anthropic's pretrained AV/AR checkpoints
> (released on Hugging Face via the kitft account) and adds a reliability-
> evaluation layer on top. The inference client at
> [`nla/nla_inference.py`](nla/nla_inference.py) is vendored unchanged from
> kitft (MIT) and provides **both** `NLAClient` (AV, used in Step 2) and
> `NLACritic` (AR, used in Step 3).
>
> - Research post: <https://www.anthropic.com/research/natural-language-autoencoders>
> - Paper (Fraser-Taliente, Kantamneni, Ong, et al., Transformer Circuits, 2026):
>   <https://transformer-circuits.pub/2026/nla/index.html>
> - AV checkpoint: <https://huggingface.co/kitft/nla-gemma3-12b-L32-av>
> - AR checkpoint: <https://huggingface.co/kitft/nla-gemma3-12b-L32-ar>
> - Vendored client: <https://github.com/kitft/natural_language_autoencoders>
>
> ```bibtex
> @article{frasertaliente2026nla,
>   title  = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
>   author = {Fraser-Taliente, Cody and Kantamneni, Sanjana and Ong, Ben and others},
>   journal= {Transformer Circuits},
>   year   = {2026},
>   url    = {https://transformer-circuits.pub/2026/nla/index.html}
> }
> ```
>
---

The four canonical **run_ids** (used everywhere in CLI flags + path helpers):

| run_id          | dataset    | prompt format                                       |
|-----------------|------------|-----------------------------------------------------|
| `prism`         | prism      | last user turn of a multi-turn chat                 |
| `biosbias`      | biosbias   | full biography (gender-scrubbed)                    |
| `mmlu_choice`   | mmlu       | `Question: … \n A.… B.… C.… D.… \n Answer:`         |
| `mmlu_nochoice` | mmlu       | question stem only (choices kept as CSV metadata)   |

---

## Setup (once per machine)

You need [uv](https://docs.astral.sh/uv/) (or plain pip), a Modal account, and
a Hugging Face token with read access to Gemma-3 + the kitft NLA checkpoints.

### Install with uv (preferred)

```bash
uv sync
uv run modal setup
modal secret create --force huggingface HF_TOKEN=<your_hf_token>
```

### Install with pip (alternative)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
modal setup
modal secret create --force huggingface HF_TOKEN=<your_hf_token>
```

`requirements.txt` is generated from `uv.lock`
(`uv export --format requirements.txt --no-dev > requirements.txt`); the
canonical lockfile is still `uv.lock` so `uv sync` is the most reliable path.

### Accept Hugging Face licenses

- [google/gemma-3-12b-pt](https://huggingface.co/google/gemma-3-12b-pt)
- [kitft/nla-gemma3-12b-L32-av](https://huggingface.co/kitft/nla-gemma3-12b-L32-av)
- [kitft/nla-gemma3-12b-L32-ar](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar)

`LabHC/bias_in_bios` and `cais/mmlu` are publicly accessible.

### Smoke test (no GPU, no Modal)

```bash
HF_TOKEN=<your_hf_token> uv run python scripts/smoke_test_loaders.py
```

Loads 5 items from each dataset and prints prompt previews. Expected ending:

```
  PASS  prism
  PASS  biosbias
  PASS  mmlu

All dataset loaders OK.
```

You can also dry-run the local analysis chain (only checks file presence and
schemas; no compute):

```bash
uv run python scripts/build_synthesis_tables.py --dry-run
```

---

## End-to-end Modal pipeline

All three Modal jobs accept `--run-id <id>` for a single run, or `--all-runs`
for all four. Outputs land on the Modal volume `nla-cache` (mounted at
`/cache` inside containers) under `/cache/runs/<run_id>/`.

> **Heads-up on `--all-runs`:** running back-to-back jobs occasionally trips
> Modal's transient "open files preventing the operation" error on the shared
> volume. If that happens, just rerun the failing run_id individually.

### Step 1 — Extract activations

```bash
uv run modal run extract_activations.py --run-id prism         --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id biosbias      --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id mmlu_choice   --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id mmlu_nochoice --n-items 400 --seed 42
# or:
uv run modal run extract_activations.py --all-runs --n-items 400 --seed 42
```

### Step 2 — Sample AV descriptions

```bash
uv run modal run sample_descriptions.py --run-id prism         --n-samples 12
uv run modal run sample_descriptions.py --run-id biosbias      --n-samples 12
uv run modal run sample_descriptions.py --run-id mmlu_choice   --n-samples 12
uv run modal run sample_descriptions.py --run-id mmlu_nochoice --n-samples 12
# or:
uv run modal run sample_descriptions.py --all-runs --n-samples 12
```

Limit parallel GPUs with `--n-shards 1` (or any value ≤ 12) if you don't have
the quota.

### Step 3 — AR reconstruction + scoring

Pass `--save-vectors` so the centered-cosine diagnostics work locally.

```bash
uv run modal run reconstruct_scores.py --run-id prism         --save-vectors
uv run modal run reconstruct_scores.py --run-id biosbias      --save-vectors
uv run modal run reconstruct_scores.py --run-id mmlu_choice   --save-vectors
uv run modal run reconstruct_scores.py --run-id mmlu_nochoice --save-vectors
# or:
uv run modal run reconstruct_scores.py --all-runs --save-vectors
```

Raw fidelity / pairwise cosines will sit near 0.99. That is **expected** —
activation space has a strong shared mean direction and raw cosines must be
read in tandem with the centered diagnostics generated downstream (see
`scripts/build_synthesis_tables.py` and the notebook).

---

## Local analysis (after the Modal pipeline finishes)

```bash
# 1. Pull all four runs into data/runs/<run_id>/
uv run python scripts/pull_from_modal.py --all-runs

# 2. Compute MPNet text consistency for each run (~minutes on CPU)
uv run python scripts/compute_text_consistency.py         --all-runs
uv run python scripts/compute_text_consistency_between.py --all-runs

# 3. G-theory study per run + metric (writes per-run g_theory_*.csv)
for r in prism biosbias mmlu_choice mmlu_nochoice; do
  uv run python scripts/g_theory_study.py --run-id "$r" --metric all
done

# 4. Linear probes (profession/gender on PRISM+BiasBios, subject on MMLU runs)
uv run python scripts/train_linear_probes.py
uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_choice
uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_nochoice

# 5. Synthesise every reports/*.csv + reports/results_table.tex
uv run python scripts/build_synthesis_tables.py

# 6. Render every figures_bundle/*.png (the single figure output dir)
uv run python scripts/generate_figure_bundle.py

# 7. Execute the analysis notebook (uses the CSVs from step 5)
uv run jupyter nbconvert --to notebook --execute \
    notebooks/analysis_synthesis.ipynb --inplace
```

If you want to inspect AV samples interactively:

```bash
uv run python scripts/preview_descriptions.py --run-id prism --num-samples 3
```
---

## Code reuse and attribution

This repo's original contribution is the reliability-evaluation layer:
Step 1 (vanilla HuggingFace activation extraction) and everything in
`scripts/`, `nla/g_theory.py`, `nla/synthesis_metrics.py`, the figure bundle,
the notebook, and the docs. NLA itself (the AV/AR architecture and the
pretrained checkpoints) is Anthropic's, and the inference client is kitft's.

| What                              | Who          | Where                                                          |
|-----------------------------------|--------------|----------------------------------------------------------------|
| NLA method                        | Anthropic    | [Transformer Circuits 2026](https://transformer-circuits.pub/2026/nla/index.html), [research post](https://www.anthropic.com/research/natural-language-autoencoders) |
| AV checkpoint (Gemma-3-12B L32)   | Anthropic via kitft HF | <https://huggingface.co/kitft/nla-gemma3-12b-L32-av>      |
| AR checkpoint (Gemma-3-12B L32)   | Anthropic via kitft HF | <https://huggingface.co/kitft/nla-gemma3-12b-L32-ar>      |
| Inference client (`NLAClient` + `NLACritic`) | kitft (MIT) | [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders), vendored at [`nla/nla_inference.py`](nla/nla_inference.py) |

