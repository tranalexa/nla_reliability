# nla_reliability

Reliability evaluation layer on top of Anthropic's **Natural Language Autoencoders (NLA)**.
We sample 400 items from four canonical runs (PRISM, Bias-in-Bios, MMLU + choices,
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
> See [NOTICE](NOTICE), [CITATION.cff](CITATION.cff), and
> [nla/ATTRIBUTION.md](nla/ATTRIBUTION.md) for the full attribution stack.

---

## Repo structure

```text
extract_activations.py        Modal Step 1 (Gemma forward pass; **not** Anthropic NLA code)
sample_descriptions.py        Modal Step 2 (NLAClient = AV; uses Anthropic NLA checkpoint)
reconstruct_scores.py         Modal Step 3 (NLACritic = AR; uses Anthropic NLA checkpoint)
nla/
  __init__.py                 Public re-exports
  nla_inference.py            VENDORED FROM kitft — NLAClient + NLACritic
  datasets.py                 PRISM / BiasBios / MMLU loaders (+ mmlu prompt modes)
  paths.py                    run_id-keyed local + Modal volume path helpers
  g_theory.py                 G-study + D-study variance decomposition
  synthesis_metrics.py        Centered fidelity/consistency, probes, summary tables
  ATTRIBUTION.md              Detailed attribution for the vendored inference client
scripts/
  pull_from_modal.py          Download artifacts from Modal volume per run_id
  compute_text_consistency.py            Within-item MPNet cosine per run_id
  compute_text_consistency_between.py    Between-item MPNet baseline per run_id
  g_theory_study.py           G-theory study per run_id and metric
  train_linear_probes.py      Majority + LR probes (profession / gender)
  train_linear_probe_mmlu.py  Subject probes for MMLU runs
  build_synthesis_tables.py   All reports/*.csv + reports/results_table.tex
  generate_figure_bundle.py   All figures_bundle/*.png (the single figure output dir)
  smoke_test_loaders.py       No-GPU sanity check that PRISM/BiasBios/MMLU load
  diagnose_*.py / preview_descriptions.py  ad-hoc inspection utilities
data/runs/<run_id>/           Local artifacts after pull_from_modal (gitignored)
reports/                      Synthesis CSVs + LaTeX table (gitignored)
figures_bundle/               17 paper figures (gitignored)
notebooks/analysis_synthesis.ipynb   Source notebook (outputs cleared at commit)
EXPERIMENT_OVERVIEW.md        Full conceptual write-up (provenance + design + metrics)
```

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

### Step 1 — Extract activations (≈ 5–15 min per run, 1× A100)

```bash
uv run modal run extract_activations.py --run-id prism         --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id biosbias      --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id mmlu_choice   --n-items 400 --seed 42
uv run modal run extract_activations.py --run-id mmlu_nochoice --n-items 400 --seed 42
# or:
uv run modal run extract_activations.py --all-runs --n-items 400 --seed 42
```

### Step 2 — Sample AV descriptions (≈ 30–90 min per run, up to 12× A100-80GB)

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

### Step 3 — AR reconstruction + scoring (≈ 15–45 min per run, up to 12× A100)

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
uv run python scripts/preview_descriptions.py --run-id prism --n 3
```

---

## Script → paper figure / table map

Every figure and table in the paper is regenerated by exactly one (or a small
pipeline of) scripts in this repo:

| Paper artifact                          | Generator                                              | Output                                                       |
|-----------------------------------------|--------------------------------------------------------|--------------------------------------------------------------|
| Headline metrics table                  | `scripts/build_synthesis_tables.py`                    | `reports/synthesis_headline_metrics.csv`                     |
| LaTeX results table                     | `scripts/build_synthesis_tables.py`                    | `reports/results_table.tex`                                  |
| Per-run summary (mean/std/p5/p95)       | `scripts/build_synthesis_tables.py`                    | `reports/summary_stats.csv`                                  |
| Data inventory                          | `scripts/build_synthesis_tables.py`                    | `reports/synthesis_inventory.csv`                            |
| Text consistency (within / between)     | `scripts/compute_text_consistency*.py` → `build_synthesis_tables.py` | `reports/synthesis_text_consistency.csv`         |
| Linear probes (incl. majority baseline) | `scripts/train_linear_probes*.py` → `build_synthesis_tables.py`      | `reports/synthesis_linear_probes.csv`            |
| G-study variance decomposition          | `scripts/g_theory_study.py` → `build_synthesis_tables.py`            | `reports/synthesis_g_theory_variance.csv`        |
| D-study (G_rel by n′)                   | `scripts/g_theory_study.py` → `build_synthesis_tables.py`            | `reports/synthesis_g_theory_d_study.csv`         |
| Fig. centered fidelity (matched/mismatched/gap) | `scripts/generate_figure_bundle.py`            | `figures_bundle/01_centered_fidelity.png`                    |
| Fig. centered consistency               | `scripts/generate_figure_bundle.py`                    | `figures_bundle/02_centered_consistency.png`                 |
| Fig. raw vs centered                    | `scripts/generate_figure_bundle.py`                    | `figures_bundle/03_raw_vs_centered.png`                      |
| Fig. cosine inflation                   | `scripts/generate_figure_bundle.py`                    | `figures_bundle/04_cosine_inflation.png`                     |
| Fig. per-item mean scatter              | `scripts/generate_figure_bundle.py`                    | `figures_bundle/05_per_item_means.png`                       |
| Fig. text consistency (MPNet)           | `scripts/generate_figure_bundle.py`                    | `figures_bundle/06_text_consistency.png`                     |
| Fig. G-study variance components        | `scripts/generate_figure_bundle.py`                    | `figures_bundle/07a_g_theory_variance.png`, `07b_…`         |
| Fig. D-study G_rel curves               | `scripts/generate_figure_bundle.py`                    | `figures_bundle/08_d_study_curves.png`                       |
| Fig. linear probe accuracies            | `scripts/generate_figure_bundle.py`                    | `figures_bundle/09_linear_probes.png`                        |
| MMLU prompt mode side-by-side           | `scripts/generate_figure_bundle.py` + notebook §6      | `figures_bundle/1?_mmlu_*.png`                               |
| Notebook narrative                      | `notebooks/analysis_synthesis.ipynb`                   | (re-renders from `reports/*.csv`)                            |

See the comments at the top of each script for input schemas.

---

## Reproducibility notes

- **Pinned seeds.** `seed=42` propagates through dataset sampling (PRISM
  shuffle, BiasBios stratified split, MMLU subject sampling) and is printed
  at job start in `extract_activations.py`, `sample_descriptions.py`, and
  `train_linear_probe*.py`. The Gemma-3 forward pass is deterministic given
  identical bf16 weights and inputs.
- **SGLang sampling is non-deterministic.** Step 2 draws 12 AV samples per
  activation at temperature 1.0 through SGLang. Reruns will produce
  different descriptions and slightly different raw cosines, but the
  *aggregate* metrics (centered fidelity gap, within-item vs between-item
  MPNet, G_rel, probe accuracy) should match within run-to-run noise. This
  is expected behavior for stochastic verbalization at T=1.0.
- **Hardware.** Step 1 needs one A100 (≈ 15 GB peak). Step 2/3 default to 12
  shards × A100 each but degrade gracefully with `--n-shards`. Local
  analysis is CPU-only and runs in a few minutes on a laptop.
- **Storage.** Each run produces about 50 MB of parquets; the four-run
  superset is ~ 200 MB and lives entirely on the Modal volume until you
  pull it.
- **Expected row counts per run.** 400 items × 12 AV samples = 4,800
  description rows and 4,800 fidelity rows; C(12, 2) × 400 = 26,400
  pairwise rows; 4,800 reconstruction vectors (with `--save-vectors`).

The grader's recipe (and the recommended pre-submission check the author
should re-run) is exactly the eight steps in
[Local analysis](#local-analysis-after-the-modal-pipeline-finishes) above,
plus the four Modal jobs that produce the inputs.

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
| Reliability evaluation layer      | this repo (MIT) | everything else                                            |

Read [NOTICE](NOTICE) for the formal attribution block,
[CITATION.cff](CITATION.cff) for citation metadata, and
[nla/ATTRIBUTION.md](nla/ATTRIBUTION.md) for the per-file vendored-code notes.
Conceptual write-up of the reliability methodology is in
[EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md).
