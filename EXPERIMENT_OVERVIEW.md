# NLA Reliability — Full Experiment Overview

Use this document to sanity-check the paper. Every claim has a source file/line citation in this repo. Numbers (means, gaps, G values) are not hardcoded here; the freshest set is in `reports/` and `figures_bundle/`.

---

## 0. Provenance

The Natural Language Autoencoder (NLA) method, including the AV (activation
verbalizer) and AR (activation reconstructor) architecture, is Anthropic /
Transformer Circuits' work (Fraser-Taliente et al., 2026 — see
<https://transformer-circuits.pub/2026/nla/index.html> and the
[research post](https://www.anthropic.com/research/natural-language-autoencoders)).
The pretrained `kitft/nla-gemma3-12b-L32-{av,ar}` checkpoints used throughout
this repo are Anthropic-released NLAs distributed via the kitft Hugging Face
account. The inference client at [`nla/nla_inference.py`](nla/nla_inference.py)
is vendored verbatim from
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
(MIT) and supplies **both** `NLAClient` (AV) and `NLACritic` (AR) used by
Steps 2 and 3 below.

This repo's original contribution is the **reliability evaluation layer**:
Step 1 activation extraction (vanilla HuggingFace, unrelated to NLA), the
mean-centered cosine diagnostics, MPNet text consistency, G-theory variance
decomposition, linear probes, and the synthesis tables/figures. NLA itself is
not retrained or modified here. See [`NOTICE`](NOTICE),
[`CITATION.cff`](CITATION.cff), and
[`nla/ATTRIBUTION.md`](nla/ATTRIBUTION.md) for the full attribution stack.

---

## 1. What the experiment is asking

We probe the **reliability** of the Natural Language Autoencoder (NLA) round-trip on Gemma-3-12B layer-32 activations, and the **validity** of those reconstructions for downstream linear probing. Concretely, for each of N=400 items per benchmark we:

1. Extract a layer-32 activation (Gemma-3-12B-pt, last token of the prompt) — Step 1.
2. Sample K=12 stochastic natural-language descriptions of that activation with the AV ("verbalizer") — Step 2.
3. Reconstruct an activation back from each description with the AR ("reconstructor") — Step 3.
4. Score:
   - **Fidelity** = how well the reconstructions point at the *correct* original (centered cosine matched vs mismatched).
   - **Consistency** = how well the 12 reconstructions of one item agree with each other (centered cosine within vs between).
   - **Text-space consistency** = MPNet pairwise cosine over the 12 AV descriptions (within and between items).
   - **Linear probe validity** = decoding accuracy of dataset metadata from originals vs from the mean of the 12 reconstructions.
   - **G-theory** = variance decomposition + D-study on per-sample fidelity and per-sample consistency, treating the 12 AV samples as random measurement occasions.

Models, checkpoints, and design parameters are fixed across the whole experiment. The four "runs" differ only in (a) the source benchmark and (b) for MMLU, the prompt format.

---

## 2. Models, checkpoints, and design constants

- Base model: `google/gemma-3-12b-pt`, layer **32** residual stream, **last non-padding token** (extracted from the backbone, not lm_head). Vectors are L2-normalized to unit norm at save time. `extract_activations.py` lines 136–168, 237–243.
- AV (verbalizer): `kitft/nla-gemma3-12b-L32-av` (HuggingFace), served on Modal via SGLang. `sample_descriptions.py` lines 28–37.
- AR (reconstructor): `kitft/nla-gemma3-12b-L32-ar` (HuggingFace), loaded as `NLACritic`. `reconstruct_scores.py` lines 105–110.
- Items per benchmark: **N = 400** (`nla/datasets.py` line 25).
- Stochastic AV samples per item: **K = 12** (`sample_descriptions.py` line 30).
- AV temperature: **T = 1.0**, max new tokens 500. AR is a deterministic forward pass (no sampling).
- Random seed: 42 for item sampling and probe splits.
- Vector dimensionality: 3,840 (Gemma-3-12B-pt layer 32 hidden size).

The NLA module code is vendored from `kitft/natural_language_autoencoders` (see `nla/ATTRIBUTION.md`). This repo never retrains AV/AR; it consumes the released checkpoints.

---

## 3. Datasets and the four experimental runs

All loaders live in `nla/datasets.py`. `SUPPORTED_DATASETS = ["prism", "biosbias", "mmlu"]` (line 23).

### 3.1 PRISM (run id `prism`)

- HuggingFace source: `Transluce/PRISM-gender-Llama-3.1-8B-Instruct` (`nla/datasets.py` `_load_prism`, lines 60–78).
- Sampling: `shuffle(seed=42).select(range(400))` — uniform random of 400 conversations.
- Prompt: multi-turn chat flattened by `_flatten_prism_conversation` (lines 34–57); keeps turns through the last *user* turn (trailing assistant turns are dropped). Format `"User: ...\nAssistant: ...\nUser: ..."`. The last Gemma token is therefore the last user-turn token.
- Metadata columns: `gender` (model-predicted), `gt_gender` (survey ground-truth). Probe target: `gender`.

### 3.2 Bias in Bios (run id `biosbias`)

- HuggingFace source: `LabHC/bias_in_bios` (`_load_biosbias`, lines 129–173).
- Sampling: stratified 50/50 male/female, 200 each, then shuffled.
- Prompt: `hard_text` field (fallback `text`); gender indicators (pronouns, honorifics, gendered nouns) are scrubbed to `[PERSON]` via regex.
- Metadata columns: `profession` (26 classes), `gender`. Probe targets: `profession`, `gender`.

### 3.3 MMLU (run ids `mmlu_choice` and `mmlu_nochoice`)

Both runs use the **same 400 items** (same `item_idx`, `question`, `choices`, `answer`, `subject`). They differ only in `prompt_text`, controlled by the `mmlu_prompt_mode` argument to `nla.datasets.load_dataset_items`:

- Source: `cais/mmlu`. Four subjects, 100 questions each → 400 total: `abstract_algebra`, `moral_scenarios`, `virology`, `astronomy` (`MMLU_SUBJECTS`, `nla/datasets.py`).
- Split preference: `test` first, `validation` fallback.
- Filtering: non-empty question, exactly four non-empty choices, non-null answer.
- `mmlu_choice` → `_format_mmlu_prompt(..., mode="with_choices")` builds `"Question: {stem}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"`.
- `mmlu_nochoice` → `_format_mmlu_prompt(..., mode="question_only")` returns the question stem only.
- Probe target: `subject` (4 classes).

### 3.4 Run table

| Run id | Label | Prompt fed to Gemma | Probe target | Local artifacts under |
|---|---|---|---|---|
| `prism` | PRISM | Last user turn of multi-turn chat | gender | `data/runs/prism/` |
| `biosbias` | Bias in Bios | Full biography text (gender-scrubbed) | profession, gender | `data/runs/biosbias/` |
| `mmlu_choice` | MMLU-Choice | `Question:` + A–D + `Answer:` | subject | `data/runs/mmlu_choice/` |
| `mmlu_nochoice` | MMLU-NoChoice | Question stem only | subject | `data/runs/mmlu_nochoice/` |

`nla/paths.py::RUN_IDS` and `build_synthesis_tables.py::RUN_LABELS` / `RUN_TARGETS` are the authoritative sources.

---

## 4. Step 1 — Extract activations

Script: `extract_activations.py`. Runs on Modal (1× A100 per dataset, ~5–15 min). CLI: `uv run modal run extract_activations.py --dataset {prism|biosbias|mmlu} --n-items 400 --seed 42`. `--all-datasets` runs all three.

Per item:

1. Tokenize `prompt_text` (Gemma tokenizer, no chat template).
2. Forward through Gemma-3-12B-pt with `output_hidden_states=True`.
3. Take `hidden_states[layer + 1][i, seq_len - 1]` — layer 32 residual at the *last non-padding token* (lines 136–168).
4. L2-normalize, cast to float32.
5. Write CSV (`{dataset}_400.csv`) and parquet (`activations_layer32_{dataset}_gemma-3-12b-pt.parquet`).

Schemas:
- `{dataset}_400.csv` (400 rows): `item_idx`, `dataset`, `prompt_text`, plus dataset-specific metadata.
- `activations_layer32_*.parquet` (400 rows): single column `activation_vector` (list[float32], length 3,840). Row order matches `item_idx = 0..399`.

Sanity invariants paper claims should respect:
- **All activations are unit-norm.** The dominant background direction is large: ‖μ‖ ≈ 0.99 (where μ is the mean of the 400 activations). This is what motivates centering. The exact ‖μ‖ per run is in `reports/synthesis_headline_metrics.csv` under `||mean activation||`.
- **Raw cosines between any two activations therefore sit at ~0.98–0.99** even for unrelated items. Centered cosines remove this background.

---

## 5. Step 2 — AV (verbalizer) sampling

Script: `sample_descriptions.py`. Runs on Modal (12× A100 per dataset by default).

Per activation, draws K=12 descriptions:

- Loads `kitft/nla-gemma3-12b-L32-av`, serves an SGLang endpoint.
- Calls `NLAClient._build_embeds()` to inject the activation as a prompt embedding (rescaled to `injection_scale` internally — see `nla/nla_inference.py`), then `_sglang_generate(temperature=1.0, max_new_tokens=500)` (lines 161–168).
- Parses the model output with `EXPLANATION_RE`; falls back to raw text on no match.
- Default concurrency 12 (a thread pool over 12 shards). Sharding key: `i % n_shards == shard_id`.

Output: `descriptions_{dataset}.parquet` (4,800 rows = 400 × 12). Schema: `activation_idx` (int), `sample_idx` (0..11), `description` (str). Lines 193–199.

Important for the paper:

- The 12 samples are **independent generations from the same prompt+activation**, differing only by sampling temperature/top-p noise. They are *not* paraphrases of one canonical sentence.
- AV runs in the same Modal job for all three benchmarks (prism, biosbias, mmlu); the only difference is the input parquet.

---

## 6. Step 3 — AR (reconstructor) scoring

Script: `reconstruct_scores.py`. Modal (12× A100, ~15–45 min per dataset).

Per activation:

1. Load all 12 descriptions.
2. For each description, call `NLACritic.reconstruct(description)` — deterministic forward pass returning a raw (non-normalized) 3,840-d vector.
3. Compute, for that activation:
   - **Pairwise consistency**: all `C(12, 2) = 66` cos(reconₐ, recon_b) pairs.
   - **Fidelity**: for each sample, cos(original, recon).
4. `cos_sim` (lines 107–110) **L2-normalizes both arguments before dot product**, so cosines are well-defined regardless of recon scale.
5. With `--save-vectors`, also store the **raw recon vector** (not unit-normalized).

Output files (Modal volume `nla-cache` → `/cache/`):

| File | Rows | Schema |
|---|---|---|
| `pairwise_consistency_{dataset}.parquet` | 26,400 = 400 × 66 | `activation_idx`, `sample_i`, `sample_j` (i < j), `cos_sim` |
| `fidelity_scores_{dataset}.parquet` | 4,800 | `activation_idx`, `sample_idx`, `fidelity_cos` |
| `recon_vectors_{dataset}.parquet` (optional) | 4,800 | `activation_idx`, `sample_idx`, `recon_vector` (list[float32] length 3,840, raw) |

Sharding splits 400 activations across 12 shards (`i % n_shards == k`) and merges parquets with row-count asserts.

---

## 7. Pulling artifacts locally

Script: `scripts/pull_from_modal.py`. Targets `data/runs/<run_id>/` via the
helpers in `nla/paths.py`. `--all-runs` pulls every canonical run;
`--only csv,activations,...` is a comma subset; `--run-id <id>` pulls a single run.

Local layout after `pull_from_modal.py --all-runs`:

```
data/runs/prism/         (8 files: csv, activations, descriptions, pairwise,
data/runs/biosbias/        fidelity, recon_vectors, text_consistency_*, text_between_item_*)
data/runs/mmlu_choice/
data/runs/mmlu_nochoice/
```

If your paper says "we use four runs", these are the four. `nla/paths.RUN_IDS` is the authoritative tuple.

---

## 8. Centered metrics (the headline reliability story)

Defined and computed in `nla/synthesis_metrics.py` (`compute_run_metrics`, which delegates to `compute_fidelity` and `compute_consistency`) and driven by `scripts/build_synthesis_tables.py`. `nla/synthesis_metrics.HEADLINE_METRICS` is the list that flows into `reports/synthesis_headline_metrics.csv`.

### 8.1 Why centering

All originals share a dominant mean direction μ. Raw cosines for *any* pair sit at ~0.99. We subtract μ from both originals and unit-normalized reconstructions, then renormalize, then cosine. This isolates **activation-specific** structure.

Formally, with `r̂ = r / ‖r‖`:

```
centered_cos(a, b) = cos(a − μ, b − μ)    after L2 renormalization
```

μ is the mean of the 400 originals for that run (`mean_orig = originals.mean(axis=0)`).

### 8.2 Fidelity (specificity vs other items)

For each (activation i, sample s):

- **Matched (centered)** = `centered_cos(orig_i, r̂_{i,s})`.
- **Mismatched (centered)** = `centered_cos(orig_j, r̂_{i,s})` for 5 random j ≠ i per recon (constant `N_MISMATCHES_PER_RECON = 5`, `nla/synthesis_metrics.py`).
- **Gap (centered)** = matched − mismatched (per-pair subtraction with `np.repeat` so both arrays have length 24,000 = 400 × 12 × 5).

The headline metric is the mean of the per-pair centered gap, recorded in `reports/synthesis_headline_metrics.csv` under `Fidelity centered gap`. Distribution stats (p5, median, p95, std) come from `describe()` (lines 209–217).

### 8.3 Consistency (specificity vs other items in recon space)

Reshape recons into `(N=400, K=12, D=3840)`. For each activation:

- **Within (centered)** = all `C(12, 2) = 66` centered cos(r̂_{i,a}, r̂_{i,b}) pairs → 26,400 values total.
- **Between (centered)** = `n_between_factor × n_within = 5 × 26,400 = 132,000` random pairs across different activations (`ai ≠ aj`, random sample indices), centered.
- **Gap (centered)** = each within value minus one randomly drawn between value, with 26,400 paired differences.

Recorded as `Consistency centered (within-item)`, `Consistency centered (between-item)`, `Consistency centered gap` in `reports/synthesis_headline_metrics.csv`.

Two scripts also generate stdout-only diagnostics with the same metric definitions but slightly different bookkeeping: `scripts/diagnose_reconstruction_fidelity.py` and `scripts/diagnose_reconstruction_consistency.py` (both compute their own μ from the supplied activation file).

---

## 9. Text-space consistency (MPNet)

Two scripts produce the MPNet numbers; both embed all 4,800 AV descriptions once with `sentence-transformers/all-mpnet-base-v2`, batched at 64, normalized.

### 9.1 Within-item (`scripts/compute_text_consistency.py`)

- Reshape embeddings `(N=400, K=12, D=768)`.
- For each activation, average all `C(K, 2)` pairwise cosines among its 12 description embeddings.
- Output: `data/text_consistency_mpnet_{dataset}.parquet` (and `..._mmlu_choices.parquet` for the MCQ variant). Schema: `activation_idx`, `mean_pairwise_text_cosine`.

### 9.2 Between-item baseline (`scripts/compute_text_consistency_between.py`)

- Per activation i: compute the mean cosine between i's 12 description embeddings and **all 12 × (N − 1)** embeddings from every *other* activation in the same benchmark (whole 4,800 × 4,800 Gram matrix masked to exclude i's own 12×12 block). Output also stores std and p5/median/p95 per activation.
- Per-run summary: `reports/synthesis_text_between_item.csv` with `within_mean`, `between_mean`, `gap_within_minus_between` for the four runs.
- Per-activation parquets: `data/text_between_item_mpnet_{prism|biosbias|mmlu|mmlu_choices}.parquet`.

These let us check whether the apparent within-item text consistency is just "this benchmark talks about similar stuff" (high between baseline → small text specificity gap) or genuine AV paraphrase agreement.

---

## 10. Linear probes (validity)

Implemented in `nla/synthesis_metrics.run_probes` (lines 131–166) and rolled up into `reports/synthesis_linear_probes.csv` by `build_synthesis_tables.py`.

For each (run, target):

1. Load `{dataset}_400.csv` → labels.
2. Load originals (L2-normalized) and recon vectors; compute the **per-activation mean of the 12 recon vectors** and L2-normalize that.
3. Train two `LogisticRegression(max_iter=3000)` probes — one on originals, one on mean-recons — using the same `train_test_split(test_size=0.2, random_state=42, stratify=y)` (with non-stratified fallback for classes too small to stratify, e.g. PRISM's tiny non-binary class).
4. Report `probe_acc` and `majority_acc` (always predicting the train-majority class).

Probe targets per run: PRISM → `gender`; BiasBios → `profession`, `gender`; both MMLU runs → `subject`. `scripts/train_linear_probes.py` is a thin CLI over `run_probes()` for sanity-checking probe numbers per run_id; `scripts/train_linear_probe_mmlu.py` does the same for the two MMLU runs with extra subject-level breakdowns.

This is the "reliability ≠ validity" check: a run can have high recon consistency (the 12 recons agree with each other) and still lose metadata decodability (mean-recon probe accuracy drops vs original-activation probe accuracy).

---

## 11. Generalizability theory

Implemented in `nla/g_theory.py`; run by `scripts/g_theory_study.py` and embedded in `scripts/build_synthesis_tables.py`. The output CSVs are `reports/synthesis_g_theory_variance.csv` and `reports/synthesis_g_theory_d_study.csv`.

### 11.1 Design

Random p × i, both facets random:
- `p` = activations (n_p = 400)
- `i` = AV samples (n_i = 12)

Two separate G-studies, each fed a 400 × 12 matrix of per-cell scores:

- **Fidelity G-study.** Cell (p, i) = `fidelity_cos` = raw cos(original, recon_{p,i}). Source: `fidelity_scores_*.parquet` pivoted by `scores_matrix_from_fidelity` (`nla/g_theory.py` lines 50–59). Recorded with metric label `fidelity_cos`.
- **Consistency G-study.** Cell (p, i) = mean within-item recon-recon cosine that involves sample i, i.e. mean of `cos_sim` over the 11 pairs `(p, i, *)` from `pairwise_consistency_*.parquet`. Built by `scores_matrix_from_pairwise` (lines 62–83). Recorded with metric label `consistency_cos`.

### 11.2 Variance components

Two-way balanced ANOVA in `g_study_pxi` (lines 86–144):

```
σ²_pi = max(MS_pi, 0)
σ²_p  = max((MS_p − MS_pi) / n_i, 0)
σ²_i  = max((MS_i − MS_pi) / n_p, 0)
```

Reported as raw σ², percent of total, and Cronbach α (line 124).

- **σ²_p**: variance across activations after averaging over the 12 samples — the **signal** for ranking items.
- **σ²_i**: variance across AV-sample slots (a whole-dataset shift if one of the 12 sample columns is systematically high/low). In our data this is essentially 0 for every run.
- **σ²_pi**: activation × sample interaction — within-item noise from AV stochasticity. This is the source of unreliability that more samples can average out.

### 11.3 D-study

`d_study_grid` (lines 147–159) evaluates n′ ∈ {1, 2, 3, 4, 6, 12}:

```
G(n′)   = σ²_p / (σ²_p + σ²_pi / n′)            # relative generalizability (rankings)
Φ(n′)   = σ²_p / (σ²_p + σ²_pi / n′ + σ²_i / n_p)  # absolute generalizability
```

`G(n′)` is the column the figures use. The number 12 in the D-study is just the maximum n′ we collected; n′ is *how many of the 12 per-sample scores we average per activation before ranking activations against each other*.

`Phi_abs` is essentially identical to `G_rel` in our data because σ²_i ≈ 0.

### 11.4 In plain language

- **Fidelity G-theory + D-study** quantify how much AV stochasticity perturbs how well each reconstruction matches its own original activation. The D-study curve tells how many samples n′ you need to average to get a stable per-activation fidelity summary, dependable for comparing the 400 activations. σ²_pi is exactly within-item AV noise on hitting the true target; raising n′ averages it out.
- **Consistency G-theory + D-study** do the same for within-item recon agreement. σ²_pi here captures wording-driven swings in how well a recon sits with its 11 siblings; raising n′ stabilizes the item-level summary for comparison.

---

## 12. Synthesis outputs

Everything in `reports/` is produced by `scripts/build_synthesis_tables.py` (which loops the four runs and writes one row per run/metric). The figure bundle in `figures_bundle/` is produced by `scripts/generate_figure_bundle.py` from those CSVs plus raw parquets.

### 12.1 `reports/` CSVs

| File | Rows | What it contains |
|---|---|---|
| `synthesis_inventory.csv` | 4 (one per run) | Which artifacts exist + `prompt_format_label`. |
| `synthesis_headline_metrics.csv` | 28 (7 HEADLINE_METRICS × 4 runs) | Centered fidelity (matched / mismatched / gap), centered consistency (within / between / gap), `‖mean activation‖`. |
| `synthesis_text_consistency.csv` | 4 | MPNet within-item text consistency per run. |
| `synthesis_text_between_item.csv` | 4 | MPNet within, between, and gap text similarity per run. |
| `synthesis_activation_vs_text_similarity.csv` | 4 | Side-by-side comparison of activation-space and text-space within / between / gap. |
| `synthesis_linear_probes.csv` | 10 (1–2 targets × 4 runs × 2 sources) | Probe accuracy original vs mean-recon. |
| `synthesis_g_theory_variance.csv` | 8 (2 metrics × 4 runs) | σ²_p, σ²_i, σ²_pi (raw and %), Cronbach α, G(1), G(12). |
| `synthesis_g_theory_d_study.csv` | 48 (2 metrics × 4 runs × 6 n′) | G(n′) and Φ(n′) curves. |

### 12.2 `figures_bundle/` PNGs

Driven by `scripts/generate_figure_bundle.py`; described in `figures_bundle/README.md` and recomputed any time you re-run the script. Labels use **MMLU-Choice** / **MMLU-NoChoice** and `G` notation. The current set:

- `01_centered_fidelity.png` — centered matched / mismatched / gap per run.
- `02_centered_consistency.png` — centered within / between / gap per run.
- `03_centered_gaps.png` — headline reliability bars (fidelity gap, consistency gap, with p5–p95 whiskers).
- `04_text_consistency_mpnet.png` — within-item MPNet text consistency (mean + p5–p95).
- `04b_text_within_vs_between_mpnet.png` — within vs between vs gap MPNet text cosine.
- `04c_text_similarity_distributions.png` — per-activation violin distributions for within, between, gap.
- `04d_activation_vs_text_similarity_table.png` — table comparing activation-space centered consistency with MPNet text similarity (within, between, gap, Δ gap).
- `05_linear_probes.png` — original vs mean-recon probe accuracy per (run, target).
- `06_gtheory_overview.png` — two stacked variance panels (fidelity, consistency) with G(1) annotation.
- `07a_fidelity_variance_components.png`, `07b_consistency_variance_components.png` — per-metric variance stacks.
- `07c_fidelity_dstudy.png`, `07d_consistency_dstudy.png` — D-study curves with a single G = 0.90 reference line.
- `08_dstudy_side_by_side.png` — fidelity vs consistency D-study, same y-axis.
- `09_grel_n1_vs_n12.png` — bar chart pairing G(n′=1) and G(n′=12) for both metrics.
- `10_raw_score_distributions.png` — violins for raw `fidelity_cos` and within-item `cos_sim`.
- `11_per_item_fid_vs_consistency.png` — per-activation scatter of mean matched fidelity vs mean within-item consistency.

### 12.3 Reproducing the bundle

The figure bundle is the only artifact directory committed in source form (as a directory skeleton); regenerate the PNGs with `scripts/generate_figure_bundle.py` after the synthesis CSVs are in place.

---

## 13. End-to-end command reproduction

This recreates everything from scratch on a fresh laptop with Modal credentials and HF access (see `README.md` §Setup).

```bash
# Step 1-3 on Modal (each line ~5-90 min depending on stage)
uv run modal run extract_activations.py --all-runs --n-items 400 --seed 42
uv run modal run sample_descriptions.py  --all-runs --n-samples 12
uv run modal run reconstruct_scores.py   --all-runs --save-vectors

# Pull artifacts to local data/runs/<run_id>/
uv run python scripts/pull_from_modal.py --all-runs

# Text consistency (within, then between)
uv run python scripts/compute_text_consistency.py         --all-runs
uv run python scripts/compute_text_consistency_between.py --all-runs

# G-theory study per run + metric
for r in prism biosbias mmlu_choice mmlu_nochoice; do
  uv run python scripts/g_theory_study.py --run-id "$r" --metric all
done

# Linear probes per run
uv run python scripts/train_linear_probes.py
uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_choice
uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_nochoice

# Synthesis CSVs (writes reports/synthesis_*.csv + summary_stats.csv + results_table.tex)
uv run python scripts/build_synthesis_tables.py

# Figure bundle (the single PNG output dir)
uv run python scripts/generate_figure_bundle.py
```

---

## 14. Things to check against the paper

When proofing the paper, walk through this list once:

1. **Model + checkpoint names.** Gemma-3-12B-pt, layer 32, `kitft/nla-gemma3-12b-L32-{av,ar}`. Source: §2 above.
2. **N, K, T.** N = 400 items per benchmark, K = 12 AV samples per item, AV temperature T = 1.0. AR is deterministic.
3. **Datasets.** PRISM (random 400), BiasBios (stratified 50/50, gender-scrubbed), MMLU four subjects × 100 (`abstract_algebra`, `moral_scenarios`, `virology`, `astronomy`).
4. **Four runs.** PRISM, BiasBios, MMLU-Choice, MMLU-NoChoice — same items for the two MMLU variants, only `prompt_text` differs.
5. **Centering definition.** μ = mean of 400 originals per run; centered cosine = cos(a − μ, b − μ) after L2 renormalization. Used for matched/mismatched fidelity and within/between consistency.
6. **Mismatch / between sampling.** 5 random mismatched originals per recon for fidelity; 5 × within-pair count random between pairs (different `activation_idx`) for consistency.
7. **Text consistency model.** `sentence-transformers/all-mpnet-base-v2`, batch 64, embeddings L2-normalized. Within = C(12, 2) pairwise per item; between = vs all descriptions of all other items in the same benchmark.
8. **Linear probe protocol.** Logistic regression, 80/20 stratified split (seed 42), `max_iter=3000`, original activations vs mean of 12 recon vectors (each L2-normalized).
9. **G-theory facets.** p × i, both random; n_p = 400, n_i = 12.
10. **Variance components.** σ²_p (between activations), σ²_pi (activation × AV sample), σ²_i (global AV sample slot, ≈ 0 in our data).
11. **Reported G coefficient.** Relative generalizability G (label as `G(n′)` in figures); Φ ≈ G because σ²_i ≈ 0.
12. **What "12" means in the D-study.** It is the maximum n′ we collected, not a separate experiment knob. n′ is how many of the 12 per-sample scores you average per activation before ranking items.

Every claim above can be re-checked from the cited file and line range.
