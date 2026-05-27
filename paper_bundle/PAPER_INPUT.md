# NLA Reliability — Paper Input Bundle

Use this file with Claude (or similar) to draft the paper. Attach the PNGs in `paper_bundle/figures/` for figure captions. Full narrative and interpretation: `docs/ANALYSIS_FINDINGS.md` (also copied here as `ANALYSIS_FINDINGS.md`).

---

## Methods (one paragraph)

We evaluate Natural Language Autoencoders (NLA) on Gemma-3-12B layer-32 activations (L2-normalized) for 400 items × 12 stochastic activation-verbalizer (AV) samples (temperature 1.0) per dataset (PRISM, Bias in Bios, MMLU), using public checkpoints `kitft/nla-gemma3-12b-L32-{av,ar}`. Reliability is assessed via reconstruction cosine (raw and mean-centered), pairwise recon consistency, G-theory on `fidelity_cos`, and MPNet (`all-mpnet-base-v2`) mean pairwise text cosine across verbalizations. Validity is assessed via logistic-regression linear probes on original vs mean-reconstructed vectors (80/20 split, seed 42).

---

## Headline results (centered gaps)

| metric | biosbias | mmlu | prism |
| --- | --- | --- | --- |
| Fidelity centered gap | 0.6455 | 0.2530 | 0.6748 |
| Consistency centered gap | 0.7929 | 0.2725 | 0.8081 |
| Fidelity centered (matched) | 0.6468 | 0.2380 | 0.6675 |
| Consistency centered (within-item) | 0.9513 | 0.9237 | 0.9528 |
| Consistency centered (between-item) | 0.1575 | 0.6508 | 0.1455 |
| ||mean activation|| | 0.9900 | 0.9966 | 0.9912 |

### Full summary statistics (`tables/summary_stats.csv`)

| dataset | metric | mean | std | n |
| --- | --- | --- | --- | --- |
| prism | Fidelity raw (matched) | 0.994282 | 0.001492 | 4800 |
| prism | Fidelity raw (mismatched) | 0.981179 | 0.004055 | 24000 |
| prism | Fidelity raw gap | 0.013103 | 0.004611 | 24000 |
| prism | Fidelity centered (matched) | 0.667470 | 0.136853 | 4800 |
| prism | Fidelity centered (mismatched) | -0.007301 | 0.118449 | 24000 |
| prism | Fidelity centered gap | 0.674770 | 0.187248 | 24000 |
| prism | Consistency raw (within-item) | 0.999151 | 0.000473 | 26400 |
| prism | Consistency raw (between-item) | 0.982757 | 0.004716 | 132000 |
| prism | Consistency raw gap | 0.016455 | 0.004721 | 26400 |
| prism | Consistency centered (within-item) | 0.952790 | 0.029309 | 26400 |
| prism | Consistency centered (between-item) | 0.145522 | 0.134331 | 132000 |
| prism | Consistency centered gap | 0.808077 | 0.136417 | 26400 |
| prism | ||mean activation|| | 0.991182 | 0.000000 | 1 |
| biosbias | Fidelity raw (matched) | 0.993385 | 0.003679 | 4800 |
| biosbias | Fidelity raw (mismatched) | 0.980913 | 0.007996 | 24000 |
| biosbias | Fidelity raw gap | 0.012472 | 0.007863 | 24000 |
| biosbias | Fidelity centered (matched) | 0.646763 | 0.098905 | 4800 |
| biosbias | Fidelity centered (mismatched) | 0.001221 | 0.169638 | 24000 |
| biosbias | Fidelity centered gap | 0.645542 | 0.199389 | 24000 |
| biosbias | Consistency raw (within-item) | 0.999130 | 0.000510 | 26400 |
| biosbias | Consistency raw (between-item) | 0.984257 | 0.006686 | 132000 |
| biosbias | Consistency raw gap | 0.014844 | 0.006724 | 26400 |
| biosbias | Consistency centered (within-item) | 0.951258 | 0.023512 | 26400 |
| biosbias | Consistency centered (between-item) | 0.157543 | 0.225935 | 132000 |
| biosbias | Consistency centered gap | 0.792868 | 0.228062 | 26400 |
| biosbias | ||mean activation|| | 0.990037 | 0.000000 | 1 |
| mmlu | Fidelity raw (matched) | 0.993081 | 0.000891 | 4800 |
| mmlu | Fidelity raw (mismatched) | 0.990692 | 0.001674 | 24000 |
| mmlu | Fidelity raw gap | 0.002389 | 0.001671 | 24000 |
| mmlu | Fidelity centered (matched) | 0.237992 | 0.148446 | 4800 |
| mmlu | Fidelity centered (mismatched) | -0.014978 | 0.128853 | 24000 |
| mmlu | Fidelity centered gap | 0.252970 | 0.204110 | 24000 |
| mmlu | Consistency raw (within-item) | 0.999098 | 0.000443 | 26400 |
| mmlu | Consistency raw (between-item) | 0.995806 | 0.001862 | 132000 |
| mmlu | Consistency raw gap | 0.003289 | 0.001910 | 26400 |
| mmlu | Consistency centered (within-item) | 0.923680 | 0.038023 | 26400 |
| mmlu | Consistency centered (between-item) | 0.650844 | 0.141615 | 132000 |
| mmlu | Consistency centered gap | 0.272531 | 0.146346 | 26400 |
| mmlu | ||mean activation|| | 0.996604 | 0.000000 | 1 |

---

## Text consistency (MPNet)

| dataset | n | mean | std | median | p5 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| prism | 400 | 0.8613 | 0.0323 | 0.8647 | 0.8048 | 0.9112 |
| biosbias | 400 | 0.8699 | 0.0327 | 0.8725 | 0.8139 | 0.9142 |
| mmlu | 400 | 0.8255 | 0.0435 | 0.8321 | 0.7454 | 0.8849 |

---

## Linear probes (validity)

| dataset | target | vector_source | n | n_classes | train_n | test_n | majority_acc | probe_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prism | gender | original | 400 | 3 | 320 | 80 | 0.5750 | 0.5750 |
| prism | gender | recon | 400 | 3 | 320 | 80 | 0.5750 | 0.5750 |
| biosbias | profession | original | 400 | 26 | 320 | 80 | 0.2625 | 0.3000 |
| biosbias | gender | original | 400 | 2 | 320 | 80 | 0.5000 | 0.5500 |
| biosbias | profession | recon | 400 | 26 | 320 | 80 | 0.2625 | 0.2625 |
| biosbias | gender | recon | 400 | 2 | 320 | 80 | 0.5000 | 0.5875 |
| mmlu | subject | original | 400 | 4 | 320 | 80 | 0.2500 | 0.9625 |
| mmlu | subject | recon | 400 | 4 | 320 | 80 | 0.2500 | 0.7625 |

**Key finding:** MMLU subject probe drops 0.96 → 0.76 after mean recon; PRISM gender unchanged; BiasBios mixed (profession −0.04, gender +0.04).

---

## G-theory variance components

| dataset | sigma2_p | sigma2_pi | var_pct_p | var_pct_pi | cronbach_alpha |
| --- | --- | --- | --- | --- | --- |
| prism | 0.000002 | 0.000000 | 90.601538 | 9.398462 | 0.991430 |
| biosbias | 0.000013 | 0.000000 | 98.630561 | 1.369439 | 0.998844 |
| mmlu | 0.000001 | 0.000000 | 75.485133 | 24.474939 | 0.973691 |

---

## G-theory D-study (G_rel by n′ samples)

| dataset | 1 | 2 | 3 | 4 | 6 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| biosbias | 0.9863 | 0.9931 | 0.9954 | 0.9965 | 0.9977 | 0.9988 |
| mmlu | 0.7552 | 0.8605 | 0.9025 | 0.9250 | 0.9487 | 0.9737 |
| prism | 0.9060 | 0.9507 | 0.9666 | 0.9747 | 0.9830 | 0.9914 |

---

## Figures (attach these images)

| File | Suggested caption |
|------|-------------------|
| `figures/fidelity_dist.png` | Distribution of mean-centered matched fidelity cosines by dataset |
| `figures/consistency_dist.png` | Within-item vs between-item centered recon consistency |
| `figures/raw_vs_centered.png` | Raw vs centered metric gaps (inflation diagnostic) |
| `figures/cosine_inflation.png` | Shared mean direction: raw cosines near 1.0 |

---

## LaTeX table

See `tables/results_table.tex` for a publication-ready two/three-column table.

---

## Limitations (for Discussion)

- Cosine similarity on L2-normalized activations is inflated by a shared mean direction (‖μ‖ ≈ 0.99); centered gaps are the interpretable reliability metrics.
- AV describes meta-linguistic state, not prompt content; text MPNet consistency (~0.83–0.87) is much lower than recon cosine (~0.999).
- Linear probes test coarse linear decodability, not full semantic preservation.
- PRISM gender has severe class imbalance (`non_binary` n=1).
- G-theory on `fidelity_cos` does not subsume text-space or probe validity.

---

## Reproduce

```bash
uv run python scripts/make_report_tables.py --data-dir data/data
uv run python scripts/g_theory_study.py --dataset all --data-dir data/data
uv run python scripts/train_linear_probes_multi_dataset.py --vector-source compare --data-dir data/data
uv run python scripts/compute_text_consistency.py --dataset all --data-dir data/data
uv run python scripts/export_paper_bundle.py
```
