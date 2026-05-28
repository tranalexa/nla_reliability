# Attribution: vendored NLA inference client

[`nla/nla_inference.py`](nla_inference.py) is vendored verbatim (subject to
minor local diffs) from the open-source Natural Language Autoencoders (NLA)
reference implementation released by kitft, which is itself a reimplementation
of the inference protocol described by Anthropic.

| Layer                           | Owner                                   | Where                                                                                  |
|---------------------------------|-----------------------------------------|----------------------------------------------------------------------------------------|
| NLA method (AV + AR)            | Anthropic / Transformer Circuits        | [Paper, 2026](https://transformer-circuits.pub/2026/nla/index.html), [research post](https://www.anthropic.com/research/natural-language-autoencoders) |
| AV checkpoint (Gemma-3-12B L32) | Anthropic, released via kitft's HF account | <https://huggingface.co/kitft/nla-gemma3-12b-L32-av>                                |
| AR checkpoint (Gemma-3-12B L32) | Anthropic, released via kitft's HF account | <https://huggingface.co/kitft/nla-gemma3-12b-L32-ar>                                |
| Inference client (this file)    | kitft (MIT licensed)                    | <https://github.com/kitft/natural_language_autoencoders/blob/main/nla_inference.py> |
| Reliability evaluation layer    | this repo (MIT)                         | everything else under `nla_reliability/`                                               |

## What's vendored

`nla/nla_inference.py` exposes **both** halves of the NLA pair:

- **`NLAClient`** — the **AV (activation verbalizer)**.  Encodes an activation
  vector into a 1-token input embedding and streams a natural-language
  description through SGLang.  Used by this repo in
  [`sample_descriptions.py`](../sample_descriptions.py) (Modal Step 2).

- **`NLACritic`** — the **AR (activation reconstructor)**.  Reads a description
  back into an activation vector with a truncated transformer + Linear(d, d)
  head, pure torch (no SGLang).  Used by this repo in
  [`reconstruct_scores.py`](../reconstruct_scores.py) (Modal Step 3).

Both checkpoints (`kitft/nla-gemma3-12b-L32-av`,
`kitft/nla-gemma3-12b-L32-ar`) are Anthropic-released NLAs distributed via
kitft's Hugging Face account.

## What this repo contributes

`nla_reliability` does **not** retrain NLAs or modify the AV/AR architecture.
Its original contribution is a reliability-evaluation layer on top of
Anthropic's pretrained NLA:

- Step 1 activation extraction (vanilla HuggingFace forward pass at layer 32 -
  unrelated to NLA).
- Mean-centered cosine fidelity / consistency diagnostics
  (`nla/synthesis_metrics.py`).
- MPNet within / between-item text consistency
  (`scripts/compute_text_consistency*.py`).
- Generalizability-theory variance decomposition (`nla/g_theory.py`,
  `scripts/g_theory_study.py`).
- Linear-probe decoding (majority + LR) on originals vs mean reconstructions
  (`nla/synthesis_metrics.py::run_probes`, `scripts/train_linear_probe*.py`).
- Synthesis CSV + LaTeX table generation (`scripts/build_synthesis_tables.py`)
  and the 17-figure paper bundle (`scripts/generate_figure_bundle.py`).

## License

The vendored client (`nla/nla_inference.py`) is MIT-licensed by kitft. The rest
of this repo is MIT-licensed (see [`LICENSE`](../LICENSE)). The full
attribution stack is also recorded in the top-level [`NOTICE`](../NOTICE) and
[`CITATION.cff`](../CITATION.cff).

When updating `nla_inference.py`, compare against upstream first and keep
local changes minimal.
