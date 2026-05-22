# Attribution: `nla_inference.py`

`nla/nla_inference.py` is vendored from the open-source NLA reference implementation:

- **Repository:** [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
- **Upstream file:** `nla_inference.py`

It supports the method described in:

- **Paper / thread:** Fraser-Taliente, Kantamneni, Ong, et al., [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html), Transformer Circuits, 2026.
- **Overview:** [Anthropic research post](https://www.anthropic.com/research/natural-language-autoencoders)

`nla_reliability` uses this client for Step 2 only (SGLang + `kitft/nla-gemma3-12b-L32-av`). Extraction (Step 1), prompt modes, and reliability analysis are separate code in this repository.

When updating `nla_inference.py`, compare against upstream first and keep local changes minimal.
