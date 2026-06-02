"""Shared library for the NLA reliability pipeline.

Module map:
  paths              - filename + path helpers for all run_ids (canonical layout)
  datasets           - HF dataset loaders (PRISM, BiasBios, MMLU)
  nla_inference      - vendored Anthropic NLA AV (NLAClient) + AR (NLACritic) clients
                       (from kitft/natural_language_autoencoders; see ATTRIBUTION.md)
  g_theory           - p x i ANOVA G-study + D-study helpers
  synthesis_metrics  - centered fidelity/consistency, text consistency, linear probes,
                       summary stats, LaTeX table builder
"""

from nla.paths import (
    DATA_DIR,
    ROOT,
    RUN_IDS,
    RUNS_DIR,
    dataset_for_run,
    mmlu_prompt_mode_for_run,
    run_dir,
    validate_run_id,
)

__all__ = [
    "DATA_DIR",
    "ROOT",
    "RUN_IDS",
    "RUNS_DIR",
    "dataset_for_run",
    "mmlu_prompt_mode_for_run",
    "run_dir",
    "validate_run_id",
]
