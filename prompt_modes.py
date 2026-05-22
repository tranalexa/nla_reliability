"""Prompt modes for SelfDescribe activation extraction (shared by Step 1 and probes)."""

from __future__ import annotations

INFOBOX_SUFFIX = (
    "Write a hypothetical but realistic Wikipedia biography infobox for me."
)

PROMPT_MODES = frozenset({"full", "persona-only", "last-persona-token"})
PERSONA_MODES = frozenset({"persona-only", "last-persona-token"})

LEGACY_PARQUET = "activations_layer32.parquet"
LEGACY_NPY = "activations_layer32.npy"


def model_slug(model_id: str) -> str:
    return model_id.replace("google/", "").replace(".", "-")


def output_paths(
    cache_dir: str, layer: int, prompt_mode: str, model_id: str
) -> tuple[str, str]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"prompt_mode {prompt_mode!r} not in {sorted(PROMPT_MODES)}"
        )
    base = f"activations_layer{layer}_{prompt_mode}_{model_slug(model_id)}"
    return f"{cache_dir}/{base}.parquet", f"{cache_dir}/{base}.npy"


def apply_prompt_mode(prompt: str, mode: str) -> tuple[str, bool]:
    """Return (text_for_forward, suffix_was_stripped)."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"unknown prompt_mode: {mode!r}")
    if mode == "full":
        return prompt, False
    if INFOBOX_SUFFIX in prompt:
        return prompt.split(INFOBOX_SUFFIX, 1)[0].strip(), True
    return prompt.strip(), False


def prepare_prompts(
    raw_prompts: list[str], mode: str, *, max_warnings: int = 20
) -> tuple[list[str], list[int]]:
    """Apply mode to each row; return texts and indices where suffix was missing."""
    missing: list[int] = []
    out: list[str] = []
    for i, p in enumerate(raw_prompts):
        text, stripped = apply_prompt_mode(p, mode)
        if mode in PERSONA_MODES and not stripped:
            missing.append(i)
        out.append(text)
    if missing:
        print(
            f"warning: {len(missing)} row(s) missing {INFOBOX_SUFFIX!r}; "
            f"using full prompt for those rows"
        )
        for idx in missing[:max_warnings]:
            snippet = raw_prompts[idx][:120].replace("\n", " ")
            print(f"  [{idx}] {snippet!r}...")
        if len(missing) > max_warnings:
            print(f"  ... and {len(missing) - max_warnings} more")
    return out, missing
