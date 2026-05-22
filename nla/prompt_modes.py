"""Prompt modes for SelfDescribe activation extraction (shared by Step 1 and probes)."""

from __future__ import annotations

INFOBOX_SUFFIX = (
    "Write a hypothetical but realistic Wikipedia biography infobox for me."
)

PROMPT_MODES = frozenset({"full", "persona-only"})

LEGACY_PARQUET = "activations_layer32.parquet"
LEGACY_NPY = "activations_layer32.npy"

DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"
DEFAULT_LAYER = 32
DEFAULT_PROMPT_MODE = "persona-only"


def model_slug(model_id: str) -> str:
    """e.g. google/gemma-3-12b-pt -> gemma-3-12b-pt"""
    return model_id.replace("google/", "").replace(".", "-")


def activations_basename(layer: int, prompt_mode: str, model_id: str) -> str:
    return f"activations_layer{layer}_{prompt_mode}_{model_slug(model_id)}.parquet"


def output_paths(
    cache_dir: str, layer: int, prompt_mode: str, model_id: str
) -> tuple[str, str]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"prompt_mode {prompt_mode!r} not in {sorted(PROMPT_MODES)}"
        )
    base = f"activations_layer{layer}_{prompt_mode}_{model_slug(model_id)}"
    return f"{cache_dir}/{base}.parquet", f"{cache_dir}/{base}.npy"


def default_activations_parquet(
    cache_dir: str = ".",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
) -> str:
    """Tagged activations path for the default prompt mode (persona-only)."""
    return output_paths(cache_dir, layer, prompt_mode, model_id)[0]


def resolve_activations_parquet(
    cache_dir: str,
    *,
    activations: str | None = None,
    prompt_mode: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = DEFAULT_LAYER,
) -> str:
    """Pick activations path: explicit path wins, else tagged file for prompt_mode."""
    if activations:
        path = activations
        if not path.startswith("/"):
            path = f"{cache_dir.rstrip('/')}/{path.lstrip('/')}"
        return path
    mode = prompt_mode or DEFAULT_PROMPT_MODE
    return output_paths(cache_dir, layer, mode, model_id)[0]


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
        if mode == "persona-only" and not stripped:
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
