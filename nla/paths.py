"""Local paths for artifacts pulled from Modal volume nla-cache."""

from __future__ import annotations

from pathlib import Path

from nla.prompt_modes import (
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_MODE,
    LEGACY_PARQUET,
    activations_basename,
)

# Repo root (parent of nla/ package)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MODAL_VOLUME = "nla-cache"
MODAL_CACHE = "/cache"


def data_path(name: str) -> Path:
    return DATA_DIR / name


def persona_activations_file() -> str:
    return activations_basename(DEFAULT_LAYER, DEFAULT_PROMPT_MODE, DEFAULT_MODEL_ID)


def full_activations_tagged_file() -> str:
    return activations_basename(DEFAULT_LAYER, "full", DEFAULT_MODEL_ID)


DEFAULT_CSV = data_path("selfdescribe_400.csv")
DEFAULT_DESCRIPTIONS = data_path("descriptions.parquet")
DEFAULT_ACTIVATIONS_PERSONA = data_path(persona_activations_file())
DEFAULT_ACTIVATIONS_FULL_LEGACY = data_path(LEGACY_PARQUET)
DEFAULT_ACTIVATIONS_FULL_TAGGED = data_path(full_activations_tagged_file())
