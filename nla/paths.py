"""Artifact path helpers for the NLA reliability pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MODAL_VOLUME = "nla-cache"
MODAL_CACHE = "/cache"

DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"
DEFAULT_LAYER = 32


def _model_slug(model_id: str) -> str:
    """e.g. google/gemma-3-12b-pt → gemma-3-12b-pt"""
    return model_id.replace("google/", "").replace(".", "-")


# ── filename helpers ──────────────────────────────────────────────────────────

def csv_filename(dataset: str, n: int) -> str:
    return f"{dataset}_{n}.csv"


def activations_filename(
    dataset: str,
    layer: int = DEFAULT_LAYER,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    return f"activations_layer{layer}_{dataset}_{_model_slug(model_id)}.parquet"


def descriptions_filename(dataset: str) -> str:
    return f"descriptions_{dataset}.parquet"


def pairwise_filename(dataset: str) -> str:
    return f"pairwise_consistency_{dataset}.parquet"


def fidelity_filename(dataset: str) -> str:
    return f"fidelity_scores_{dataset}.parquet"


def recon_vectors_filename(dataset: str) -> str:
    return f"recon_vectors_{dataset}.parquet"


# ── Modal volume paths (/cache/...) ───────────────────────────────────────────

def volume_csv_path(dataset: str, n: int) -> str:
    return f"{MODAL_CACHE}/{csv_filename(dataset, n)}"


def volume_activations_path(
    dataset: str,
    layer: int = DEFAULT_LAYER,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    return f"{MODAL_CACHE}/{activations_filename(dataset, layer, model_id)}"


def volume_descriptions_path(dataset: str) -> str:
    return f"{MODAL_CACHE}/{descriptions_filename(dataset)}"


def volume_pairwise_path(dataset: str) -> str:
    return f"{MODAL_CACHE}/{pairwise_filename(dataset)}"


def volume_fidelity_path(dataset: str) -> str:
    return f"{MODAL_CACHE}/{fidelity_filename(dataset)}"


def volume_recon_vectors_path(dataset: str) -> str:
    return f"{MODAL_CACHE}/{recon_vectors_filename(dataset)}"


# ── Local data/ paths ─────────────────────────────────────────────────────────

def data_path(name: str) -> Path:
    return DATA_DIR / name


def local_csv_path(dataset: str, n: int) -> Path:
    return DATA_DIR / csv_filename(dataset, n)


def local_activations_path(
    dataset: str,
    layer: int = DEFAULT_LAYER,
    model_id: str = DEFAULT_MODEL_ID,
) -> Path:
    return DATA_DIR / activations_filename(dataset, layer, model_id)


def local_descriptions_path(dataset: str) -> Path:
    return DATA_DIR / descriptions_filename(dataset)


def local_pairwise_path(dataset: str) -> Path:
    return DATA_DIR / pairwise_filename(dataset)


def local_fidelity_path(dataset: str) -> Path:
    return DATA_DIR / fidelity_filename(dataset)


def local_recon_vectors_path(dataset: str) -> Path:
    return DATA_DIR / recon_vectors_filename(dataset)


# ── Legacy constants (used by nla/activation_utils.py and train_linear_probes.py) ──

_LEGACY_LAYER = 32
_LEGACY_MODEL = "google/gemma-3-12b-pt"

DEFAULT_CSV = DATA_DIR / "selfdescribe_400.csv"
DEFAULT_DESCRIPTIONS = DATA_DIR / "descriptions.parquet"
DEFAULT_ACTIVATIONS_PERSONA = (
    DATA_DIR / f"activations_layer{_LEGACY_LAYER}_persona-only_{_model_slug(_LEGACY_MODEL)}.parquet"
)
DEFAULT_ACTIVATIONS_FULL_LEGACY = DATA_DIR / "activations_layer32.parquet"
DEFAULT_ACTIVATIONS_FULL_TAGGED = (
    DATA_DIR / f"activations_layer{_LEGACY_LAYER}_full_{_model_slug(_LEGACY_MODEL)}.parquet"
)
# These are kept for backward compatibility with scripts/train_linear_probes.py.
# They are not used by the main PRISM / BiasBios pipeline.

