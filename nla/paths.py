"""Artifact path helpers for the NLA reliability pipeline.

Each pipeline run is identified by a `run_id`. Four canonical runs are supported:

  prism          PRISM gender conversations (Transluce/PRISM-gender-Llama-3.1-8B-Instruct).
  biosbias       Bias-in-Bios biographies (LabHC/bias_in_bios), gender-scrubbed.
  mmlu_choice    MMLU question + A-D choices inside prompt_text.
  mmlu_nochoice  MMLU question stem only; choices kept as CSV metadata.

Layouts:

  Local:    data/runs/<run_id>/<filename>          (after `pull_from_modal.py --run-id …`)
  Modal:    /cache/runs/<run_id>/<filename>        (written by extract/sample/reconstruct
                                                    on the `nla-cache` volume)

The filename inside the per-run subdir is keyed by the underlying dataset
("prism", "biosbias", "mmlu") rather than the run_id. Disambiguation between
the two MMLU runs is provided by the directory, not the filename.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"

MODAL_VOLUME = "nla-cache"
MODAL_CACHE = "/cache"
MODAL_RUNS = f"{MODAL_CACHE}/runs"

DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"
DEFAULT_LAYER = 32

# ── Run-id taxonomy ───────────────────────────────────────────────────────────

#: Canonical run identifiers, in display order.
RUN_IDS: tuple[str, ...] = ("prism", "biosbias", "mmlu_choice", "mmlu_nochoice")

#: Maps run_id -> underlying dataset name. The dataset name is what appears in
#: filenames (e.g. ``activations_layer32_mmlu_*``); the run_id is what appears
#: in directory paths (e.g. ``data/runs/mmlu_choice/``).
RUN_TO_DATASET: dict[str, str] = {
    "prism": "prism",
    "biosbias": "biosbias",
    "mmlu_choice": "mmlu",
    "mmlu_nochoice": "mmlu",
}

#: MMLU runs differ only in how prompt_text is built. "with_choices" puts the
#: four A-D options inside prompt_text; "question_only" sends the question stem
#: alone (the original setup from the early experiments).
MMLU_PROMPT_MODE: dict[str, str] = {
    "mmlu_choice": "with_choices",
    "mmlu_nochoice": "question_only",
}


def validate_run_id(run_id: str) -> None:
    """Raise ValueError if ``run_id`` is not one of the four canonical runs."""
    if run_id not in RUN_IDS:
        raise ValueError(
            f"unknown run_id {run_id!r}; expected one of {RUN_IDS}"
        )


def dataset_for_run(run_id: str) -> str:
    """Return the underlying dataset name ("prism" | "biosbias" | "mmlu") for a run_id."""
    validate_run_id(run_id)
    return RUN_TO_DATASET[run_id]


def mmlu_prompt_mode_for_run(run_id: str) -> str | None:
    """Return "with_choices" or "question_only" for MMLU runs, else None."""
    return MMLU_PROMPT_MODE.get(run_id)


def _model_slug(model_id: str) -> str:
    """e.g. google/gemma-3-12b-pt -> gemma-3-12b-pt"""
    return model_id.replace("google/", "").replace(".", "-")


# ── Filename helpers (same per run; directory disambiguates) ──────────────────

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


def text_consistency_mpnet_filename(dataset: str) -> str:
    return f"text_consistency_mpnet_{dataset}.parquet"


def text_between_item_mpnet_filename(dataset: str) -> str:
    return f"text_between_item_mpnet_{dataset}.parquet"


# ── Modal volume paths (write side) ───────────────────────────────────────────

def volume_run_dir(run_id: str) -> str:
    """Path of the per-run subdir on the Modal volume (e.g. /cache/runs/prism)."""
    validate_run_id(run_id)
    return f"{MODAL_RUNS}/{run_id}"


def volume_csv_path(run_id: str, n: int) -> str:
    ds = dataset_for_run(run_id)
    return f"{volume_run_dir(run_id)}/{csv_filename(ds, n)}"


def volume_activations_path(
    run_id: str,
    layer: int = DEFAULT_LAYER,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    ds = dataset_for_run(run_id)
    return f"{volume_run_dir(run_id)}/{activations_filename(ds, layer, model_id)}"


def volume_descriptions_path(run_id: str) -> str:
    return f"{volume_run_dir(run_id)}/{descriptions_filename(dataset_for_run(run_id))}"


def volume_pairwise_path(run_id: str) -> str:
    return f"{volume_run_dir(run_id)}/{pairwise_filename(dataset_for_run(run_id))}"


def volume_fidelity_path(run_id: str) -> str:
    return f"{volume_run_dir(run_id)}/{fidelity_filename(dataset_for_run(run_id))}"


def volume_recon_vectors_path(run_id: str) -> str:
    return f"{volume_run_dir(run_id)}/{recon_vectors_filename(dataset_for_run(run_id))}"


# ── Local paths (read side) ───────────────────────────────────────────────────

def run_dir(run_id: str) -> Path:
    """Local directory for a single run's artifacts."""
    validate_run_id(run_id)
    return RUNS_DIR / run_id


def data_path(name: str) -> Path:
    """A path inside ``data/`` (not inside a run subdir). Used for shared inputs."""
    return DATA_DIR / name


def local_csv_path(run_id: str, n: int = 400) -> Path:
    return run_dir(run_id) / csv_filename(dataset_for_run(run_id), n)


def local_activations_path(
    run_id: str,
    layer: int = DEFAULT_LAYER,
    model_id: str = DEFAULT_MODEL_ID,
) -> Path:
    return run_dir(run_id) / activations_filename(dataset_for_run(run_id), layer, model_id)


def local_descriptions_path(run_id: str) -> Path:
    return run_dir(run_id) / descriptions_filename(dataset_for_run(run_id))


def local_pairwise_path(run_id: str) -> Path:
    return run_dir(run_id) / pairwise_filename(dataset_for_run(run_id))


def local_fidelity_path(run_id: str) -> Path:
    return run_dir(run_id) / fidelity_filename(dataset_for_run(run_id))


def local_recon_vectors_path(run_id: str) -> Path:
    return run_dir(run_id) / recon_vectors_filename(dataset_for_run(run_id))


def local_text_consistency_mpnet_path(run_id: str) -> Path:
    return run_dir(run_id) / text_consistency_mpnet_filename(dataset_for_run(run_id))


def local_text_between_item_mpnet_path(run_id: str) -> Path:
    return run_dir(run_id) / text_between_item_mpnet_filename(dataset_for_run(run_id))
