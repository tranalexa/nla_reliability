"""Dataset loading for the NLA reliability evaluation pipeline.

Provides load_dataset_items() with a common schema:

  item_idx    (int)  — zero-based index; aligned with activation_idx in all outputs
  dataset     (str)  — "prism" | "biosbias"
  prompt_text (str)  — text passed to Gemma for activation extraction

Additional metadata columns are dataset-specific and are not required by the core
pipeline (Steps 1–3). They are preserved in the CSV for downstream diagnostic grouping.

PRISM metadata:    gender (model-predicted), gt_gender (survey ground truth)
BiasBios metadata: profession, gender
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SUPPORTED_DATASETS: list[str] = ["prism", "biosbias"]

DEFAULT_N_ITEMS = 400
DEFAULT_SEED = 42


# ── PRISM ─────────────────────────────────────────────────────────────────────

PRISM_HF_SLUG = "Transluce/PRISM-gender-Llama-3.1-8B-Instruct"


def _flatten_prism_conversation(record: dict) -> str:
    """Serialize a PRISM multi-turn conversation to a single string.

    All turns up to and including the last user turn are kept; trailing
    assistant turns are dropped so the final token in the Gemma forward pass
    is the last user token.  Each turn is prefixed "User:" / "Assistant:".
    """
    conv = record.get("conversation") or record.get("messages") or record.get("turns")
    if conv is None:
        raise KeyError(
            f"No conversation field in PRISM record. Keys: {sorted(record.keys())}"
        )
    turns = list(conv)
    while turns and turns[-1].get("role", "").lower() in ("assistant", "model"):
        turns = turns[:-1]
    if not turns:
        raise ValueError("PRISM record has no user turns after stripping assistant tail")
    parts: list[str] = []
    for t in turns:
        role = t.get("role", "user").lower()
        content = t.get("content", "")
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def _load_prism(n_items: int, seed: int, *, hf_token: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset as hf_load  # noqa: PLC0415

    ds = hf_load(PRISM_HF_SLUG, split="train", token=hf_token)
    ds = ds.shuffle(seed=seed).select(range(n_items))
    rows = []
    for i, record in enumerate(ds):
        rows.append(
            {
                "item_idx": i,
                "dataset": "prism",
                "prompt_text": _flatten_prism_conversation(record),
                "gender": str(record.get("attr", "")),
                "gt_gender": str(record.get("gt_attr", "")),
            }
        )
    df = pd.DataFrame(rows)
    assert len(df) == n_items, f"Expected {n_items} PRISM items, got {len(df)}"
    return df


# ── Bias in Bios ──────────────────────────────────────────────────────────────

BIOSBIAS_HF_SLUG = "LabHC/bias_in_bios"

_PRONOUNS = [
    r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b", r"\bhis\b", r"\bhers\b",
    r"\bhimself\b", r"\bherself\b",
    r"\bhe'd\b", r"\bshe'd\b", r"\bhe's\b", r"\bshe's\b",
    r"\bhe'll\b", r"\bshe'll\b",
]
_HONORIFICS = [
    r"\bMr\.?\b", r"\bMrs\.?\b", r"\bMs\.?\b", r"\bMiss\b",
    r"\bSir\b", r"\bMadam\b", r"\bMa'am\b",
]
_GENDERED_NOUNS = [
    r"\bbrother\b", r"\bsister\b", r"\bson\b", r"\bdaughter\b",
    r"\bfather\b", r"\bmother\b", r"\bhusband\b", r"\bwife\b",
    r"\bboyfriend\b", r"\bgirlfriend\b", r"\buncle\b", r"\baunt\b",
    r"\bnephew\b", r"\bniece\b", r"\bgrandfather\b", r"\bgrandmother\b",
    r"\bgrandson\b", r"\bgranddaughter\b", r"\bwidower\b", r"\bwidow\b",
    r"\bactor\b", r"\bactress\b",
]

_SCRUB_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in _PRONOUNS + _HONORIFICS + _GENDERED_NOUNS
]
_PLACEHOLDER = "[PERSON]"

PROFESSION_LABELS: dict[int, str] = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher",
}


def _scrub_gender(text: str) -> str:
    """Remove explicit gender indicators from biography text."""
    for pat in _SCRUB_PATTERNS:
        text = pat.sub(_PLACEHOLDER, text)
    text = re.sub(r"(\[PERSON\]\s*){2,}", _PLACEHOLDER + " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _load_biosbias(n_items: int, seed: int, *, hf_token: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset as hf_load  # noqa: PLC0415

    ds = hf_load(BIOSBIAS_HF_SLUG, split="train", token=hf_token)
    df = ds.to_pandas()

    # Normalise integer-encoded gender (LabHC schema: 0=M, 1=F)
    if str(df["gender"].dtype).startswith("int"):
        df["gender"] = df["gender"].map({0: "M", 1: "F"})

    # Normalise integer-encoded profession
    if str(df["profession"].dtype).startswith("int"):
        df["profession"] = df["profession"].map(PROFESSION_LABELS)

    # hard_text already strips the opening name-sentence; fall back to text
    text_col = "hard_text" if "hard_text" in df.columns else "text"
    df["bio_text"] = df[text_col].astype(str)
    df = df[df["bio_text"].str.len() >= 20].reset_index(drop=True)
    df["bio_text"] = df["bio_text"].apply(_scrub_gender)

    # Stratified sample: half male / half female
    rng = np.random.default_rng(seed)
    half = n_items // 2
    male = df[df["gender"] == "M"].sample(half, random_state=int(rng.integers(1_000_000)))
    female = df[df["gender"] == "F"].sample(n_items - half, random_state=int(rng.integers(1_000_000)))
    sampled = (
        pd.concat([male, female])
        .sample(frac=1, random_state=int(rng.integers(1_000_000)))
        .reset_index(drop=True)
    )

    rows = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        rows.append(
            {
                "item_idx": i,
                "dataset": "biosbias",
                "prompt_text": str(row["bio_text"]),
                "profession": str(row.get("profession", "")),
                "gender": str(row.get("gender", "")),
            }
        )
    df_out = pd.DataFrame(rows)
    assert len(df_out) == n_items, f"Expected {n_items} BiasBios items, got {len(df_out)}"
    return df_out


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset_items(
    dataset: str,
    n_items: int = DEFAULT_N_ITEMS,
    seed: int = DEFAULT_SEED,
    *,
    hf_token: str | None = None,
) -> pd.DataFrame:
    """Load and sample n_items from the specified dataset.

    Returns DataFrame with columns: item_idx, dataset, prompt_text,
    plus dataset-specific metadata columns.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from: {SUPPORTED_DATASETS}"
        )
    if dataset == "prism":
        return _load_prism(n_items, seed, hf_token=hf_token)
    return _load_biosbias(n_items, seed, hf_token=hf_token)
