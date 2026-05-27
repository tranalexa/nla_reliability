"""Dataset adapters for Step 1 (extract_activations.py).

Each adapter provides the dataset-specific bits needed to build the CSV that
glues the pipeline together:

  hf_slug      — HuggingFace dataset identifier
  dataset_tag  — appended to activations parquet filenames; "" keeps SelfDescribe
                 filenames unchanged (backward compat)
  n_default    — default row count; None = full dataset
  sample_seed  — shuffle seed used when sampling n_default rows
  csv_filename — distinct per-dataset CSV name keeps volume artefacts separate
  build_row    — maps one raw HF record → dict with at minimum
                 {user_prompt, attr_class, attr}; adapters may include extra
                 columns (PRISM adds gt_attr for validity analysis).

The required CSV columns {user_prompt, attr_class, attr} and integer row-index
alignment (activation_idx == row i) are the contracts shared with Steps 2 and 3.
Extra columns are ignored by all downstream consumers.
"""

from __future__ import annotations


class SelfDescribeAdapter:
    """Adapter for Transluce/SelfDescribe-Llama-3.1-8B-Instruct (default)."""

    hf_slug = "Transluce/SelfDescribe-Llama-3.1-8B-Instruct"
    dataset_tag = ""        # no tag → filenames are backward-compatible
    n_default: int | None = 400
    sample_seed = 42

    def csv_filename(self, n: int) -> str:
        # Fixed name regardless of n to keep backward compat with existing pipeline.
        return "selfdescribe_400.csv"

    def build_row(self, record: dict) -> dict:
        return {
            "user_prompt": record["user_prompt"],
            "attr_class":  record["attr_class"],
            "attr":        record["attr"],
        }

    def disagreement_report(self, ds) -> str | None:
        return None


def _flatten_prism_conversation(record: dict) -> str:
    """Serialize a PRISM multi-turn conversation to a single string.

    Serialization strategy (confirmed by scripts/sanity_check_prism.py):
      • Field name: "conversation", list of {role: str, content: str} dicts.
      • Drop any trailing assistant/model turns so the last token in the
        forward pass is the final user token — that is the activation we extract.
      • Each turn is prefixed with its role label so turn boundaries are
        explicit in the text: "User: <content>" / "Assistant: <content>".
      • Turns are joined with "\n".

    In persona-only mode, INFOBOX_SUFFIX is absent from PRISM text, so
    apply_prompt_mode("persona-only") falls through and returns this full
    flattened string unchanged — the correct behaviour for PRISM.
    """
    conv = record.get("conversation") or record.get("messages") or record.get("turns")
    if conv is None:
        raise KeyError(
            f"No conversation field found in PRISM record. "
            f"Available keys: {sorted(record.keys())}"
        )

    turns = list(conv)
    # Drop trailing assistant turns so activation is at the final user token.
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


class PRISMGenderAdapter:
    """Adapter for Transluce/PRISM-gender-Llama-3.1-8B-Instruct.

    PRISM is a multi-turn conversational benchmark.  Each record is one
    conversation with gender metadata.  We flatten all turns up to the last
    user message into a single user_prompt string (see _flatten_prism_conversation).

    Schema confirmed by scripts/sanity_check_prism.py:
      attr_class — 'Gender' in the source dataset
      attr       — Llama-3.1-8B-Instruct model-predicted gender; written as the
                   primary CSV label because it describes what the activation encodes
                   (right grouping variable for consistency/reliability analysis)
      gt_attr    — survey ground-truth gender; kept as a second CSV column so
                   validity analysis (grouped by actual gender) can use the same CSV

    dataset_tag="prism" is injected into activations parquet filenames so
    SelfDescribe and PRISM artefacts coexist on the volume without collision.
    """

    hf_slug = "Transluce/PRISM-gender-Llama-3.1-8B-Instruct"
    dataset_tag = "prism"
    n_default: int | None = None  # full dataset; row count printed at runtime
    sample_seed = 42

    def csv_filename(self, n: int) -> str:
        return f"prism_gender_{n}.csv"

    def build_row(self, record: dict) -> dict:
        return {
            "user_prompt": _flatten_prism_conversation(record),
            "attr_class":  str(record["attr_class"]),
            "attr":        str(record["attr"]),      # model-predicted: primary reliability label
            "gt_attr":     str(record["gt_attr"]),   # survey ground truth: validity analysis
        }

    def disagreement_report(self, ds) -> str | None:
        predicted = ds["attr"]       # Llama-3.1-8B-Instruct model-predicted label
        ground_truth = ds["gt_attr"] # survey ground truth
        n = len(ds)
        n_disagree = sum(p != g for p, g in zip(predicted, ground_truth))
        pct = 100.0 * n_disagree / n if n > 0 else 0.0
        return (
            f"PRISM gender: attr (model) disagrees with gt_attr (survey) "
            f"on {pct:.1f}% of {n} rows — both columns kept"
        )


ADAPTERS: dict[str, SelfDescribeAdapter | PRISMGenderAdapter] = {
    "selfdescribe": SelfDescribeAdapter(),
    "prism":        PRISMGenderAdapter(),
}


def get_adapter(name: str) -> SelfDescribeAdapter | PRISMGenderAdapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from: {sorted(ADAPTERS)}"
        )
