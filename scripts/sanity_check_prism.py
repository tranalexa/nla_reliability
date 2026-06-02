#!/usr/bin/env python3
"""Local sanity check: load ~5 PRISM rows, print raw record + flattened user_prompt.

Run this before any Modal work to verify conversation serialization looks right.
No GPU or Modal account needed.

Usage:
  uv run python scripts/sanity_check_prism.py
  uv run python scripts/sanity_check_prism.py --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRISM_DATASET = "Transluce/PRISM-gender-Llama-3.1-8B-Instruct"


def _tentative_flatten(record: dict) -> tuple[str | None, str | None, str | None]:
    """Best-effort flatten + gender extraction using the likely field names.

    Returns (flattened_text, gt_attr_value, predicted_attr_value).
    Any element may be None if the field is not found — the caller prints the
    raw record so you can identify the real names and update ``nla/datasets.py``.

    gt_attr is the survey ground-truth gender written to the CSV attr column.
    predicted_attr is the model-predicted gender (attr in the source dataset),
    shown for comparison only.
    """
    conv = record.get("conversation") or record.get("messages") or record.get("turns")
    if conv is None:
        return None, None, None

    turns = list(conv)
    # Drop trailing assistant turn so activation is at the final user token.
    while turns and turns[-1].get("role", "").lower() in ("assistant", "model"):
        turns = turns[:-1]

    parts: list[str] = []
    for t in turns:
        role = t.get("role", "user").lower()
        content = t.get("content", "")
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")

    flat = "\n".join(parts)

    gt_attr = record.get("gt_attr")                  # ground-truth → goes into CSV
    predicted = record.get("attr") or record.get("label") or record.get("attribute")
    return flat, str(gt_attr) if gt_attr is not None else None, str(predicted) if predicted is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5, help="number of rows to inspect")
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="print full flattened text (default: truncate at 600 chars)",
    )
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading {PRISM_DATASET!r}  (streaming, first {args.n} rows)\n")
    ds = load_dataset(PRISM_DATASET, split="train", streaming=True)

    print("=== Dataset features ===")
    print(ds.features)
    print()

    for i, record in enumerate(ds.take(args.n)):
        print("=" * 72)
        print(f"Row {i}  —  keys: {list(record.keys())}")
        for k, v in record.items():
            if isinstance(v, list):
                print(f"  {k} ({len(v)} items):")
                for j, item in enumerate(v[:3]):
                    print(f"    [{j}] {item!r}")
                if len(v) > 3:
                    print(f"    ... and {len(v) - 3} more")
            else:
                print(f"  {k}: {v!r}")

        flat, gt_attr, predicted = _tentative_flatten(record)
        print()
        agree = " [agree]" if gt_attr == predicted else " [DISAGREE]"
        print(f"--- CSV columns ---")
        print(f"  attr (model-predicted, primary label) : {predicted!r}")
        print(f"  gt_attr (survey ground truth, extra)  : {gt_attr!r}{agree}")

        print(f"--- flattened user_prompt (what goes into the CSV) ---")
        if flat is None:
            print("  ERROR: could not find conversation field — inspect keys above and update")
            print("         nla/datasets.py _flatten_prism_conversation() accordingly.")
        else:
            limit = len(flat) if args.no_truncate else 600
            print(flat[:limit])
            if not args.no_truncate and len(flat) > limit:
                print(f"  ... [{len(flat)} chars total, pass --no-truncate to see all]")
        print()


if __name__ == "__main__":
    main()
