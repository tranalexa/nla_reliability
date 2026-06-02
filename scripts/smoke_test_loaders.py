#!/usr/bin/env python3
"""Smoke test: validate dataset loaders without Gemma, Modal, or GPU.

Loads 5 items from each dataset and checks the common schema.
Requires a HuggingFace token in HF_TOKEN or HUGGING_FACE_HUB_TOKEN.

Usage:
  uv run python scripts/smoke_test_loaders.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.datasets import SUPPORTED_DATASETS, load_dataset_items  # noqa: E402

REQUIRED_COLUMNS = {"item_idx", "dataset", "prompt_text"}
N_ITEMS = 5
SEED = 42


def check_dataset(name: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {name}  (n_items={N_ITEMS}, seed={SEED})")
    print(f"{'=' * 60}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        df = load_dataset_items(name, n_items=N_ITEMS, seed=SEED, hf_token=token)
    except Exception as e:
        print(f"  FAIL  load_dataset_items raised: {e}")
        return False

    print(f"  columns : {list(df.columns)}")
    print(f"  shape   : {df.shape}")

    ok = True

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"  FAIL  missing columns: {sorted(missing)}")
        ok = False
    else:
        print(f"  OK    required columns present: {sorted(REQUIRED_COLUMNS)}")

    if "item_idx" in df.columns:
        expected = list(range(N_ITEMS))
        actual = df["item_idx"].tolist()
        if actual == expected:
            print(f"  OK    item_idx is 0..{N_ITEMS - 1}")
        else:
            print(f"  FAIL  item_idx expected {expected}, got {actual}")
            ok = False

    if "dataset" in df.columns:
        wrong = df[df["dataset"] != name]["dataset"].unique().tolist()
        if wrong:
            print(f"  FAIL  dataset column contains unexpected values: {wrong}")
            ok = False
        else:
            print(f"  OK    dataset column is {name!r} for all rows")

    if "prompt_text" in df.columns:
        empty = df["prompt_text"].apply(lambda t: not str(t).strip()).sum()
        if empty:
            print(f"  FAIL  {empty} row(s) have empty prompt_text")
            ok = False
        else:
            print(f"  OK    prompt_text is non-empty for all {N_ITEMS} rows")

        print(f"\n  --- prompt_text previews ---")
        for i, row in df.iterrows():
            preview = str(row["prompt_text"])[:120].replace("\n", " ")
            print(f"  [{row['item_idx']}] {preview}...")

    return ok


def main() -> None:
    results: dict[str, bool] = {}
    for ds in SUPPORTED_DATASETS:
        results[ds] = check_dataset(ds)

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    all_passed = True
    for ds, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {ds}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll dataset loaders OK.")
    else:
        print("\nSome loaders failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
