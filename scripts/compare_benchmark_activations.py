#!/usr/bin/env python3
"""Compare raw cosine similarity across canonical activation spaces.

For each pair of run_ids (and within each run), reports the mean raw cosine
similarity of the layer-32 activation vectors. Useful for confirming the
"shared mean direction" inflation diagnostic at the dataset level.

Usage:
  uv run python scripts/compare_benchmark_activations.py
  uv run python scripts/compare_benchmark_activations.py --run-ids prism biosbias
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import RUN_IDS, local_activations_path, validate_run_id  # noqa: E402


def load(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    return np.array(df["activation_vector"].tolist(), dtype=np.float32)


def mean_cosine(A: np.ndarray, B: np.ndarray, exclude_diagonal: bool = False) -> float:
    """Mean pairwise cosine similarity between rows of A and rows of B."""
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    cos = A @ B.T
    if exclude_diagonal:
        np.fill_diagonal(cos, np.nan)
    return float(np.nanmean(cos))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-ids", nargs="+", default=list(RUN_IDS),
                   help="run_ids to compare (default: all four canonical runs)")
    args = p.parse_args()

    for rid in args.run_ids:
        validate_run_id(rid)

    mats: dict[str, np.ndarray] = {}
    for rid in args.run_ids:
        path = local_activations_path(rid)
        if not path.exists():
            print(f"WARN: skipping {rid} ({path} missing)")
            continue
        mats[rid] = load(path)
        print(f"{rid:14s} n={mats[rid].shape[0]:4d} d={mats[rid].shape[1]}")

    print()
    print("== within-run mean cosine (off-diagonal) ==")
    for rid, M in mats.items():
        print(f"  {rid:14s} {mean_cosine(M, M, exclude_diagonal=True):.6f}")

    print()
    print("== between-run mean cosine ==")
    for a, b in combinations(mats.keys(), 2):
        print(f"  {a:14s} vs {b:14s} {mean_cosine(mats[a], mats[b]):.6f}")


if __name__ == "__main__":
    main()
