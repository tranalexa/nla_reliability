#!/usr/bin/env python3
"""Compare raw cosine similarity across PRISM, BiosBias, and MMLU activation spaces.

Usage:
  uv run python scripts/compare_benchmark_activations.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
PRISM_PATH    = DATA / "activations_layer32_prism_gemma-3-12b-pt.parquet"
BIOSBIAS_PATH = DATA / "activations_layer32_biosbias_gemma-3-12b-pt.parquet"
MMLU_PATH     = DATA / "activations_layer32_mmlu_gemma-3-12b-pt.parquet"


def load(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    return np.array(df["activation_vector"].tolist(), dtype=np.float32)


def mean_cosine(A: np.ndarray, B: np.ndarray, exclude_diagonal: bool = False) -> float:
    """Mean pairwise cosine similarity between rows of A and rows of B."""
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    cos = A @ B.T  # (n_A, n_B)
    if exclude_diagonal:
        np.fill_diagonal(cos, np.nan)
    return float(np.nanmean(cos))


P = load(PRISM_PATH)
B = load(BIOSBIAS_PATH)
M = load(MMLU_PATH)

print(f"PRISM within mean cosine:          {mean_cosine(P, P, exclude_diagonal=True):.6f}")
print(f"BiosBias within mean cosine:       {mean_cosine(B, B, exclude_diagonal=True):.6f}")
print(f"MMLU within mean cosine:           {mean_cosine(M, M, exclude_diagonal=True):.6f}")
print()
print(f"PRISM-BiosBias mean cosine:        {mean_cosine(P, B):.6f}")
print(f"PRISM-MMLU mean cosine:            {mean_cosine(P, M):.6f}")
print(f"BiosBias-MMLU mean cosine:         {mean_cosine(B, M):.6f}")
