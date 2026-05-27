#!/usr/bin/env python3
"""MMLU-specific consistency breakdown: within-item vs same-subject vs cross-subject.

The standard consistency diagnostic pools all between-item pairs regardless of
subject. For MMLU (4 subjects × 100 items) this inflates the between-item
baseline because same-subject pairs are structurally similar. This script
separates them.

Usage:
  uv run python scripts/mmlu_subject_consistency.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
N_SAMPLE = 26_400  # match within-item pair count


def main() -> None:
    csv_df = pd.read_csv(DATA / "mmlu_400.csv")
    act_df = pd.read_parquet(DATA / "activations_layer32_mmlu_gemma-3-12b-pt.parquet")
    rv_df  = pd.read_parquet(DATA / "recon_vectors_mmlu.parquet")

    subjects = sorted(csv_df["subject"].unique())
    subj_idxs = {s: csv_df[csv_df["subject"] == s]["item_idx"].values for s in subjects}

    print("Subjects:")
    for s, v in subj_idxs.items():
        print(f"  {s}: {len(v)} items")

    # Global mean of original activations (for centering)
    acts = np.array(act_df["activation_vector"].tolist(), dtype=np.float32)
    mu   = acts.mean(axis=0)

    # Load recon vectors, unit-norm → center → re-norm
    rv = rv_df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)
    R  = np.array(rv["recon_vector"].tolist(), dtype=np.float32)
    R  = R / np.linalg.norm(R, axis=1, keepdims=True)
    R -= mu
    R  = R / np.linalg.norm(R, axis=1, keepdims=True)
    R3 = R.reshape(400, 12, -1)  # (400, 12, 3840)

    # ── Within-item: all C(12,2) pairs per activation (vectorised) ───────────
    pair_idx = list(combinations(range(12), 2))
    s1 = np.array([p[0] for p in pair_idx])
    s2 = np.array([p[1] for p in pair_idx])
    A  = R3[:, s1, :]   # (400, 66, 3840)
    B  = R3[:, s2, :]
    wi = (A * B).sum(axis=2).ravel()  # (26400,)

    # ── Build exhaustive (i, j) index arrays for same- and cross-subject ─────
    same_ii, same_jj = [], []
    for idxs in subj_idxs.values():
        pairs = np.array(list(combinations(idxs, 2)))
        same_ii.append(pairs[:, 0])
        same_jj.append(pairs[:, 1])
    same_ii = np.concatenate(same_ii)
    same_jj = np.concatenate(same_jj)

    cross_ii, cross_jj = [], []
    for s1_subj, s2_subj in combinations(subjects, 2):
        g1, g2 = subj_idxs[s1_subj], subj_idxs[s2_subj]
        ii, jj = np.meshgrid(g1, g2, indexing="ij")
        cross_ii.append(ii.ravel())
        cross_jj.append(jj.ravel())
    cross_ii = np.concatenate(cross_ii)
    cross_jj = np.concatenate(cross_jj)

    print(f"\nAvailable same-subject item pairs:   {len(same_ii):,}")
    print(f"Available cross-subject item pairs:  {len(cross_ii):,}")

    def sample_cos(ia: np.ndarray, ja: np.ndarray, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        sel = rng.integers(len(ia), size=n)
        sa  = rng.integers(12, size=n)
        sb  = rng.integers(12, size=n)
        return (R3[ia[sel], sa] * R3[ja[sel], sb]).sum(axis=1)

    same_cos  = sample_cos(same_ii,  same_jj,  N_SAMPLE, seed=1)
    cross_cos = sample_cos(cross_ii, cross_jj, N_SAMPLE, seed=2)

    def show(arr: np.ndarray, label: str) -> None:
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        print(f"  {label}")
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"p5={p5:.4f}  p50={p50:.4f}  p95={p95:.4f}  n={len(arr):,}")

    print("\n" + "=" * 72)
    print("  MMLU CONSISTENCY BREAKDOWN")
    print("=" * 72)
    show(wi,        "Within-item              (same activation, 2 of 12 samples)")
    show(same_cos,  "Between-item same-subj   (diff activation, same subject)")
    show(cross_cos, "Between-item cross-subj  (diff activation, diff subject)")

    print()
    print(f"  Gap within − same-subj:         {wi.mean() - same_cos.mean():.4f}")
    print(f"  Gap within − cross-subj:        {wi.mean() - cross_cos.mean():.4f}")
    print(f"  Gap same-subj − cross-subj:     {same_cos.mean() - cross_cos.mean():.4f}")
    print()
    print("  Interpretation:")
    print(f"  The original between-item baseline (0.6504) blends same-subj")
    print(f"  ({same_cos.mean():.4f}) and cross-subj ({cross_cos.mean():.4f}) pairs.")
    print(f"  The true discriminability gap (within vs cross-subj) is "
          f"{wi.mean() - cross_cos.mean():.4f}.")


if __name__ == "__main__":
    main()
