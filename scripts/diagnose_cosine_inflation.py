#!/usr/bin/env python3
"""
Diagnostic: Is the ~0.99 cosine similarity real or inflated by a shared mean direction?

Background
----------
All vectors produced by Step 1 are L2-normalized (unit sphere). If every activation
points nearly in the *same direction* — i.e. a dominant "mean direction" exists —
then the cosine similarity between ANY two vectors will be close to 1 regardless of
their individual content. In that case, the fidelity score "original vs reconstruction"
tells you almost nothing about reconstruction quality.

The mean-centering test is the standard diagnostic: subtract the global mean activation
from every vector, then recompute. If cosines drop dramatically, the shared mean direction
was dominating. If they stay high, the signal is real.

What this script computes
-------------------------
Using original activation vectors (Step 1 output, L2-normalized) and the saved
cosine scores (Steps 3 outputs):

  RAW (no mean-centering):
    (A) Matched fidelity        — cos(original_i,  reconstruction_i)  [from fidelity parquet]
    (B) Within-activ recon-recon— cos(recon_{i,s1}, recon_{i,s2})     [from pairwise parquet]
    (C) Random orig-orig        — cos(original_i,  original_j), i≠j   [computed here]

  MEAN-CENTERED (subtract global mean activation from originals):
    (D) Centered orig-orig      — cos(centered_i, centered_j), i≠j    [computed here]

  MEAN DIRECTION STATS:
    (E) cos(original_i, global_mean_vec) — how aligned are originals with their mean?
        If this is ~1.0, the mean direction dominates.

Note: cross-activation reconstruction cosines (e.g. original_i vs recon_j, or
recon_i vs recon_j for i≠j) are not saved in the pipeline outputs — only cosine
values are stored, not vectors. To compute those, reconstruct_scores.py would need
to save the raw reconstructed vectors. This script covers the original-side
diagnostics and the saved reconstruction-side cosine distributions.

Run:
  # First pull activations if not in data/:
  modal volume get nla-cache activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
    data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet

  uv run python scripts/diagnose_cosine_inflation.py
  uv run python scripts/diagnose_cosine_inflation.py --n-pairs 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_ACTIVATIONS = DATA / "activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet"
DEFAULT_PAIRWISE = DATA / "pairwise_consistency_prism.parquet"
DEFAULT_FIDELITY = DATA / "fidelity_scores_prism.parquet"
DEFAULT_N_PAIRS = 5000
DEFAULT_SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def load_activation_matrix(path: Path) -> np.ndarray:
    """Load activation vectors from parquet into a float32 (N, D) array."""
    df = pd.read_parquet(path)
    return np.stack(df["activation_vector"].values).astype(np.float32)


def batched_cos_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between corresponding rows of A and B."""
    A_n = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B_n = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (A_n * B_n).sum(axis=1)


def summarize(label: str, vals: np.ndarray, comment: str = "") -> None:
    p = np.percentile(vals, [5, 25, 50, 75, 95])
    print(
        f"  {label:<52}  "
        f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
        f"[p5={p[0]:.3f}  p25={p[1]:.3f}  p50={p[2]:.3f}  p75={p[3]:.3f}  p95={p[4]:.3f}]"
    )
    if comment:
        print(f"    → {comment}")


def section(title: str) -> None:
    print()
    print(f"{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS,
                   help="Activation vectors parquet (Step 1 output)")
    p.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE,
                   help="Pairwise consistency parquet (Step 3 output)")
    p.add_argument("--fidelity", type=Path, default=DEFAULT_FIDELITY,
                   help="Fidelity scores parquet (Step 3 output)")
    p.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS,
                   help="Number of random pairs to sample for orig-orig comparisons")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    # ── check prerequisites ───────────────────────────────────────────────────
    missing = [f for f in [args.activations, args.pairwise, args.fidelity]
               if not f.exists()]
    if missing:
        print("ERROR: missing input files:")
        for f in missing:
            print(f"  {f}")
        if args.activations in missing:
            print()
            print("Pull activations from Modal:")
            print("  modal volume get nla-cache "
                  "activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet "
                  "data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"Loading activations from {args.activations.name} ...")
    originals = load_activation_matrix(args.activations)
    N, D = originals.shape
    print(f"  shape: ({N}, {D})")

    pairwise_df = pd.read_parquet(args.pairwise)
    fidelity_df = pd.read_parquet(args.fidelity)
    print(f"  pairwise rows: {len(pairwise_df)}")
    print(f"  fidelity rows: {len(fidelity_df)}")

    # ── norm sanity check ─────────────────────────────────────────────────────
    section("VECTOR NORMS (Step 1 L2-normalizes — expect norms ≈ 1.0)")
    norms = np.linalg.norm(originals, axis=1)
    print(f"  original norms  mean={norms.mean():.6f}  std={norms.std():.6f}  "
          f"min={norms.min():.6f}  max={norms.max():.6f}")
    if norms.mean() > 0.999:
        print("  ✓ L2-normalized as expected")
    else:
        print("  ⚠ Norms are not ~1.0 — vectors may not be L2-normalized")

    # ── mean direction ────────────────────────────────────────────────────────
    section("MEAN ACTIVATION DIRECTION")
    # The mean of L2-normalized vectors. If all vectors are nearly identical in
    # direction, ||mean_vec|| ≈ 1.0. If they point in many directions, it is small.
    mean_vec = originals.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    mean_unit = mean_vec / mean_norm

    # Cosine of each original with the global mean direction.
    # Since originals are unit vectors: cos = originals @ mean_unit
    cos_with_mean = originals @ mean_unit

    print(f"  ||global mean vec||                          = {mean_norm:.4f}")
    print(f"  (max possible = 1.0 if all vectors identical;")
    print(f"   close to 0 if vectors uniformly distributed)")
    print()
    print(f"  cos(original_i, mean_vec):  "
          f"mean={cos_with_mean.mean():.4f}  std={cos_with_mean.std():.4f}  "
          f"min={cos_with_mean.min():.4f}  max={cos_with_mean.max():.4f}")
    print()
    if cos_with_mean.mean() > 0.95:
        print("  ⚠ VERY HIGH alignment with mean direction.")
        print("    Almost all variance lies in a single background direction.")
        print("    Cosine similarity between any two vectors will be ≈ 1.0 by construction.")
    elif cos_with_mean.mean() > 0.80:
        print("  ⚡ MODERATE alignment — shared direction present but not dominant.")
    else:
        print("  ✓ Low alignment — activations are spread across many directions.")

    # ── random pair indices ───────────────────────────────────────────────────
    idx_i = rng.integers(0, N, args.n_pairs)
    idx_j = rng.integers(0, N, args.n_pairs)
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % N  # guarantee i ≠ j

    # ── RAW metrics ───────────────────────────────────────────────────────────
    section("RAW COSINE SIMILARITIES (no mean-centering)")

    # (A) Matched fidelity — loaded from Step 3 parquet
    # cos(L2-norm(original_i), L2-norm(reconstruction_{i, sample_j}))
    matched_cos = fidelity_df["fidelity_cos"].values.astype(np.float32)
    summarize(
        "(A) matched fidelity  [orig_i vs recon_i]",
        matched_cos,
        "Headline metric. High = genuine reconstruction OR shared background direction."
    )

    # (B) Within-activation reconstruction pairwise — loaded from Step 3 parquet
    # cos(L2-norm(recon_{i,s1}), L2-norm(recon_{i,s2})) for s1 ≠ s2, same i
    # This measures how consistent 12 stochastic descriptions of the same activation are.
    recon_recon_within = pairwise_df["cos_sim"].values.astype(np.float32)
    summarize(
        "(B) within-activ recon-recon  [pairwise consist.]",
        recon_recon_within,
        "Agreement between different descriptions of THE SAME activation."
    )

    # (C) Random original-original pairs — computed here
    # cos(original_i, original_j) for random i ≠ j
    # KEY DIAGNOSTIC: if this ≈ matched fidelity (A), cosine is not informative.
    orig_orig_raw = batched_cos_sim(originals[idx_i], originals[idx_j])
    summarize(
        "(C) random orig-orig  [baseline]",
        orig_orig_raw,
        "BASELINE. If ≈ (A): cosine is dominated by shared direction, not reconstruction quality."
    )

    # ── MEAN-CENTERED metrics ─────────────────────────────────────────────────
    section("MEAN-CENTERED COSINE SIMILARITIES")
    print("  Subtracting the global mean activation vector from every original.")
    print("  This removes the shared background direction before computing cosine.")
    print()

    originals_c = originals - mean_vec[None, :]

    norms_c = np.linalg.norm(originals_c, axis=1)
    print(f"  centered vector norms:  mean={norms_c.mean():.4f}  std={norms_c.std():.4f}")
    print(f"  (fraction of original norm remaining: {(norms_c.mean() / norms.mean()):.4f})")
    print(f"  If this fraction is small (e.g. <0.3), most information was in the mean direction.")
    print()

    # (D) Centered original-original pairs
    orig_orig_c = batched_cos_sim(originals_c[idx_i], originals_c[idx_j])
    summarize(
        "(D) centered random orig-orig",
        orig_orig_c,
        "Cosine between originals after removing shared direction. "
        "Low values = originals differ once mean is removed."
    )

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    section("SUMMARY TABLE")
    header = f"  {'Metric':<54} {'mean':>7} {'std':>7} {'p5':>7} {'p50':>7} {'p95':>7}"
    print(header)
    print("  " + "-" * 90)

    rows = [
        ("(A) RAW matched fidelity [orig_i vs recon_i]",      matched_cos),
        ("(B) RAW within-activ recon-recon [pairwise]",       recon_recon_within),
        ("(C) RAW random orig-orig [baseline]",               orig_orig_raw),
        ("(D) CENTERED random orig-orig",                     orig_orig_c),
    ]
    for label, vals in rows:
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        print(f"  {label:<54} {vals.mean():>7.4f} {vals.std():>7.4f} "
              f"{p5:>7.4f} {p50:>7.4f} {p95:>7.4f}")

    # ── INTERPRETATION ────────────────────────────────────────────────────────
    section("INTERPRETATION")

    gap_fidelity_vs_random = float(np.mean(matched_cos)) - float(np.mean(orig_orig_raw))
    drop_after_centering = float(np.mean(orig_orig_raw)) - float(np.mean(orig_orig_c))

    print(f"  Gap (A) − (C):  matched fidelity − random orig-orig  =  {gap_fidelity_vs_random:+.4f}")
    print(f"  Drop (C) → (D): raw orig-orig − centered orig-orig   =  {drop_after_centering:+.4f}")
    print(f"  Mean ||mean_vec||:                                       {mean_norm:.4f}")
    print(f"  Mean cos(orig, mean_vec):                               {cos_with_mean.mean():.4f}")
    print()

    if abs(gap_fidelity_vs_random) < 0.02:
        print("  ⚠ INFLATION LIKELY")
        print("    Matched fidelity (A) ≈ random orig-orig baseline (C).")
        print("    The 0.99 score is not informative — cosine similarity is dominated")
        print("    by a shared background direction that all vectors share.")
        print()
        print("  Recommended alternatives:")
        print("    1. Mean-centered cosine: subtract global mean before computing similarity.")
        print("    2. Angular distance (arccos of cosine) — more sensitive near 1.0.")
        print("    3. Residual cosine: project out the top-k PCA directions first.")
    elif abs(gap_fidelity_vs_random) < 0.05:
        print("  ⚡ PARTIAL INFLATION")
        print("    Small but real gap between matched fidelity and random baseline.")
        print("    Consider reporting mean-centered cosine alongside raw cosine.")
    else:
        print("  ✓ GENUINE SIGNAL")
        print("    Matched fidelity is meaningfully higher than the random baseline.")
        print("    The high cosine similarity reflects real reconstruction accuracy.")

    if drop_after_centering > 0.05:
        print()
        print("  ⚠ Large drop after centering confirms a dominant shared direction.")
        print("    Mean-centered cosine (D) is a more discriminative metric.")

    print()
    print("  Note: cross-activation reconstruction diagnostics")
    print("  (e.g. orig_i vs recon_j, or recon_i vs recon_j for i≠j) are not")
    print("  computable here because reconstruct_scores.py does not save the raw")
    print("  reconstructed vectors — only cosine values are stored. To enable")
    print("  those comparisons, add a 'save_vectors=True' option to Step 3.")


if __name__ == "__main__":
    main()
