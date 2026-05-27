#!/usr/bin/env python3
"""
Compute the full pairwise cosine similarity matrix among the 400 original
activation vectors and its mean-centered counterpart.

Run:
  uv run python scripts/activation_cosine_matrix.py
  uv run python scripts/activation_cosine_matrix.py --save
  uv run python scripts/activation_cosine_matrix.py \\
    --activations data/activations_layer32_prism_gemma-3-12b-pt.parquet \\
    --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DEFAULT_ACTIVATIONS = DATA / "activations_layer32_prism_gemma-3-12b-pt.parquet"


def off_diagonal(M: np.ndarray) -> np.ndarray:
    """Return the upper-triangle off-diagonal elements of a square matrix."""
    N = M.shape[0]
    idx = np.triu_indices(N, k=1)
    return M[idx]


def stats(label: str, vals: np.ndarray) -> None:
    p = np.percentile(vals, [5, 25, 50, 75, 95])
    print(f"  {label}")
    print(f"    mean={vals.mean():.6f}  std={vals.std():.6f}  "
          f"min={vals.min():.6f}  max={vals.max():.6f}")
    print(f"    p5={p[0]:.6f}  p25={p[1]:.6f}  p50={p[2]:.6f}  "
          f"p75={p[3]:.6f}  p95={p[4]:.6f}")


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    ap.add_argument("--save", action="store_true",
                    help="Save cosine matrices as .npy files alongside the parquet")
    args = ap.parse_args()

    if not args.activations.exists():
        raise FileNotFoundError(args.activations)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.activations.name} ...")
    df = pd.read_parquet(args.activations)
    X = np.stack(df["activation_vector"].values).astype(np.float32)
    N, D = X.shape
    print(f"  shape: ({N}, {D})")

    # ── L2-norm check ─────────────────────────────────────────────────────────
    section("VECTOR NORMS")
    norms = np.linalg.norm(X, axis=1)
    print(f"  mean={norms.mean():.8f}  std={norms.std():.8f}  "
          f"min={norms.min():.8f}  max={norms.max():.8f}")
    if norms.max() - norms.min() < 1e-4 and abs(norms.mean() - 1.0) < 1e-3:
        print("  ✓ All vectors are L2-normalized (norms ≈ 1.0)")
    else:
        print("  ⚠ Norms deviate from 1.0 — vectors are NOT L2-normalized")

    # ── Global mean vector ────────────────────────────────────────────────────
    section("GLOBAL MEAN VECTOR")
    mean_vec = X.mean(axis=0)          # (D,)
    mean_norm = float(np.linalg.norm(mean_vec))
    print(f"  ||mean_vec|| = {mean_norm:.6f}")
    print(f"  (1.0 = all vectors identical direction; ~0 = isotropic spread)")
    if mean_norm > 0.95:
        print("  ⚠ VERY HIGH — a dominant shared direction is present.")
        print("    Cosine similarity between any two vectors will be ≈ 1 by construction.")
    elif mean_norm > 0.5:
        print("  ⚡ MODERATE — shared direction present but not fully dominant.")
    else:
        print("  ✓ LOW — vectors are spread across many directions.")

    # ── Raw cosine similarity matrix ──────────────────────────────────────────
    section("RAW COSINE SIMILARITY MATRIX  (X @ X.T, off-diagonal)")
    # X is already L2-normalized, so X @ X.T is the exact cosine matrix.
    C_raw = X @ X.T                    # (N, N), diagonal = 1.0
    off_raw = off_diagonal(C_raw)      # N*(N-1)/2 values
    print(f"  Pairs compared: {len(off_raw):,}  ({N}×{N-1}//2)")
    stats("off-diagonal cosine similarities", off_raw)

    # ── Mean-centered cosine similarity matrix ────────────────────────────────
    section("MEAN-CENTERED COSINE SIMILARITY MATRIX  (off-diagonal)")
    X_c = X - mean_vec[None, :]        # subtract global mean from every row
    c_norms = np.linalg.norm(X_c, axis=1, keepdims=True).clip(1e-12)
    X_cn = X_c / c_norms              # re-normalize centered vectors to unit sphere
    C_cen = X_cn @ X_cn.T            # (N, N)
    off_cen = off_diagonal(C_cen)

    centered_norms = c_norms.ravel()
    print(f"  Centered vector norms:  mean={centered_norms.mean():.6f}  "
          f"std={centered_norms.std():.6f}  "
          f"min={centered_norms.min():.6f}  max={centered_norms.max():.6f}")
    print(f"  Fraction of original norm retained: "
          f"{centered_norms.mean() / norms.mean():.4f}")
    print(f"  (small fraction → most information was in the shared mean direction)")
    print()
    stats("off-diagonal centered cosine similarities", off_cen)

    # ── Side-by-side summary ──────────────────────────────────────────────────
    section("SUMMARY")
    header = f"  {'Metric':<20} {'raw':>10} {'centered':>10} {'drop':>10}"
    print(header)
    print("  " + "-" * 52)
    for label, raw_val, cen_val in [
        ("mean",  off_raw.mean(),                    off_cen.mean()),
        ("std",   off_raw.std(),                     off_cen.std()),
        ("min",   off_raw.min(),                     off_cen.min()),
        ("p5",    np.percentile(off_raw, 5),         np.percentile(off_cen, 5)),
        ("p25",   np.percentile(off_raw, 25),        np.percentile(off_cen, 25)),
        ("p50",   np.percentile(off_raw, 50),        np.percentile(off_cen, 50)),
        ("p75",   np.percentile(off_raw, 75),        np.percentile(off_cen, 75)),
        ("p95",   np.percentile(off_raw, 95),        np.percentile(off_cen, 95)),
        ("max",   off_raw.max(),                     off_cen.max()),
    ]:
        print(f"  {label:<20} {raw_val:>10.6f} {cen_val:>10.6f} {raw_val - cen_val:>+10.6f}")

    print()
    print(f"  ||mean_vec||:  {mean_norm:.6f}")
    drop = float(off_raw.mean()) - float(off_cen.mean())
    print(f"  Mean cosine drop after centering:  {drop:+.6f}")
    if drop > 0.05:
        print("  ⚠ Large drop — the shared mean direction was dominating raw cosine.")
    elif drop > 0.01:
        print("  ⚡ Moderate drop — shared direction present but not fully dominant.")
    else:
        print("  ✓ Small drop — raw cosines reflect genuine vector spread.")

    # ── Optional save ─────────────────────────────────────────────────────────
    if args.save:
        out_dir = args.activations.parent
        raw_path = out_dir / "original_activation_cosine_matrix.npy"
        cen_path = out_dir / "centered_activation_cosine_matrix.npy"
        np.save(raw_path, C_raw)
        np.save(cen_path, C_cen)
        print()
        print(f"  Saved: {raw_path.name}  ({C_raw.nbytes / 1e6:.1f} MB)")
        print(f"  Saved: {cen_path.name}  ({C_cen.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
