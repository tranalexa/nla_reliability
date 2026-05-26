#!/usr/bin/env python3
"""
Diagnostic: After removing the shared mean direction, do reconstructions match their
correct original activations better than wrong/random activations?

Background
----------
diagnose_cosine_inflation.py established that raw cosine similarities around 0.99
are inflated by a dominant shared mean direction. This script goes further: it loads
the raw reconstructed vectors (saved by reconstruct_scores.py --save-vectors) and
checks whether the reconstruction is DISCRIMINATIVE — i.e., whether recon_i points
more toward orig_i than toward a random orig_j.

The key question: after subtracting the shared mean direction, does matched
(orig_i vs recon_i) cosine exceed mismatched (orig_j vs recon_i, j≠i) cosine?

Metrics computed
----------------
  RAW:
    matched         — cos(orig_i,          recon_i)      for each (i, s)
    mismatched      — cos(orig_j,          recon_i)      random j ≠ i
    gap             — matched − mismatched  (per-pair and mean)

  CENTERED (subtract global mean of original activations from both sides):
    matched         — cos(orig_i − μ,      recon_i_hat − μ)
    mismatched      — cos(orig_j − μ,      recon_i_hat − μ)   random j ≠ i
    gap             — matched − mismatched  (per-pair and mean)

  where recon_i_hat = recon_i / ||recon_i||  (unit-normalized reconstruction)
  and   μ = mean of all original activation vectors.

Normalizing recon vectors to unit norm before centering ensures the mean subtraction
is geometrically symmetric: both sides start on the unit sphere before the shared
mean direction is removed.

Grouping
---------
There are N_SAMPLES reconstructions per activation (one per description/sample_idx).
All samples from the same activation_idx are treated as matched to the same orig.
For mismatched, each reconstruction is paired with a random orig from a DIFFERENT
activation_idx.

Prerequisites
--------------
1. Original activation vectors in data/:
     modal volume get nla-cache \\
       activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
       data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet

2. Reconstructed vectors (requires re-running Step 3 with --save-vectors):
     uv run modal run reconstruct_scores.py \\
       --activations activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
       --descriptions descriptions_prism.parquet \\
       --pairwise-out pairwise_consistency_prism.parquet \\
       --fidelity-out fidelity_scores_prism.parquet \\
       --save-vectors

     modal volume get nla-cache recon_vectors_prism.parquet \\
       data/recon_vectors_prism.parquet

Run:
  uv run python scripts/diagnose_reconstruction_fidelity.py
  uv run python scripts/diagnose_reconstruction_fidelity.py \\
    --activations data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
    --recon-vectors data/recon_vectors_prism.parquet \\
    --n-mismatch 5 --seed 42
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

DEFAULT_ACTIVATIONS = DATA / "activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet"
DEFAULT_RECON_VECTORS = DATA / "recon_vectors_prism.parquet"
DEFAULT_N_MISMATCH = 5
DEFAULT_SEED = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def load_activation_matrix(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    if "activation_idx" in df.columns:
        df = df.sort_values("activation_idx").reset_index(drop=True)
    return np.stack(df["activation_vector"].values).astype(np.float32)


def load_recon_vectors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"activation_idx", "sample_idx", "recon_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"recon_vectors parquet missing columns: {missing}")
    return df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)


def batched_cos_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between corresponding rows of A and B."""
    A_n = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B_n = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (A_n * B_n).sum(axis=1)


def summarize(label: str, vals: np.ndarray) -> dict:
    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    return {"label": label, "mean": float(vals.mean()), "std": float(vals.std()),
            "p5": float(p5), "p50": float(p50), "p95": float(p95), "n": len(vals)}


def print_table(rows: list[dict]) -> None:
    header = f"  {'Metric':<60} {'mean':>7} {'std':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'n':>6}"
    print(header)
    print("  " + "-" * 102)
    for r in rows:
        print(
            f"  {r['label']:<60} {r['mean']:>7.4f} {r['std']:>7.4f} "
            f"{r['p5']:>7.4f} {r['p50']:>7.4f} {r['p95']:>7.4f} {r['n']:>6}"
        )


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS,
                   help="Activation vectors parquet (Step 1 output)")
    p.add_argument("--recon-vectors", type=Path, default=DEFAULT_RECON_VECTORS,
                   help="Reconstructed vectors parquet (Step 3 --save-vectors output)")
    p.add_argument("--n-mismatch", type=int, default=DEFAULT_N_MISMATCH,
                   help="Mismatched originals to sample per reconstruction (default 5)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    missing = [f for f in [args.activations, args.recon_vectors] if not f.exists()]
    if missing:
        print("ERROR: missing input files:")
        for f in missing:
            print(f"  {f}")
        if args.recon_vectors in missing:
            print()
            print("Re-run Step 3 with --save-vectors to generate reconstructed vectors:")
            print("  uv run modal run reconstruct_scores.py \\")
            print("    --activations activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\")
            print("    --descriptions descriptions_prism.parquet \\")
            print("    --pairwise-out pairwise_consistency_prism.parquet \\")
            print("    --fidelity-out fidelity_scores_prism.parquet \\")
            print("    --save-vectors")
            print()
            print("Then pull the output:")
            print("  modal volume get nla-cache recon_vectors_prism.parquet \\")
            print("    data/recon_vectors_prism.parquet")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading activations   : {args.activations.name}")
    originals = load_activation_matrix(args.activations)
    N, D = originals.shape
    print(f"  shape: ({N}, {D})")

    print(f"Loading recon vectors : {args.recon_vectors.name}")
    recon_df = load_recon_vectors(args.recon_vectors)
    M = len(recon_df)
    print(f"  rows: {M}  "
          f"(activation_idx {recon_df['activation_idx'].min()}–{recon_df['activation_idx'].max()}, "
          f"sample_idx {recon_df['sample_idx'].min()}–{recon_df['sample_idx'].max()})")

    # Build recon matrix robustly — the recon_vector column may be stored as Python
    # lists, numpy arrays, or a pyarrow ChunkedArray depending on the parquet engine
    # and pandas/pyarrow version.  np.stack fails on ChunkedArrays; the fallback
    # iterates element-wise, which handles all three storage types.
    _col = recon_df["recon_vector"]
    try:
        recon_vecs_raw = np.stack(_col.values).astype(np.float32)
    except (TypeError, ValueError):
        recon_vecs_raw = np.array(
            [np.asarray(v, dtype=np.float32) for v in _col], dtype=np.float32
        )
    recon_act_idx = recon_df["activation_idx"].values  # (M,) — which orig each recon belongs to

    # ── Sanity checks ─────────────────────────────────────────────────────────
    section("SANITY CHECKS")

    # [1] Number of original activations
    print(f"  [1]  Original activations:              {N}")

    # [2] Number of reconstructed vectors (rows in parquet)
    print(f"  [2]  Reconstructed vectors (rows):      {M}")

    # [3] Unique activation_idx values present in the recon file
    _n_unique_act = recon_df["activation_idx"].nunique()
    print(f"  [3]  Unique activation_idx in recon:    {_n_unique_act}")

    # [4] Expected row count = unique activations × samples-per-activation (if uniform)
    _samples_per = recon_df.groupby("activation_idx")["sample_idx"].count()
    _min_s, _max_s = int(_samples_per.min()), int(_samples_per.max())
    if _min_s == _max_s:
        _expected = _n_unique_act * _min_s
        _match_str = "✓ matches" if _expected == M else f"✗ MISMATCH — got {M}"
        print(f"  [4]  Expected recon count:              "
              f"{_n_unique_act} × {_min_s} samples = {_expected}  {_match_str}")
    else:
        print(f"  [4]  Samples per activation:            "
              f"min={_min_s}  max={_max_s}  (uneven — no single expected count)")

    # [5] Every activation_idx in recon must map to a valid row in originals
    _orig_idx_set  = set(range(N))
    _recon_idx_set = set(recon_df["activation_idx"].unique().tolist())
    _extra  = _recon_idx_set - _orig_idx_set   # recon refers to originals that don't exist
    _absent = _orig_idx_set  - _recon_idx_set  # originals that have no reconstruction
    if _extra:
        _shown = sorted(_extra)[:10]
        print(f"  [5]  ✗ recon has activation_idx absent from originals: {_shown}"
              + ("  ..." if len(_extra) > 10 else ""))
    elif _absent:
        print(f"  [5]  ⚠ {len(_absent)} original(s) have no reconstruction")
    else:
        print(f"  [5]  ✓ all recon activation_idx values exist in originals; "
              f"all {N} originals covered")

    # [6] sample_idx must be unique within each activation_idx (no duplicate descriptions)
    _n_dups = int(recon_df.duplicated(subset=["activation_idx", "sample_idx"]).sum())
    if _n_dups:
        print(f"  [6]  ✗ {_n_dups} duplicate (activation_idx, sample_idx) pair(s) found")
    else:
        print(f"  [6]  ✓ sample_idx is unique within every activation_idx")

    # [7] Original activation norm statistics (Step 1 L2-normalizes, so expect ≈ 1.0)
    _orig_norms = np.linalg.norm(originals, axis=1)
    print(f"  [7]  Original activation norms:         "
          f"mean={_orig_norms.mean():.6f}  std={_orig_norms.std():.6f}  "
          f"min={_orig_norms.min():.6f}  max={_orig_norms.max():.6f}")

    # [8] Reconstructed vector norm statistics — raw output of critic.reconstruct(),
    #     before unit-normalization.  Typically O(10–1000) for Gemma-3 AR checkpoints.
    _recon_norms = np.linalg.norm(recon_vecs_raw, axis=1)
    print(f"  [8]  Recon vector norms (raw, pre-norm): "
          f"mean={_recon_norms.mean():.4f}  std={_recon_norms.std():.4f}  "
          f"min={_recon_norms.min():.4f}  max={_recon_norms.max():.4f}")

    # [9] Confirm recon_vecs_raw is a proper 2D float32 numpy array
    _ok_2d = recon_vecs_raw.ndim == 2
    print(f"  [9]  recon_vecs_raw loaded:              "
          f"shape={recon_vecs_raw.shape}  dtype={recon_vecs_raw.dtype}  "
          + ("✓ 2D float32 array" if _ok_2d else "✗ not 2D — check storage format"))
    if not _ok_2d:
        print("       Cannot continue with non-2D recon matrix.")
        sys.exit(1)

    # [10] Mismatch sampling strategy note
    print(f"  [10] Mismatch sampling: for each recon (activation_idx=i), draws a random "
          f"original with activation_idx j ≠ i — different activation identity, not "
          f"merely a different dataframe row index.")
    print()

    # Normalize recon vectors to unit sphere before centering so the mean subtraction
    # is geometrically symmetric with the (already unit-norm) original activations.
    recon_norms = np.linalg.norm(recon_vecs_raw, axis=1, keepdims=True).clip(1e-12)
    recon_vecs = recon_vecs_raw / recon_norms  # (M, D) unit-normalized

    # ── Global mean direction ──────────────────────────────────────────────────
    section("GLOBAL MEAN DIRECTION")
    orig_norms = np.linalg.norm(originals, axis=1)
    recon_norms_flat = recon_norms.ravel()
    mean_orig = originals.mean(axis=0)           # (D,) — the shared direction to remove
    mean_norm = float(np.linalg.norm(mean_orig))
    cos_with_mean = originals @ (mean_orig / mean_norm)

    print(f"  original norms :       mean={orig_norms.mean():.6f}  std={orig_norms.std():.6f}")
    print(f"  recon raw norms:       mean={recon_norms_flat.mean():.4f}  std={recon_norms_flat.std():.4f}")
    print(f"  ||global mean orig||:  {mean_norm:.4f}  "
          f"(1.0 = all originals identical direction; 0 = isotropic)")
    print(f"  cos(orig_i, mean_vec): mean={cos_with_mean.mean():.4f}  std={cos_with_mean.std():.4f}")

    orig_c = originals - mean_orig[None, :]        # centered originals (M, D)
    recon_c = recon_vecs - mean_orig[None, :]      # centered unit-normed recons (M, D)
    orig_c_norms = np.linalg.norm(orig_c, axis=1)
    recon_c_norms = np.linalg.norm(recon_c, axis=1)
    print(f"  centered orig norms:   mean={orig_c_norms.mean():.4f}  "
          f"({orig_c_norms.mean() / orig_norms.mean():.3f}× of original)")
    print(f"  centered recon norms:  mean={recon_c_norms.mean():.4f}  "
          f"({recon_c_norms.mean():.3f}× of unit sphere)")

    # ── Build matched arrays ───────────────────────────────────────────────────
    # matched_orig[m] = originals[recon_act_idx[m]]:  the true original for each recon
    matched_orig = originals[recon_act_idx]        # (M, D)
    matched_orig_c = orig_c[recon_act_idx]          # (M, D) centered

    # ── Build mismatched arrays (n_mismatch samples per recon) ────────────────
    # For each recon, sample n_mismatch random originals from DIFFERENT activation_idx.
    # Stack into (M * n_mismatch, D) arrays so the same batched_cos_sim can be used.
    all_mismatch_orig: list[np.ndarray] = []
    all_mismatch_recon: list[np.ndarray] = []
    all_mismatch_orig_c: list[np.ndarray] = []
    all_mismatch_recon_c: list[np.ndarray] = []

    for _ in range(args.n_mismatch):
        j = rng.integers(0, N, M)
        # Guarantee j != recon_act_idx for every row
        collisions = j == recon_act_idx
        j[collisions] = (j[collisions] + 1) % N
        all_mismatch_orig.append(originals[j])
        all_mismatch_recon.append(recon_vecs)
        all_mismatch_orig_c.append(orig_c[j])
        all_mismatch_recon_c.append(recon_c)

    mis_orig   = np.concatenate(all_mismatch_orig,   axis=0)   # (M*n_mismatch, D)
    mis_recon  = np.concatenate(all_mismatch_recon,  axis=0)
    mis_orig_c = np.concatenate(all_mismatch_orig_c, axis=0)
    mis_recon_c = np.concatenate(all_mismatch_recon_c, axis=0)

    # ── Compute all six metrics ────────────────────────────────────────────────
    raw_matched    = batched_cos_sim(matched_orig,   recon_vecs)   # (M,)
    raw_mismatch   = batched_cos_sim(mis_orig,       mis_recon)    # (M*n,)
    cen_matched    = batched_cos_sim(matched_orig_c, recon_c)      # (M,)
    cen_mismatch   = batched_cos_sim(mis_orig_c,     mis_recon_c)  # (M*n,)

    # Per-pair gap: tile matched to align with mismatched for element-wise subtraction
    raw_matched_tiled = np.tile(raw_matched, args.n_mismatch)      # (M*n,)
    cen_matched_tiled = np.tile(cen_matched, args.n_mismatch)
    raw_gap = raw_matched_tiled - raw_mismatch
    cen_gap = cen_matched_tiled - cen_mismatch

    # ── Summary table ─────────────────────────────────────────────────────────
    section("SUMMARY TABLE")
    rows = [
        summarize("RAW matched       [cos(orig_i,       recon_i)]",       raw_matched),
        summarize("RAW mismatched    [cos(orig_j,       recon_i), j≠i]",  raw_mismatch),
        summarize("RAW gap           [matched − mismatched]",              raw_gap),
        summarize("CENTERED matched  [cos(orig_i−μ,  recon_i_hat−μ)]",    cen_matched),
        summarize("CENTERED mismatch [cos(orig_j−μ,  recon_i_hat−μ)]",    cen_mismatch),
        summarize("CENTERED gap      [matched − mismatched]",              cen_gap),
    ]
    print_table(rows)

    # ── Interpretation ────────────────────────────────────────────────────────
    section("INTERPRETATION")

    raw_gap_mean = float(np.mean(raw_matched)) - float(np.mean(raw_mismatch))
    cen_gap_mean = float(np.mean(cen_matched)) - float(np.mean(cen_mismatch))
    inflation = raw_gap_mean - cen_gap_mean

    print(f"  N activations:             {N}")
    print(f"  M reconstructions:         {M}")
    print(f"  Mismatches per recon:      {args.n_mismatch}  ({len(raw_mismatch)} total mismatched pairs)")
    print(f"  ||global mean orig||:      {mean_norm:.4f}")
    print()
    print(f"  RAW gap     (matched − mismatched mean):      {raw_gap_mean:+.4f}")
    print(f"  CENTERED gap (matched − mismatched mean):     {cen_gap_mean:+.4f}")
    print(f"  Inflation from mean direction:                {inflation:+.4f}  "
          f"({'raw inflated' if inflation > 0.005 else 'minimal inflation'})")
    print()

    if cen_gap_mean > 0.05:
        print("  ✓  GENUINE CENTERED SIGNAL")
        print(f"     After removing the shared mean direction, reconstructions still match")
        print(f"     their correct originals by {cen_gap_mean:+.4f} more than wrong ones.")
        print(f"     The verbalizer captures activation-specific directional information.")
    elif cen_gap_mean > 0.01:
        print("  ⚡  WEAK CENTERED SIGNAL")
        print(f"     Small but positive centered gap: {cen_gap_mean:+.4f}.")
        print(f"     Some activation-specific direction is preserved after centering.")
    elif cen_gap_mean > -0.005:
        print("  ⚠  NO CENTERED SIGNAL")
        print(f"     After centering, matched ≈ mismatched (gap ≈ {cen_gap_mean:+.4f}).")
        print(f"     Reconstructions do not discriminate between activations once")
        print(f"     the shared mean direction is removed.")
    else:
        print("  ✗  NEGATIVE CENTERED SIGNAL")
        print(f"     Centered mismatched > centered matched (gap = {cen_gap_mean:+.4f}).")
        print(f"     Unexpected — warrants inspection.")

    if inflation > 0.01:
        print()
        print(f"  Note: {inflation:.4f} of the raw gap ({raw_gap_mean:.4f}) is explained by")
        print(f"  the shared mean direction, not by reconstruction quality.")


if __name__ == "__main__":
    main()
