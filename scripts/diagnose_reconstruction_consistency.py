#!/usr/bin/env python3
"""
Diagnostic: Within-item reconstruction consistency.

For each of the 400 original activations, 12 descriptions were generated and
each was reconstructed back to a vector.  This script asks:

  Do the 12 reconstructions for the SAME activation agree with each other
  more than reconstructions from DIFFERENT activations do?

If yes (especially after mean-centering), the verbalization is reliably
capturing activation-specific directional information.  If the gap vanishes
after centering, consistency is an artefact of the shared mean direction
that all activations share.

Metrics
-------
  RAW within-item     — cos(recon_{i,s1},     recon_{i,s2})    same activation_idx
  RAW between-item    — cos(recon_{i,s},      recon_{j,t})     different activation_idx
  RAW gap             — within − between  (per matched pair)

  CENTERED within     — cos(rhat_{i,s1} − μ,  rhat_{i,s2} − μ)
  CENTERED between    — cos(rhat_{i,s}  − μ,  rhat_{j,t}  − μ)
  CENTERED gap        — within − between  (per matched pair)

  where  rhat = recon_vec / ‖recon_vec‖   (unit-normalise first so the
         mean-subtraction is geometrically symmetric with the L2-normalised
         original activations)
  and    μ = mean of all 400 original activation vectors

Within-item pair count: C(12, 2) = 66 pairs × 400 activations = 26 400.
Between-item pairs:     n_between × 26 400 sampled at random (activation_idx
                        guaranteed to differ).
Gap distribution:       N = 26 400 (each within pair is matched to exactly
                        one randomly chosen between pair for an element-wise
                        gap; the remaining between pairs extend the baseline
                        distribution reported separately).

Run:
  uv run python scripts/diagnose_reconstruction_consistency.py
  uv run python scripts/diagnose_reconstruction_consistency.py \\
    --activations data/activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
    --recon-vectors data/recon_vectors_prism.parquet \\
    --n-between 5 --seed 42
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

DEFAULT_ACTIVATIONS   = DATA / "activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet"
DEFAULT_RECON_VECTORS = DATA / "recon_vectors_prism.parquet"
DEFAULT_N_BETWEEN     = 5
DEFAULT_SEED          = 42
_CHUNK                = 10_000   # rows per chunk for memory-safe cosine computation


# ── helpers ───────────────────────────────────────────────────────────────────

def load_activation_matrix(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    if "activation_idx" in df.columns:
        df = df.sort_values("activation_idx").reset_index(drop=True)
    return np.stack(df["activation_vector"].values).astype(np.float32)


def load_recon_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = {"activation_idx", "sample_idx", "recon_vector"} - set(df.columns)
    if missing:
        raise ValueError(f"recon_vectors parquet missing columns: {missing}")
    return df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)


def stack_recon_column(series: pd.Series) -> np.ndarray:
    """Stack recon_vector column robustly regardless of storage type.

    Handles Python lists, numpy arrays, and pyarrow ChunkedArrays.
    """
    try:
        return np.stack(series.values).astype(np.float32)
    except (TypeError, ValueError):
        return np.array([np.asarray(v, dtype=np.float32) for v in series],
                        dtype=np.float32)


def unit_norm(A: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    return A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)


def chunked_dot(U: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """cos_sim(U[idx1[k]], U[idx2[k]]) for all k.

    U must be row-unit-normalised so the dot product equals cosine similarity.
    Computed in chunks of _CHUNK rows to bound peak memory at
    ~2 × _CHUNK × D × 4 bytes (~300 MB with _CHUNK=10 000, D=3840).
    """
    N = len(idx1)
    out = np.empty(N, dtype=np.float32)
    for s in range(0, N, _CHUNK):
        e = min(s + _CHUNK, N)
        out[s:e] = (U[idx1[s:e]] * U[idx2[s:e]]).sum(axis=1)
    return out


def summarize(label: str, vals: np.ndarray) -> dict:
    p = np.percentile(vals, [5, 25, 50, 75, 95])
    return {
        "label": label,
        "mean": float(vals.mean()), "std": float(vals.std()),
        "p5":  float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
        "p75": float(p[3]), "p95": float(p[4]),
        "n":   len(vals),
    }


def print_table(rows: list[dict]) -> None:
    hdr = (f"  {'Metric':<52} {'mean':>7} {'std':>7} "
           f"{'p5':>7} {'p25':>6} {'p50':>7} {'p75':>6} {'p95':>7} {'n':>8}")
    print(hdr)
    print("  " + "-" * 112)
    for r in rows:
        print(
            f"  {r['label']:<52} {r['mean']:>7.4f} {r['std']:>7.4f} "
            f"{r['p5']:>7.4f} {r['p25']:>6.4f} {r['p50']:>7.4f} "
            f"{r['p75']:>6.4f} {r['p95']:>7.4f} {r['n']:>8,}"
        )


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations",   type=Path, default=DEFAULT_ACTIVATIONS)
    ap.add_argument("--recon-vectors", type=Path, default=DEFAULT_RECON_VECTORS)
    ap.add_argument("--n-between", type=int, default=DEFAULT_N_BETWEEN,
                    help="Between-item pairs = n_between × within-item pairs (default 5)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    missing = [f for f in (args.activations, args.recon_vectors) if not f.exists()]
    if missing:
        print("ERROR: missing input files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading activations   : {args.activations.name}")
    originals = load_activation_matrix(args.activations)
    N, D = originals.shape
    print(f"  shape: ({N}, {D})")

    print(f"Loading recon vectors : {args.recon_vectors.name}")
    recon_df = load_recon_df(args.recon_vectors)
    M = len(recon_df)
    print(f"  rows: {M}")

    recon_vecs_raw = stack_recon_column(recon_df["recon_vector"])
    act_idxs       = recon_df["activation_idx"].values   # (M,)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    section("SANITY CHECKS")

    # [1] Original activations
    print(f"  [1]  Original activations:              {N}")

    # [2] Reconstructed vectors
    print(f"  [2]  Reconstructed vectors (rows):      {M}")

    # [3] Unique activation_idx values in recon
    _unique_acts = sorted(recon_df["activation_idx"].unique().tolist())
    _n_unique    = len(_unique_acts)
    print(f"  [3]  Unique activation_idx in recon:    {_n_unique}")

    # [4] Sample counts per activation_idx
    _counts   = recon_df.groupby("activation_idx")["sample_idx"].count()
    _cnt_mean = _counts.mean()
    _cnt_min  = int(_counts.min())
    _cnt_max  = int(_counts.max())
    print(f"  [4]  Samples per activation_idx:        "
          f"mean={_cnt_mean:.1f}  min={_cnt_min}  max={_cnt_max}")

    # [5] Expected within-pair count
    from math import comb
    _pairs_per = comb(_cnt_min, 2) if _cnt_min == _cnt_max else None
    if _pairs_per is not None:
        _expected_within = _n_unique * _pairs_per
        print(f"  [5]  Expected within-item pairs:        "
              f"{_n_unique} × C({_cnt_min},2) = {_n_unique} × {_pairs_per} = {_expected_within:,}")
    else:
        _expected_within = sum(comb(int(c), 2) for c in _counts)
        print(f"  [5]  Expected within-item pairs:        "
              f"{_expected_within:,}  (uneven sample counts)")

    # [6] Recon vector raw norm statistics
    _raw_norms = np.linalg.norm(recon_vecs_raw, axis=1)
    print(f"  [6]  Recon vector norms (raw):          "
          f"mean={_raw_norms.mean():.4f}  std={_raw_norms.std():.4f}  "
          f"min={_raw_norms.min():.4f}  max={_raw_norms.max():.4f}")

    # [7] Confirm 2D float32 array
    _ok = recon_vecs_raw.ndim == 2
    print(f"  [7]  recon_vecs_raw:                    "
          f"shape={recon_vecs_raw.shape}  dtype={recon_vecs_raw.dtype}  "
          + ("✓ 2D float32" if _ok else "✗ not 2D — check storage format"))
    if not _ok:
        print("       Cannot continue."); sys.exit(1)

    # [8] All recon activation_idx values must exist in originals
    _orig_set  = set(range(N))
    _recon_set = set(_unique_acts)
    _extra     = _recon_set - _orig_set
    _absent    = _orig_set  - _recon_set
    if _extra:
        print(f"  [8]  ✗ recon activation_idx absent from originals: "
              f"{sorted(_extra)[:10]}{'...' if len(_extra) > 10 else ''}")
    elif _absent:
        print(f"  [8]  ⚠ {len(_absent)} original(s) have no reconstruction")
    else:
        print(f"  [8]  ✓ all recon activation_idx values exist in originals; "
              f"all {N} originals covered")
    print()

    # ── Prepare vectors ───────────────────────────────────────────────────────
    mean_orig    = originals.mean(axis=0)              # (D,) shared direction to remove
    mean_norm    = float(np.linalg.norm(mean_orig))

    recon_hat     = unit_norm(recon_vecs_raw)           # (M, D)  unit-normalised
    recon_cen     = recon_hat - mean_orig[None, :]      # (M, D)  centred
    recon_cen_hat = unit_norm(recon_cen)                # (M, D)  unit-normalised centred

    # ── Within-item pairwise cosines ──────────────────────────────────────────
    section("WITHIN-ITEM PAIRWISE COSINES")
    raw_within_list: list[float] = []
    cen_within_list: list[float] = []

    for act in _unique_acts:
        rows = np.where(act_idxs == act)[0]
        if len(rows) < 2:
            continue
        # Cosine matrix via dot product of unit-normed rows: (n_s, n_s)
        mat_raw = recon_hat[rows]     @ recon_hat[rows].T
        mat_cen = recon_cen_hat[rows] @ recon_cen_hat[rows].T
        tri     = np.triu_indices(len(rows), k=1)
        raw_within_list.extend(mat_raw[tri].tolist())
        cen_within_list.extend(mat_cen[tri].tolist())

    raw_within = np.array(raw_within_list, dtype=np.float32)
    cen_within = np.array(cen_within_list, dtype=np.float32)
    N_within   = len(raw_within)
    print(f"  Within-item pairs computed: {N_within:,}  "
          f"(expected {_expected_within:,})")

    # ── Between-item pairs ────────────────────────────────────────────────────
    section("BETWEEN-ITEM PAIRS  (different activation_idx)")
    N_between = args.n_between * N_within
    print(f"  Sampling {N_between:,} pairs  "
          f"({args.n_between} × {N_within:,} within-item pairs)  …")

    idx1 = rng.integers(0, M, N_between)
    idx2 = rng.integers(0, M, N_between)
    # Iteratively resample collisions (E[collisions] ≈ 0.25 % per pass)
    for _pass in range(10):
        same = act_idxs[idx1] == act_idxs[idx2]
        if not same.any():
            break
        idx2[same] = rng.integers(0, M, int(same.sum()))
    # Final deterministic fallback for any residual collisions
    still_same = act_idxs[idx1] == act_idxs[idx2]
    if still_same.any():
        idx2[still_same] = (idx2[still_same] + M // 2) % M
    n_residual = int((act_idxs[idx1] == act_idxs[idx2]).sum())
    print(f"  Same-activation residual after sampling: {n_residual}  "
          f"{'✓' if n_residual == 0 else '⚠ small — will not materially affect results'}")

    print(f"  Computing raw between cosines …")
    raw_between = chunked_dot(recon_hat,     idx1, idx2)
    print(f"  Computing centered between cosines …")
    cen_between = chunked_dot(recon_cen_hat, idx1, idx2)

    # ── Gap (matched 1-to-1 using first N_within between pairs) ───────────────
    # Each within pair[k] is matched to between pair[k] for the gap distribution.
    # The remaining (n_between - 1) × N_within between pairs extend the baseline.
    raw_gap = raw_within - raw_between[:N_within]
    cen_gap = cen_within - cen_between[:N_within]

    # ── Summary table ─────────────────────────────────────────────────────────
    section("SUMMARY TABLE")
    print(f"  mean_orig norm: {mean_norm:.4f}  "
          f"(fraction of variance in shared direction ≈ {mean_norm**2:.4f})")
    print()
    rows = [
        summarize("RAW within-item  [cos(recon_i_s1, recon_i_s2)]",  raw_within),
        summarize("RAW between-item [cos(recon_i_s,  recon_j_t)]",   raw_between),
        summarize("RAW gap          [within − between, per pair]",    raw_gap),
        summarize("CEN within-item  [cos(cen_i_s1, cen_i_s2)]",      cen_within),
        summarize("CEN between-item [cos(cen_i_s,  cen_j_t)]",       cen_between),
        summarize("CEN gap          [within − between, per pair]",    cen_gap),
    ]
    print_table(rows)

    # ── Interpretation ────────────────────────────────────────────────────────
    section("INTERPRETATION")

    raw_w_mean   = float(raw_within.mean())
    raw_b_mean   = float(raw_between.mean())
    raw_gap_mean = raw_w_mean - raw_b_mean
    cen_w_mean   = float(cen_within.mean())
    cen_b_mean   = float(cen_between.mean())
    cen_gap_mean = cen_w_mean - cen_b_mean
    inflation    = raw_gap_mean - cen_gap_mean

    print(f"  mean_orig norm:         {mean_norm:.4f}")
    print(f"  RAW  within mean:       {raw_w_mean:+.4f}")
    print(f"  RAW  between mean:      {raw_b_mean:+.4f}")
    print(f"  RAW  gap (w − b):       {raw_gap_mean:+.4f}")
    print(f"  CEN  within mean:       {cen_w_mean:+.4f}")
    print(f"  CEN  between mean:      {cen_b_mean:+.4f}")
    print(f"  CEN  gap (w − b):       {cen_gap_mean:+.4f}")
    print(f"  Inflation (raw − cen):  {inflation:+.4f}")
    print()

    # Inflation diagnosis
    if raw_w_mean > 0.97 and raw_b_mean > 0.97:
        print("  ⚠  RAW INFLATION: both within and between cosines are near 0.99.")
        print("     The shared mean direction dominates raw cosine — "
              "raw gap alone is not informative.")
    elif raw_gap_mean > 0.01:
        print(f"  Raw gap {raw_gap_mean:+.4f}: some within-item structure even in raw space.")

    # Main verdict
    print()
    if cen_gap_mean > 0.05:
        print("  ✓  STRONG CENTERED CONSISTENCY")
        print(f"     After removing the shared mean direction, within-item recon-recon")
        print(f"     cosine ({cen_w_mean:.4f}) exceeds between-item ({cen_b_mean:.4f}) "
              f"by {cen_gap_mean:+.4f}.")
        print(f"     The 12 reconstructions for the same activation reliably agree,")
        print(f"     and that agreement is specific to the activation's direction.")
    elif cen_gap_mean > 0.01:
        print("  ⚡  WEAK CENTERED CONSISTENCY")
        print(f"     Small but positive centered gap: {cen_gap_mean:+.4f}.")
        print(f"     Some activation-specific consistency, but the effect is modest.")
    elif cen_gap_mean > -0.005:
        print("  ⚠  NO CENTERED CONSISTENCY")
        print(f"     Centered within ≈ centered between (gap ≈ {cen_gap_mean:+.4f}).")
        print(f"     After removing the shared direction, the 12 reconstructions for")
        print(f"     the same activation are no more similar to each other than")
        print(f"     reconstructions from completely different activations.")
    else:
        print(f"  ✗  NEGATIVE CENTERED GAP ({cen_gap_mean:+.4f}) — warrants inspection.")

    if inflation > 0.01:
        print()
        print(f"  Note: {inflation:.4f} of the raw gap is explained by the shared mean")
        print(f"  direction, not by activation-specific content.")


if __name__ == "__main__":
    main()
