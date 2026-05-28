#!/usr/bin/env python3
"""Mean pairwise within-item text consistency per activation, using MPNet embeddings.

Reads ``data/runs/<run_id>/descriptions_<dataset>.parquet`` (Step 2 output) and
embeds every row once with ``sentence-transformers/all-mpnet-base-v2`` in a
single batched pass. For each activation, averages C(K,2) pairwise cosine
similarities between its K description embeddings.

Output: ``data/runs/<run_id>/text_consistency_mpnet_<dataset>.parquet`` with
one row per activation (``activation_idx``, ``mean_pairwise_text_cosine``).

Usage:
  uv run python scripts/compute_text_consistency.py --run-id prism
  uv run python scripts/compute_text_consistency.py --all-runs
  uv run python scripts/compute_text_consistency.py --run-id mmlu_choice --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import (  # noqa: E402
    RUN_IDS,
    local_descriptions_path,
    local_text_consistency_mpnet_path,
)

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_one(
    *,
    run_id: str,
    descriptions_path: Path,
    output_path: Path,
    model_id: str,
    batch_size: int,
    device: str,
) -> None:
    df = pd.read_parquet(descriptions_path)
    required = {"activation_idx", "sample_idx", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"descriptions parquet missing columns {missing}")

    df = df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)
    activations = np.sort(df["activation_idx"].unique())
    N = len(activations)
    K = df["sample_idx"].nunique()
    if len(df) != N * K:
        raise ValueError(f"expected {N}*{K}={N * K} rows, got {len(df)}")
    counts = df.groupby("activation_idx").size()
    if not (counts == K).all():
        bad = counts[counts != K]
        raise ValueError(f"{len(bad)} activations do not have exactly K={K} samples")

    resolved_device = resolve_device(device)
    print(f"\n=== {run_id} ===")
    print(f"  descriptions: {descriptions_path}  ({len(df)} rows, N={N}, K={K})")
    print(f"  output:       {output_path}")
    print(f"  model:        {model_id}")
    print(f"  device:       {resolved_device}")
    print(f"  batch-size:   {batch_size}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=resolved_device)
    t0 = time.time()
    emb = model.encode(
        df["description"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  encoded {emb.shape[0]} descriptions in {time.time() - t0:.1f}s -> shape {emb.shape}")

    D = emb.shape[1]
    E = emb.reshape(N, K, D).astype(np.float32, copy=False)
    sims = np.einsum("nkd,nld->nkl", E, E)
    iu = np.triu_indices(K, k=1)
    pair_sims = sims[:, iu[0], iu[1]]
    mean_per_act = pair_sims.mean(axis=1)

    out = pd.DataFrame({
        "activation_idx": activations.astype(np.int64),
        "mean_pairwise_text_cosine": mean_per_act.astype(np.float32),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    v = mean_per_act
    print(f"\n  mean_pairwise_text_cosine distribution (per activation, N={N}):")
    print(f"    mean   {v.mean():.4f}")
    print(f"    std    {v.std(ddof=1):.4f}")
    print(f"    min    {v.min():.4f}")
    print(f"    median {np.median(v):.4f}")
    print(f"    max    {v.max():.4f}")
    print(f"  wrote {len(out)} rows -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-id",
        default=None,
        choices=list(RUN_IDS),
        help="run to embed",
    )
    p.add_argument("--all-runs", action="store_true", help="embed every canonical run sequentially")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"sentence-transformers model id (default: {DEFAULT_MODEL})")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (default: auto)")
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="with --all-runs, skip runs whose descriptions parquet is absent",
    )
    args = p.parse_args()

    if not args.all_runs and args.run_id is None:
        p.error("either --run-id or --all-runs is required")

    run_ids = list(RUN_IDS) if args.all_runs else [args.run_id]
    skip_missing = args.skip_missing or args.all_runs
    failed = False

    for r in run_ids:
        desc_path = local_descriptions_path(r)
        out_path = local_text_consistency_mpnet_path(r)
        if not desc_path.exists():
            msg = f"missing {desc_path}"
            if skip_missing:
                print(f"\n[skip] {r}: {msg}")
                continue
            print(f"ERROR [{r}]: {msg}", file=sys.stderr)
            print(
                f"  Pull first: uv run python scripts/pull_from_modal.py --run-id {r} --only descriptions",
                file=sys.stderr,
            )
            failed = True
            continue

        try:
            compute_one(
                run_id=r,
                descriptions_path=desc_path,
                output_path=out_path,
                model_id=args.model,
                batch_size=args.batch_size,
                device=args.device,
            )
        except ValueError as e:
            print(f"ERROR [{r}]: {e}", file=sys.stderr)
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
