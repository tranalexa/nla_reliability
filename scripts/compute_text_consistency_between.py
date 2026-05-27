#!/usr/bin/env python3
"""Between-item text consistency (MPNet) baseline.

For each activation we already compute the mean **within-item** pairwise MPNet cosine
across its 12 descriptions (`scripts/compute_text_consistency.py`).

This script computes the **between-item** baseline: for each activation i, the mean
cosine between any of its 12 description embeddings and any description embedding from
a *different* activation j != i in the same benchmark.

Interpretation:
- within − between ≈ how much MORE the 12 paraphrases of one item agree than
  random descriptions from the same benchmark. If within ≈ between, the high
  within-item number is just "this benchmark talks about similar stuff" rather than
  "AV is paraphrase-stable for each item."

Output: parquet with one row per activation, plus a stdout summary that pairs this
with the existing within-item numbers for the same descriptions file.

Examples:
  uv run python scripts/compute_text_consistency_between.py --dataset prism
  uv run python scripts/compute_text_consistency_between.py --dataset biosbias
  uv run python scripts/compute_text_consistency_between.py --dataset mmlu
  uv run python scripts/compute_text_consistency_between.py \\
      --dataset mmlu \\
      --descriptions data/data/descriptions_mmlu.parquet \\
      --output       data/text_between_item_mpnet_mmlu_choices.parquet \\
      --within       data/text_consistency_mpnet_mmlu_choices.parquet
  uv run python scripts/compute_text_consistency_between.py --all-runs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.datasets import SUPPORTED_DATASETS  # noqa: E402
from nla.paths import (  # noqa: E402
    local_descriptions_path,
    local_text_consistency_mpnet_path,
)

DATA = ROOT / "data"
INNER = DATA / "data"
REPORTS = ROOT / "reports"

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Bundled "runs": each (run_id, descriptions, within file, output) so --all-runs covers
# every benchmark variant currently in the repo (PRISM, BiasBios, MMLU with choices,
# MMLU question-only).
ALL_RUNS: list[dict] = [
    {
        "run_id": "prism",
        "descriptions": INNER / "descriptions_prism.parquet",
        "within": DATA / "text_consistency_mpnet_prism.parquet",
        "output": DATA / "text_between_item_mpnet_prism.parquet",
    },
    {
        "run_id": "biosbias",
        "descriptions": INNER / "descriptions_biosbias.parquet",
        "within": DATA / "text_consistency_mpnet_biosbias.parquet",
        "output": DATA / "text_between_item_mpnet_biosbias.parquet",
    },
    {
        "run_id": "mmlu_with_choices",
        "descriptions": INNER / "descriptions_mmlu.parquet",
        "within": DATA / "text_consistency_mpnet_mmlu_choices.parquet",
        "output": DATA / "text_between_item_mpnet_mmlu_choices.parquet",
    },
    {
        "run_id": "mmlu_question_only",
        "descriptions": DATA / "descriptions_mmlu.parquet",
        "within": DATA / "text_consistency_mpnet_mmlu.parquet",
        "output": DATA / "text_between_item_mpnet_mmlu.parquet",
    },
]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_descriptions(
    descriptions_path: Path,
    model_id: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
        raise ValueError("not all activations have the same number of samples K")

    print(f"  descriptions: {descriptions_path}  ({len(df)} rows, N={N}, K={K})")
    print(f"  model:        {model_id}")
    print(f"  device:       {device}")
    print(f"  batch-size:   {batch_size}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device)
    t0 = time.time()
    emb = model.encode(
        df["description"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  encoded {emb.shape[0]} descriptions in {time.time() - t0:.1f}s -> shape {emb.shape}")
    return emb.astype(np.float32, copy=False), activations.astype(np.int64), N, K


def _between_item_stats(emb: np.ndarray, N: int, K: int) -> dict[str, np.ndarray]:
    """Per-activation mean/std/quantiles of cosines vs ALL descriptions of OTHER activations.

    emb shape (N*K, D), unit-norm rows.
    """
    G = emb  # (N*K, D)
    sims = G @ G.T  # (N*K, N*K) — fine for N=400, K=12, D=768 (~92 MB).

    mean_between = np.empty(N, dtype=np.float32)
    std_between = np.empty(N, dtype=np.float32)
    p5 = np.empty(N, dtype=np.float32)
    p50 = np.empty(N, dtype=np.float32)
    p95 = np.empty(N, dtype=np.float32)

    full_mask = np.ones(N * K, dtype=bool)
    for i in range(N):
        block_rows = np.arange(i * K, (i + 1) * K)
        col_mask = full_mask.copy()
        col_mask[block_rows] = False  # exclude same-activation columns
        rows = sims[block_rows][:, col_mask]  # (K, (N-1)*K)
        flat = rows.ravel()
        mean_between[i] = flat.mean()
        std_between[i] = flat.std(ddof=1)
        p5[i], p50[i], p95[i] = np.quantile(flat, [0.05, 0.5, 0.95])
    return {
        "mean_pairwise_text_cosine_between": mean_between,
        "std_pairwise_text_cosine_between": std_between,
        "p5_pairwise_text_cosine_between": p5,
        "median_pairwise_text_cosine_between": p50,
        "p95_pairwise_text_cosine_between": p95,
    }


def _summary(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "p5": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _within_summary(within_path: Path) -> dict[str, float] | None:
    if within_path is None or not within_path.exists():
        return None
    df = pd.read_parquet(within_path)
    col = "mean_pairwise_text_cosine"
    if col not in df.columns:
        return None
    return _summary(df[col].to_numpy(dtype=np.float32))


def compute_one(
    *,
    run_id: str,
    descriptions: Path,
    within: Path | None,
    output: Path,
    model_id: str,
    batch_size: int,
    device: str,
) -> dict:
    print(f"\n=== {run_id} ===")
    emb, activations, N, K = _encode_descriptions(descriptions, model_id, batch_size, device)
    stats = _between_item_stats(emb, N, K)

    out = pd.DataFrame({"activation_idx": activations, **stats})
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    between_per_act = stats["mean_pairwise_text_cosine_between"]
    summary = _summary(between_per_act)
    print(f"\n  between-item mean pairwise cosine (per activation, N={N}):")
    print(f"    mean   {summary['mean']:.4f}")
    print(f"    std    {summary['std']:.4f}")
    print(f"    p5     {summary['p5']:.4f}")
    print(f"    median {summary['median']:.4f}")
    print(f"    p95    {summary['p95']:.4f}")

    within_summary = _within_summary(within) if within else None
    if within_summary is not None:
        gap = within_summary["mean"] - summary["mean"]
        print("\n  within (existing)   vs   between (new):")
        print(f"    mean within:  {within_summary['mean']:.4f}")
        print(f"    mean between: {summary['mean']:.4f}")
        print(f"    gap (within − between): {gap:+.4f}")
    print(f"\n  wrote {len(out)} rows -> {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")

    row = {
        "run_id": run_id,
        "n_activations": int(N),
        "k_samples": int(K),
        "between_mean": summary["mean"],
        "between_std": summary["std"],
        "between_p5": summary["p5"],
        "between_median": summary["median"],
        "between_p95": summary["p95"],
        "within_mean": float(within_summary["mean"]) if within_summary else np.nan,
        "within_std": float(within_summary["std"]) if within_summary else np.nan,
        "gap_within_minus_between": (
            float(within_summary["mean"] - summary["mean"]) if within_summary else np.nan
        ),
    }
    return row


def _runs_from_args(args: argparse.Namespace) -> list[dict]:
    if args.all_runs:
        return ALL_RUNS

    if args.dataset is None and args.descriptions is None:
        raise SystemExit("must pass --dataset, --descriptions/--output, or --all-runs")

    if args.descriptions is not None and args.output is not None:
        return [{
            "run_id": args.dataset or args.descriptions.stem,
            "descriptions": args.descriptions,
            "within": args.within,
            "output": args.output,
        }]

    ds = args.dataset
    if ds not in SUPPORTED_DATASETS:
        raise SystemExit(f"--dataset must be one of {SUPPORTED_DATASETS}, got {ds}")
    return [{
        "run_id": ds,
        "descriptions": local_descriptions_path(ds),
        "within": local_text_consistency_mpnet_path(ds),
        "output": DATA / f"text_between_item_mpnet_{ds}.parquet",
    }]


def _write_summary_csv(rows: Iterable[dict], path: Path) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nwrote summary -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=None, choices=[*SUPPORTED_DATASETS, None])
    p.add_argument("--all-runs", action="store_true",
                   help="run on PRISM, BiasBios, MMLU+choices, MMLU question-only")
    p.add_argument("--descriptions", type=Path, default=None,
                   help="explicit input descriptions parquet (overrides --dataset)")
    p.add_argument("--output", type=Path, default=None,
                   help="explicit output parquet (overrides --dataset)")
    p.add_argument("--within", type=Path, default=None,
                   help="existing within-item text consistency parquet for the same descriptions; "
                        "used for an on-screen within vs between comparison")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--summary-csv", type=Path,
                   default=REPORTS / "synthesis_text_between_item.csv",
                   help="aggregated CSV of within/between per run")
    args = p.parse_args()

    device = resolve_device(args.device)
    runs = _runs_from_args(args)

    rows: list[dict] = []
    for run in runs:
        if not run["descriptions"].exists():
            print(f"\n[skip] {run['run_id']}: missing {run['descriptions']}")
            continue
        rows.append(compute_one(
            run_id=run["run_id"],
            descriptions=run["descriptions"],
            within=run.get("within"),
            output=run["output"],
            model_id=args.model,
            batch_size=args.batch_size,
            device=device,
        ))

    if rows:
        print("\n=========================================")
        print("Summary (within vs between MPNet cosine):")
        print("=========================================")
        header = f"{'run':22s} {'within':>8s} {'between':>9s} {'gap':>9s}"
        print(header)
        print("-" * len(header))
        for r in rows:
            within = f"{r['within_mean']:.4f}" if pd.notna(r['within_mean']) else "—"
            print(f"{r['run_id']:22s} {within:>8s} {r['between_mean']:>9.4f} "
                  f"{r['gap_within_minus_between']:>+9.4f}"
                  if pd.notna(r['gap_within_minus_between']) else
                  f"{r['run_id']:22s} {within:>8s} {r['between_mean']:>9.4f} {'—':>9s}")
        _write_summary_csv(rows, args.summary_csv)


if __name__ == "__main__":
    main()
