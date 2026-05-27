"""Mean pairwise text-space consistency per activation using MPNet sentence embeddings.

Reads AV descriptions and embeds every row once with sentence-transformers/all-mpnet-base-v2
in a single batched pass. For each activation, averages C(K,2) pairwise cosine similarities
between its K description embeddings.

Output: one row per activation (activation_idx, mean_pairwise_text_cosine).

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

from nla.datasets import SUPPORTED_DATASETS  # noqa: E402
from nla.paths import (  # noqa: E402
    DEFAULT_DESCRIPTIONS,
    DEFAULT_TEXT_CONSISTENCY_MPNET,
    local_descriptions_path,
    local_text_consistency_mpnet_path,
)

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
LEGACY_DATASET = "selfdescribe"
TEXT_CONSISTENCY_DATASETS: list[str] = [*SUPPORTED_DATASETS, LEGACY_DATASET]
LEGACY_PAIRWISE = ROOT / "data" / "text_recon_pairwise.parquet"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_paths(
    dataset: str,
    descriptions: Path | None,
    output: Path | None,
) -> tuple[Path, Path]:
    if descriptions is not None and output is not None:
        return descriptions, output

    if dataset == LEGACY_DATASET:
        desc = descriptions or DEFAULT_DESCRIPTIONS
        out = output or DEFAULT_TEXT_CONSISTENCY_MPNET
        return desc, out

    desc = descriptions or local_descriptions_path(dataset)
    out = output or local_text_consistency_mpnet_path(dataset)
    return desc, out


def compute_one(
    descriptions_path: Path,
    output_path: Path,
    *,
    model_id: str,
    batch_size: int,
    device: str,
    dataset_label: str,
    sanity_pairwise: Path | None,
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
    print(f"\n=== {dataset_label} ===")
    print(f"descriptions: {descriptions_path}  ({len(df)} rows, N={N}, K={K})")
    print(f"output:       {output_path}")
    print(f"model:        {model_id}")
    print(f"device:       {resolved_device}")
    print(f"batch-size:   {batch_size}")

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
    enc_secs = time.time() - t0
    print(f"encoded {emb.shape[0]} descriptions in {enc_secs:.1f}s -> shape {emb.shape}")

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
    print(f"\nmean_pairwise_text_cosine distribution (per activation, N={N}):")
    print(f"  mean   {v.mean():.4f}")
    print(f"  std    {v.std(ddof=1):.4f}")
    print(f"  min    {v.min():.4f}")
    print(f"  median {np.median(v):.4f}")
    print(f"  max    {v.max():.4f}")

    if sanity_pairwise is not None and sanity_pairwise.exists():
        legacy = pd.read_parquet(sanity_pairwise)
        if {"activation_idx", "text_cos_sim"} <= set(legacy.columns):
            legacy_mean = legacy.groupby("activation_idx")["text_cos_sim"].mean().rename("minilm")
            joined = out.set_index("activation_idx").join(legacy_mean, how="inner")
            if len(joined) >= 3:
                r = np.corrcoef(joined["mean_pairwise_text_cosine"], joined["minilm"])[0, 1]
                print(f"\nsanity: Pearson r vs MiniLM per-activation mean text cos = {r:.3f} "
                      f"(n={len(joined)})")

    print(f"wrote {len(out)} rows -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=LEGACY_DATASET,
        choices=TEXT_CONSISTENCY_DATASETS,
        help=f"dataset for default path inference (default: {LEGACY_DATASET})",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help=f"run for all of: {', '.join(TEXT_CONSISTENCY_DATASETS)} (skip missing inputs)",
    )
    p.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="input descriptions parquet (default: inferred from --dataset)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output parquet (default: inferred from --dataset)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | mps (default: auto)",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="with --all-datasets, skip datasets whose descriptions file is absent",
    )
    args = p.parse_args()

    if args.all_datasets and (args.descriptions or args.output):
        print("ERROR: --all-datasets cannot be combined with --descriptions/--output",
              file=sys.stderr)
        sys.exit(1)

    datasets = TEXT_CONSISTENCY_DATASETS if args.all_datasets else [args.dataset]
    skip_missing = args.skip_missing or args.all_datasets
    failed = False

    for ds in datasets:
        desc_path, out_path = resolve_paths(ds, args.descriptions, args.output)
        if not desc_path.exists():
            if skip_missing:
                print(f"skip {ds}: {desc_path} not found")
                continue
            print(f"ERROR: {desc_path} not found.", file=sys.stderr)
            print(f"  Pull: uv run python scripts/pull_from_modal.py --dataset {ds} "
                  f"--only descriptions", file=sys.stderr)
            if ds == LEGACY_DATASET:
                print(f"  Legacy SelfDescribe uses: {DEFAULT_DESCRIPTIONS}", file=sys.stderr)
            failed = True
            continue

        sanity = LEGACY_PAIRWISE if ds == LEGACY_DATASET else None
        try:
            compute_one(
                desc_path,
                out_path,
                model_id=args.model,
                batch_size=args.batch_size,
                device=args.device,
                dataset_label=ds,
                sanity_pairwise=sanity,
            )
        except ValueError as e:
            print(f"ERROR [{ds}]: {e}", file=sys.stderr)
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
