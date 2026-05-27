#!/usr/bin/env python3
"""Download pipeline artifacts from Modal volume nla-cache into ./data/.

Usage:
  # Single dataset
  uv run python scripts/pull_from_modal.py --dataset prism
  uv run python scripts/pull_from_modal.py --dataset biosbias

  # Both datasets
  uv run python scripts/pull_from_modal.py --all-datasets

  # Subset of artifact types
  uv run python scripts/pull_from_modal.py --dataset prism --only csv,activations,descriptions

Artifact types (--only):
  csv           sampled items CSV
  activations   activation vectors parquet (Step 1)
  descriptions  AV description parquet (Step 2)
  pairwise      pairwise consistency parquet (Step 3)
  fidelity      fidelity scores parquet (Step 3)
  vectors       reconstructed vectors parquet (Step 3 --save-vectors)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.datasets import SUPPORTED_DATASETS  # noqa: E402
from nla.paths import (  # noqa: E402
    DATA_DIR,
    MODAL_CACHE,
    MODAL_VOLUME,
    activations_filename,
    csv_filename,
    descriptions_filename,
    fidelity_filename,
    pairwise_filename,
    recon_vectors_filename,
)

DEFAULT_N_ITEMS = 400


def pull(remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # modal volume get uses volume-relative paths; MODAL_CACHE is the container
    # mount point (/cache), not a subdirectory inside the volume.
    remote_path = remote.removeprefix(MODAL_CACHE) or "/"
    print(f"  pull  {remote}")
    print(f"     -> {dest}")
    result = subprocess.run(
        ["modal", "volume", "get", "--force", MODAL_VOLUME, remote_path, str(dest)],
        check=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: pull failed (exit {result.returncode}) — file may not exist yet")


def pull_dataset(dataset: str, n_items: int, only: set[str] | None) -> None:
    def want(name: str) -> bool:
        return only is None or name in only

    pulls: list[tuple[str, Path]] = []

    if want("csv"):
        pulls.append((
            f"{MODAL_CACHE}/{csv_filename(dataset, n_items)}",
            DATA_DIR / csv_filename(dataset, n_items),
        ))
    if want("activations"):
        pulls.append((
            f"{MODAL_CACHE}/{activations_filename(dataset)}",
            DATA_DIR / activations_filename(dataset),
        ))
    if want("descriptions"):
        pulls.append((
            f"{MODAL_CACHE}/{descriptions_filename(dataset)}",
            DATA_DIR / descriptions_filename(dataset),
        ))
    if want("pairwise"):
        pulls.append((
            f"{MODAL_CACHE}/{pairwise_filename(dataset)}",
            DATA_DIR / pairwise_filename(dataset),
        ))
    if want("fidelity"):
        pulls.append((
            f"{MODAL_CACHE}/{fidelity_filename(dataset)}",
            DATA_DIR / fidelity_filename(dataset),
        ))
    if want("vectors"):
        pulls.append((
            f"{MODAL_CACHE}/{recon_vectors_filename(dataset)}",
            DATA_DIR / recon_vectors_filename(dataset),
        ))

    if not pulls:
        print(f"  nothing selected to pull for {dataset}")
        return

    print(f"\n=== Pulling {dataset} ({len(pulls)} files) ===")
    for remote, dest in pulls:
        pull(remote, dest)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="prism",
        choices=SUPPORTED_DATASETS,
        help="dataset to pull (default: prism)",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help="pull artifacts for all datasets (prism and biosbias)",
    )
    p.add_argument(
        "--n-items",
        type=int,
        default=DEFAULT_N_ITEMS,
        help=f"number of items sampled in Step 1 (default: {DEFAULT_N_ITEMS})",
    )
    p.add_argument(
        "--only",
        default="",
        help="comma-separated subset: csv,activations,descriptions,pairwise,fidelity,vectors",
    )
    args = p.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    datasets = SUPPORTED_DATASETS if args.all_datasets else [args.dataset]

    print(f"volume={MODAL_VOLUME}  dest={DATA_DIR.resolve()}")
    for ds in datasets:
        pull_dataset(ds, args.n_items, only)

    print(f"\ndone — pulled {len(datasets)} dataset(s) into {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
