#!/usr/bin/env python3
"""Download pipeline artifacts for one run_id from the Modal volume into data/runs/<run_id>/.

Usage:
  uv run python scripts/pull_from_modal.py --run-id prism
  uv run python scripts/pull_from_modal.py --run-id biosbias
  uv run python scripts/pull_from_modal.py --run-id mmlu_choice
  uv run python scripts/pull_from_modal.py --run-id mmlu_nochoice
  uv run python scripts/pull_from_modal.py --all-runs
  uv run python scripts/pull_from_modal.py --run-id prism --only csv,activations,descriptions

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

from nla.paths import (  # noqa: E402
    MODAL_CACHE,
    MODAL_VOLUME,
    RUN_IDS,
    local_activations_path,
    local_csv_path,
    local_descriptions_path,
    local_fidelity_path,
    local_pairwise_path,
    local_recon_vectors_path,
    run_dir,
    volume_activations_path,
    volume_csv_path,
    volume_descriptions_path,
    volume_fidelity_path,
    volume_pairwise_path,
    volume_recon_vectors_path,
)

DEFAULT_N_ITEMS = 400
ARTIFACT_TYPES = ("csv", "activations", "descriptions", "pairwise", "fidelity", "vectors")


def pull(remote: str, dest: Path) -> None:
    """Run ``modal volume get`` from ``remote`` to local ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # modal volume get uses volume-relative paths (the volume root is the
    # /cache mount inside containers).
    remote_path = remote.removeprefix(MODAL_CACHE) or "/"
    print(f"  pull  {remote}")
    print(f"     -> {dest}")
    result = subprocess.run(
        ["modal", "volume", "get", "--force", MODAL_VOLUME, remote_path, str(dest)],
        check=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: pull failed (exit {result.returncode}) - file may not exist yet")


def pull_run(run_id: str, n_items: int, only: set[str] | None) -> None:
    def want(name: str) -> bool:
        return only is None or name in only

    pulls: list[tuple[str, Path]] = []

    if want("csv"):
        pulls.append((volume_csv_path(run_id, n_items), local_csv_path(run_id, n_items)))
    if want("activations"):
        pulls.append((volume_activations_path(run_id), local_activations_path(run_id)))
    if want("descriptions"):
        pulls.append((volume_descriptions_path(run_id), local_descriptions_path(run_id)))
    if want("pairwise"):
        pulls.append((volume_pairwise_path(run_id), local_pairwise_path(run_id)))
    if want("fidelity"):
        pulls.append((volume_fidelity_path(run_id), local_fidelity_path(run_id)))
    if want("vectors"):
        pulls.append((volume_recon_vectors_path(run_id), local_recon_vectors_path(run_id)))

    if not pulls:
        print(f"  nothing selected to pull for {run_id}")
        return

    print(f"\n=== Pulling {run_id} ({len(pulls)} files) into {run_dir(run_id).relative_to(ROOT)}/ ===")
    for remote, dest in pulls:
        pull(remote, dest)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id",
        default=None,
        choices=list(RUN_IDS),
        help="which run to pull (one of: prism, biosbias, mmlu_choice, mmlu_nochoice)",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="pull artifacts for every canonical run_id in sequence",
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
        help=f"comma-separated subset of artifact types: {','.join(ARTIFACT_TYPES)}",
    )
    args = p.parse_args()

    if not args.all_runs and args.run_id is None:
        p.error("either --run-id or --all-runs is required")

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    if only is not None:
        bad = only - set(ARTIFACT_TYPES)
        if bad:
            p.error(f"unknown --only types: {sorted(bad)} (allowed: {ARTIFACT_TYPES})")

    run_ids = list(RUN_IDS) if args.all_runs else [args.run_id]
    print(f"volume={MODAL_VOLUME}  dest={(ROOT / 'data' / 'runs').resolve()}")
    for r in run_ids:
        pull_run(r, args.n_items, only)

    print(f"\ndone - pulled {len(run_ids)} run(s) into data/runs/")


if __name__ == "__main__":
    main()
