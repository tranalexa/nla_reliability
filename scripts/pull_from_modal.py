#!/usr/bin/env python3
"""Download Step 1/2 artifacts from Modal volume nla-cache into ./data/.

Usage:
  uv run python scripts/pull_from_modal.py
  uv run python scripts/pull_from_modal.py --full
  uv run python scripts/pull_from_modal.py --only csv,descriptions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import (  # noqa: E402
    DATA_DIR,
    MODAL_CACHE,
    MODAL_VOLUME,
    DEFAULT_ACTIVATIONS_FULL_LEGACY,
    DEFAULT_ACTIVATIONS_FULL_TAGGED,
    DEFAULT_ACTIVATIONS_PERSONA,
    DEFAULT_CSV,
    DEFAULT_DESCRIPTIONS,
    persona_activations_file,
    full_activations_tagged_file,
)


def pull(remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_path = remote if remote.startswith("/") else f"{MODAL_CACHE}/{remote}"
    print(f"pull {remote_path} -> {dest}")
    subprocess.run(
        ["modal", "volume", "get", MODAL_VOLUME, remote_path, str(dest)],
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="also pull full-prompt activations (legacy + tagged parquet)",
    )
    p.add_argument(
        "--only",
        default="",
        help="comma-separated subset: csv,activations,descriptions,full",
    )
    args = p.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    def want(name: str) -> bool:
        return only is None or name in only

    pulls: list[tuple[str, Path]] = []

    if want("csv"):
        pulls.append((f"{MODAL_CACHE}/selfdescribe_400.csv", DEFAULT_CSV))
    if want("activations"):
        pulls.append(
            (
                f"{MODAL_CACHE}/{persona_activations_file()}",
                DEFAULT_ACTIVATIONS_PERSONA,
            )
        )
    if want("descriptions"):
        pulls.append((f"{MODAL_CACHE}/descriptions.parquet", DEFAULT_DESCRIPTIONS))
    if args.full or want("full"):
        pulls.append(
            (f"{MODAL_CACHE}/{full_activations_tagged_file()}", DEFAULT_ACTIVATIONS_FULL_TAGGED)
        )
        pulls.append((f"{MODAL_CACHE}/activations_layer32.parquet", DEFAULT_ACTIVATIONS_FULL_LEGACY))

    if not pulls:
        p.error("nothing selected to pull")

    print(f"volume={MODAL_VOLUME} dest={DATA_DIR.resolve()}\n")
    for remote, dest in pulls:
        pull(remote, dest)
    print(f"\ndone — {len(pulls)} file(s) in {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
