"""Train majority-baseline + logistic probes per run_id.

This script is a thin CLI wrapper around :func:`nla.synthesis_metrics.run_probes`.
It is provided so graders can sanity-check probe accuracy independently of the
full ``build_synthesis_tables.py`` pipeline. ``build_synthesis_tables.py``
also runs probes (using the same code path) and writes the consolidated
``reports/synthesis_linear_probes.csv``.

For each run_id, probes train on layer-32 originals (``original``) and on the
per-item mean of the 12 AR reconstructions (``recon_mean``):

  * prism         → target = gender
  * biosbias      → targets = profession, gender
  * mmlu_choice   → target = subject
  * mmlu_nochoice → target = subject

Random seed: ``--seed`` controls the stratified train/test split (default 42).
The majority baseline uses the same split as the LR probe so the two numbers
are directly comparable.

Usage:
  uv run python scripts/train_linear_probes.py                    # all four run_ids
  uv run python scripts/train_linear_probes.py --run-id biosbias  # one run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import RUN_IDS, validate_run_id  # noqa: E402
from nla.synthesis_metrics import run_probes  # noqa: E402

RUN_TARGETS: dict[str, list[str]] = {
    "prism": ["gender"],
    "biosbias": ["profession", "gender"],
    "mmlu_choice": ["subject"],
    "mmlu_nochoice": ["subject"],
}


def _print_probe_block(run_id: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"  (no probe data for {run_id} — required parquets missing)")
        return
    cols = ["target", "vector_source", "n", "n_classes", "majority_acc", "probe_acc"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-id",
        default="all",
        help="run_id to probe, or 'all' for every canonical run (default: all)",
    )
    p.add_argument("--seed", type=int, default=42, help="train/test split seed (default: 42)")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()

    runs = list(RUN_IDS) if args.run_id == "all" else [args.run_id]
    for r in runs:
        validate_run_id(r)

    print(f"linear_probes seed={args.seed} test_size={args.test_size}")
    print()
    for run_id in runs:
        targets = RUN_TARGETS.get(run_id, [])
        if not targets:
            print(f"=== {run_id} === (no probe targets configured, skipping)")
            print()
            continue
        print(f"=== {run_id} === targets={targets}")
        probes = run_probes(run_id, targets, seed=args.seed, test_size=args.test_size)
        _print_probe_block(run_id, probes)
        print()


if __name__ == "__main__":
    main()
