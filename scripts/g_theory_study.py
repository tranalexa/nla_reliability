#!/usr/bin/env python3
"""G-study and D-study on per-sample fidelity or consistency scores (p × i design).

Usage:
  uv run python scripts/g_theory_study.py --dataset mmlu --data-dir data
  uv run python scripts/g_theory_study.py --dataset all --data-dir data/data
  uv run python scripts/g_theory_study.py --dataset mmlu --metric consistency --data-dir data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.datasets import SUPPORTED_DATASETS  # noqa: E402
from nla.g_theory import (  # noqa: E402
    d_study_grid,
    g_study_pxi,
    scores_matrix_from_fidelity,
    scores_matrix_from_pairwise,
)
from nla.paths import fidelity_filename, pairwise_filename  # noqa: E402

DEFAULT_N_PRIME = [1, 2, 3, 4, 6, 12]

METRIC_SPECS: dict[str, tuple[str, Callable[[pd.DataFrame], object], str]] = {
    "fidelity": (fidelity_filename, scores_matrix_from_fidelity, "fidelity_cos"),
    "consistency": (pairwise_filename, scores_matrix_from_pairwise, "consistency_cos"),
}


def run_one(
    dataset: str,
    data_dir: Path,
    n_prime: list[int],
    metric_key: str,
) -> tuple[dict, pd.DataFrame] | None:
    path_fn, matrix_fn, metric_label = METRIC_SPECS[metric_key]
    path = data_dir / path_fn(dataset)
    if not path.exists():
        print(f"  skip {dataset}/{metric_key}: {path} not found")
        return None
    df = pd.read_parquet(path)
    mat = matrix_fn(df)
    result = g_study_pxi(mat)
    dstudy = d_study_grid(result, n_samples=n_prime)

    row = {
        "dataset": dataset,
        "metric": metric_label,
        "n_p": result.n_p,
        "n_i": result.n_i,
        "ms_p": result.ms_p,
        "ms_i": result.ms_i,
        "ms_pi": result.ms_pi,
        "sigma2_p": result.sigma2_p,
        "sigma2_i": result.sigma2_i,
        "sigma2_pi": result.sigma2_pi,
        "var_pct_p": result.var_pct_p,
        "var_pct_i": result.var_pct_i,
        "var_pct_pi": result.var_pct_pi,
        "cronbach_alpha": result.cronbach_alpha,
    }
    dstudy.insert(0, "dataset", dataset)
    dstudy.insert(1, "metric", metric_label)
    return row, dstudy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mmlu", choices=[*SUPPORTED_DATASETS, "all"])
    p.add_argument(
        "--metric",
        default="all",
        choices=[*METRIC_SPECS.keys(), "all"],
        help="fidelity_cos from fidelity_scores_* or consistency_cos from pairwise_consistency_*",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--n-prime", type=int, nargs="*", default=None)
    args = p.parse_args()

    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_prime = args.n_prime if args.n_prime else DEFAULT_N_PRIME
    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    metrics = list(METRIC_SPECS.keys()) if args.metric == "all" else [args.metric]

    var_rows: list[dict] = []
    d_frames: list[pd.DataFrame] = []

    for metric_key in metrics:
        for ds in datasets:
            print(f"\n=== G-study: {ds} / {metric_key} ({data_dir}) ===")
            out = run_one(ds, data_dir, n_prime, metric_key)
            if out is None:
                continue
            row, dstudy = out
            var_rows.append(row)
            d_frames.append(dstudy)
            print(f"  σ²_p={row['sigma2_p']:.2e} ({row['var_pct_p']:.1f}%)  "
                  f"σ²_pi={row['sigma2_pi']:.2e} ({row['var_pct_pi']:.1f}%)  "
                  f"α={row['cronbach_alpha']:.4f}")
            print(f"  G_rel(n′=1)={dstudy.loc[dstudy['n_samples']==1,'G_rel'].iloc[0]:.4f}  "
                  f"G_rel(n′={row['n_i']})="
                  f"{dstudy.loc[dstudy['n_samples']==row['n_i'],'G_rel'].iloc[0]:.4f}")

    if not var_rows:
        print("No datasets processed.")
        sys.exit(1)

    pd.DataFrame(var_rows).to_csv(out_dir / "g_theory_variance_components.csv", index=False)
    pd.concat(d_frames, ignore_index=True).to_csv(out_dir / "g_theory_d_study.csv", index=False)
    print(f"\nWrote {out_dir}/g_theory_variance_components.csv")
    print(f"Wrote {out_dir}/g_theory_d_study.csv")


if __name__ == "__main__":
    main()
