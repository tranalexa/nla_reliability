#!/usr/bin/env python3
"""G-study and D-study on per-sample fidelity or consistency scores (p x i design).

For each run_id this script:
  1. Loads ``data/runs/<run_id>/fidelity_scores_<dataset>.parquet`` (fidelity_cos)
     or ``data/runs/<run_id>/pairwise_consistency_<dataset>.parquet`` (consistency_cos).
  2. Builds the per-(activation, sample) score matrix.
  3. Runs a two-way random-effects p x i ANOVA (g_study_pxi) to estimate
     sigma^2_p, sigma^2_i, sigma^2_pi, and Cronbach's alpha.
  4. Sweeps a D-study grid over n' averaged samples to report relative G(n').

Outputs written to ``reports/``:
  g_theory_variance_components.csv     one row per (run_id, metric)
  g_theory_d_study.csv                 long-format G(n') grid

Examples:
  uv run python scripts/g_theory_study.py                                # all runs, both metrics
  uv run python scripts/g_theory_study.py --run-id mmlu_choice --metric fidelity
  uv run python scripts/g_theory_study.py --run-id all --metric all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.g_theory import (  # noqa: E402
    d_study_grid,
    g_study_pxi,
    scores_matrix_from_fidelity,
    scores_matrix_from_pairwise,
)
from nla.paths import (  # noqa: E402
    RUN_IDS,
    local_fidelity_path,
    local_pairwise_path,
    validate_run_id,
)

DEFAULT_N_PRIME = [1, 2, 3, 4, 6, 12]

#: metric_key -> (path_fn(run_id) -> Path, matrix_builder(df) -> np.ndarray, metric_label).
METRIC_SPECS: dict[str, tuple[Callable, Callable, str]] = {
    "fidelity": (local_fidelity_path, scores_matrix_from_fidelity, "fidelity_cos"),
    "consistency": (local_pairwise_path, scores_matrix_from_pairwise, "consistency_cos"),
}


def run_one(
    run_id: str,
    n_prime: list[int],
    metric_key: str,
) -> tuple[dict, pd.DataFrame] | None:
    """Run a single (run_id, metric_key) G+D study; returns (var_row, dstudy) or None if missing."""
    path_fn, matrix_fn, metric_label = METRIC_SPECS[metric_key]
    path = path_fn(run_id)
    if not path.exists():
        print(f"  skip {run_id}/{metric_key}: {path} not found")
        return None
    df = pd.read_parquet(path)
    mat = matrix_fn(df)
    result = g_study_pxi(mat)
    dstudy = d_study_grid(result, n_samples=n_prime)

    row = {
        "run_id": run_id,
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
    dstudy.insert(0, "run_id", run_id)
    dstudy.insert(1, "metric", metric_label)
    return row, dstudy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-id",
        default="all",
        choices=[*RUN_IDS, "all"],
        help="which run to analyse; 'all' iterates over every canonical run",
    )
    p.add_argument(
        "--metric",
        default="all",
        choices=[*METRIC_SPECS.keys(), "all"],
        help="fidelity_cos from fidelity_scores_*, consistency_cos from pairwise_consistency_*",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--n-prime", type=int, nargs="*", default=None)
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_prime = args.n_prime if args.n_prime else DEFAULT_N_PRIME
    run_ids = list(RUN_IDS) if args.run_id == "all" else [args.run_id]
    for r in run_ids:
        validate_run_id(r)
    metrics = list(METRIC_SPECS.keys()) if args.metric == "all" else [args.metric]

    var_rows: list[dict] = []
    d_frames: list[pd.DataFrame] = []

    for metric_key in metrics:
        for r in run_ids:
            print(f"\n=== G-study: {r} / {metric_key} ===")
            out = run_one(r, n_prime, metric_key)
            if out is None:
                continue
            row, dstudy = out
            var_rows.append(row)
            d_frames.append(dstudy)
            print(
                f"  sigma^2_p={row['sigma2_p']:.2e} ({row['var_pct_p']:.1f}%)  "
                f"sigma^2_pi={row['sigma2_pi']:.2e} ({row['var_pct_pi']:.1f}%)  "
                f"alpha={row['cronbach_alpha']:.4f}"
            )
            g1 = dstudy.loc[dstudy["n_samples"] == 1, "G_rel"].iloc[0]
            gK = dstudy.loc[dstudy["n_samples"] == row["n_i"], "G_rel"].iloc[0]
            print(f"  G(n'=1)={g1:.4f}   G(n'={row['n_i']})={gK:.4f}")

    if not var_rows:
        print("No runs/metrics produced output (check that the required parquets are present).")
        sys.exit(1)

    var_csv = out_dir / "g_theory_variance_components.csv"
    d_csv = out_dir / "g_theory_d_study.csv"
    pd.DataFrame(var_rows).to_csv(var_csv, index=False)
    pd.concat(d_frames, ignore_index=True).to_csv(d_csv, index=False)
    print(f"\nWrote {var_csv.relative_to(ROOT)}")
    print(f"Wrote {d_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
