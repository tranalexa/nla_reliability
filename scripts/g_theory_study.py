#!/usr/bin/env python3
"""G-study and D-study for NLA reliability (activation × AV sample design).

Treats each activation as the object of measurement (facet p) and each of K=12
stochastic AV→AR scores as the sample facet (facet i). Default outcome:
fidelity_cos from Step 3 fidelity parquets (one score per activation_idx × sample_idx).

Also reports Cronbach's α (samples as items) for classical internal consistency.

Usage:
  uv run python scripts/g_theory_study.py --dataset prism
  uv run python scripts/g_theory_study.py --dataset all --data-dir data/data
  uv run python scripts/g_theory_study.py --dataset biosbias --metric fidelity_cos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.g_theory import run_g_study  # noqa: E402
from nla.paths import fidelity_filename  # noqa: E402

DATASETS = ("prism", "biosbias", "mmlu")
DEFAULT_N_PRIME = (1, 2, 3, 4, 6, 12)


def load_long_scores(data_dir: Path, dataset: str, metric: str) -> pd.DataFrame:
    path = data_dir / fidelity_filename(dataset)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {"activation_idx", "sample_idx", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)


def format_vc_table(result, dataset: str, metric: str) -> pd.DataFrame:
    vc = result.vc
    rows = [
        {
            "dataset": dataset,
            "metric": metric,
            "n_p": vc.n_p,
            "n_i": vc.n_i,
            "ms_p": vc.ms_p,
            "ms_i": vc.ms_i,
            "ms_pi": vc.ms_pi,
            "sigma2_p": vc.sigma2_p,
            "sigma2_i": vc.sigma2_i,
            "sigma2_pi": vc.sigma2_pi,
            "var_pct_p": 100 * vc.sigma2_p / max(vc.sigma2_p + vc.sigma2_i + vc.sigma2_pi, 1e-18),
            "var_pct_i": 100 * vc.sigma2_i / max(vc.sigma2_p + vc.sigma2_i + vc.sigma2_pi, 1e-18),
            "var_pct_pi": 100 * vc.sigma2_pi / max(vc.sigma2_p + vc.sigma2_i + vc.sigma2_pi, 1e-18),
            "cronbach_alpha": result.cronbach_alpha,
        }
    ]
    return pd.DataFrame(rows)


def format_d_study_table(result, dataset: str, metric: str) -> pd.DataFrame:
    rows = []
    for n in sorted(result.g_rel):
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n_samples": n,
                "G_rel": result.g_rel[n],
                "Phi_abs": result.phi_abs[n],
            }
        )
    return pd.DataFrame(rows)


def print_report(result, dataset: str, metric: str) -> None:
    vc = result.vc
    total_var = vc.sigma2_p + vc.sigma2_i + vc.sigma2_pi
    print(f"\n{'=' * 72}")
    print(f"G-study: {dataset}  |  metric={metric}  |  design p×i ({vc.n_p}×{vc.n_i})")
    print(f"{'=' * 72}")
    print("\nMean squares:")
    print(f"  MS_p   = {vc.ms_p:.6e}")
    print(f"  MS_i   = {vc.ms_i:.6e}")
    print(f"  MS_pi  = {vc.ms_pi:.6e}")
    print("\nVariance components (negative estimates clamped to 0):")
    print(f"  σ²_p   = {vc.sigma2_p:.6e}  ({100 * vc.sigma2_p / max(total_var, 1e-18):.1f}% of comp total)")
    print(f"  σ²_i   = {vc.sigma2_i:.6e}  ({100 * vc.sigma2_i / max(total_var, 1e-18):.1f}%)")
    print(f"  σ²_pi  = {vc.sigma2_pi:.6e}  ({100 * vc.sigma2_pi / max(total_var, 1e-18):.1f}%)")
    print(f"\nCronbach's α (samples as items): {result.cronbach_alpha:.4f}")
    print("\nD-study (projected coefficients if each activation uses mean of n' samples):")
    print(f"  {'n':>4}  {'G_rel':>10}  {'Phi_abs':>10}")
    print("  " + "-" * 28)
    for n in sorted(result.g_rel):
        print(f"  {n:>4}  {result.g_rel[n]:>10.4f}  {result.phi_abs[n]:>10.4f}")
    print("\nInterpretation:")
    print("  G_rel  — dependability for ranking/discriminating activations")
    print("  Phi_abs — dependability for absolute level of an activation's score")
    print("  σ²_pi  — instability across AV samples (lower → more reliable averaging)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="all", choices=("all", *DATASETS))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "data")
    p.add_argument("--metric", default="fidelity_cos", help="score column in fidelity parquet")
    p.add_argument(
        "--n-prime",
        default="1,2,3,4,6,12",
        help="comma-separated sample sizes for D-study projection",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = p.parse_args()

    n_prime_grid = tuple(int(x.strip()) for x in args.n_prime.split(",") if x.strip())
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    vc_frames: list[pd.DataFrame] = []
    d_frames: list[pd.DataFrame] = []

    for ds in datasets:
        df = load_long_scores(args.data_dir, ds, args.metric)
        y = df[args.metric].to_numpy(dtype=np.float64)
        result = run_g_study(
            y,
            df["activation_idx"].to_numpy(),
            df["sample_idx"].to_numpy(),
            n_prime_grid=n_prime_grid,
        )
        print_report(result, ds, args.metric)
        vc_frames.append(format_vc_table(result, ds, args.metric))
        d_frames.append(format_d_study_table(result, ds, args.metric))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vc_path = args.out_dir / "g_theory_variance_components.csv"
    d_path = args.out_dir / "g_theory_d_study.csv"
    pd.concat(vc_frames, ignore_index=True).to_csv(vc_path, index=False)
    pd.concat(d_frames, ignore_index=True).to_csv(d_path, index=False)
    print(f"\nwrote {vc_path}")
    print(f"wrote {d_path}")


if __name__ == "__main__":
    main()
