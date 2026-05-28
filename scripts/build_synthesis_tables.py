#!/usr/bin/env python3
"""Build every reports/*.csv (+ results_table.tex) consumed by the notebook + paper.

Reads ``data/runs/<run_id>/`` for each of the four canonical run_ids and writes:

  reports/summary_stats.csv                  long-format per-run metric summary
  reports/results_table.tex                  LaTeX table for the paper
  reports/synthesis_inventory.csv            on-disk presence per run_id
  reports/synthesis_headline_metrics.csv     seven HEADLINE_METRICS per run_id
  reports/synthesis_text_consistency.csv     within-item MPNet summary per run_id
  reports/synthesis_linear_probes.csv        majority + probe accuracy per (run, target, source)
  reports/synthesis_g_theory_variance.csv    G-study sigma^2 + alpha per (run, metric)
  reports/synthesis_g_theory_d_study.csv     D-study G(n') per (run, metric, n')

Usage:
  uv run python scripts/build_synthesis_tables.py
  uv run python scripts/build_synthesis_tables.py --dry-run     # only check file presence

Random seeds:
  - RNG_SEED=0 inside nla.synthesis_metrics for mismatch / between-item sampling.
  - PROBE_SEED=42 for the probe train/test split.
Both are fixed and printed at job start so reruns are deterministic on a given dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    dataset_for_run,
    local_csv_path,
    local_fidelity_path,
    local_pairwise_path,
    local_text_between_item_mpnet_path,
    run_dir,
)
from nla.synthesis_metrics import (  # noqa: E402
    HEADLINE_METRICS,
    PROBE_SEED,
    RNG_SEED,
    build_summary_df,
    compute_run_metrics,
    headline_metrics_for_run,
    prompt_format_label,
    run_probes,
    text_consistency_summary,
    write_latex_table,
)

OUT = ROOT / "reports"

#: Display names for each canonical run.
RUN_LABELS: dict[str, str] = {
    "prism": "PRISM",
    "biosbias": "Bias in Bios",
    "mmlu_choice": "MMLU-Choice",
    "mmlu_nochoice": "MMLU-NoChoice",
}

#: Probe target columns per run (must exist in the run's <dataset>_400.csv).
RUN_TARGETS: dict[str, list[str]] = {
    "prism": ["gender"],
    "biosbias": ["profession", "gender"],
    "mmlu_choice": ["subject"],
    "mmlu_nochoice": ["subject"],
}


def _g_study_rows(run_id: str, metric: str, result) -> tuple[dict, pd.DataFrame]:
    """Pack one G-study result into a flat row + its n'-grid frame."""
    dstudy = d_study_grid(result)
    row = {
        "run_id": run_id,
        "metric": metric,
        "n_p": result.n_p,
        "n_i": result.n_i,
        "sigma2_p": result.sigma2_p,
        "sigma2_i": result.sigma2_i,
        "sigma2_pi": result.sigma2_pi,
        "var_pct_p": result.var_pct_p,
        "var_pct_i": result.var_pct_i,
        "var_pct_pi": result.var_pct_pi,
        "cronbach_alpha": result.cronbach_alpha,
        "G_rel_n1": result.g_rel(1),
        "G_rel_n12": result.g_rel(result.n_i),
    }
    dstudy.insert(0, "run_id", run_id)
    dstudy.insert(1, "metric", metric)
    return row, dstudy


def _between_item_text_summary(run_id: str) -> dict | None:
    """Aggregate within / between / gap text-space cosine for one run."""
    between_path = local_text_between_item_mpnet_path(run_id)
    if not between_path.exists():
        return None
    df = pd.read_parquet(between_path)
    if "mean_pairwise_text_cosine_between" not in df.columns:
        return None
    between = df["mean_pairwise_text_cosine_between"].astype(float)
    within_summary = text_consistency_summary(run_id) or {}
    return {
        "run_id": run_id,
        "n_activations": int(len(df)),
        "between_mean": float(between.mean()),
        "between_std": float(between.std(ddof=1)),
        "between_p5": float(between.quantile(0.05)),
        "between_median": float(between.median()),
        "between_p95": float(between.quantile(0.95)),
        "within_mean": within_summary.get("mean", np.nan),
        "within_std": within_summary.get("std", np.nan),
        "gap_within_minus_between": (
            within_summary.get("mean", np.nan) - float(between.mean())
            if within_summary
            else np.nan
        ),
    }


def _inventory_row(run_id: str) -> dict:
    """File-presence snapshot for one run, surfaced via synthesis_inventory.csv."""
    ds = dataset_for_run(run_id)
    rd = run_dir(run_id)
    csv_path = local_csv_path(run_id, 400)
    return {
        "run_id": run_id,
        "dataset": ds,
        "run_dir": str(rd.relative_to(ROOT)) if rd.is_relative_to(ROOT) else str(rd),
        "prompt_format": prompt_format_label(csv_path),
        "has_csv": csv_path.exists(),
        "has_activations": (rd / f"activations_layer32_{ds}_gemma-3-12b-pt.parquet").exists(),
        "has_descriptions": (rd / f"descriptions_{ds}.parquet").exists(),
        "has_recon_vectors": (rd / f"recon_vectors_{ds}.parquet").exists(),
        "has_fidelity_scores": local_fidelity_path(run_id).exists(),
        "has_pairwise_consistency": local_pairwise_path(run_id).exists(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="just print the inventory; do not compute or write tables",
    )
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"build_synthesis_tables   seeds: RNG_SEED={RNG_SEED}  PROBE_SEED={PROBE_SEED}")
    print(f"out dir: {OUT.relative_to(ROOT)}")

    inventory_rows = [_inventory_row(r) for r in RUN_IDS]
    inv = pd.DataFrame(inventory_rows)
    print("\nInventory (file presence per run):")
    print(inv.to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: skipping metric computation")
        return

    inv.to_csv(OUT / "synthesis_inventory.csv", index=False)

    headline_rows: list[dict] = []
    text_within_rows: list[dict] = []
    text_between_rows: list[dict] = []
    probe_frames: list[pd.DataFrame] = []
    gvar_rows: list[dict] = []
    gdstudy_frames: list[pd.DataFrame] = []
    all_stats: dict[str, dict] = {}

    for run_id in RUN_IDS:
        print(f"\n=== {run_id} ({RUN_LABELS[run_id]}) ===")

        stats, _ = compute_run_metrics(run_id, seed=RNG_SEED)
        if stats:
            all_stats[run_id] = stats
            headline = headline_metrics_for_run(run_id, seed=RNG_SEED)
            for metric in HEADLINE_METRICS:
                if metric in headline:
                    headline_rows.append({"run_id": run_id, "metric": metric, **headline[metric]})

        tc = text_consistency_summary(run_id)
        if tc:
            text_within_rows.append({"run_id": run_id, **tc})

        bt = _between_item_text_summary(run_id)
        if bt:
            text_between_rows.append(bt)

        targets = RUN_TARGETS.get(run_id, [])
        if targets:
            probes = run_probes(run_id, targets, seed=PROBE_SEED)
            if not probes.empty:
                probes.insert(0, "run_id", run_id)
                probe_frames.append(probes)

        fid_path = local_fidelity_path(run_id)
        if fid_path.exists():
            result = g_study_pxi(scores_matrix_from_fidelity(pd.read_parquet(fid_path)))
            row, dstudy = _g_study_rows(run_id, "fidelity_cos", result)
            gvar_rows.append(row)
            gdstudy_frames.append(dstudy)

        pairwise_path = local_pairwise_path(run_id)
        if pairwise_path.exists():
            result = g_study_pxi(scores_matrix_from_pairwise(pd.read_parquet(pairwise_path)))
            row, dstudy = _g_study_rows(run_id, "consistency_cos", result)
            gvar_rows.append(row)
            gdstudy_frames.append(dstudy)

    if all_stats:
        summary_df = build_summary_df(all_stats)
        summary_df.to_csv(OUT / "summary_stats.csv", index=False, float_format="%.6f")
        write_latex_table(all_stats, RUN_LABELS, OUT / "results_table.tex")

    pd.DataFrame(headline_rows).to_csv(OUT / "synthesis_headline_metrics.csv", index=False)
    pd.DataFrame(text_within_rows).to_csv(OUT / "synthesis_text_consistency.csv", index=False)
    if text_between_rows:
        pd.DataFrame(text_between_rows).to_csv(OUT / "synthesis_text_between_item.csv", index=False)
    if probe_frames:
        pd.concat(probe_frames, ignore_index=True).to_csv(
            OUT / "synthesis_linear_probes.csv", index=False
        )
    if gvar_rows:
        pd.DataFrame(gvar_rows).to_csv(OUT / "synthesis_g_theory_variance.csv", index=False)
    if gdstudy_frames:
        pd.concat(gdstudy_frames, ignore_index=True).to_csv(
            OUT / "synthesis_g_theory_d_study.csv", index=False
        )

    print(f"\nwrote tables to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
