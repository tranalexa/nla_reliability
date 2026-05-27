#!/usr/bin/env python3
"""Build CSV tables used by notebooks/analysis_synthesis.ipynb."""
from __future__ import annotations

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
from nla.paths import fidelity_filename, pairwise_filename  # noqa: E402
from nla.synthesis_metrics import (  # noqa: E402
    HEADLINE_METRICS,
    prompt_format_label,
    recon_metrics_from_dir,
    run_probes,
    text_consistency_summary,
)

DATA = ROOT / "data"
INNER = DATA / "data"
OUT = ROOT / "reports"

RUNS = [
    {
        "run_id": "prism",
        "dataset": "prism",
        "data_dir": INNER,
        "targets": ["gender"],
    },
    {
        "run_id": "biosbias",
        "dataset": "biosbias",
        "data_dir": INNER,
        "targets": ["profession", "gender"],
    },
    {
        "run_id": "mmlu_with_choices",
        "dataset": "mmlu",
        "data_dir": INNER,
        "targets": ["subject"],
    },
    {
        "run_id": "mmlu_question_only",
        "dataset": "mmlu",
        "data_dir": DATA,
        "targets": ["subject"],
    },
]


def _g_study_rows(run_id: str, metric: str, result) -> tuple[dict, pd.DataFrame]:
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    headline_rows = []
    text_rows = []
    probe_frames = []
    gvar_rows = []
    gdstudy_frames = []
    inventory_rows = []

    for run in RUNS:
        data_dir = run["data_dir"]
        ds = run["dataset"]
        run_id = run["run_id"]
        csv_path = data_dir / f"{ds}_400.csv"
        act = data_dir / f"activations_layer32_{ds}_gemma-3-12b-pt.parquet"
        recon = data_dir / f"recon_vectors_{ds}.parquet"
        desc = data_dir / f"descriptions_{ds}.parquet"
        fid = data_dir / fidelity_filename(ds)
        pairwise = data_dir / pairwise_filename(ds)

        inventory_rows.append(
            {
                "run_id": run_id,
                "data_dir": str(data_dir.relative_to(ROOT)),
                "prompt_format": prompt_format_label(csv_path),
                "has_csv": csv_path.exists(),
                "has_activations": act.exists(),
                "has_descriptions": desc.exists(),
                "has_recon_vectors": recon.exists(),
                "has_fidelity_scores": fid.exists(),
                "has_pairwise_consistency": pairwise.exists(),
            }
        )

        if act.exists() and recon.exists():
            metrics = recon_metrics_from_dir(data_dir, ds, rng)
            for metric in HEADLINE_METRICS:
                if metric in metrics:
                    headline_rows.append(
                        {"run_id": run_id, "metric": metric, **metrics[metric]}
                    )

        if run_id == "mmlu_with_choices":
            choices_tc = DATA / "text_consistency_mpnet_mmlu_choices.parquet"
            if choices_tc.exists():
                col = "mean_pairwise_text_cosine"
                s = pd.read_parquet(choices_tc)[col].astype(float)
                tc = {
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "median": float(s.median()),
                    "p5": float(s.quantile(0.05)),
                    "p95": float(s.quantile(0.95)),
                }
            else:
                tc = None
        else:
            tc = text_consistency_summary(DATA, ds)
            if tc is None:
                tc = text_consistency_summary(data_dir, ds)
        if tc:
            text_rows.append({"run_id": run_id, **tc})

        probes = run_probes(data_dir, ds, run["targets"])
        if not probes.empty:
            probes.insert(0, "run_id", run_id)
            probe_frames.append(probes)

        if fid.exists():
            result = g_study_pxi(scores_matrix_from_fidelity(pd.read_parquet(fid)))
            row, dstudy = _g_study_rows(run_id, "fidelity_cos", result)
            gvar_rows.append(row)
            gdstudy_frames.append(dstudy)

        if pairwise.exists():
            result = g_study_pxi(scores_matrix_from_pairwise(pd.read_parquet(pairwise)))
            row, dstudy = _g_study_rows(run_id, "consistency_cos", result)
            gvar_rows.append(row)
            gdstudy_frames.append(dstudy)

    pd.DataFrame(inventory_rows).to_csv(OUT / "synthesis_inventory.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(OUT / "synthesis_headline_metrics.csv", index=False)
    pd.DataFrame(text_rows).to_csv(OUT / "synthesis_text_consistency.csv", index=False)
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

    print(f"wrote tables to {OUT}/")


if __name__ == "__main__":
    main()
