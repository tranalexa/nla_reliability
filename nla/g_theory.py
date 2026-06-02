"""Generalizability theory helpers for p × i designs (activations × AV samples).

Design: ``n_p`` activations × ``n_i`` stochastic AV samples, one cosine score
per cell. Both facets are treated as random. Used here for the two reliability
metrics:

  * ``fidelity_cos``     — cosine(reconstruction, original activation)
  * ``consistency_cos``  — per-sample mean within-item recon cosine

Variance components (estimated from expected mean squares of a two-way random
ANOVA):

  σ²_p  — between activations (signal we want to rank by)
  σ²_i  — between sample occasions (global AV bias across all activations)
  σ²_pi — activation × sample interaction (residual)

``G_rel(n')`` — dependability of the mean of ``n'`` samples as an estimator of
the activation's true score; used as the D-study reliability curve.

This module is **not** vendored. It is pure-Python / pandas / numpy and
written for this repo. See ``scripts/g_theory_study.py`` for the CLI that
calls it per run_id, and ``scripts/build_synthesis_tables.py`` for the
aggregate that lands in ``reports/synthesis_g_theory_*.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GStudyResult:
    n_p: int
    n_i: int
    ms_p: float
    ms_i: float
    ms_pi: float
    sigma2_p: float
    sigma2_i: float
    sigma2_pi: float
    var_pct_p: float
    var_pct_i: float
    var_pct_pi: float
    cronbach_alpha: float

    def g_rel(self, n_prime: int) -> float:
        denom = self.sigma2_p + self.sigma2_pi / max(n_prime, 1)
        if denom <= 0:
            return 0.0
        return float(self.sigma2_p / denom)

    def phi_abs(self, n_prime: int) -> float:
        denom = self.sigma2_p + self.sigma2_pi / max(n_prime, 1) + self.sigma2_i / max(self.n_p, 1)
        if denom <= 0:
            return 0.0
        return float(self.sigma2_p / denom)


def scores_matrix_from_fidelity(df: pd.DataFrame, score_col: str = "fidelity_cos") -> np.ndarray:
    """Pivot long fidelity table to (n_p, n_i) matrix."""
    required = {"activation_idx", "sample_idx", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"fidelity table needs columns {required}, got {df.columns.tolist()}")
    wide = df.pivot(index="activation_idx", columns="sample_idx", values=score_col).sort_index()
    wide = wide.sort_index(axis=1)
    if wide.isna().any().any():
        raise ValueError("fidelity table has missing (activation_idx, sample_idx) cells")
    return wide.to_numpy(dtype=np.float64)


def scores_matrix_from_pairwise(df: pd.DataFrame, score_col: str = "cos_sim") -> np.ndarray:
    """Per-sample mean within-item recon cosine → (n_p, n_i) matrix.

    For activation p and sample i, the score is the mean cos_sim between recon(p, i)
    and the other K−1 within-item recons (from pairwise_consistency_* parquet).
    """
    required = {"activation_idx", "sample_i", "sample_j", score_col}
    if not required.issubset(df.columns):
        raise ValueError(f"pairwise table needs columns {required}, got {df.columns.tolist()}")
    long = pd.concat(
        [
            df[["activation_idx", "sample_i", score_col]].rename(columns={"sample_i": "sample_idx"}),
            df[["activation_idx", "sample_j", score_col]].rename(columns={"sample_j": "sample_idx"}),
        ],
        ignore_index=True,
    )
    per = long.groupby(["activation_idx", "sample_idx"], as_index=False)[score_col].mean()
    wide = per.pivot(index="activation_idx", columns="sample_idx", values=score_col).sort_index()
    wide = wide.sort_index(axis=1)
    if wide.isna().any().any():
        raise ValueError("pairwise table missing within-item scores for some (activation_idx, sample_idx)")
    return wide.to_numpy(dtype=np.float64)


def g_study_pxi(matrix: np.ndarray) -> GStudyResult:
    """Two-way random-effects ANOVA on a balanced p × i matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-D (n_p, n_i)")
    n_p, n_i = arr.shape
    if n_p < 2 or n_i < 2:
        raise ValueError(f"need n_p>=2 and n_i>=2, got {n_p}, {n_i}")

    grand = arr.mean()
    mean_p = arr.mean(axis=1, keepdims=True)
    mean_i = arr.mean(axis=0, keepdims=True)

    ss_p = float(n_i * np.sum((mean_p - grand) ** 2))
    ss_i = float(n_p * np.sum((mean_i - grand) ** 2))
    resid = arr - mean_p - mean_i + grand
    ss_pi = float(np.sum(resid**2))

    df_p = n_p - 1
    df_i = n_i - 1
    df_pi = (n_p - 1) * (n_i - 1)

    ms_p = ss_p / df_p
    ms_i = ss_i / df_i
    ms_pi = ss_pi / df_pi

    sigma2_pi = max(ms_pi, 0.0)
    sigma2_p = max((ms_p - ms_pi) / n_i, 0.0)
    sigma2_i = max((ms_i - ms_pi) / n_p, 0.0)

    total = sigma2_p + sigma2_i + sigma2_pi
    if total > 0:
        var_pct_p = 100.0 * sigma2_p / total
        var_pct_i = 100.0 * sigma2_i / total
        var_pct_pi = 100.0 * sigma2_pi / total
    else:
        var_pct_p = var_pct_i = var_pct_pi = 0.0

    # Cronbach's α (parallel form from G_rel at n′=1; stable for narrow score bands)
    g1 = sigma2_p / (sigma2_p + sigma2_pi) if (sigma2_p + sigma2_pi) > 0 else 0.0
    if n_i > 1 and g1 > 0:
        alpha = float((n_i * g1) / (1.0 + (n_i - 1) * g1))
    else:
        alpha = float("nan")

    return GStudyResult(
        n_p=n_p,
        n_i=n_i,
        ms_p=ms_p,
        ms_i=ms_i,
        ms_pi=ms_pi,
        sigma2_p=sigma2_p,
        sigma2_i=sigma2_i,
        sigma2_pi=sigma2_pi,
        var_pct_p=var_pct_p,
        var_pct_i=var_pct_i,
        var_pct_pi=var_pct_pi,
        cronbach_alpha=alpha,
    )


def d_study_grid(result: GStudyResult, n_samples: list[int] | None = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = [1, 2, 3, 4, 6, result.n_i]
    rows = []
    for n_prime in n_samples:
        rows.append(
            {
                "n_samples": int(n_prime),
                "G_rel": result.g_rel(n_prime),
                "Phi_abs": result.phi_abs(n_prime),
            }
        )
    return pd.DataFrame(rows)
