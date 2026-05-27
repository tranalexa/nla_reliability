"""Generalizability theory helpers for p × i designs (activations × AV samples).

Design: each of n_p activations is measured on n_i stochastic samples (e.g. K=12
AV→AR fidelity scores). Both facets are treated as random.

Variance components (Brennan, 2001; one observation per cell):
  σ²_p  — activation (object of measurement)
  σ²_i  — sample / occasion facet
  σ²_pi — activation × sample interaction (includes residual when n=1 per cell)

Coefficients:
  G_rel(n') — relative generalization over activations when each activation is
              the mean of n' sample scores
  Phi_abs(n') — absolute generalization for a single activation mean
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarianceComponents:
    sigma2_p: float
    sigma2_i: float
    sigma2_pi: float
    ms_p: float
    ms_i: float
    ms_pi: float
    n_p: int
    n_i: int


@dataclass(frozen=True)
class GStudyResult:
    vc: VarianceComponents
    g_rel: dict[int, float]
    phi_abs: dict[int, float]
    cronbach_alpha: float


def _two_way_anova_ms(
    y: np.ndarray,
    p: np.ndarray,
    i: np.ndarray,
) -> tuple[float, float, float, int, int]:
    """Cell means ANOVA mean squares for fully crossed p × i (one obs per cell)."""
    df = pd.DataFrame({"y": y.astype(np.float64), "p": p, "i": i})
    n_p = df["p"].nunique()
    n_i = df["i"].nunique()
    if len(df) != n_p * n_i:
        raise ValueError(f"unbalanced design: {len(df)} rows but {n_p}×{n_i}={n_p * n_i}")

    grand = df["y"].mean()
    cell = df.groupby(["p", "i"], as_index=False)["y"].mean()
    p_means = cell.groupby("p")["y"].mean()
    i_means = cell.groupby("i")["y"].mean()

    ss_p = float(n_i * ((p_means - grand) ** 2).sum())
    ss_i = float(n_p * ((i_means - grand) ** 2).sum())
    ss_pi = float(
        ((cell.merge(p_means.rename("pm"), on="p").merge(i_means.rename("im"), on="i")
          .assign(
              resid=lambda x: x["y"] - x["pm"] - x["im"] + grand
          )["resid"]
          ** 2).sum())
    )
    ms_p = ss_p / (n_p - 1)
    ms_i = ss_i / (n_i - 1)
    ms_pi = ss_pi / ((n_p - 1) * (n_i - 1))
    return ms_p, ms_i, ms_pi, n_p, n_i


def estimate_variance_components_p_x_i(
    y: np.ndarray,
    activation_idx: np.ndarray,
    sample_idx: np.ndarray,
) -> VarianceComponents:
    """Random p × i design; one score per (activation, sample)."""
    ms_p, ms_i, ms_pi, n_p, n_i = _two_way_anova_ms(y, activation_idx, sample_idx)
    sigma2_pi = max(0.0, ms_pi)
    sigma2_p = max(0.0, (ms_p - ms_pi) / n_i)
    sigma2_i = max(0.0, (ms_i - ms_pi) / n_p)
    return VarianceComponents(
        sigma2_p=sigma2_p,
        sigma2_i=sigma2_i,
        sigma2_pi=sigma2_pi,
        ms_p=ms_p,
        ms_i=ms_i,
        ms_pi=ms_pi,
        n_p=n_p,
        n_i=n_i,
    )


def g_rel(vc: VarianceComponents, n_prime: int) -> float:
    """Relative G for comparing activations using mean of n' samples each."""
    denom = vc.sigma2_p + vc.sigma2_pi / n_prime
    if denom <= 0:
        return float("nan")
    return vc.sigma2_p / denom


def phi_abs(vc: VarianceComponents, n_prime: int) -> float:
    """Absolute Φ for activation level using mean of n' samples each."""
    denom = (
        vc.sigma2_p
        + vc.sigma2_i / vc.n_p
        + vc.sigma2_pi / (vc.n_p * n_prime)
    )
    if denom <= 0:
        return float("nan")
    return vc.sigma2_p / denom


def cronbach_alpha_samples(
    y: np.ndarray,
    activation_idx: np.ndarray,
    sample_idx: np.ndarray,
) -> float:
    """Cronbach's α treating samples as items (columns), activations as persons."""
    df = pd.DataFrame({"y": y, "p": activation_idx, "i": sample_idx})
    wide = df.pivot(index="p", columns="i", values="y")
    k = wide.shape[1]
    if k < 2:
        return float("nan")
    item_var = wide.var(axis=0, ddof=1)
    total_var = wide.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def run_g_study(
    y: np.ndarray,
    activation_idx: np.ndarray,
    sample_idx: np.ndarray,
    *,
    n_prime_grid: tuple[int, ...] = (1, 2, 3, 4, 6, 12),
) -> GStudyResult:
    vc = estimate_variance_components_p_x_i(y, activation_idx, sample_idx)
    n_max = vc.n_i
    grid = tuple(sorted({n for n in n_prime_grid if 1 <= n <= n_max}))
    g_map = {n: g_rel(vc, n) for n in grid}
    phi_map = {n: phi_abs(vc, n) for n in grid}
    alpha = cronbach_alpha_samples(y, activation_idx, sample_idx)
    return GStudyResult(vc=vc, g_rel=g_map, phi_abs=phi_map, cronbach_alpha=alpha)
