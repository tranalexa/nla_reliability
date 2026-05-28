"""Centered fidelity / consistency metrics + linear probes + summary tables.

This module is the single source of truth for the per-run reliability metrics
consumed by ``scripts/build_synthesis_tables.py`` and the synthesis notebook.

It reads Step-1 originals (``activations_layer32_*.parquet``) and Step-3 raw
reconstructed vectors (``recon_vectors_*.parquet``), removes the shared mean
direction, and reports both fidelity (matched vs mismatched) and consistency
(within-item vs between-item) statistics.

Random seed: ``RNG_SEED = 0`` for the mismatch / between-item samplers
(centered diagnostics) and ``seed = 42`` for the linear-probe train/test split.
Override seeds via ``compute_run_metrics(seed=…)`` or ``run_probes(seed=…)``.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nla.paths import (
    dataset_for_run,
    local_activations_path,
    local_csv_path,
    local_recon_vectors_path,
    local_text_consistency_mpnet_path,
)

# ── Constants ─────────────────────────────────────────────────────────────────

#: Number of mismatched originals drawn per reconstruction for fidelity sampling.
N_MISMATCHES_PER_RECON = 5

#: Between-item baseline samples = factor x number of within-item pairs.
N_BETWEEN_FACTOR = 5

#: Default RNG seed for centered metric sampling.
RNG_SEED = 0

#: Default RNG seed for the linear-probe train/test split.
PROBE_SEED = 42

#: Headline metrics surfaced by build_synthesis_tables.py.
HEADLINE_METRICS = [
    "Fidelity centered (matched)",
    "Fidelity centered (mismatched)",
    "Fidelity centered gap",
    "Consistency centered (within-item)",
    "Consistency centered (between-item)",
    "Consistency centered gap",
    "||mean activation||",
]


# ── Vector loading ────────────────────────────────────────────────────────────

def _load_vec_col(path: Path, col: str) -> np.ndarray:
    return np.array(pd.read_parquet(path)[col].tolist(), dtype=np.float32)


def load_activations(path: Path) -> np.ndarray:
    """Load Step-1 activation matrix (N, D)."""
    return _load_vec_col(path, "activation_vector")


def load_recon(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Step-3 reconstructed vectors. Returns (recon_raw, activation_idx, sample_idx)."""
    df = (
        pd.read_parquet(path)
        .sort_values(["activation_idx", "sample_idx"])
        .reset_index(drop=True)
    )
    return (
        np.array(df["recon_vector"].tolist(), dtype=np.float32),
        df["activation_idx"].to_numpy(),
        df["sample_idx"].to_numpy(),
    )


# ── Math helpers ──────────────────────────────────────────────────────────────

def unit_norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize a matrix."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms < 1e-12, 1.0, norms)


def center(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Subtract a global mean vector from each row."""
    return x - mu[None, :]


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


# ── Metric computation ────────────────────────────────────────────────────────

def _sample_mismatches(
    act_idx: np.ndarray,
    n_orig: int,
    n_per: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return shape (n_recon, n_per) mismatch indices, all != act_idx (no collisions)."""
    n_recon = len(act_idx)
    j = rng.integers(0, n_orig - 1, size=(n_recon, n_per))
    # Shift candidate indices >= true index up by one so j != act_idx pairwise.
    for k in range(n_per):
        j[:, k] = np.where(j[:, k] >= act_idx, j[:, k] + 1, j[:, k])
    return j


def compute_fidelity(
    originals: np.ndarray,
    recon_raw: np.ndarray,
    act_idx: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Raw + centered fidelity arrays per reconstruction sample.

    Returns keys raw_matched, raw_mismatched, raw_gap, cen_matched, cen_mismatched, cen_gap.
    """
    mu = originals.mean(axis=0)
    recon_hat = unit_norm(recon_raw)
    cen_orig_hat = unit_norm(center(originals, mu))
    cen_recon_hat = unit_norm(center(recon_hat, mu))

    raw_matched = (originals[act_idx] * recon_hat).sum(axis=-1)
    cen_matched = (cen_orig_hat[act_idx] * cen_recon_hat).sum(axis=-1)

    mis = _sample_mismatches(act_idx, len(originals), N_MISMATCHES_PER_RECON, rng)
    raw_mis = np.stack(
        [(originals[mis[:, k]] * recon_hat).sum(axis=-1) for k in range(N_MISMATCHES_PER_RECON)],
        axis=1,
    )
    cen_mis = np.stack(
        [(cen_orig_hat[mis[:, k]] * cen_recon_hat).sum(axis=-1) for k in range(N_MISMATCHES_PER_RECON)],
        axis=1,
    )
    raw_mismatched = raw_mis.ravel()
    cen_mismatched = cen_mis.ravel()

    raw_gap = np.repeat(raw_matched, N_MISMATCHES_PER_RECON) - raw_mismatched
    cen_gap = np.repeat(cen_matched, N_MISMATCHES_PER_RECON) - cen_mismatched

    return {
        "raw_matched": raw_matched,
        "raw_mismatched": raw_mismatched,
        "raw_gap": raw_gap,
        "cen_matched": cen_matched,
        "cen_mismatched": cen_mismatched,
        "cen_gap": cen_gap,
    }


def compute_consistency(
    originals: np.ndarray,
    recon_raw: np.ndarray,
    act_idx: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Raw + centered within-item / between-item consistency arrays.

    Returns keys raw_within, raw_between, raw_gap, cen_within, cen_between, cen_gap.
    """
    n_orig, D = originals.shape[0], recon_raw.shape[1]
    mu = originals.mean(axis=0)

    recon_hat = unit_norm(recon_raw)
    cen_recon_hat = unit_norm(center(recon_hat, mu))

    # Reshape relies on rows being sorted by (activation_idx, sample_idx).
    rh = recon_hat.reshape(n_orig, n_samples, D)
    cr = cen_recon_hat.reshape(n_orig, n_samples, D)

    pairs = list(combinations(range(n_samples), 2))
    s1 = np.array([p[0] for p in pairs])
    s2 = np.array([p[1] for p in pairs])

    raw_within, cen_within = [], []
    for i in range(n_orig):
        raw_within.append((rh[i][s1] * rh[i][s2]).sum(axis=-1))
        cen_within.append((cr[i][s1] * cr[i][s2]).sum(axis=-1))
    raw_within_arr = np.concatenate(raw_within)
    cen_within_arr = np.concatenate(cen_within)

    n_within = len(raw_within_arr)
    n_between = n_within * N_BETWEEN_FACTOR

    ai = rng.integers(0, n_orig, size=n_between)
    aj = rng.integers(0, n_orig - 1, size=n_between)
    aj = np.where(aj >= ai, aj + 1, aj)
    si = rng.integers(0, n_samples, size=n_between)
    sj = rng.integers(0, n_samples, size=n_between)

    raw_between = (rh[ai, si] * rh[aj, sj]).sum(axis=-1)
    cen_between = (cr[ai, si] * cr[aj, sj]).sum(axis=-1)

    bi = rng.integers(0, n_between, size=n_within)
    raw_gap = raw_within_arr - raw_between[bi]
    cen_gap = cen_within_arr - cen_between[bi]

    return {
        "raw_within": raw_within_arr,
        "raw_between": raw_between,
        "raw_gap": raw_gap,
        "cen_within": cen_within_arr,
        "cen_between": cen_between,
        "cen_gap": cen_gap,
    }


# ── Summary statistics ────────────────────────────────────────────────────────

def describe(arr: np.ndarray) -> dict[str, float]:
    """Five-number summary for a 1-D numeric array."""
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


METRIC_ORDER: list[tuple[str, str, str]] = [
    ("fidelity",    "raw_matched",    "Fidelity raw (matched)"),
    ("fidelity",    "raw_mismatched", "Fidelity raw (mismatched)"),
    ("fidelity",    "raw_gap",        "Fidelity raw gap"),
    ("fidelity",    "cen_matched",    "Fidelity centered (matched)"),
    ("fidelity",    "cen_mismatched", "Fidelity centered (mismatched)"),
    ("fidelity",    "cen_gap",        "Fidelity centered gap"),
    ("consistency", "raw_within",     "Consistency raw (within-item)"),
    ("consistency", "raw_between",    "Consistency raw (between-item)"),
    ("consistency", "raw_gap",        "Consistency raw gap"),
    ("consistency", "cen_within",     "Consistency centered (within-item)"),
    ("consistency", "cen_between",    "Consistency centered (between-item)"),
    ("consistency", "cen_gap",        "Consistency centered gap"),
    ("inflation",   "mean_vec_norm",  "||mean activation||"),
]


# ── Per-run driver ────────────────────────────────────────────────────────────

def compute_run_metrics(
    run_id: str,
    *,
    seed: int = RNG_SEED,
) -> tuple[dict[str, Any], dict[str, Any]] | tuple[dict, dict]:
    """Compute centered fidelity + consistency stats for one run_id.

    Returns ``(stats, raw_arrays)`` where ``stats`` is a nested dict of summary
    statistics by metric group and ``raw_arrays`` holds the underlying per-sample
    arrays. Both are empty if the run has no Step-1 activations on disk.
    """
    act_path = local_activations_path(run_id)
    recon_path = local_recon_vectors_path(run_id)

    if not act_path.exists():
        return {}, {}

    rng = np.random.default_rng(seed)

    originals = load_activations(act_path)
    mu = originals.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mu))
    stats: dict[str, Any] = {
        "inflation": {"mean_vec_norm": describe(np.array([mean_vec_norm]))}
    }
    raw_arrays: dict[str, Any] = {}

    if recon_path.exists():
        recon_raw, act_idx, sample_idx = load_recon(recon_path)
        n_samples = int(sample_idx.max()) + 1

        fid = compute_fidelity(originals, recon_raw, act_idx, rng)
        stats["fidelity"] = {k: describe(v) for k, v in fid.items()}
        raw_arrays["fidelity"] = fid

        con = compute_consistency(originals, recon_raw, act_idx, n_samples, rng)
        stats["consistency"] = {k: describe(v) for k, v in con.items()}
        raw_arrays["consistency"] = con

    return stats, raw_arrays


def headline_metrics_for_run(run_id: str, *, seed: int = RNG_SEED) -> dict[str, dict[str, float]]:
    """Extract just the seven HEADLINE_METRICS rows for a run_id."""
    stats, _ = compute_run_metrics(run_id, seed=seed)
    label_map = {
        ("fidelity", "cen_matched"): "Fidelity centered (matched)",
        ("fidelity", "cen_mismatched"): "Fidelity centered (mismatched)",
        ("fidelity", "cen_gap"): "Fidelity centered gap",
        ("consistency", "cen_within"): "Consistency centered (within-item)",
        ("consistency", "cen_between"): "Consistency centered (between-item)",
        ("consistency", "cen_gap"): "Consistency centered gap",
        ("inflation", "mean_vec_norm"): "||mean activation||",
    }
    out: dict[str, dict[str, float]] = {}
    for (grp, key), label in label_map.items():
        if grp in stats and key in stats[grp]:
            out[label] = stats[grp][key]
    return out


# ── Summary CSV + LaTeX builders ──────────────────────────────────────────────

def build_summary_df(all_stats: dict[str, dict]) -> pd.DataFrame:
    """Build a long-format DataFrame of (run_id, metric, n/mean/std/p5/median/p95)."""
    rows = []
    for run_id, stats in all_stats.items():
        for grp, key, label in METRIC_ORDER:
            if grp not in stats or key not in stats[grp]:
                continue
            rows.append({"run_id": run_id, "metric": label, **stats[grp][key]})
    return pd.DataFrame(rows)


def write_latex_table(
    all_stats: dict[str, dict],
    run_labels: dict[str, str],
    path: Path,
) -> None:
    """Emit a LaTeX results table summarising centered fidelity + consistency.

    ``run_labels`` maps each run_id present in ``all_stats`` to its display name.
    Rows kept: centered matched / mismatched / gap fidelity, centered within /
    between / gap consistency, plus ||mu||.
    """
    paper_rows = [
        ("fidelity",    "cen_matched",   r"Centered fidelity (matched)"),
        ("fidelity",    "cen_mismatched", r"Centered fidelity (mismatched)"),
        ("fidelity",    "cen_gap",        r"\textbf{Centered fidelity gap} $\uparrow$"),
        ("consistency", "cen_within",     r"Centered consistency (within-item)"),
        ("consistency", "cen_between",    r"Centered consistency (between-item)"),
        ("consistency", "cen_gap",        r"\textbf{Centered consistency gap} $\uparrow$"),
        ("inflation",   "mean_vec_norm",  r"$\|\mu\|$ (mean activation norm)"),
    ]

    run_ids = [r for r in run_labels if r in all_stats]
    if not run_ids:
        return

    col_header = " & ".join(run_labels[r] for r in run_ids)
    n_runs = len(run_ids)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{NLA reliability results across runs. "
            r"Metrics are mean-centered cosine similarities (mean $\pm$ std, $n{=}400$ items). "
            r"Higher gap $\uparrow$ indicates stronger reliability.}"
        ),
        r"\label{tab:nla_reliability}",
        r"\begin{tabular}{l" + "c" * n_runs + r"}",
        r"\toprule",
        f"Metric & {col_header} \\\\",
        r"\midrule",
    ]

    prev_grp: str | None = None
    for grp, key, label in paper_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r"\midrule")
        prev_grp = grp
        cells = []
        for r in run_ids:
            d = all_stats.get(r, {}).get(grp, {}).get(key)
            cells.append(f"{d['mean']:.3f} $\\pm$ {d['std']:.3f}" if d else "---")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


# ── Text-space consistency aggregator ─────────────────────────────────────────

def text_consistency_summary(run_id: str) -> dict[str, float] | None:
    """Five-number summary of the per-activation within-item MPNet cosine."""
    path = local_text_consistency_mpnet_path(run_id)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "mean_pairwise_text_cosine" not in df.columns:
        return None
    s = df["mean_pairwise_text_cosine"].astype(float)
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "p5": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


# ── Linear probes (majority baseline + logistic regression) ───────────────────

def _load_originals_for_probe(path: Path) -> np.ndarray:
    return np.asarray(pd.read_parquet(path)["activation_vector"].tolist(), dtype=np.float32)


def _load_recon_means(path: Path, n_items: int) -> np.ndarray:
    """Average the K reconstructions per activation_idx into one (N, D) matrix."""
    df = pd.read_parquet(path)
    sums = None
    counts = np.zeros(n_items, dtype=np.int32)
    for idx, vec in zip(df["activation_idx"].to_numpy(), df["recon_vector"].to_list()):
        i = int(idx)
        v = np.asarray(vec, dtype=np.float32)
        if sums is None:
            sums = np.zeros((n_items, v.shape[0]), dtype=np.float32)
        sums[i] += v
        counts[i] += 1
    if sums is None or np.any(counts == 0):
        raise ValueError(f"incomplete recon vectors in {path}")
    return sums / counts[:, None]


def _majority_acc(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of the always-predict-most-common-train-label classifier."""
    labels, counts = np.unique(y_train, return_counts=True)
    majority = labels[int(np.argmax(counts))]
    return float(accuracy_score(y_test, np.full_like(y_test, majority)))


def _probe_acc(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
) -> dict[str, float]:
    idx = np.arange(len(y))
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed
        )
    y_train, y_test = y[train_idx], y[test_idx]
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X[train_idx], y_train)
    return {
        "majority_acc": _majority_acc(y_train, y_test),
        "probe_acc": float(accuracy_score(y_test, clf.predict(X[test_idx]))),
    }


def run_probes(
    run_id: str,
    targets: list[str],
    *,
    n_items: int = 400,
    seed: int = PROBE_SEED,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Train + score the majority baseline and logistic probe on original vs mean-recon."""
    csv_path = local_csv_path(run_id, n_items)
    act_path = local_activations_path(run_id)
    recon_path = local_recon_vectors_path(run_id)
    if not all(p.exists() for p in (csv_path, act_path, recon_path)):
        return pd.DataFrame()

    csv_df = pd.read_csv(csv_path)
    y_by_target = {t: csv_df[t].astype(str).to_numpy() for t in targets if t in csv_df.columns}
    n = len(csv_df)
    X_orig = _l2_normalize_rows(_load_originals_for_probe(act_path))
    X_recon = _l2_normalize_rows(_load_recon_means(recon_path, n_items=n))

    dataset = dataset_for_run(run_id)
    rows: list[dict[str, Any]] = []
    for target, y in y_by_target.items():
        for source, X in (("original", X_orig), ("recon_mean", X_recon)):
            scores = _probe_acc(X, y, seed=seed, test_size=test_size)
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "vector_source": source,
                    "n": n,
                    "n_classes": int(np.unique(y).size),
                    **scores,
                }
            )
    return pd.DataFrame(rows)


# ── Misc display helpers ──────────────────────────────────────────────────────

def prompt_format_label(csv_path: Path) -> str:
    """Heuristic label for the prompt_text format used in a run's CSV."""
    if not csv_path.exists():
        return "missing"
    sample = str(pd.read_csv(csv_path, nrows=1)["prompt_text"].iloc[0])
    if "\nA." in sample or sample.startswith("Question:"):
        return "MCQ (question + A-D + Answer:)"
    return "question stem only"
