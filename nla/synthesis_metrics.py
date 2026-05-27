"""Helpers to aggregate NLA reliability metrics across datasets and data roots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nla.paths import (
    activations_filename,
    csv_filename,
    local_text_consistency_mpnet_path,
    recon_vectors_filename,
)

HEADLINE_METRICS = [
    "Fidelity centered (matched)",
    "Fidelity centered (mismatched)",
    "Fidelity centered gap",
    "Consistency centered (within-item)",
    "Consistency centered (between-item)",
    "Consistency centered gap",
    "||mean activation||",
]


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def _load_originals(path: Path) -> np.ndarray:
    return np.asarray(pd.read_parquet(path)["activation_vector"].tolist(), dtype=np.float32)


def _load_recon_means(path: Path, n_items: int) -> np.ndarray:
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
    labels, counts = np.unique(y_train, return_counts=True)
    majority = labels[int(np.argmax(counts))]
    return float(accuracy_score(y_test, np.full_like(y_test, majority)))


def _probe_acc(X: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> dict[str, float]:
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


def recon_metrics_from_dir(data_dir: Path, dataset: str, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.make_report_tables import process_dataset

    stats, _ = process_dataset(dataset, data_dir, rng)
    out: dict[str, dict[str, float]] = {}
    label_map = {
        ("fidelity", "cen_matched"): "Fidelity centered (matched)",
        ("fidelity", "cen_mismatched"): "Fidelity centered (mismatched)",
        ("fidelity", "cen_gap"): "Fidelity centered gap",
        ("consistency", "cen_within"): "Consistency centered (within-item)",
        ("consistency", "cen_between"): "Consistency centered (between-item)",
        ("consistency", "cen_gap"): "Consistency centered gap",
        ("inflation", "mean_vec_norm"): "||mean activation||",
    }
    for (grp, key), label in label_map.items():
        if grp in stats and key in stats[grp]:
            out[label] = stats[grp][key]
    return out


def headline_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def text_consistency_summary(data_dir: Path, dataset: str) -> dict[str, float] | None:
    path = data_dir / local_text_consistency_mpnet_path(dataset).name
    if not path.exists():
        path = local_text_consistency_mpnet_path(dataset)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_csv(path)
    col = "mean_pairwise_text_cosine"
    s = df[col].astype(float)
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "p5": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def run_probes(
    data_dir: Path,
    dataset: str,
    targets: list[str],
    *,
    n_items: int = 400,
    seed: int = 42,
    test_size: float = 0.2,
) -> pd.DataFrame:
    csv_path = data_dir / csv_filename(dataset, n_items)
    act_path = data_dir / activations_filename(dataset)
    recon_path = data_dir / recon_vectors_filename(dataset)
    if not all(p.exists() for p in (csv_path, act_path, recon_path)):
        return pd.DataFrame()

    csv_df = pd.read_csv(csv_path)
    y_by_target = {t: csv_df[t].astype(str).to_numpy() for t in targets if t in csv_df.columns}
    n = len(csv_df)
    X_orig = _l2_normalize_rows(_load_originals(act_path))
    X_recon = _l2_normalize_rows(_load_recon_means(recon_path, n_items=n))

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


def prompt_format_label(csv_path: Path) -> str:
    if not csv_path.exists():
        return "missing"
    sample = str(pd.read_csv(csv_path, nrows=1)["prompt_text"].iloc[0])
    if "\nA." in sample or sample.startswith("Question:"):
        return "MCQ (question + A–D + Answer:)"
    return "question stem only"
