#!/usr/bin/env python3
"""Linear probe for MMLU: original vs mean-reconstructed activations.

Trains a multinomial logistic-regression probe to predict the MMLU ``subject``
label (4 classes: abstract_algebra, moral_scenarios, virology, astronomy) from
either:
  1. The original Step-1 activation vector, or
  2. The mean of the 12 Step-3 reconstructions for the same activation.

Both probes share the same train/test split (seed 42, 80/20 stratified) so
they are directly comparable. A majority-class baseline is reported alongside.

Usage:
  uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_choice
  uv run python scripts/train_linear_probe_mmlu.py --run-id mmlu_nochoice --output reports/linear_probe_mmlu_nochoice.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import (  # noqa: E402
    local_activations_path,
    local_csv_path,
    local_recon_vectors_path,
)


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def _load_originals(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    if "activation_vector" not in df.columns:
        raise ValueError(f"{path} missing activation_vector column")
    return np.asarray(df["activation_vector"].tolist(), dtype=np.float32)


def _load_recon_means(path: Path, n_items: int) -> np.ndarray:
    df = pd.read_parquet(path)
    required = {"activation_idx", "recon_vector"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns: {required}")

    sums = None
    counts = np.zeros(n_items, dtype=np.int32)
    for idx, vec in zip(df["activation_idx"].to_numpy(), df["recon_vector"].to_list()):
        i = int(idx)
        if i < 0 or i >= n_items:
            raise ValueError(f"activation_idx out of range: {i} (n_items={n_items})")
        v = np.asarray(vec, dtype=np.float32)
        if sums is None:
            sums = np.zeros((n_items, v.shape[0]), dtype=np.float32)
        sums[i] += v
        counts[i] += 1

    if sums is None:
        raise ValueError(f"{path} has no recon rows")
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"missing recon rows for activation_idx values: {missing[:10]}")
    return sums / counts[:, None]


def _majority_acc(y_train: np.ndarray, y_test: np.ndarray) -> float:
    labels, counts = np.unique(y_train, return_counts=True)
    majority = labels[int(np.argmax(counts))]
    return float(accuracy_score(y_test, np.full_like(y_test, majority)))


def _fit_and_score(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_iter: int,
    c: float,
) -> float:
    clf = LogisticRegression(max_iter=max_iter, C=c)
    clf.fit(X_train, y_train)
    return float(accuracy_score(y_test, clf.predict(X_test)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-id",
        default="mmlu_choice",
        choices=["mmlu_choice", "mmlu_nochoice"],
        help="which MMLU run to probe",
    )
    p.add_argument("--target", default="subject", help="CSV column to predict (default: subject)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42, help="train/test split seed (default: 42)")
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--no-l2-normalize", action="store_true")
    p.add_argument("--output", type=Path, default=None, help="optional CSV output path")
    args = p.parse_args()

    csv_path = local_csv_path(args.run_id, 400)
    act_path = local_activations_path(args.run_id)
    recon_path = local_recon_vectors_path(args.run_id)

    csv_df = pd.read_csv(csv_path)
    if args.target not in csv_df.columns:
        raise ValueError(f"{csv_path} missing target column: {args.target}")
    y = csv_df[args.target].astype(str).to_numpy()
    n_items = len(csv_df)

    X_orig = _load_originals(act_path)
    if X_orig.shape[0] != n_items:
        raise ValueError(f"{act_path} rows={X_orig.shape[0]} != csv rows={n_items}")
    X_recon = _load_recon_means(recon_path, n_items=n_items)

    if not args.no_l2_normalize:
        X_orig = _l2_normalize_rows(X_orig)
        X_recon = _l2_normalize_rows(X_recon)

    idx = np.arange(n_items)
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    y_train, y_test = y[train_idx], y[test_idx]

    majority = _majority_acc(y_train, y_test)
    orig_acc = _fit_and_score(
        X_orig[train_idx], X_orig[test_idx], y_train, y_test,
        max_iter=args.max_iter, c=args.c,
    )
    recon_acc = _fit_and_score(
        X_recon[train_idx], X_recon[test_idx], y_train, y_test,
        max_iter=args.max_iter, c=args.c,
    )

    out_df = pd.DataFrame(
        [
            {
                "run_id": args.run_id,
                "dataset": "mmlu",
                "target": args.target,
                "vector_source": "original",
                "n": n_items,
                "n_classes": int(np.unique(y).size),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "majority_acc": majority,
                "probe_acc": orig_acc,
            },
            {
                "run_id": args.run_id,
                "dataset": "mmlu",
                "target": args.target,
                "vector_source": "recon_mean",
                "n": n_items,
                "n_classes": int(np.unique(y).size),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "majority_acc": majority,
                "probe_acc": recon_acc,
            },
        ]
    )

    print(f"\nMMLU linear probe ({args.run_id}, seed={args.seed}, same split for both sources)")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDelta (recon_mean - original): {recon_acc - orig_acc:+.4f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False, float_format="%.6f")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
