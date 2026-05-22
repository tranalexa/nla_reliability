"""Train linear probes on Gemma activations to predict SelfDescribe attribute labels.

Assumes row order in CSV matches activations parquet (Step 1, 400-row sample, seed 42).

Usage:
  uv run python scripts/pull_from_modal.py
  uv run python scripts/train_linear_probes.py
  uv run python scripts/pull_from_modal.py --full
  uv run python scripts/train_linear_probes.py \\
    --compare-activations data/activations_layer32_full_gemma-3-12b-pt.parquet
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.activation_utils import load_activation_matrix, load_selfdescribe  # noqa: E402
from nla.paths import DEFAULT_ACTIVATIONS_PERSONA, DEFAULT_CSV  # noqa: E402

ATTR_CLASSES = ("Gender", "Religion", "Occupation", "Country")

DEFAULT_PARQUET = DEFAULT_ACTIVATIONS_PERSONA


def _split_xy(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError as e:
        warnings.warn(
            f"stratified split failed ({e}); using random split without stratify",
            stacklevel=2,
        )
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def majority_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    majority = Counter(y_train.tolist()).most_common(1)[0][0]
    preds = np.full_like(y_test, fill_value=majority)
    return float(accuracy_score(y_test, preds))


def run_task(
    attr_class: str,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    verbose: bool,
) -> dict:
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)
    n_train_min = int(n * (1 - test_size))
    if n_classes > n_train_min / 10:
        warnings.warn(
            f"{attr_class}: {n_classes} classes on ~{n_train_min} train rows — "
            "probe may be weak",
            stacklevel=2,
        )

    X_train, X_test, y_train, y_test = _split_xy(X, y, test_size, seed)
    maj_acc = majority_accuracy(y_train, y_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    probe_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    out = {
        "attr_class": attr_class,
        "n": n,
        "n_classes": n_classes,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "majority_acc": maj_acc,
        "probe_acc": probe_acc,
    }

    if verbose:
        print(f"\n--- {attr_class} confusion matrix (rows=true, cols=pred) ---")
        labels = sorted(classes.tolist())
        cm = confusion_matrix(y_test, clf.predict(X_test), labels=labels)
        print("labels:", labels)
        print(cm)

    return out


def evaluate_parquet(
    parquet_path: Path,
    dataset: list[dict],
    *,
    test_size: float,
    seed: int,
    verbose: bool,
) -> list[dict]:
    X_all = load_activation_matrix(parquet_path)
    if X_all.shape[0] != len(dataset):
        raise ValueError(
            f"{parquet_path}: {X_all.shape[0]} activations vs {len(dataset)} CSV rows"
        )
    rows = []
    for attr_class in ATTR_CLASSES:
        idx = [i for i, r in enumerate(dataset) if r["attribute_class"] == attr_class]
        if not idx:
            warnings.warn(f"no rows for attr_class={attr_class!r}", stacklevel=2)
            continue
        X = X_all[idx]
        y = np.array([dataset[i]["label"] for i in idx])
        rows.append(run_task(attr_class, X, y, test_size, seed, verbose))
    return rows


def print_results(rows: list[dict], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(label)
    print(
        f"{'attr_class':<12} {'n':>4} {'classes':>7} {'train':>6} {'test':>5} "
        f"{'majority':>9} {'probe':>9}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"{r['attr_class']:<12} {r['n']:>4} {r['n_classes']:>7} "
            f"{r['train_n']:>6} {r['test_n']:>5} "
            f"{r['majority_acc']:>9.3f} {r['probe_acc']:>9.3f}"
        )
    if rows:
        macro_maj = float(np.mean([r["majority_acc"] for r in rows]))
        macro_probe = float(np.mean([r["probe_acc"] for r in rows]))
        print("-" * 72)
        print(
            f"{'overall (macro)':<12} {'':>4} {'':>7} {'':>6} {'':>5} "
            f"{macro_maj:>9.3f} {macro_probe:>9.3f}"
        )


def print_compare(primary: list[dict], baseline: list[dict], primary_name: str, baseline_name: str) -> None:
    base_by = {r["attr_class"]: r for r in baseline}
    print(f"\n{'=' * 72}")
    print(f"compare: primary={primary_name}  baseline={baseline_name}")
    print(
        f"{'attr_class':<12} {'probe_pri':>10} {'probe_base':>10} {'delta':>8}"
    )
    print("-" * 72)
    deltas = []
    for r in primary:
        b = base_by.get(r["attr_class"])
        if not b:
            continue
        delta = r["probe_acc"] - b["probe_acc"]
        deltas.append(delta)
        print(
            f"{r['attr_class']:<12} {r['probe_acc']:>10.3f} {b['probe_acc']:>10.3f} "
            f"{delta:>+8.3f}"
        )
    if deltas:
        print("-" * 72)
        print(f"{'mean delta':<12} {'':>10} {'':>10} {float(np.mean(deltas)):>+8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--activations",
        type=Path,
        default=DEFAULT_PARQUET,
        help="activations parquet from Step 1 (default: persona-only tagged file)",
    )
    p.add_argument(
        "--compare-activations",
        type=Path,
        default=None,
        help="second parquet for side-by-side probe comparison (e.g. data/activations_layer32_full_…parquet)",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dataset = load_selfdescribe(args.csv)
    rows = evaluate_parquet(
        args.activations,
        dataset,
        test_size=args.test_size,
        seed=args.seed,
        verbose=args.verbose,
    )
    print_results(rows, f"activations: {args.activations.resolve()}")

    if args.compare_activations is not None:
        baseline_rows = evaluate_parquet(
            args.compare_activations,
            dataset,
            test_size=args.test_size,
            seed=args.seed,
            verbose=False,
        )
        print_results(baseline_rows, f"activations: {args.compare_activations.resolve()}")
        print_compare(
            rows,
            baseline_rows,
            str(args.activations.name),
            str(args.compare_activations.name),
        )

    print(
        "\nCountry in SelfDescribe corresponds to nationality. "
        "Last-token layer-32 residual; compare extraction parquets from Step 1."
    )


if __name__ == "__main__":
    main()
