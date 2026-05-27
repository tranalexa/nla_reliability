#!/usr/bin/env python3
"""
Train linear probes on Gemma activation vectors to predict dataset labels.

Datasets:
  - PRISM:        predicts `gender`
  - Bias in Bios: predicts `profession` and `gender`
  - MMLU:         predicts `subject`

Vector sources (--vector-source):
  original  Step 1 activations (activation_vector), row order = CSV
  recon     Step 3: mean of 12 recon vectors per activation_idx, L2-normalized
  compare   run both and print orig vs recon probe accuracy

Usage:
  uv run python scripts/train_linear_probes_multi_dataset.py --vector-source original
  uv run python scripts/train_linear_probes_multi_dataset.py --vector-source recon
  uv run python scripts/train_linear_probes_multi_dataset.py --vector-source compare
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import recon_vectors_filename  # noqa: E402

VECTOR_SOURCES = ("original", "recon", "compare")


@dataclass(frozen=True)
class DatasetSpec:
    csv_path: Path
    activations_path: Path
    recon_vectors_path: Path
    default_targets: tuple[str, ...]


def load_activation_matrix(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    if "activation_vector" not in df.columns:
        raise ValueError(
            f"{parquet_path} must have `activation_vector` column; got {df.columns.tolist()}"
        )
    col = df["activation_vector"]
    return np.stack(col.to_list()).astype(np.float32)


def load_mean_recon_matrix(parquet_path: Path, *, l2_normalize: bool = True) -> np.ndarray:
    """Mean of K recon vectors per activation_idx; rows sorted by activation_idx."""
    df = pd.read_parquet(parquet_path)
    missing = {"activation_idx", "sample_idx", "recon_vector"} - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} missing columns {missing}")
    df = df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)
    vecs = np.stack(df["recon_vector"].to_list()).astype(np.float32)
    act_idx = df["activation_idx"].to_numpy()
    unique = np.sort(np.unique(act_idx))
    rows = [vecs[act_idx == a].mean(axis=0) for a in unique]
    X = np.stack(rows).astype(np.float32)
    if l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-12, None)
    return X


def majority_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    majority = Counter(y_train.tolist()).most_common(1)[0][0]
    preds = np.full_like(y_test, fill_value=majority)
    return float(accuracy_score(y_test, preds))


def split_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(object)
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


def run_task(
    *,
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    target: str,
    vector_source: str,
    test_size: float,
    seed: int,
    verbose: bool,
) -> dict:
    target_name = f"{dataset}.{target}"
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    n_train_min = int(n * (1 - test_size))
    if n_classes > n_train_min / 10:
        warnings.warn(
            f"{target_name} [{vector_source}]: {n_classes} classes on ~{n_train_min} train rows",
            stacklevel=2,
        )

    X_train, X_test, y_train, y_test = split_xy(X, y, test_size=test_size, seed=seed)
    maj_acc = majority_accuracy(y_train, y_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    probe_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    if verbose:
        print(f"\n--- {vector_source} {target_name} confusion (rows=true, cols=pred) ---")
        labels = sorted(classes.tolist())
        cm = confusion_matrix(y_test, clf.predict(X_test), labels=labels)
        print("labels:", labels)
        print(cm)

    return {
        "dataset": dataset,
        "target": target,
        "vector_source": vector_source,
        "n": int(n),
        "n_classes": int(n_classes),
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "majority_acc": float(maj_acc),
        "probe_acc": float(probe_acc),
    }


def print_results(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(
        f"{'target':<22} {'n':>4} {'classes':>7} {'train':>6} {'test':>5} "
        f"{'majority':>9} {'probe':>9}"
    )
    print("-" * 72)
    for r in rows:
        label = f"{r['dataset']}.{r['target']}"
        print(
            f"{label:<22} {r['n']:>4} {r['n_classes']:>7} "
            f"{r['train_n']:>6} {r['test_n']:>5} "
            f"{r['majority_acc']:>9.3f} {r['probe_acc']:>9.3f}"
        )
    if rows:
        print("-" * 72)
        print(
            f"{'macro':<22} {'':>4} {'':>7} {'':>6} {'':>5} "
            f"{float(np.mean([r['majority_acc'] for r in rows])):>9.3f} "
            f"{float(np.mean([r['probe_acc'] for r in rows])):>9.3f}"
        )


def print_compare(orig_rows: list[dict], recon_rows: list[dict], dataset: str) -> None:
    recon_by = {(r["dataset"], r["target"]): r for r in recon_rows}
    print(f"\n{'=' * 72}")
    print(f"Compare original vs mean-recon probes: {dataset}")
    print(f"{'target':<22} {'orig':>8} {'recon':>8} {'delta':>8}")
    print("-" * 72)
    deltas = []
    for o in orig_rows:
        r = recon_by.get((o["dataset"], o["target"]))
        if not r:
            continue
        delta = r["probe_acc"] - o["probe_acc"]
        deltas.append(delta)
        label = f"{o['dataset']}.{o['target']}"
        print(f"{label:<22} {o['probe_acc']:>8.3f} {r['probe_acc']:>8.3f} {delta:>+8.3f}")
    if deltas:
        print("-" * 72)
        print(f"{'mean delta (recon - orig)':<22} {'':>8} {'':>8} {float(np.mean(deltas)):>+8.3f}")


def spec_for_dataset(dataset: str, data_dir: Path) -> DatasetSpec:
    if dataset == "prism":
        return DatasetSpec(
            csv_path=data_dir / "prism_400.csv",
            activations_path=data_dir / "activations_layer32_prism_gemma-3-12b-pt.parquet",
            recon_vectors_path=data_dir / recon_vectors_filename("prism"),
            default_targets=("gender",),
        )
    if dataset == "biosbias":
        return DatasetSpec(
            csv_path=data_dir / "biosbias_400.csv",
            activations_path=data_dir / "activations_layer32_biosbias_gemma-3-12b-pt.parquet",
            recon_vectors_path=data_dir / recon_vectors_filename("biosbias"),
            default_targets=("profession", "gender"),
        )
    if dataset == "mmlu":
        return DatasetSpec(
            csv_path=data_dir / "mmlu_400.csv",
            activations_path=data_dir / "activations_layer32_mmlu_gemma-3-12b-pt.parquet",
            recon_vectors_path=data_dir / recon_vectors_filename("mmlu"),
            default_targets=("subject",),
        )
    raise ValueError(f"unknown dataset {dataset!r}; expected prism|biosbias|mmlu")


def evaluate_dataset(
    ds: str,
    spec: DatasetSpec,
    *,
    vector_source: str,
    targets: tuple[str, ...],
    test_size: float,
    seed: int,
    verbose: bool,
) -> list[dict]:
    if not spec.csv_path.exists():
        raise FileNotFoundError(spec.csv_path)

    df = pd.read_csv(spec.csv_path)
    if len(df) == 0:
        raise ValueError(f"{spec.csv_path} is empty")

    if vector_source == "original":
        if not spec.activations_path.exists():
            raise FileNotFoundError(spec.activations_path)
        X = load_activation_matrix(spec.activations_path)
        src_label = "original activations"
    else:
        if not spec.recon_vectors_path.exists():
            raise FileNotFoundError(spec.recon_vectors_path)
        X = load_mean_recon_matrix(spec.recon_vectors_path)
        src_label = "mean recon (12 samples, L2-norm)"

    if X.shape[0] != len(df):
        raise ValueError(f"{ds}: {X.shape[0]} vectors vs {len(df)} csv rows")

    rows: list[dict] = []
    for target in targets:
        if target not in df.columns:
            raise ValueError(
                f"{ds} csv missing {target!r}; columns: {df.columns.tolist()}"
            )
        y = df[target].astype(str).to_numpy()
        rows.append(
            run_task(
                X=X,
                y=y,
                dataset=ds,
                target=target,
                vector_source=vector_source,
                test_size=test_size,
                seed=seed,
                verbose=verbose,
            )
        )

    print_results(rows, title=f"Linear probe on {ds} — {src_label}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="all", choices=("all", "prism", "biosbias", "mmlu"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "data")
    p.add_argument(
        "--vector-source",
        default="original",
        choices=VECTOR_SOURCES,
        help="original | recon (mean per activation) | compare (both)",
    )
    p.add_argument("--targets", default=None, help="comma-separated target columns")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=Path, default=None, help="optional output CSV")
    args = p.parse_args()

    datasets = ["prism", "biosbias", "mmlu"] if args.dataset == "all" else [args.dataset]
    sources = ("original", "recon") if args.vector_source == "compare" else (args.vector_source,)

    out_rows: list[dict] = []
    for ds in datasets:
        spec = spec_for_dataset(ds, args.data_dir)
        targets = (
            tuple(t.strip() for t in args.targets.split(",") if t.strip())
            if args.targets
            else spec.default_targets
        )

        if args.vector_source == "compare":
            orig_rows = evaluate_dataset(
                ds, spec, vector_source="original", targets=targets,
                test_size=args.test_size, seed=args.seed, verbose=args.verbose,
            )
            recon_rows = evaluate_dataset(
                ds, spec, vector_source="recon", targets=targets,
                test_size=args.test_size, seed=args.seed, verbose=args.verbose,
            )
            print_compare(orig_rows, recon_rows, ds)
            out_rows.extend(orig_rows)
            out_rows.extend(recon_rows)
        else:
            out_rows.extend(
                evaluate_dataset(
                    ds, spec, vector_source=args.vector_source, targets=targets,
                    test_size=args.test_size, seed=args.seed, verbose=args.verbose,
                )
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
