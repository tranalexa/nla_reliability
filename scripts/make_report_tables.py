#!/usr/bin/env python3
"""Report tables and figures from local NLA reliability artifacts.

Reads data/ outputs for PRISM and BiosBias and produces:
  reports/summary_stats.csv       — per-dataset metric summary statistics
  reports/results_table.tex       — LaTeX table for the paper
  figures/fidelity_dist.png       — centered fidelity distributions by dataset
  figures/consistency_dist.png    — within-item vs between-item consistency
  figures/raw_vs_centered.png     — raw vs centered gap bar chart
  figures/cosine_inflation.png    — cosine inflation diagnostic

Usage:
  uv run python scripts/make_report_tables.py
  uv run python scripts/make_report_tables.py --data-dir data --out-dir reports --fig-dir figures

Required local artifacts (Step F: pull_from_modal.py --all-datasets):
  data/activations_layer32_{prism,biosbias}_gemma-3-12b-pt.parquet
  data/recon_vectors_{prism,biosbias}.parquet          (Step 3 --save-vectors)

Fallback (if recon_vectors absent, raw scores only):
  data/fidelity_scores_{prism,biosbias}.parquet
  data/pairwise_consistency_{prism,biosbias}.parquet
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import (  # noqa: E402
    activations_filename,
    fidelity_filename,
    pairwise_filename,
    recon_vectors_filename,
)

DATASETS = ["prism", "biosbias"]
DATASET_LABELS = {"prism": "PRISM", "biosbias": "Bias in Bios"}
COLORS = {"prism": "#2176ae", "biosbias": "#e05c2a"}

N_MISMATCHES_PER_RECON = 5   # mismatched fidelity samples per reconstruction
N_BETWEEN_FACTOR = 5          # between-item pairs = factor × within-item pairs
RNG_SEED = 0


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_vec_col(path: Path, col: str) -> np.ndarray:
    return np.array(pd.read_parquet(path)[col].tolist(), dtype=np.float32)


def load_activations(path: Path) -> np.ndarray:
    return _load_vec_col(path, "activation_vector")


def load_recon(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = (
        pd.read_parquet(path)
        .sort_values(["activation_idx", "sample_idx"])
        .reset_index(drop=True)
    )
    return (
        np.array(df["recon_vector"].tolist(), dtype=np.float32),
        df["activation_idx"].values,
        df["sample_idx"].values,
    )


# ── Math helpers ───────────────────────────────────────────────────────────────

def unit_norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms < 1e-12, 1.0, norms)


def center(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return x - mu[None, :]


# ── Metric computation ─────────────────────────────────────────────────────────

def _sample_mismatches(
    act_idx: np.ndarray,
    n_orig: int,
    n_per: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return shape (n_recon, n_per) mismatch indices, all != act_idx."""
    n_recon = len(act_idx)
    j = rng.integers(0, n_orig - 1, size=(n_recon, n_per))
    # shift indices that would equal act_idx upward by one
    for k in range(n_per):
        j[:, k] = np.where(j[:, k] >= act_idx, j[:, k] + 1, j[:, k])
    return j


def compute_fidelity(
    originals: np.ndarray,
    recon_raw: np.ndarray,
    act_idx: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    mu = originals.mean(axis=0)
    recon_hat = unit_norm(recon_raw)
    cen_orig_hat = unit_norm(center(originals, mu))
    cen_recon_hat = unit_norm(center(recon_hat, mu))

    # matched — one value per reconstruction sample
    raw_matched = (originals[act_idx] * recon_hat).sum(axis=-1)
    cen_matched = (cen_orig_hat[act_idx] * cen_recon_hat).sum(axis=-1)

    # mismatched — N_MISMATCHES_PER_RECON values per reconstruction sample
    mis = _sample_mismatches(act_idx, len(originals), N_MISMATCHES_PER_RECON, rng)
    raw_mis = np.stack(
        [(originals[mis[:, k]] * recon_hat).sum(axis=-1) for k in range(N_MISMATCHES_PER_RECON)],
        axis=1,
    )  # (n_recon, n_per)
    cen_mis = np.stack(
        [(cen_orig_hat[mis[:, k]] * cen_recon_hat).sum(axis=-1) for k in range(N_MISMATCHES_PER_RECON)],
        axis=1,
    )
    raw_mismatched = raw_mis.ravel()
    cen_mismatched = cen_mis.ravel()

    # gap: each matched value minus each of its N_MISMATCHES_PER_RECON mismatches
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
    n_orig, D = originals.shape[0], recon_raw.shape[1]
    mu = originals.mean(axis=0)

    recon_hat = unit_norm(recon_raw)
    cen_recon_hat = unit_norm(center(recon_hat, mu))

    # reshape to (n_orig, n_samples, D) — works because data is sorted
    rh = recon_hat.reshape(n_orig, n_samples, D)
    cr = cen_recon_hat.reshape(n_orig, n_samples, D)

    # within-item: all C(n_samples, 2) pairs per activation
    pairs = list(combinations(range(n_samples), 2))
    s1 = np.array([p[0] for p in pairs])
    s2 = np.array([p[1] for p in pairs])

    raw_within, cen_within = [], []
    for i in range(n_orig):
        raw_within.append((rh[i][s1] * rh[i][s2]).sum(axis=-1))
        cen_within.append((cr[i][s1] * cr[i][s2]).sum(axis=-1))
    raw_within = np.concatenate(raw_within)
    cen_within = np.concatenate(cen_within)

    # between-item: sample N_BETWEEN_FACTOR × as many pairs as within-item
    n_within = len(raw_within)
    n_between = n_within * N_BETWEEN_FACTOR

    ai = rng.integers(0, n_orig, size=n_between)
    aj = rng.integers(0, n_orig - 1, size=n_between)
    aj = np.where(aj >= ai, aj + 1, aj)
    si = rng.integers(0, n_samples, size=n_between)
    sj = rng.integers(0, n_samples, size=n_between)

    raw_between = (rh[ai, si] * rh[aj, sj]).sum(axis=-1)
    cen_between = (cr[ai, si] * cr[aj, sj]).sum(axis=-1)

    # gap: pair each within value with a randomly drawn between value
    bi = rng.integers(0, n_between, size=n_within)
    raw_gap = raw_within - raw_between[bi]
    cen_gap = cen_within - cen_between[bi]

    return {
        "raw_within": raw_within,
        "raw_between": raw_between,
        "raw_gap": raw_gap,
        "cen_within": cen_within,
        "cen_between": cen_between,
        "cen_gap": cen_gap,
    }


# ── Summary statistics ─────────────────────────────────────────────────────────

def describe(arr: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


METRIC_ORDER = [
    # (group, key, display label)
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


# ── Table generation ───────────────────────────────────────────────────────────

def build_summary_df(all_stats: dict) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        if ds not in all_stats:
            continue
        for grp, key, label in METRIC_ORDER:
            if grp not in all_stats[ds] or key not in all_stats[ds][grp]:
                continue
            rows.append({"dataset": ds, "metric": label, **all_stats[ds][grp][key]})
    return pd.DataFrame(rows)


def write_latex(all_stats: dict, path: Path) -> None:
    paper_rows = [
        ("fidelity",    "cen_matched",   r"Centered fidelity (matched)"),
        ("fidelity",    "cen_mismatched", r"Centered fidelity (mismatched)"),
        ("fidelity",    "cen_gap",        r"\textbf{Centered fidelity gap} $\uparrow$"),
        ("consistency", "cen_within",     r"Centered consistency (within-item)"),
        ("consistency", "cen_between",    r"Centered consistency (between-item)"),
        ("consistency", "cen_gap",        r"\textbf{Centered consistency gap} $\uparrow$"),
        ("inflation",   "mean_vec_norm",  r"$\|\mu\|$ (mean activation norm)"),
    ]

    ds_present = [ds for ds in DATASETS if ds in all_stats]
    col_header = " & ".join(DATASET_LABELS[ds] for ds in ds_present)
    n_ds = len(ds_present)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{NLA reliability results on PRISM and Bias in Bios. "
            r"Metrics are mean-centered cosine similarities (mean $\pm$ std, $n{=}400$ items). "
            r"Higher gap $\uparrow$ indicates stronger reliability.}"
        ),
        r"\label{tab:nla_reliability}",
        r"\begin{tabular}{l" + "c" * n_ds + r"}",
        r"\toprule",
        f"Metric & {col_header} \\\\",
        r"\midrule",
    ]

    prev_grp = None
    for grp, key, label in paper_rows:
        if prev_grp is not None and grp != prev_grp:
            lines.append(r"\midrule")
        prev_grp = grp
        cells = []
        for ds in ds_present:
            d = all_stats.get(ds, {}).get(grp, {}).get(key)
            cells.append(f"{d['mean']:.3f} $\\pm$ {d['std']:.3f}" if d else "---")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _violin_panel(
    ax: plt.Axes,
    arrays: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    title: str,
    hline: float | None = 0.0,
) -> None:
    parts = ax.violinplot(arrays, positions=list(range(len(arrays))), showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax.boxplot(
        arrays,
        positions=list(range(len(arrays))),
        widths=0.12,
        showfliers=False,
        patch_artist=False,
        medianprops=dict(linewidth=0),
    )
    if hline is not None:
        ax.axhline(hline, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(arrays)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)


def _subsample(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(arr) <= max_n:
        return arr
    return rng.choice(arr, size=max_n, replace=False)


# ── Figures ────────────────────────────────────────────────────────────────────

def plot_fidelity_dist(raw_data: dict, fig_dir: Path) -> None:
    ds_avail = [
        ds for ds in DATASETS
        if "cen_matched" in raw_data.get(ds, {}).get("fidelity", {})
    ]
    if not ds_avail:
        print("  skip fidelity_dist.png — no centered fidelity data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(
        axes,
        ["cen_matched", "cen_mismatched"],
        ["Centered matched fidelity", "Centered mismatched fidelity"],
    ):
        _violin_panel(
            ax,
            [raw_data[ds]["fidelity"][key] for ds in ds_avail],
            [DATASET_LABELS[ds] for ds in ds_avail],
            [COLORS[ds] for ds in ds_avail],
            ylabel="Centered cosine similarity",
            title=title,
        )
    fig.suptitle("Reconstruction Fidelity (mean-centered)", fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_dir / "fidelity_dist.png")


def plot_consistency_dist(raw_data: dict, fig_dir: Path, rng: np.random.Generator) -> None:
    ds_avail = [
        ds for ds in DATASETS
        if "cen_within" in raw_data.get(ds, {}).get("consistency", {})
    ]
    if not ds_avail:
        print("  skip consistency_dist.png — no centered consistency data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(
        axes,
        ["cen_within", "cen_between"],
        ["Centered within-item consistency", "Centered between-item baseline"],
    ):
        arrays = [
            _subsample(raw_data[ds]["consistency"][key], 50_000, rng)
            for ds in ds_avail
        ]
        _violin_panel(
            ax,
            arrays,
            [DATASET_LABELS[ds] for ds in ds_avail],
            [COLORS[ds] for ds in ds_avail],
            ylabel="Centered cosine similarity",
            title=title,
        )
    fig.suptitle("Reconstruction Consistency (mean-centered)", fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_dir / "consistency_dist.png")


def plot_raw_vs_centered(all_stats: dict, fig_dir: Path) -> None:
    ds_avail = [ds for ds in DATASETS if ds in all_stats]
    if not ds_avail:
        return

    bar_defs = [
        ("fidelity",    "raw_gap",  "Fidelity\nraw gap"),
        ("fidelity",    "cen_gap",  "Fidelity\ncentered gap"),
        ("consistency", "raw_gap",  "Consistency\nraw gap"),
        ("consistency", "cen_gap",  "Consistency\ncentered gap"),
    ]
    avail = [
        (g, k, lbl)
        for g, k, lbl in bar_defs
        if any(
            k in all_stats.get(ds, {}).get(g, {})
            for ds in ds_avail
        )
    ]
    if not avail:
        print("  skip raw_vs_centered.png — no gap data")
        return

    x = np.arange(len(avail))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for di, ds in enumerate(ds_avail):
        means = [all_stats[ds].get(g, {}).get(k, {}).get("mean", 0.0) for g, k, _ in avail]
        errs  = [all_stats[ds].get(g, {}).get(k, {}).get("std",  0.0) for g, k, _ in avail]
        offset = (di - len(ds_avail) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=errs, capsize=4,
            label=DATASET_LABELS[ds], color=COLORS[ds], alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in avail])
    ax.set_ylabel("Mean cosine gap (mean ± std)")
    ax.set_title("Raw vs Centered Cosine Gaps")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, fig_dir / "raw_vs_centered.png")


def plot_cosine_inflation(all_stats: dict, fig_dir: Path) -> None:
    ds_avail = [ds for ds in DATASETS if ds in all_stats]
    if not ds_avail:
        return

    bar_defs = [
        ("inflation",   "mean_vec_norm",  r"$\|\mu\|$"),
        ("fidelity",    "raw_matched",    "Raw matched"),
        ("fidelity",    "raw_mismatched", "Raw mismatched"),
        ("fidelity",    "cen_matched",    "Cen. matched"),
        ("fidelity",    "cen_mismatched", "Cen. mismatched"),
    ]
    avail = [
        (g, k, lbl)
        for g, k, lbl in bar_defs
        if any(k in all_stats.get(ds, {}).get(g, {}) for ds in ds_avail)
    ]
    if not avail:
        print("  skip cosine_inflation.png — insufficient data")
        return

    x = np.arange(len(avail))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for di, ds in enumerate(ds_avail):
        means = [all_stats[ds].get(g, {}).get(k, {}).get("mean", 0.0) for g, k, _ in avail]
        offset = (di - len(ds_avail) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=DATASET_LABELS[ds], color=COLORS[ds], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in avail], fontsize=9)
    ax.set_ylabel("Mean value")
    ax.set_title("Cosine Inflation: Raw vs Centered Fidelity")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    _save(fig, fig_dir / "cosine_inflation.png")


# ── Per-dataset processing ─────────────────────────────────────────────────────

def process_dataset(
    ds: str, data_dir: Path, rng: np.random.Generator
) -> tuple[dict, dict]:
    act_path   = data_dir / activations_filename(ds)
    recon_path = data_dir / recon_vectors_filename(ds)
    fid_path   = data_dir / fidelity_filename(ds)
    pw_path    = data_dir / pairwise_filename(ds)

    if not act_path.exists():
        print(f"  [{ds}] skip — {act_path.name} not found")
        return {}, {}

    print(f"  [{ds}] loading activations ...")
    originals = load_activations(act_path)
    mu = originals.mean(axis=0)
    mean_vec_norm = float(np.linalg.norm(mu))
    stats: dict = {"inflation": {"mean_vec_norm": describe(np.array([mean_vec_norm]))}}
    raw_arrays: dict = {}

    if recon_path.exists():
        print(f"  [{ds}] loading recon vectors ...")
        recon_raw, act_idx, sample_idx = load_recon(recon_path)
        n_samples = int(sample_idx.max()) + 1

        print(f"  [{ds}] fidelity  ({len(originals)} orig × {n_samples} samples) ...")
        fid = compute_fidelity(originals, recon_raw, act_idx, rng)
        stats["fidelity"] = {k: describe(v) for k, v in fid.items()}
        raw_arrays["fidelity"] = fid

        print(f"  [{ds}] consistency ...")
        con = compute_consistency(originals, recon_raw, act_idx, n_samples, rng)
        stats["consistency"] = {k: describe(v) for k, v in con.items()}
        raw_arrays["consistency"] = con

    elif fid_path.exists() and pw_path.exists():
        print(f"  [{ds}] recon_vectors missing — using pre-computed raw scores")
        fid_df = pd.read_parquet(fid_path)
        pw_df  = pd.read_parquet(pw_path)
        raw_matched = fid_df["fidelity_cos"].values
        raw_within  = pw_df["cos_sim"].values
        stats["fidelity"]     = {"raw_matched": describe(raw_matched)}
        stats["consistency"]  = {"raw_within":  describe(raw_within)}
        raw_arrays["fidelity"]    = {"raw_matched": raw_matched}
        raw_arrays["consistency"] = {"raw_within": raw_within}

    else:
        print(f"  [{ds}] skip metrics — no recon_vectors or pre-computed score files found")

    return stats, raw_arrays


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default="data",    type=Path)
    p.add_argument("--out-dir",  default="reports", type=Path)
    p.add_argument("--fig-dir",  default="figures", type=Path)
    args = p.parse_args()

    def resolve(path: Path) -> Path:
        return ROOT / path if not path.is_absolute() else path

    data_dir = resolve(args.data_dir)
    out_dir  = resolve(args.out_dir)
    fig_dir  = resolve(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    all_stats: dict = {}
    all_raw: dict   = {}

    print("=== Computing metrics ===")
    for ds in DATASETS:
        stats, raw = process_dataset(ds, data_dir, rng)
        if stats:
            all_stats[ds] = stats
            all_raw[ds]   = raw

    if not all_stats:
        print("No datasets processed — check that data/ contains the required artifacts.")
        sys.exit(1)

    print("\n=== Writing tables ===")
    summary_df = build_summary_df(all_stats)
    csv_path = out_dir / "summary_stats.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  wrote {csv_path}")
    write_latex(all_stats, out_dir / "results_table.tex")

    print("\n=== Writing figures ===")
    plot_fidelity_dist(all_raw, fig_dir)
    plot_consistency_dist(all_raw, fig_dir, rng)
    plot_raw_vs_centered(all_stats, fig_dir)
    plot_cosine_inflation(all_stats, fig_dir)

    print(f"\n=== Done ===")
    print(f"  tables  → {out_dir}/")
    print(f"  figures → {fig_dir}/")


if __name__ == "__main__":
    main()
