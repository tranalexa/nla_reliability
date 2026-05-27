#!/usr/bin/env python3
"""Generate the full figure bundle for the NLA reliability synthesis.

Reads:
  reports/synthesis_headline_metrics.csv
  reports/synthesis_text_consistency.csv
  reports/synthesis_linear_probes.csv
  reports/synthesis_g_theory_variance.csv
  reports/synthesis_g_theory_d_study.csv
  data/data/{fidelity_scores,pairwise_consistency}_{prism,biosbias,mmlu}.parquet
  data/{fidelity_scores,pairwise_consistency}_mmlu.parquet  (MMLU question-only)

Writes all PNGs into figures_bundle/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
DATA = ROOT / "data"
INNER = DATA / "data"
OUT = ROOT / "figures_bundle"

RUN_ORDER = ["prism", "biosbias", "mmlu_with_choices", "mmlu_question_only"]
RUN_LABELS = {
    "prism": "PRISM",
    "biosbias": "Bias in Bios",
    "mmlu_with_choices": "MMLU-Choice",
    "mmlu_question_only": "MMLU-NoChoice",
}
RUN_COLORS = {
    "prism": "#2176ae",
    "biosbias": "#e05c2a",
    "mmlu_with_choices": "#7a5195",
    "mmlu_question_only": "#bc5090",
}

DATA_DIR_BY_RUN = {
    "prism": INNER,
    "biosbias": INNER,
    "mmlu_with_choices": INNER,
    "mmlu_question_only": DATA,
}
DS_BY_RUN = {
    "prism": "prism",
    "biosbias": "biosbias",
    "mmlu_with_choices": "mmlu",
    "mmlu_question_only": "mmlu",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _legend_outside(ax: plt.Axes, *, ncol: int = 1) -> None:
    """Keep legends from covering bars/lines in saved figures."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=ncol)


def _x(runs: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    pos = np.arange(len(runs))
    labels = [RUN_LABELS[r] for r in runs]
    colors = [RUN_COLORS[r] for r in runs]
    return pos, labels, colors


def fig_centered_fidelity(headline: pd.DataFrame) -> None:
    runs = [r for r in RUN_ORDER if r in headline["run_id"].unique()]
    matched = []
    mismatched = []
    gap = []
    for r in runs:
        sub = headline[headline["run_id"] == r].set_index("metric")
        matched.append(sub.loc["Fidelity centered (matched)", "mean"])
        mismatched.append(sub.loc["Fidelity centered (mismatched)", "mean"])
        gap.append(sub.loc["Fidelity centered gap", "mean"])
    pos, labels, colors = _x(runs)
    w = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(pos - w, matched, width=w, label="Matched", color="#2a9d8f")
    ax.bar(pos, mismatched, width=w, label="Mismatched", color="#d4a373")
    ax.bar(pos + w, gap, width=w, label="Gap (matched − mismatched)", color="#1d3557")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Centered cosine")
    ax.set_title("Centered fidelity: matched vs mismatched vs gap")
    _legend_outside(ax)
    _save(fig, "01_centered_fidelity.png")


def fig_centered_consistency(headline: pd.DataFrame) -> None:
    runs = [r for r in RUN_ORDER if r in headline["run_id"].unique()]
    within = []
    between = []
    gap = []
    for r in runs:
        sub = headline[headline["run_id"] == r].set_index("metric")
        within.append(sub.loc["Consistency centered (within-item)", "mean"])
        between.append(sub.loc["Consistency centered (between-item)", "mean"])
        gap.append(sub.loc["Consistency centered gap", "mean"])
    pos, labels, _ = _x(runs)
    w = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(pos - w, within, width=w, label="Within-item", color="#2a9d8f")
    ax.bar(pos, between, width=w, label="Between-item", color="#d4a373")
    ax.bar(pos + w, gap, width=w, label="Gap (within − between)", color="#1d3557")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Centered cosine")
    ax.set_title("Centered consistency: within vs between vs gap")
    _legend_outside(ax)
    _save(fig, "02_centered_consistency.png")


def fig_gap_comparison(headline: pd.DataFrame) -> None:
    runs = [r for r in RUN_ORDER if r in headline["run_id"].unique()]
    fid_gap, fid_p5, fid_p95 = [], [], []
    con_gap, con_p5, con_p95 = [], [], []
    for r in runs:
        sub = headline[headline["run_id"] == r].set_index("metric")
        fid_gap.append(sub.loc["Fidelity centered gap", "mean"])
        fid_p5.append(sub.loc["Fidelity centered gap", "p5"])
        fid_p95.append(sub.loc["Fidelity centered gap", "p95"])
        con_gap.append(sub.loc["Consistency centered gap", "mean"])
        con_p5.append(sub.loc["Consistency centered gap", "p5"])
        con_p95.append(sub.loc["Consistency centered gap", "p95"])
    pos, labels, _ = _x(runs)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(pos - w / 2, fid_gap, width=w,
           yerr=[np.array(fid_gap) - np.array(fid_p5), np.array(fid_p95) - np.array(fid_gap)],
           label="Centered fidelity gap", color="#264653", capsize=4)
    ax.bar(pos + w / 2, con_gap, width=w,
           yerr=[np.array(con_gap) - np.array(con_p5), np.array(con_p95) - np.array(con_gap)],
           label="Centered consistency gap", color="#e76f51", capsize=4)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap (centered cosine)")
    ax.set_title("Reliability headline: centered gaps by run\n(error bars = p5–p95 across pairs)")
    _legend_outside(ax)
    _save(fig, "03_centered_gaps.png")


def fig_text_consistency(text_df: pd.DataFrame) -> None:
    runs = [r for r in RUN_ORDER if r in text_df["run_id"].unique()]
    means = []
    p5 = []
    p95 = []
    for r in runs:
        sub = text_df[text_df["run_id"] == r].iloc[0]
        means.append(sub["mean"])
        p5.append(sub["p5"])
        p95.append(sub["p95"])
    pos, labels, colors = _x(runs)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(pos, means, color=colors,
           yerr=[np.array(means) - np.array(p5), np.array(p95) - np.array(means)],
           capsize=5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean pairwise MPNet cosine")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Text-space consistency (MPNet) across 12 AV descriptions\n(error bars = p5–p95 across activations)")
    _save(fig, "04_text_consistency_mpnet.png")


def fig_text_within_between(text_between: pd.DataFrame) -> None:
    runs = [r for r in RUN_ORDER if r in text_between["run_id"].unique()]
    within = []
    between = []
    gap = []
    for r in runs:
        sub = text_between[text_between["run_id"] == r].iloc[0]
        within.append(sub["within_mean"])
        between.append(sub["between_mean"])
        gap.append(sub["gap_within_minus_between"])
    pos, labels, _ = _x(runs)
    w = 0.28
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(pos - w, within, width=w, label="Within-item", color="#2a9d8f")
    ax.bar(pos, between, width=w, label="Between-item", color="#d4a373")
    ax.bar(pos + w, gap, width=w, label="Gap (within − between)", color="#1d3557")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean pairwise MPNet cosine")
    ax.set_ylim(0, 1.0)
    ax.set_title("Text-space specificity: within-item vs between-item MPNet cosine")
    _legend_outside(ax)
    _save(fig, "04b_text_within_vs_between_mpnet.png")


def _load_text_within_between_per_activation(run_id: str) -> pd.DataFrame | None:
    ds = DS_BY_RUN[run_id]
    if run_id == "mmlu_with_choices":
        within_path = DATA / "text_consistency_mpnet_mmlu_choices.parquet"
        between_path = DATA / "text_between_item_mpnet_mmlu_choices.parquet"
    else:
        within_path = DATA / f"text_consistency_mpnet_{ds}.parquet"
        between_path = DATA / f"text_between_item_mpnet_{ds}.parquet"
    if not within_path.exists() or not between_path.exists():
        return None
    within = pd.read_parquet(within_path)[["activation_idx", "mean_pairwise_text_cosine"]]
    between = pd.read_parquet(between_path)[
        ["activation_idx", "mean_pairwise_text_cosine_between"]
    ]
    out = within.merge(between, on="activation_idx", how="inner")
    out["gap_within_minus_between"] = (
        out["mean_pairwise_text_cosine"] - out["mean_pairwise_text_cosine_between"]
    )
    out["run_id"] = run_id
    return out


def fig_text_specificity_distributions() -> None:
    frames = []
    for run_id in RUN_ORDER:
        df = _load_text_within_between_per_activation(run_id)
        if df is not None:
            frames.append(df)
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    metrics = [
        ("mean_pairwise_text_cosine", "Within-item"),
        ("mean_pairwise_text_cosine_between", "Between-item"),
        ("gap_within_minus_between", "Gap"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    for ax, (col, label) in zip(axes, metrics):
        data = []
        colors = []
        xticklabels = []
        for run_id in RUN_ORDER:
            sub = df[df["run_id"] == run_id]
            if sub.empty:
                continue
            data.append(sub[col].to_numpy())
            colors.append(RUN_COLORS[run_id])
            xticklabels.append(RUN_LABELS[run_id])

        parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.85)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor("black")
        for key in ("cmins", "cmaxes", "cbars", "cmeans"):
            if key in parts:
                parts[key].set_color("black")
        ax.set_xticks(np.arange(1, len(xticklabels) + 1))
        ax.set_xticklabels(xticklabels, rotation=18, ha="right")
        ax.set_title(label)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("MPNet cosine per activation")
    fig.suptitle(
        "Text-space similarity distributions across activations\n"
        "within-item, between-item baseline, and within−between gap",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "04c_text_similarity_distributions.png")


def fig_linear_probes(probes: pd.DataFrame) -> None:
    sub = probes.copy()
    sub["key"] = sub["run_id"] + " · " + sub["target"]
    pivot_acc = sub.pivot(index="key", columns="vector_source", values="probe_acc")
    pivot_acc = pivot_acc.loc[[
        f"{r} · {t}"
        for r, t in zip(sub["run_id"], sub["target"])
    ][::1]]
    pivot_acc = pivot_acc[~pivot_acc.index.duplicated()]
    pivot_acc = pivot_acc.reindex([
        "prism · gender",
        "biosbias · profession",
        "biosbias · gender",
        "mmlu_with_choices · subject",
        "mmlu_question_only · subject",
    ])
    pivot_acc = pivot_acc.dropna(how="all")

    pretty = {
        "prism · gender": "PRISM · gender",
        "biosbias · profession": "Bias in Bios · profession",
        "biosbias · gender": "Bias in Bios · gender",
        "mmlu_with_choices · subject": "MMLU-Choice · subject",
        "mmlu_question_only · subject": "MMLU-NoChoice · subject",
    }
    labels = [pretty[i] for i in pivot_acc.index]
    pos = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(pos - w / 2, pivot_acc["original"], width=w, label="Original activation", color="#264653")
    ax.bar(pos + w / 2, pivot_acc["recon_mean"], width=w, label="Mean of 12 recons", color="#e76f51")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Linear probe accuracy: original vs reconstructed activations")
    _legend_outside(ax)
    _save(fig, "05_linear_probes.png")


def fig_g_variance_components(gvar: pd.DataFrame, metric: str, title: str, fname: str) -> None:
    sub = gvar[gvar["metric"] == metric].set_index("run_id").reindex(RUN_ORDER).dropna(how="all")
    runs = sub.index.tolist()
    pos, labels, _ = _x(runs)
    p = sub["var_pct_p"].to_numpy()
    pi = sub["var_pct_pi"].to_numpy()
    i = sub["var_pct_i"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(pos, p, label=r"$\sigma^2_p$ · between activations", color="#264653")
    ax.bar(pos, pi, bottom=p, label=r"$\sigma^2_{pi}$ · activation × sample", color="#e76f51")
    ax.bar(pos, i, bottom=p + pi, label=r"$\sigma^2_i$ · global sample", color="#f4a261")
    for x, total_p, total_pi in zip(pos, p, pi):
        ax.text(x, total_p / 2, f"{total_p:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if total_pi > 4:
            ax.text(x, total_p + total_pi / 2, f"{total_pi:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of total variance")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, fname)


def fig_d_study(gdstudy: pd.DataFrame, metric: str, title: str, fname: str) -> None:
    sub = gdstudy[gdstudy["metric"] == metric]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for r in RUN_ORDER:
        s = sub[sub["run_id"] == r]
        if s.empty:
            continue
        ax.plot(s["n_samples"], s["G_rel"], marker="o", lw=2,
                color=RUN_COLORS[r], label=RUN_LABELS[r])
    ax.axhline(0.90, color="grey", lw=0.8, ls="--", label=r"$G = 0.90$")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":", label=r"$G = 0.95$")
    ax.set_xticks([1, 2, 3, 4, 6, 12])
    ax.set_xlabel("n′ (averaged AV samples per activation)")
    ax.set_ylabel(r"Relative generalizability, $G(n')$")
    ax.set_ylim(0.5, 1.01)
    ax.set_title(title)
    _legend_outside(ax)
    _save(fig, fname)


def fig_d_study_compare(gdstudy: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    for ax, metric, title in [
        (axes[0], "fidelity_cos", r"Fidelity $G(n')$"),
        (axes[1], "consistency_cos", r"Consistency $G(n')$"),
    ]:
        sub = gdstudy[gdstudy["metric"] == metric]
        for r in RUN_ORDER:
            s = sub[sub["run_id"] == r]
            if s.empty:
                continue
            ax.plot(s["n_samples"], s["G_rel"], marker="o", lw=2,
                    color=RUN_COLORS[r], label=RUN_LABELS[r])
        ax.axhline(0.90, color="grey", lw=0.8, ls="--")
        ax.axhline(0.95, color="grey", lw=0.8, ls=":")
        ax.set_xticks([1, 2, 3, 4, 6, 12])
        ax.set_xlabel("n′ averaged samples")
        ax.set_title(title)
    axes[0].set_ylabel(r"Relative generalizability, $G(n')$")
    axes[0].set_ylim(0.5, 1.01)
    _legend_outside(axes[1])
    fig.suptitle("D-study side-by-side: fidelity vs consistency", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_dstudy_side_by_side.png")


def fig_g_rel_n1_n12_bars(gvar: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    for ax, metric, title in [
        (axes[0], "fidelity_cos", "Fidelity"),
        (axes[1], "consistency_cos", "Consistency"),
    ]:
        sub = gvar[gvar["metric"] == metric].set_index("run_id").reindex(RUN_ORDER).dropna(how="all")
        runs = sub.index.tolist()
        pos, labels, colors = _x(runs)
        w = 0.35
        ax.bar(pos - w / 2, sub["G_rel_n1"], width=w, color="#e76f51", label="n′=1")
        ax.bar(pos + w / 2, sub["G_rel_n12"], width=w, color="#264653", label="n′=12")
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(fr"{title}: $G$ at n′=1 vs n′=12")
        ax.axhline(0.90, color="grey", lw=0.7, ls="--")
        for x, v1, v12 in zip(pos, sub["G_rel_n1"], sub["G_rel_n12"]):
            ax.text(x - w / 2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=9)
            ax.text(x + w / 2, v12 + 0.02, f"{v12:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel(r"Relative generalizability, $G$")
    _legend_outside(axes[1])
    fig.tight_layout()
    _save(fig, "09_grel_n1_vs_n12.png")


def _load_fidelity(run_id: str) -> np.ndarray:
    ds = DS_BY_RUN[run_id]
    p = DATA_DIR_BY_RUN[run_id] / f"fidelity_scores_{ds}.parquet"
    if not p.exists():
        return np.array([])
    return pd.read_parquet(p)["fidelity_cos"].to_numpy(dtype=np.float32)


def _load_pairwise_within(run_id: str) -> np.ndarray:
    ds = DS_BY_RUN[run_id]
    p = DATA_DIR_BY_RUN[run_id] / f"pairwise_consistency_{ds}.parquet"
    if not p.exists():
        return np.array([])
    return pd.read_parquet(p)["cos_sim"].to_numpy(dtype=np.float32)


def fig_raw_score_violins() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, loader, title, ylim in [
        (axes[0], _load_fidelity, "Raw fidelity_cos (recon vs original)", (0.94, 1.001)),
        (axes[1], _load_pairwise_within, "Raw pairwise within-item cos_sim", (0.94, 1.001)),
    ]:
        data = []
        labels = []
        colors = []
        for r in RUN_ORDER:
            arr = loader(r)
            if arr.size == 0:
                continue
            data.append(arr)
            labels.append(RUN_LABELS[r])
            colors.append(RUN_COLORS[r])
        parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.85)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.55)
            body.set_edgecolor("black")
        for key in ("cmins", "cmaxes", "cbars", "cmeans"):
            if key in parts:
                parts[key].set_color("black")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_ylabel("cosine")
    fig.suptitle("Raw per-sample distributions feeding G-theory",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_raw_score_distributions.png")


def fig_per_item_mean_scatter() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=False, sharey=False)
    for ax, run_id in zip(axes.flat, RUN_ORDER):
        fid_path = DATA_DIR_BY_RUN[run_id] / f"fidelity_scores_{DS_BY_RUN[run_id]}.parquet"
        pw_path = DATA_DIR_BY_RUN[run_id] / f"pairwise_consistency_{DS_BY_RUN[run_id]}.parquet"
        if not fid_path.exists() or not pw_path.exists():
            ax.set_visible(False)
            continue
        fid = pd.read_parquet(fid_path)
        pw = pd.read_parquet(pw_path)
        fid_mean = fid.groupby("activation_idx")["fidelity_cos"].mean()
        pw_mean = pw.groupby("activation_idx")["cos_sim"].mean()
        df = pd.concat([fid_mean.rename("fid"), pw_mean.rename("con")], axis=1).dropna()
        ax.scatter(df["fid"], df["con"], s=12, alpha=0.55, color=RUN_COLORS[run_id])
        ax.set_xlabel("Per-item mean fidelity_cos (12 samples)")
        ax.set_ylabel("Per-item mean within-item cos_sim")
        r = df["fid"].corr(df["con"])
        ax.set_title(f"{RUN_LABELS[run_id]}   (n={len(df)}, r={r:.2f})")
    fig.suptitle("Per-activation: matched fidelity vs within-item consistency",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_per_item_fid_vs_consistency.png")


def fig_gtheory_overview(gvar: pd.DataFrame) -> None:
    """One figure summarizing both G-studies + key D-study numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, metric, title in [
        (axes[0], "fidelity_cos", "Fidelity variance split"),
        (axes[1], "consistency_cos", "Consistency variance split"),
    ]:
        sub = gvar[gvar["metric"] == metric].set_index("run_id").reindex(RUN_ORDER).dropna(how="all")
        runs = sub.index.tolist()
        pos, labels, _ = _x(runs)
        p = sub["var_pct_p"].to_numpy()
        pi = sub["var_pct_pi"].to_numpy()
        i = sub["var_pct_i"].to_numpy()
        ax.bar(pos, p, color="#264653", label=r"$\sigma^2_p$")
        ax.bar(pos, pi, bottom=p, color="#e76f51", label=r"$\sigma^2_{pi}$")
        ax.bar(pos, i, bottom=p + pi, color="#f4a261", label=r"$\sigma^2_i$")
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("% variance")
        ax.set_title(title)
        for x, vp, vpi, g1 in zip(pos, p, pi, sub["G_rel_n1"].to_numpy()):
            ax.text(x, 102, rf"$G(1)$={g1:.2f}", ha="center", fontsize=9)
    _legend_outside(axes[0])
    fig.suptitle("G-theory variance components (with relative G at n′=1)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_gtheory_overview.png")


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    headline = pd.read_csv(REPORTS / "synthesis_headline_metrics.csv")
    text_df = pd.read_csv(REPORTS / "synthesis_text_consistency.csv")
    text_between_path = REPORTS / "synthesis_text_between_item.csv"
    text_between = pd.read_csv(text_between_path) if text_between_path.exists() else None
    probes = pd.read_csv(REPORTS / "synthesis_linear_probes.csv")
    gvar = pd.read_csv(REPORTS / "synthesis_g_theory_variance.csv")
    gdstudy = pd.read_csv(REPORTS / "synthesis_g_theory_d_study.csv")
    if "metric" not in gvar.columns:
        gvar["metric"] = "fidelity_cos"
        gdstudy["metric"] = "fidelity_cos"

    print("Generating figure bundle ...")
    fig_centered_fidelity(headline)
    fig_centered_consistency(headline)
    fig_gap_comparison(headline)
    fig_text_consistency(text_df)
    if text_between is not None:
        fig_text_within_between(text_between)
        fig_text_specificity_distributions()
    fig_linear_probes(probes)
    fig_gtheory_overview(gvar)
    fig_g_variance_components(
        gvar, "fidelity_cos",
        "Fidelity G-study: variance components (% of total)",
        "07a_fidelity_variance_components.png",
    )
    fig_g_variance_components(
        gvar, "consistency_cos",
        "Consistency G-study: variance components (% of total)",
        "07b_consistency_variance_components.png",
    )
    fig_d_study(gdstudy, "fidelity_cos",
                "Fidelity D-study: relative G vs n′ (averaged AV samples)",
                "07c_fidelity_dstudy.png")
    fig_d_study(gdstudy, "consistency_cos",
                "Consistency D-study: relative G vs n′ (averaged AV samples)",
                "07d_consistency_dstudy.png")
    fig_d_study_compare(gdstudy)
    fig_g_rel_n1_n12_bars(gvar)
    fig_raw_score_violins()
    fig_per_item_mean_scatter()
    print(f"done -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
