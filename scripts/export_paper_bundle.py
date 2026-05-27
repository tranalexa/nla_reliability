#!/usr/bin/env python3
"""Export a single folder for paper writing (Claude, Overleaf, etc.).

Copies figures, CSV/LaTeX tables, and writes docs/PAPER_INPUT.md with embedded
numbers. Re-run after regenerating reports:

  uv run python scripts/make_report_tables.py --data-dir data/data
  uv run python scripts/g_theory_study.py --dataset all --data-dir data/data
  uv run python scripts/train_linear_probes_multi_dataset.py --vector-source compare --data-dir data/data
  uv run python scripts/compute_text_consistency.py --dataset all --data-dir data/data
  uv run python scripts/export_paper_bundle.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "paper_bundle"
FIG_SRC = ROOT / "figures"
REP_SRC = ROOT / "reports"
FINDINGS = ROOT / "docs" / "ANALYSIS_FINDINGS.md"


def _df_to_md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, float_fmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read_csv(name: str) -> pd.DataFrame | None:
    p = REP_SRC / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def _text_consistency_summary() -> pd.DataFrame:
    rows = []
    for ds in ["prism", "biosbias", "mmlu"]:
        p = REP_SRC / f"text_consistency_mpnet_{ds}.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            df = pd.read_csv(p)
        col = "mean_pairwise_text_cosine"
        s = df[col].astype(float)
        rows.append(
            {
                "dataset": ds,
                "n": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "median": s.median(),
                "p5": s.quantile(0.05),
                "p95": s.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def _pivot_summary_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    want = [
        "Fidelity centered gap",
        "Consistency centered gap",
        "Fidelity centered (matched)",
        "Consistency centered (within-item)",
        "Consistency centered (between-item)",
        "||mean activation||",
    ]
    sub = summary[summary["metric"].isin(want)].copy()
    if sub.empty:
        return sub
    out = sub.pivot_table(index="metric", columns="dataset", values="mean", aggfunc="first")
    out = out.reindex(want)
    return out.reset_index()


def build_paper_input() -> str:
    parts: list[str] = [
        "# NLA Reliability — Paper Input Bundle",
        "",
        "Use this file with Claude (or similar) to draft the paper. Attach the PNGs in "
        "`paper_bundle/figures/` for figure captions. Full narrative and interpretation: "
        "`docs/ANALYSIS_FINDINGS.md` (also copied here as `ANALYSIS_FINDINGS.md`).",
        "",
        "---",
        "",
        "## Methods (one paragraph)",
        "",
        "We evaluate Natural Language Autoencoders (NLA) on Gemma-3-12B layer-32 "
        "activations (L2-normalized) for 400 items × 12 stochastic activation-verbalizer "
        "(AV) samples (temperature 1.0) per dataset (PRISM, Bias in Bios, MMLU), using "
        "public checkpoints `kitft/nla-gemma3-12b-L32-{av,ar}`. Reliability is assessed "
        "via reconstruction cosine (raw and mean-centered), pairwise recon consistency, "
        "G-theory on `fidelity_cos`, and MPNet (`all-mpnet-base-v2`) mean pairwise text "
        "cosine across verbalizations. Validity is assessed via logistic-regression linear "
        "probes on original vs mean-reconstructed vectors (80/20 split, seed 42).",
        "",
        "---",
        "",
        "## Headline results (centered gaps)",
        "",
    ]

    summary = _read_csv("summary_stats.csv")
    if summary is not None:
        gaps = _pivot_summary_gaps(summary)
        if not gaps.empty:
            parts.append(_df_to_md_table(gaps))
        parts.append("")
        parts.append("### Full summary statistics (`tables/summary_stats.csv`)")
        parts.append("")
        parts.append(_df_to_md_table(summary[["dataset", "metric", "mean", "std", "n"]], ".6f"))
    else:
        parts.append("_Run `make_report_tables.py --data-dir data/data` to generate summary_stats.csv._")

    parts.extend(["", "---", "", "## Text consistency (MPNet)", ""])
    tc = _text_consistency_summary()
    if not tc.empty:
        parts.append(_df_to_md_table(tc))
    else:
        parts.append("_Run `compute_text_consistency.py` for each dataset._")

    parts.extend(["", "---", "", "## Linear probes (validity)", ""])
    probes = _read_csv("linear_probe_compare.csv")
    if probes is not None:
        parts.append(_df_to_md_table(probes, ".4f"))
        parts.append("")
        parts.append(
            "**Key finding:** MMLU subject probe drops 0.96 → 0.76 after mean recon; "
            "PRISM gender unchanged; BiasBios mixed (profession −0.04, gender +0.04)."
        )

    parts.extend(["", "---", "", "## G-theory variance components", ""])
    gvar = _read_csv("g_theory_variance_components.csv")
    if gvar is not None:
        show = gvar[
            [
                "dataset",
                "sigma2_p",
                "sigma2_pi",
                "var_pct_p",
                "var_pct_pi",
                "cronbach_alpha",
            ]
        ].copy()
        parts.append(_df_to_md_table(show, ".6f"))

    parts.extend(["", "---", "", "## G-theory D-study (G_rel by n′ samples)", ""])
    dstudy = _read_csv("g_theory_d_study.csv")
    if dstudy is not None:
        pivot = dstudy.pivot_table(
            index="dataset", columns="n_samples", values="G_rel", aggfunc="first"
        )
        parts.append(_df_to_md_table(pivot.reset_index(), ".4f"))

    parts.extend(
        [
            "",
            "---",
            "",
            "## Figures (attach these images)",
            "",
            "| File | Suggested caption |",
            "|------|-------------------|",
            "| `figures/fidelity_dist.png` | Distribution of mean-centered matched fidelity cosines by dataset |",
            "| `figures/consistency_dist.png` | Within-item vs between-item centered recon consistency |",
            "| `figures/raw_vs_centered.png` | Raw vs centered metric gaps (inflation diagnostic) |",
            "| `figures/cosine_inflation.png` | Shared mean direction: raw cosines near 1.0 |",
            "",
            "---",
            "",
            "## LaTeX table",
            "",
            "See `tables/results_table.tex` for a publication-ready two/three-column table.",
            "",
            "---",
            "",
            "## Limitations (for Discussion)",
            "",
            "- Cosine similarity on L2-normalized activations is inflated by a shared mean direction (‖μ‖ ≈ 0.99); centered gaps are the interpretable reliability metrics.",
            "- AV describes meta-linguistic state, not prompt content; text MPNet consistency (~0.83–0.87) is much lower than recon cosine (~0.999).",
            "- Linear probes test coarse linear decodability, not full semantic preservation.",
            "- PRISM gender has severe class imbalance (`non_binary` n=1).",
            "- G-theory on `fidelity_cos` does not subsume text-space or probe validity.",
            "",
            "---",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python scripts/make_report_tables.py --data-dir data/data",
            "uv run python scripts/g_theory_study.py --dataset all --data-dir data/data",
            "uv run python scripts/train_linear_probes_multi_dataset.py --vector-source compare --data-dir data/data",
            "uv run python scripts/compute_text_consistency.py --dataset all --data-dir data/data",
            "uv run python scripts/export_paper_bundle.py",
            "```",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-dir", type=Path, default=BUNDLE)
    args = p.parse_args()
    bundle = args.bundle_dir if args.bundle_dir.is_absolute() else ROOT / args.bundle_dir

    fig_dst = bundle / "figures"
    tab_dst = bundle / "tables"
    fig_dst.mkdir(parents=True, exist_ok=True)
    tab_dst.mkdir(parents=True, exist_ok=True)

    if FIG_SRC.exists():
        for png in sorted(FIG_SRC.glob("*.png")):
            shutil.copy2(png, fig_dst / png.name)

    for pattern in ("*.csv", "*.tex"):
        for f in sorted(REP_SRC.glob(pattern)):
            shutil.copy2(f, tab_dst / f.name)

    if FINDINGS.exists():
        shutil.copy2(FINDINGS, bundle / "ANALYSIS_FINDINGS.md")

    (bundle / "PAPER_INPUT.md").write_text(build_paper_input(), encoding="utf-8")

    manifest = [
        "# Paper bundle manifest",
        "",
        "Upload to Claude:",
        "",
        "1. **`PAPER_INPUT.md`** — all tables + methods + figure list",
        "2. **`ANALYSIS_FINDINGS.md`** — full interpretation (optional if PAPER_INPUT is enough)",
        "3. **`figures/*.png`** — four plots",
        "4. **`tables/*.csv`** and **`tables/results_table.tex`** — raw numbers",
        "",
        "Or zip the folder:",
        "",
        "```bash",
        "cd paper_bundle && zip -r ../nla_paper_bundle.zip .",
        "```",
        "",
        f"Generated from `{ROOT.name}/` at export time.",
        "",
    ]
    (bundle / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

    print(f"Wrote {bundle}/")
    print(f"  PAPER_INPUT.md")
    print(f"  MANIFEST.md")
    print(f"  figures/ ({len(list(fig_dst.glob('*.png')))} PNGs)")
    print(f"  tables/ ({len(list(tab_dst.iterdir()))} files)")


if __name__ == "__main__":
    main()
