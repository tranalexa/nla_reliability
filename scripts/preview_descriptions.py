"""Preview AV descriptions alongside the matching prompt text.

Reads ``data/runs/<run_id>/descriptions_<dataset>.parquet`` and the matching
item CSV (pull with ``scripts/pull_from_modal.py --run-id <run_id>``).

Usage:
  uv run python scripts/preview_descriptions.py --run-id prism
  uv run python scripts/preview_descriptions.py --run-id biosbias -n 10
  uv run python scripts/preview_descriptions.py --run-id mmlu_choice
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nla.paths import (  # noqa: E402
    RUN_IDS,
    local_csv_path,
    local_descriptions_path,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "description_preview.txt"
DEFAULT_N_ITEMS = 400


def load_merged(descriptions_path: Path, prompts_path: Path) -> pd.DataFrame:
    descs = pd.read_parquet(descriptions_path)
    prompts = pd.read_csv(prompts_path)

    # item_idx (new schema) or falling back to row index
    if "item_idx" in prompts.columns:
        prompts = prompts.rename(columns={"item_idx": "activation_idx"})
    else:
        prompts = prompts.reset_index().rename(columns={"index": "activation_idx"})

    # Keep prompt_text (new schema) or user_prompt (legacy)
    prompt_col = "prompt_text" if "prompt_text" in prompts.columns else "user_prompt"
    keep_cols = ["activation_idx", prompt_col] + [
        c for c in prompts.columns if c not in ("activation_idx", prompt_col)
    ]
    merged = descs.merge(
        prompts[keep_cols],
        on="activation_idx",
        how="left",
        validate="many_to_one",
    )
    missing = merged[prompt_col].isna().sum()
    if missing:
        raise ValueError(f"{missing} descriptions have no matching prompt row")
    merged["_prompt_text"] = merged[prompt_col]
    return merged


def format_block(row: pd.Series) -> str:
    meta_cols = [c for c in row.index if c not in
                 ("activation_idx", "sample_idx", "description", "_prompt_text")]
    meta_parts = "  |  ".join(f"{c}: {row[c]}" for c in meta_cols if pd.notna(row.get(c)))
    lines = [
        "=" * 72,
        f"activation_idx: {row.activation_idx}  |  sample_idx: {row.sample_idx}",
    ]
    if meta_parts:
        lines.append(meta_parts)
    lines.extend([
        "-" * 72,
        "Prompt text (input to Gemma for activation extraction):",
        str(row._prompt_text),
        "-" * 72,
        "AV description (from injected activation vector only):",
        str(row.description),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-id",
        default="prism",
        choices=list(RUN_IDS),
        help="run to preview (default: prism)",
    )
    p.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="path to descriptions parquet (default: inferred from --run-id)",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="path to items CSV (default: inferred from --run-id)",
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("-n", type=int, default=10, help="description rows to sample (default: 10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS,
                   help="items count used in Step 1 (for CSV filename, default: 400)")
    args = p.parse_args()

    descriptions_path = args.descriptions or local_descriptions_path(args.run_id)
    prompts_path = args.prompts or local_csv_path(args.run_id, args.n_items)

    if not descriptions_path.exists():
        print(f"ERROR: {descriptions_path} not found. Pull it first:")
        print(f"  uv run python scripts/pull_from_modal.py --run-id {args.run_id}")
        sys.exit(1)
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found. Pull it first:")
        print(f"  uv run python scripts/pull_from_modal.py --run-id {args.run_id}")
        sys.exit(1)

    merged = load_merged(descriptions_path, prompts_path)
    n = min(args.n, len(merged))
    sample = merged.sample(n=n, random_state=args.seed).sort_values(
        ["activation_idx", "sample_idx"]
    )

    header = (
        f"NLA description preview ({n} samples, seed={args.seed})\n"
        f"run_id:       {args.run_id}\n"
        f"descriptions: {descriptions_path}\n"
        f"prompts:      {prompts_path}\n"
        f"total rows:   {len(merged)}\n"
    )
    body = "\n".join(format_block(row) for _, row in sample.iterrows())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + "\n" + body, encoding="utf-8")
    print(f"wrote {n} entries -> {args.output}")


if __name__ == "__main__":
    main()
