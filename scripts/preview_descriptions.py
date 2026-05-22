"""Sample AV descriptions and write them beside the matching SelfDescribe prompt.

Defaults expect local copies (e.g. from modal volume get):
  descriptions.parquet
  selfdescribe_400.csv

Usage:
  uv run python scripts/preview_descriptions.py
  uv run python scripts/preview_descriptions.py -n 10 -o scripts/description_preview.txt
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTIONS = ROOT / "descriptions.parquet"
DEFAULT_PROMPTS = ROOT / "selfdescribe_400.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "description_preview.txt"


def load_merged(descriptions_path: Path, prompts_path: Path) -> pd.DataFrame:
    descs = pd.read_parquet(descriptions_path)
    prompts = pd.read_csv(prompts_path)
    prompts = prompts.reset_index().rename(columns={"index": "activation_idx"})
    merged = descs.merge(
        prompts[["activation_idx", "user_prompt", "attr_class", "attr"]],
        on="activation_idx",
        how="left",
        validate="many_to_one",
    )
    missing = merged["user_prompt"].isna().sum()
    if missing:
        raise ValueError(f"{missing} descriptions have no matching SelfDescribe row")
    return merged


def format_block(row: pd.Series) -> str:
    lines = [
        "=" * 72,
        f"activation_idx: {row.activation_idx}  |  sample_idx: {row.sample_idx}",
        f"attr_class: {row.attr_class}  |  attr: {row.attr}",
        "-" * 72,
        "SelfDescribe user_prompt:",
        str(row.user_prompt),
        "-" * 72,
        "AV description:",
        str(row.description),
        "",
    ]
    return "\n".join(lines)


def main(
    descriptions_path: Path = DEFAULT_DESCRIPTIONS,
    prompts_path: Path = DEFAULT_PROMPTS,
    output_path: Path = DEFAULT_OUTPUT,
    n: int = 10,
    seed: int = 0,
) -> None:
    merged = load_merged(descriptions_path, prompts_path)
    n = min(n, len(merged))
    sample = merged.sample(n=n, random_state=seed).sort_values(
        ["activation_idx", "sample_idx"]
    )

    header = (
        f"NLA description preview ({n} samples, seed={seed})\n"
        f"descriptions: {descriptions_path}\n"
        f"prompts: {prompts_path}\n"
        f"total descriptions in file: {len(merged)}\n"
    )

    body = "\n".join(format_block(row) for _, row in sample.iterrows())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n" + body, encoding="utf-8")
    print(f"wrote {n} entries -> {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("-n", type=int, default=10, help="number of description rows to sample")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(
        descriptions_path=args.descriptions,
        prompts_path=args.prompts,
        output_path=args.output,
        n=args.n,
        seed=args.seed,
    )
