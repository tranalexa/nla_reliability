"""Sample AV descriptions and write them beside the matching SelfDescribe prompt.

Defaults expect artifacts in data/ (pull with scripts/pull_from_modal.py):
  data/descriptions.parquet
  data/selfdescribe_400.csv

Usage:
  uv run python scripts/preview_descriptions.py
  uv run python scripts/preview_descriptions.py -n 10 -o scripts/description_preview.txt
  uv run python scripts/preview_descriptions.py --activations-source persona-only
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import DEFAULT_CSV, DEFAULT_DESCRIPTIONS  # noqa: E402
from prompt_modes import INFOBOX_SUFFIX, apply_prompt_mode  # noqa: E402

DEFAULT_PROMPTS = DEFAULT_CSV
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
    persona_texts = []
    for p in merged["user_prompt"]:
        text, _ = apply_prompt_mode(str(p), "persona-only")
        persona_texts.append(text)
    merged["persona_prompt"] = persona_texts
    return merged


def format_block(row: pd.Series, *, show_full_prompt: bool) -> str:
    lines = [
        "=" * 72,
        f"activation_idx: {row.activation_idx}  |  sample_idx: {row.sample_idx}",
        f"attr_class: {row.attr_class}  |  attr: {row.attr}",
        "-" * 72,
        "Persona text (what Step 1 persona-only mode forwards; AV never sees this):",
        str(row.persona_prompt),
    ]
    if show_full_prompt:
        lines.extend(
            [
                "-" * 72,
                "Full CSV user_prompt (includes infobox suffix):",
                str(row.user_prompt),
            ]
        )
    else:
        lines.append(
            f"(Full CSV still ends with infobox suffix; stripped for extraction only.)"
        )
    lines.extend(
        [
            "-" * 72,
            "AV description (from injected activation only):",
            str(row.description),
            "",
        ]
    )
    return "\n".join(lines)


def main(
    descriptions_path: Path = DEFAULT_DESCRIPTIONS,
    prompts_path: Path = DEFAULT_PROMPTS,
    output_path: Path = DEFAULT_OUTPUT,
    n: int = 10,
    seed: int = 0,
    activations_source: str = "persona-only",
    show_full_prompt: bool = False,
) -> None:
    merged = load_merged(descriptions_path, prompts_path)
    n = min(n, len(merged))
    sample = merged.sample(n=n, random_state=seed).sort_values(
        ["activation_idx", "sample_idx"]
    )

    source_line = (
        f"activations source (Step 2): {activations_source}\n"
        if activations_source
        else ""
    )
    header = (
        f"NLA description preview ({n} samples, seed={seed})\n"
        f"descriptions: {descriptions_path}\n"
        f"prompts: {prompts_path}\n"
        f"total descriptions in file: {len(merged)}\n"
        f"{source_line}"
        f"infobox suffix (not shown in persona text): {INFOBOX_SUFFIX!r}\n"
    )

    body = "\n".join(
        format_block(row, show_full_prompt=show_full_prompt) for _, row in sample.iterrows()
    )
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
    p.add_argument(
        "--activations-source",
        default="",
        help="label for which Step 1/2 activations produced descriptions (e.g. persona-only)",
    )
    p.add_argument(
        "--show-full-prompt",
        action="store_true",
        help="also print full CSV user_prompt including Wikipedia infobox suffix",
    )
    args = p.parse_args()
    main(
        descriptions_path=args.descriptions,
        prompts_path=args.prompts,
        output_path=args.output,
        n=args.n,
        seed=args.seed,
        activations_source=args.activations_source,
        show_full_prompt=args.show_full_prompt,
    )
