"""preprocessing_biosbias.py

Adapted from microsoft/biosbias preprocess.py for use with LabHC/bias_in_bios
on HuggingFace. Applies the original scrubbing logic (pronouns + person names
extracted via spaCy NER) and outputs a CSV for the nla_reliability pipeline.

Key differences from the original:
  - Loads from HuggingFace instead of Common Crawl .pkl files
  - Uses spaCy NER to extract person names (original used crawled HTML metadata)
  - Dedup logic is adapted since HF version lacks path/raw fields
  - Outputs CSV with user_prompt / attr_class / attr columns

Usage:
    pip install datasets pandas numpy spacy
    python -m spacy download en_core_web_sm
    python prepare_biosbias.py --n 400 --task both --output biosbias_nla.csv
"""

import argparse
import re
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Profession label map (LabHC integer → string)
# ---------------------------------------------------------------------------

PROFESSION_LABELS = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher"
}

TITLES_TO_IGNORE = set()  # add profession strings here to exclude them

# ---------------------------------------------------------------------------
# Scrubbing — adapted directly from microsoft/biosbias preprocess.py
# ---------------------------------------------------------------------------

def extract_names(text: str, nlp) -> list[str]:
    """Use spaCy NER to extract PERSON entity tokens from text."""
    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Add each token of the name individually, as the original does
            for token in ent:
                if len(token.text) > 0:
                    names.add(token.text)
    return list(names)


def process(bio_text: str, names: list[str], replacement: str = "_") -> str:
    """
    Adapted from microsoft/biosbias preprocess.py process().
    Scrubs pronouns and person names from bio text.
    """
    # Build regex exactly as the original does
    regExp = (
        r"\b(?:[Hh]e|[Ss]he|[Hh]er|[Hh]is|[Hh]im|[Hh]ers|"
        r"[Hh]imself|[Hh]erself|[Mm][Rr]|[Mm][Rr][sS]|[Mm][Ss]|"
    )
    name_parts = [re.escape(n) for n in names if len(n) > 0]
    if name_parts:
        regExp += "|".join(name_parts)
    else:
        # No names found — close the group without name alternates
        regExp = regExp.rstrip("|")
    regExp += r")\b"

    bio_text = re.sub(regExp, replacement, bio_text)
    return bio_text

# ---------------------------------------------------------------------------
# Dedup — adapted from microsoft/biosbias preprocess.py
# ---------------------------------------------------------------------------

def group_by(l, func):
    ans = {}
    for i in l:
        k = func(i)
        if k not in ans:
            ans[k] = [i]
        else:
            ans[k].append(i)
    return ans


def dedup(records: list[dict]) -> list[dict]:
    """Deduplicate by (hard_text, profession) keeping longest text."""
    by_text_title = group_by(records, lambda r: (r["hard_text"], r["profession"]))
    return [
        sorted(rs, key=lambda r: -len(r["hard_text"]))[0]
        for rs in by_text_title.values()
    ]

# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratify_occupation(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample n_total rows, balanced male/female, across professions."""
    rng = np.random.default_rng(seed)
    half = n_total // 2
    male = df[df["gender"] == "M"].sample(half, random_state=int(rng.integers(1e6)))
    female = df[df["gender"] == "F"].sample(half, random_state=int(rng.integers(1e6)))
    result = pd.concat([male, female]).sample(frac=1, random_state=int(rng.integers(1e6)))
    return result.reset_index(drop=True)


def stratify_gender(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Sample n_total rows, 50/50 male/female."""
    rng = np.random.default_rng(seed)
    half = n_total // 2
    male = df[df["gender"] == "M"].sample(half, random_state=int(rng.integers(1e6)))
    female = df[df["gender"] == "F"].sample(half, random_state=int(rng.integers(1e6)))
    result = pd.concat([male, female]).sample(frac=1, random_state=int(rng.integers(1e6)))
    return result.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=400,
                        help="Total rows to sample (default: 400)")
    parser.add_argument("--task", choices=["occupation", "gender", "both"], default="both",
                        help="Label to use as attr_class (default: both)")
    parser.add_argument("--split", default="train",
                        help="HuggingFace dataset split (default: train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="biosbias_nla.csv",
                        help="Output CSV path (default: biosbias_nla.csv)")
    parser.add_argument("--hf-dataset", default="LabHC/bias_in_bios")
    parser.add_argument("--keep-pronouns", action="store_true",
                        help="Skip scrubbing (ablation baseline)")
    parser.add_argument("--spacy-model", default="en_core_web_sm",
                        help="spaCy model for NER (default: en_core_web_sm)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.hf_dataset} split='{args.split}' ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    df_raw = ds.to_pandas()
    print(f"Loaded {len(df_raw):,} rows.")

    # Decode profession integers
    df_raw["profession"] = df_raw["profession"].map(PROFESSION_LABELS)
    df_raw = df_raw[~df_raw["profession"].isin(TITLES_TO_IGNORE)].reset_index(drop=True)

    # Decode gender integers (HF: 0=Male, 1=Female per dataset card)
    df_raw["gender"] = df_raw["gender"].map({0: "M", 1: "F"})

    # Drop short bios
    n_before = len(df_raw)
    df_raw = df_raw[df_raw["hard_text"].str.len() >= 20].reset_index(drop=True)
    print(f"Dropped {n_before - len(df_raw)} short bios.")

    # Dedup
    records = df_raw.to_dict("records")
    records = dedup(records)
    df_raw = pd.DataFrame(records)
    print(f"After dedup: {len(df_raw):,} rows.")

    # Scrub
    if args.keep_pronouns:
        print("Skipping scrubbing (--keep-pronouns).")
        df_raw["user_prompt"] = df_raw["hard_text"]
    else:
        print(f"Loading spaCy model '{args.spacy_model}' for NER ...")
        nlp = spacy.load(args.spacy_model, disable=["parser", "tagger", "lemmatizer"])

        print("Scrubbing pronouns and names ...")
        def scrub(text):
            names = extract_names(text, nlp)
            return process(text, names)

        df_raw["user_prompt"] = df_raw["hard_text"].apply(scrub)

        # Sanity check
        pronoun_pat = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.IGNORECASE)
        n_remaining = df_raw["user_prompt"].apply(lambda t: bool(pronoun_pat.search(t))).sum()
        print(f"After scrubbing: {n_remaining} rows ({100*n_remaining/len(df_raw):.1f}%) still contain a pronoun.")

        dr_remaining = df_raw["user_prompt"].str.contains(r'Dr\.\s+[A-Z][a-z]+', regex=True).sum()
        print(f"After scrubbing: {dr_remaining} rows still contain Dr. + name.")

    # Stratify
    print(f"Stratifying to {args.n} rows, task='{args.task}' ...")
    if args.task == "occupation":
        sampled = stratify_occupation(df_raw, args.n, args.seed)
        sampled["attr_class"] = "Occupation"
        sampled["attr"] = sampled["profession"]

    elif args.task == "gender":
        sampled = stratify_gender(df_raw, args.n, args.seed)
        sampled["attr_class"] = "Gender"
        sampled["attr"] = sampled["gender"].map({"M": "male", "F": "female"})

    elif args.task == "both":
        half = args.n // 2
        occ = stratify_occupation(df_raw, half, args.seed)
        occ["attr_class"] = "Occupation"
        occ["attr"] = occ["profession"]

        gen = stratify_gender(df_raw, half, args.seed + 1)
        gen["attr_class"] = "Gender"
        gen["attr"] = gen["gender"].map({"M": "male", "F": "female"})

        sampled = pd.concat([occ, gen]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Output
    out = sampled[["user_prompt", "attr_class", "attr"]].reset_index(drop=True)
    assert out["user_prompt"].isna().sum() == 0
    assert out["attr"].isna().sum() == 0

    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows → {args.output}")
    print(f"\nattr_class distribution:\n{out['attr_class'].value_counts().to_string()}")
    print(f"\nattr distribution (top 20):\n{out['attr'].value_counts().head(20).to_string()}")

    print("\n--- Sample (first 3 rows) ---")
    for _, row in out.head(3).iterrows():
        preview = row["user_prompt"][:200].replace("\n", " ")
        print(f"  [{row['attr_class']}={row['attr']}] {preview}...")

    print("\nDone. Upload with:")
    print(f"  modal volume put --force nla-cache {args.output} selfdescribe_400.csv")


if __name__ == "__main__":
    main()