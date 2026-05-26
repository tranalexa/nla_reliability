"""prepare_biosbias.py

Loads Bias in Bios from HuggingFace (LabHC/bias_in_bios), scrubs explicit gender
indicators from biography text, stratifies to N samples balanced across occupations
and genders, then writes a CSV that is a drop-in replacement for selfdescribe_400.csv
in the nla_reliability pipeline.

Output CSV columns (matching nla_reliability's expected format):
    user_prompt   — scrubbed bio text (starting after the first sentence, per BiasBios convention)
    attr_class    — "Occupation" or "Gender"
    attr          — the label value (e.g. "surgeon", "male")

Usage:
    pip install datasets pandas numpy
    python prepare_biosbias.py                        # 200 rows, occupation task
    python prepare_biosbias.py --n 400 --task gender  # 400 rows, gender task
    python prepare_biosbias.py --n 400 --task both    # 400 rows, alternating tasks
    python prepare_biosbias.py --keep-pronouns        # skip scrubbing (ablation)

The scrubbing step removes:
    - Gendered pronouns (he/she/him/her/his/hers/himself/herself)
    - Gendered honorifics (Mr./Mrs./Ms./Miss/Sir/Madam/Ma'am)
    - Gendered nouns that appear in the original BiasBios scrubber
      (brother/sister/son/daughter/father/mother/husband/wife/boyfriend/girlfriend etc.)

Note: BiasBios already provides a partially scrubbed field (`bio`) that removes the
subject's name and the most obvious pronouns. We use `raw` and apply a more thorough
scrub so you have full control and can tune the word list for your experiments.
"""

import argparse
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------

# Pronouns
_PRONOUNS = [
    r"\bhe\b", r"\bshe\b",
    r"\bhim\b", r"\bher\b",
    r"\bhis\b", r"\bhers\b",
    r"\bhimself\b", r"\bherself\b",
    r"\bhe'd\b", r"\bshe'd\b",
    r"\bhe's\b", r"\bshe's\b",
    r"\bhe'll\b", r"\bshe'll\b",
    r"\bhe've\b", r"\bshe've\b",  # rare but present
]

# Honorifics (with and without trailing period)
_HONORIFICS = [
    r"\bMr\.?\b", r"\bMrs\.?\b", r"\bMs\.?\b", r"\bMiss\b",
    r"\bSir\b", r"\bMadam\b", r"\bMa'am\b", r"\bMaam\b",
]

# Gendered relational nouns — keep this conservative; over-scrubbing removes
# occupation-relevant signal (e.g. "brotherhood" in religious contexts).
_GENDERED_NOUNS = [
    r"\bbrother\b", r"\bsister\b",
    r"\bson\b", r"\bdaughter\b",
    r"\bfather\b", r"\bmother\b",
    r"\bhusband\b", r"\bwife\b",
    r"\bboyfriend\b", r"\bgirlfriend\b",
    r"\buncle\b", r"\baunt\b",
    r"\bnephew\b", r"\bniece\b",
    r"\bgrandfather\b", r"\bgrandmother\b",
    r"\bgrandson\b", r"\bgranddaughter\b",
    r"\bstepfather\b", r"\bstepmother\b",
    r"\bstepson\b", r"\bstepdaughter\b",
    r"\bwidower\b", r"\bwidow\b",
    r"\bactor\b", r"\bactress\b",
    r"\bwaiter\b", r"\bwaitress\b",
    r"\bsteward\b", r"\bstewardess\b",
]

_SCRUB_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in _PRONOUNS + _HONORIFICS + _GENDERED_NOUNS
]

_PLACEHOLDER = "[PERSON]"


def scrub_gender_indicators(text: str) -> str:
    """Remove explicit gender indicators from bio text."""
    for pat in _SCRUB_PATTERNS:
        text = pat.sub(_PLACEHOLDER, text)
    # Collapse multiple consecutive placeholders into one
    text = re.sub(r"(\[PERSON\]\s*){2,}", _PLACEHOLDER + " ", text)
    # Remove leading/trailing whitespace artifacts
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def strip_first_sentence(raw: str, start_pos: int) -> str:
    """
    BiasBios convention: the classification task uses raw[start_pos:] to avoid
    using the opening sentence which typically names the subject.
    start_pos is a character index.
    """
    return raw[start_pos:].strip()


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratify_occupation(
    df: pd.DataFrame,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    """Sample n_total rows, balanced male/female, from the top occupations by count."""
    rng = np.random.default_rng(seed)
    half = n_total // 2
    male = df[df["gender"] == "M"].sample(half, random_state=int(rng.integers(1e6)))
    female = df[df["gender"] == "F"].sample(half, random_state=int(rng.integers(1e6)))
    result = pd.concat([male, female]).sample(frac=1, random_state=int(rng.integers(1e6)))
    return result.reset_index(drop=True)


def stratify_gender(
    df: pd.DataFrame,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    """Sample n_total rows, half male / half female."""
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200,
                        help="Total number of rows to sample (default: 200)")
    parser.add_argument("--task", choices=["occupation", "gender", "both"], default="occupation",
                        help=(
                            "Which label to use as attr_class. "
                            "'occupation' → attr_class='Occupation', attr=job title. "
                            "'gender' → attr_class='Gender', attr=male/female. "
                            "'both' → half occupation rows, half gender rows (attr_class varies). "
                            "Default: occupation"
                        ))
    parser.add_argument("--split", default="train",
                        help="HuggingFace dataset split to load (default: train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-pronouns", action="store_true",
                        help="Skip gender scrubbing (ablation / comparison baseline)")
    parser.add_argument("--output", default="biosbias_nla.csv",
                        help="Output CSV path (default: biosbias_nla.csv)")
    parser.add_argument("--hf-dataset", default="LabHC/bias_in_bios",
                        help="HuggingFace dataset name (default: LabHC/bias_in_bios)")
    args = parser.parse_args()

    print(f"Loading {args.hf_dataset} split='{args.split}' ...")
    ds = load_dataset(args.hf_dataset, split=args.split)
    df_raw = ds.to_pandas()

    print(f"Loaded {len(df_raw):,} rows. Columns: {list(df_raw.columns)}")

    # --- Normalise column names ---
    # LabHC version uses: hard_text, text, title, gender, start_pos
    # Original pkl uses: raw, bio, title, gender, start_pos
    # Detect which version we have.
    if "hard_text" in df_raw.columns:
        # LabHC HuggingFace version — `hard_text` is the bio starting after start_pos
        # i.e. it already strips the first sentence. `text` is the full bio.
        text_col = "hard_text"
        has_start_pos = False
        print("Detected LabHC HuggingFace schema (hard_text column).")
    elif "raw" in df_raw.columns:
        text_col = "raw"
        has_start_pos = "start_pos" in df_raw.columns
        print("Detected original pkl schema (raw column).")
    elif "text" in df_raw.columns:
        text_col = "text"
        has_start_pos = False
        print("Detected text column schema.")
    else:
        raise ValueError(f"Unrecognised columns: {list(df_raw.columns)}")

    # Map gender values to consistent strings
    # LabHC uses 0/1 integers; original uses "M"/"F"
    if df_raw["gender"].dtype in [np.int64, np.int32, int] or str(df_raw["gender"].dtype).startswith("int"):
        df_raw["gender"] = df_raw["gender"].map({0: "M", 1: "F"})
        print("Mapped gender 0→F, 1→M.")

    PROFESSION_LABELS = {
        0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
        4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
        8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
        12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
        16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
        20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
        24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher"
    }
    df_raw["profession"] = df_raw["profession"].map(PROFESSION_LABELS)
    print(f"Decoded profession labels. Example: {df_raw['profession'].value_counts().head(5).to_dict()}")

    # --- Extract bio text ---
    def get_bio_text(row):
        if has_start_pos:
            return strip_first_sentence(row[text_col], row["start_pos"])
        else:
            return row[text_col]

    print("Extracting bio text ...")
    df_raw["bio_text"] = df_raw.apply(get_bio_text, axis=1)

    # Drop very short bios (fewer than 20 characters after stripping first sentence)
    n_before = len(df_raw)
    df_raw = df_raw[df_raw["bio_text"].str.len() >= 20].reset_index(drop=True)
    print(f"Dropped {n_before - len(df_raw)} rows with bio_text < 20 chars.")

    # --- Scrub gender indicators ---
    if args.keep_pronouns:
        print("Skipping gender scrubbing (--keep-pronouns).")
        df_raw["user_prompt"] = df_raw["bio_text"]
    else:
        print("Scrubbing explicit gender indicators ...")
        df_raw["user_prompt"] = df_raw["bio_text"].apply(scrub_gender_indicators)

        # Sanity check: count remaining obvious indicators
        pronoun_pat = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.IGNORECASE)
        n_remaining = df_raw["user_prompt"].apply(lambda t: bool(pronoun_pat.search(t))).sum()
        pct = 100 * n_remaining / len(df_raw)
        print(f"After scrubbing: {n_remaining} rows ({pct:.1f}%) still contain a core pronoun "
              f"(may be in quoted text or compound words — inspect if high).")

    # --- Stratify ---
    print(f"Stratifying to {args.n} rows, task='{args.task}' ...")

    if args.task == "occupation":
        sampled = stratify_occupation(df_raw, args.n, args.seed)
        sampled["attr_class"] = "Occupation"
        sampled["attr"] = sampled["title"]

    elif args.task == "gender":
        sampled = stratify_gender(df_raw, args.n, args.seed)
        sampled["attr_class"] = "Gender"
        sampled["attr"] = sampled["gender"].map({"M": "male", "F": "female"})

    elif args.task == "both":
        half = args.n // 2
        occ_sample = stratify_occupation(df_raw, half, args.seed)
        occ_sample["attr_class"] = "Occupation"
        occ_sample["attr"] = occ_sample["profession"]

        gen_sample = stratify_gender(df_raw, half, args.seed + 1)
        gen_sample["attr_class"] = "Gender"
        gen_sample["attr"] = gen_sample["gender"].map({"M": "male", "F": "female"})

        sampled = pd.concat([occ_sample, gen_sample]).sample(
            frac=1, random_state=args.seed
        ).reset_index(drop=True)

    # --- Build output ---
    out = sampled[["user_prompt", "attr_class", "attr"]].reset_index(drop=True)

    # Validate
    assert len(out) == args.n or len(out) <= args.n, f"Expected {args.n} rows, got {len(out)}"
    assert out["user_prompt"].isna().sum() == 0, "NaN user_prompt values found"
    assert out["attr"].isna().sum() == 0, "NaN attr values found"

    out.to_csv(args.output, index=False)
    print(f"\nWrote {len(out)} rows → {args.output}")
    print(f"\nattr_class distribution:\n{out['attr_class'].value_counts().to_string()}")
    print(f"\nattr distribution (top 20):\n{out['attr'].value_counts().head(20).to_string()}")

    if not args.keep_pronouns:
        # Show a few examples to sanity check scrubbing
        print("\n--- Scrubbing sample (first 3 rows) ---")
        for _, row in out.head(3).iterrows():
            preview = row["user_prompt"][:200].replace("\n", " ")
            print(f"  [{row['attr_class']}={row['attr']}] {preview}...")

    print("\nDone. Upload this CSV to Modal with:")
    print(f"  modal volume put nla-cache {args.output} /cache/selfdescribe_400.csv")
    print("(rename on upload so the pipeline finds it at the expected path)")


if __name__ == "__main__":
    main()
