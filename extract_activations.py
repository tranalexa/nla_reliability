"""Step 1: extract layer-32 residual activations from Gemma-3-12B on Modal.

Supported datasets (--dataset):
  prism     — Transluce/PRISM-gender-Llama-3.1-8B-Instruct
              Samples n_items rows (default 400, seed 42).
              CSV: /cache/prism_400.csv
              Activations: /cache/activations_layer32_prism_gemma-3-12b-pt.parquet

  biosbias  — LabHC/bias_in_bios
              Samples n_items rows (default 400, seed 42), stratified 50/50 by gender.
              Gender indicators are scrubbed from bio text before extraction.
              CSV: /cache/biosbias_400.csv
              Activations: /cache/activations_layer32_biosbias_gemma-3-12b-pt.parquet

Run:
  uv run modal run extract_activations.py --dataset prism
  uv run modal run extract_activations.py --dataset biosbias
  uv run modal run extract_activations.py --all-datasets
  uv run modal run extract_activations.py --all-datasets --n-items 400 --seed 42
"""

from pathlib import Path

import modal

NLA_PKG = Path(__file__).resolve().parent / "nla"
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"
LAYER = 32
BATCH_SIZE = 4  # conservative for long PRISM sequences; safe for BiasBios too

CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{HF_CACHE}/hub"

app = modal.App("nla-extract")
vol = modal.Volume.from_name("nla-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.49",
        "accelerate",
        "pandas",
        "numpy",
        "datasets",
        "huggingface_hub",
        "pyarrow",
    )
    .add_local_dir(NLA_PKG, "/root/nla")
)


def setup_hf_cache() -> None:
    import os

    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"


def hf_token() -> str:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing in container. Set it with:\n"
            "  modal secret create --force huggingface HF_TOKEN=<token>"
        )
    os.environ["HF_TOKEN"] = token
    return token


def load_model(model_id: str):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        model_dir = snapshot_download(model_id, cache_dir=HF_HUB_CACHE, token=token)
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError(
                "HF 403: token cannot access gated repos. Accept the license on the model "
                "page and use a classic Read token or enable gated-repo access."
            ) from e
        raise
    print(f"model cache: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        token=token,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    if hasattr(model, "language_model"):
        inner = model.language_model
        if hasattr(inner, "lm_head"):
            model = inner
        else:
            wrapper = AutoModelForCausalLM.from_config(inner.config)
            wrapper.model = inner
            if getattr(inner.config, "tie_word_embeddings", False):
                wrapper.tie_weights()
            model = wrapper.eval()

    return model, tokenizer


def extract_activations(model, tokenizer, prompts: list[str], layer: int) -> np.ndarray:
    device = model.get_input_embeddings().weight.device
    all_vecs = []

    with torch.no_grad():
        for start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Call model.model (backbone) directly to skip the lm_head projection,
            # which would allocate ~3.5 GiB and OOM on long PRISM sequences.
            out = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = out.hidden_states[layer + 1]

            lengths = attention_mask.sum(dim=1).cpu()
            for i, seq_len in enumerate(lengths.tolist()):
                vec = hidden[i, seq_len - 1].float().cpu().numpy()
                all_vecs.append(vec)

    return np.stack(all_vecs, axis=0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(
    dataset: str,
    n_items: int,
    seed: int,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = LAYER,
):
    import sys

    sys.path.insert(0, "/root")
    from nla.datasets import load_dataset_items
    from nla.paths import (
        activations_filename,
        csv_filename,
        volume_activations_path,
        volume_csv_path,
    )

    vol.reload()
    setup_hf_cache()

    token = hf_token()
    print(f"dataset={dataset}  n_items={n_items}  seed={seed}  layer={layer}")

    # ── Load / build dataset CSV ───────────────────────────────────────────────
    csv_path = volume_csv_path(dataset, n_items)
    if Path(csv_path).exists():
        existing = pd.read_csv(csv_path)
        if len(existing) == n_items:
            print(f"using existing CSV ({n_items} rows) -> {csv_path}")
            df = existing
        else:
            print(f"CSV row count mismatch ({len(existing)} != {n_items}); regenerating")
            df = None
    else:
        df = None

    if df is None:
        df = load_dataset_items(dataset, n_items=n_items, seed=seed, hf_token=token)
        assert len(df) == n_items
        df.to_csv(csv_path, index=False)
        vol.commit()
        print(f"wrote {n_items} rows -> {csv_path}")

    # ── Extract activations ────────────────────────────────────────────────────
    prompts: list[str] = df["prompt_text"].tolist()
    assert len(prompts) == n_items

    parquet_path = volume_activations_path(dataset, layer, model_id)
    print(f"output -> {parquet_path}")

    model, tokenizer = load_model(model_id)
    vol.commit()

    acts = extract_activations(model, tokenizer, prompts, layer)
    acts = l2_normalize(acts).astype(np.float32)

    assert acts.shape[0] == n_items, f"Expected {n_items} activations, got {acts.shape[0]}"

    table = pa.table({"activation_vector": acts.tolist()})
    pq.write_table(table, parquet_path)
    vol.commit()

    norms = np.linalg.norm(acts, axis=1)
    print(f"shape:      {acts.shape}")
    print(f"mean norm:  {norms.mean():.6f}")
    print(f"saved ->    {parquet_path}")


@app.local_entrypoint()
def main(
    dataset: str = "prism",
    all_datasets: bool = False,
    n_items: int = 400,
    seed: int = 42,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = LAYER,
):
    """
    --dataset       prism | biosbias  (default: prism)
    --all-datasets  run both prism and biosbias in sequence
    --n-items       items to sample per dataset (default: 400)
    --seed          shuffle seed (default: 42)
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.datasets import SUPPORTED_DATASETS

    datasets = SUPPORTED_DATASETS if all_datasets else [dataset]
    for ds in datasets:
        print(f"\n=== Extracting activations: {ds} (n_items={n_items}, seed={seed}) ===")
        run.remote(
            dataset=ds,
            n_items=n_items,
            seed=seed,
            model_id=model_id,
            layer=layer,
        )
