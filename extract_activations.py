"""Step 1: extract layer-32 residual activations from Gemma-3-12B on Modal.

Make sure to set up HF token and accept the Gemma license at https://huggingface.co/google/gemma-3-12b-pt

Run:
  modal run extract_activations.py
"""

import os
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- constants ---
# google/gemma-3-12b is the family name only; weights are at -pt (pretrained) or -it (instruct)
MODEL_ID = "google/gemma-3-12b-pt"
LAYER = 32
N_PROMPTS = 400
BATCH_SIZE = 4  # 12B on A100; increase if memory allows
SELF_DESCRIBE_DATASET = "Transluce/SelfDescribe-Llama-3.1-8B-Instruct"
SAMPLE_SEED = 42

CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{HF_CACHE}/hub"
CSV_PATH = f"{CACHE}/selfdescribe_400.csv"
ACTIVATIONS_PATH = f"{CACHE}/activations_layer32.npy"
ACTIVATIONS_PARQUET_PATH = f"{CACHE}/activations_layer32.parquet"

app = modal.App("nla-extract")
vol = modal.Volume.from_name("nla-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.49",
    "accelerate",
    "pandas",
    "numpy",
    "datasets",
    "huggingface_hub",
    "pyarrow",
)


def setup_hf_cache() -> None:
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing in container. Overwrite secret:\n"
            "  modal secret create --force huggingface HF_TOKEN=<token>"
        )
    os.environ["HF_TOKEN"] = token
    return token


def ensure_csv(volume: modal.Volume) -> None:
    if Path(CSV_PATH).exists():
        return
    ds = load_dataset(SELF_DESCRIBE_DATASET, split="train", token=hf_token())
    ds = ds.shuffle(seed=SAMPLE_SEED).select(range(N_PROMPTS))
    # Stratified: ds = ds.shuffle(seed=SAMPLE_SEED).to_pandas().groupby("attr_class", group_keys=False).apply(lambda g: g.sample(n=100, random_state=SAMPLE_SEED))
    df = ds.to_pandas()[["user_prompt", "attr_class", "attr"]]
    df.to_csv(CSV_PATH, index=False)
    volume.commit()
    print(f"wrote {len(df)} rows -> {CSV_PATH}")


def load_model():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        model_dir = snapshot_download(MODEL_ID, cache_dir=HF_HUB_CACHE, token=token)
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError(
                "HF 403: token cannot access gated repos. At huggingface.co/settings/tokens "
                "use a classic Read token, or enable 'Read access to public gated repos' "
                "on your fine-grained token. Also accept the license on the model page."
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

    # Gemma-3 loads as a multimodal wrapper; use the text CausalLM for forward.
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


def read_prompts(path: str) -> list[str]:
    df = pd.read_csv(path)
    assert len(df) == N_PROMPTS, f"expected {N_PROMPTS} rows, got {len(df)}"
    return df["user_prompt"].tolist()


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

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states[0] = embeddings; hidden_states[layer+1] = post-layer-{layer} residual
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
    timeout=3600,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run():
    vol.reload()
    setup_hf_cache()

    ensure_csv(vol)
    prompts = read_prompts(CSV_PATH)
    model, tokenizer = load_model()
    vol.commit()  # persist hub cache across runs

    acts = extract_activations(model, tokenizer, prompts, LAYER)
    acts = l2_normalize(acts).astype(np.float32)
    np.save(ACTIVATIONS_PATH, acts)
    table = pa.table({"activation_vector": acts.tolist()})
    pq.write_table(table, ACTIVATIONS_PARQUET_PATH)
    vol.commit()

    norms = np.linalg.norm(acts, axis=1)
    print(f"shape: {acts.shape}")
    print(f"mean norm: {norms.mean():.6f}")
    print(f"saved -> {ACTIVATIONS_PATH}")
    print(f"saved -> {ACTIVATIONS_PARQUET_PATH}")


@app.local_entrypoint()
def main():
    run.remote()
