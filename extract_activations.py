"""Step 1: extract layer-32 residual activations from Gemma-3-12B on Modal.

Samples 400 rows from Transluce/SelfDescribe-Llama-3.1-8B-Instruct (shuffle seed 42).

Prompt modes (--prompt-mode; default persona-only):
  persona-only — strip INFOBOX_SUFFIX, last-token on persona text (default)
  full         — entire user_prompt; pass --full or --prompt-mode full (writes legacy parquet)

Run:
  uv run modal run extract_activations.py
  uv run modal run extract_activations.py --full
"""

import shutil
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "google/gemma-3-12b-pt"
LAYER = 32
BATCH_SIZE = 8  # 12B on A100; lower if OOM
SELF_DESCRIBE_DATASET = "Transluce/SelfDescribe-Llama-3.1-8B-Instruct"
N_PROMPTS = 400
SAMPLE_SEED = 42

CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{HF_CACHE}/hub"
CSV_PATH = f"{CACHE}/selfdescribe_400.csv"

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
    .add_local_file("prompt_modes.py", "/root/prompt_modes.py")
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
            "HF_TOKEN missing in container. Overwrite secret:\n"
            "  modal secret create --force huggingface HF_TOKEN=<token>"
        )
    os.environ["HF_TOKEN"] = token
    return token


def ensure_csv(volume: modal.Volume) -> int:
    if Path(CSV_PATH).exists():
        n = len(pd.read_csv(CSV_PATH))
        if n == N_PROMPTS:
            print(f"using existing CSV ({n} rows) -> {CSV_PATH}")
            return n
        print(f"regenerating CSV ({n} != {N_PROMPTS} rows)")
    ds = load_dataset(SELF_DESCRIBE_DATASET, split="train", token=hf_token())
    ds = ds.shuffle(seed=SAMPLE_SEED).select(range(N_PROMPTS))
    df = ds.to_pandas()[["user_prompt", "attr_class", "attr"]]
    assert len(df) == N_PROMPTS
    df.to_csv(CSV_PATH, index=False)
    volume.commit()
    print(f"wrote {len(df)} rows -> {CSV_PATH}")
    return len(df)


def load_model(model_id: str):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        model_dir = snapshot_download(model_id, cache_dir=HF_HUB_CACHE, token=token)
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
    if len(df) == 0:
        raise ValueError(f"empty CSV at {path}")
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
            hidden = out.hidden_states[layer + 1]

            lengths = attention_mask.sum(dim=1).cpu()
            for i, seq_len in enumerate(lengths.tolist()):
                vec = hidden[i, seq_len - 1].float().cpu().numpy()
                all_vecs.append(vec)

    return np.stack(all_vecs, axis=0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def _write_activations(
    acts: np.ndarray, parquet_path: str, npy_path: str
) -> None:
    np.save(npy_path, acts)
    table = pa.table({"activation_vector": acts.tolist()})
    pq.write_table(table, parquet_path)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(
    prompt_mode: str = "persona-only",
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = LAYER,
):
    import sys

    sys.path.insert(0, "/root")
    from prompt_modes import (
        LEGACY_NPY,
        LEGACY_PARQUET,
        output_paths,
        prepare_prompts,
    )

    vol.reload()
    setup_hf_cache()

    n_rows = ensure_csv(vol)
    raw_prompts = read_prompts(CSV_PATH)
    assert len(raw_prompts) == n_rows
    prompts, missing_suffix = prepare_prompts(raw_prompts, prompt_mode)

    parquet_path, npy_path = output_paths(CACHE, layer, prompt_mode, model_id)
    print(f"prompt_mode={prompt_mode} model_id={model_id} layer={layer}")
    print(f"output parquet -> {parquet_path}")

    model, tokenizer = load_model(model_id)
    vol.commit()

    acts = extract_activations(model, tokenizer, prompts, layer)
    acts = l2_normalize(acts).astype(np.float32)
    _write_activations(acts, parquet_path, npy_path)

    if prompt_mode == "full":
        legacy_parquet = f"{CACHE}/{LEGACY_PARQUET}"
        legacy_npy = f"{CACHE}/{LEGACY_NPY}"
        shutil.copy2(parquet_path, legacy_parquet)
        shutil.copy2(npy_path, legacy_npy)
        print(f"legacy copy -> {legacy_parquet}")
        print(f"legacy copy -> {legacy_npy}")

    vol.commit()

    norms = np.linalg.norm(acts, axis=1)
    print(f"shape: {acts.shape}")
    print(f"mean norm: {norms.mean():.6f}")
    print(f"suffix missing on {len(missing_suffix)} row(s)")
    print(f"saved -> {npy_path}")
    print(f"saved -> {parquet_path}")


@app.local_entrypoint()
def main(
    prompt_mode: str = "persona-only",
    full: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    layer: int = LAYER,
):
    if full:
        prompt_mode = "full"
    run.remote(prompt_mode=prompt_mode, model_id=model_id, layer=layer)
