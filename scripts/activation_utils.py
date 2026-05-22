"""Load SelfDescribe rows and Gemma residual activations (parquet or live forward)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompt_modes import (  # noqa: E402
    INFOBOX_SUFFIX,
    activations_basename,
    apply_prompt_mode,
    prepare_prompts,
)

MODEL_ID = "google/gemma-3-12b-pt"
DEFAULT_LAYER = 32
BATCH_SIZE = 4

# Backward-compatible alias
INFOBOX_BOILERPLATE = INFOBOX_SUFFIX

PromptMode = Literal["full", "persona-only", "persona"]
TokenPosition = Literal["last"]

_model_cache: tuple[object, object] | None = None


def strip_infobox_boilerplate(prompt: str) -> str:
    text, _ = apply_prompt_mode(prompt, "persona-only")
    return text


def prompt_for_mode(prompt: str, mode: PromptMode) -> str:
    if mode == "persona":
        mode = "persona-only"
    text, _ = apply_prompt_mode(prompt, mode)
    return text


def load_selfdescribe(csv_path: str | Path) -> list[dict]:
    """Return rows with keys prompt, label, attribute_class."""
    df = pd.read_csv(csv_path)
    required = {"user_prompt", "attr", "attr_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "prompt": str(r["user_prompt"]),
                "label": str(r["attr"]),
                "attribute_class": str(r["attr_class"]),
            }
        )
    return rows


def load_activation_matrix(parquet_path: str | Path) -> np.ndarray:
    table = pq.read_table(parquet_path)
    if "activation_vector" not in table.column_names:
        raise ValueError(
            f"{parquet_path} must have activation_vector column, got {table.column_names}"
        )
    rows = table.column("activation_vector").to_pylist()
    return np.asarray(rows, dtype=np.float32)


class ActivationSource:
    """Parquet-first activations; live Gemma forward when needed."""

    def __init__(
        self,
        *,
        prompt_mode: PromptMode = "persona-only",
        layer: int = DEFAULT_LAYER,
        parquet_path: str | Path | None = None,
        force_live: bool = False,
        l2_normalize: bool = True,
        model_id: str = MODEL_ID,
    ) -> None:
        if prompt_mode == "persona":
            prompt_mode = "persona-only"
        self.prompt_mode: PromptMode = prompt_mode
        self.layer = layer
        self.model_id = model_id
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.force_live = force_live
        self.l2_normalize = l2_normalize
        self._matrix: np.ndarray | None = None

        if (
            not force_live
            and self.parquet_path is not None
            and self.parquet_path.exists()
        ):
            self._matrix = load_activation_matrix(self.parquet_path)

    def build_matrix(self, dataset: list[dict]) -> np.ndarray:
        n = len(dataset)
        if self._matrix is not None:
            if self._matrix.shape[0] != n:
                raise ValueError(
                    f"parquet has {self._matrix.shape[0]} rows, dataset has {n}"
                )
            return self._matrix

        if not self.force_live:
            path = self.parquet_path
            hint = f" ({path})" if path else ""
            expected = activations_basename(
                self.layer, self.prompt_mode, self.model_id
            )
            raise FileNotFoundError(
                f"parquet not found{hint}. Run Step 1 or pass --live. "
                f"Expected naming like {expected}"
            )

        print(
            f"Running live Gemma forward (mode={self.prompt_mode}, "
            f"layer={self.layer}, model={self.model_id})..."
        )
        raw = [r["prompt"] for r in dataset]
        texts, _ = prepare_prompts(raw, self.prompt_mode)
        vecs = _forward_batch(texts, self.layer, model_id=self.model_id)
        if self.l2_normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs.astype(np.float32)


def get_activation(
    prompt: str,
    layer: int,
    token_position: TokenPosition = "last",
    *,
    mode: PromptMode = "persona-only",
    row_index: int | None = None,
    source: ActivationSource | None = None,
) -> np.ndarray:
    """Return residual activation at the given layer and token position."""
    if token_position != "last":
        raise ValueError(
            f"token_position {token_position!r} not supported; only 'last' is implemented"
        )
    if source is not None and source._matrix is not None and row_index is not None:
        return source._matrix[row_index].copy()

    text = prompt_for_mode(prompt, mode)
    mid = source.model_id if source else MODEL_ID
    vec = _forward_batch([text], layer, model_id=mid)[0]
    if source is None or source.l2_normalize:
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
    return vec.astype(np.float32)


def _load_gemma(model_id: str = MODEL_ID):
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    if hasattr(model, "language_model"):
        inner = model.language_model
        if hasattr(inner, "lm_head"):
            model = inner
        else:
            wrapper = AutoModelForCausalLM.from_config(inner.config)
            wrapper.model = inner
            if getattr(inner.config, "tie_word_embeddings", False):
                wrapper.tie_weights()
            model = wrapper.eval().to(device)

    _model_cache = (model, tokenizer)
    return _model_cache


def _forward_batch(
    prompts: list[str],
    layer: int,
    batch_size: int = 4,
    *,
    model_id: str = MODEL_ID,
) -> np.ndarray:
    import torch

    model, tokenizer = _load_gemma(model_id)
    device = model.get_input_embeddings().weight.device
    all_vecs = []

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
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
