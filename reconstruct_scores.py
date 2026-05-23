"""Step 3: AR reconstruction, pairwise consistency, and fidelity on Modal.

Prerequisites:
  1. Step 1 persona activations on nla-cache
  2. Step 2 descriptions.parquet (4800 rows)
  3. Accept license for kitft/nla-gemma3-12b-L32-ar on HuggingFace
  4. modal secret create --force huggingface HF_TOKEN=<token>

Run:
  uv run modal run reconstruct_scores.py
  uv run modal run reconstruct_scores.py --n-shards 1
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import modal
import numpy as np
import pandas as pd

AR_MODEL = "kitft/nla-gemma3-12b-L32-ar"
N_ACTIVATIONS = 400
N_SAMPLES = 12
N_PAIRS = 66
N_SHARDS_DEFAULT = 12
CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{CACHE}/hf/hub"
DESCRIPTIONS_PARQUET = f"{CACHE}/descriptions.parquet"
PAIRWISE_OUT = f"{CACHE}/pairwise_consistency.parquet"
FIDELITY_OUT = f"{CACHE}/fidelity_scores.parquet"

NLA_PKG = Path(__file__).resolve().parent / "nla"

app = modal.App("nla-reconstruct")
vol = modal.Volume.from_name("nla-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.49",
        "safetensors",
        "pyyaml",
        "orjson",
        "httpx",
        "pandas",
        "pyarrow",
        "numpy",
        "huggingface_hub",
    )
    .add_local_dir(NLA_PKG, "/root/nla")
)


def pairwise_shard_parquet(shard_id: int) -> str:
    return f"{CACHE}/pairwise_consistency_shard{shard_id}.parquet"


def fidelity_shard_parquet(shard_id: int) -> str:
    return f"{CACHE}/fidelity_scores_shard{shard_id}.parquet"


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


def download_ar() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        model_dir = snapshot_download(AR_MODEL, cache_dir=HF_HUB_CACHE, token=token)
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError(
                "HF 403: token cannot access gated repos. At huggingface.co/settings/tokens "
                "use a classic Read token, or enable 'Read access to public gated repos' "
                "on your fine-grained token. Also accept the license on the model page."
            ) from e
        raise
    print(f"AR cache: {model_dir}")
    return model_dir


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def _process_shard(
    critic,
    act_df: pd.DataFrame,
    grouped: dict,
    indices: list[int],
    shard_id: int,
    n_shards: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    n = len(indices)

    for n_done, activation_idx in enumerate(indices, 1):
        orig = np.asarray(
            act_df.iloc[activation_idx]["activation_vector"], dtype=np.float32
        )
        g = grouped[activation_idx]
        if len(g) != N_SAMPLES:
            raise ValueError(
                f"activation {activation_idx}: expected {N_SAMPLES} samples, got {len(g)}"
            )
        descs = g["description"].tolist()
        sample_idxs = g["sample_idx"].tolist()
        if sample_idxs != list(range(N_SAMPLES)):
            raise ValueError(
                f"activation {activation_idx}: sample_idx not 0..{N_SAMPLES - 1}"
            )

        vecs = [critic.reconstruct(d).numpy() for d in descs]

        for i, j in combinations(range(N_SAMPLES), 2):
            pairwise_rows.append(
                {
                    "activation_idx": activation_idx,
                    "sample_i": i,
                    "sample_j": j,
                    "cos_sim": cos_sim(vecs[i], vecs[j]),
                }
            )

        for sample_idx, desc in zip(sample_idxs, descs):
            _, fidelity_cos = critic.score(desc, orig)
            fidelity_rows.append(
                {
                    "activation_idx": activation_idx,
                    "sample_idx": sample_idx,
                    "fidelity_cos": fidelity_cos,
                }
            )

        if n_done % 20 == 0 or n_done == n:
            print(f"shard {shard_id}/{n_shards}: {n_done}/{n} activations")

    pairwise_df = pd.DataFrame(pairwise_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)
    assert len(pairwise_df) == n * N_PAIRS
    assert len(fidelity_df) == n * N_SAMPLES
    return pairwise_df, fidelity_df


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_shard(shard_id: int, n_shards: int, activations_parquet: str):
    sys.path.insert(0, "/root")
    from nla.nla_inference import NLACritic

    setup_hf_cache()

    if not Path(activations_parquet).exists():
        raise FileNotFoundError(f"missing activations: {activations_parquet}")
    if not Path(DESCRIPTIONS_PARQUET).exists():
        raise FileNotFoundError(f"missing descriptions: {DESCRIPTIONS_PARQUET}")

    act_df = pd.read_parquet(activations_parquet)
    if len(act_df) != N_ACTIVATIONS:
        raise ValueError(f"expected {N_ACTIVATIONS} activations, got {len(act_df)}")

    desc_df = pd.read_parquet(DESCRIPTIONS_PARQUET)
    expected_desc = N_ACTIVATIONS * N_SAMPLES
    if len(desc_df) != expected_desc:
        raise ValueError(f"expected {expected_desc} descriptions, got {len(desc_df)}")

    grouped = {
        k: g.sort_values("sample_idx")
        for k, g in desc_df.groupby("activation_idx", sort=True)
    }
    if set(grouped.keys()) != set(range(N_ACTIVATIONS)):
        raise ValueError("descriptions missing some activation_idx 0..399")

    indices = [i for i in range(N_ACTIVATIONS) if i % n_shards == shard_id]
    pairwise_path = pairwise_shard_parquet(shard_id)
    fidelity_path = fidelity_shard_parquet(shard_id)
    print(
        f"shard {shard_id}/{n_shards}: {len(indices)} activations "
        f"from {activations_parquet} -> {pairwise_path}, {fidelity_path}"
    )

    ar_dir = download_ar()
    critic = NLACritic(ar_dir, device="cuda")

    pairwise_df, fidelity_df = _process_shard(
        critic, act_df, grouped, indices, shard_id, n_shards
    )
    pairwise_df.to_parquet(pairwise_path, index=False)
    fidelity_df.to_parquet(fidelity_path, index=False)
    vol.commit()
    print(
        f"shard {shard_id}: wrote {len(pairwise_df)} pairwise, "
        f"{len(fidelity_df)} fidelity rows"
    )


@app.function(
    image=image,
    timeout=600,
    volumes={CACHE: vol},
)
def merge_shards(n_shards: int):
    vol.reload()
    for path in (PAIRWISE_OUT, FIDELITY_OUT):
        p = Path(path)
        if p.exists():
            p.unlink()
            print(f"removed stale {path}")

    pairwise_dfs = []
    fidelity_dfs = []
    for shard_id in range(n_shards):
        pw = pairwise_shard_parquet(shard_id)
        fid = fidelity_shard_parquet(shard_id)
        if not Path(pw).exists():
            raise FileNotFoundError(f"missing {pw} — shard {shard_id} did not finish")
        if not Path(fid).exists():
            raise FileNotFoundError(f"missing {fid} — shard {shard_id} did not finish")
        pairwise_dfs.append(pd.read_parquet(pw))
        fidelity_dfs.append(pd.read_parquet(fid))

    pairwise_df = pd.concat(pairwise_dfs, ignore_index=True)
    fidelity_df = pd.concat(fidelity_dfs, ignore_index=True)
    pairwise_df = pairwise_df.sort_values(
        ["activation_idx", "sample_i", "sample_j"]
    ).reset_index(drop=True)
    fidelity_df = fidelity_df.sort_values(
        ["activation_idx", "sample_idx"]
    ).reset_index(drop=True)

    assert len(pairwise_df) == N_ACTIVATIONS * N_PAIRS
    assert len(fidelity_df) == N_ACTIVATIONS * N_SAMPLES

    pairwise_df.to_parquet(PAIRWISE_OUT, index=False)
    fidelity_df.to_parquet(FIDELITY_OUT, index=False)
    vol.commit()

    print(f"pairwise_consistency.parquet: {len(pairwise_df)} rows")
    print(f"fidelity_scores.parquet:      {len(fidelity_df)} rows")
    print(f"mean pairwise cos_sim:        {pairwise_df['cos_sim'].mean():.3f}")
    print(f"std pairwise cos_sim:         {pairwise_df['cos_sim'].std():.3f}")
    print(f"mean fidelity cos:            {fidelity_df['fidelity_cos'].mean():.3f}")
    print(f"any NaN in pairwise:          {pairwise_df['cos_sim'].isna().any()}")
    print(f"any NaN in fidelity:          {fidelity_df['fidelity_cos'].isna().any()}")


@app.local_entrypoint()
def main(n_shards: int = N_SHARDS_DEFAULT):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.prompt_modes import default_activations_parquet

    activations_parquet = default_activations_parquet(CACHE)
    print(f"activations: {activations_parquet}")
    print(f"n_shards: {n_shards}")

    list(
        run_shard.map(
            range(n_shards),
            kwargs={
                "n_shards": n_shards,
                "activations_parquet": activations_parquet,
            },
        )
    )
    merge_shards.remote(n_shards)
