"""Step 3: AR reconstruction, pairwise consistency, and fidelity on Modal.

Prerequisites:
  1. Step 1 persona activations on nla-cache
  2. Step 2 descriptions.parquet (4800 rows)
  3. Accept license for kitft/nla-gemma3-12b-L32-ar on HuggingFace
  4. modal secret create --force huggingface HF_TOKEN=<token>

Run:
  # SelfDescribe (default)
  uv run modal run reconstruct_scores.py
  uv run modal run reconstruct_scores.py --n-shards 1

  # PRISM
  uv run modal run reconstruct_scores.py \\
    --activations activations_layer32_persona-only_prism_gemma-3-12b-pt.parquet \\
    --descriptions descriptions_prism.parquet \\
    --pairwise-out pairwise_consistency_prism.parquet \\
    --fidelity-out fidelity_scores_prism.parquet \\
    --n-shards 10
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
VECTORS_OUT = f"{CACHE}/recon_vectors.parquet"

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


def pairwise_shard_parquet(shard_id: int, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{CACHE}/pairwise_consistency{suffix}_shard{shard_id}.parquet"


def fidelity_shard_parquet(shard_id: int, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{CACHE}/fidelity_scores{suffix}_shard{shard_id}.parquet"


def vectors_shard_parquet(shard_id: int, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    return f"{CACHE}/recon_vectors{suffix}_shard{shard_id}.parquet"


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
    save_vectors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None"]:
    pairwise_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    vector_rows: list[dict] | None = [] if save_vectors else None
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

        # Reuse already-reconstructed vecs for fidelity (avoids a second forward pass).
        # cos_sim(vec, orig) == critic.score(desc, orig)[1] since cosine is scale-invariant.
        for sample_idx, vec in zip(sample_idxs, vecs):
            fidelity_rows.append(
                {
                    "activation_idx": activation_idx,
                    "sample_idx": sample_idx,
                    "fidelity_cos": cos_sim(vec, orig),
                }
            )
            if save_vectors:
                vector_rows.append(  # type: ignore[union-attr]
                    {
                        "activation_idx": activation_idx,
                        "sample_idx": sample_idx,
                        "recon_vector": vec.tolist(),
                    }
                )

        if n_done % 20 == 0 or n_done == n:
            print(f"shard {shard_id}/{n_shards}: {n_done}/{n} activations")

    pairwise_df = pd.DataFrame(pairwise_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)
    assert len(pairwise_df) == n * N_PAIRS
    assert len(fidelity_df) == n * N_SAMPLES

    vectors_df: pd.DataFrame | None = None
    if save_vectors:
        vectors_df = pd.DataFrame(vector_rows)
        assert len(vectors_df) == n * N_SAMPLES

    return pairwise_df, fidelity_df, vectors_df


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_shard(
    shard_id: int,
    n_shards: int,
    activations_parquet: str,
    descriptions_parquet: str = DESCRIPTIONS_PARQUET,
    shard_tag: str = "",
    save_vectors: bool = False,
):
    sys.path.insert(0, "/root")
    from nla.nla_inference import NLACritic

    setup_hf_cache()

    if not Path(activations_parquet).exists():
        raise FileNotFoundError(f"missing activations: {activations_parquet}")
    if not Path(descriptions_parquet).exists():
        raise FileNotFoundError(f"missing descriptions: {descriptions_parquet}")

    act_df = pd.read_parquet(activations_parquet)
    if len(act_df) != N_ACTIVATIONS:
        raise ValueError(f"expected {N_ACTIVATIONS} activations, got {len(act_df)}")

    desc_df = pd.read_parquet(descriptions_parquet)
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
    pairwise_path = pairwise_shard_parquet(shard_id, shard_tag)
    fidelity_path = fidelity_shard_parquet(shard_id, shard_tag)
    print(
        f"shard {shard_id}/{n_shards}: {len(indices)} activations "
        f"from {activations_parquet} -> {pairwise_path}, {fidelity_path}"
    )

    ar_dir = download_ar()
    critic = NLACritic(ar_dir, device="cuda")

    pairwise_df, fidelity_df, vectors_df = _process_shard(
        critic, act_df, grouped, indices, shard_id, n_shards, save_vectors=save_vectors
    )
    pairwise_df.to_parquet(pairwise_path, index=False)
    fidelity_df.to_parquet(fidelity_path, index=False)
    if save_vectors and vectors_df is not None:
        vectors_path = vectors_shard_parquet(shard_id, shard_tag)
        vectors_df.to_parquet(vectors_path, index=False)
    vol.commit()
    print(
        f"shard {shard_id}: wrote {len(pairwise_df)} pairwise, "
        f"{len(fidelity_df)} fidelity rows"
        + (f", {len(vectors_df)} vector rows" if save_vectors and vectors_df is not None else "")
    )


@app.function(
    image=image,
    timeout=600,
    volumes={CACHE: vol},
)
def merge_shards(
    n_shards: int,
    pairwise_out: str = PAIRWISE_OUT,
    fidelity_out: str = FIDELITY_OUT,
    vectors_out: str = "",
    shard_tag: str = "",
):
    vol.reload()
    for path in (pairwise_out, fidelity_out):
        p = Path(path)
        if p.exists():
            p.unlink()
            print(f"removed stale {path}")
    if vectors_out and Path(vectors_out).exists():
        Path(vectors_out).unlink()
        print(f"removed stale {vectors_out}")

    pairwise_dfs = []
    fidelity_dfs = []
    vectors_dfs = []
    has_vectors = vectors_out and Path(vectors_shard_parquet(0, shard_tag)).exists()
    for shard_id in range(n_shards):
        pw = pairwise_shard_parquet(shard_id, shard_tag)
        fid = fidelity_shard_parquet(shard_id, shard_tag)
        if not Path(pw).exists():
            raise FileNotFoundError(f"missing {pw} — shard {shard_id} did not finish")
        if not Path(fid).exists():
            raise FileNotFoundError(f"missing {fid} — shard {shard_id} did not finish")
        pairwise_dfs.append(pd.read_parquet(pw))
        fidelity_dfs.append(pd.read_parquet(fid))
        if has_vectors:
            vp = vectors_shard_parquet(shard_id, shard_tag)
            if not Path(vp).exists():
                raise FileNotFoundError(f"missing {vp} — shard {shard_id} did not finish")
            vectors_dfs.append(pd.read_parquet(vp))

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

    pairwise_df.to_parquet(pairwise_out, index=False)
    fidelity_df.to_parquet(fidelity_out, index=False)

    if has_vectors and vectors_out:
        vectors_df = pd.concat(vectors_dfs, ignore_index=True)
        vectors_df = vectors_df.sort_values(
            ["activation_idx", "sample_idx"]
        ).reset_index(drop=True)
        assert len(vectors_df) == N_ACTIVATIONS * N_SAMPLES
        vectors_df.to_parquet(vectors_out, index=False)
        print(f"{Path(vectors_out).name}: {len(vectors_df)} rows")

    vol.commit()

    print(f"{Path(pairwise_out).name}: {len(pairwise_df)} rows")
    print(f"{Path(fidelity_out).name}: {len(fidelity_df)} rows")
    print(f"mean pairwise cos_sim:        {pairwise_df['cos_sim'].mean():.3f}")
    print(f"std pairwise cos_sim:         {pairwise_df['cos_sim'].std():.3f}")
    print(f"mean fidelity cos:            {fidelity_df['fidelity_cos'].mean():.3f}")
    print(f"any NaN in pairwise:          {pairwise_df['cos_sim'].isna().any()}")
    print(f"any NaN in fidelity:          {fidelity_df['fidelity_cos'].isna().any()}")


@app.local_entrypoint()
def main(
    n_shards: int = N_SHARDS_DEFAULT,
    activations: str = "",
    descriptions: str = "",
    pairwise_out: str = "",
    fidelity_out: str = "",
    save_vectors: bool = False,
    vectors_out: str = "",
):
    """
    --activations    filename (not full path) of activations parquet on the volume
                     (default: selfdescribe persona-only parquet)
    --descriptions   filename of descriptions parquet on the volume
                     (default: descriptions.parquet)
    --pairwise-out   filename for merged pairwise output (default: pairwise_consistency.parquet)
    --fidelity-out   filename for merged fidelity output (default: fidelity_scores.parquet)
    --save-vectors   also save raw reconstructed vectors to a separate parquet
    --vectors-out    filename for merged vectors output (default: recon_vectors[_TAG].parquet)
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.prompt_modes import default_activations_parquet

    activations_parquet = f"{CACHE}/{activations}" if activations else default_activations_parquet(CACHE)
    descriptions_parquet = f"{CACHE}/{descriptions}" if descriptions else DESCRIPTIONS_PARQUET
    resolved_pairwise_out = f"{CACHE}/{pairwise_out}" if pairwise_out else PAIRWISE_OUT
    resolved_fidelity_out = f"{CACHE}/{fidelity_out}" if fidelity_out else FIDELITY_OUT

    # Derive a short tag from the pairwise output filename to namespace shard files.
    # e.g. "pairwise_consistency_prism.parquet" -> tag "prism"
    #      "pairwise_consistency.parquet"        -> tag "" (no suffix on shard files)
    pw_stem = Path(resolved_pairwise_out).stem  # e.g. "pairwise_consistency_prism"
    default_stem = Path(PAIRWISE_OUT).stem       # "pairwise_consistency"
    shard_tag = pw_stem[len(default_stem):].lstrip("_")  # "" or "prism"

    # Derive vectors output filename from shard_tag when not explicitly provided.
    resolved_vectors_out = ""
    if save_vectors:
        vf = vectors_out if vectors_out else f"recon_vectors{'_' + shard_tag if shard_tag else ''}.parquet"
        resolved_vectors_out = f"{CACHE}/{vf}"

    print(f"activations:      {activations_parquet}")
    print(f"descriptions:     {descriptions_parquet}")
    print(f"pairwise_out:     {resolved_pairwise_out}")
    print(f"fidelity_out:     {resolved_fidelity_out}")
    print(f"n_shards:         {n_shards}")
    print(f"shard_tag:        {shard_tag!r}")
    print(f"save_vectors:     {save_vectors}")
    if resolved_vectors_out:
        print(f"vectors_out:      {resolved_vectors_out}")

    list(
        run_shard.map(
            range(n_shards),
            kwargs={
                "n_shards": n_shards,
                "activations_parquet": activations_parquet,
                "descriptions_parquet": descriptions_parquet,
                "shard_tag": shard_tag,
                "save_vectors": save_vectors,
            },
        )
    )
    merge_shards.remote(
        n_shards,
        pairwise_out=resolved_pairwise_out,
        fidelity_out=resolved_fidelity_out,
        vectors_out=resolved_vectors_out,
        shard_tag=shard_tag,
    )
