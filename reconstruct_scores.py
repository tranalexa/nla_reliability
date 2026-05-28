"""Step 3: AR reconstruction + pairwise consistency + fidelity scoring on Modal.

Uses **Anthropic NLA** infrastructure: the AR (activation reconstructor)
checkpoint ``kitft/nla-gemma3-12b-L32-ar`` released by Anthropic and the
vendored ``NLACritic`` from kitft/natural_language_autoencoders (see
``nla/ATTRIBUTION.md``). NLACritic takes each AV description, regenerates an
activation vector, and we compare it to:

  * the matching original activation (fidelity_cos), and
  * each of the other 11 reconstructions of the same activation (pairwise cos).

Outputs per run on the Modal volume:

  /cache/runs/<run_id>/pairwise_consistency_<dataset>.parquet     (~26,400 rows)
  /cache/runs/<run_id>/fidelity_scores_<dataset>.parquet          (~4,800 rows)
  /cache/runs/<run_id>/recon_vectors_<dataset>.parquet            (with --save-vectors)

Prerequisites:
  1. Step 1 + Step 2 completed for this run_id (volume parquets present).
  2. License accepted for kitft/nla-gemma3-12b-L32-ar on HuggingFace.
  3. modal secret create --force huggingface HF_TOKEN=<token>

Run:
  uv run modal run reconstruct_scores.py --run-id prism --save-vectors
  uv run modal run reconstruct_scores.py --run-id biosbias --save-vectors
  uv run modal run reconstruct_scores.py --run-id mmlu_choice --save-vectors
  uv run modal run reconstruct_scores.py --run-id mmlu_nochoice --save-vectors
  uv run modal run reconstruct_scores.py --all-runs --save-vectors
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import modal
import numpy as np
import pandas as pd

AR_MODEL = "kitft/nla-gemma3-12b-L32-ar"
N_ITEMS = 400
N_SAMPLES = 12
N_PAIRS = 66  # C(12, 2)
N_SHARDS_DEFAULT = 12

CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{CACHE}/hf/hub"

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


def pairwise_shard_parquet(shard_id: int, run_id: str, dataset: str) -> str:
    return f"{CACHE}/runs/{run_id}/pairwise_consistency_{dataset}_shard{shard_id}.parquet"


def fidelity_shard_parquet(shard_id: int, run_id: str, dataset: str) -> str:
    return f"{CACHE}/runs/{run_id}/fidelity_scores_{dataset}_shard{shard_id}.parquet"


def vectors_shard_parquet(shard_id: int, run_id: str, dataset: str) -> str:
    return f"{CACHE}/runs/{run_id}/recon_vectors_{dataset}_shard{shard_id}.parquet"


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
            "HF_TOKEN missing in container. Set it with:\n"
            "  modal secret create --force huggingface HF_TOKEN=<token>"
        )
    os.environ["HF_TOKEN"] = token
    return token


def download_ar() -> str:
    """Snapshot-download the Anthropic AR checkpoint to the volume HF cache."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        model_dir = snapshot_download(AR_MODEL, cache_dir=HF_HUB_CACHE, token=token)
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError(
                "HF 403: token cannot access gated repos. Accept the license on the model "
                "page and use a classic Read token or enable gated-repo access."
            ) from e
        raise
    print(f"AR cache: {model_dir}")
    return model_dir


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two raw float vectors."""
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
    n_samples: int,
    save_vectors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None"]:
    """Reconstruct + score one shard's worth of activations."""
    pairwise_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    vector_rows: list[dict] | None = [] if save_vectors else None
    n = len(indices)

    for n_done, activation_idx in enumerate(indices, 1):
        orig = np.asarray(
            act_df.iloc[activation_idx]["activation_vector"], dtype=np.float32
        )
        g = grouped[activation_idx]
        if len(g) != n_samples:
            raise ValueError(
                f"activation {activation_idx}: expected {n_samples} samples, got {len(g)}"
            )
        descs = g["description"].tolist()
        sample_idxs = g["sample_idx"].tolist()
        if sample_idxs != list(range(n_samples)):
            raise ValueError(
                f"activation {activation_idx}: sample_idx not 0..{n_samples - 1}"
            )

        vecs = [critic.reconstruct(d).numpy() for d in descs]

        # All C(n_samples, 2) within-item pairs.
        for i, j in combinations(range(n_samples), 2):
            pairwise_rows.append(
                {
                    "activation_idx": activation_idx,
                    "sample_i": i,
                    "sample_j": j,
                    "cos_sim": cos_sim(vecs[i], vecs[j]),
                }
            )

        # Per-sample fidelity vs the matching original.
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

    n_pairs = len(list(combinations(range(n_samples), 2)))
    pairwise_df = pd.DataFrame(pairwise_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)
    assert len(pairwise_df) == n * n_pairs
    assert len(fidelity_df) == n * n_samples

    vectors_df: pd.DataFrame | None = None
    if save_vectors:
        vectors_df = pd.DataFrame(vector_rows)
        assert len(vectors_df) == n * n_samples

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
    descriptions_parquet: str,
    run_id: str,
    dataset: str,
    n_items: int,
    n_samples: int,
    save_vectors: bool = False,
):
    """One GPU shard: load AR checkpoint, reconstruct + score activations modulo shard."""
    sys.path.insert(0, "/root")
    from nla.nla_inference import NLACritic

    setup_hf_cache()

    if not Path(activations_parquet).exists():
        raise FileNotFoundError(f"missing activations: {activations_parquet}")
    if not Path(descriptions_parquet).exists():
        raise FileNotFoundError(f"missing descriptions: {descriptions_parquet}")

    act_df = pd.read_parquet(activations_parquet)
    if len(act_df) != n_items:
        raise ValueError(f"expected {n_items} activations, got {len(act_df)}")

    desc_df = pd.read_parquet(descriptions_parquet)
    expected_desc = n_items * n_samples
    if len(desc_df) != expected_desc:
        raise ValueError(f"expected {expected_desc} descriptions, got {len(desc_df)}")

    grouped = {
        k: g.sort_values("sample_idx")
        for k, g in desc_df.groupby("activation_idx", sort=True)
    }
    if set(grouped.keys()) != set(range(n_items)):
        raise ValueError(f"descriptions missing some activation_idx 0..{n_items - 1}")

    indices = [i for i in range(n_items) if i % n_shards == shard_id]
    pairwise_path = pairwise_shard_parquet(shard_id, run_id, dataset)
    fidelity_path = fidelity_shard_parquet(shard_id, run_id, dataset)
    Path(pairwise_path).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"shard {shard_id}/{n_shards}: {len(indices)} activations "
        f"from {activations_parquet} -> {pairwise_path}, {fidelity_path}"
    )

    ar_dir = download_ar()
    critic = NLACritic(ar_dir, device="cuda")

    pairwise_df, fidelity_df, vectors_df = _process_shard(
        critic, act_df, grouped, indices, shard_id, n_shards, n_samples, save_vectors=save_vectors
    )
    pairwise_df.to_parquet(pairwise_path, index=False)
    fidelity_df.to_parquet(fidelity_path, index=False)
    if save_vectors and vectors_df is not None:
        vectors_path = vectors_shard_parquet(shard_id, run_id, dataset)
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
    run_id: str,
    dataset: str,
    pairwise_out: str,
    fidelity_out: str,
    vectors_out: str,
    n_items: int,
    n_samples: int,
    save_vectors: bool,
):
    """Concatenate per-shard parquets into the canonical run outputs and sanity-check."""
    vol.reload()
    n_pairs = len(list(combinations(range(n_samples), 2)))

    for path in (pairwise_out, fidelity_out):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
            print(f"removed stale {path}")
    if vectors_out and Path(vectors_out).exists():
        Path(vectors_out).unlink()
        print(f"removed stale {vectors_out}")

    pairwise_dfs = []
    fidelity_dfs = []
    vectors_dfs = []
    has_vectors = save_vectors and Path(vectors_shard_parquet(0, run_id, dataset)).exists()

    for shard_id in range(n_shards):
        pw = pairwise_shard_parquet(shard_id, run_id, dataset)
        fid = fidelity_shard_parquet(shard_id, run_id, dataset)
        if not Path(pw).exists():
            raise FileNotFoundError(f"missing {pw} - shard {shard_id} did not finish")
        if not Path(fid).exists():
            raise FileNotFoundError(f"missing {fid} - shard {shard_id} did not finish")
        pairwise_dfs.append(pd.read_parquet(pw))
        fidelity_dfs.append(pd.read_parquet(fid))
        if has_vectors:
            vp = vectors_shard_parquet(shard_id, run_id, dataset)
            if not Path(vp).exists():
                raise FileNotFoundError(f"missing {vp} - shard {shard_id} did not finish")
            vectors_dfs.append(pd.read_parquet(vp))

    pairwise_df = pd.concat(pairwise_dfs, ignore_index=True)
    fidelity_df = pd.concat(fidelity_dfs, ignore_index=True)
    pairwise_df = pairwise_df.sort_values(
        ["activation_idx", "sample_i", "sample_j"]
    ).reset_index(drop=True)
    fidelity_df = fidelity_df.sort_values(
        ["activation_idx", "sample_idx"]
    ).reset_index(drop=True)

    assert len(pairwise_df) == n_items * n_pairs, (
        f"expected {n_items * n_pairs} pairwise rows, got {len(pairwise_df)}"
    )
    assert len(fidelity_df) == n_items * n_samples, (
        f"expected {n_items * n_samples} fidelity rows, got {len(fidelity_df)}"
    )

    pairwise_df.to_parquet(pairwise_out, index=False)
    fidelity_df.to_parquet(fidelity_out, index=False)

    if has_vectors and vectors_out:
        vectors_df = pd.concat(vectors_dfs, ignore_index=True)
        vectors_df = vectors_df.sort_values(
            ["activation_idx", "sample_idx"]
        ).reset_index(drop=True)
        assert len(vectors_df) == n_items * n_samples
        vectors_df.to_parquet(vectors_out, index=False)
        print(f"{Path(vectors_out).name}: {len(vectors_df)} rows")

    vol.commit()
    print(f"{Path(pairwise_out).name}: {len(pairwise_df)} rows")
    print(f"{Path(fidelity_out).name}: {len(fidelity_df)} rows")
    print(f"mean pairwise cos_sim: {pairwise_df['cos_sim'].mean():.3f}")
    print(f"std pairwise cos_sim:  {pairwise_df['cos_sim'].std():.3f}")
    print(f"mean fidelity cos:     {fidelity_df['fidelity_cos'].mean():.3f}")
    print(f"any NaN in pairwise:   {pairwise_df['cos_sim'].isna().any()}")
    print(f"any NaN in fidelity:   {fidelity_df['fidelity_cos'].isna().any()}")


def _run_one(
    run_id: str,
    n_shards: int,
    n_items: int,
    n_samples: int,
    save_vectors: bool,
) -> None:
    """Drive shards + merge for a single run_id."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.paths import (
        dataset_for_run,
        volume_activations_path,
        volume_descriptions_path,
        volume_fidelity_path,
        volume_pairwise_path,
        volume_recon_vectors_path,
    )

    dataset = dataset_for_run(run_id)
    activations_parquet = volume_activations_path(run_id)
    descriptions_parquet = volume_descriptions_path(run_id)
    pairwise_out = volume_pairwise_path(run_id)
    fidelity_out = volume_fidelity_path(run_id)
    vectors_out = volume_recon_vectors_path(run_id) if save_vectors else ""

    print(f"\n=== Reconstructing scores: {run_id} ({dataset}) ===")
    print(f"  activations:  {activations_parquet}")
    print(f"  descriptions: {descriptions_parquet}")
    print(f"  pairwise_out: {pairwise_out}")
    print(f"  fidelity_out: {fidelity_out}")
    print(f"  n_items:      {n_items}  n_samples: {n_samples}  n_shards: {n_shards}")
    print(f"  save_vectors: {save_vectors}")
    if vectors_out:
        print(f"  vectors_out:  {vectors_out}")

    list(
        run_shard.map(
            range(n_shards),
            kwargs={
                "n_shards": n_shards,
                "activations_parquet": activations_parquet,
                "descriptions_parquet": descriptions_parquet,
                "run_id": run_id,
                "dataset": dataset,
                "n_items": n_items,
                "n_samples": n_samples,
                "save_vectors": save_vectors,
            },
        )
    )
    merge_shards.remote(
        n_shards=n_shards,
        run_id=run_id,
        dataset=dataset,
        pairwise_out=pairwise_out,
        fidelity_out=fidelity_out,
        vectors_out=vectors_out,
        n_items=n_items,
        n_samples=n_samples,
        save_vectors=save_vectors,
    )


@app.local_entrypoint()
def main(
    run_id: str = "prism",
    all_runs: bool = False,
    n_shards: int = N_SHARDS_DEFAULT,
    n_items: int = N_ITEMS,
    n_samples: int = N_SAMPLES,
    save_vectors: bool = False,
):
    """
    --run-id        prism | biosbias | mmlu_choice | mmlu_nochoice  (default: prism)
    --all-runs      run all four canonical runs in sequence
    --save-vectors  also save raw reconstructed vectors (required for centered diagnostics)
    --n-shards      parallel GPU workers per run (default: 12)
    --n-items       items per run (default: 400; must match Step 1)
    --n-samples     AV samples per activation (default: 12; must match Step 2)
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.paths import RUN_IDS, validate_run_id

    runs = list(RUN_IDS) if all_runs else [run_id]
    for r in runs:
        validate_run_id(r)
        _run_one(
            run_id=r,
            n_shards=n_shards,
            n_items=n_items,
            n_samples=n_samples,
            save_vectors=save_vectors,
        )
