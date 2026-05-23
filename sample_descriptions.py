"""Step 2: sample AV descriptions from activations via SGLang + NLAClient on Modal.

Prerequisites:
  1. Step 1 done on nla-cache (activations parquet on volume)
  2. Accept license for kitft/nla-gemma3-12b-L32-av on HuggingFace
  3. modal secret create --force huggingface HF_TOKEN=<token>

Run:
  uv run modal run sample_descriptions.py
  uv run modal run sample_descriptions.py --full
  uv run modal run sample_descriptions.py \\
    --activations /cache/activations_layer32_full_gemma-3-12b-pt.parquet
"""

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import modal
import pandas as pd
import pyarrow.parquet as pq

AV_MODEL = "kitft/nla-gemma3-12b-L32-av"
N_SAMPLES = 12
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 500
PORT = 30000
N_RETRIES = 3
RETRY_SLEEP = 5
READY_TIMEOUT = 600
CONCURRENCY = 12
N_SHARDS_DEFAULT = 12

CACHE = "/cache"
HF_CACHE = f"{CACHE}/hf"
HF_HUB_CACHE = f"{CACHE}/hf/hub"
DEFAULT_DESCRIPTIONS_PARQUET = f"{CACHE}/descriptions.parquet"
SGLANG_URL = f"http://127.0.0.1:{PORT}"

NLA_PKG = Path(__file__).resolve().parent / "nla"

app = modal.App("nla-sample")
vol = modal.Volume.from_name("nla-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime")
    .entrypoint([])
    .pip_install(
        "httpx",
        "orjson",
        "pyyaml",
        "safetensors",
        "pandas",
        "pyarrow",
        "huggingface_hub",
    )
    .add_local_dir(NLA_PKG, "/root/nla")
)


def shard_parquet(shard_id: int) -> str:
    return f"{CACHE}/descriptions_shard{shard_id}.parquet"


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


def download_av() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = hf_token()
    try:
        return snapshot_download(AV_MODEL, cache_dir=HF_HUB_CACHE, token=token)
    except HfHubHTTPError as e:
        if e.response.status_code == 403:
            raise RuntimeError(
                "HF 403: accept the AV model license and use a token with gated-repo read access."
            ) from e
        raise


def start_sglang(model_dir: str) -> subprocess.Popen:
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_dir,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--disable-radix-cache",
        "--context-length",
        "512",
        "--mem-fraction-static",
        "0.75",
        "--cuda-graph-max-bs",
        str(CONCURRENCY),
        "--skip-server-warmup",
        "--trust-remote-code",
    ]
    return subprocess.Popen(cmd)


def wait_ready(process: subprocess.Popen, timeout: int = READY_TIMEOUT) -> None:
    deadline = time.time() + timeout
    with httpx.Client() as http:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"SGLang exited with code {process.returncode} "
                    "(often -9/OOM or CUDA graph failure — check logs above)"
                )
            try:
                http.get(f"{SGLANG_URL}/health", timeout=5.0).raise_for_status()
                return
            except (httpx.HTTPError, httpx.RequestError):
                time.sleep(1)
    raise TimeoutError(f"SGLang not ready within {timeout}s")


def _make_client(av_dir: str):
    import numpy as np
    import torch
    from nla.nla_inference import NLAClient

    client = NLAClient(av_dir, sglang_url=SGLANG_URL)
    client._http.close()
    client._http = httpx.Client(
        timeout=httpx.Timeout(120.0),
        limits=httpx.Limits(
            max_connections=CONCURRENCY,
            max_keepalive_connections=CONCURRENCY,
        ),
    )
    return client, np, torch


def _text_from_out(out: dict) -> str:
    from nla.nla_inference import EXPLANATION_RE

    text = out["text"]
    m = EXPLANATION_RE.search(text)
    return m.group(1).strip() if m else text


def _draw(client, embeds_np) -> str:
    for attempt in range(N_RETRIES):
        try:
            out = client._sglang_generate(
                embeds_np,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            return _text_from_out(out)
        except httpx.HTTPError:
            if attempt == N_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP)
    raise RuntimeError("unreachable")


class _EmbedCache:
    def __init__(self, client, np, torch):
        self._client = client
        self._np = np
        self._torch = torch
        self._cache: dict[int, object] = {}
        self._lock = threading.Lock()

    def get(self, activation_idx: int, vec) -> object:
        with self._lock:
            if activation_idx not in self._cache:
                v = self._torch.as_tensor(self._np.asarray(vec, dtype=self._np.float32))
                self._cache[activation_idx] = self._client._build_embeds(v, None)[0]
            return self._cache[activation_idx]


def _one_sample(cache: _EmbedCache, client, activation_idx: int, sample_idx: int, vec) -> dict:
    embeds = cache.get(activation_idx, vec)
    return {
        "activation_idx": activation_idx,
        "sample_idx": sample_idx,
        "description": _draw(client, embeds),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={CACHE: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_shard(shard_id: int, n_shards: int, activations_parquet: str):
    sys.path.insert(0, "/root")

    vol.reload()
    setup_hf_cache()

    if not Path(activations_parquet).exists():
        raise FileNotFoundError(
            f"missing {activations_parquet} — run extract_activations.py first "
            "(e.g. --prompt-mode persona-only)"
        )

    table = pq.read_table(activations_parquet)
    vectors = table.column("activation_vector").to_pylist()
    n_activations = len(vectors)
    if n_activations == 0:
        raise ValueError(f"empty {activations_parquet}")

    indices = [i for i in range(n_activations) if i % n_shards == shard_id]
    out_path = shard_parquet(shard_id)
    print(
        f"shard {shard_id}/{n_shards}: {len(indices)} activations "
        f"from {activations_parquet} -> {out_path}"
    )

    av_dir = download_av()
    vol.commit()

    process = start_sglang(av_dir)
    try:
        wait_ready(process)
        client, np, torch = _make_client(av_dir)
        cache = _EmbedCache(client, np, torch)
        print(
            f"shard {shard_id}: concurrency={CONCURRENCY} "
            f"max_new_tokens={MAX_NEW_TOKENS} embed_cache=on"
        )

        total = len(indices) * N_SAMPLES
        rows = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [
                pool.submit(_one_sample, cache, client, i, s, vectors[i])
                for i in indices
                for s in range(N_SAMPLES)
            ]
            for n, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if n % 50 == 0 or n == total:
                    print(f"shard {shard_id}: {n}/{total} samples")

        df = pd.DataFrame(rows).sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)
        assert len(df) == total
        df.to_parquet(out_path, index=False)
        vol.commit()
        print(f"shard {shard_id}: wrote {len(df)} rows -> {out_path}")
    finally:
        process.terminate()
        process.wait(timeout=60)


@app.function(
    image=image,
    timeout=600,
    volumes={CACHE: vol},
)
def merge_shards(n_shards: int, output_parquet: str = DEFAULT_DESCRIPTIONS_PARQUET):
    vol.reload()
    out = Path(output_parquet)
    if out.exists():
        out.unlink()
        print(f"removed stale {output_parquet}")
    dfs = []
    for shard_id in range(n_shards):
        path = shard_parquet(shard_id)
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {path} — shard {shard_id} did not finish")
        dfs.append(pd.read_parquet(path))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["activation_idx", "sample_idx"]).reset_index(drop=True)
    n_act = df["activation_idx"].nunique()
    expected = n_act * N_SAMPLES
    assert len(df) == expected, (
        f"expected {expected} rows ({n_act} activations × {N_SAMPLES}), got {len(df)}"
    )
    df.to_parquet(output_parquet, index=False)
    vol.commit()
    print(f"merged {len(df)} rows -> {output_parquet}")


@app.local_entrypoint()
def main(
    n_shards: int = N_SHARDS_DEFAULT,
    activations: str = "",
    prompt_mode: str = "persona-only",
    full: bool = False,
    output: str = DEFAULT_DESCRIPTIONS_PARQUET,
    model_id: str = "google/gemma-3-12b-pt",
    layer: int = 32,
):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla.prompt_modes import resolve_activations_parquet

    if full:
        prompt_mode = "full"
    activations_parquet = resolve_activations_parquet(
        CACHE,
        activations=activations or None,
        prompt_mode=prompt_mode,
        model_id=model_id,
        layer=layer,
    )
    output_parquet = output if output.startswith("/") else f"{CACHE}/{output.lstrip('/')}"
    print(f"activations: {activations_parquet}")
    print(f"output: {output_parquet}")

    list(
        run_shard.map(
            range(n_shards),
            kwargs={
                "n_shards": n_shards,
                "activations_parquet": activations_parquet,
            },
        )
    )
    merge_shards.remote(n_shards, output_parquet=output_parquet)
