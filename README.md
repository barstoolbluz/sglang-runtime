# SGLang Runtime

Production SGLang inference server as a Flox environment.

- **SGLang 0.5.9** with Python 3.12
- **CUDA 12.8** (driver 550+)
- **SM75–SM120** (T4, A10, A100, L40, H100, B200, RTX 3090/4090/5090)
- **AVX2** CPU instructions, x86_64-linux only

> To target a specific GPU family instead of the all-SM build, swap the
> package in `.flox/env/manifest.toml` — e.g.
> `flox/sglang-python312-cuda12_8-sm89-avx2` for Ada Lovelace only.

## Quick start

```bash
# Start the server with the default model (Llama-3.1-8B-Instruct)
flox activate --start-services

# Or override the model at activation time
SGLANG_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B flox activate -s
```

### Verify it's running

```bash
# Health check
curl http://127.0.0.1:30000/health

# List loaded models
curl http://127.0.0.1:30000/v1/models

# Chat completion
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

## Architecture

```
manifest.toml
  ├─ [install]   sglang + optional model bundle from Flox catalog
  ├─ [vars]      SGLANG_HOST, SGLANG_PORT
  ├─ [hook]      on-activate: resolve closure → PYTHONPATH + CUDA JIT env + model resolution
  ├─ [services]  sglang: python3.12 -m sglang.launch_server …
  └─ [profile]   banner with version info
```

The environment has no external scripts. Everything is driven by the manifest:

1. **Install** — the `[install]` section pulls the SGLang package from the
   Flox catalog (`flox/sglang-python312-cuda12_8-all-avx2`), which places
   the `sglang` binary on `PATH`.

2. **Hook (on-activate)** — runs at activation time and sets up the Python
   and CUDA environment in four phases:
   - **Phase 1 — Isolation**: unsets `PYTHONPATH` and `PYTHONHOME` so no
     outer Python environment leaks in.
   - **Phase 2 — Python/PYTHONPATH**: resolves the sglang store path via
     `which sglang`, locates Python 3.12 in the Nix closure, adds it to
     `PATH`, then walks the full transitive closure to build `PYTHONPATH`
     from all `site-packages` directories.
   - **Phase 3 — CUDA JIT**: exports `CUDA_HOME` (for deep_gemm JIT),
     `CPATH` (CUDA headers), and `LIBRARY_PATH` (CUDA libraries) from
     `cuda12.8-*` store paths. Creates a writable `FLASHINFER_JIT_DIR`
     since the Nix store is read-only.
   - **Phase 4 — Model resolution**: if `SGLANG_MODEL` is a HuggingFace
     model ID (contains `/` and is not already a local path), checks for a
     bundled model in two layouts: HF cache (`share/models/hub/`) then flat
     (`share/models/<name>/`). If found, rewrites `SGLANG_MODEL` to the
     local path and sets `HF_HUB_OFFLINE=1`.

3. **Service** — `[services.sglang]` runs
   `python3.12 -m sglang.launch_server` with model, host, and port from
   `[vars]`.

> **Why `launch_server` instead of `sglang serve`?** The `sglang serve`
> entry point eagerly imports multimodal/diffusion modules (`remote_pdb`,
> `diffusers`, etc.) that are not included in this build. Using
> `launch_server` directly avoids those imports.

## API reference

SGLang exposes an OpenAI-compatible API plus its own native endpoints.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check — returns 200 when the server is ready |
| `/get_model_info` | `GET` | Model metadata (architecture, context length, etc.) |
| `/v1/models` | `GET` | OpenAI-compatible model list |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat completion |
| `/v1/completions` | `POST` | OpenAI-compatible text completion |
| `/generate` | `POST` | SGLang native generation endpoint |

### Chat completion

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain tensor parallelism in two sentences."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Streaming

```bash
curl --no-buffer http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "max_tokens": 128,
    "stream": true
  }'
```

### Text completion

```bash
curl http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }'
```

### Native generate

```bash
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meaning of life is",
    "sampling_params": {
      "max_new_tokens": 128,
      "temperature": 0.8
    }
  }'
```

## Configuration reference

### Runtime environment variables

`SGLANG_HOST` and `SGLANG_PORT` are set in `[vars]`. `SGLANG_MODEL` is not
in `[vars]` — it is resolved at service start time via the shell expansion
`${SGLANG_MODEL:-meta-llama/Llama-3.1-8B-Instruct}` in the service command.
All three can be overridden at activation time.

| Variable | Default | Set in | Description |
|----------|---------|--------|-------------|
| `SGLANG_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | service command | HuggingFace model ID or local path |
| `SGLANG_HOST` | `0.0.0.0` | `[vars]` | Bind address for the server |
| `SGLANG_PORT` | `30000` | `[vars]` | Listen port for the server |

Override any variable at activation time:

```bash
SGLANG_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
SGLANG_PORT=8080 \
  flox activate --start-services
```

### Engine tuning

SGLang accepts additional launch arguments in the service command. To
customize engine behavior, edit the `command` in `[services.sglang]` to
append flags.

| Flag | Default | Description |
|------|---------|-------------|
| `--tp-size` | `1` | Tensor parallelism degree (number of GPUs) |
| `--mem-fraction-static` | `0.88` | Fraction of GPU memory reserved for KV cache |
| `--dtype` | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `--context-length` | model default | Override maximum context length |
| `--chunked-prefill-size` | `8192` | Chunk size for prefill phase |
| `--max-running-requests` | auto | Cap on concurrent requests |
| `--disable-cuda-graph` | off | Disable CUDA graph capture (saves memory, slower) |
| `--quantization` | none | Quantization method (`awq`, `gptq`, `fp8`, etc.) |
| `--schedule-policy` | `lpm` | Scheduling policy (`lpm`, `random`, `fcfs`, `dfs-weight`) |

Example — serve a 70B model across 4 GPUs with reduced memory:

```bash
# Edit [services.sglang] command to include:
command = "exec python3.12 -m sglang.launch_server --model-path \"${SGLANG_MODEL:-meta-llama/Llama-3.1-70B-Instruct}\" --host $SGLANG_HOST --port $SGLANG_PORT --tp-size 4 --mem-fraction-static 0.80"
```

### Hook-managed variables

These are set automatically by the on-activate hook. They do not need to be
configured manually but are documented for reference.

| Variable | Source | Purpose |
|----------|--------|---------|
| `PYTHONPATH` | Nix closure walk | All `site-packages` from transitive deps |
| `CUDA_HOME` | `cuda_nvcc` store path | `nvcc` location for deep_gemm JIT |
| `CPATH` | `cuda12.8-*` store paths | CUDA headers for JIT compilation |
| `LIBRARY_PATH` | `cuda12.8-*` store paths | CUDA libraries for JIT linking |
| `FLASHINFER_JIT_DIR` | `$FLOX_ENV_CACHE/flashinfer-jit` | Writable cache for FlashInfer JIT kernels |
| `SGLANG_MODEL` | Bundled model snapshot path | Rewritten from HF model ID if bundled model found |
| `SGLANG_BUNDLED_FROM` | Original HF model ID | Set when bundled model detected (used by profile banner) |
| `HF_HUB_OFFLINE` | `1` (when bundled) | Prevents HF Hub network access when serving bundled model |

## Multi-GPU

SGLang supports tensor parallelism for serving large models across multiple
GPUs. Set `--tp-size` in the service command to the number of GPUs:

```bash
# In [services.sglang], append --tp-size:
command = "exec python3.12 -m sglang.launch_server --model-path \"${SGLANG_MODEL:-meta-llama/Llama-3.1-70B-Instruct}\" --host $SGLANG_HOST --port $SGLANG_PORT --tp-size 4"
```

Common configurations:

| Model size | `--tp-size` | GPUs |
|-----------|-------------|------|
| 7–8B | `1` | 1x (any 16 GB+ GPU) |
| 13–14B | `1` or `2` | 1x 24 GB or 2x 16 GB |
| 30–34B | `2` | 2x 24 GB+ |
| 70B | `4` | 4x 24 GB+ or 2x 80 GB |
| 405B | `8` | 8x 80 GB |

All GPUs must be visible to the process. SGLang uses NCCL for cross-GPU
communication.

## Swapping models

Override the model at activation time without editing the manifest:

```bash
# Community model
SGLANG_MODEL=mistralai/Mistral-7B-Instruct-v0.3 flox activate -s

# Local path
SGLANG_MODEL=/data/models/my-fine-tune flox activate -s
```

## Bundled models

Model packages from the Flox catalog (or a custom catalog) can be installed
alongside SGLang so that model weights are included in the Nix closure. When
a bundled model is detected, the hook rewrites `SGLANG_MODEL` to the local
snapshot path and sets `HF_HUB_OFFLINE=1` — no network access is needed at
startup.

### How it works

The hook supports two model package layouts:

**Layout 1 — HF cache** (used by `build-hf-models`):

```
$FLOX_ENV/share/models/hub/
  models--meta-llama--Llama-3.1-8B-Instruct/
    refs/main                    # commit hash
    snapshots/<hash>/
      config.json
      model-00001-of-00004.safetensors
      ...
```

**Layout 2 — Flat** (model name directory):

```
$FLOX_ENV/share/models/
  Qwen2.5-1.5B-Instruct/
    config.json
    model.safetensors
    tokenizer.json
    ...
```

The on-activate hook converts `SGLANG_MODEL` (e.g.
`meta-llama/Llama-3.1-8B-Instruct`) to the HF cache slug and checks for
a matching snapshot directory. If not found, it falls back to checking
`$FLOX_ENV/share/models/<model-name>/` (the part after `/` in the model
ID). Both layouts are validated by checking for `config.json`.

### Installing a model bundle

Add the model package to `[install]` in `manifest.toml`, or install
imperatively with `flox install`:

```bash
flox install flox/qwen2-5-1-5b-instruct
```

Or declaratively:

```toml
[install]
sglang.pkg-path = "flox/sglang-python312-cuda12_8-all-avx2"
sglang.systems = ["x86_64-linux"]

qwen-1-5b.pkg-path = "flox/qwen2-5-1-5b-instruct"
qwen-1-5b.systems = ["x86_64-linux"]
```

Then activate with the matching HF model ID:

```bash
SGLANG_MODEL=Qwen/Qwen2.5-1.5B-Instruct flox activate -s
```

No other configuration is needed — the hook auto-detects the bundled model.
The profile banner will show `(bundled)` instead of `(will download from HF)`.

### Gated models

Some HuggingFace models (Llama, Gemma, etc.) require accepting a license
agreement and providing an access token:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx \
SGLANG_MODEL=meta-llama/Llama-3.1-70B-Instruct \
  flox activate --start-services
```

## Service management

```bash
# Check service status
flox services status

# View logs (follow mode)
flox services logs sglang -f

# View recent logs
flox services logs sglang

# Restart after configuration changes
flox services restart sglang

# Stop all services
flox services stop

# Start fresh
flox activate --start-services
```

## How it works

The on-activate hook performs the following steps in order:

1. **Isolate from outer Python** — unsets `PYTHONPATH` and `PYTHONHOME` to
   prevent any system or virtualenv Python packages from leaking into the
   environment.

2. **Resolve the sglang store path** — follows the `sglang` binary
   (`which sglang → readlink`) back to its Nix store path. This is the
   root from which all transitive dependencies are discovered.

3. **Discover Python 3.12** — runs `nix-store -qR` on the sglang store
   path and finds the `python3-3.12` derivation. Adds its `bin/` to
   `PATH` so `python3.12` is available interactively.

4. **Build PYTHONPATH** — walks the full Nix closure and collects every
   `lib/python3.12/site-packages` directory into `PYTHONPATH`. This gives
   interactive Python access to the entire SGLang dependency tree (torch,
   transformers, flashinfer, etc.).

5. **Set up CUDA JIT environment** — exports:
   - `CUDA_HOME` → the `cuda_nvcc` store path (deep_gemm needs `nvcc`)
   - `CPATH` → all `cuda12.8-*/include` directories (`cuda_runtime.h`,
     `nv/target`, etc.)
   - `LIBRARY_PATH` → all `cuda12.8-*/lib` and `lib64` directories
     (`libcudart.so`, etc.)

6. **Resolve bundled model** — if `SGLANG_MODEL` looks like a HuggingFace
   model ID (contains `/`), checks for a matching model package in two
   layouts: HF cache (`share/models/hub/models--<slug>/snapshots/`) then
   flat (`share/models/<name>/`). If found, rewrites `SGLANG_MODEL` to
   the local path and sets `HF_HUB_OFFLINE=1` to prevent network access.

7. **Create FlashInfer JIT cache** — sets `FLASHINFER_JIT_DIR` to a
   writable directory under `$FLOX_ENV_CACHE` (the Nix store is read-only,
   so JIT-compiled kernels need a mutable location).

## Troubleshooting

### HuggingFace download fails with xet/CAS errors

HuggingFace's newer xet download backend may fail in Nix-built
environments. Disable it:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 flox activate --start-services
```

### Gated model returns 401 Unauthorized

The model requires a HuggingFace access token:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx flox activate --start-services
```

Accept the model's license at `https://huggingface.co/<model>` before
downloading.

### Out of memory (OOM)

Reduce memory pressure with one or more of:

- Lower KV cache fraction: add `--mem-fraction-static 0.70` to the service
  command
- Reduce context length: add `--context-length 4096`
- Use tensor parallelism: add `--tp-size 2` (or more) to spread across GPUs
- Use quantization: add `--quantization fp8` or `--quantization awq`

### `sglang serve` crashes on import

This is expected. The `sglang serve` entry point imports multimodal
dependencies (`diffusers`, `remote_pdb`) not included in this build. The
service already uses `python3.12 -m sglang.launch_server` which avoids
these imports.

### JIT compilation fails

Verify CUDA JIT environment variables are set:

```bash
flox activate
echo $CUDA_HOME      # Should point to cuda_nvcc store path
echo $CPATH          # Should contain cuda12.8-*/include paths
echo $LIBRARY_PATH   # Should contain cuda12.8-*/lib paths
```

If any are empty, the hook may have failed to resolve the Nix closure.
Check that `which sglang` returns a valid store path.

### GPU not detected

```bash
# Check driver
nvidia-smi

# Check CUDA visibility inside the environment
flox activate
python3.12 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Requires NVIDIA driver 550+ with CUDA 12.8 support.

### Port conflict

Change the port:

```bash
SGLANG_PORT=30001 flox activate --start-services
```

Or check what's using port 30000:

```bash
ss -tlnp | grep 30000
```

### FlashInfer JIT cache errors

If FlashInfer fails to compile kernels, clear the JIT cache:

```bash
rm -rf "${FLOX_ENV_CACHE:-$HOME/.cache/flox}/flashinfer-jit"
flox services restart sglang
```

## File structure

```
sglang-runtime/
  .flox/
    env/
      manifest.toml    # Environment definition: packages, hook, service, vars
  README.md            # This file
```

## Known limitations

- **`sglang serve` is not usable** — eagerly imports multimodal/diffusion
  dependencies not included in this build. The service uses
  `python3.12 -m sglang.launch_server` instead.
- **x86_64-linux only** — built with AVX2 CPU instructions. No macOS or
  aarch64 support.
- **Single-node only** — tensor parallelism works across GPUs on the same
  machine. Distributed multi-node serving is not configured.
