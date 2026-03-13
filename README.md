# SGLang Runtime Environment — All GPUs (SM75–SM120), AVX2

Flox runtime environment wrapping a pre-built SGLang Nix store path. Supports all NVIDIA GPUs from T4 through RTX 5090 (SM75–SM120), compiled with AVX2 CPU instructions and CUDA 12.8.

## Prerequisites

- **NVIDIA driver 550+** (run `nvidia-smi` — "CUDA Version" must show 12.8 or higher)
- **SGLang store path** — built via `flox build sglang-python312-cuda12_8-all-avx2` in the build environment
- **Flox** — [flox.dev](https://flox.dev)

## Quick Start

```bash
cd sglang-runtime
flox activate

# Python with full SGLang stack
python3.12 -c "import sglang; print(sglang.__version__)"

# Or use the sglang CLI wrapper directly
sglang --help
```

## Serving a Model

### Via Flox service

```bash
# Default model (Llama-3.1-8B-Instruct)
flox activate -s

# Custom model
SGLANG_MODEL=mistralai/Mistral-7B-Instruct-v0.3 flox activate -s
```

### Via command line

```bash
flox activate
python3.12 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000
```

### Check CUDA availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
```

## How It Works

The manifest at `.flox/env/manifest.toml` does the following:

1. **Installs the package** — the `[install]` section references the SGLang store path directly, making the `sglang` binary available on PATH
2. **Builds PYTHONPATH from the full closure** — the `on-activate` hook runs `nix-store -qR` on the store path to discover all transitive dependencies, then filters for `lib/python3.12/site-packages` directories
3. **Sets up JIT compilation** — exports `CUDA_HOME` (for deep_gemm), `CPATH` and `LIBRARY_PATH` (for sgl-kernel JIT) from CUDA packages in the Nix closure
4. **Isolates the environment** — unsets any outer `PYTHONPATH`/`PYTHONHOME` so system Python packages cannot interfere, and sets `FLASHINFER_JIT_DIR` to a writable cache directory (the Nix store is read-only)
5. **Configures the service** — `[services.sglang]` runs `python3.12 -m sglang.launch_server` with configurable model, host, and port via `[vars]`

## After Rebuilds

When you rebuild the SGLang package, update the store path in `.flox/env/manifest.toml`:

```bash
# In the build environment
flox build sglang-python312-cuda12_8-all-avx2
readlink result-sglang-python312-cuda12_8-all-avx2
# Copy the store path and update manifest.toml [install] section
```

The hook re-resolves all dependent store paths (Python, CUDA, etc.) dynamically — only the top-level store path needs updating.

## Troubleshooting

- **HuggingFace model downloads fail with xet/CAS errors**: HuggingFace's newer xet download backend may fail in this environment. Set `HF_HUB_ENABLE_HF_TRANSFER=0` to fall back to standard HTTP downloads:
  ```bash
  export HF_HUB_ENABLE_HF_TRANSFER=0
  python3.12 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
  ```

## Known Limitations

- **`sglang serve` command**: Eagerly imports multimodal/diffusion deps not included in this build. Use `python3.12 -m sglang.launch_server` or the Flox service instead
- **x86_64-linux only**: This environment targets `x86_64-linux` with AVX2 CPU instructions
