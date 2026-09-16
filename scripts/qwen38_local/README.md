# Qwen3.8-27B local quantized models

Official source: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0, Aug 2026).  
Quantized GGUFs: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (Dynamic V3, from those same weights).  
Ollama tags: [`qwen3.8:27b`](https://ollama.com/library/qwen3.8:27b) (needs Ollama ≥ 0.32.12).  
llama.cpp: `b10419+` (Gated DeltaNet). This launcher pins `b10964`.

This host-side toolkit is for **your** GPU/CPU box. A 15GiB CPU-only machine can download weights; it cannot run a 27B quant at usable speed.

## One command

From the repo root:

```bash
uv run python -m scripts.qwen38_local list
uv run python -m scripts.qwen38_local plan --all --dry-run
uv run python -m scripts.qwen38_local start --all
```

`start --all` does, in order:

1. Snapshot RAM/disk/GPU and skip anything that cannot fit (unless `--force-all`).
2. Download the official `Qwen/Qwen3.8-27B` safetensors.
3. Download every Unsloth GGUF + vision `mmproj` that still fits, pinning `UD-Q4_K_XL` first (the 24GB default).
4. If the machine has ≥32GiB RAM, convert the official weights to BF16 GGUF and run `llama-quantize` for `Q2_K` … `Q8_0`.
5. Start `llama-server` in router mode (`--models-dir`, `--models-max 1`) on `http://127.0.0.1:8080/v1`.
6. `ollama pull` every Linux-compatible `qwen3.8:27b-*` tag (MLX tags are skipped on Linux).

Artifacts land in `~/.cache/browser-use/qwen3.8-27b` (override with `QWEN38_MODELS_DIR` or `--models-dir`). A `local.env` file is written next to them.

## Talk to it from browser-use

```bash
set -a && source ~/.cache/browser-use/qwen3.8-27b/local.env && set +a
uv run python examples/models/qwen38_27b_local.py
```

OpenAI-compatible:

```python
from browser_use.llm import ChatOpenAI

llm = ChatOpenAI(
	model='Qwen3.8-27B-UD-Q4_K_XL',
	base_url='http://127.0.0.1:8080/v1',
	api_key='local',
	temperature=0.7,
	top_p=0.8,
)
```

Ollama:

```python
from browser_use.llm import ChatOllama

llm = ChatOllama(model='qwen3.8:27b', host='http://127.0.0.1:11434')
```

Switch GGUF by sending a different `model` name (the GGUF stem). llama-server loads on demand and evicts with LRU.

## Hardware cheat sheet

| Box | Use |
| --- | --- |
| 16GB | `UD-IQ3_S` / `UD-IQ4_XS` |
| 24GB | `UD-Q4_K_XL` (default) |
| 32GB | `UD-Q6_K_XL` |
| 48GB+ | `Q8_0` / `UD-Q8_K_XL` |
| 64GB+ | official BF16 / `qwen3.8:27b-bf16` |

Qwen3.8 thinking mode is **off** in the llama-server router so browser-use JSON actions stay parseable. Turn it back on in the preset if you want reasoning traces.

## Flags

- `--backend llama|ollama|both` (default `both`)
- `--force-all` ignore disk/RAM packing
- `--force-self-quant` convert official BF16 even on small RAM hosts (will likely OOM)
- `--skip-self-quant` Unsloth/Ollama only
- `--dry-run` print the plan
- `--foreground` do not daemonize llama-server
