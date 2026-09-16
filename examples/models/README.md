# Gemini
Detailed video on how to integrate browser-use with Gemini: https://www.youtube.com/watch?v=JluZiWBV_Tc

# Local Qwen3.8-27B

Download the official `Qwen/Qwen3.8-27B` weights and start every quantized GGUF/Ollama tag that fits this machine:

```bash
uv run python -m scripts.qwen38_local start --all
uv run python examples/models/qwen38_27b_local.py
```

See `scripts/qwen38_local/README.md` for the quant ladder, ports, and hardware table.
