#!/usr/bin/env bash
# Pull and start official Qwen3.8-27B quantized models (Ollama GGUF + optional vLLM FP8).
# Usage:
#   ./bin/start_qwen38_27b.sh
#   ./bin/start_qwen38_27b.sh --dry-run
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR/.."

if command -v uv >/dev/null 2>&1; then
	exec uv run python -m browser_use.llm.qwen38_27b start "$@"
fi
exec python -m browser_use.llm.qwen38_27b start "$@"
