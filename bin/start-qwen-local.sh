#!/usr/bin/env bash
# Start Ollama and pull all Linux-compatible Qwen3.8-27B quantized models for local browser-use.
# Usage:
#   ./bin/start-qwen-local.sh          # pull models + start server
#   ./bin/start-qwen-local.sh --serve  # only start ollama serve (models already pulled)

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

# Linux x86_64 quantized variants (skip MLX / Apple-only tags).
QWEN_27B_MODELS=(
	qwen3.8:27b
	qwen3.8:27b-q4_K_M
	qwen3.8:27b-q8_0
	qwen3.8:27b-mtp-q4_K_M
	qwen3.8:27b-mtp-q8_0
)

_log() {
	printf '[qwen-local] %s\n' "$*"
}

_ensure_ollama() {
	if ! command -v ollama >/dev/null 2>&1; then
		_log 'Installing Ollama...'
		if ! command -v zstd >/dev/null 2>&1; then
			sudo apt-get update -qq
			sudo apt-get install -y zstd
		fi
		curl -fsSL https://ollama.com/install.sh | sh
	fi
}

_ollama_running() {
	curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1
}

_start_ollama() {
	if _ollama_running; then
		_log "Ollama already running at $OLLAMA_HOST"
		return
	fi

	_log 'Starting ollama serve in background...'
	nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
	for _ in $(seq 1 30); do
		if _ollama_running; then
			_log "Ollama ready at $OLLAMA_HOST"
			return
		fi
		sleep 1
	done
	_log "Failed to start Ollama. Check /tmp/ollama-serve.log"
	exit 1
}

_pull_models() {
	_start_ollama
	for model in "${QWEN_27B_MODELS[@]}"; do
		_log "Pulling $model ..."
		ollama pull "$model"
	done
	_log 'Installed models:'
	ollama list
}

_setup_python() {
	if ! command -v uv >/dev/null 2>&1; then
		_log 'Installing uv...'
		curl -LsSf https://astral.sh/uv/install.sh | sh
		export PATH="$HOME/.local/bin:$PATH"
	fi

	_log 'Syncing browser-use Python environment...'
	uv venv --python 3.11
	# shellcheck disable=SC1091
	source .venv/bin/activate
	uv sync
}

_print_usage() {
	cat <<EOF

Ready. Example:

  source .venv/bin/activate
  python -c "
import asyncio
from browser_use import Agent, ChatOllama

llm = ChatOllama(
    model='qwen3.8:27b-q4_K_M',
    ollama_options={'num_ctx': 32768},
)
asyncio.run(Agent(task='Find the founders of browser-use', llm=llm).run())
"

Default model tags:
  - qwen3.8:27b            (Q4_K_M, ~18GB, recommended)
  - qwen3.8:27b-q4_K_M     (explicit Q4_K_M)
  - qwen3.8:27b-q8_0       (higher quality, ~30GB)
  - qwen3.8:27b-mtp-q4_K_M (MTP speculative decoding, Q4)
  - qwen3.8:27b-mtp-q8_0   (MTP speculative decoding, Q8)

EOF
}

_ensure_ollama

if [[ "${1:-}" == '--serve' ]]; then
	_start_ollama
	exit 0
fi

_pull_models
_setup_python
_print_usage
