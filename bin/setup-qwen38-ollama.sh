#!/usr/bin/env bash
# @file purpose: Install Ollama and pull all published Qwen3.8-27B quantized tags for local browser-use.
#
# Upstream model: https://huggingface.co/Qwen/Qwen3.8-27B
# Ollama quants:  https://ollama.com/oamazonasgabriel/qwen3.8-27b
#
# Usage:
#   ./bin/setup-qwen38-ollama.sh          # install + pull all tags + start serve
#   ./bin/setup-qwen38-ollama.sh pull     # pull only
#   ./bin/setup-qwen38-ollama.sh serve    # start ollama serve in background

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail
IFS=$'\n'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL_REPO="oamazonasgabriel/qwen3.8-27b"

# All published Ollama tags for Qwen3.8-27B (as of 2026-03).
QWEN38_TAGS=(
	"iq4-xs-256k-text-q4kv"
	"iq4-xs-64k-text-q4kv"
	"q4-km-256k-text-q4kv"
	"q4-km-32gbGPU"
)

_log() {
	echo "[qwen3.8-27b] $*"
}

_ensure_ollama_installed() {
	if command -v ollama >/dev/null 2>&1; then
		_log "Ollama already installed: $(ollama --version 2>/dev/null || ollama -v)"
		return
	fi

	_log "Installing Ollama..."
	curl -fsSL https://ollama.com/install.sh | sh
}

_ollama_is_running() {
	curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1
}

_start_ollama_serve() {
	if _ollama_is_running; then
		_log "Ollama already running at $OLLAMA_HOST"
		return
	fi

	_log "Starting ollama serve in background..."
	nohup ollama serve >"$REPO_ROOT/.ollama-serve.log" 2>&1 &
	disown || true

	for _ in $(seq 1 60); do
		if _ollama_is_running; then
			_log "Ollama ready at $OLLAMA_HOST"
			return
		fi
		sleep 1
	done

	echo "ERROR: ollama serve did not become ready at $OLLAMA_HOST" >&2
	echo "Check $REPO_ROOT/.ollama-serve.log" >&2
	exit 1
}

_pull_all_tags() {
	_start_ollama_serve

	for tag in "${QWEN38_TAGS[@]}"; do
		full="${MODEL_REPO}:${tag}"
		_log "Pulling $full ..."
		ollama pull "$full"
	done

	_log "Installed models:"
	ollama list | rg -i "qwen3\.8-27b|NAME" || ollama list
}

_print_usage() {
	cat <<EOF

Done. Recommended defaults by VRAM:
  32GB+ GPU : ${MODEL_REPO}:q4-km-32gbGPU
  tight VRAM: ${MODEL_REPO}:iq4-xs-64k-text-q4kv

Browser-use example:

  from browser_use import Agent, ChatOllama

  llm = ChatOllama(
      model="${MODEL_REPO}:q4-km-32gbGPU",
      ollama_options={"num_ctx": 65536},
  )
  Agent("your task", llm=llm).run_sync()

Ollama API: $OLLAMA_HOST
Logs:       $REPO_ROOT/.ollama-serve.log
EOF
}

cmd="${1:-all}"

case "$cmd" in
all)
	_ensure_ollama_installed
	_pull_all_tags
	_print_usage
	;;
pull)
	_ensure_ollama_installed
	_pull_all_tags
	;;
serve)
	_ensure_ollama_installed
	_start_ollama_serve
	;;
*)
	echo "Usage: $0 [all|pull|serve]" >&2
	exit 1
	;;
esac
