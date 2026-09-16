#!/usr/bin/env bash
# Download and register all official Ollama Qwen3.8-27B quantizations for local use.
# Usage:
#   ./bin/setup-qwen38-local.sh          # pull all tags + start ollama serve
#   ./bin/setup-qwen38-local.sh --pull   # pull only
#   ./bin/setup-qwen38-local.sh --serve  # start serve only

set -o errexit
set -o errtrace
set -o nounset
set -o pipefail
IFS=$'\n'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Official Ollama library tags for Qwen3.8-27B (source: https://ollama.com/library/qwen3.8/tags)
ALL_TAGS=(
	qwen3.8:latest
	qwen3.8:27b
	qwen3.8:27b-q4_K_M
	qwen3.8:27b-q8_0
	qwen3.8:27b-bf16
	qwen3.8:27b-mxfp8
	qwen3.8:27b-nvfp4
	qwen3.8:27b-mtp-q4_K_M
	qwen3.8:27b-mtp-q8_0
	qwen3.8:27b-mtp-bf16
	qwen3.8:27b-mlx
	qwen3.8:27b-mlx-bf16
)

DO_PULL=true
DO_SERVE=true

for arg in "$@"; do
	case "$arg" in
		--pull) DO_SERVE=false ;;
		--serve) DO_PULL=false ;;
		--help|-h)
			echo "Usage: $0 [--pull] [--serve]"
			exit 0
			;;
		*)
			echo "Unknown argument: $arg" >&2
			exit 1
			;;
	esac
done

_log() {
	echo "[qwen3.8-local] $*"
}

ensure_ollama() {
	if command -v ollama >/dev/null 2>&1; then
		_log "ollama already installed: $(ollama --version 2>/dev/null || true)"
		return
	fi

	_log "Installing Ollama..."
	curl -fsSL https://ollama.com/install.sh | sh
}

tags_for_platform() {
	local tag
	for tag in "${ALL_TAGS[@]}"; do
		case "$tag" in
			*mlx*)
				if [[ "$(uname -s)" == "Darwin" ]]; then
					echo "$tag"
				fi
				;;
			*)
				echo "$tag"
				;;
		esac
	done
}

ensure_ollama_serve() {
	if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
		_log "ollama serve already running on :11434"
		return
	fi

	_log "Starting ollama serve in background..."
	nohup ollama serve >"$REPO_ROOT/.ollama-serve.log" 2>&1 &
	local pid=$!

	for _ in $(seq 1 60); do
		if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
			_log "ollama serve ready (pid=$pid, log=$REPO_ROOT/.ollama-serve.log)"
			return
		fi
		sleep 1
	done

	echo "Timed out waiting for ollama serve" >&2
	exit 1
}

pull_all_tags() {
	local tags=()
	while IFS= read -r tag; do
		[[ -n "$tag" ]] && tags+=("$tag")
	done < <(tags_for_platform)

	_log "Pulling ${#tags[@]} Qwen3.8-27B tags from Ollama library..."
	for tag in "${tags[@]}"; do
		_log "pull $tag"
		ollama pull "$tag"
	done

	_log "Installed models:"
	ollama list | rg 'qwen3\.8' || ollama list
}

print_usage() {
	cat <<EOF

Done. Local Qwen3.8-27B models are ready via Ollama.

Quick test:
  ollama run qwen3.8:27b-q4_K_M "hello"

Browser Use (recommended daily driver on 16-24GB VRAM):
  from browser_use import Agent, ChatOllama
  llm = ChatOllama(model="qwen3.8:27b-q4_K_M", num_ctx=32000)
  Agent("your task", llm=llm).run_sync()

Tags pulled on this machine:
$(tags_for_platform | sed 's/^/  - /')

Original weights (BF16, not required for Ollama): https://huggingface.co/Qwen/Qwen3.8-27B
EOF
}

ensure_ollama

if $DO_SERVE; then
	ensure_ollama_serve
fi

if $DO_PULL; then
	pull_all_tags
fi

print_usage
