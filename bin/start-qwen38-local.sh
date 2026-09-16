#!/usr/bin/env bash
# @file purpose: Single-tier Qwen3.8-27B local setup for browser-use (no overlapping quants).
#
# Each tier is mutually exclusive — only one Ollama tag is pulled/kept per machine.
# MTP variants supersede plain quants (same backbone + draft head = faster inference).
#
# Usage:
#   ./bin/start-qwen38-local.sh detect          # print recommended tier + model
#   ./bin/start-qwen38-local.sh pull [tier]     # pull one tier (default: auto)
#   ./bin/start-qwen38-local.sh serve           # start ollama with perf env
#   ./bin/start-qwen38-local.sh prune           # remove superseded / duplicate tags
#   ./bin/start-qwen38-local.sh setup           # serve + pull recommended tier

set -o errexit
set -o nounset
set -o pipefail

# --- tiers (newest/highest tier wins when pruning) ---
TIER_BALANCED='balanced'   # 24 GB GPU — best speed/quality tradeoff
TIER_QUALITY='quality'     # 32 GB+ GPU
TIER_MAX='max'             # 48 GB+ VRAM

MODEL_BALANCED='qwen3.8:27b-mtp-q4_K_M'
MODEL_QUALITY='qwen3.8:27b-mtp-q8_0'
MODEL_MAX='qwen3.8:27b-mtp-bf16'

# Plain tags superseded by MTP (never pull these when MTP exists).
SUPERSEDED_MODELS=(
	'qwen3.8:27b'
	'qwen3.8:27b-q4_K_M'
	'qwen3.8:27b-q8_0'
	'qwen3.8:27b-bf16'
)

TIER_VRAM_GB_BALANCED=20
TIER_VRAM_GB_QUALITY=30
TIER_VRAM_GB_MAX=46

OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

_log() { printf '[qwen38] %s\n' "$*" >&2; }

_model_for_tier() {
	case "$1" in
		"${TIER_BALANCED}") echo "${MODEL_BALANCED}" ;;
		"${TIER_QUALITY}") echo "${MODEL_QUALITY}" ;;
		"${TIER_MAX}") echo "${MODEL_MAX}" ;;
		*) return 1 ;;
	esac
}

_all_tier_models() {
	printf '%s\n' "${MODEL_BALANCED}" "${MODEL_QUALITY}" "${MODEL_MAX}"
}

_detect_vram_gb() {
	if command -v nvidia-smi >/dev/null 2>&1; then
		nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.0f\n", $1/1024}'
		return
	fi
	awk '/MemTotal/ {printf "%.0f\n", $2/1024/1024}' /proc/meminfo
}

_recommend_tier() {
	local vram_gb
	vram_gb="$(_detect_vram_gb)"
	if [[ "${vram_gb}" -ge "${TIER_VRAM_GB_MAX}" ]]; then
		echo "${TIER_MAX}"
	elif [[ "${vram_gb}" -ge "${TIER_VRAM_GB_QUALITY}" ]]; then
		echo "${TIER_QUALITY}"
	elif [[ "${vram_gb}" -ge "${TIER_VRAM_GB_BALANCED}" ]]; then
		echo "${TIER_BALANCED}"
	else
		echo "${TIER_BALANCED}"
		_log "WARN: detected ~${vram_gb}GB — 27B needs ~20GB+; expect CPU offload / slow runs."
	fi
}

_ensure_ollama() {
	command -v ollama >/dev/null 2>&1 || {
		_log 'Install Ollama: https://ollama.com/download'
		exit 1
	}
}

_apply_perf_env() {
	# Single-stream agent workloads: one parallel slot, flash attention, modest KV cache.
	export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
	export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
	export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
	export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
}

_start_serve() {
	_apply_perf_env
	if curl -fsS "http://${OLLAMA_HOST}/" >/dev/null 2>&1; then
		_log "Ollama already running at http://${OLLAMA_HOST}"
		return
	fi
	_log "Starting ollama serve (flash-attn=1, parallel=1)..."
	nohup ollama serve >"${HOME}/.ollama/serve.log" 2>&1 &
	for _ in $(seq 1 30); do
		curl -fsS "http://${OLLAMA_HOST}/" >/dev/null 2>&1 && {
			_log 'Ollama ready.'
			return
		}
		sleep 1
	done
	_log 'Timed out waiting for Ollama.'
	exit 1
}

_model_installed() {
	ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$1" >/dev/null 2>&1
}

_prune_superseded() {
	local model
	for model in "${SUPERSEDED_MODELS[@]}"; do
		if _model_installed "${model}"; then
			_log "Removing superseded tag: ${model}"
			ollama rm "${model}" || true
		fi
	done
}

_prune_other_tiers() {
	local keep_model="$1"
	local model
	while IFS= read -r model; do
		[[ "${model}" == "${keep_model}" ]] && continue
		if _model_installed "${model}"; then
			_log "Removing other tier (keep ${keep_model}): ${model}"
			ollama rm "${model}" || true
		fi
	done < <(_all_tier_models)
}

_pull_tier() {
	local tier="${1:-$(_recommend_tier)}"
	local model
	model="$(_model_for_tier "${tier}")" || {
		_log "Unknown tier: ${tier} (use: balanced | quality | max)"
		exit 1
	}

	_ensure_ollama
	_start_serve

	_log "Tier=${tier} model=${model} (~$(
		case "${tier}" in
			balanced) echo '18GB' ;;
			quality) echo '30GB' ;;
			max) echo '56GB' ;;
		esac
	))"
	ollama pull "${model}"
	_prune_superseded
	_prune_other_tiers "${model}"
	_log "Ready. Only ${model} is kept — no overlapping quants."
	ollama list
}

_cmd_detect() {
	local tier model vram
	vram="$(_detect_vram_gb)"
	tier="$(_recommend_tier)"
	model="$(_model_for_tier "${tier}")"
	cat <<EOF
detected_vram_gb=${vram}
recommended_tier=${tier}
recommended_model=${model}
browser_use_snippet:
  from browser_use import Agent, ChatOllama
  llm = ChatOllama(
      model='${model}',
      ollama_options={'num_ctx': 8192, 'temperature': 0.1, 'think': False},
  )
  Agent('your task', llm=llm).run_sync()
EOF
}

_mode="${1:-setup}"
shift || true

case "${_mode}" in
	detect) _cmd_detect ;;
	serve) _ensure_ollama; _start_serve ;;
	pull) _pull_tier "${1:-}" ;;
	prune)
		_ensure_ollama
		tier="$(_recommend_tier)"
		model="$(_model_for_tier "${tier}")"
		_prune_superseded
		if _model_installed "${model}"; then
			_prune_other_tiers "${model}"
		fi
		ollama list
		;;
	setup)
		_pull_tier "${1:-}"
		_cmd_detect
		;;
	*)
		echo "Usage: $0 {detect|serve|pull [tier]|prune|setup [tier]}"
		echo "Tiers: balanced (~24GB) | quality (~32GB) | max (~48GB+)"
		exit 1
		;;
esac
