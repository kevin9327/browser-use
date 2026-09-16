#!/usr/bin/env bash
# @file purpose: Single-tier Qwen3.8-27B local setup for browser-use (no overlapping quants).
#
# Each tier is mutually exclusive — only one Ollama tag is pulled/kept per machine.
# MTP variants supersede plain quants (same backbone + draft head = faster inference).
#
# Usage:
#   ./bin/start-qwen38-local.sh detect          # tier + num_ctx recommendation
#   ./bin/start-qwen38-local.sh setup [tier]    # serve + pull + prune
#   ./bin/start-qwen38-local.sh verify [tier]   # health + optional inference smoke test
#   ./bin/start-qwen38-local.sh status          # one-line readiness summary
#   ./bin/start-qwen38-local.sh serve           # start ollama with perf env
#   ./bin/start-qwen38-local.sh pull [tier]     # pull one tier (default: auto)
#   ./bin/start-qwen38-local.sh prune           # remove superseded / duplicate tags

set -o errexit
set -o nounset
set -o pipefail

TIER_BALANCED='balanced'
TIER_QUALITY='quality'
TIER_MAX='max'

MODEL_BALANCED='qwen3.8:27b-mtp-q4_K_M'
MODEL_QUALITY='qwen3.8:27b-mtp-q8_0'
MODEL_MAX='qwen3.8:27b-mtp-bf16'

SUPERSEDED_MODELS=(
	'qwen3.8:27b'
	'qwen3.8:27b-q4_K_M'
	'qwen3.8:27b-q8_0'
	'qwen3.8:27b-bf16'
)

# Minimum memory (GB) to run inference for each tier (weights + KV headroom).
TIER_MIN_INFER_GB_BALANCED=20
TIER_MIN_INFER_GB_QUALITY=32
TIER_MIN_INFER_GB_MAX=50

# Minimum memory (GB) to recommend pulling each tier.
TIER_VRAM_GB_BALANCED=20
TIER_VRAM_GB_QUALITY=30
TIER_VRAM_GB_MAX=46

OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_URL="http://${OLLAMA_HOST}"

export PATH="${HOME}/.local/bin:${PATH}"

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
	awk '/MemAvailable/ {printf "%.0f\n", $2/1024/1024; exit}' /proc/meminfo
}

_min_infer_gb_for_tier() {
	case "$1" in
		"${TIER_BALANCED}") echo "${TIER_MIN_INFER_GB_BALANCED}" ;;
		"${TIER_QUALITY}") echo "${TIER_MIN_INFER_GB_QUALITY}" ;;
		"${TIER_MAX}") echo "${TIER_MIN_INFER_GB_MAX}" ;;
		*) return 1 ;;
	esac
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
		_log "WARN: ~${vram_gb}GB available — 27B inference needs ~${TIER_MIN_INFER_GB_BALANCED}GB+ (pull OK, run on bigger GPU/RAM)."
	fi
}

_recommend_num_ctx() {
	local tier="$1"
	local vram_gb="$(_detect_vram_gb)"
	local min_gb
	min_gb="$(_min_infer_gb_for_tier "${tier}")"

	if [[ "${vram_gb}" -lt "${min_gb}" ]]; then
		echo 2048
	elif [[ "${tier}" == "${TIER_MAX}" && "${vram_gb}" -ge 64 ]]; then
		echo 32768
	elif [[ "${vram_gb}" -ge 32 ]]; then
		echo 8192
	else
		echo 4096
	fi
}

_can_run_inference() {
	local tier="$1"
	local vram_gb="$(_detect_vram_gb)"
	local min_gb
	min_gb="$(_min_infer_gb_for_tier "${tier}")"
	[[ "${vram_gb}" -ge "${min_gb}" ]]
}

_ensure_ollama() {
	command -v ollama >/dev/null 2>&1 || {
		_log 'Install Ollama: https://ollama.com/download'
		exit 1
	}
}

_apply_perf_env() {
	export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
	export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
	export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
	export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
}

_wait_ollama() {
	local attempts="${1:-30}"
	for _ in $(seq 1 "${attempts}"); do
		curl -fsS "${OLLAMA_URL}/" >/dev/null 2>&1 && return 0
		sleep 1
	done
	return 1
}

_start_serve() {
	_apply_perf_env
	if curl -fsS "${OLLAMA_URL}/" >/dev/null 2>&1; then
		_log "Ollama already running at ${OLLAMA_URL}"
		return
	fi
	_log 'Starting ollama serve (flash-attn=1, parallel=1)...'
	nohup ollama serve >"${HOME}/.ollama/serve.log" 2>&1 &
	if ! _wait_ollama 30; then
		_log 'Timed out waiting for Ollama. See ~/.ollama/serve.log'
		exit 1
	fi
	_log 'Ollama ready.'
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
	local attempt
	for attempt in 1 2 3; do
		if ollama pull "${model}"; then
			break
		fi
		_log "Pull failed (attempt ${attempt}/3), retrying in 5s..."
		sleep 5
		[[ "${attempt}" -eq 3 ]] && exit 1
	done

	_prune_superseded
	_prune_other_tiers "${model}"
	_log "Ready. Only ${model} is kept — no overlapping quants."
	ollama list
}

_smoke_inference() {
	local model="$1"
	local num_ctx="$2"
	local payload
	payload=$(printf '{"model":"%s","prompt":"Reply with exactly: OK","stream":false,"options":{"num_predict":4,"num_ctx":%s,"think":false}}' "${model}" "${num_ctx}")

	local response http_code body
	response=$(curl -sS -w '\n%{http_code}' --max-time 120 -X POST "${OLLAMA_URL}/api/generate" -d "${payload}" 2>&1) || {
		_log "Inference request failed (timeout or connection error)."
		return 1
	}
	http_code=$(printf '%s' "${response}" | tail -1)
	body=$(printf '%s' "${response}" | sed '$d')

	if [[ "${http_code}" != "200" ]]; then
		_log "Inference HTTP ${http_code}: ${body}"
		return 1
	fi
	if printf '%s' "${body}" | grep -q '"error"'; then
		_log "Inference error: ${body}"
		return 1
	fi
	_log 'Inference smoke test passed.'
	return 0
}

_cmd_verify() {
	local tier="${1:-$(_recommend_tier)}"
	local model num_ctx
	model="$(_model_for_tier "${tier}")"
	num_ctx="$(_recommend_num_ctx "${tier}")"

	_ensure_ollama
	_start_serve

	local ok=0
	if curl -fsS "${OLLAMA_URL}/" >/dev/null 2>&1; then
		_log "OK: Ollama reachable at ${OLLAMA_URL}"
	else
		_log 'FAIL: Ollama not reachable'
		ok=1
	fi

	if _model_installed "${model}"; then
		_log "OK: Model installed (${model})"
	else
		_log "FAIL: Model missing (${model}) — run: $0 pull ${tier}"
		ok=1
	fi

	if [[ "${ok}" -ne 0 ]]; then
		exit 1
	fi

	if _can_run_inference "${tier}"; then
		_log "Running inference smoke test (num_ctx=${num_ctx})..."
		if _smoke_inference "${model}" "${num_ctx}"; then
			_log 'Running ChatOllama Python smoke test...'
			if QWEN38_TIER="${tier}" QWEN38_NUM_CTX="${num_ctx}" uv run python "${BASH_SOURCE%/*}/test-qwen38-chat.py"; then
				_log 'VERIFY: all checks passed (curl + ChatOllama).'
				exit 0
			fi
			_log 'FAIL: ChatOllama smoke test failed'
			exit 1
		fi
		_log 'FAIL: inference smoke test failed (OOM? check dmesg / free -h)'
		exit 1
	fi

	_log "SKIP: inference (~$(_detect_vram_gb)GB available, need ~$(_min_infer_gb_for_tier "${tier}")GB+)"
	_log 'VERIFY: install OK; run inference on a machine with enough VRAM/RAM.'
	exit 0
}

_cmd_detect() {
	local tier model vram num_ctx infer_ready
	vram="$(_detect_vram_gb)"
	tier="$(_recommend_tier)"
	model="$(_model_for_tier "${tier}")"
	num_ctx="$(_recommend_num_ctx "${tier}")"
	if _can_run_inference "${tier}"; then infer_ready=true; else infer_ready=false; fi
	cat <<EOF
detected_vram_gb=${vram}
recommended_tier=${tier}
recommended_model=${model}
recommended_num_ctx=${num_ctx}
inference_ready=${infer_ready}
browser_use_snippet:
  from browser_use import Agent, ChatOllama
  llm = ChatOllama(
      model='${model}',
      ollama_options={'num_ctx': ${num_ctx}, 'temperature': 0.1, 'think': False},
  )
  Agent('your task', llm=llm).run_sync()
EOF
}

_cmd_status() {
	local tier model
	tier="$(_recommend_tier)"
	model="$(_model_for_tier "${tier}")"
	printf 'ollama=%s model=%s installed=%s inference_ready=%s num_ctx=%s\n' \
		"$(curl -fsS "${OLLAMA_URL}/" >/dev/null 2>&1 && echo up || echo down)" \
		"${model}" \
		"$(_model_installed "${model}" && echo yes || echo no)" \
		"$(_can_run_inference "${tier}" && echo yes || echo no)" \
		"$(_recommend_num_ctx "${tier}")"
}

_mode="${1:-setup}"
shift || true

case "${_mode}" in
	detect) _cmd_detect ;;
	serve) _ensure_ollama; _start_serve ;;
	pull) _pull_tier "${1:-}" ;;
	verify) _cmd_verify "${1:-}" ;;
	status) _cmd_status ;;
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
		echo "Usage: $0 {detect|setup [tier]|verify [tier]|status|serve|pull [tier]|prune}"
		echo "Tiers: balanced (~24GB) | quality (~32GB) | max (~48GB+)"
		exit 1
		;;
esac
