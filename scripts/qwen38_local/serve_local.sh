#!/usr/bin/env bash
# Serve a local Qwen3.8-27B GGUF on OpenAI-compatible :8080 for browser-use.
#
# Prefers Q4_K_M, then the largest remaining quant that exists. Attach mmproj
# and native MTP draft when those files are present.
#
#   ./scripts/qwen38_local/serve_local.sh
#   QUANT=Q5_K_M NGL=99 CTX=8192 ./scripts/qwen38_local/serve_local.sh
#
# Then:
#   from browser_use import Agent
#   from browser_use.llm import ChatOpenAI
#   llm = ChatOpenAI(model='Qwen3.8-27B', base_url='http://127.0.0.1:8080/v1', api_key='local')

set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen38}"
GGUF_DIR="${GGUF_DIR:-$WORK_DIR/gguf}"
SRC_DIR="${SRC_DIR:-$WORK_DIR/src/llama.cpp}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
NGL="${NGL:-99}"
QUANT="${QUANT:-}"

pick_quant() {
	local preferred=(
		Q4_K_M
		Q5_K_M
		Q4_K_S
		Q6_K
		Q8_0
		Q3_K_M
		IQ4_XS
		Q4_0
		Q2_K
	)
	local q path
	if [[ -n "$QUANT" ]]; then
		path="$GGUF_DIR/Qwen3.8-27B-${QUANT}.gguf"
		if [[ ! -f "$path" ]]; then
			echo "missing $path" >&2
			exit 1
		fi
		echo "$path"
		return
	fi
	for q in "${preferred[@]}"; do
		path="$GGUF_DIR/Qwen3.8-27B-${q}.gguf"
		if [[ -f "$path" ]]; then
			echo "$path"
			return
		fi
	done
	echo "no quantized GGUF in $GGUF_DIR" >&2
	ls -lh "$GGUF_DIR" >&2 || true
	exit 1
}

MODEL="$(pick_quant)"
ARGS=(
	-m "$MODEL"
	--host "$HOST"
	--port "$PORT"
	-c "$CTX"
	-ngl "$NGL"
	--jinja
)

shopt -s nullglob
mmproj_files=("$GGUF_DIR"/mmproj-Qwen3.8-27B-*.gguf)
if (( ${#mmproj_files[@]} > 0 )); then
	ARGS+=(--mmproj "${mmproj_files[0]}")
fi
mtp_files=()
if [[ -f "$GGUF_DIR/mtp-Qwen3.8-27B-Q4_K_M.gguf" ]]; then
	mtp_files+=("$GGUF_DIR/mtp-Qwen3.8-27B-Q4_K_M.gguf")
elif [[ -f "$GGUF_DIR/mtp-Qwen3.8-27B-Q8_0.gguf" ]]; then
	mtp_files+=("$GGUF_DIR/mtp-Qwen3.8-27B-Q8_0.gguf")
fi
if (( ${#mtp_files[@]} > 0 )); then
	ARGS+=(--spec-type draft-mtp --spec-draft-model "${mtp_files[0]}")
fi

echo "serving $MODEL on http://${HOST}:${PORT}/v1"
exec "$SRC_DIR/build/bin/llama-server" "${ARGS[@]}"
