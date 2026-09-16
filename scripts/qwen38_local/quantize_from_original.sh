#!/usr/bin/env bash
# Quantize official Qwen/Qwen3.8-27B (latest main) into local GGUF files.
#
# Downloads the original Hugging Face weights, converts them with a current
# llama.cpp (qwen35 arch + --no-mtp / --mmproj / --mtp split), then emits every
# practical GGUF quant. IQ/TQ presets wait for an importance matrix.
#
# Usage:
#   ./scripts/qwen38_local/quantize_from_original.sh
#   WORK_DIR=/data/qwen38 ./scripts/qwen38_local/quantize_from_original.sh
#
# Outputs land in $WORK_DIR/gguf/. This VM has ~230G disk and 15G RAM, so the
# script deletes the HF snapshot after a successful BF16 convert, skips alias
# types, and stops a quant if free disk drops below MIN_FREE_GB.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/qwen38}"
HF_DIR="${HF_DIR:-$WORK_DIR/hf}"
GGUF_DIR="${GGUF_DIR:-$WORK_DIR/gguf}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
SRC_DIR="${SRC_DIR:-$WORK_DIR/src/llama.cpp}"
VENV="${VENV:-$WORK_DIR/.venv}"
HF_REPO="${HF_REPO:-Qwen/Qwen3.8-27B}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
NPROC_COUNT="$(nproc)"
export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC_COUNT}"
export CMAKE_BUILD_PARALLEL_LEVEL="${NPROC_COUNT}"

mkdir -p "$HF_DIR" "$GGUF_DIR" "$LOG_DIR" "$(dirname "$SRC_DIR")"

log() {
	printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/pipeline.log"
}

free_gb() {
	df -BG --output=avail "$WORK_DIR" | tail -n1 | tr -dc '0-9'
}

need_disk() {
	local need="$1"
	local have
	have="$(free_gb)"
	if (( have < need + MIN_FREE_GB )); then
		log "skip: need ${need}G + ${MIN_FREE_GB}G free, have ${have}G"
		return 1
	fi
	return 0
}

ensure_uv() {
	if ! command -v uv >/dev/null 2>&1; then
		curl -LsSf https://astral.sh/uv/install.sh | sh
	fi
}

ensure_venv() {
	ensure_uv
	if [[ ! -x "$VENV/bin/python" ]]; then
		uv venv "$VENV" --python 3.12
	fi
	# shellcheck disable=SC1091
	source "$VENV/bin/activate"
	uv pip install -U huggingface_hub hf_xet
}

ensure_llama_cpp() {
	if [[ ! -d "$SRC_DIR/.git" ]]; then
		git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
	else
		git -C "$SRC_DIR" fetch --depth 1 origin HEAD && git -C "$SRC_DIR" reset --hard FETCH_HEAD || true
	fi
	if [[ ! -x "$SRC_DIR/build/bin/llama-quantize" ]]; then
		rm -f "$SRC_DIR/build/CMakeCache.txt"
		cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_C_COMPILER=gcc \
			-DCMAKE_CXX_COMPILER=g++ \
			-DGGML_NATIVE=ON \
			-DLLAMA_BUILD_TESTS=OFF \
			-DLLAMA_CURL=ON
		cmake --build "$SRC_DIR/build" -j"$NPROC_COUNT" \
			--target llama-quantize llama-imatrix llama-server llama-cli llama-gguf-split
	fi
	# Converter deps pin an older huggingface_hub; install them first so a
	# background download cannot race a wheel uninstall.
	uv pip install -r "$SRC_DIR/requirements/requirements-convert_hf_to_gguf.txt"
	uv pip install -e "$SRC_DIR/gguf-py"
	uv pip install -U hf_xet
}

download_original() {
	log "download original ${HF_REPO} -> ${HF_DIR}"
	python - <<PY
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
repo = "$HF_REPO"
dest = "$HF_DIR"
snapshot_download(repo_id=repo, local_dir=dest)
info = HfApi().repo_info(repo)
text = f"repo={info.id}\nsha={info.sha}\nlast_modified={info.last_modified}\n"
Path("$LOG_DIR/revision.txt").write_text(text, encoding="utf-8")
print(text, end="")
PY
}

convert_one() {
	local extra_args="$1"
	local outfile="$2"
	if [[ -f "$outfile" ]]; then
		log "exists, skip convert: $outfile"
		return 0
	fi
	need_disk 56 || return 1
	log "convert ${extra_args} -> ${outfile}"
	python "$SRC_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
		--outtype bf16 \
		--use-temp-file \
		--outfile "$outfile" \
		${extra_args}
}

convert_all() {
	local outfile="$GGUF_DIR/Qwen3.8-27B-BF16.gguf"
	convert_one "--no-mtp --model-name Qwen3.8-27B" "$outfile" || log "local BF16 convert returned $?"
	if [[ ! -f "$outfile" ]]; then
		log "local BF16 convert failed; fetching lossless BF16 GGUF converted from the same original"
		mkdir -p "$WORK_DIR/bf16_fallback"
		python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="unsloth/Qwen3.8-27B-GGUF", local_dir="$WORK_DIR/bf16_fallback", allow_patterns=["BF16/*"])
PY
		shopt -s nullglob
		local splits=("$WORK_DIR"/bf16_fallback/BF16/*.gguf)
		if (( ${#splits[@]} == 1 )); then
			mv "${splits[0]}" "$outfile"
		elif (( ${#splits[@]} > 1 )); then
			"$SRC_DIR/build/bin/llama-gguf-split" --merge "${splits[0]}" "$outfile"
		fi
		shopt -u nullglob
		rm -rf "$WORK_DIR/bf16_fallback"
	fi
	if [[ ! -f "$outfile" ]]; then
		log "ERROR: no BF16 GGUF after convert + fallback"
		return 1
	fi
	local bf16_gb
	bf16_gb="$(du -BG "$outfile" | awk '{print $1}' | tr -dc '0-9')"
	if (( bf16_gb < 40 )); then
		log "ERROR: $outfile is only ${bf16_gb}G (expected ~54G). Refusing to quantize a truncated/MTP file."
		return 1
	fi
	# llama.cpp only auto-prefixes mmproj-/mtp- when --outfile is a DIRECTORY.
	# Always pass the final filename so a second convert cannot clobber the backbone.
	if [[ ! -f "$GGUF_DIR/mmproj-Qwen3.8-27B-F16.gguf" && ! -f "$GGUF_DIR/mmproj-Qwen3.8-27B-f16.gguf" ]]; then
		need_disk 2 || true
		log "convert mmproj"
		python "$SRC_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
			--outtype f16 \
			--mmproj \
			--use-temp-file \
			--outfile "$GGUF_DIR/mmproj-Qwen3.8-27B-F16.gguf" || log "mmproj convert failed"
	fi
	if [[ ! -f "$GGUF_DIR/mtp-Qwen3.8-27B-BF16.gguf" && ! -f "$GGUF_DIR/mtp-Qwen3.8-27B-bf16.gguf" ]]; then
		need_disk 8 || true
		log "convert mtp"
		python "$SRC_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
			--outtype bf16 \
			--mtp \
			--use-temp-file \
			--outfile "$GGUF_DIR/mtp-Qwen3.8-27B-BF16.gguf" || log "mtp convert failed"
	fi
}

maybe_delete_hf() {
	local bf16="$GGUF_DIR/Qwen3.8-27B-BF16.gguf"
	if [[ -f "$bf16" ]]; then
		local gb
		gb="$(du -BG "$bf16" | awk '{print $1}' | tr -dc '0-9')"
		if (( gb < 40 )); then
			log "refusing to delete HF snapshot; $bf16 is only ${gb}G"
			return 1
		fi
		log "BF16 GGUF ready (${gb}G); removing HF snapshot to free disk"
		rm -rf "$HF_DIR"
		mkdir -p "$HF_DIR"
	fi
}

quantize_one() {
	local src="$1"
	local dst="$2"
	local ftype="$3"
	shift 3
	if [[ -f "$dst" ]]; then
		log "exists, skip quant: $dst"
		return 0
	fi
	if [[ ! -f "$src" ]]; then
		log "missing source, skip: $src"
		return 1
	fi
	local src_gb
	src_gb="$(du -BG "$src" | awk '{print $1}' | tr -dc '0-9')"
	# Worst case output ~= source size (Q8/F16). Require that plus MIN_FREE.
	need_disk "$src_gb" || return 1
	log "quantize $* $ftype -> $dst"
	"$SRC_DIR/build/bin/llama-quantize" "$@" "$src" "$dst" "$ftype"
}

# K-quants and legacy presets that do not require an importance matrix.
K_QUANTS=(
	Q4_K_M
	Q5_K_M
	Q8_0
	Q6_K
	Q3_K_M
	Q4_K_S
	Q5_K_S
	Q3_K_S
	Q2_K
	Q4_0
	Q4_1
	Q5_0
	Q5_1
	Q3_K_L
	Q2_K_S
	Q2_0
	Q1_0
)

# Higher-quality "L" mixes: keep embeddings/output at Q8_0.
L_QUANTS=(
	Q4_K_M
	Q5_K_M
	Q6_K
	Q3_K_M
)

IQ_QUANTS=(
	IQ4_XS
	IQ4_NL
	IQ3_M
	IQ3_S
	IQ3_XS
	IQ3_XXS
	IQ2_M
	IQ2_S
	IQ2_XS
	IQ2_XXS
	IQ1_M
	IQ1_S
	TQ2_0
	TQ1_0
)

quantize_k() {
	local src="$GGUF_DIR/Qwen3.8-27B-BF16.gguf"
	local q
	for q in "${K_QUANTS[@]}"; do
		quantize_one "$src" "$GGUF_DIR/Qwen3.8-27B-${q}.gguf" "$q" || true
	done
	for q in "${L_QUANTS[@]}"; do
		quantize_one "$src" "$GGUF_DIR/Qwen3.8-27B-${q}_L.gguf" "$q" \
			--token-embedding-type Q8_0 --output-tensor-type Q8_0 || true
	done
	if [[ -f "$GGUF_DIR/mtp-Qwen3.8-27B-BF16.gguf" ]]; then
		quantize_one "$GGUF_DIR/mtp-Qwen3.8-27B-BF16.gguf" \
			"$GGUF_DIR/mtp-Qwen3.8-27B-Q8_0.gguf" Q8_0 || true
		quantize_one "$GGUF_DIR/mtp-Qwen3.8-27B-BF16.gguf" \
			"$GGUF_DIR/mtp-Qwen3.8-27B-Q4_K_M.gguf" Q4_K_M || true
	fi
}

prepare_calib() {
	local calib="$WORK_DIR/calib.txt"
	if [[ -f "$calib" ]]; then
		return 0
	fi
	log "write small calibration corpus for imatrix"
	python - <<'PY' "$calib"
from pathlib import Path
import sys
out = Path(sys.argv[1])
# Compact, mixed-domain text so imatrix has something to work with on CPU.
passages = [
	"The importance matrix estimates per-row activation scale so low-bit GGUF quants keep outlier channels.",
	"Qwen3.8-27B is a dense hybrid Gated DeltaNet vision-language model with native 262144 context.",
	"def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
	"SELECT user_id, COUNT(*) FROM events WHERE ts >= '2026-01-01' GROUP BY user_id HAVING COUNT(*) > 10;",
	"브라우저 에이전트는 DOM 스냅샷을 보고 click, type, navigate 액션을 고른다. tool calling JSON은 스키마를 지켜야 한다.",
	"Speculative decoding with a native MTP draft predicts several tokens and verifies them on the target model.",
	"Quantization maps BF16 weights to 2-8 bit codes. K-quants mix super-blocks; IQ quants need an imatrix.",
	"HTTP/1.1 200 OK\ncontent-type: application/json\n\n{\"ok\": true, \"items\": [1, 2, 3]}",
]
out.write_text("\n\n".join(passages * 40) + "\n", encoding="utf-8")
print(out, "bytes", out.stat().st_size)
PY
}

run_imatrix() {
	local src="$GGUF_DIR/Qwen3.8-27B-BF16.gguf"
	local imatrix="$GGUF_DIR/Qwen3.8-27B-imatrix.dat"
	if [[ -f "$imatrix" ]]; then
		log "exists, skip imatrix"
		return 0
	fi
	if [[ ! -f "$src" ]]; then
		return 1
	fi
	prepare_calib
	log "imatrix on CPU (slow; 15G RAM, mmap weights)"
	# Tiny context so the 27B graph fits in system RAM. More chunks = better IQ.
	"$SRC_DIR/build/bin/llama-imatrix" \
		-m "$src" \
		-f "$WORK_DIR/calib.txt" \
		-o "$imatrix" \
		--chunk 256 \
		-c 256 \
		-ngl 0 \
		--n-chunks "${IMATRIX_CHUNKS:-64}" \
		-b 64 || log "imatrix failed; IQ quants will be skipped"
}

quantize_iq() {
	local src="$GGUF_DIR/Qwen3.8-27B-BF16.gguf"
	local imatrix="$GGUF_DIR/Qwen3.8-27B-imatrix.dat"
	if [[ ! -f "$imatrix" ]]; then
		log "no imatrix, skip IQ/TQ"
		return 0
	fi
	local q
	for q in "${IQ_QUANTS[@]}"; do
		quantize_one "$src" "$GGUF_DIR/Qwen3.8-27B-${q}.gguf" "$q" --imatrix "$imatrix" || true
	done
}

write_manifest() {
	log "manifest"
	{
		echo "# Qwen3.8-27B local GGUF"
		echo "source=${HF_REPO}"
		echo "llama.cpp=$(git -C "$SRC_DIR" rev-parse --short HEAD)"
		echo "generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo
		(cd "$GGUF_DIR" && du -h --apparent-size *.gguf *.dat 2>/dev/null | sort -h) || true
	} | tee "$GGUF_DIR/MANIFEST.txt"
}

serve_hint() {
	cat <<EOF | tee "$GGUF_DIR/SERVE.txt"
# Pick the biggest quant that fits VRAM (weights + KV). Then:
$SRC_DIR/build/bin/llama-server \\
  -m $GGUF_DIR/Qwen3.8-27B-Q4_K_M.gguf \\
  --mmproj $GGUF_DIR/mmproj-Qwen3.8-27B-F16.gguf \\
  --spec-type draft-mtp \\
  --spec-draft-model $GGUF_DIR/mtp-Qwen3.8-27B-Q4_K_M.gguf \\
  --port 8080 --host 127.0.0.1 -c 16384 -ngl 99 --jinja

# browser-use against that server:
#   ChatOpenAI(model='Qwen3.8-27B', base_url='http://127.0.0.1:8080/v1', api_key='local')
EOF
}

main() {
	log "work_dir=$WORK_DIR free=$(free_gb)G"
	ensure_venv
	ensure_llama_cpp
	download_original
	convert_all
	maybe_delete_hf
	quantize_k
	run_imatrix
	quantize_iq
	write_manifest
	serve_hint
	log "done"
}

main "$@"
