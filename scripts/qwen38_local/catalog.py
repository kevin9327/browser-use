"""Catalog of official Qwen3.8-27B weights and every quantized variant we start locally.

# @file purpose: Size/quality catalog for Qwen3.8-27B original, Unsloth GGUF, and Ollama tags
"""

from .views import DownloadTarget

OFFICIAL_REPO = 'Qwen/Qwen3.8-27B'
UNSLOTH_REPO = 'unsloth/Qwen3.8-27B-GGUF'
LLAMA_CPP_RELEASE = 'b10964'
LLAMA_CPP_MIN_RELEASE = 'b10419'

# Sum of the 18 official safetensor shards on Qwen/Qwen3.8-27B (Hugging Face, Aug 2026).
ORIGINAL_SAFETENSOR_BYTES = (
	3_966_730_552
	+ 3_043_080_328
	+ 2_542_796_952
	+ 3_988_973_152
	+ 2_099_339_864
	+ 3_979_553_696
	+ 2_108_759_344
	+ 3_979_553_696
	+ 2_108_759_344
	+ 3_979_553_696
	+ 2_108_759_344
	+ 3_979_553_696
	+ 2_108_759_344
	+ 3_979_553_696
	+ 2_108_759_344
	+ 3_979_564_040
	+ 2_108_759_344
	+ 3_392_197_344
)

# llama-quantize presets that do not require an importance matrix.
LLAMA_QUANT_TYPES: tuple[str, ...] = (
	'Q2_K',
	'Q3_K_S',
	'Q3_K_M',
	'Q3_K_L',
	'Q4_0',
	'Q4_1',
	'Q4_K_S',
	'Q4_K_M',
	'Q5_0',
	'Q5_1',
	'Q5_K_S',
	'Q5_K_M',
	'Q6_K',
	'Q8_0',
)

# Rough output sizes when quantizing the official 27B BF16 GGUF with llama-quantize.
# Used only for disk planning; actual files replace these after quantization.
SELF_QUANT_SIZE_ESTIMATES: dict[str, int] = {
	'Q2_K': 10_800_000_000,
	'Q3_K_S': 12_700_000_000,
	'Q3_K_M': 13_400_000_000,
	'Q3_K_L': 14_100_000_000,
	'Q4_0': 16_350_000_000,
	'Q4_1': 17_540_000_000,
	'Q4_K_S': 16_360_000_000,
	'Q4_K_M': 17_770_000_000,
	'Q5_0': 18_900_000_000,
	'Q5_1': 20_500_000_000,
	'Q5_K_S': 19_570_000_000,
	'Q5_K_M': 20_920_000_000,
	'Q6_K': 23_460_000_000,
	'Q8_0': 29_120_000_000,
}

UNSLOTH_GGUFS: tuple[DownloadTarget, ...] = (
	DownloadTarget(
		key='ud-iq1-s',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ1_S',
		size_bytes=6_192_222_208,
		filename='Qwen3.8-27B-UD-IQ1_S.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ1_S',
		quality_rank=10,
		notes='Smallest Dynamic V3 quant. Last resort for 12GB machines.',
	),
	DownloadTarget(
		key='ud-iq1-m',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ1_M',
		size_bytes=6_729_166_848,
		filename='Qwen3.8-27B-UD-IQ1_M.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ1_M',
		quality_rank=12,
	),
	DownloadTarget(
		key='ud-iq2-xxs',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ2_XXS',
		size_bytes=7_266_070_528,
		filename='Qwen3.8-27B-UD-IQ2_XXS.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ2_XXS',
		quality_rank=18,
	),
	DownloadTarget(
		key='ud-iq2-s',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ2_S',
		size_bytes=8_371_970_048,
		filename='Qwen3.8-27B-UD-IQ2_S.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ2_S',
		quality_rank=22,
	),
	DownloadTarget(
		key='ud-q2-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q2_K_XL',
		size_bytes=9_828_981_664,
		filename='Qwen3.8-27B-UD-Q2_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q2_K_XL',
		quality_rank=28,
	),
	DownloadTarget(
		key='ud-iq3-xxs',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ3_XXS',
		size_bytes=10_934_860_704,
		filename='Qwen3.8-27B-UD-IQ3_XXS.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ3_XXS',
		quality_rank=32,
	),
	DownloadTarget(
		key='ud-iq3-s',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ3_S',
		size_bytes=12_040_883_104,
		filename='Qwen3.8-27B-UD-IQ3_S.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ3_S',
		quality_rank=36,
		notes='Practical 16GB-class quant.',
	),
	DownloadTarget(
		key='ud-q3-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q3_K_XL',
		size_bytes=13_146_393_504,
		filename='Qwen3.8-27B-UD-Q3_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q3_K_XL',
		quality_rank=40,
	),
	DownloadTarget(
		key='ud-iq4-xs',
		kind='unsloth-gguf',
		label='Unsloth UD-IQ4_XS',
		size_bytes=14_252_845_984,
		filename='Qwen3.8-27B-UD-IQ4_XS.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='IQ4_XS',
		quality_rank=48,
	),
	DownloadTarget(
		key='ud-q4-k-s',
		kind='unsloth-gguf',
		label='Unsloth UD-Q4_K_S',
		size_bytes=15_358_213_024,
		filename='Qwen3.8-27B-UD-Q4_K_S.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q4_K_S',
		quality_rank=52,
	),
	DownloadTarget(
		key='q4-0',
		kind='unsloth-gguf',
		label='Unsloth Q4_0',
		size_bytes=16_056_478_688,
		filename='Qwen3.8-27B-Q4_0.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q4_0',
		quality_rank=50,
	),
	DownloadTarget(
		key='ud-q4-k-m',
		kind='unsloth-gguf',
		label='Unsloth UD-Q4_K_M',
		size_bytes=16_464_440_224,
		filename='Qwen3.8-27B-UD-Q4_K_M.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q4_K_M',
		quality_rank=56,
	),
	DownloadTarget(
		key='q4-1',
		kind='unsloth-gguf',
		label='Unsloth Q4_1',
		size_bytes=17_540_705_248,
		filename='Qwen3.8-27B-Q4_1.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q4_1',
		quality_rank=54,
	),
	DownloadTarget(
		key='ud-q4-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q4_K_XL',
		size_bytes=17_559_178_144,
		filename='Qwen3.8-27B-UD-Q4_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q4_K_XL',
		quality_rank=62,
		notes='Default 24GB pick. Unsloth Dynamic V3.',
	),
	DownloadTarget(
		key='ud-q5-k-s',
		kind='unsloth-gguf',
		label='Unsloth UD-Q5_K_S',
		size_bytes=18_665_753_504,
		filename='Qwen3.8-27B-UD-Q5_K_S.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q5_K_S',
		quality_rank=66,
	),
	DownloadTarget(
		key='ud-q5-k-m',
		kind='unsloth-gguf',
		label='Unsloth UD-Q5_K_M',
		size_bytes=19_771_509_664,
		filename='Qwen3.8-27B-UD-Q5_K_M.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q5_K_M',
		quality_rank=70,
	),
	DownloadTarget(
		key='ud-q5-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q5_K_XL',
		size_bytes=20_876_938_144,
		filename='Qwen3.8-27B-UD-Q5_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q5_K_XL',
		quality_rank=74,
	),
	DownloadTarget(
		key='ud-q6-k',
		kind='unsloth-gguf',
		label='Unsloth UD-Q6_K',
		size_bytes=21_983_677_344,
		filename='Qwen3.8-27B-UD-Q6_K.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q6_K',
		quality_rank=78,
	),
	DownloadTarget(
		key='ud-q6-k-m',
		kind='unsloth-gguf',
		label='Unsloth UD-Q6_K_M',
		size_bytes=23_088_409_504,
		filename='Qwen3.8-27B-UD-Q6_K_M.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q6_K_M',
		quality_rank=82,
	),
	DownloadTarget(
		key='ud-q6-k-l',
		kind='unsloth-gguf',
		label='Unsloth UD-Q6_K_L',
		size_bytes=24_193_919_904,
		filename='Qwen3.8-27B-UD-Q6_K_L.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q6_K_L',
		quality_rank=84,
	),
	DownloadTarget(
		key='ud-q6-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q6_K_XL',
		size_bytes=25_299_061_664,
		filename='Qwen3.8-27B-UD-Q6_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q6_K_XL',
		quality_rank=88,
		notes='High-fidelity 32GB pick.',
	),
	DownloadTarget(
		key='ud-q8-k-l',
		kind='unsloth-gguf',
		label='Unsloth UD-Q8_K_L',
		size_bytes=28_045_695_904,
		filename='Qwen3.8-27B-UD-Q8_K_L.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q8_K_L',
		quality_rank=94,
	),
	DownloadTarget(
		key='q8-0',
		kind='unsloth-gguf',
		label='Unsloth Q8_0',
		size_bytes=29_047_086_048,
		filename='Qwen3.8-27B-Q8_0.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q8_0',
		quality_rank=96,
	),
	DownloadTarget(
		key='ud-q8-k-xl',
		kind='unsloth-gguf',
		label='Unsloth UD-Q8_K_XL',
		size_bytes=31_457_991_680,
		filename='Qwen3.8-27B-UD-Q8_K_XL.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='Q8_K_XL',
		quality_rank=98,
		notes='Near-BF16. Needs ~48GB+ once KV cache is included.',
	),
)

MMPROJ_FILES: tuple[DownloadTarget, ...] = (
	DownloadTarget(
		key='mmproj-bf16',
		kind='mmproj',
		label='Vision projector BF16',
		size_bytes=931_146_432,
		filename='mmproj-BF16.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='mmproj-BF16',
		quality_rank=99,
		notes='Pairs with any language GGUF for image/video input.',
	),
	DownloadTarget(
		key='mmproj-f16',
		kind='mmproj',
		label='Vision projector F16',
		size_bytes=927_607_488,
		filename='mmproj-F16.gguf',
		repo_id=UNSLOTH_REPO,
		quant_type='mmproj-F16',
		quality_rank=97,
	),
)

OLLAMA_TAGS: tuple[DownloadTarget, ...] = (
	DownloadTarget(
		key='ollama-27b',
		kind='ollama',
		label='Ollama qwen3.8:27b (MTP Q4_K_M)',
		size_bytes=18_000_000_000,
		ollama_tag='qwen3.8:27b',
		quant_type='Q4_K_M',
		quality_rank=60,
		notes='Official library default. Same blob as latest / 27b-mtp-q4_K_M.',
	),
	DownloadTarget(
		key='ollama-q4-k-m',
		kind='ollama',
		label='Ollama qwen3.8:27b-q4_K_M',
		size_bytes=18_000_000_000,
		ollama_tag='qwen3.8:27b-q4_K_M',
		quant_type='Q4_K_M',
		quality_rank=58,
		notes='Non-MTP Q4_K_M tag.',
	),
	DownloadTarget(
		key='ollama-q8-0',
		kind='ollama',
		label='Ollama qwen3.8:27b-q8_0',
		size_bytes=30_000_000_000,
		ollama_tag='qwen3.8:27b-q8_0',
		quant_type='Q8_0',
		quality_rank=90,
	),
	DownloadTarget(
		key='ollama-mtp-q8-0',
		kind='ollama',
		label='Ollama qwen3.8:27b-mtp-q8_0',
		size_bytes=30_000_000_000,
		ollama_tag='qwen3.8:27b-mtp-q8_0',
		quant_type='Q8_0',
		quality_rank=91,
	),
	DownloadTarget(
		key='ollama-bf16',
		kind='ollama',
		label='Ollama qwen3.8:27b-bf16',
		size_bytes=56_000_000_000,
		ollama_tag='qwen3.8:27b-bf16',
		quant_type='BF16',
		quality_rank=100,
	),
	DownloadTarget(
		key='ollama-mtp-bf16',
		kind='ollama',
		label='Ollama qwen3.8:27b-mtp-bf16',
		size_bytes=56_000_000_000,
		ollama_tag='qwen3.8:27b-mtp-bf16',
		quant_type='BF16',
		quality_rank=101,
	),
	DownloadTarget(
		key='ollama-mlx',
		kind='ollama',
		label='Ollama qwen3.8:27b-mlx',
		size_bytes=18_000_000_000,
		ollama_tag='qwen3.8:27b-mlx',
		quant_type='MLX',
		linux_ok=False,
		quality_rank=60,
		notes='Apple Silicon only.',
	),
	DownloadTarget(
		key='ollama-mlx-bf16',
		kind='ollama',
		label='Ollama qwen3.8:27b-mlx-bf16',
		size_bytes=56_000_000_000,
		ollama_tag='qwen3.8:27b-mlx-bf16',
		quant_type='MLX-BF16',
		linux_ok=False,
		quality_rank=100,
		notes='Apple Silicon only.',
	),
	DownloadTarget(
		key='ollama-mxfp8',
		kind='ollama',
		label='Ollama qwen3.8:27b-mxfp8',
		size_bytes=32_000_000_000,
		ollama_tag='qwen3.8:27b-mxfp8',
		quant_type='MXFP8',
		linux_ok=False,
		quality_rank=86,
		notes='Apple Silicon MLX FP8.',
	),
	DownloadTarget(
		key='ollama-nvfp4',
		kind='ollama',
		label='Ollama qwen3.8:27b-nvfp4',
		size_bytes=18_000_000_000,
		ollama_tag='qwen3.8:27b-nvfp4',
		quant_type='NVFP4',
		linux_ok=False,
		quality_rank=61,
		notes='Ollama lists this under MLX; skip on Linux.',
	),
)

ORIGINAL_TARGET = DownloadTarget(
	key='original',
	kind='original',
	label='Official Qwen/Qwen3.8-27B safetensors',
	size_bytes=ORIGINAL_SAFETENSOR_BYTES,
	repo_id=OFFICIAL_REPO,
	quant_type='BF16',
	quality_rank=100,
	notes='Apache-2.0 source weights. Required for self-quantization.',
)

BF16_GGUF_ESTIMATE = DownloadTarget(
	key='self-bf16-gguf',
	kind='self-quant',
	label='Self-converted BF16 GGUF',
	size_bytes=54_660_000_000,
	filename='Qwen3.8-27B-bf16.gguf',
	quant_type='BF16',
	quality_rank=100,
	notes='Lossless GGUF conversion of the official safetensors.',
)


def self_quant_targets() -> tuple[DownloadTarget, ...]:
	items = [BF16_GGUF_ESTIMATE]
	for quant_type in LLAMA_QUANT_TYPES:
		items.append(
			DownloadTarget(
				key=f'self-{quant_type.lower()}',
				kind='self-quant',
				label=f'Self-quant {quant_type}',
				size_bytes=SELF_QUANT_SIZE_ESTIMATES[quant_type],
				filename=f'Qwen3.8-27B-self-{quant_type}.gguf',
				quant_type=quant_type,
				quality_rank={'Q8_0': 93, 'Q6_K': 80, 'Q5_K_M': 68, 'Q4_K_M': 55}.get(quant_type, 40),
				notes='llama-quantize from official BF16 GGUF (no imatrix).',
			)
		)
	return tuple(items)


def all_download_targets(*, include_self_quant: bool = True, linux_only: bool = False) -> list[DownloadTarget]:
	targets = [ORIGINAL_TARGET, *UNSLOTH_GGUFS, *MMPROJ_FILES, *OLLAMA_TAGS]
	if include_self_quant:
		targets.extend(self_quant_targets())
	if linux_only:
		targets = [target for target in targets if target.linux_ok]
	return targets


def fit_by_bytes(
	targets: list[DownloadTarget],
	budget_bytes: int,
	*,
	reserve_bytes: int = 8_000_000_000,
	pinned_keys: tuple[str, ...] = (),
	priority_keys: tuple[str, ...] = (),
) -> tuple[list[DownloadTarget], list[DownloadTarget]]:
	"""Pack as many artifacts as possible under a disk budget.

	Pinned keys are reserved first (default local quant, projector, official weights).
	Priority keys then cover the rest of the bit-width ladder (Q5/Q6/Q8, 16GB IQ3, …)
	before leftover disk is filled smallest-first.
	"""
	assert budget_bytes >= 0
	usable = max(0, budget_bytes - reserve_bytes)
	by_key = {target.key: target for target in targets}
	chosen: list[DownloadTarget] = []
	skipped: list[DownloadTarget] = []
	used = 0

	def _try_add(target: DownloadTarget) -> None:
		nonlocal used
		if target.key in {item.key for item in chosen} or target.key in {item.key for item in skipped}:
			return
		if used + target.size_bytes <= usable:
			chosen.append(target)
			used += target.size_bytes
		else:
			skipped.append(target)

	for key in pinned_keys:
		if key in by_key:
			_try_add(by_key[key])
	for key in priority_keys:
		if key in by_key:
			_try_add(by_key[key])

	rest = [target for target in targets if target.key not in set(pinned_keys) | set(priority_keys)]
	for target in sorted(rest, key=lambda item: (item.size_bytes, item.key)):
		_try_add(target)

	chosen.sort(key=lambda item: (item.kind, -item.quality_rank, item.key))
	skipped.sort(key=lambda item: (item.kind, -item.quality_rank, item.key))
	return chosen, skipped


DEFAULT_PINNED_KEYS: tuple[str, ...] = ('ud-q4-k-xl', 'mmproj-bf16', 'original')
# After the 24GB default, cover the rest of the bit-width ladder before filling
# leftover disk with duplicate Q4_0/Q4_1 clones.
COVERAGE_KEYS: tuple[str, ...] = (
	'ud-q5-k-xl',
	'ud-q6-k-xl',
	'ud-q8-k-xl',
	'q8-0',
	'ud-iq4-xs',
	'ud-iq3-s',
	'ud-q3-k-xl',
	'ud-q2-k-xl',
	'ud-iq1-s',
	'mmproj-f16',
	'ollama-27b',
	'ollama-q8-0',
)


def default_serve_quant(available_ram_bytes: int) -> DownloadTarget:
	"""Pick the strongest Unsloth GGUF that can stay resident with an 8K KV cache."""
	headroom = 2 * 1024**3
	usable = max(0, available_ram_bytes - headroom)
	candidates = sorted(UNSLOTH_GGUFS, key=lambda item: item.quality_rank, reverse=True)
	for target in candidates:
		if int(target.size_bytes * 1.15) <= usable:
			return target
	return UNSLOTH_GGUFS[0]
