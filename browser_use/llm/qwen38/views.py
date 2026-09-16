"""Qwen3.8-27B local quant catalog and fleet state.

Pinned to the official Qwen/Qwen3.8-27B weights (Gated DeltaNet hybrid, native VLM)
and bartowski's llama.cpp imatrix GGUFs. The live Hugging Face tree is the source of
truth; FALLBACK_QUANT_FILES is only used when the network is unavailable.
"""

# @file purpose: Defines Qwen3.8-27B original + GGUF quant metadata for local serving

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

ORIGINAL_MODEL_ID = 'Qwen/Qwen3.8-27B'
ORIGINAL_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
QUANT_REPO_ID = 'bartowski/Qwen3.8-27B-GGUF'
MIN_LLAMA_CPP_RELEASE = 'b10896'
DEFAULT_QUANT = 'Q4_K_M'
DEFAULT_BASE_PORT = 8080
DEFAULT_ALIAS_PREFIX = 'qwen3.8-27b'
OLLAMA_MODEL = 'qwen3.8:27b'

ReasoningEffort = Literal['xhigh', 'medium', 'low']

# Quality ladder used for stable port assignment (highest quality first).
QUANT_QUALITY_ORDER: tuple[str, ...] = (
	'Q8_0',
	'Q6_K_L',
	'Q6_K',
	'Q6_K_S',
	'Q5_K_L',
	'Q5_K_M',
	'Q5_K_S',
	'Q4_K_L',
	'Q4_1',
	'Q4_K_M',
	'IQ4_NL',
	'Q4_K_S',
	'Q4_0',
	'IQ4_XS',
	'IQ3_M',
	'Q3_K_XL',
	'Q3_K_L',
	'Q3_K_M',
	'IQ3_XS',
	'Q3_K_S',
	'IQ3_XXS',
	'Q2_K_L',
	'Q2_K',
	'IQ2_M',
	'IQ2_S',
	'IQ2_XS',
	'IQ2_XXS',
)

RECOMMENDED_QUANTS: frozenset[str] = frozenset(
	{
		'Q6_K_L',
		'Q6_K',
		'Q6_K_S',
		'Q5_K_M',
		'Q5_K_S',
		'Q4_K_L',
		'Q4_K_M',
		'Q4_K_S',
		'IQ4_XS',
	}
)

# Filename -> size in bytes, captured from bartowski/Qwen3.8-27B-GGUF on 2026-09-16.
FALLBACK_QUANT_FILES: dict[str, int] = {
	'Qwen3.8-27B-Q8_0.gguf': 29116388960,
	'Qwen3.8-27B-Q6_K_L.gguf': 24958908128,
	'Qwen3.8-27B-Q6_K.gguf': 23860565728,
	'Qwen3.8-27B-Q6_K_S.gguf': 22857455328,
	'Qwen3.8-27B-Q5_K_L.gguf': 21537478240,
	'Qwen3.8-27B-Q5_K_M.gguf': 20923877088,
	'Qwen3.8-27B-Q5_K_S.gguf': 19566913248,
	'Qwen3.8-27B-Q4_K_L.gguf': 18820703968,
	'Qwen3.8-27B-Q4_1.gguf': 17825457888,
	'Qwen3.8-27B-Q4_K_M.gguf': 17442399968,
	'Qwen3.8-27B-IQ4_NL.gguf': 17442399968,
	'Qwen3.8-27B-Q4_K_S.gguf': 16363513568,
	'Qwen3.8-27B-Q3_K_XL.gguf': 16391919200,
	'Qwen3.8-27B-Q4_0.gguf': 16348767968,
	'Qwen3.8-27B-IQ4_XS.gguf': 15475951328,
	'Qwen3.8-27B-IQ3_M.gguf': 14860629728,
	'Qwen3.8-27B-Q3_K_L.gguf': 14122817248,
	'Qwen3.8-27B-Q3_K_M.gguf': 13404051168,
	'Qwen3.8-27B-Q2_K_L.gguf': 13081040480,
	'Qwen3.8-27B-IQ3_XS.gguf': 12799604448,
	'Qwen3.8-27B-Q3_K_S.gguf': 12739884768,
	'Qwen3.8-27B-IQ3_XXS.gguf': 12320167648,
	'Qwen3.8-27B-Q2_K.gguf': 10821543648,
	'Qwen3.8-27B-IQ2_M.gguf': 10522248928,
	'Qwen3.8-27B-IQ2_S.gguf': 9684043488,
	'Qwen3.8-27B-IQ2_XS.gguf': 9088526048,
	'Qwen3.8-27B-IQ2_XXS.gguf': 8881268448,
}

MMPROJ_FILES: tuple[str, ...] = (
	'mmproj-Qwen3.8-27B-f16.gguf',
	'mmproj-Qwen3.8-27B-bf16.gguf',
)


class Qwen38Quant(BaseModel):
	"""One language-trunk GGUF of Qwen3.8-27B."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	quant: str
	filename: str
	size_bytes: int
	recommended: bool = False
	legacy: bool = False
	port: int = Field(gt=0, lt=65536)

	@computed_field
	@property
	def alias(self) -> str:
		return f'{DEFAULT_ALIAS_PREFIX}-{self.quant}'

	@computed_field
	@property
	def hf_spec(self) -> str:
		return f'{QUANT_REPO_ID}:{self.quant}'

	@computed_field
	@property
	def size_gib(self) -> float:
		return round(self.size_bytes / (1024**3), 2)


class Qwen38Mmproj(BaseModel):
	"""Vision projector that pairs with any language GGUF."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	filename: str
	size_bytes: int = 0


class Qwen38Catalog(BaseModel):
	"""Latest original checkpoint + every bartowski language GGUF."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	original_repo: str = ORIGINAL_MODEL_ID
	original_sha: str = ORIGINAL_REVISION
	quant_repo: str = QUANT_REPO_ID
	min_llama_cpp_release: str = MIN_LLAMA_CPP_RELEASE
	source: Literal['huggingface', 'fallback'] = 'fallback'
	quants: list[Qwen38Quant]
	mmproj: list[Qwen38Mmproj] = Field(default_factory=list)

	def get(self, quant: str) -> Qwen38Quant:
		wanted = quant.strip().upper().replace('-', '_')
		for item in self.quants:
			if item.quant.upper() == wanted:
				return item
		available = ', '.join(item.quant for item in self.quants)
		raise KeyError(f'Unknown Qwen3.8-27B quant {quant!r}. Available: {available}')

	def language_quants(self) -> list[Qwen38Quant]:
		return list(self.quants)


class Qwen38ServerSpec(BaseModel):
	"""Ready-to-spawn llama-server invocation for one quant."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	quant: Qwen38Quant
	argv: list[str]
	host: str = '127.0.0.1'
	fits_hardware: bool
	skip_reason: str | None = None

	@computed_field
	@property
	def base_url(self) -> str:
		return f'http://{self.host}:{self.quant.port}/v1'


class Qwen38Fleet(BaseModel):
	"""All quantized endpoints the local machine should expose."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	catalog: Qwen38Catalog
	servers: list[Qwen38ServerSpec]
	llama_server_bin: str | None = None

	def started(self) -> list[Qwen38ServerSpec]:
		return [spec for spec in self.servers if spec.skip_reason is None]


class Qwen38Hardware(BaseModel):
	"""RAM + optional VRAM used to decide which quants can load."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	ram_bytes: int
	vram_bytes: int = 0
	gpu_layers: int = 0

	@computed_field
	@property
	def usable_bytes(self) -> int:
		# Keep ~2 GiB for OS / browser-use / KV cache headroom.
		reserve = 2 * 1024**3
		return max(0, self.ram_bytes + self.vram_bytes - reserve)
