"""Pydantic models for official Qwen3.8-27B quantized checkpoints.

This catalog is the source of truth for which original (Qwen/Ollama) quants
browser-use will pull and serve locally. Community GGUF ladders are intentionally
out of scope — those are not the upstream release.
"""

# @file purpose: Defines Qwen3.8-27B official quantized-model catalog schemas

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuantRuntime = Literal['gguf', 'mlx', 'vllm']
QuantSource = Literal['ollama', 'huggingface']
StartAction = Literal['planned', 'pulled', 'loaded', 'skipped', 'failed']


class Qwen3827BQuant(BaseModel):
	"""One official quantized checkpoint of Qwen/Qwen3.8-27B."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	tag: str
	quant: str
	size_gb: float = Field(gt=0)
	runtime: QuantRuntime
	source: QuantSource
	platforms: tuple[str, ...]
	quantized: bool = True
	mtp: bool = False
	recommended: bool = False
	base_model: str = 'Qwen/Qwen3.8-27B'
	notes: str = ''


class CommandResult(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	returncode: int
	stdout: str = ''
	stderr: str = ''


class QuantStartResult(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	tag: str
	action: StartAction
	reason: str | None = None


class StartReport(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	ollama_running: bool
	results: list[QuantStartResult]


OFFICIAL_QUANTIZED_MODELS: tuple[Qwen3827BQuant, ...] = (
	Qwen3827BQuant(
		tag='qwen3.8:27b-q4_K_M',
		quant='Q4_K_M',
		size_gb=18,
		runtime='gguf',
		source='ollama',
		platforms=('linux', 'darwin', 'win32'),
		notes='Standard 4-bit GGUF. Fits a 24GB GPU at normal context.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-q8_0',
		quant='Q8_0',
		size_gb=30,
		runtime='gguf',
		source='ollama',
		platforms=('linux', 'darwin', 'win32'),
		notes='High-fidelity 8-bit GGUF. Needs ~40GB VRAM to stay on GPU.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-mtp-q4_K_M',
		quant='Q4_K_M',
		size_gb=18,
		runtime='gguf',
		source='ollama',
		platforms=('linux', 'darwin', 'win32'),
		mtp=True,
		recommended=True,
		notes='Official Ollama default (also tagged qwen3.8:27b / qwen3.8:latest) with MTP.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-mtp-q8_0',
		quant='Q8_0',
		size_gb=30,
		runtime='gguf',
		source='ollama',
		platforms=('linux', 'darwin', 'win32'),
		mtp=True,
		notes='8-bit GGUF with native multi-token prediction draft.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-mlx',
		quant='MLX',
		size_gb=18,
		runtime='mlx',
		source='ollama',
		platforms=('darwin',),
		notes='Apple Silicon MLX 4-bit pack. Linux/Windows skip this.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-mxfp8',
		quant='MXFP8',
		size_gb=32,
		runtime='mlx',
		source='ollama',
		platforms=('darwin',),
		notes='Apple Silicon MXFP8 quant.',
	),
	Qwen3827BQuant(
		tag='qwen3.8:27b-nvfp4',
		quant='NVFP4',
		size_gb=18,
		runtime='mlx',
		source='ollama',
		platforms=('darwin',),
		notes='Apple Silicon NVFP4 quant published in the Ollama library.',
	),
	Qwen3827BQuant(
		tag='Qwen/Qwen3.8-27B-FP8',
		quant='FP8',
		size_gb=31,
		runtime='vllm',
		source='huggingface',
		platforms=('linux', 'win32'),
		notes='Official Qwen FP8 (block-128). Served with vLLM/SGLang when CUDA is present.',
	),
)

# Canonical Ollama alias for the recommended MTP Q4_K_M blob.
DEFAULT_OLLAMA_TAG = 'qwen3.8:27b'
DEFAULT_VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'
DEFAULT_NUM_CTX = 32768
