"""Qwen3.8-27B local quantized model support.

Exposes ChatQwen38 (OpenAI-compatible client) and Qwen38LocalService (catalog +
llama-server fleet for every bartowski GGUF of the official Qwen/Qwen3.8-27B).
"""

# @file purpose: Public exports for local Qwen3.8-27B quantized inference

from browser_use.llm.qwen38.chat import ChatQwen38
from browser_use.llm.qwen38.service import Qwen38LocalService, fallback_catalog
from browser_use.llm.qwen38.views import (
	DEFAULT_QUANT,
	OLLAMA_MODEL,
	ORIGINAL_MODEL_ID,
	QUANT_REPO_ID,
	Qwen38Catalog,
	Qwen38Fleet,
)

__all__ = [
	'ChatQwen38',
	'DEFAULT_QUANT',
	'OLLAMA_MODEL',
	'ORIGINAL_MODEL_ID',
	'QUANT_REPO_ID',
	'Qwen38Catalog',
	'Qwen38Fleet',
	'Qwen38LocalService',
	'fallback_catalog',
]
