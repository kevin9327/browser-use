"""Official Qwen3.8-27B local quantized models for browser-use agents.

Use `chat_qwen38_27b()` after starting the Ollama (and optional vLLM) servers:
`uv run python -m browser_use.llm.qwen38_27b start`
"""

# @file purpose: Exports Qwen3.8-27B local quant catalog and ChatOllama factory

from browser_use.llm.qwen38_27b.service import (
	Qwen3827BLocalRuntime,
	chat_qwen38_27b,
	format_start_report,
	is_qwen38_27b_name,
	resolve_quant,
	resolve_quant_tag,
	select_quants_for_platform,
)
from browser_use.llm.qwen38_27b.views import (
	DEFAULT_OLLAMA_TAG,
	OFFICIAL_QUANTIZED_MODELS,
	Qwen3827BQuant,
	StartReport,
)

__all__ = [
	'DEFAULT_OLLAMA_TAG',
	'OFFICIAL_QUANTIZED_MODELS',
	'Qwen3827BLocalRuntime',
	'Qwen3827BQuant',
	'StartReport',
	'chat_qwen38_27b',
	'format_start_report',
	'is_qwen38_27b_name',
	'resolve_quant',
	'resolve_quant_tag',
	'select_quants_for_platform',
]
