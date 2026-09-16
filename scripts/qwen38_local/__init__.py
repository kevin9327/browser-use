"""Local launcher for official Qwen3.8-27B and every public quantized build.

# @file purpose: Package entry for Qwen3.8-27B original + quantized local serving
"""

from .catalog import (
	LLAMA_QUANT_TYPES,
	OLLAMA_TAGS,
	UNSLOTH_GGUFS,
	all_download_targets,
	fit_by_bytes,
)
from .views import ArtifactKind, DownloadTarget, HardwareInfo, StartPlan

__all__ = [
	'ArtifactKind',
	'DownloadTarget',
	'HardwareInfo',
	'LLAMA_QUANT_TYPES',
	'OLLAMA_TAGS',
	'StartPlan',
	'UNSLOTH_GGUFS',
	'all_download_targets',
	'fit_by_bytes',
]
