# Local models via Ollama: https://github.com/ollama/ollama
#
# Qwen3.8-27B (recommended): run once — picks one non-overlapping MTP quant by VRAM
#   ./bin/start-qwen38-local.sh setup
#
# Generic small model:
#   ollama pull llama3.1:8b && ollama serve

import os

from browser_use import Agent, ChatOllama

# Override tier: balanced | quality | max — see ./bin/start-qwen38-local.sh detect
_tier_model = {
	'balanced': 'qwen3.8:27b-mtp-q4_K_M',
	'quality': 'qwen3.8:27b-mtp-q8_0',
	'max': 'qwen3.8:27b-mtp-bf16',
}
_model = _tier_model.get(os.getenv('QWEN38_TIER', 'balanced'), _tier_model['balanced'])

llm = ChatOllama(
	model=_model,
	ollama_options={
		'num_ctx': 8192,
		'temperature': 0.1,
		'think': False,  # agent tasks: disable thinking for lower latency
	},
)

Agent('find the founders of browser-use', llm=llm).run_sync()
