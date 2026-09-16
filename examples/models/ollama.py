# Local models via Ollama: https://github.com/ollama/ollama
#
# Qwen3.8-27B setup (one non-overlapping MTP quant):
#   ./bin/start-qwen38-local.sh setup
#   ./bin/start-qwen38-local.sh verify

import os

from browser_use import Agent, ChatOllama

_tier_model = {
	'balanced': 'qwen3.8:27b-mtp-q4_K_M',
	'quality': 'qwen3.8:27b-mtp-q8_0',
	'max': 'qwen3.8:27b-mtp-bf16',
}
_tier = os.getenv('QWEN38_TIER', 'balanced')
_model = _tier_model.get(_tier, _tier_model['balanced'])
_num_ctx = int(os.getenv('QWEN38_NUM_CTX', '4096'))

llm = ChatOllama(
	model=_model,
	ollama_options={
		'num_ctx': _num_ctx,
		'temperature': 0.1,
		'think': False,
	},
)

Agent('find the founders of browser-use', llm=llm).run_sync()
