#!/usr/bin/env python3
"""ChatOllama end-to-end smoke test for Qwen3.8 local setup.

Used by bin/start-qwen38-local.sh verify when hardware has enough memory.
"""

from __future__ import annotations

import asyncio
import os
import sys

from browser_use.llm.messages import UserMessage
from browser_use.llm.ollama.chat import ChatOllama


async def _main() -> int:
	tier = os.getenv('QWEN38_TIER', 'balanced')
	models = {
		'balanced': 'qwen3.8:27b-mtp-q4_K_M',
		'quality': 'qwen3.8:27b-mtp-q8_0',
		'max': 'qwen3.8:27b-mtp-bf16',
	}
	model = models.get(tier, models['balanced'])
	num_ctx = int(os.getenv('QWEN38_NUM_CTX', '4096'))

	llm = ChatOllama(
		model=model,
		ollama_options={'num_ctx': num_ctx, 'temperature': 0.1, 'think': False},
	)
	result = await llm.ainvoke([UserMessage(content='Reply with exactly: OK')])
	text = (result.completion or '').strip()
	if not text:
		print('FAIL: empty completion', file=sys.stderr)
		return 1
	print(f'OK: ChatOllama responded ({len(text)} chars)')
	return 0


if __name__ == '__main__':
	raise SystemExit(asyncio.run(_main()))
