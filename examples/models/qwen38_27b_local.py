"""Run browser-use against a local Qwen3.8-27B quant served by llama-server or Ollama.

Start the local stack first:

	uv run python -m scripts.qwen38_local start --all

Then:

	set -a && source ~/.cache/browser-use/qwen3.8-27b/local.env && set +a
	uv run python examples/models/qwen38_27b_local.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from browser_use import Agent
from browser_use.llm import ChatOllama, ChatOpenAI

for env_file in (
	Path.home() / '.cache' / 'browser-use' / 'qwen3.8-27b' / 'local.env',
	Path('.env'),
):
	if env_file.exists():
		load_dotenv(env_file, override=False)


def _local_llm():
	backend = os.getenv('QWEN38_BACKEND', 'llama').strip().lower()
	if backend == 'ollama':
		return ChatOllama(
			model=os.getenv('QWEN38_OLLAMA_MODEL', 'qwen3.8:27b'),
			host=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'),
		)
	return ChatOpenAI(
		model=os.getenv('QWEN38_MODEL', 'Qwen3.8-27B-UD-Q4_K_XL'),
		base_url=os.getenv('QWEN38_BASE_URL', 'http://127.0.0.1:8080/v1'),
		api_key=os.getenv('QWEN38_API_KEY', 'local'),
		temperature=0.7,
		top_p=0.8,
	)


async def main():
	agent = Agent(
		task='Go to example.com, click on the first link, and give me the title of the page',
		llm=_local_llm(),
		use_vision=False,
	)
	await agent.run(max_steps=10)


if __name__ == '__main__':
	asyncio.run(main())
