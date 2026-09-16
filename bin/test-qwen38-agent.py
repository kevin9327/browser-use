#!/usr/bin/env python3
"""Agent + browser-use e2e smoke for Qwen3.8 local Ollama setup."""

from __future__ import annotations

import asyncio
import os
import sys

from pytest_httpserver import HTTPServer

from browser_use.agent.service import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
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

	html = """
	<!DOCTYPE html>
	<html><body>
		<button id="test-button" onclick="document.getElementById('result').innerText='SUCCESS'">Click Me</button>
		<div id="result">NOT_CLICKED</div>
	</body></html>
	"""

	server = HTTPServer()
	server.expect_request('/').respond_with_data(html, content_type='text/html')
	server.start()

	llm = ChatOllama(
		model=model,
		ollama_options={'num_ctx': num_ctx, 'temperature': 0.1, 'think': False},
	)
	browser = BrowserSession(browser_profile=BrowserProfile(headless=True, user_data_dir=None))

	try:
		await browser.start()
		test_url = server.url_for('/')
		agent = Agent(
			task=f'{test_url} - Click the button',
			llm=llm,
			browser_session=browser,
			max_steps=2,
		)
		result = await agent.run()
		assert result is not None and len(result.history) > 0

		clicked = any(step.state_message and 'SUCCESS' in step.state_message for step in result.history)
		if not clicked:
			print('FAIL: button not clicked (SUCCESS not in page state)', file=sys.stderr)
			return 1
		print('OK: Agent clicked button via Qwen3.8 + ChatOllama')
		return 0
	finally:
		await browser.kill()
		server.clear()
		if server.is_running():
			server.stop()


if __name__ == '__main__':
	raise SystemExit(asyncio.run(_main()))
