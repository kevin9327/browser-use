"""CI tests for Qwen3.8 local Ollama setup script and optional live e2e."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_SCRIPT = REPO_ROOT / 'bin' / 'start-qwen38-local.sh'


def _mem_available_gb() -> float:
	with open('/proc/meminfo') as f:
		for line in f:
			if line.startswith('MemAvailable:'):
				return int(line.split()[1]) / 1024 / 1024
	raise RuntimeError('MemAvailable not found')


def _ollama_e2e_enabled() -> bool:
	return os.getenv('OLLAMA_E2E', '').lower() in {'1', 'true', 'yes'}


def _ollama_reachable() -> bool:
	try:
		import urllib.request

		with urllib.request.urlopen('http://127.0.0.1:11434/', timeout=2) as resp:
			return resp.status == 200
	except OSError:
		return False


def test_setup_script_exists_and_is_executable():
	assert SETUP_SCRIPT.is_file()
	assert os.access(SETUP_SCRIPT, os.X_OK)


def test_tier_models_are_mutually_exclusive():
	text = SETUP_SCRIPT.read_text()
	models = re.findall(r"MODEL_\w+='([^']+)'", text)
	assert len(models) == len(set(models))
	assert 'qwen3.8:27b-mtp-q4_K_M' in models
	assert 'qwen3.8:27b' not in models


def test_verify_script_reports_install_health():
	result = subprocess.run(
		[str(SETUP_SCRIPT), 'verify'],
		capture_output=True,
		text=True,
		cwd=REPO_ROOT,
		timeout=60,
		check=False,
	)
	combined = result.stdout + result.stderr
	assert result.returncode == 0, combined
	assert 'OK: Ollama reachable' in combined
	assert 'OK: Model available' in combined or 'OK: Model installed' in combined or 'FAIL: Model missing' in combined
	assert 'VERIFY: install OK' in combined or 'VERIFY: all checks passed' in combined


def test_status_script_one_line_format():
	result = subprocess.run(
		[str(SETUP_SCRIPT), 'status'],
		capture_output=True,
		text=True,
		cwd=REPO_ROOT,
		timeout=30,
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert re.search(r'ollama=(up|down) host=\S+ model=\S+ installed=(yes|no)', result.stdout + result.stderr)


@pytest.mark.skipif(not _ollama_e2e_enabled(), reason='Set OLLAMA_E2E=1 for live Ollama chat test')
@pytest.mark.skipif(_mem_available_gb() < 20, reason='Need ~20GB+ MemAvailable for 27B inference')
@pytest.mark.skipif(not _ollama_reachable(), reason='Ollama not reachable at 127.0.0.1:11434')
async def test_chatollama_live_smoke():
	from browser_use.llm.messages import UserMessage
	from browser_use.llm.ollama.chat import ChatOllama

	llm = ChatOllama(
		model='qwen3.8:27b-mtp-q4_K_M',
		ollama_options={'num_ctx': 2048, 'temperature': 0.1, 'think': False},
	)
	result = await llm.ainvoke([UserMessage(content='Reply with exactly: OK')])
	assert result.completion


@pytest.mark.skipif(not _ollama_e2e_enabled(), reason='Set OLLAMA_E2E=1 for live agent test')
@pytest.mark.skipif(_mem_available_gb() < 20, reason='Need ~20GB+ MemAvailable for 27B agent run')
@pytest.mark.skipif(not _ollama_reachable(), reason='Ollama not reachable at 127.0.0.1:11434')
async def test_agent_button_click_with_qwen38(httpserver):
	from tests.ci.models.model_test_helper import run_model_button_click_test

	from browser_use.llm.ollama.chat import ChatOllama

	await run_model_button_click_test(
		ChatOllama,
		'qwen3.8:27b-mtp-q4_K_M',
		api_key_env=None,
		extra_kwargs={'ollama_options': {'num_ctx': 2048, 'temperature': 0.1, 'think': False}},
		httpserver=httpserver,
	)
