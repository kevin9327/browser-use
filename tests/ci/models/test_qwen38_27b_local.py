"""Official Qwen3.8-27B local quantized-model catalog and launcher.

These tests pin the original Ollama/HF quant set (not community GGUF ladders)
and the commands used to start them for local browser-use.
"""

import sys
from typing import Any

from browser_use.llm.ollama.chat import ChatOllama
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.qwen38_27b.service import (
	Qwen3827BLocalRuntime,
	chat_qwen38_27b,
	is_qwen38_27b_name,
	resolve_quant,
	resolve_quant_tag,
	select_quants_for_platform,
)
from browser_use.llm.qwen38_27b.views import CommandResult, OFFICIAL_QUANTIZED_MODELS


def test_catalog_is_original_quantized_qwen38_27b_only():
	tags = {spec.tag for spec in OFFICIAL_QUANTIZED_MODELS}
	assert tags == {
		'qwen3.8:27b-q4_K_M',
		'qwen3.8:27b-q8_0',
		'qwen3.8:27b-mtp-q4_K_M',
		'qwen3.8:27b-mtp-q8_0',
		'qwen3.8:27b-mlx',
		'qwen3.8:27b-mxfp8',
		'qwen3.8:27b-nvfp4',
		'Qwen/Qwen3.8-27B-FP8',
	}
	assert all(spec.quantized for spec in OFFICIAL_QUANTIZED_MODELS)
	assert all('bf16' not in spec.tag.lower() for spec in OFFICIAL_QUANTIZED_MODELS)
	assert all(spec.base_model == 'Qwen/Qwen3.8-27B' for spec in OFFICIAL_QUANTIZED_MODELS)


def test_linux_starts_gguf_quants_and_skips_apple_mlx():
	selected = select_quants_for_platform('linux')
	ollama_tags = {spec.tag for spec in selected if spec.source == 'ollama'}
	assert ollama_tags == {
		'qwen3.8:27b-q4_K_M',
		'qwen3.8:27b-q8_0',
		'qwen3.8:27b-mtp-q4_K_M',
		'qwen3.8:27b-mtp-q8_0',
	}
	assert all(spec.runtime != 'mlx' for spec in selected)


def test_darwin_includes_mlx_quants():
	selected = select_quants_for_platform('darwin')
	tags = {spec.tag for spec in selected}
	assert 'qwen3.8:27b-mlx' in tags
	assert 'qwen3.8:27b-mxfp8' in tags
	assert 'qwen3.8:27b-nvfp4' in tags
	assert 'qwen3.8:27b-q4_K_M' in tags


def test_resolve_quant_tag_accepts_local_aliases():
	assert resolve_quant_tag('default') == 'qwen3.8:27b'
	assert resolve_quant_tag('qwen38_27b') == 'qwen3.8:27b'
	assert resolve_quant_tag('q4') == 'qwen3.8:27b-q4_K_M'
	assert resolve_quant_tag('qwen38_27b_q4_k_m') == 'qwen3.8:27b-q4_K_M'
	assert resolve_quant_tag('q8_0') == 'qwen3.8:27b-q8_0'
	assert resolve_quant_tag('mtp-q8_0') == 'qwen3.8:27b-mtp-q8_0'
	assert resolve_quant_tag('mxfp8') == 'qwen3.8:27b-mxfp8'
	assert resolve_quant_tag('fp8') == 'Qwen/Qwen3.8-27B-FP8'
	assert resolve_quant('qwen3.8:27b').recommended is True


def test_is_qwen38_27b_name_detects_factory_aliases():
	assert is_qwen38_27b_name('qwen38_27b')
	assert is_qwen38_27b_name('qwen3_8_27b_q8_0')
	assert is_qwen38_27b_name('qwen38-27b-mtp-q4-k-m')
	assert not is_qwen38_27b_name('llama3_1_8b')
	assert not is_qwen38_27b_name('qwen_3_32b')


def test_chat_qwen38_27b_defaults_to_official_ollama_tag_without_thinking():
	llm = chat_qwen38_27b()
	assert isinstance(llm, ChatOllama)
	assert llm.model == 'qwen3.8:27b'
	assert llm.ollama_options is not None
	assert llm.ollama_options['think'] is False
	assert llm.ollama_options['num_ctx'] == 32768


def test_chat_qwen38_27b_fp8_uses_openai_compatible_local_endpoint():
	llm = chat_qwen38_27b('fp8')
	assert isinstance(llm, ChatOpenAI)
	assert llm.model == 'Qwen/Qwen3.8-27B-FP8'
	assert str(llm.base_url).rstrip('/') in {'http://127.0.0.1:8000/v1', 'http://127.0.0.1:8000/v1/'}


async def test_start_all_dry_run_plans_platform_ollama_pulls(monkeypatch: Any):
	calls: list[list[str]] = []

	async def fake_run(argv: list[str], timeout: float | None = None) -> CommandResult:
		calls.append(argv)
		return CommandResult(returncode=0, stdout='ok')

	runtime = Qwen3827BLocalRuntime(
		platform_name='linux',
		which=lambda _: '/usr/bin/ollama',
		run=fake_run,
		cuda_available=lambda: False,
	)
	report = await runtime.start_all(dry_run=True)

	planned = [result.tag for result in report.results if result.action == 'planned']
	assert planned == [
		'qwen3.8:27b',
		'qwen3.8:27b-q4_K_M',
		'qwen3.8:27b-q8_0',
		'qwen3.8:27b-mtp-q4_K_M',
		'qwen3.8:27b-mtp-q8_0',
	]
	assert any(result.tag == 'Qwen/Qwen3.8-27B-FP8' and result.action == 'skipped' for result in report.results)
	assert all(result.action != 'planned' or 'mlx' not in result.tag for result in report.results)
	assert calls == []


async def test_start_all_pulls_every_linux_quant_and_loads_the_default():
	calls: list[list[str]] = []

	async def fake_run(argv: list[str], timeout: float | None = None) -> CommandResult:
		calls.append(argv)
		if argv[:2] == ['ollama', 'list']:
			return CommandResult(returncode=0, stdout='NAME\n')
		return CommandResult(returncode=0, stdout='ok')

	runtime = Qwen3827BLocalRuntime(
		platform_name='linux',
		which=lambda name: '/usr/bin/ollama' if name == 'ollama' else None,
		run=fake_run,
		cuda_available=lambda: False,
		healthcheck=lambda: True,
	)
	report = await runtime.start_all()

	pulls = [' '.join(argv) for argv in calls if argv[:2] == ['ollama', 'pull']]
	assert pulls == [
		'ollama pull qwen3.8:27b',
		'ollama pull qwen3.8:27b-q4_K_M',
		'ollama pull qwen3.8:27b-q8_0',
		'ollama pull qwen3.8:27b-mtp-q4_K_M',
		'ollama pull qwen3.8:27b-mtp-q8_0',
	]
	assert any(argv[:2] == ['ollama', 'serve'] or argv[0].endswith('ollama') and 'serve' in argv for argv in calls) or report.ollama_running
	assert {result.tag for result in report.results if result.action == 'pulled'} >= {
		'qwen3.8:27b-q4_K_M',
		'qwen3.8:27b-q8_0',
		'qwen3.8:27b-mtp-q4_K_M',
		'qwen3.8:27b-mtp-q8_0',
	}
	assert any(result.tag == 'qwen3.8:27b' and result.action in {'pulled', 'loaded'} for result in report.results)


def test_get_llm_by_name_wires_local_qwen38_27b():
	from browser_use.llm.models import get_llm_by_name

	llm = get_llm_by_name('ollama_qwen38_27b')
	assert isinstance(llm, ChatOllama)
	assert llm.model == 'qwen3.8:27b'

	q8 = get_llm_by_name('ollama_qwen38_27b_q8_0')
	assert isinstance(q8, ChatOllama)
	assert q8.model == 'qwen3.8:27b-q8_0'


def test_module_runs_on_current_interpreter():
	assert sys.version_info >= (3, 11)
