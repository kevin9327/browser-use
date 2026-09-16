"""Local Qwen3.8-27B quantized catalog, llama-server fleet, and ChatQwen38 client.

Uses pytest-httpserver for the OpenAI-compatible endpoint. Does not download GGUFs
or call real remote model hosts.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pytest_httpserver import HTTPServer

from browser_use.llm.messages import UserMessage
from browser_use.llm.qwen38.chat import ChatQwen38
from browser_use.llm.qwen38.service import (
	Qwen38LocalService,
	catalog_from_hf_tree,
	fallback_catalog,
	fits_hardware,
	llama_server_argv,
	quant_name_from_filename,
)
from browser_use.llm.qwen38.views import (
	DEFAULT_QUANT,
	ORIGINAL_MODEL_ID,
	ORIGINAL_REVISION,
	QUANT_REPO_ID,
	RECOMMENDED_QUANTS,
	Qwen38Hardware,
	Qwen38Quant,
)


class CapitalResponse(BaseModel):
	country: str
	capital: str


HF_TREE = [
	{'path': 'layouts', 'type': 'directory', 'size': 0},
	{'path': 'README.md', 'type': 'file', 'size': 100},
	{'path': 'Qwen3.8-27B-imatrix.gguf', 'type': 'file', 'size': 13642688},
	{'path': 'mmproj-Qwen3.8-27B-f16.gguf', 'type': 'file', 'size': 927607008},
	{'path': 'mmproj-Qwen3.8-27B-bf16.gguf', 'type': 'file', 'size': 931145952},
	{'path': 'Qwen3.8-27B-Q4_K_M.gguf', 'type': 'file', 'size': 17442399968},
	{'path': 'Qwen3.8-27B-Q8_0.gguf', 'type': 'file', 'size': 29116388960},
	{'path': 'Qwen3.8-27B-IQ2_XXS.gguf', 'type': 'file', 'size': 8881268448},
	{'path': 'Qwen3.8-27B-Q5_K_L.gguf', 'type': 'file', 'size': 21537478240},
	{'path': 'Qwen3.8-27B-bf16/Qwen3.8-27B-bf16-00001-of-00002.gguf', 'type': 'file', 'size': 1},
]


def test_quant_name_from_filename_skips_non_language_ggufs():
	assert quant_name_from_filename('Qwen3.8-27B-Q4_K_M.gguf') == 'Q4_K_M'
	assert quant_name_from_filename('mmproj-Qwen3.8-27B-f16.gguf') is None
	assert quant_name_from_filename('Qwen3.8-27B-imatrix.gguf') is None
	assert quant_name_from_filename('README.md') is None


def test_catalog_from_hf_tree_assigns_unique_ports_and_keeps_original_sha():
	catalog = catalog_from_hf_tree(HF_TREE, original_sha='abc123', source='huggingface')
	assert catalog.original_repo == ORIGINAL_MODEL_ID
	assert catalog.original_sha == 'abc123'
	assert catalog.quant_repo == QUANT_REPO_ID
	names = [item.quant for item in catalog.quants]
	assert names == ['Q8_0', 'Q5_K_L', 'Q4_K_M', 'IQ2_XXS']
	assert [item.port for item in catalog.quants] == [8080, 8081, 8082, 8083]
	assert len({item.port for item in catalog.quants}) == len(catalog.quants)
	assert catalog.get('Q4_K_M').recommended is True
	assert catalog.get('q4_k_m').filename == 'Qwen3.8-27B-Q4_K_M.gguf'
	assert catalog.get('Q5_K_L').legacy is True
	assert catalog.get('Q8_0').recommended is False
	assert {item.filename for item in catalog.mmproj} == {
		'mmproj-Qwen3.8-27B-f16.gguf',
		'mmproj-Qwen3.8-27B-bf16.gguf',
	}


def test_fallback_catalog_includes_every_pinned_language_quant():
	catalog = fallback_catalog()
	assert catalog.source == 'fallback'
	assert catalog.original_sha == ORIGINAL_REVISION
	assert catalog.get(DEFAULT_QUANT).quant == DEFAULT_QUANT
	assert RECOMMENDED_QUANTS <= {item.quant for item in catalog.quants}
	assert len(catalog.quants) >= 20
	try:
		catalog.get('not-a-quant')
		raise AssertionError('expected KeyError')
	except KeyError as exc:
		assert 'not-a-quant' in str(exc)


def test_hardware_fit_uses_ram_plus_vram_minus_reserve():
	tiny = Qwen38Hardware(ram_bytes=8 * 1024**3, vram_bytes=0)
	gpu = Qwen38Hardware(ram_bytes=16 * 1024**3, vram_bytes=24 * 1024**3, gpu_layers=99)
	q4 = 17_442_399_968
	assert fits_hardware(q4, tiny) is False
	assert fits_hardware(q4, gpu) is True


def test_llama_server_argv_targets_official_bartowski_hf_spec():
	quant = Qwen38Quant(
		quant='Q4_K_M',
		filename='Qwen3.8-27B-Q4_K_M.gguf',
		size_bytes=17442399968,
		recommended=True,
		port=8080,
	)
	argv = llama_server_argv(
		quant,
		llama_server_bin='/opt/llama-server',
		hardware=Qwen38Hardware(ram_bytes=32 * 1024**3, vram_bytes=24 * 1024**3, gpu_layers=99),
		mmproj='mmproj-Qwen3.8-27B-f16.gguf',
	)
	assert argv[0] == '/opt/llama-server'
	assert '-hf' in argv and 'bartowski/Qwen3.8-27B-GGUF:Q4_K_M' in argv
	assert '--jinja' in argv
	assert argv[argv.index('--spec-type') + 1] == 'draft-mtp'
	assert argv[argv.index('-a') + 1] == 'qwen3.8-27b-Q4_K_M'
	assert argv[argv.index('-ngl') + 1] == '99'
	assert argv[argv.index('--mmproj') + 1] == 'mmproj-Qwen3.8-27B-f16.gguf'


async def test_refresh_catalog_reads_live_huggingface_payload(httpserver: HTTPServer):
	httpserver.expect_request('/Qwen/Qwen3.8-27B', method='GET').respond_with_json({'sha': 'deadbeefcafebabe'})
	httpserver.expect_request('/bartowski/Qwen3.8-27B-GGUF/tree/main', method='GET').respond_with_json(HF_TREE)
	service = Qwen38LocalService(hf_api_base=httpserver.url_for('').rstrip('/'))
	async with __import__('httpx').AsyncClient(timeout=5.0) as client:
		catalog = await service.refresh_catalog(client=client)
	assert catalog.source == 'huggingface'
	assert catalog.original_sha == 'deadbeefcafebabe'
	assert catalog.get('Q4_K_M').size_bytes == 17442399968


async def test_refresh_catalog_falls_back_when_huggingface_is_down(httpserver: HTTPServer):
	httpserver.expect_request('/Qwen/Qwen3.8-27B', method='GET').respond_with_data('nope', status=500)
	service = Qwen38LocalService(hf_api_base=httpserver.url_for('').rstrip('/'))
	async with __import__('httpx').AsyncClient(timeout=5.0) as client:
		catalog = await service.refresh_catalog(client=client)
	assert catalog.source == 'fallback'
	assert catalog.get(DEFAULT_QUANT).quant == DEFAULT_QUANT


def test_chat_qwen38_default_base_url_uses_catalog_port():
	llm = ChatQwen38(quant='Q4_K_M')
	assert llm.base_url == 'http://127.0.0.1:8089/v1'
	ollama = ChatQwen38(quant='qwen3.8:27b')
	assert ollama.base_url == 'http://127.0.0.1:11434/v1'
	assert ollama.model == 'qwen3.8:27b'


def test_plan_fleet_cumulative_ram_does_not_overcommit(tmp_path: Path):
	stub = tmp_path / 'llama-server'
	stub.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
	stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
	service = Qwen38LocalService(cache_dir=tmp_path, llama_server_bin=str(stub))
	catalog = catalog_from_hf_tree(HF_TREE)
	# 32 GiB usable after reserve: Q8_0 (27.12) fits, remaining cannot take Q5_K_L (20.06).
	fleet = service.plan_fleet(catalog, hardware=Qwen38Hardware(ram_bytes=34 * 1024**3))
	started = [spec.quant.quant for spec in fleet.servers if spec.skip_reason is None]
	assert started == ['Q8_0']
	assert fleet.servers[0].fits_hardware is True
	assert fleet.servers[1].skip_reason is not None
	service = Qwen38LocalService(cache_dir=tmp_path, llama_server_bin='/definitely/missing/llama-server')
	catalog = catalog_from_hf_tree(HF_TREE)
	fleet = service.plan_fleet(catalog, hardware=Qwen38Hardware(ram_bytes=8 * 1024**3), force=True)
	assert fleet.llama_server_bin is None
	assert all(spec.skip_reason for spec in fleet.servers)
	script = service.write_launch_scripts(fleet)
	assert script.is_file()
	text = script.read_text(encoding='utf-8')
	assert ORIGINAL_MODEL_ID in text
	assert 'Qwen3.8-27B' in text


async def test_start_all_spawns_stub_llama_server_for_every_fitting_quant(tmp_path: Path):
	stub = tmp_path / 'llama-server'
	log = tmp_path / 'argv.log'
	stub.write_text(
		'#!/usr/bin/env python3\n'
		'import sys\n'
		f'open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")\n'
		'import time; time.sleep(30)\n',
		encoding='utf-8',
	)
	stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
	service = Qwen38LocalService(cache_dir=tmp_path / 'cache', llama_server_bin=str(stub), base_port=18080)
	catalog = catalog_from_hf_tree(HF_TREE)
	fleet = await service.start_all(
		catalog=catalog,
		hardware=Qwen38Hardware(ram_bytes=256 * 1024**3, gpu_layers=0),
		spawn=True,
		force=True,
	)
	try:
		assert fleet.llama_server_bin == str(stub)
		assert len(fleet.started()) == len(catalog.quants)
		for _ in range(20):
			if log.exists() and log.read_text(encoding='utf-8').count('\n') >= len(catalog.quants):
				break
			await asyncio.sleep(0.05)
		logged = log.read_text(encoding='utf-8')
		assert 'bartowski/Qwen3.8-27B-GGUF:Q4_K_M' in logged
		assert '--jinja' in logged
		assert 'qwen3.8-27b-IQ2_XXS' in logged
	finally:
		await service.stop_all()


async def test_chat_qwen38_text_and_structured_against_local_openai_server(httpserver: HTTPServer):
	captured: dict[str, Any] = {}

	def handler(request):
		payload = json.loads(request.data.decode('utf-8'))
		captured.update(payload)
		content = payload['messages'][-1]['content']
		if 'json_schema' in json.dumps(payload.get('response_format', {})):
			body = {
				'id': 'chatcmpl-qwen38',
				'object': 'chat.completion',
				'choices': [
					{
						'index': 0,
						'message': {
							'role': 'assistant',
							'content': '{"country":"France","capital":"Paris"}',
							'reasoning_content': 'checking capitals',
						},
						'finish_reason': 'stop',
					}
				],
				'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14},
			}
		else:
			body = {
				'id': 'chatcmpl-qwen38',
				'object': 'chat.completion',
				'choices': [
					{
						'index': 0,
						'message': {'role': 'assistant', 'content': f'echo:{content}'},
						'finish_reason': 'stop',
					}
				],
				'usage': {'prompt_tokens': 8, 'completion_tokens': 2, 'total_tokens': 12},
			}
		from werkzeug import Response

		return Response(json.dumps(body), status=200, content_type='application/json')

	httpserver.expect_request('/v1/chat/completions', method='POST').respond_with_handler(handler)
	llm = ChatQwen38(quant='Q4_K_M', base_url=httpserver.url_for('/v1'), timeout=10.0)
	assert llm.model == 'qwen3.8-27b-Q4_K_M'
	assert llm.provider == 'qwen38-local'
	text = await llm.ainvoke([UserMessage(content='hello')])
	assert text.completion == 'echo:hello'
	assert captured['model'] == 'qwen3.8-27b-Q4_K_M'
	assert captured['temperature'] == 1.0
	assert captured['top_p'] == 0.95
	assert captured['chat_template_kwargs']['enable_thinking'] is True
	assert captured['reasoning_effort'] == 'xhigh'

	structured = await llm.ainvoke([UserMessage(content='capital?')], output_format=CapitalResponse)
	assert structured.completion.capital == 'Paris'
	assert structured.thinking == 'checking capitals'


async def test_chat_qwen38_instruct_sampling_disables_thinking(httpserver: HTTPServer):
	captured: dict[str, Any] = {}

	def handler(request):
		captured.update(json.loads(request.data.decode('utf-8')))
		from werkzeug import Response

		body = {
			'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
			'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
		}
		return Response(json.dumps(body), status=200, content_type='application/json')

	httpserver.expect_request('/v1/chat/completions', method='POST').respond_with_handler(handler)
	llm = ChatQwen38(quant='Q6_K', enable_thinking=False, base_url=httpserver.url_for('/v1'), timeout=10.0)
	await llm.ainvoke([UserMessage(content='hi')])
	assert captured['model'] == 'qwen3.8-27b-Q6_K'
	assert captured['temperature'] == 0.7
	assert captured['top_p'] == 0.80
	assert captured['chat_template_kwargs']['enable_thinking'] is False
