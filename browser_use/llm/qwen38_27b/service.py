"""Pull and serve official Qwen3.8-27B quantized models for local browser-use.

Fits into the LLM layer as the local path for Qwen3.8-27B: the original Ollama
library quants plus Qwen's own FP8 checkpoint. ChatOllama talks to Ollama;
ChatOpenAI talks to a local vLLM OpenAI-compatible server when CUDA is available.
"""

# @file purpose: Starts official Qwen3.8-27B quantized models for local agents

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Awaitable, Callable
from typing import Literal

from browser_use.llm.base import BaseChatModel
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.qwen38_27b.views import (
	DEFAULT_NUM_CTX,
	DEFAULT_OLLAMA_TAG,
	DEFAULT_VLLM_BASE_URL,
	CommandResult,
	OFFICIAL_QUANTIZED_MODELS,
	QuantStartResult,
	Qwen3827BQuant,
	StartReport,
)

logger = logging.getLogger(__name__)

RunFn = Callable[..., Awaitable[CommandResult]]
WhichFn = Callable[[str], str | None]
HealthFn = Callable[[], bool | Awaitable[bool]]
CudaFn = Callable[[], bool]

_QWEN38_NAME_RE = re.compile(r'qwen3[\._-]?8')
_FOLD_RE = re.compile(r'[^a-z0-9]+')

_TAG_BY_FOLDED: dict[str, str] = {
	'default': DEFAULT_OLLAMA_TAG,
	'latest': DEFAULT_OLLAMA_TAG,
	'27b': DEFAULT_OLLAMA_TAG,
	'qwen3827b': DEFAULT_OLLAMA_TAG,
	'qwen38': DEFAULT_OLLAMA_TAG,
	'qwen3827blatest': DEFAULT_OLLAMA_TAG,
	'q4': 'qwen3.8:27b-q4_K_M',
	'q4km': 'qwen3.8:27b-q4_K_M',
	'q4k': 'qwen3.8:27b-q4_K_M',
	'qwen3827bq4': 'qwen3.8:27b-q4_K_M',
	'qwen3827bq4km': 'qwen3.8:27b-q4_K_M',
	'q8': 'qwen3.8:27b-q8_0',
	'q80': 'qwen3.8:27b-q8_0',
	'qwen3827bq8': 'qwen3.8:27b-q8_0',
	'qwen3827bq80': 'qwen3.8:27b-q8_0',
	'mtp': 'qwen3.8:27b-mtp-q4_K_M',
	'mtpq4': 'qwen3.8:27b-mtp-q4_K_M',
	'mtpq4km': 'qwen3.8:27b-mtp-q4_K_M',
	'qwen3827bmtp': 'qwen3.8:27b-mtp-q4_K_M',
	'qwen3827bmtpq4km': 'qwen3.8:27b-mtp-q4_K_M',
	'mtpq8': 'qwen3.8:27b-mtp-q8_0',
	'mtpq80': 'qwen3.8:27b-mtp-q8_0',
	'qwen3827bmtpq80': 'qwen3.8:27b-mtp-q8_0',
	'mlx': 'qwen3.8:27b-mlx',
	'qwen3827bmlx': 'qwen3.8:27b-mlx',
	'mxfp8': 'qwen3.8:27b-mxfp8',
	'qwen3827bmxfp8': 'qwen3.8:27b-mxfp8',
	'nvfp4': 'qwen3.8:27b-nvfp4',
	'qwen3827bnvfp4': 'qwen3.8:27b-nvfp4',
	'fp8': 'Qwen/Qwen3.8-27B-FP8',
	'qwen3827bfp8': 'Qwen/Qwen3.8-27B-FP8',
}

for _spec in OFFICIAL_QUANTIZED_MODELS:
	_TAG_BY_FOLDED[_FOLD_RE.sub('', _spec.tag.lower())] = _spec.tag
_TAG_BY_FOLDED[_FOLD_RE.sub('', DEFAULT_OLLAMA_TAG.lower())] = DEFAULT_OLLAMA_TAG


def _fold(name: str) -> str:
	return _FOLD_RE.sub('', name.strip().lower())


def _log_skip(tag: str, reason: str) -> None:
	logger.info('skip %s (%s)', tag, reason)


def _log_pull(tag: str) -> None:
	logger.info('pull %s', tag)


def _log_serve(message: str) -> None:
	logger.info('%s', message)


def is_qwen38_27b_name(name: str) -> bool:
	"""True when a factory/model string refers to local Qwen3.8-27B."""
	folded = _fold(name)
	if folded.startswith('ollama'):
		folded = folded[len('ollama') :]
	if folded in _TAG_BY_FOLDED:
		return True
	return bool(_QWEN38_NAME_RE.search(name.lower())) and '27' in folded


def resolve_quant_tag(name: str) -> str:
	"""Map a human alias (`q4`, `ollama_qwen38_27b_q8_0`) to a concrete tag."""
	assert name.strip(), 'quant name must be non-empty'
	raw = name.strip()
	if raw == DEFAULT_OLLAMA_TAG or raw in {spec.tag for spec in OFFICIAL_QUANTIZED_MODELS}:
		return raw

	folded = _fold(raw)
	if folded.startswith('ollama'):
		folded = folded[len('ollama') :]
	if folded in _TAG_BY_FOLDED:
		tag = _TAG_BY_FOLDED[folded]
		assert tag
		return tag
	raise ValueError(
		f"Unknown Qwen3.8-27B quant '{name}'. "
		f'Known: default, q4_K_M, q8_0, mtp-q4_K_M, mtp-q8_0, mlx, mxfp8, nvfp4, fp8'
	)


def resolve_quant(name: str) -> Qwen3827BQuant:
	"""Resolve an alias to the catalog entry (the default alias is MTP Q4_K_M)."""
	tag = resolve_quant_tag(name)
	if tag == DEFAULT_OLLAMA_TAG:
		recommended = next(spec for spec in OFFICIAL_QUANTIZED_MODELS if spec.recommended)
		return recommended.model_copy(update={'tag': DEFAULT_OLLAMA_TAG})
	match = next((spec for spec in OFFICIAL_QUANTIZED_MODELS if spec.tag == tag), None)
	if match is None:
		raise ValueError(f"Resolved tag '{tag}' is not in the official catalog")
	return match


def select_quants_for_platform(platform_name: str, *, include_mlx: bool | None = None) -> list[Qwen3827BQuant]:
	"""Official quants that can actually run on this OS."""
	assert platform_name, 'platform_name is required'
	want_mlx = platform_name == 'darwin' if include_mlx is None else include_mlx
	selected: list[Qwen3827BQuant] = []
	for spec in OFFICIAL_QUANTIZED_MODELS:
		if platform_name not in spec.platforms:
			continue
		if spec.runtime == 'mlx' and not want_mlx:
			continue
		selected.append(spec)
	assert selected, f'no official Qwen3.8-27B quants for platform {platform_name}'
	return selected


def chat_qwen38_27b(
	quant: str = 'default',
	*,
	host: str | None = None,
	think: bool = False,
	num_ctx: int = DEFAULT_NUM_CTX,
	keep_alive: str = '30m',
	base_url: str | None = None,
	api_key: str | None = None,
) -> BaseChatModel:
	"""Chat client pointed at an official local Qwen3.8-27B quant.

	Thinking is off by default so browser-use structured action JSON stays intact.
	"""
	assert num_ctx > 0, 'num_ctx must be positive'
	spec = resolve_quant(quant)
	if spec.source == 'huggingface':
		return ChatOpenAI(
			model=spec.tag,
			base_url=base_url or os.getenv('QWEN38_VLLM_BASE_URL', DEFAULT_VLLM_BASE_URL),
			api_key=api_key or os.getenv('QWEN38_VLLM_API_KEY', 'local'),
			temperature=0.7,
		)
	return ChatOllama(
		model=resolve_quant_tag(quant),
		host=host,
		ollama_options={
			'think': think,
			'num_ctx': num_ctx,
			'keep_alive': keep_alive,
			'temperature': 0.7,
			'top_p': 0.8,
		},
	)


def _default_cuda_available() -> bool:
	return shutil.which('nvidia-smi') is not None


class Qwen3827BLocalRuntime:
	"""Installs/starts Ollama, pulls every compatible official quant, optionally vLLM FP8."""

	def __init__(
		self,
		*,
		platform_name: str | None = None,
		which: WhichFn = shutil.which,
		run: RunFn | None = None,
		cuda_available: CudaFn | None = None,
		healthcheck: HealthFn | None = None,
		host: str = 'http://127.0.0.1:11434',
	) -> None:
		self.platform_name = platform_name or sys.platform
		self.which = which
		self._run = run or self._run_command
		self.cuda_available = cuda_available or _default_cuda_available
		self._healthcheck = healthcheck
		self.host = host

	async def _run_command(self, argv: list[str], timeout: float | None = None) -> CommandResult:
		assert argv, 'command argv must be non-empty'
		proc = await asyncio.create_subprocess_exec(
			*argv,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.STDOUT,
		)
		try:
			stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
		except TimeoutError:
			proc.kill()
			await proc.wait()
			return CommandResult(returncode=124, stdout='', stderr='timeout')
		text = (stdout_bytes or b'').decode('utf-8', errors='replace')
		return CommandResult(returncode=proc.returncode or 0, stdout=text, stderr='')

	async def _is_ollama_up(self) -> bool:
		if self._healthcheck is not None:
			result = self._healthcheck()
			if asyncio.iscoroutine(result):
				return bool(await result)
			return bool(result)
		try:
			proc = await asyncio.create_subprocess_exec(
				'curl',
				'-fsS',
				f'{self.host}/api/tags',
				stdout=asyncio.subprocess.DEVNULL,
				stderr=asyncio.subprocess.DEVNULL,
			)
			return (await proc.wait()) == 0
		except OSError:
			return False

	async def ensure_ollama(self) -> str:
		"""Return the ollama binary path, installing it on Linux if missing."""
		path = self.which('ollama')
		if path:
			return path
		if self.platform_name.startswith('linux'):
			install = await self._run(['bash', '-lc', 'curl -fsSL https://ollama.com/install.sh | sh'])
			if install.returncode != 0:
				raise RuntimeError(f'Failed to install Ollama: {install.stdout}')
			path = self.which('ollama')
			if path:
				return path
		raise RuntimeError('Ollama is not installed. Install it from https://ollama.com/download and re-run.')

	async def ensure_ollama_running(self) -> bool:
		if await self._is_ollama_up():
			_log_serve(f'ollama already running at {self.host}')
			return True
		return False

	def _ollama_start_list(self, *, include_mlx: bool | None) -> list[str]:
		tags = [DEFAULT_OLLAMA_TAG]
		for spec in select_quants_for_platform(self.platform_name, include_mlx=include_mlx):
			if spec.source != 'ollama':
				continue
			if spec.tag not in tags:
				tags.append(spec.tag)
		return tags

	async def start_all(
		self,
		*,
		dry_run: bool = False,
		include_mlx: bool | None = None,
		load_default: bool = True,
	) -> StartReport:
		"""Pull every official quantized 27B checkpoint that can run here, then load the default."""
		assert include_mlx in (True, False, None)
		results: list[QuantStartResult] = []
		ollama_tags = self._ollama_start_list(include_mlx=include_mlx)

		if dry_run:
			results.extend(QuantStartResult(tag=tag, action='planned') for tag in ollama_tags)
			fp8 = next(spec for spec in OFFICIAL_QUANTIZED_MODELS if spec.runtime == 'vllm')
			if self.cuda_available() and self.platform_name in fp8.platforms:
				results.append(QuantStartResult(tag=fp8.tag, action='planned', reason='vllm'))
			else:
				results.append(
					QuantStartResult(
						tag=fp8.tag,
						action='skipped',
						reason='CUDA/vLLM not available on this host',
					)
				)
			for spec in OFFICIAL_QUANTIZED_MODELS:
				if spec.runtime == 'mlx' and spec.tag not in ollama_tags:
					results.append(QuantStartResult(tag=spec.tag, action='skipped', reason='mlx requires macOS'))
			report = StartReport(ollama_running=False, results=results)
			assert report.results
			return report

		running = await self.ensure_ollama_running()
		binary = await self.ensure_ollama()

		if not running:
			_log_serve(f'starting {binary} serve')
			await asyncio.create_subprocess_exec(
				binary,
				'serve',
				stdout=asyncio.subprocess.DEVNULL,
				stderr=asyncio.subprocess.DEVNULL,
			)
			for _ in range(50):
				if await self._is_ollama_up():
					running = True
					break
				await asyncio.sleep(0.2)

		for tag in ollama_tags:
			_log_pull(tag)
			pulled = await self._run([binary, 'pull', tag])
			if pulled.returncode != 0:
				results.append(QuantStartResult(tag=tag, action='failed', reason=pulled.stdout[-2000:] or 'ollama pull failed'))
				continue
			action: Literal['pulled', 'loaded'] = 'pulled'
			if load_default and tag == DEFAULT_OLLAMA_TAG:
				payload = json.dumps({'model': tag, 'prompt': 'ok', 'keep_alive': '30m'})
				loaded = await self._run(['curl', '-fsS', f'{self.host}/api/generate', '-d', payload])
				if loaded.returncode == 0:
					action = 'loaded'
			results.append(QuantStartResult(tag=tag, action=action))

		fp8 = next(spec for spec in OFFICIAL_QUANTIZED_MODELS if spec.runtime == 'vllm')
		if self.cuda_available() and self.platform_name in fp8.platforms and self.which('vllm'):
			_log_serve(f'starting vllm serve {fp8.tag}')
			await asyncio.create_subprocess_exec(
				'vllm',
				'serve',
				fp8.tag,
				'--host',
				'127.0.0.1',
				'--port',
				'8000',
				stdout=asyncio.subprocess.DEVNULL,
				stderr=asyncio.subprocess.DEVNULL,
			)
			results.append(QuantStartResult(tag=fp8.tag, action='loaded', reason='vllm'))
		else:
			_log_skip(fp8.tag, 'CUDA/vLLM not available on this host')
			results.append(QuantStartResult(tag=fp8.tag, action='skipped', reason='CUDA/vLLM not available on this host'))

		for spec in OFFICIAL_QUANTIZED_MODELS:
			if spec.runtime == 'mlx' and spec.tag not in ollama_tags:
				_log_skip(spec.tag, 'mlx requires macOS')
				results.append(QuantStartResult(tag=spec.tag, action='skipped', reason='mlx requires macOS'))

		report = StartReport(ollama_running=running or await self._is_ollama_up(), results=results)
		assert report.results, 'start_all produced no results'
		return report


def format_start_report(report: StartReport) -> str:
	lines = [f'ollama_running={report.ollama_running}']
	for result in report.results:
		suffix = f' ({result.reason})' if result.reason else ''
		lines.append(f'{result.action:8} {result.tag}{suffix}')
	return '\n'.join(lines)
