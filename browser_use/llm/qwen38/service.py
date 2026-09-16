"""Local Qwen3.8-27B fleet: refresh official weights metadata and start every GGUF quant.

Pulls the live Hugging Face tree for Qwen/Qwen3.8-27B and bartowski's llama.cpp
imatrix GGUFs, then builds llama-server argv for each language-trunk quant so a
developer machine can serve them as OpenAI-compatible endpoints.
"""

# @file purpose: Fetches latest Qwen3.8-27B quants and starts local llama-server fleet

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import httpx
import psutil

from browser_use.llm.qwen38.views import (
	DEFAULT_BASE_PORT,
	FALLBACK_QUANT_FILES,
	MIN_LLAMA_CPP_RELEASE,
	MMPROJ_FILES,
	ORIGINAL_MODEL_ID,
	ORIGINAL_REVISION,
	QUANT_QUALITY_ORDER,
	QUANT_REPO_ID,
	RECOMMENDED_QUANTS,
	Qwen38Catalog,
	Qwen38Fleet,
	Qwen38Hardware,
	Qwen38Mmproj,
	Qwen38Quant,
	Qwen38ServerSpec,
)

_HF_API = 'https://huggingface.co/api/models'
_LANGUAGE_PREFIX = 'Qwen3.8-27B-'
_LANGUAGE_SUFFIX = '.gguf'
_LEGACY_QUANTS = frozenset({'Q5_K_L', 'Q3_K_XL', 'Q2_K_L'})


def _log_pretty_size(size_bytes: int) -> str:
	gib = size_bytes / (1024**3)
	return f'{gib:.2f} GiB'


def _log_quant_line(quant: Qwen38Quant) -> str:
	flag = ' recommended' if quant.recommended else ''
	legacy = ' legacy' if quant.legacy else ''
	return f'{quant.quant:10}  port={quant.port}  {_log_pretty_size(quant.size_bytes)}{flag}{legacy}'


def quant_name_from_filename(filename: str) -> str | None:
	"""Return the quant tag from a bartowski language GGUF filename, or None."""
	name = Path(filename).name
	if not name.startswith(_LANGUAGE_PREFIX) or not name.endswith(_LANGUAGE_SUFFIX):
		return None
	if name.startswith('mmproj-') or 'imatrix' in name.lower():
		return None
	# Skip bf16 directory shards; those live under Qwen3.8-27B-bf16/.
	stem = name[len(_LANGUAGE_PREFIX) : -len(_LANGUAGE_SUFFIX)]
	if not stem or '/' in name:
		return None
	if stem.lower() in {'bf16', 'f16', 'f32', 'calibration-v6.txt'}:
		return None
	return stem


def _quality_index(quant: str) -> int:
	try:
		return QUANT_QUALITY_ORDER.index(quant)
	except ValueError:
		return len(QUANT_QUALITY_ORDER)


def detect_hardware() -> Qwen38Hardware:
	"""Read RAM and optional NVIDIA VRAM for load-fit checks."""
	ram_bytes = int(psutil.virtual_memory().total)
	vram_bytes = 0
	gpu_layers = 0
	nvidia_smi = shutil.which('nvidia-smi')
	if nvidia_smi is not None:
		# Best-effort; absence of nvidia-smi just means CPU-only.
		import subprocess

		try:
			proc = subprocess.run(
				[nvidia_smi, '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
				check=False,
				capture_output=True,
				text=True,
				timeout=5,
			)
			if proc.returncode == 0:
				mb = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()]
				if mb:
					vram_bytes = sum(mb) * 1024 * 1024
					gpu_layers = 99
		except (OSError, ValueError, subprocess.TimeoutExpired):
			vram_bytes = 0
			gpu_layers = 0
	return Qwen38Hardware(ram_bytes=ram_bytes, vram_bytes=vram_bytes, gpu_layers=gpu_layers)


def fits_hardware(size_bytes: int, hardware: Qwen38Hardware) -> bool:
	"""True if the GGUF plus KV-cache headroom can map into RAM+VRAM."""
	assert size_bytes > 0
	return size_bytes <= hardware.usable_bytes


def find_llama_server(explicit: str | None = None) -> str | None:
	"""Resolve llama-server from an explicit path, PATH, or llama.app install dirs."""
	if explicit:
		path = Path(explicit).expanduser()
		return str(path) if path.is_file() and os.access(path, os.X_OK) else None
	which = shutil.which('llama-server')
	if which:
		return which
	home = Path.home()
	candidates = [
		home / '.llama' / 'bin' / 'llama-server',
		home / '.local' / 'bin' / 'llama-server',
		Path('/usr/local/bin/llama-server'),
	]
	for candidate in candidates:
		if candidate.is_file() and os.access(candidate, os.X_OK):
			return str(candidate)
	return None


def catalog_from_hf_tree(
	tree: list[dict[str, object]],
	*,
	original_sha: str = ORIGINAL_REVISION,
	source: str = 'huggingface',
	base_port: int = DEFAULT_BASE_PORT,
) -> Qwen38Catalog:
	"""Build a catalog from a Hugging Face `/tree/main` JSON payload."""
	assert isinstance(tree, list)
	quants: list[Qwen38Quant] = []
	mmproj: list[Qwen38Mmproj] = []
	for entry in tree:
		if not isinstance(entry, dict):
			continue
		path = str(entry.get('path') or '')
		entry_type = str(entry.get('type') or 'file')
		raw_size = entry.get('size') or 0
		size = int(raw_size) if isinstance(raw_size, int | float | str) else 0
		name = Path(path).name
		if entry_type != 'file' or '/' in path:
			continue
		if name in MMPROJ_FILES or name.startswith('mmproj-'):
			mmproj.append(Qwen38Mmproj(filename=name, size_bytes=size))
			continue
		quant = quant_name_from_filename(name)
		if quant is None:
			continue
		quants.append(
			Qwen38Quant(
				quant=quant,
				filename=name,
				size_bytes=size,
				recommended=quant in RECOMMENDED_QUANTS,
				legacy=quant in _LEGACY_QUANTS,
				port=DEFAULT_BASE_PORT,
			)
		)
	assert quants, 'Hugging Face tree contained no Qwen3.8-27B language GGUFs'
	quants.sort(key=lambda item: (_quality_index(item.quant), item.quant))
	assigned: list[Qwen38Quant] = []
	for index, item in enumerate(quants):
		assigned.append(item.model_copy(update={'port': base_port + index}))
	assert len({item.port for item in assigned}) == len(assigned)
	assert len({item.quant for item in assigned}) == len(assigned)
	return Qwen38Catalog(
		original_repo=ORIGINAL_MODEL_ID,
		original_sha=original_sha,
		quant_repo=QUANT_REPO_ID,
		min_llama_cpp_release=MIN_LLAMA_CPP_RELEASE,
		source='huggingface' if source == 'huggingface' else 'fallback',
		quants=assigned,
		mmproj=mmproj,
	)


def fallback_catalog(base_port: int = DEFAULT_BASE_PORT) -> Qwen38Catalog:
	"""Offline catalog pinned to the bartowski tree snapshot in views.py."""
	tree = [{'path': name, 'type': 'file', 'size': size} for name, size in FALLBACK_QUANT_FILES.items()]
	for name in MMPROJ_FILES:
		tree.append({'path': name, 'type': 'file', 'size': 0})
	return catalog_from_hf_tree(tree, original_sha=ORIGINAL_REVISION, source='fallback', base_port=base_port)


def llama_server_argv(
	quant: Qwen38Quant,
	*,
	llama_server_bin: str,
	hardware: Qwen38Hardware,
	host: str = '127.0.0.1',
	ctx_size: int | None = None,
	mmproj: str | None = None,
	use_mtp: bool = True,
) -> list[str]:
	"""llama.cpp argv for one quant. Requires b10896+ for the qwen35 hybrid graph."""
	assert llama_server_bin
	assert quant.port > 0
	n_ctx = ctx_size if ctx_size is not None else (32768 if hardware.gpu_layers else 8192)
	n_gpu_layers = hardware.gpu_layers
	argv = [
		llama_server_bin,
		'-hf',
		quant.hf_spec,
		'--host',
		host,
		'--port',
		str(quant.port),
		'-a',
		quant.alias,
		'--jinja',
		'-c',
		str(n_ctx),
		'-ngl',
		str(n_gpu_layers),
		'-np',
		'1',
		'--reasoning-format',
		'deepseek',
	]
	if use_mtp:
		argv.extend(['--spec-type', 'draft-mtp'])
	if mmproj:
		argv.extend(['--mmproj', mmproj])
	return argv


class Qwen38LocalService:
	"""Refresh the official Qwen3.8-27B catalog and start every quantized endpoint."""

	def __init__(
		self,
		*,
		cache_dir: Path | None = None,
		llama_server_bin: str | None = None,
		host: str = '127.0.0.1',
		base_port: int = DEFAULT_BASE_PORT,
		timeout: float = 30.0,
		hf_api_base: str = _HF_API,
	) -> None:
		self.cache_dir = cache_dir or (Path.home() / '.cache' / 'browser-use' / 'qwen38')
		self.llama_server_bin = llama_server_bin
		self.host = host
		self.base_port = base_port
		self.timeout = timeout
		self.hf_api_base = hf_api_base.rstrip('/')
		self._processes: dict[str, asyncio.subprocess.Process] = {}

	async def refresh_catalog(self, client: httpx.AsyncClient | None = None) -> Qwen38Catalog:
		"""Fetch the live official SHA and bartowski GGUF tree. Falls back if HF is down."""
		owns_client = client is None
		http = client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
		try:
			model_resp = await http.get(f'{self.hf_api_base}/{ORIGINAL_MODEL_ID}')
			model_resp.raise_for_status()
			model_payload = model_resp.json()
			sha = str(model_payload.get('sha') or ORIGINAL_REVISION)
			tree_resp = await http.get(f'{self.hf_api_base}/{QUANT_REPO_ID}/tree/main')
			tree_resp.raise_for_status()
			tree = tree_resp.json()
			assert isinstance(tree, list)
			catalog = catalog_from_hf_tree(tree, original_sha=sha, source='huggingface', base_port=self.base_port)
			assert catalog.original_sha
			assert catalog.quants
			return catalog
		except (httpx.HTTPError, AssertionError, KeyError, ValueError, TypeError):
			return fallback_catalog(base_port=self.base_port)
		finally:
			if owns_client:
				await http.aclose()

	def plan_fleet(
		self,
		catalog: Qwen38Catalog,
		*,
		hardware: Qwen38Hardware | None = None,
		force: bool = False,
		ctx_size: int | None = None,
		use_mtp: bool = True,
	) -> Qwen38Fleet:
		"""Build llama-server specs for every language quant in the catalog."""
		hw = hardware or detect_hardware()
		bin_path = find_llama_server(self.llama_server_bin)
		mmproj_name = next((item.filename for item in catalog.mmproj if item.filename.endswith('f16.gguf')), None)
		servers: list[Qwen38ServerSpec] = []
		remaining = hw.usable_bytes
		for quant in catalog.language_quants():
			fits_alone = fits_hardware(quant.size_bytes, hw)
			fits_remaining = quant.size_bytes <= remaining
			fits = force or (fits_alone and fits_remaining)
			skip_reason: str | None = None
			if bin_path is None:
				skip_reason = 'llama-server not found (install llama.cpp b10896+ / llama.app)'
			elif not fits:
				skip_reason = (
					f'{quant.filename} is {_log_pretty_size(quant.size_bytes)}; '
					f'{_log_pretty_size(remaining)} remaining of {_log_pretty_size(hw.usable_bytes)} usable RAM+VRAM'
				)
			else:
				remaining = max(0, remaining - quant.size_bytes)
			argv = (
				llama_server_argv(
					quant,
					llama_server_bin=bin_path or 'llama-server',
					hardware=hw,
					host=self.host,
					ctx_size=ctx_size,
					mmproj=mmproj_name,
					use_mtp=use_mtp,
				)
				if bin_path
				else []
			)
			servers.append(
				Qwen38ServerSpec(
					quant=quant,
					argv=argv,
					host=self.host,
					fits_hardware=fits,
					skip_reason=skip_reason,
				)
			)
		fleet = Qwen38Fleet(catalog=catalog, servers=servers, llama_server_bin=bin_path)
		assert len(fleet.servers) == len(catalog.quants)
		return fleet

	def write_launch_scripts(self, fleet: Qwen38Fleet) -> Path:
		"""Write a start-all shell script the user can run on a GPU box."""
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		script = self.cache_dir / 'start-all-qwen38-quants.sh'
		lines = [
			'#!/usr/bin/env bash',
			'set -euo pipefail',
			f'# Qwen3.8-27B original: {fleet.catalog.original_repo}@{fleet.catalog.original_sha}',
			f'# GGUFs: {fleet.catalog.quant_repo} (llama.cpp {fleet.catalog.min_llama_cpp_release}+)',
			f'# Catalog source: {fleet.catalog.source}',
			'',
		]
		for spec in fleet.servers:
			comment = ''
			if spec.skip_reason:
				comment = f'  # skipped here: {spec.skip_reason}'
			quoted = (
				' '.join(_shell_quote(part) for part in spec.argv)
				if spec.argv
				else f'# missing llama-server for {spec.quant.quant}'
			)
			lines.append(f'{quoted} &{comment}')
		lines.append('wait')
		script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
		script.chmod(0o755)
		return script

	async def start_all(
		self,
		*,
		catalog: Qwen38Catalog | None = None,
		hardware: Qwen38Hardware | None = None,
		force: bool = False,
		spawn: bool = True,
		ctx_size: int | None = None,
	) -> Qwen38Fleet:
		"""Refresh if needed, write launch scripts, and spawn every quant that fits."""
		resolved = catalog or await self.refresh_catalog()
		fleet = self.plan_fleet(catalog=resolved, hardware=hardware, force=force, ctx_size=ctx_size)
		self.write_launch_scripts(fleet)
		if not spawn:
			return fleet
		for spec in fleet.servers:
			if spec.skip_reason or not spec.argv:
				continue
			await self._spawn(spec)
		return fleet

	async def _spawn(self, spec: Qwen38ServerSpec) -> None:
		assert spec.argv
		proc = await asyncio.create_subprocess_exec(
			*spec.argv,
			stdout=asyncio.subprocess.DEVNULL,
			stderr=asyncio.subprocess.DEVNULL,
			start_new_session=True,
		)
		self._processes[spec.quant.quant] = proc

	async def stop_all(self) -> None:
		for proc in self._processes.values():
			if proc.returncode is None:
				proc.terminate()
		await asyncio.gather(*[proc.wait() for proc in self._processes.values()], return_exceptions=True)
		self._processes.clear()


def _shell_quote(value: str) -> str:
	if not value:
		return "''"
	if all(ch.isalnum() or ch in '._-:=/@+' for ch in value):
		return value
	return "'" + value.replace("'", "'\\''") + "'"


def _log_fleet_summary(fleet: Qwen38Fleet) -> str:
	started = fleet.started()
	return (
		f'Qwen3.8-27B {fleet.catalog.original_sha[:12]}  '
		f'{len(fleet.servers)} quants, {len(started)} spawning, '
		f'llama-server={fleet.llama_server_bin or "missing"}'
	)


if __name__ == '__main__':
	# Keep `python -m browser_use.llm.qwen38.service` useful for a one-shot plan dump.
	async def _main() -> None:
		service = Qwen38LocalService()
		fleet = await service.start_all(spawn=False)
		print(_log_fleet_summary(fleet))
		for spec in fleet.servers:
			print(_log_quant_line(spec.quant), spec.skip_reason or spec.base_url)

	asyncio.run(_main())
	sys.exit(0)
