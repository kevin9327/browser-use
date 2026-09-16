"""Download official Qwen3.8-27B, quantize it, and serve every local GGUF/Ollama tag.

# @file purpose: Local download/quantize/serve pipeline for Qwen3.8-27B
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

from .catalog import (
	COVERAGE_KEYS,
	DEFAULT_PINNED_KEYS,
	LLAMA_CPP_RELEASE,
	LLAMA_QUANT_TYPES,
	MMPROJ_FILES,
	OFFICIAL_REPO,
	OLLAMA_TAGS,
	ORIGINAL_TARGET,
	UNSLOTH_GGUFS,
	UNSLOTH_REPO,
	all_download_targets,
	default_serve_quant,
	fit_by_bytes,
	self_quant_targets,
)
from .views import BackendName, DownloadTarget, HardwareInfo, StartPlan

DEFAULT_MODELS_DIR = Path.home() / '.cache' / 'browser-use' / 'qwen3.8-27b'
LLAMA_SERVER_PORT = 8080
OLLAMA_PORT = 11434
GITHUB_LLAMA_RELEASE = f'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}'


def detect_hardware(models_dir: Path) -> HardwareInfo:
	models_dir.mkdir(parents=True, exist_ok=True)
	ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') if hasattr(os, 'sysconf') else 0
	disk_free_bytes = shutil.disk_usage(models_dir).free
	has_nvidia = shutil.which('nvidia-smi') is not None
	return HardwareInfo(
		cpu_count=os.cpu_count() or 1,
		ram_bytes=int(ram_bytes),
		disk_free_bytes=int(disk_free_bytes),
		has_nvidia=has_nvidia,
		platform=sys.platform,
	)


def _log_hardware(hardware: HardwareInfo) -> None:
	gpu = 'NVIDIA' if hardware.has_nvidia else 'none'
	print(
		f'[qwen38] host  cpu={hardware.cpu_count}  ram={hardware.ram_gb:.1f}GiB  '
		f'disk_free={hardware.disk_free_gb:.1f}GiB  gpu={gpu}  os={hardware.platform}'
	)


def _log_target(prefix: str, target: DownloadTarget) -> None:
	print(f'[qwen38] {prefix}  {target.key:18}  {target.size_gb:6.1f}GB  {target.label}')


def _log_pretty_path(path: Path) -> str:
	try:
		return str(path.expanduser().resolve())
	except OSError:
		return str(path)


def default_models_dir() -> Path:
	override = os.environ.get('QWEN38_MODELS_DIR')
	return Path(override).expanduser() if override else DEFAULT_MODELS_DIR


def build_start_plan(
	hardware: HardwareInfo,
	*,
	models_dir: Path,
	backend: BackendName = 'both',
	include_self_quant: bool = True,
	force_all: bool = False,
	force_self_quant: bool = False,
) -> StartPlan:
	"""Decide which original/quant artifacts to fetch and which GGUFs to serve."""
	linux_only = hardware.platform.startswith('linux')
	notes: list[str] = []
	wanted: list[DownloadTarget] = []

	if backend in ('llama', 'both'):
		wanted.append(ORIGINAL_TARGET)
		wanted.extend(UNSLOTH_GGUFS)
		wanted.extend(MMPROJ_FILES)
		if include_self_quant:
			if hardware.ram_bytes >= 32 * 1024**3 or force_all or force_self_quant:
				wanted.extend(self_quant_targets())
				if hardware.ram_bytes < 32 * 1024**3:
					notes.append('Self-quantization forced on <32GiB RAM; convert_hf_to_gguf may OOM.')
			else:
				notes.append(
					'Self-quantization from official BF16 needs ~32GiB RAM; '
					'downloading Unsloth GGUFs instead. Pass --force-self-quant to convert anyway.'
				)

	if backend in ('ollama', 'both'):
		wanted.extend([tag for tag in OLLAMA_TAGS if (tag.linux_ok or not linux_only)])

	if force_all:
		download = [target for target in wanted if target.linux_ok or not linux_only]
		skipped = [target for target in wanted if target.key not in {item.key for item in download}]
		notes.append('force_all=True: disk/RAM filters disabled. This can fill the disk or OOM.')
	else:
		download, skipped = fit_by_bytes(
			wanted,
			hardware.disk_free_bytes,
			pinned_keys=DEFAULT_PINNED_KEYS,
			priority_keys=COVERAGE_KEYS,
		)

	serve_ggufs = [
		target.filename
		for target in download
		if target.kind in {'unsloth-gguf', 'self-quant'} and target.filename and target.quant_type != 'BF16'
	]
	if not serve_ggufs:
		fallback = default_serve_quant(hardware.ram_bytes)
		serve_ggufs = [fallback.filename or fallback.key]
		notes.append(f'No GGUF fitted the disk budget; will still try {fallback.key}.')

	ollama_pull = [target.ollama_tag for target in download if target.kind == 'ollama' and target.ollama_tag]
	self_quant_types = [
		target.quant_type
		for target in download
		if target.kind == 'self-quant' and target.quant_type and target.quant_type != 'BF16'
	]

	if hardware.ram_bytes < 16 * 1024**3:
		notes.append(
			f'This host has {hardware.ram_gb:.1f}GiB RAM. A 27B quant will page hard on CPU. '
			'Run the same command on a 24GB+ GPU box for real local use.'
		)

	return StartPlan(
		hardware=hardware,
		models_dir=models_dir,
		backend=backend,
		download=download,
		skipped=skipped,
		self_quant_types=[item for item in self_quant_types if item is not None],
		serve_ggufs=[item for item in serve_ggufs if item is not None],
		ollama_pull=ollama_pull,
		llama_port=LLAMA_SERVER_PORT,
		ollama_port=OLLAMA_PORT,
		notes=notes,
	)


def _which_or_cache(name: str, cache_bin: Path) -> Path | None:
	found = shutil.which(name)
	if found:
		return Path(found)
	candidate = cache_bin / name
	return candidate if candidate.exists() else None


def _llama_asset_name(hardware: HardwareInfo) -> str:
	system = hardware.platform
	machine = platform.machine().lower()
	if system == 'darwin':
		arch = 'arm64' if machine in {'arm64', 'aarch64'} else 'x64'
		return f'llama-{LLAMA_CPP_RELEASE}-bin-macos-{arch}.tar.gz'
	if machine in {'arm64', 'aarch64'}:
		return f'llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-arm64.tar.gz'
	if hardware.has_nvidia:
		# CPU tarball still works; CUDA users should install a CUDA build themselves.
		return f'llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz'
	return f'llama-{LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz'


def ensure_llama_cpp(models_dir: Path, hardware: HardwareInfo, *, dry_run: bool = False) -> dict[str, Path]:
	"""Return paths to llama-server / llama-quantize, installing a GitHub release if needed."""
	cache_bin = models_dir / 'tools' / 'llama.cpp' / 'bin'
	cache_src = models_dir / 'tools' / 'llama.cpp' / 'src'
	server = _which_or_cache('llama-server', cache_bin)
	quantize = _which_or_cache('llama-quantize', cache_bin)
	if server and quantize:
		return {'server': server, 'quantize': quantize, 'src': cache_src}

	asset = _llama_asset_name(hardware)
	url = f'{GITHUB_LLAMA_RELEASE}/{asset}'
	archive = models_dir / 'tools' / asset
	print(f'[qwen38] installing llama.cpp {LLAMA_CPP_RELEASE} from {url}')
	if dry_run:
		return {
			'server': cache_bin / 'llama-server',
			'quantize': cache_bin / 'llama-quantize',
			'src': cache_src,
		}

	archive.parent.mkdir(parents=True, exist_ok=True)
	cache_bin.mkdir(parents=True, exist_ok=True)
	if not archive.exists():
		urlretrieve(url, archive)
	subprocess.run(['tar', '-xzf', str(archive), '-C', str(cache_bin.parent)], check=True)

	# Release tarballs unpack either into bin/ or a nested folder of binaries.
	for binary in ('llama-server', 'llama-quantize'):
		matches = list(cache_bin.parent.rglob(binary))
		if not matches:
			raise FileNotFoundError(f'{binary} missing from {archive}')
		dest = cache_bin / binary
		if matches[0].resolve() != dest.resolve():
			dest.write_bytes(matches[0].read_bytes())
			dest.chmod(0o755)

	if not cache_src.exists():
		print('[qwen38] cloning llama.cpp (convert_hf_to_gguf.py)')
		subprocess.run(
			[
				'git',
				'clone',
				'--depth',
				'1',
				'--branch',
				LLAMA_CPP_RELEASE,
				'https://github.com/ggml-org/llama.cpp',
				str(cache_src),
			],
			check=False,
		)
		if not cache_src.exists():
			subprocess.run(
				['git', 'clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', str(cache_src)],
				check=True,
			)

	server = cache_bin / 'llama-server'
	quantize = cache_bin / 'llama-quantize'
	assert server.exists() and quantize.exists()
	return {'server': server, 'quantize': quantize, 'src': cache_src}


def _ensure_huggingface_hub() -> None:
	try:
		import huggingface_hub  # noqa: F401
	except ImportError:
		installer = (
			['uv', 'pip', 'install', 'huggingface_hub']
			if shutil.which('uv')
			else [sys.executable, '-m', 'pip', 'install', 'huggingface_hub']
		)
		print(f'[qwen38] installing huggingface_hub via {installer[0]}')
		subprocess.run(installer, check=True)


def download_original(models_dir: Path, *, dry_run: bool = False) -> Path:
	dest = models_dir / 'original' / 'Qwen3.8-27B'
	print(f'[qwen38] original  {OFFICIAL_REPO} -> {_log_pretty_path(dest)}')
	if dry_run:
		return dest
	_ensure_huggingface_hub()
	from huggingface_hub import snapshot_download

	snapshot_download(repo_id=OFFICIAL_REPO, local_dir=str(dest))
	return dest


def download_unsloth_artifacts(targets: list[DownloadTarget], models_dir: Path, *, dry_run: bool = False) -> list[Path]:
	gguf_dir = models_dir / 'gguf'
	gguf_dir.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	hf_hub_download = None
	if not dry_run:
		_ensure_huggingface_hub()
		from huggingface_hub import hf_hub_download as _hf_hub_download

		hf_hub_download = _hf_hub_download
	for target in targets:
		if target.kind not in {'unsloth-gguf', 'mmproj'} or not target.filename:
			continue
		dest = gguf_dir / target.filename
		_log_target('download', target)
		if dest.exists() and dest.stat().st_size > 0:
			print(f'[qwen38] exists    {dest.name}')
			paths.append(dest)
			continue
		if dry_run or hf_hub_download is None:
			paths.append(dest)
			continue
		downloaded = hf_hub_download(repo_id=UNSLOTH_REPO, filename=target.filename, local_dir=str(gguf_dir))
		paths.append(Path(downloaded))
	return paths


def convert_and_quantize(
	original_dir: Path,
	models_dir: Path,
	tools: dict[str, Path],
	quant_types: list[str],
	*,
	dry_run: bool = False,
) -> list[Path]:
	"""Convert official safetensors to BF16 GGUF, then run every requested llama-quantize preset."""
	gguf_dir = models_dir / 'self-quant'
	gguf_dir.mkdir(parents=True, exist_ok=True)
	bf16 = gguf_dir / 'Qwen3.8-27B-bf16.gguf'
	convert_py = tools['src'] / 'convert_hf_to_gguf.py'
	outputs = [bf16]

	print(f'[qwen38] convert   {original_dir} -> {bf16.name}')
	if not dry_run:
		if not convert_py.exists():
			raise FileNotFoundError(f'convert_hf_to_gguf.py not found at {convert_py}')
		if not bf16.exists():
			cmd = [sys.executable, str(convert_py), str(original_dir), '--outtype', 'bf16', '--outfile', str(bf16)]
			print('[qwen38] $ ' + ' '.join(cmd))
			subprocess.run(cmd, check=True)

	for quant_type in quant_types:
		assert quant_type in LLAMA_QUANT_TYPES
		out = gguf_dir / f'Qwen3.8-27B-self-{quant_type}.gguf'
		outputs.append(out)
		print(f'[qwen38] quantize  {quant_type} -> {out.name}')
		if dry_run or out.exists():
			continue
		cmd = [str(tools['quantize']), str(bf16), str(out), quant_type]
		print('[qwen38] $ ' + ' '.join(cmd))
		subprocess.run(cmd, check=True)
	return outputs


def write_models_preset(models_dir: Path, gguf_files: list[Path], mmproj: Path | None, hardware: HardwareInfo) -> Path:
	preset = models_dir / 'models.ini'
	ngl = 99 if hardware.has_nvidia else 0
	lines = ['[DEFAULT]', f'n-gpu-layers = {ngl}', 'flash-attn = on', 'jinja = true', '']
	for path in gguf_files:
		if not path.name.endswith('.gguf') or path.name.startswith('mmproj'):
			continue
		alias = path.stem
		lines.extend(
			[
				f'[{alias}]',
				f'model = {path}',
				'ctx-size = 8192',
			]
		)
		if mmproj is not None:
			lines.append(f'mmproj = {mmproj}')
		lines.append('')
	preset.write_text('\n'.join(lines) + '\n', encoding='utf-8')
	return preset


def start_llama_server(
	tools: dict[str, Path],
	models_dir: Path,
	gguf_files: list[Path],
	hardware: HardwareInfo,
	*,
	port: int = LLAMA_SERVER_PORT,
	dry_run: bool = False,
	foreground: bool = False,
) -> subprocess.Popen[bytes] | None:
	gguf_dir = models_dir / 'gguf'
	self_dir = models_dir / 'self-quant'
	serve_dir = models_dir / 'serve'
	serve_dir.mkdir(parents=True, exist_ok=True)

	linked: list[Path] = []
	for path in gguf_files:
		if not path.exists() and not dry_run:
			continue
		dest = serve_dir / path.name
		if path.exists() and dest.resolve() != path.resolve():
			if dest.exists() or dest.is_symlink():
				dest.unlink()
			dest.symlink_to(path.resolve())
		linked.append(dest if dest.exists() or dry_run else path)

	# Include already-downloaded GGUFs even if the planner listed a subset.
	if gguf_dir.exists():
		linked.extend(p for p in gguf_dir.glob('Qwen3.8-27B-*.gguf') if p not in linked)
	if self_dir.exists():
		linked.extend(p for p in self_dir.glob('Qwen3.8-27B-*.gguf') if p not in linked)

	mmproj = next((p for p in (gguf_dir / 'mmproj-BF16.gguf', gguf_dir / 'mmproj-F16.gguf') if p.exists()), None)
	unique_ggufs: list[Path] = []
	seen: set[str] = set()
	for path in linked:
		resolved = path.resolve() if path.exists() else path
		if resolved.name in seen:
			continue
		seen.add(resolved.name)
		unique_ggufs.append(resolved)
	preset = write_models_preset(models_dir, unique_ggufs, mmproj, hardware)

	cmd = [
		str(tools['server']),
		'--models-preset',
		str(preset),
		'--models-dir',
		str(serve_dir),
		'--models-max',
		'1',
		'--host',
		'127.0.0.1',
		'--port',
		str(port),
		'--jinja',
		'--chat-template-kwargs',
		json.dumps({'enable_thinking': False}),
	]
	if hardware.has_nvidia:
		cmd.extend(['-ngl', '99'])
	print('[qwen38] serve    llama-server router  ' + ' '.join(cmd))
	if dry_run:
		return None

	logs = models_dir / 'logs'
	logs.mkdir(parents=True, exist_ok=True)
	log_file = logs / 'llama-server.log'
	pid_file = logs / 'llama-server.pid'
	if foreground:
		return subprocess.Popen(cmd)
	with log_file.open('ab') as handle:
		proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
	pid_file.write_text(str(proc.pid), encoding='utf-8')
	print(f'[qwen38] llama-server pid={proc.pid}  log={_log_pretty_path(log_file)}  http://127.0.0.1:{port}/v1')
	return proc


def ensure_ollama(*, dry_run: bool = False) -> Path | None:
	found = shutil.which('ollama')
	if found:
		return Path(found)
	if dry_run:
		print('[qwen38] ollama    not on PATH (dry-run; would install)')
		return None
	print('[qwen38] ollama    installing via official script')
	subprocess.run('curl -fsSL https://ollama.com/install.sh | sh', shell=True, check=True)
	found = shutil.which('ollama')
	return Path(found) if found else None


def start_ollama_and_pull(tags: list[str], *, dry_run: bool = False) -> None:
	if not tags:
		return
	binary = ensure_ollama(dry_run=dry_run)
	if binary is None:
		print('[qwen38] ollama    skipped (binary unavailable)')
		return
	print('[qwen38] ollama    serve')
	if not dry_run:
		subprocess.Popen([str(binary), 'serve'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		deadline = time.time() + 30
		while time.time() < deadline:
			probe = subprocess.run([str(binary), 'list'], capture_output=True, text=True)
			if probe.returncode == 0:
				break
			time.sleep(1)
	for tag in tags:
		print(f'[qwen38] ollama    pull {tag}')
		if dry_run:
			continue
		subprocess.run([str(binary), 'pull', tag], check=False)


def write_env_file(models_dir: Path, plan: StartPlan, default_model: str) -> Path:
	env_path = models_dir / 'local.env'
	env_path.write_text(
		'\n'.join(
			[
				f'QWEN38_MODELS_DIR={models_dir}',
				f'QWEN38_BASE_URL=http://127.0.0.1:{plan.llama_port}/v1',
				'QWEN38_API_KEY=local',
				f'QWEN38_MODEL={default_model}',
				f'OLLAMA_HOST=http://127.0.0.1:{plan.ollama_port}',
				f'QWEN38_OLLAMA_MODEL={plan.ollama_pull[0] if plan.ollama_pull else "qwen3.8:27b"}',
				'',
			]
		),
		encoding='utf-8',
	)
	return env_path


def execute_start_plan(
	plan: StartPlan, *, dry_run: bool = False, foreground: bool = False, skip_self_quant: bool = False
) -> None:
	models_dir = plan.models_dir
	models_dir.mkdir(parents=True, exist_ok=True)
	_log_hardware(plan.hardware)
	for note in plan.notes:
		print(f'[qwen38] note     {note}')

	print('[qwen38] plan     download:')
	for target in plan.download:
		_log_target('keep    ', target)
	if plan.skipped:
		print('[qwen38] plan     skipped (disk/platform):')
		for target in plan.skipped:
			_log_target('skip    ', target)

	unsloth_targets = [t for t in plan.download if t.kind in {'unsloth-gguf', 'mmproj'}]
	want_original = any(t.kind == 'original' for t in plan.download)
	want_self_quant = [t.quant_type for t in plan.download if t.kind == 'self-quant' and t.quant_type and t.quant_type != 'BF16']
	tools: dict[str, Path] | None = None
	if plan.backend in {'llama', 'both'}:
		tools = ensure_llama_cpp(models_dir, plan.hardware, dry_run=dry_run)

	original_dir: Path | None = None
	if want_original:
		original_dir = download_original(models_dir, dry_run=dry_run)

	ggufs = download_unsloth_artifacts(unsloth_targets, models_dir, dry_run=dry_run)

	if want_self_quant and not skip_self_quant and tools is not None and original_dir is not None:
		try:
			ggufs.extend(
				convert_and_quantize(
					original_dir,
					models_dir,
					tools,
					[item for item in want_self_quant if item in LLAMA_QUANT_TYPES],
					dry_run=dry_run,
				)
			)
		except Exception as exc:
			print(f'[qwen38] self-quant failed ({exc!r}); continuing with downloaded GGUFs')

	if tools is not None and plan.backend in {'llama', 'both'}:
		existing = list((models_dir / 'gguf').glob('Qwen3.8-27B-*.gguf')) if (models_dir / 'gguf').exists() else []
		existing += list((models_dir / 'self-quant').glob('Qwen3.8-27B-*.gguf')) if (models_dir / 'self-quant').exists() else []
		to_serve = existing or ggufs
		start_llama_server(
			tools, models_dir, to_serve, plan.hardware, port=plan.llama_port, dry_run=dry_run, foreground=foreground
		)

	if plan.backend in {'ollama', 'both'}:
		start_ollama_and_pull(plan.ollama_pull, dry_run=dry_run)

	default_model = Path(plan.serve_ggufs[0]).stem if plan.serve_ggufs else 'Qwen3.8-27B-UD-Q4_K_XL'
	env_path = write_env_file(models_dir, plan, default_model)
	print(f'[qwen38] env      {_log_pretty_path(env_path)}')
	print('[qwen38] browser-use:')
	print(f'         source {_log_pretty_path(env_path)}')
	print('         uv run python examples/models/qwen38_27b_local.py')


def list_catalog() -> None:
	print(f'Official source: {OFFICIAL_REPO}')
	print(f'Unsloth GGUFs:   {UNSLOTH_REPO}')
	print(f'llama.cpp:       {LLAMA_CPP_RELEASE}+ (Qwen3.8 Gated DeltaNet)')
	print()
	print(f'{"key":20} {"kind":14} {"GB":>6}  label')
	for target in all_download_targets(include_self_quant=True, linux_only=False):
		linux = '' if target.linux_ok else '  [macos]'
		print(f'{target.key:20} {target.kind:14} {target.size_gb:6.1f}  {target.label}{linux}')
