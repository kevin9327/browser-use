"""CLI: refresh the official Qwen3.8-27B catalog and start every local GGUF quant.

uv run python -m browser_use.llm.qwen38 refresh
uv run python -m browser_use.llm.qwen38 serve --all
uv run python -m browser_use.llm.qwen38 status
"""

# @file purpose: CLI entry for refreshing and serving all Qwen3.8-27B quantized models

from __future__ import annotations

import argparse
import asyncio
import sys

from browser_use.llm.qwen38.service import (
	Qwen38LocalService,
	_log_fleet_summary,
	_log_quant_line,
	detect_hardware,
	find_llama_server,
)
from browser_use.llm.qwen38.views import ORIGINAL_MODEL_ID, QUANT_REPO_ID


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog='python -m browser_use.llm.qwen38',
		description='Serve official Qwen3.8-27B bartowski GGUF quants via llama.cpp',
	)
	sub = parser.add_subparsers(dest='command', required=True)

	refresh = sub.add_parser('refresh', help='Fetch the latest original SHA + GGUF tree from Hugging Face')
	refresh.add_argument('--json', action='store_true', help='Print catalog as JSON')

	serve = sub.add_parser('serve', help='Start quantized llama-server endpoints')
	serve.add_argument('--all', action='store_true', default=True, help='Start every language GGUF (default)')
	serve.add_argument('--force', action='store_true', help='Spawn even when the GGUF does not fit RAM+VRAM')
	serve.add_argument('--no-spawn', action='store_true', help='Write launch scripts only; do not exec llama-server')
	serve.add_argument('--llama-server', default=None, help='Path to llama-server (b10896+)')
	serve.add_argument('--host', default='127.0.0.1')
	serve.add_argument('--base-port', type=int, default=8080)
	serve.add_argument('--ctx', type=int, default=None, help='Override -c (default 8192 CPU / 32768 GPU)')

	sub.add_parser('status', help='Show hardware, llama-server, and the current catalog plan')
	return parser


async def _refresh(as_json: bool) -> int:
	service = Qwen38LocalService()
	catalog = await service.refresh_catalog()
	if as_json:
		print(catalog.model_dump_json(indent=2))
		return 0
	print(f'original  {catalog.original_repo}@{catalog.original_sha}')
	print(f'quants    {catalog.quant_repo}  source={catalog.source}  llama.cpp>={catalog.min_llama_cpp_release}')
	print(f'mmproj    {", ".join(item.filename for item in catalog.mmproj) or "(none)"}')
	for quant in catalog.quants:
		print(_log_quant_line(quant))
	print(f'{len(catalog.quants)} quantized models')
	return 0


async def _serve(args: argparse.Namespace) -> int:
	service = Qwen38LocalService(
		llama_server_bin=args.llama_server,
		host=args.host,
		base_port=args.base_port,
	)
	fleet = await service.start_all(force=args.force, spawn=not args.no_spawn, ctx_size=args.ctx)
	script = service.cache_dir / 'start-all-qwen38-quants.sh'
	print(_log_fleet_summary(fleet))
	print(f'launch script  {script}')
	print(f'from {ORIGINAL_MODEL_ID}  via {QUANT_REPO_ID}')
	for spec in fleet.servers:
		state = spec.skip_reason or f'spawned {spec.base_url}  model={spec.quant.alias}'
		print(f'  {spec.quant.quant:10}  {state}')
	if fleet.llama_server_bin is None:
		print(
			'\nllama-server is not on PATH. Install llama.cpp '
			f'{fleet.catalog.min_llama_cpp_release}+ then re-run, e.g.\n'
			'  curl -LsSf https://llama.app/install.sh | sh\n'
			'  uv run python -m browser_use.llm.qwen38 serve --all',
			file=sys.stderr,
		)
		return 0 if args.no_spawn else 2
	if not fleet.started() and not args.no_spawn:
		print(
			'\nNo quant fitted this machine. Re-run on a GPU box, or pass --force '
			'(will mmap and likely thrash). The launch script is ready for local use.',
			file=sys.stderr,
		)
		return 0
	if not args.no_spawn and fleet.started():
		print('llama-server processes are running. Ctrl+C does not reap them; kill by port if needed.')
	return 0


async def _status() -> int:
	hw = detect_hardware()
	bin_path = find_llama_server()
	service = Qwen38LocalService()
	catalog = await service.refresh_catalog()
	fleet = service.plan_fleet(catalog, hardware=hw)
	print(f'hardware   RAM={hw.ram_bytes / 1024**3:.1f} GiB  VRAM={hw.vram_bytes / 1024**3:.1f} GiB  ngl={hw.gpu_layers}')
	print(f'llama-server  {bin_path or "NOT FOUND"}')
	print(_log_fleet_summary(fleet))
	for spec in fleet.servers:
		print(f'  {_log_quant_line(spec.quant)}  {spec.skip_reason or spec.base_url}')
	return 0


async def _async_main(argv: list[str] | None = None) -> int:
	args = _parser().parse_args(argv)
	if args.command == 'refresh':
		return await _refresh(args.json)
	if args.command == 'serve':
		return await _serve(args)
	if args.command == 'status':
		return await _status()
	raise AssertionError(f'unknown command {args.command}')


def main(argv: list[str] | None = None) -> int:
	return asyncio.run(_async_main(argv))


if __name__ == '__main__':
	sys.exit(main())
