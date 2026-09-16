"""CLI: list or start official Qwen3.8-27B quantized models locally.

Usage:
    uv run python -m browser_use.llm.qwen38_27b list
    uv run python -m browser_use.llm.qwen38_27b start
    uv run python -m browser_use.llm.qwen38_27b start --dry-run
"""

# @file purpose: CLI entry point to list/start official Qwen3.8-27B quants

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from browser_use.llm.qwen38_27b.service import Qwen3827BLocalRuntime, format_start_report, select_quants_for_platform
from browser_use.llm.qwen38_27b.views import OFFICIAL_QUANTIZED_MODELS


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Official Qwen3.8-27B local quantized models')
	sub = parser.add_subparsers(dest='command', required=True)
	sub.add_parser('list', help='Show the official quantized catalog')
	start = sub.add_parser('start', help='Pull and start every compatible official quant')
	start.add_argument('--dry-run', action='store_true', help='Print the plan without calling Ollama')
	start.add_argument('--include-mlx', action='store_true', help='Also pull Apple MLX tags (macOS)')
	return parser


def _print_catalog() -> None:
	print('Official quantized Qwen/Qwen3.8-27B checkpoints:\n')
	for spec in OFFICIAL_QUANTIZED_MODELS:
		flags = []
		if spec.recommended:
			flags.append('recommended')
		if spec.mtp:
			flags.append('mtp')
		suffix = f' [{", ".join(flags)}]' if flags else ''
		print(f'  {spec.tag:28}  {spec.quant:8}  {spec.size_gb:5.0f} GB  {spec.runtime:4}  {",".join(spec.platforms)}{suffix}')
	print('\nThis host:')
	for spec in select_quants_for_platform(sys.platform):
		print(f'  start {spec.tag}')


async def _start(dry_run: bool, include_mlx: bool) -> int:
	logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
	runtime = Qwen3827BLocalRuntime()
	report = await runtime.start_all(dry_run=dry_run, include_mlx=include_mlx or None)
	print(format_start_report(report))
	failed = any(result.action == 'failed' for result in report.results)
	return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	if args.command == 'list':
		_print_catalog()
		return 0
	if args.command == 'start':
		return asyncio.run(_start(dry_run=args.dry_run, include_mlx=args.include_mlx))
	parser.error(f'unknown command {args.command}')
	return 2


if __name__ == '__main__':
	raise SystemExit(main())
