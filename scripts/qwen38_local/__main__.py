"""CLI for downloading official Qwen3.8-27B and starting every local quantized build.

# @file purpose: argparse entrypoint for the Qwen3.8-27B local quant launcher
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .service import (
	build_start_plan,
	default_models_dir,
	detect_hardware,
	execute_start_plan,
	list_catalog,
)
from .views import BackendName


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog='python -m scripts.qwen38_local',
		description='Fetch Qwen/Qwen3.8-27B and start every local quantized variant that fits this machine.',
	)
	sub = parser.add_subparsers(dest='command', required=True)

	sub.add_parser('list', help='Print the original + Unsloth + Ollama + self-quant catalog')

	plan = sub.add_parser('plan', help='Show what start --all would download/serve on this host')
	_add_start_flags(plan)

	start = sub.add_parser('start', help='Download original weights, fetch/quantize every variant, and serve them')
	_add_start_flags(start)
	start.add_argument('--foreground', action='store_true', help='Keep llama-server in the foreground')

	return parser


def _add_start_flags(parser: argparse.ArgumentParser) -> None:
	parser.add_argument(
		'--models-dir', type=Path, default=None, help='Override QWEN38_MODELS_DIR / ~/.cache/browser-use/qwen3.8-27b'
	)
	parser.add_argument('--all', action='store_true', default=True, help='Include every quantized variant that fits')
	parser.add_argument('--backend', choices=('llama', 'ollama', 'both'), default='both')
	parser.add_argument('--force-all', action='store_true', help='Ignore disk/RAM filters and try every artifact')
	parser.add_argument('--force-self-quant', action='store_true', help='Convert official BF16 even on <32GiB RAM hosts')
	parser.add_argument('--skip-self-quant', action='store_true', help='Do not convert official safetensors with llama-quantize')
	parser.add_argument('--dry-run', action='store_true', help='Print the plan without downloading or spawning servers')


def main(argv: list[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)
	models_dir = (getattr(args, 'models_dir', None) or default_models_dir()).expanduser()

	if args.command == 'list':
		list_catalog()
		return 0

	hardware = detect_hardware(models_dir)
	backend: BackendName = args.backend
	plan = build_start_plan(
		hardware,
		models_dir=models_dir,
		backend=backend,
		include_self_quant=(not args.skip_self_quant) or args.force_self_quant,
		force_all=bool(args.force_all),
		force_self_quant=bool(args.force_self_quant),
	)

	if args.command == 'plan' or args.dry_run:
		execute_start_plan(
			plan,
			dry_run=True,
			foreground=False,
			skip_self_quant=bool(args.skip_self_quant),
		)
		return 0

	execute_start_plan(
		plan,
		dry_run=False,
		foreground=bool(getattr(args, 'foreground', False)),
		skip_self_quant=bool(args.skip_self_quant),
	)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
