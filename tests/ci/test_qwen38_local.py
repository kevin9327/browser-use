"""Catalog and packing tests for the Qwen3.8-27B local quant launcher.

These tests never download weights or talk to Hugging Face. They only verify
the catalog invariants and the disk/RAM planner that `start --all` uses.
"""

from pathlib import Path

from scripts.qwen38_local.catalog import (
	COVERAGE_KEYS,
	DEFAULT_PINNED_KEYS,
	LLAMA_QUANT_TYPES,
	OFFICIAL_REPO,
	OLLAMA_TAGS,
	ORIGINAL_SAFETENSOR_BYTES,
	UNSLOTH_REPO,
	all_download_targets,
	default_serve_quant,
	fit_by_bytes,
	self_quant_targets,
)
from scripts.qwen38_local.service import build_start_plan, detect_hardware
from scripts.qwen38_local.views import HardwareInfo


def test_official_repo_is_qwen_source():
	assert OFFICIAL_REPO == 'Qwen/Qwen3.8-27B'
	assert UNSLOTH_REPO == 'unsloth/Qwen3.8-27B-GGUF'
	assert ORIGINAL_SAFETENSOR_BYTES > 50_000_000_000


def test_catalog_keys_and_filenames_are_unique():
	targets = all_download_targets(include_self_quant=True, linux_only=False)
	keys = [target.key for target in targets]
	assert len(keys) == len(set(keys))
	filenames = [target.filename for target in targets if target.filename]
	assert len(filenames) == len(set(filenames))
	ollama_tags = [target.ollama_tag for target in OLLAMA_TAGS]
	assert len(ollama_tags) == len(set(ollama_tags))
	assert 'qwen3.8:27b' in ollama_tags
	assert 'qwen3.8:27b-q8_0' in ollama_tags
	assert 'qwen3.8:27b-bf16' in ollama_tags


def test_self_quant_covers_the_full_k_quant_ladder():
	assert LLAMA_QUANT_TYPES == (
		'Q2_K',
		'Q3_K_S',
		'Q3_K_M',
		'Q3_K_L',
		'Q4_0',
		'Q4_1',
		'Q4_K_S',
		'Q4_K_M',
		'Q5_0',
		'Q5_1',
		'Q5_K_S',
		'Q5_K_M',
		'Q6_K',
		'Q8_0',
	)
	produced = {target.quant_type for target in self_quant_targets()}
	assert produced == {'BF16', *LLAMA_QUANT_TYPES}


def test_mlx_ollama_tags_are_not_linux_ok():
	mlx = [
		target
		for target in OLLAMA_TAGS
		if 'mlx' in (target.ollama_tag or '') or target.quant_type in {'MLX', 'MLX-BF16', 'MXFP8', 'NVFP4'}
	]
	assert mlx
	assert all(not target.linux_ok for target in mlx)
	linux = all_download_targets(include_self_quant=False, linux_only=True)
	assert all(target.linux_ok for target in linux)
	assert not any(target.ollama_tag and 'mlx' in target.ollama_tag for target in linux)


def test_fit_by_bytes_pins_original_and_default_quant():
	pinned = ('ud-q4-k-xl', 'mmproj-bf16', 'original')
	assert DEFAULT_PINNED_KEYS == pinned
	# 90GB free after the 8GB reserve -> 82GB usable. That holds Q4_K_XL (17.6)
	# + mmproj (0.93) + original (55.6) and must not spend leftover on a second
	# BF16 copy before smaller quants.
	chosen, skipped = fit_by_bytes(
		all_download_targets(include_self_quant=False, linux_only=True),
		budget_bytes=90_000_000_000,
		pinned_keys=DEFAULT_PINNED_KEYS,
		priority_keys=COVERAGE_KEYS,
	)
	keys = {target.key for target in chosen}
	assert 'original' in keys
	assert 'ud-q4-k-xl' in keys
	assert 'mmproj-bf16' in keys
	assert 'ollama-bf16' not in keys
	assert skipped


def test_fit_by_bytes_prefers_default_quant_over_original_when_disk_is_tight():
	chosen, skipped = fit_by_bytes(
		all_download_targets(include_self_quant=False, linux_only=True),
		budget_bytes=80_000_000_000,
		pinned_keys=DEFAULT_PINNED_KEYS,
	)
	keys = {target.key for target in chosen}
	assert 'ud-q4-k-xl' in keys
	assert 'mmproj-bf16' in keys
	assert 'original' not in keys
	assert any(target.key == 'original' for target in skipped)


def test_fit_by_bytes_fills_quant_ladder_smallest_first_after_pins():
	# 220GB usable after reserve: original + default Q4 + the coverage ladder
	# (Q5/Q6/Q8) must land before leftover duplicate Q4_0/Q4_1 files.
	chosen, skipped = fit_by_bytes(
		all_download_targets(include_self_quant=False, linux_only=True),
		budget_bytes=230_000_000_000,
		pinned_keys=DEFAULT_PINNED_KEYS,
		priority_keys=COVERAGE_KEYS,
	)
	keys = {target.key for target in chosen}
	assert 'original' in keys
	assert 'ud-q4-k-xl' in keys
	assert 'ud-q5-k-xl' in keys
	assert 'ud-q6-k-xl' in keys
	assert 'q8-0' in keys
	assert skipped


def test_default_serve_quant_tracks_ram():
	tiny = default_serve_quant(12 * 1024**3)
	assert tiny.size_bytes < 10_000_000_000
	mid = default_serve_quant(24 * 1024**3)
	assert mid.quality_rank >= tiny.quality_rank
	big = default_serve_quant(64 * 1024**3)
	assert big.key == 'ud-q8-k-xl'


def test_start_plan_on_small_cpu_box_skips_self_quant_without_force():
	hardware = HardwareInfo(
		cpu_count=4,
		ram_bytes=15 * 1024**3,
		disk_free_bytes=200_000_000_000,
		has_nvidia=False,
		platform='linux',
	)
	plan = build_start_plan(hardware, models_dir=Path('/tmp/qwen38-test'), backend='llama', include_self_quant=True)
	assert plan.hardware.ram_gb < 16
	assert any('15' in note or 'RAM' in note for note in plan.notes)
	assert not any(target.kind == 'self-quant' for target in plan.download)
	assert any(target.key == 'original' for target in plan.download)
	assert any(target.kind == 'unsloth-gguf' for target in plan.download)
	assert plan.llama_port == 8080


def test_start_plan_force_all_includes_self_quant_on_small_ram():
	hardware = HardwareInfo(
		cpu_count=4,
		ram_bytes=15 * 1024**3,
		disk_free_bytes=200_000_000_000,
		has_nvidia=False,
		platform='linux',
	)
	plan = build_start_plan(
		hardware,
		models_dir=Path('/tmp/qwen38-test'),
		backend='llama',
		include_self_quant=True,
		force_all=True,
	)
	assert any(target.kind == 'self-quant' for target in plan.download)
	assert {target.quant_type for target in plan.download if target.kind == 'self-quant'} >= set(LLAMA_QUANT_TYPES)


def test_detect_hardware_reads_real_machine(tmp_path: Path):
	hardware = detect_hardware(tmp_path)
	assert hardware.cpu_count >= 1
	assert hardware.ram_bytes > 0
	assert hardware.disk_free_bytes > 0
	assert hardware.platform


def test_cli_list_and_plan_dry_run(tmp_path: Path, capsys):
	from scripts.qwen38_local.__main__ import main

	assert main(['list']) == 0
	listed = capsys.readouterr().out
	assert 'Qwen/Qwen3.8-27B' in listed
	assert 'UD-Q4_K_XL' in listed
	assert 'qwen3.8:27b' in listed

	assert main(['plan', '--all', '--dry-run', '--backend', 'llama', '--models-dir', str(tmp_path)]) == 0
	planned = capsys.readouterr().out
	assert 'original' in planned
	assert 'llama-server' in planned
