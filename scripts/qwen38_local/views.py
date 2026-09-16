"""Pydantic models for the Qwen3.8-27B local quant catalog and start planner.

# @file purpose: Typed catalog/start-plan schemas for Qwen3.8-27B local serving
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactKind = Literal['original', 'unsloth-gguf', 'mmproj', 'self-quant', 'ollama']
BackendName = Literal['llama', 'ollama', 'both']


class DownloadTarget(BaseModel):
	"""One artifact the launcher can fetch or produce locally."""

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	key: str
	kind: ArtifactKind
	label: str
	size_bytes: int
	filename: str | None = None
	repo_id: str | None = None
	ollama_tag: str | None = None
	quant_type: str | None = None
	needs_gpu: bool = False
	linux_ok: bool = True
	quality_rank: int = Field(ge=0, description='Higher is closer to BF16.')
	notes: str = ''

	@property
	def size_gb(self) -> float:
		return self.size_bytes / 1_000_000_000


class HardwareInfo(BaseModel):
	"""Snapshot of the machine we are about to download/serve on."""

	model_config = ConfigDict(extra='forbid')

	cpu_count: int
	ram_bytes: int
	disk_free_bytes: int
	has_nvidia: bool
	platform: str

	@property
	def ram_gb(self) -> float:
		return self.ram_bytes / (1024**3)

	@property
	def disk_free_gb(self) -> float:
		return self.disk_free_bytes / (1024**3)


class StartPlan(BaseModel):
	"""What `start --all` will download, quantize, and serve given current hardware."""

	model_config = ConfigDict(extra='forbid')

	hardware: HardwareInfo
	models_dir: Path
	backend: BackendName
	download: list[DownloadTarget]
	skipped: list[DownloadTarget]
	self_quant_types: list[str]
	serve_ggufs: list[str]
	ollama_pull: list[str]
	llama_port: int = 8080
	ollama_port: int = 11434
	notes: list[str] = Field(default_factory=list)
