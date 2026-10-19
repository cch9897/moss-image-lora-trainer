"""YAML configuration with strict key validation and CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULT_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(set(DTYPE_MAP))}.")
    return DTYPE_MAP[name]


@dataclass
class ModelConfig:
    path: str = ""
    revision: str | None = None
    dtype: str = "bf16"
    local_files_only: bool = False

    def torch_dtype(self) -> torch.dtype:
        return parse_dtype(self.dtype)


@dataclass
class DataConfig:
    cache_dir: str = "cache"
    num_workers: int = 4
    batch_size: int = 1
    drop_last: bool = True
    pin_memory: bool = True


@dataclass
class TrainConfig:
    device: str = "cuda"
    output_dir: str = "runs/moss_lora"
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    warmup_steps: int = 50
    lr_scheduler: str = "cosine"  # cosine | constant
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = True
    compile: bool = False
    compile_mode: str = "default"
    sdpa_backend: str = "auto"
    ema_decay: float = 0.0
    timestep_sampling: str = "logit_normal"  # logit_normal | uniform_grid
    logit_normal_mean: float = -1.09
    logit_normal_std: float = 0.95
    sigma_cutoff: float = 0.95
    uniform_mix: float = 0.01

    def __post_init__(self):
        self.betas = tuple(self.betas)  # YAML lists -> tuple


@dataclass
class LoraConfig:
    rank: int = 32
    alpha: float | None = None  # None -> rank
    dropout: float = 0.0
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    text_projection: bool = False
    final_linear: bool = False


@dataclass
class LoggingConfig:
    log_every: int = 10
    save_every: int = 200
    # Write <run>/comfyui_lora.safetensors after training (raw adapter_model.safetensors stays).
    comfy_export: bool = True
    # Also export inside every checkpoints/step_XXXXXX directory. Off by default: convert a
    # checkpoint afterwards with `moss-lora-export-comfy --run-dir <run>`.
    comfy_export_checkpoints: bool = False
    sample_every: int = 0
    sample_steps: int = 30
    sample_cfg: float = 3.0
    sample_skip_layer: int = 22  # -1 disables negative-branch block skipping
    sample_negative_prompt: str = ""
    sample_val_ids: list[str] = field(default_factory=list)


@dataclass
class TrainerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig,
             "lora": LoraConfig, "logging": LoggingConfig}


def _fill(section: Any, values: dict[str, Any], name: str) -> Any:
    allowed = {f.name for f in fields(section)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown config key(s) in '{name}': {unknown}. Allowed: {sorted(allowed)}.")
    kwargs = {}
    for f in fields(section):
        if f.name in values:
            kwargs[f.name] = values[f.name]
    return section(**kwargs)


def load_config(path: str | Path) -> TrainerConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown_sections = sorted(set(raw) - set(_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {unknown_sections}. Allowed: {sorted(_SECTIONS)}.")
    return TrainerConfig(
        **{key: _fill(section_cls, raw.get(key, {}) or {}, key) for key, section_cls in _SECTIONS.items()},
    )


def apply_cli_overrides(config: TrainerConfig, *, max_steps: int | None = None,
                        output_dir: str | None = None, device: str | None = None,
                        batch_size: int | None = None, val_ids: list[str] | None = None,
                        comfy_export: bool | None = None) -> TrainerConfig:
    if max_steps is not None:
        config.train.max_steps = max_steps
    if output_dir is not None:
        config.train.output_dir = output_dir
    if device is not None:
        config.train.device = device
    if batch_size is not None:
        config.data.batch_size = batch_size
    if val_ids:
        config.logging.sample_val_ids = val_ids
    if comfy_export is not None:
        config.logging.comfy_export = comfy_export
    return config
