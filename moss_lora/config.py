"""YAML configuration with strict key validation and CLI overrides."""

from __future__ import annotations

import math
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
    # Absolute per-kind weights for the five v2 variants; empty = use each item's
    # index weights (or uniform for old caches).
    variant_weights: dict[str, float] = field(default_factory=dict)
    # Optional convenience: layout_weight is absolute (0-0.3), original_weight is
    # the original share of the remaining caption branch (spec 5.1).
    layout_weight: float | None = None
    original_weight: float | None = None

    def __post_init__(self) -> None:
        self.variant_weights = {str(key): float(value)
                                for key, value in (self.variant_weights or {}).items()}
        for key, value in self.variant_weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"data.variant_weights[{key!r}] must be finite and >= 0")
        if self.variant_weights and sum(self.variant_weights.values()) <= 0:
            raise ValueError("data.variant_weights must contain at least one positive weight")
        if self.layout_weight is not None and not 0.0 <= self.layout_weight <= 0.3:
            raise ValueError("data.layout_weight must be in [0, 0.3]")
        if self.original_weight is not None and not 0.0 <= self.original_weight <= 1.0:
            raise ValueError("data.original_weight must be in [0, 1]")

    def resolved_variant_weights(self) -> dict[str, float]:
        """Absolute weights for all kinds, or ``{}`` to keep per-item index weights."""
        if self.variant_weights:
            return dict(self.variant_weights)
        if self.layout_weight is None and self.original_weight is None:
            return {}
        layout = 0.20 if self.layout_weight is None else float(self.layout_weight)
        original_share = 0.20 if self.original_weight is None else float(self.original_weight)
        caption_share = 1.0 - layout
        return {
            "original": original_share * caption_share,
            "long": caption_share * (1.0 - original_share) / 3.0,
            "medium": caption_share * (1.0 - original_share) / 3.0,
            "short": caption_share * (1.0 - original_share) / 3.0,
            "layout": layout,
        }


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
    # Selective activation checkpointing: save expensive op outputs (attention, and
    # optionally matmuls) across the checkpoint boundary and recompute only the cheap
    # elementwise chains. "off" | "attention" | "matmul". Trades extra VRAM for less
    # recompute; "matmul" costs the most memory. Requires gradient_checkpointing.
    selective_checkpointing: str = "off"
    # Whole-batch forward with masked padding (mathematically equivalent to the
    # per-sample loop, much faster for batch_size > 1). False restores the vendored
    # per-sample path bit-for-bit.
    batched_forward: bool = True
    compile: bool = False
    compile_mode: str = "default"
    sdpa_backend: str = "auto"
    ema_decay: float = 0.0
    timestep_sampling: str = "logit_normal"  # logit_normal | uniform_grid
    logit_normal_mean: float = -1.09
    logit_normal_std: float = 0.95
    sigma_cutoff: float = 0.95
    uniform_mix: float = 0.01
    keep_last_checkpoints: int = 0
    save_epoch: bool = False

    def __post_init__(self):
        self.betas = tuple(self.betas)  # YAML lists -> tuple
        if self.keep_last_checkpoints < 0:
            raise ValueError("train.keep_last_checkpoints must be >= 0")
        if self.selective_checkpointing is False:  # YAML 1.1 parses unquoted `off` as False
            self.selective_checkpointing = "off"
        if self.selective_checkpointing not in ("off", "attention", "matmul"):
            raise ValueError(
                "train.selective_checkpointing must be off/attention/matmul, "
                f"got {self.selective_checkpointing!r}")
        if self.selective_checkpointing != "off" and not self.gradient_checkpointing:
            raise ValueError("train.selective_checkpointing requires gradient_checkpointing: true")


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
