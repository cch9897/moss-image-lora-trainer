"""Custom LoRA: frozen nn.Linear base + fp32 A/B adapters with zero-initialized B.

Adapter layout (adapter_model.safetensors, fp32):
    layers.<i>.proj.<target>.lora_A  [r, in]
    layers.<i>.proj.<target>.lora_B  [out, r]
    text_embedder.proj.lora_A/lora_B          (optional target)
    final_linear_video.lora_A/lora_B          (optional target)
adapter_config.json:
    {"r", "alpha", "dropout", "targets", "text_projection", "final_linear",
     "base_model", "revision", "moss_lora_version"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from moss_lora import __version__

ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_EMA_NAME = "adapter_ema.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
DEFAULT_TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
_REGISTRY_ATTR = "_moss_lora_adapters"


class LoRALinear(nn.Module):
    """base(x) + ((dropout(x) @ A^T) @ B^T) * scaling, with A/B kept in fp32.

    B starts at zero so the wrapped layer is an exact identity at injection.
    Inputs may be bf16: A and B are cast to x.dtype for the matmuls while their
    parameters (and therefore accumulated gradients) stay fp32.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}.")
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError(f"Invalid LoRA rank {rank} for Linear({base.in_features}, {base.out_features}).")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout)
        # Weights may already sit on an accelerator (injection after .to(device));
        # allocate alongside them so the first matmul never mixes devices.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        # Gaussian A; B stays zero so the adapter is an exact identity before training.
        nn.init.normal_(self.lora_A, std=1.0 / rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.lora_dropout(x)
        lora_a = self.lora_A.to(dropped.dtype)
        lora_b = self.lora_B.to(dropped.dtype)
        update = (dropped @ lora_a.transpose(0, 1)) @ lora_b.transpose(0, 1)
        # The vendored blocks feed activations in the base weight dtype; keep that contract
        # even when a caller passes another dtype, while A/B compute in the input dtype.
        base_input = x if x.dtype == self.base.weight.dtype else x.to(self.base.weight.dtype)
        return self.base(base_input) + update * self.scaling

    # Delegated attributes: the vendored blocks read `.weight.dtype` on wrapped
    # projections, and merge/save need the true shapes.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling}"


def _resolve_parent(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent: nn.Module = model
    for part in path.split(".")[:-1]:
        parent = getattr(parent, part)
    return parent, path.split(".")[-1]


def lora_target_paths(num_layers: int, targets: list[str] | tuple[str, ...],
                      text_projection: bool = False, final_linear: bool = False) -> list[str]:
    invalid = sorted(set(targets) - set(DEFAULT_TARGETS))
    if invalid:
        raise ValueError(f"Unknown LoRA targets {invalid}; allowed: {list(DEFAULT_TARGETS)}.")
    paths = [f"layers.{index}.proj.{target}" for index in range(num_layers) for target in targets]
    if text_projection:
        paths.append("text_embedder.proj")
    if final_linear:
        paths.append("final_linear_video")
    return paths


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0.0,
                targets: list[str] | tuple[str, ...] = DEFAULT_TARGETS,
                text_projection: bool = False, final_linear: bool = False) -> dict[str, LoRALinear]:
    """Wrap every requested projection in-place; returns {path: LoRALinear}."""
    if hasattr(model, _REGISTRY_ATTR):
        raise RuntimeError("LoRA is already injected; call merge_lora() or start from a clean model.")
    wrappers: dict[str, LoRALinear] = {}
    for path in lora_target_paths(len(model.layers), targets, text_projection, final_linear):
        parent, attribute = _resolve_parent(model, path)
        base = getattr(parent, attribute)
        wrapper = LoRALinear(base, rank, alpha, dropout)
        setattr(parent, attribute, wrapper)
        wrappers[path] = wrapper
    if not wrappers:
        raise RuntimeError("LoRA injection matched no modules.")
    setattr(model, _REGISTRY_ATTR, wrappers)
    return wrappers


def get_adapters(model: nn.Module) -> dict[str, LoRALinear]:
    return getattr(model, _REGISTRY_ATTR, {})


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """{path.lora_A/lora_B: fp32 cpu tensor} for every injected adapter."""
    state = {}
    for path, wrapper in get_adapters(model).items():
        state[f"{path}.lora_A"] = wrapper.lora_A.detach().cpu().float()
        state[f"{path}.lora_B"] = wrapper.lora_B.detach().cpu().float()
    return state


def adapter_config_payload(model: nn.Module, base_model: str | None = None,
                           revision: str | None = None) -> dict[str, Any]:
    wrappers = get_adapters(model)
    first = next(iter(wrappers.values()))
    text_projection = "text_embedder.proj" in wrappers
    final_linear = "final_linear_video" in wrappers
    targets = sorted({path.rsplit(".", 1)[-1] for path in wrappers if path.startswith("layers.")},
                     key=DEFAULT_TARGETS.index)
    return {
        "r": first.rank,
        "alpha": first.scaling * first.rank,
        "dropout": first.lora_dropout.p,
        "targets": targets,
        "text_projection": text_projection,
        "final_linear": final_linear,
        "base_model": base_model,
        "revision": revision,
        "moss_lora_version": __version__,
    }


def save_adapter(model: nn.Module, output_dir: str | Path, *, base_model: str | None = None,
                 revision: str | None = None, weights_name: str = ADAPTER_WEIGHTS_NAME) -> Path:
    """Write adapter_model.safetensors + adapter_config.json (atomic)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from moss_lora.utils import atomic_save_safetensors, atomic_write_json

    atomic_save_safetensors(lora_state_dict(model), output_dir / weights_name)
    atomic_write_json(output_dir / ADAPTER_CONFIG_NAME, adapter_config_payload(model, base_model, revision))
    return output_dir / weights_name


def _expected_keys(config: dict[str, Any], model: nn.Module) -> set[str]:
    paths = lora_target_paths(len(model.layers), config.get("targets", list(DEFAULT_TARGETS)),
                              bool(config.get("text_projection", False)),
                              bool(config.get("final_linear", False)))
    return {f"{path}.{name}" for path in paths for name in ("lora_A", "lora_B")}


def load_adapter(model: nn.Module, adapter: str | Path) -> dict[str, Any]:
    """Load adapter weights into an already-injected model with strict key validation."""
    adapter = Path(adapter)
    if adapter.is_dir():
        adapter = adapter / ADAPTER_WEIGHTS_NAME
    if not adapter.is_file():
        raise FileNotFoundError(f"Adapter weights not found: {adapter}")
    config = read_adapter_config(adapter.parent)
    state = load_file(str(adapter))
    expected = _expected_keys(config, model)
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"Adapter {adapter} is missing tensors: {missing}")
    if unexpected:
        raise ValueError(f"Adapter {adapter} has unexpected tensors: {unexpected}")
    wrappers = get_adapters(model)
    for path, wrapper in wrappers.items():
        with torch.no_grad():
            wrapper.lora_A.copy_(state[f"{path}.lora_A"].float())
            wrapper.lora_B.copy_(state[f"{path}.lora_B"].float())
    return config


def read_adapter_config(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / ADAPTER_CONFIG_NAME if Path(directory).is_dir() else Path(directory)
    return json.loads(path.read_text(encoding="utf-8"))


def load_lora_into_model(model: nn.Module, adapter: str | Path,
                         base_model: str | None = None) -> dict[str, Any]:
    adapter = Path(adapter)
    directory = adapter if adapter.is_dir() else adapter.parent
    config = read_adapter_config(directory)
    inject_lora(
        model,
        rank=int(config["r"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
        targets=list(config.get("targets", DEFAULT_TARGETS)),
        text_projection=bool(config.get("text_projection", False)),
        final_linear=bool(config.get("final_linear", False)),
    )
    load_adapter(model, adapter)
    if base_model and config.get("base_model") not in (None, base_model):
        raise ValueError(
            f"Adapter base_model {config.get('base_model')!r} does not match target model {base_model!r}.")
    return config


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base Linear (weight += B @ A * scaling) and unwrap.

    Restores the original module tree and state-dict key names.
    """
    wrappers = get_adapters(model)
    if not wrappers:
        raise RuntimeError("No LoRA adapters to merge.")
    for path, wrapper in wrappers.items():
        parent, attribute = _resolve_parent(model, path)
        base = wrapper.base
        delta = (wrapper.lora_B.detach().float() @ wrapper.lora_A.detach().float()) * wrapper.scaling
        merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                           dtype=base.weight.dtype, device=base.weight.device)
        merged.weight.copy_((base.weight.detach().float() + delta).to(base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias.detach())
        merged.requires_grad_(False)
        setattr(parent, attribute, merged)
    delattr(model, _REGISTRY_ATTR)
    return model
