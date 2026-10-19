"""Convert a MOSS-LoRA-Trainer adapter into a ComfyUI-compatible LoRA file.

ComfyUI's LoRA loader maps a model weight ``diffusion_model.layers.N.proj.q.weight``
to the base key ``diffusion_model.layers.N.proj.q`` and then looks for
``{base}.lora_A.weight`` / ``{base}.lora_B.weight`` (diffusers naming) and
``{base}.alpha``.  Trainer adapters use ``layers.N.proj.q.lora_A`` instead, so
they must be renamed before ComfyUI can load them.

This module only reads the trained adapter and writes a new safetensors file;
the source adapter is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moss_lora.lora import ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, DEFAULT_TARGETS

COMFY_PREFIX = "diffusion_model."
COMFY_METADATA_KEY = "moss_lora_comfy"
COMFY_OUTPUT_NAME = "comfyui_lora.safetensors"
KEY_PATTERN = re.compile(r"^layers\.(\d+)\.proj\.([A-Za-z0-9_]+)\.lora_([AB])$")


class ConversionError(ValueError):
    """The source adapter cannot be converted without guessing or dropping data."""


def _resolve_sources(adapter: str | Path) -> tuple[Path, Path]:
    """Return (weights file, adapter_config.json) for a directory or weights file."""
    path = Path(adapter)
    if path.is_dir():
        weights = path / ADAPTER_WEIGHTS_NAME
        config = path / ADAPTER_CONFIG_NAME
    else:
        weights = path
        config = path.parent / ADAPTER_CONFIG_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"Adapter weights not found: {weights}")
    if not config.is_file():
        raise FileNotFoundError(
            f"Adapter config not found: {config} (expected next to the weights file)"
        )
    return weights, config


def _read_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("r", "alpha"):
        if field not in config:
            raise ConversionError(f"{config_path} is missing the {field!r} field")
    try:
        rank = int(config["r"])
        alpha = float(config["alpha"])
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"Invalid rank/alpha in {config_path}: {exc}") from exc
    if rank <= 0:
        raise ConversionError(f"Invalid LoRA rank r={config['r']!r} in {config_path}")
    config["r"] = rank
    config["alpha"] = alpha

    targets = config.get("targets", list(DEFAULT_TARGETS))
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ConversionError(f"Invalid targets in {config_path}: {targets!r}")
    unknown_targets = sorted(set(targets) - set(DEFAULT_TARGETS))
    if unknown_targets:
        raise ConversionError(
            f"Unknown LoRA targets in {config_path}: {unknown_targets}; "
            f"supported targets are {list(DEFAULT_TARGETS)}"
        )
    if len(set(targets)) != len(targets):
        raise ConversionError(f"Duplicate targets in {config_path}: {targets!r}")
    config["targets"] = list(targets)
    return config


def _collect_pairs(keys: list[str], targets: list[str]) -> dict[tuple[int, str], dict[str, str]]:
    """Group source keys into {(layer, target): {'A': key, 'B': key}} with validation."""
    pairs: dict[tuple[int, str], dict[str, str]] = {}
    unexpected: list[str] = []
    for key in keys:
        match = KEY_PATTERN.match(key)
        if match is None:
            unexpected.append(key)
            continue
        layer, target, which = int(match.group(1)), match.group(2), match.group(3)
        if target not in targets:
            unexpected.append(key)
            continue
        slot = pairs.setdefault((layer, target), {})
        slot[which] = key

    if unexpected:
        raise ConversionError(
            "Adapter contains keys that are not MOSS LoRA A/B pairs: "
            + ", ".join(sorted(unexpected)[:10])
            + (" ..." if len(unexpected) > 10 else "")
        )
    if not pairs:
        raise ConversionError("Adapter contains no layers.*.proj.*.lora_A/lora_B tensors")

    layers = sorted({layer for layer, _ in pairs})
    if layers != list(range(layers[0], layers[-1] + 1)):
        missing = sorted(set(range(layers[0], layers[-1] + 1)) - set(layers))
        raise ConversionError(f"Adapter is missing whole layers: {missing}")
    if layers[0] != 0:
        raise ConversionError(f"Adapter layer indices must start at 0, found {layers[0]}")

    incomplete = sorted(
        f"layers.{layer}.proj.{target}.lora_{which}"
        for (layer, target), slots in pairs.items()
        for which in ("A", "B")
        if which not in slots
    )
    if incomplete:
        raise ConversionError("Adapter is missing A/B pairs: " + ", ".join(incomplete[:10]))

    missing_pairs = [
        (layer, target)
        for layer in layers
        for target in targets
        if (layer, target) not in pairs
    ]
    if missing_pairs:
        formatted = ", ".join(f"layers.{layer}.proj.{target}" for layer, target in missing_pairs[:10])
        raise ConversionError(f"Adapter is missing configured targets: {formatted}")

    # The regex already guarantees that only A/B keys are present; per-pair
    # shapes and dtypes are validated while reading tensors.
    return pairs


def _validate_shapes(handle: safe_open, pairs: dict[tuple[int, str], dict[str, str]],
                     rank: int) -> None:
    """Reject rank/shape mismatches before any output file is created."""
    for (layer, target), slots in sorted(pairs.items()):
        a_key, b_key = slots["A"], slots["B"]
        a_slice = handle.get_slice(a_key)
        b_slice = handle.get_slice(b_key)
        a_shape = tuple(a_slice.get_shape())
        b_shape = tuple(b_slice.get_shape())
        name = f"layers.{layer}.proj.{target}"
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ConversionError(
                f"{name}: lora_A/lora_B must be 2-D, got {a_shape} and {b_shape}"
            )
        if a_shape[0] != rank:
            raise ConversionError(
                f"{name}: lora_A rank dimension is {a_shape[0]}, config rank is {rank}"
            )
        if b_shape[1] != rank:
            raise ConversionError(
                f"{name}: lora_B rank dimension is {b_shape[1]}, config rank is {rank}"
            )
        if a_slice.get_dtype() != b_slice.get_dtype():
            raise ConversionError(
                f"{name}: lora_A dtype {a_slice.get_dtype()} != lora_B dtype {b_slice.get_dtype()}"
            )


def convert_adapter(adapter: str | Path, output: str | Path) -> dict[str, Any]:
    """Convert a trainer adapter to ComfyUI naming and return a summary."""
    weights_path, config_path = _resolve_sources(adapter)
    config = _read_config(config_path)
    rank, alpha, targets = config["r"], config["alpha"], config["targets"]
    output = Path(output)

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        pairs = _collect_pairs(keys, targets)
        _validate_shapes(handle, pairs, rank)

        converted: dict[str, torch.Tensor] = {}
        alpha_value = torch.tensor(alpha, dtype=torch.float32)
        for (layer, target), slots in sorted(pairs.items()):
            base = f"{COMFY_PREFIX}layers.{layer}.proj.{target}"
            converted[f"{base}.lora_A.weight"] = handle.get_tensor(slots["A"])
            converted[f"{base}.lora_B.weight"] = handle.get_tensor(slots["B"])
            # Fresh tensor per key: safetensors rejects shared storage.
            converted[f"{base}.alpha"] = alpha_value.clone()

    metadata = {
        "format": "pt",
        COMFY_METADATA_KEY: "1",
        "base_model": str(config.get("base_model") or ""),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(output.name + ".tmp")
    try:
        save_file(converted, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, output)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    sample_keys = sorted(converted)[:6]
    return {
        "output": str(output),
        "keys": len(converted),
        "weights": sum(1 for key in converted if key.endswith((".lora_A.weight", ".lora_B.weight"))),
        "alphas": sum(1 for key in converted if key.endswith(".alpha")),
        "layers": len({layer for layer, _ in pairs}),
        "targets": list(targets),
        "rank": rank,
        "alpha": alpha,
        "base_model": metadata["base_model"],
        "source": str(weights_path),
        "bytes": output.stat().st_size,
        "sample_keys": sample_keys,
    }


def find_latest_adapter(run_dir: str | Path) -> Path:
    """Return the newest complete adapter directory in a training run.

    Checkpoints are considered newest-first, then the run root (the final save after
    training).  A directory is complete when both the weights and adapter_config.json
    exist, so half-written or unrelated ``step_*`` directories are ignored.
    """
    root = Path(run_dir)
    checkpoints_dir = root / "checkpoints"
    checkpoints = sorted(checkpoints_dir.glob("step_*")) if checkpoints_dir.is_dir() else []
    for candidate in [*reversed(checkpoints), root]:
        if (candidate / ADAPTER_WEIGHTS_NAME).is_file() and (candidate / ADAPTER_CONFIG_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"No complete adapter ({ADAPTER_WEIGHTS_NAME} + {ADAPTER_CONFIG_NAME}) in "
        f"{checkpoints_dir} or {root}"
    )


def export_latest_checkpoint(run_dir: str | Path,
                             output: str | Path | None = None) -> dict[str, Any]:
    """Convert the newest adapter in a run (or an interrupted run's checkpoint).

    Defaults to ``<run>/comfyui_lora.safetensors`` so the result is ready to drop
    into ComfyUI's ``models/loras`` directory.
    """
    run_dir = Path(run_dir)
    adapter = find_latest_adapter(run_dir)
    return convert_adapter(adapter, Path(output) if output is not None
                           else run_dir / COMFY_OUTPUT_NAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moss-lora-export-comfy",
        description="Convert a MOSS-LoRA-Trainer adapter to ComfyUI LoRA naming.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter",
                        help="Adapter directory (adapter_model.safetensors + adapter_config.json) "
                             "or path to a weights file with adapter_config.json beside it")
    source.add_argument("--run-dir",
                        help="Training run directory: exports the newest checkpoints/step_* adapter, "
                             "falling back to the run root")
    parser.add_argument("--output",
                        help=f"Output .safetensors path (default: {COMFY_OUTPUT_NAME} beside the "
                             "adapter, or at the run root with --run-dir)")
    args = parser.parse_args(argv)

    try:
        if args.run_dir:
            summary = export_latest_checkpoint(args.run_dir, args.output)
        else:
            adapter = Path(args.adapter)
            output = Path(args.output) if args.output else (
                (adapter if adapter.is_dir() else adapter.parent) / COMFY_OUTPUT_NAME)
            summary = convert_adapter(adapter, output)
    except (ConversionError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        parser.exit(2, f"moss-lora-export-comfy: error: {exc}\n")

    print(
        f"Exported {summary['weights']} LoRA weights + {summary['alphas']} alpha scalars "
        f"({summary['keys']} keys, {summary['bytes']} bytes) from {summary['source']}\n"
        f"  -> {summary['output']}\n"
        f"  rank={summary['rank']} alpha={summary['alpha']} layers={summary['layers']} "
        f"targets={','.join(summary['targets'])}"
    )
    print("  sample keys: " + ", ".join(summary["sample_keys"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
