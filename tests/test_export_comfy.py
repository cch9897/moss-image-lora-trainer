"""ComfyUI export tests: renaming, alpha scalars, validation, and loader compatibility."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from moss_lora import export_comfy
from moss_lora.export_comfy import (
    ConversionError,
    convert_adapter,
    export_latest_checkpoint,
    find_latest_adapter,
)
from moss_lora.lora import DEFAULT_TARGETS, get_adapters, inject_lora, save_adapter
from moss_lora.utils import sha256_file
from tests.helpers import tiny_transformer

COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", "ComfyUI")).resolve()


def make_adapter(tmp_path: Path, *, rank: int = 4, alpha: float = 8.0,
                 targets: tuple[str, ...] = DEFAULT_TARGETS) -> tuple[Path, dict[str, torch.Tensor]]:
    model = tiny_transformer().float()
    model.requires_grad_(False)
    inject_lora(model, rank=rank, alpha=alpha, targets=targets)
    torch.manual_seed(7)
    state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for path, wrapper in get_adapters(model).items():
            wrapper.lora_A.normal_(0.0, 0.02)
            wrapper.lora_B.normal_(0.0, 0.02)
            state[f"{path}.lora_A"] = wrapper.lora_A.detach().clone()
            state[f"{path}.lora_B"] = wrapper.lora_B.detach().clone()
    adapter_dir = tmp_path / "adapter"
    save_adapter(model, adapter_dir, base_model="tiny-base")
    return adapter_dir, state


def write_raw_adapter(tmp_path: Path, tensors: dict[str, torch.Tensor],
                      config: dict) -> Path:
    adapter_dir = tmp_path / "raw_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return adapter_dir


def test_export_renames_keys_and_preserves_values(tmp_path: Path):
    adapter_dir, state = make_adapter(tmp_path)
    weights_before = sha256_file(adapter_dir / "adapter_model.safetensors")
    config_before = (adapter_dir / "adapter_config.json").read_bytes()

    output = tmp_path / "comfy.safetensors"
    summary = convert_adapter(adapter_dir, output)

    converted = load_file(str(output))
    expected = set()
    for layer in range(2):
        for target in DEFAULT_TARGETS:
            base = f"diffusion_model.layers.{layer}.proj.{target}"
            expected |= {f"{base}.lora_A.weight", f"{base}.lora_B.weight", f"{base}.alpha"}
    assert set(converted) == expected
    assert not any(key.startswith("layers.") for key in converted)

    base = "diffusion_model.layers.0.proj.q"
    assert converted[f"{base}.lora_A.weight"].shape == (4, 64)
    assert converted[f"{base}.lora_B.weight"].shape == (64, 4)
    assert converted[f"{base}.lora_A.weight"].dtype == torch.float32
    assert torch.equal(converted[f"{base}.lora_A.weight"], state["layers.0.proj.q.lora_A"])
    assert torch.equal(converted[f"{base}.lora_B.weight"], state["layers.0.proj.q.lora_B"])

    for key, tensor in converted.items():
        if key.endswith(".alpha"):
            assert tensor.dtype == torch.float32
            assert tensor.ndim == 0
            assert tensor.item() == 8.0
    assert summary["keys"] == 2 * len(DEFAULT_TARGETS) * 3
    assert summary["weights"] == 2 * len(DEFAULT_TARGETS) * 2
    assert summary["alphas"] == 2 * len(DEFAULT_TARGETS)

    with safe_open(str(output), framework="pt") as handle:
        assert handle.metadata() == {
            "format": "pt",
            "moss_lora_comfy": "1",
            "base_model": "tiny-base",
        }

    # The source adapter must be untouched.
    assert sha256_file(adapter_dir / "adapter_model.safetensors") == weights_before
    assert (adapter_dir / "adapter_config.json").read_bytes() == config_before


def test_rejects_unknown_keys(tmp_path: Path):
    config = {"r": 4, "alpha": 4.0, "targets": ["q"]}
    tensors = {
        "layers.0.proj.q.lora_A": torch.zeros(4, 64),
        "layers.0.proj.q.lora_B": torch.zeros(64, 4),
        "layers.0.proj.q.lora_X": torch.zeros(4, 4),
    }
    adapter_dir = write_raw_adapter(tmp_path, tensors, config)
    output = tmp_path / "comfy.safetensors"
    with pytest.raises(ConversionError, match="not MOSS LoRA A/B pairs"):
        convert_adapter(adapter_dir, output)
    assert not output.exists()


def test_rejects_missing_half_pair(tmp_path: Path):
    config = {"r": 4, "alpha": 4.0, "targets": ["q"]}
    tensors = {"layers.0.proj.q.lora_A": torch.zeros(4, 64)}
    adapter_dir = write_raw_adapter(tmp_path, tensors, config)
    with pytest.raises(ConversionError, match="missing A/B pairs"):
        convert_adapter(adapter_dir, tmp_path / "comfy.safetensors")


def test_rejects_rank_shape_mismatch(tmp_path: Path):
    config = {"r": 4, "alpha": 4.0, "targets": ["q"]}
    tensors = {
        "layers.0.proj.q.lora_A": torch.zeros(4, 64),
        "layers.0.proj.q.lora_B": torch.zeros(64, 8),
    }
    adapter_dir = write_raw_adapter(tmp_path, tensors, config)
    with pytest.raises(ConversionError, match="lora_B rank dimension is 8"):
        convert_adapter(adapter_dir, tmp_path / "comfy.safetensors")


def test_rejects_missing_config(tmp_path: Path):
    adapter_dir = tmp_path / "weights_only"
    adapter_dir.mkdir()
    save_file({"layers.0.proj.q.lora_A": torch.zeros(4, 64)},
              str(adapter_dir / "adapter_model.safetensors"))
    with pytest.raises(FileNotFoundError, match="Adapter config not found"):
        convert_adapter(adapter_dir, tmp_path / "comfy.safetensors")


def test_find_latest_adapter_prefers_newest_complete_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    old, _ = make_adapter(tmp_path / "old")
    new, _ = make_adapter(tmp_path / "new")
    shutil.copytree(old, run / "checkpoints" / "step_000002")
    shutil.copytree(new, run / "checkpoints" / "step_000010")
    # An incomplete newer directory must be ignored.
    (run / "checkpoints" / "step_000012").mkdir()

    assert find_latest_adapter(run) == run / "checkpoints" / "step_000010"
    summary = export_latest_checkpoint(run)
    assert Path(summary["output"]) == run / "comfyui_lora.safetensors"
    source = Path(summary["source"])
    assert source == run / "checkpoints" / "step_000010" / "adapter_model.safetensors"
    assert summary["keys"] == 2 * len(DEFAULT_TARGETS) * 3


def test_export_latest_checkpoint_falls_back_to_run_root(tmp_path: Path):
    run = tmp_path / "run"
    adapter_dir, _ = make_adapter(tmp_path / "src")
    run.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(adapter_dir / name, run / name)

    summary = export_latest_checkpoint(run)
    assert Path(summary["output"]) == run / "comfyui_lora.safetensors"
    assert (run / "comfyui_lora.safetensors").is_file()


def test_cli_run_dir_exports_latest_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    adapter_dir, _ = make_adapter(tmp_path / "src")
    shutil.copytree(adapter_dir, run / "checkpoints" / "step_000007")

    assert export_comfy.main(["--run-dir", str(run)]) == 0
    output = run / "comfyui_lora.safetensors"
    assert output.is_file()
    assert all(key.startswith("diffusion_model.") for key in load_file(str(output)))


def comfy_modules():
    """Import ComfyUI's LoRA loader, or skip when its runtime is unavailable."""
    if not COMFY_ROOT.is_dir():
        pytest.skip(f"ComfyUI not found at {COMFY_ROOT}")
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    try:
        from comfy.lora import load_lora
        from comfy.weight_adapter.lora import LoRAAdapter
    except Exception as exc:  # noqa: BLE001 - missing host deps must skip, not fail
        pytest.skip(f"ComfyUI import unavailable in this environment: {exc}")
    return LoRAAdapter, load_lora


def test_comfyui_loads_converted_adapter(tmp_path: Path, caplog):
    LoRAAdapter, load_lora = comfy_modules()
    adapter_dir, _ = make_adapter(tmp_path)
    output = tmp_path / "comfy.safetensors"
    convert_adapter(adapter_dir, output)
    lora = load_file(str(output))

    base = "diffusion_model.layers.0.proj.q"
    loaded: set[str] = set()
    adapter = LoRAAdapter.load(base, lora, lora[f"{base}.alpha"].item(), None, loaded)
    assert adapter is not None
    assert loaded == {f"{base}.lora_A.weight", f"{base}.lora_B.weight"}

    alpha = lora[f"{base}.alpha"].item()
    rank = lora[f"{base}.lora_A.weight"].shape[0]
    weight = torch.zeros(64, 64)
    updated = adapter.calculate_weight(weight, base, 1.0, 1.0, None, lambda value: value)
    expected = (lora[f"{base}.lora_B.weight"].float() @ lora[f"{base}.lora_A.weight"].float())
    expected *= alpha / rank
    assert torch.allclose(updated, expected, atol=1e-6)

    # ComfyUI builds this map from the model state dict: base key -> <base>.weight.
    to_load = {key.rsplit(".lora_", 1)[0]: key.rsplit(".lora_", 1)[0] + ".weight"
               for key in lora if ".lora_" in key}
    with caplog.at_level(logging.WARNING):
        patch = load_lora(lora, to_load)
    assert set(patch) == set(to_load.values())
    assert all(isinstance(value, LoRAAdapter) for value in patch.values())
    assert not [record for record in caplog.records if "lora key not loaded" in record.message]
