"""Sample/merge CLI tests, component path resolution and TextEncoder device placement."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import moss_lora.sample as sample_module
from moss_lora.merge import main as merge_main
from moss_lora.model import MOSSImageTransformer
from moss_lora.sample import main as sample_main
from moss_lora.sample import resolve_adapter_weights
from moss_lora.text import TextEncoder
from moss_lora.utils import resolve_pretrained_target, resolve_tokenizer_path
from tests.helpers import FakeTextEncoder, make_tiny_model_dir

# --------------------------------------------------------------------------- resolution


def test_resolve_pretrained_target_full_checkout(tmp_path):
    make_tiny_model_dir(tmp_path / "model")
    (tmp_path / "model" / "vae").mkdir()
    (tmp_path / "model" / "vae" / "config.json").write_text("{}\n", encoding="utf-8")
    root = tmp_path / "model"
    # Full checkout: components load from their subfolder with subfolder=None (no double nesting).
    path, subfolder = resolve_pretrained_target(root, "vae")
    assert subfolder is None
    assert Path(path) == root / "vae"
    assert (Path(path) / "config.json").is_file()
    path, subfolder = resolve_pretrained_target(root, "transformer")
    assert Path(path) == root / "transformer" and subfolder is None
    path, subfolder = resolve_pretrained_target(root, "text_encoder")
    assert Path(path) == root / "text_encoder" and subfolder is None
    # Direct component dir: root itself.
    path, subfolder = resolve_pretrained_target(root / "transformer", "transformer")
    assert Path(path) == root / "transformer" and subfolder is None
    path, subfolder = resolve_pretrained_target(root / "vae", "vae")
    assert Path(path) == root / "vae" and subfolder is None
    path, subfolder = resolve_pretrained_target(root / "text_encoder", "text_encoder")
    assert Path(path) == root / "text_encoder" and subfolder is None
    # Hub id passthrough keeps the subfolder.
    assert resolve_pretrained_target("OpenMOSS-Team/MOSS-Image-Base", "vae") == (
        "OpenMOSS-Team/MOSS-Image-Base", "vae")


def test_load_sampling_vae_component_resolution(tmp_path, monkeypatch):
    from diffusers import AutoencoderKLFlux2

    from moss_lora.sample import load_sampling_vae

    calls: list[tuple[str, str | None]] = []

    def fake_from_pretrained(path, **kwargs):
        calls.append((str(path), kwargs.get("subfolder")))
        return nn.Linear(1, 1)

    monkeypatch.setattr(AutoencoderKLFlux2, "from_pretrained", fake_from_pretrained)
    root = make_tiny_model_dir(tmp_path / "model")

    # Local checkout without vae/: no VAE, and no load attempt.
    assert load_sampling_vae(str(root), "cpu", True, None) is None
    assert calls == []

    (root / "vae").mkdir()
    (root / "vae" / "config.json").write_text("{}\n", encoding="utf-8")
    assert load_sampling_vae(str(root), "cpu", True, None) is not None
    assert calls[-1] == (str(root / "vae"), None)
    assert load_sampling_vae(str(root / "vae"), "cpu", True, None) is not None
    assert calls[-1] == (str(root / "vae"), None)
    # Hub ids are not local dirs: from_pretrained resolves the vae/ subfolder itself.
    assert load_sampling_vae("OpenMOSS-Team/MOSS-Image-Base", "cpu", True, "rev") is not None
    assert calls[-1] == ("OpenMOSS-Team/MOSS-Image-Base", "vae")


def test_resolve_tokenizer_path_layouts(tmp_path):
    root = tmp_path / "model"
    (root / "tokenizer").mkdir(parents=True)
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    assert resolve_tokenizer_path(root) == str(root / "tokenizer")

    plain = tmp_path / "te"
    plain.mkdir()
    (plain / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    assert resolve_tokenizer_path(plain) == str(plain)

    assert resolve_tokenizer_path("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"  # hub id passthrough

    bare = tmp_path / "nothing"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_tokenizer_path(bare)


def test_text_encoder_moves_model_to_device():
    model = nn.Linear(4, 4)
    encoder = TextEncoder(model, tokenizer=None, device="meta")
    assert next(encoder.model.parameters()).device.type == "meta"


# --------------------------------------------------------------------------- sample CLI


def test_sample_cli_latent_output(tmp_path, tiny_model_dir, monkeypatch):
    encoder = FakeTextEncoder(text_dim=32)
    seen = {}
    def fake_loader(path, **kwargs):
        seen["path"] = str(path)
        seen["kwargs"] = kwargs
        return encoder

    monkeypatch.setattr(sample_module, "load_text_encoder", fake_loader)
    output = tmp_path / "out"
    sample_main([
        "--model", str(tiny_model_dir),
        "--prompt", "a red circle on white background", "a blue square",
        "--steps", "2", "--cfg", "3.0", "--skip-layer", "-1",
        "--height", "256", "--width", "256", "--seed", "5",
        "--output-type", "latent", "--output", str(output),
        "--device", "cpu", "--dtype", "fp32",
    ])
    assert seen["path"].endswith("text_encoder")
    assert encoder.unloaded  # TE freed before the transformer loads
    files = sorted(output.glob("image_*.pt"))
    assert len(files) == 2
    tensor = torch.load(files[0], weights_only=False)
    assert tensor.shape == (8, 16, 16) and tensor.dtype.is_floating_point


def test_sample_cli_skip_layer_validation(tmp_path, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(sample_module, "load_text_encoder", lambda path, **kwargs: FakeTextEncoder(32))
    with pytest.raises(ValueError, match="must be a directory"):
        sample_main([
            "--model", str(tiny_model_dir), "--prompt", "a", "b", "--steps", "1", "--cfg", "3.0",
            "--skip-layer", "-1", "--height", "128", "--width", "128", "--output-type", "latent",
            "--output", str(tmp_path / "single.pt"), "--device", "cpu", "--dtype", "fp32",
        ])
    with pytest.raises(ValueError, match="block indices"):
        sample_main([
            "--model", str(tiny_model_dir), "--prompt", "a", "--steps", "1", "--cfg", "3.0",
            "--skip-layer", "22", "--height", "128", "--width", "128", "--output-type", "latent",
            "--output", str(tmp_path / "out"), "--device", "cpu", "--dtype", "fp32",
        ])


def test_resolve_adapter_weights(tmp_path):
    (tmp_path / "adapter_model.safetensors").touch()
    (tmp_path / "adapter_ema.safetensors").touch()
    assert resolve_adapter_weights(tmp_path, use_ema=False).name == "adapter_model.safetensors"
    assert resolve_adapter_weights(str(tmp_path / "adapter_model.safetensors"), False).is_file()
    assert resolve_adapter_weights(tmp_path, use_ema=True).name == "adapter_ema.safetensors"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_adapter_weights(empty, use_ema=False)
    with pytest.raises(FileNotFoundError):
        resolve_adapter_weights(empty, use_ema=True)


# --------------------------------------------------------------------------- merge CLI


def test_merge_cli_equivalence(tmp_path, tiny_model_dir, tiny_config):
    from moss_lora import trainer
    from moss_lora.config import load_config
    from moss_lora.lora import get_adapters, inject_lora

    # Train a tiny adapter so the merge has non-trivial weights.
    config = load_config(tiny_config)
    config.train.max_steps = 2
    config.train.ema_decay = 0.0
    trainer.train(config, resume_path=None)
    adapter_dir = Path(config.train.output_dir)

    model = MOSSImageTransformer.from_pretrained(str(tiny_model_dir / "transformer"),
                                                 torch_dtype=torch.float32).eval()
    inject_lora(model, rank=config.lora.rank, alpha=config.lora.alpha, targets=config.lora.targets)
    from moss_lora.lora import load_adapter

    load_adapter(model, adapter_dir)
    with torch.no_grad():
        for wrapper in get_adapters(model).values():
            assert torch.count_nonzero(wrapper.lora_B) > 0  # actually trained
    x = torch.randn(1, 8, 8, 8)
    ctx = torch.randn(1, 3, 32)
    with torch.no_grad():
        adapted = model(x, [400.0], ctx, return_dict=False)[0]

    merged_dir = tmp_path / "merged"
    merge_main(["--model", str(tiny_model_dir), "--lora", str(adapter_dir),
                "--output", str(merged_dir), "--dtype", "fp32"])
    merged = MOSSImageTransformer.from_pretrained(str(merged_dir), torch_dtype=torch.float32).eval()
    assert "layers.0.proj.q.weight" in set(merged.state_dict())
    with torch.no_grad():
        out = merged(x, [400.0], ctx, return_dict=False)[0]
    assert torch.allclose(adapted, out, rtol=1e-4, atol=1e-5)
