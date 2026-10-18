"""LoRA wrapper tests: zero-init identity, math, save/load, merge, validation."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from moss_lora.lora import (
    LoRALinear,
    get_adapters,
    inject_lora,
    load_adapter,
    load_lora_into_model,
    lora_state_dict,
    lora_target_paths,
    merge_lora,
    save_adapter,
)
from tests.helpers import tiny_transformer


def wrapped_model():
    model = tiny_transformer().float()
    model.requires_grad_(False)  # matches the trainer: freeze base before injection
    adapters = inject_lora(model, rank=4, alpha=4, dropout=0.0)
    return model, adapters


def test_zero_init_identity():
    model, adapters = wrapped_model()
    model.eval()
    assert len(adapters) == 2 * 6  # 2 layers x 6 targets
    x = torch.randn(3, 64)
    with torch.no_grad():
        wrapper = adapters["layers.0.proj.q"]
        assert torch.equal(wrapper(x), wrapper.base(x))
        assert torch.count_nonzero(wrapper.lora_B) == 0
        assert wrapper.lora_A.dtype == torch.float32
        assert wrapper.lora_B.dtype == torch.float32


def test_only_lora_params_trainable():
    model, adapters = wrapped_model()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 2 * len(adapters)
    assert all(name.endswith((".lora_A", ".lora_B")) for name in trainable)
    base_params = [name for name, _ in model.named_parameters() if not name.endswith((".lora_A", ".lora_B"))]
    assert all(not parameter.requires_grad
               for name, parameter in model.named_parameters() if name in set(base_params))


def test_forward_matches_reference_math():
    model, adapters = wrapped_model()
    wrapper = adapters["layers.1.proj.ffn_down"]
    x = torch.randn(5, 128)
    expected = wrapper.base(x) + ((x @ wrapper.lora_A.T) @ wrapper.lora_B.T) * wrapper.scaling
    model.eval()
    with torch.no_grad():
        actual = wrapper(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_grads_accumulate_in_fp32():
    model, adapters = wrapped_model()
    model.train()
    x = torch.randn(2, 64).to(torch.bfloat16)
    adapters["layers.0.proj.q"](x).square().mean().backward()
    assert adapters["layers.0.proj.q"].lora_A.grad.dtype == torch.float32
    assert adapters["layers.0.proj.q"].lora_B.grad.dtype == torch.float32
    assert adapters["layers.0.proj.q"].base.weight.grad is None


def test_wrapper_exposes_weight_dtype():
    model, adapters = wrapped_model()
    assert adapters["layers.0.proj.q"].weight.dtype == torch.float32
    assert adapters["layers.0.proj.q"].in_features == 64
    assert adapters["layers.0.proj.q"].out_features == 64
    assert adapters["layers.0.proj.ffn_down"].out_features == 64
    assert adapters["layers.0.proj.ffn_up_gate"].weight.shape == (256, 64)


def test_save_load_roundtrip(tmp_path):
    model, _ = wrapped_model()
    with torch.no_grad():
        for wrapper in get_adapters(model).values():
            wrapper.lora_B.normal_(0, 0.1)
    out = save_adapter(model, tmp_path, base_model="tiny", revision="rev1")
    assert out.name == "adapter_model.safetensors"
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == 4 and config["alpha"] == 4
    assert config["targets"] == ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
    assert config["base_model"] == "tiny" and config["revision"] == "rev1"

    other, _ = wrapped_model()
    load_adapter(other, tmp_path)
    for (path_a, wrapper_a), (path_b, wrapper_b) in zip(get_adapters(model).items(),
                                                        get_adapters(other).items()):
        assert path_a == path_b
        assert torch.equal(wrapper_a.lora_A, wrapper_b.lora_A)
        assert torch.equal(wrapper_a.lora_B, wrapper_b.lora_B)


def test_load_rejects_missing_and_unexpected_keys(tmp_path):
    model, _ = wrapped_model()
    save_adapter(model, tmp_path)
    from safetensors.torch import save_file

    state = lora_state_dict(model)
    del state["layers.0.proj.q.lora_B"]
    state["bogus.key.lora_A"] = torch.zeros(1)
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp_path / "broken.safetensors"))

    with pytest.raises(ValueError, match="missing"):
        load_adapter(model, tmp_path / "broken.safetensors")

    other = tiny_transformer().float()
    other.requires_grad_(False)
    inject_lora(other, rank=8, alpha=8)  # wrong rank -> shape mismatch
    with pytest.raises(RuntimeError):
        load_adapter(other, tmp_path)


def test_merge_equivalence_and_key_restore(tmp_path):
    model, _ = wrapped_model()
    with torch.no_grad():
        for wrapper in get_adapters(model).values():
            wrapper.lora_B.normal_(0, 0.05)
    model.eval()
    x = torch.randn(2, 8, 8, 8)
    ctx = torch.randn(2, 3, 32)
    with torch.no_grad():
        adapted = model(x, [500.0], ctx, return_dict=False)[0]

    state_before_keys = set(lora_state_dict(model))
    assert any("layers.0.proj.q.lora_A" == key for key in state_before_keys)
    merge_lora(model)
    assert get_adapters(model) == {}
    assert not hasattr(model, "_moss_lora_adapters")
    assert isinstance(model.layers[0].proj.q, nn.Linear)  # original key names restored
    state_keys = set(model.state_dict())
    assert "layers.0.proj.q.weight" in state_keys
    assert not any("lora_A" in key for key in state_keys)
    with torch.no_grad():
        merged = model(x, [500.0], ctx, return_dict=False)[0]
    assert torch.allclose(adapted, merged, rtol=1e-4, atol=1e-5)


def test_merge_without_adapters_raises():
    model = tiny_transformer()
    with pytest.raises(RuntimeError, match="No LoRA adapters"):
        merge_lora(model)


def test_inject_twice_raises():
    model, _ = wrapped_model()
    with pytest.raises(RuntimeError, match="already injected"):
        inject_lora(model, rank=4, alpha=4)


def test_target_paths_and_validation():
    paths = lora_target_paths(2, ["q", "o"])
    assert paths == ["layers.0.proj.q", "layers.0.proj.o", "layers.1.proj.q", "layers.1.proj.o"]
    with pytest.raises(ValueError, match="Unknown LoRA targets"):
        lora_target_paths(1, ["nope"])
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(8, 8), rank=0, alpha=1)
    with pytest.raises(TypeError):
        LoRALinear(nn.Identity(), rank=2, alpha=2)


def test_optional_targets():
    model = tiny_transformer().float()
    adapters = inject_lora(model, rank=2, alpha=2, targets=["q"], text_projection=True, final_linear=True)
    assert set(adapters) == {"text_embedder.proj", "final_linear_video"} | {
        f"layers.{i}.proj.q" for i in range(2)}
    x = torch.randn(1, 8, 8, 8)
    out = model(x, [100.0], torch.randn(1, 3, 32), return_dict=False)[0]
    assert out.shape == x.shape


def test_load_lora_into_model_uses_adapter_config(tmp_path):
    model = tiny_transformer().float()
    source, _ = wrapped_model()
    with torch.no_grad():
        for wrapper in get_adapters(source).values():
            wrapper.lora_B.normal_(0, 0.1)
    save_adapter(source, tmp_path, base_model="tiny", revision="rev9")
    load_lora_into_model(model, tmp_path, base_model="tiny")
    with pytest.raises(ValueError, match="base_model"):
        load_lora_into_model(tiny_transformer().float(), tmp_path, base_model="other-model")
