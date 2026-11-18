"""P1 batched-forward tests: equivalence to the per-sample loop, padding safety, compile.

The batched path must be mathematically equivalent (not bit-identical: batched
GEMMs/SDPA round differently). RoPE values for real tokens must be bit-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moss_lora.model import MOSSImageTransformer
from tests.helpers import tiny_transformer

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _inputs(device, dtype, batch=3):
    torch.manual_seed(11)
    latents = torch.randn(batch, 8, 8, 8, device=device, dtype=dtype)
    context = torch.randn(batch, 6, 32, device=device, dtype=dtype)
    mask = torch.zeros(batch, 6, dtype=torch.bool, device=device)
    for row, length in enumerate([6, 3, 1][:batch]):
        mask[row, :length] = True
    return latents, context, mask


def _tolerances(dtype):
    return (dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32
            else dict(rtol=2e-2, atol=2e-2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_batched_matches_per_sample_forward(device, dtype):
    if device == "cpu" and dtype == torch.bfloat16:
        pytest.skip("bf16 CPU SDPA masking is not the target path")
    latents, context, mask = _inputs(device, dtype)
    model = tiny_transformer().to(device=device, dtype=dtype).eval()
    timesteps = [250, 750, 100]
    with torch.no_grad():
        per_sample = model(latents, timesteps, context, mask, return_dict=False)[0]
        batched = model.forward_batched(latents, timesteps, context, mask, return_dict=False)[0]
    assert torch.allclose(per_sample, batched, **_tolerances(dtype))
    # B=1 and skip_layers also match.
    with torch.no_grad():
        one_a = model(latents[:1], [250], context[:1], mask[:1], return_dict=False)[0]
        one_b = model.forward_batched(latents[:1], [250], context[:1], mask[:1], return_dict=False)[0]
        skip_a = model(latents, timesteps, context, mask, return_dict=False,
                       skip_layers=[1])[0]
        skip_b = model.forward_batched(latents, timesteps, context, mask, return_dict=False,
                                       skip_layers=[1])[0]
    assert torch.allclose(one_a, one_b, **_tolerances(dtype))
    assert torch.allclose(skip_a, skip_b, **_tolerances(dtype))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_batched_padding_cannot_leak(device):
    latents, context, mask = _inputs(device, torch.float32)
    model = tiny_transformer().to(device).eval()
    with torch.no_grad():
        base = model.forward_batched(latents, [250, 750, 100], context, mask, return_dict=False)[0]
        tweaked = context.clone()
        tweaked[1, 3:] += 10.0
        tweaked[2, 1:] -= 7.0
        other = model.forward_batched(latents, [250, 750, 100], tweaked, mask, return_dict=False)[0]
    assert torch.allclose(base, other, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_batched_gradient_parity(device, checkpointing):
    latents, context, mask = _inputs(device, torch.float32)
    model = tiny_transformer().to(device)
    model.gradient_checkpointing = checkpointing
    model.train()

    def gradients(batched: bool):
        for parameter in model.parameters():
            parameter.grad = None
        fn = model.forward_batched if batched else model
        out = fn(latents, [120, 880, 300], context, mask, return_dict=False)[0]
        out.square().mean().backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    per_sample = gradients(False)
    batched = gradients(True)
    assert len(per_sample) == len(batched) > 0
    for a, b in zip(per_sample, batched):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_batched_rope_bit_identical_for_real_tokens(device):
    model = tiny_transformer().to(device)
    lengths = [6, 3, 1]
    mask = torch.zeros(3, 6, dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    batched = model.rope.forward_batched(8, 8, 6, mask.sum(1), torch.device(device))
    for row, length in enumerate(lengths):
        scalar = model.rope(8, 8, length, torch.device(device))[0]
        assert torch.equal(batched[row, :64], scalar[:64])  # image part: f = true length
        assert torch.equal(batched[row, 64:64 + length], scalar[64:])  # real text part


def test_batched_flag_dispatch_and_default_off():
    model = tiny_transformer().eval()
    assert model._batched_forward is False
    latents, context, mask = _inputs("cpu", torch.float32)
    with torch.no_grad():
        default = model(latents, [250, 750, 100], context, mask, return_dict=False)[0]
        model.enable_batched_forward()
        dispatched = model(latents, [250, 750, 100], context, mask, return_dict=False)[0]
        direct = model.forward_batched(latents, [250, 750, 100], context, mask, return_dict=False)[0]
        model.disable_batched_forward()
        restored = model(latents, [250, 750, 100], context, mask, return_dict=False)[0]
    assert torch.equal(dispatched, direct)
    assert torch.equal(default, restored)
    assert torch.allclose(default, dispatched, rtol=1e-5, atol=1e-6)


@cuda_only
def test_batched_forward_compile_cuda():
    torch.manual_seed(5)
    model = MOSSImageTransformer(dim=256, ffn_dim=512, num_heads=4, num_layers=4,
                                 text_dim=64, freq_dim=32, latent_channels=8)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.requires_grad_(False)
    from moss_lora.lora import inject_lora

    inject_lora(model, rank=8, alpha=8)
    model.enable_gradient_checkpointing()
    model.enable_batched_forward()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    compiled = torch.compile(model, dynamic=True)

    torch.manual_seed(9)
    latents = torch.randn(2, 8, 8, 8, device="cuda")
    context = torch.randn(2, 6, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3], device="cuda")

    def step(fn):
        for p in params:
            p.grad = None
        out = fn(latents, [200, 800], context, mask, return_dict=False)[0]
        loss = out.float().square().mean()
        loss.backward()
        return float(loss.detach()), [p.grad.clone() for p in params]

    eager_loss, eager_grads = step(model)
    compiled_loss, compiled_grads = step(compiled)
    assert abs(eager_loss - compiled_loss) < 1e-2 * max(1.0, eager_loss)
    for a, b in zip(eager_grads, compiled_grads):
        assert torch.allclose(a, b, rtol=3e-2, atol=3e-2)
    # A second shape must not crash (dynamic=True) and must stay finite.
    latents2 = torch.randn(2, 8, 16, 8, device="cuda")
    out = compiled(latents2, [300, 700], context, mask, return_dict=False)[0]
    assert out.shape == (2, 8, 16, 8) and torch.isfinite(out.float()).all()


def test_trainer_batched_forward_flag(tmp_path, tiny_model_dir, tiny_cache):
    """batched_forward=false must restore the per-sample path end-to-end."""
    import csv
    from pathlib import Path

    from moss_lora import trainer
    from moss_lora.config import load_config
    from tests.helpers import write_config

    config_path = write_config(
        tmp_path / "tiny_unbatched.yaml",
        model={"path": str(tiny_model_dir)},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run"), "batched_forward": False},
    )
    config = load_config(config_path)
    assert config.train.batched_forward is False
    result = trainer.train(config)
    assert result["steps"] == 2
    with open(Path(result["output_dir"]) / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(0.0 < float(row["loss"]) < 1e4 for row in rows)


@cuda_only
def test_trainer_compile_per_block_cuda(tmp_path, tiny_model_dir, tiny_cache):
    """compile=true compiles DiT blocks; adapter/EMA keys must stay clean and resume works."""
    from safetensors.torch import load_file

    from moss_lora import trainer
    from moss_lora.config import load_config
    from tests.helpers import write_config

    overrides = dict(
        model={"path": str(tiny_model_dir), "dtype": "bf16"},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run"), "device": "cuda",
               "max_steps": 4, "compile": True},
    )
    result = trainer.train(load_config(write_config(tmp_path / "c1.yaml", **overrides)))
    assert result["steps"] == 4
    run_dir = Path(result["output_dir"])
    raw = load_file(str(run_dir / "adapter_model.safetensors"))
    assert raw and all("_orig_mod" not in key for key in raw)
    ema = load_file(str(run_dir / "adapter_ema.safetensors"))
    assert ema and all("_orig_mod" not in key for key in ema)
    overrides["train"]["max_steps"] = 8
    resumed = trainer.train(load_config(write_config(tmp_path / "c2.yaml", **overrides)),
                            resume_path=str(run_dir / "checkpoints" / "step_000004"))
    assert resumed["steps"] == 8
