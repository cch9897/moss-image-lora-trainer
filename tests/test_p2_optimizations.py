"""P2 tests: foreach EMA parity, selective activation checkpointing parity + smoke."""

from __future__ import annotations

import pytest
import torch

from moss_lora.trainer import ParamEMA
from tests.helpers import tiny_transformer

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_ema_foreach_matches_loop_bitwise():
    torch.manual_seed(3)
    named = [(f"p{i}", torch.nn.Parameter(torch.randn(16, 8))) for i in range(20)]
    ema = ParamEMA(named, decay=0.9)
    shadow_ref = {name: p.detach().clone() for name, p in named}
    for _ in range(5):
        with torch.no_grad():
            for _, p in named:
                p.add_(torch.randn_like(p) * 0.01)
        ema.update(named)
        with torch.no_grad():
            for name, p in named:
                shadow_ref[name].mul_(0.9).add_(p.detach(), alpha=0.1)
    for name, _ in named:
        assert torch.equal(ema.shadow[name], shadow_ref[name])


@pytest.mark.parametrize("policy", ["attention", "matmul"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_sac_gradient_parity(device, batched, policy):
    torch.manual_seed(11)
    model = tiny_transformer().to(device)
    latents = torch.randn(2, 8, 8, 8, device=device)
    context = torch.randn(2, 6, 32, device=device)
    mask = torch.zeros(2, 6, dtype=torch.bool, device=device)
    mask[0] = True
    mask[1, :3] = True
    model.train()
    if batched:
        model.enable_batched_forward()

    def gradients(sac: str | None):
        model.gradient_checkpointing = True
        if sac is None:
            model.disable_selective_checkpointing()
        else:
            model.enable_selective_checkpointing(sac)
        for p in model.parameters():
            p.grad = None
        out = model(latents, [120, 880], context, mask, return_dict=False)[0]
        out.square().mean().backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    full = gradients(None)
    selective = gradients(policy)
    assert len(full) == len(selective) > 0
    for a, b in zip(full, selective):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), policy


@cuda_only
@pytest.mark.parametrize("policy", ["attention", "matmul"])
def test_sac_with_per_block_compile_cuda(policy):
    from moss_lora.lora import inject_lora
    from moss_lora.model import MOSSImageTransformer

    torch.manual_seed(5)
    model = MOSSImageTransformer(dim=256, ffn_dim=512, num_heads=4, num_layers=4,
                                 text_dim=64, freq_dim=32, latent_channels=8)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.requires_grad_(False)
    inject_lora(model, rank=8, alpha=8)
    model.enable_gradient_checkpointing()
    model.enable_batched_forward()
    model.enable_selective_checkpointing(policy)
    model.train()
    for index, layer in enumerate(model.layers):
        model.layers[index] = torch.compile(layer, dynamic=True)
    params = [p for p in model.parameters() if p.requires_grad]

    torch.manual_seed(9)
    latents = torch.randn(2, 8, 8, 8, device="cuda")
    context = torch.randn(2, 6, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3], device="cuda")

    def step():
        for p in params:
            p.grad = None
        out = model(latents, [200, 800], context, mask, return_dict=False)[0]
        loss = out.float().square().mean()
        loss.backward()
        return float(loss.detach()), [p.grad.clone() for p in params]

    loss_a, grads_a = step()
    loss_b, grads_b = step()  # warm compiled graph
    assert abs(loss_a - loss_b) < 1e-2 * max(1.0, loss_a)
    for a, b in zip(grads_a, grads_b):
        assert torch.isfinite(a).all() and torch.allclose(a, b, rtol=3e-2, atol=3e-2)


@cuda_only
def test_trainer_sac_compile_smoke_cuda(tmp_path, tiny_model_dir, tiny_cache):
    from moss_lora import trainer
    from moss_lora.config import load_config
    from tests.helpers import write_config

    config_path = write_config(
        tmp_path / "tiny_sac.yaml",
        model={"path": str(tiny_model_dir), "dtype": "bf16"},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run"), "device": "cuda", "max_steps": 3,
               "compile": True, "selective_checkpointing": "matmul"},
    )
    result = trainer.train(load_config(config_path))
    assert result["steps"] == 3
    assert (tmp_path / "run" / "adapter_model.safetensors").is_file()


def test_selective_checkpointing_config_validation(tmp_path):
    from moss_lora.config import TrainConfig

    with pytest.raises(ValueError, match="off/attention/matmul"):
        TrainConfig(selective_checkpointing="bogus")
    with pytest.raises(ValueError, match="requires gradient_checkpointing"):
        TrainConfig(selective_checkpointing="matmul", gradient_checkpointing=False)
    assert TrainConfig(selective_checkpointing="attention").selective_checkpointing == "attention"
