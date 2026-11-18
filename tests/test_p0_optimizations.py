"""P0 optimization tests: RoPE cache, timestep table, bit-parity with upstream, CUDA smoke.

The optimizations must be numerically invisible: every cached/fast path is checked
bit-identical against the original inline math (and against the vendored upstream
module when available). CUDA tests run on any available GPU with tiny models only.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from moss_lora.model import MOSSImageTransformer
from tests.helpers import TINY_KWARGS, tiny_transformer

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _reference_sinusoid(times: torch.Tensor, freq_dim: int, device) -> torch.Tensor:
    """The original upstream inline computation, verbatim."""
    times = torch.as_tensor(times, device=device, dtype=torch.float32).reshape(-1)
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


def _load_upstream():
    path = Path(os.environ.get("MOSS_UPSTREAM_PATH", "/tmp/moss_upstream/pipeline_moss_image.py"))
    if not path.is_file():
        pytest.skip(f"upstream pipeline not available at {path}")
    spec = importlib.util.spec_from_file_location("official_moss_image_p0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- timestep table


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_time_embedding_bit_identical(device):
    model = tiny_transformer().to(device)
    freq_dim = model.config.freq_dim
    # Integer timesteps (trainer path): table lookup must equal the inline fp64 math.
    for values in ([1, 500, 1000], [333], [0], [7, 7, 7, 7]):
        times = torch.tensor(values, dtype=torch.float32).expand(len(values))
        got = model._time_embedding_input(times, torch.device(device))
        want = _reference_sinusoid(values, freq_dim, device)
        assert torch.equal(got, want), values
    # Non-integer / out-of-range timesteps fall back to the compute path.
    for values in ([333.5, 0.25], [-5.0, 1001.0]):
        times = torch.tensor(values, dtype=torch.float32)
        got = model._time_embedding_input(times, torch.device(device))
        want = _reference_sinusoid(values, freq_dim, device)
        assert torch.equal(got, want), values
    # NaN timesteps stay NaN on both paths (torch.equal rejects NaN).
    times = torch.tensor([float("nan")])
    got = model._time_embedding_input(times, torch.device(device))
    assert torch.isnan(got).all()
    # CUDA-tensor input (sampling path) keeps the inline computation.
    if device == "cuda":
        times = torch.tensor([333.0, 700.0], device="cuda")
        got = model._time_embedding_input(times, torch.device("cuda"))
        assert torch.equal(got, _reference_sinusoid([333.0, 700.0], freq_dim, "cuda"))


# --------------------------------------------------------------------------- RoPE cache


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_rope_cache_hit_values_and_eviction(device):
    model = tiny_transformer().to(device)
    rope = model.rope
    a = rope(8, 8, 5, torch.device(device))
    b = rope(8, 8, 5, torch.device(device))
    assert a is b  # cache hit returns the identical tensor
    fresh = tiny_transformer().to(device)
    assert torch.equal(a, fresh.rope(8, 8, 5, torch.device(device)))
    other = rope(8, 8, 6, torch.device(device))
    assert other is not a
    assert not torch.equal(a[:, :64], other[:, :64])
    for height in range(4, 40, 2):  # > 16 distinct shapes
        rope(height, 8, 5, torch.device(device))
    assert len(rope._rope_cache) <= 16
    # _apply (device/dtype moves) clears the cache; values stay correct afterwards.
    model.to(device)
    assert len(rope._rope_cache) == 0
    assert torch.equal(rope(8, 8, 5, torch.device(device)), a)


# --------------------------------------------------------------------------- forward parity


def _parity_inputs(device, dtype):
    torch.manual_seed(11)
    latents = torch.randn(2, 8, 8, 8, device=device, dtype=dtype)
    context = torch.randn(2, 6, 32, device=device, dtype=dtype)
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3], device=device)
    return latents, context, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_forward_bit_identical_to_upstream(device, dtype):
    upstream = _load_upstream()
    latents, context, mask = _parity_inputs(device, dtype)
    torch.manual_seed(123)
    theirs = upstream.MOSSImageTransformer(**TINY_KWARGS).to(device=device, dtype=dtype).eval()
    torch.manual_seed(123)
    ours = MOSSImageTransformer(**TINY_KWARGS).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for timesteps in ([250, 750], [333.5, 0.25], torch.tensor([250.0, 750.0], device=device)):
            out_theirs = theirs(latents, timesteps, context, mask, return_dict=False)[0]
            out_ours = ours(latents, timesteps, context, mask, return_dict=False)[0]
            assert torch.equal(out_theirs, out_ours), (device, dtype, timesteps)
        # Second call exercises the RoPE cache / timestep table warm paths.
        out_theirs = theirs(latents, [250, 750], context, mask, return_dict=False)[0]
        out_ours = ours(latents, [250, 750], context, mask, return_dict=False)[0]
        assert torch.equal(out_theirs, out_ours)


@cuda_only
def test_cuda_gradient_checkpointing_grad_parity():
    model = tiny_transformer().cuda()
    latents = torch.randn(2, 8, 8, 8, device="cuda")
    context = torch.randn(2, 4, 32, device="cuda")
    mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")

    def gradients(use_checkpointing: bool):
        model.gradient_checkpointing = use_checkpointing
        model.train()
        for parameter in model.parameters():
            parameter.grad = None
        out = model(latents, [120, 880], context, mask, return_dict=False)[0]
        out.square().mean().backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    plain = gradients(False)
    checkpointed = gradients(True)
    assert len(plain) == len(checkpointed) > 0
    for a, b in zip(plain, checkpointed):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- CUDA trainer smoke


@cuda_only
def test_trainer_smoke_cuda_bf16(tmp_path, tiny_model_dir, tiny_cache):
    from moss_lora import trainer
    from moss_lora.config import load_config
    from tests.helpers import write_config

    config_path = write_config(
        tmp_path / "tiny_cuda.yaml",
        model={"path": str(tiny_model_dir), "dtype": "bf16"},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run"), "device": "cuda", "max_steps": 4},
    )
    result = trainer.train(load_config(config_path))
    assert result["steps"] == 4
    run_dir = Path(result["output_dir"])
    assert (run_dir / "adapter_model.safetensors").is_file()
    assert (run_dir / "adapter_ema.safetensors").is_file()
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2", "3", "4"]
    assert all(0.0 < float(row["loss"]) < 1e4 for row in rows)

    # Resume from the step-2 checkpoint and finish at 6 steps.
    config_path = write_config(
        tmp_path / "tiny_cuda_resume.yaml",
        model={"path": str(tiny_model_dir), "dtype": "bf16"},
        data={"cache_dir": str(tiny_cache)},
        train={"output_dir": str(tmp_path / "run"), "device": "cuda", "max_steps": 6},
    )
    resumed = trainer.train(load_config(config_path),
                            resume_path=str(run_dir / "checkpoints" / "step_000002"))
    assert resumed["steps"] == 6
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as handle:
        steps = [row["step"] for row in csv.DictReader(handle)]
    assert steps == ["1", "2", "3", "4", "3", "4", "5", "6"]
