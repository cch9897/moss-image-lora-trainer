"""Trainer smoke tests: tiny end-to-end run, resume, timestep sampling."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest
import torch
import yaml

from moss_lora import trainer
from moss_lora.config import load_config
from moss_lora.lora import inject_lora
from moss_lora.trainer import NUM_TIMESTEPS, sample_timestep
from tests.helpers import base_config_dict, tiny_transformer


class Schedule:
    """Mimics TrainConfig fields for sample_timestep unit tests."""

    def __init__(self, **kwargs):
        values = dict(timestep_sampling="logit_normal", logit_normal_mean=-1.09, logit_normal_std=0.95,
                      sigma_cutoff=0.95, uniform_mix=0.01)
        values.update(kwargs)
        self.__dict__.update(values)


def test_sample_timestep_bounds_and_determinism():
    assert sample_timestep(random.Random(0), Schedule()) == sample_timestep(random.Random(0), Schedule())
    rng = random.Random(1)
    draws = [sample_timestep(rng, Schedule()) for _ in range(2000)]
    assert all(1 <= j <= NUM_TIMESTEPS for j in draws)
    assert all(isinstance(j, int) for j in draws)


def test_sample_timestep_logit_normal_distribution():
    rng = random.Random(2)
    sigmas = [sample_timestep(rng, Schedule(uniform_mix=0.0)) / NUM_TIMESTEPS for _ in range(20000)]
    mean = sum(sigmas) / len(sigmas)
    # s = 1 - sigmoid(z) with z ~ N(-1.09, 0.95) concentrates sigma near ~0.75 (then rescaled).
    assert 0.70 < mean < 0.85
    assert max(sigmas) <= 1.0
    assert min(sigmas) >= 0.0


def test_sample_timestep_uniform_mix_and_grid():
    rng = random.Random(3)
    draws = [sample_timestep(rng, Schedule(uniform_mix=1.0)) for _ in range(500)]
    assert all(1 <= j <= NUM_TIMESTEPS for j in draws)
    uniform = Schedule(timestep_sampling="uniform_grid")
    assert all(1 <= sample_timestep(rng, uniform) <= NUM_TIMESTEPS for _ in range(100))
    with pytest.raises(ValueError):
        sample_timestep(rng, Schedule(timestep_sampling="bogus"))


def test_sample_timestep_rejection_guard():
    with pytest.raises(RuntimeError, match="converge"):
        sample_timestep(random.Random(4), Schedule(uniform_mix=0.0, logit_normal_mean=-12.0,
                                                   logit_normal_std=0.01, sigma_cutoff=0.5))


def test_only_lora_params_trainable_after_injection():
    model = tiny_transformer().float()
    model.requires_grad_(False)
    adapters = inject_lora(model, rank=4, alpha=4)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 2 * len(adapters)
    assert all(name.endswith((".lora_A", ".lora_B")) for name in trainable)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_trainer_smoke_end_to_end(tmp_path, tiny_config):
    result = trainer.train(load_config(tiny_config))
    run_dir = Path(result["output_dir"])
    assert result["steps"] == 2

    adapter = run_dir / "adapter_model.safetensors"
    assert adapter.is_file()
    assert (run_dir / "adapter_config.json").is_file()
    assert (run_dir / "adapter_ema.safetensors").is_file()  # ema_decay=0.9

    from safetensors.torch import load_file

    raw = load_file(str(adapter))
    comfy = run_dir / "comfyui_lora.safetensors"
    assert comfy.is_file()
    converted = load_file(str(comfy))
    assert raw and all(key.startswith("layers.") for key in raw)
    assert any(key.endswith(".lora_A") for key in raw)
    assert converted and all(key.startswith("diffusion_model.") for key in converted)
    assert any(key.endswith(".lora_A.weight") for key in converted)
    assert any(key.endswith(".lora_B.weight") for key in converted)
    assert any(key.endswith(".alpha") for key in converted)
    assert not any(key.startswith("layers.") for key in converted)
    assert len(converted) == len(raw) // 2 * 3  # A + B + alpha per pair

    checkpoint = run_dir / "checkpoints" / "step_000002"
    for name in ("adapter_model.safetensors", "optimizer.pt", "lr_scheduler.pt", "rng.pt",
                 "trainer_state.json"):
        assert (checkpoint / name).is_file(), name
    assert not (checkpoint / "comfyui_lora.safetensors").exists()  # opt-in per checkpoint
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["step"] == 2 and "config" in state
    assert (checkpoint / "adapter_ema.safetensors").is_file()

    rows = _read_metrics(run_dir / "metrics.csv")
    assert len(rows) == 2
    assert all(math.isfinite(float(row["loss"])) for row in rows)
    assert {row["step"] for row in rows} == {"1", "2"}

    # Baseline (step 0) and step-2 validation latents: tiny model has no VAE -> .latent.pt
    baseline = run_dir / "samples" / "step_000000" / "item_00000.latent.pt"
    sampled = run_dir / "samples" / "step_000002" / "item_00000.latent.pt"
    assert baseline.is_file() and sampled.is_file()
    assert torch.load(sampled, weights_only=False).shape[0] == 8


def test_trainer_comfy_export_disabled(tmp_path, tiny_config):
    config = load_config(tiny_config)
    config.train.max_steps = 1
    config.logging.comfy_export = False
    result = trainer.train(config)
    run_dir = Path(result["output_dir"])
    assert (run_dir / "adapter_model.safetensors").is_file()
    assert not (run_dir / "comfyui_lora.safetensors").exists()
    assert not (run_dir / "checkpoints" / "step_000001" / "comfyui_lora.safetensors").exists()


def test_comfy_export_failure_is_non_fatal(tmp_path, capsys):
    assert trainer.export_comfy_adapter(tmp_path / "missing") is None
    assert "ComfyUI LoRA export failed" in capsys.readouterr().err


def test_trainer_resume_exact(tmp_path, tiny_config):
    config = load_config(tiny_config)
    trainer.train(config)
    resume_config = load_config(tiny_config)
    resume_config.train.max_steps = 4
    result = trainer.train(resume_config, resume_path=str(Path(config.train.output_dir)))
    assert result["steps"] == 4
    run_dir = Path(result["output_dir"])
    final_state = json.loads((run_dir / "checkpoints" / "step_000004" / "trainer_state.json").read_text())
    assert final_state["step"] == 4
    rows = _read_metrics(run_dir / "metrics.csv")
    assert [row["step"] for row in rows] == ["1", "2", "3", "4"]
    assert all(math.isfinite(float(row["loss"])) for row in rows)
    from safetensors.torch import load_file

    saved = load_file(str(run_dir / "checkpoints" / "step_000002" / "adapter_model.safetensors"))
    final_adapter = load_file(str(run_dir / "adapter_model.safetensors"))
    assert set(saved) == set(final_adapter)


def test_base_config_dict_matches_dataclasses(tmp_path):
    """The test fixture config must satisfy the strict key validation."""
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base_config_dict()), encoding="utf-8")
    config = load_config(path)
    assert config.train.seed == 42
