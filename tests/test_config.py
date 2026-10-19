"""Config parsing, CLI overrides and shipped config validation."""

from __future__ import annotations

import pytest
import yaml

from moss_lora.config import (
    DataConfig,
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    TrainConfig,
    apply_cli_overrides,
    load_config,
    parse_dtype,
)


def test_defaults():
    config = TrainConfig()
    assert config.betas == (0.9, 0.95)
    assert config.learning_rate == 1e-4
    assert config.timestep_sampling == "logit_normal"
    assert config.logit_normal_mean == -1.09
    assert config.logit_normal_std == 0.95
    assert config.sigma_cutoff == 0.95
    assert config.uniform_mix == 0.01
    assert config.lr_scheduler == "cosine"
    assert config.adamw_eps == 1e-8
    assert LoraConfig().targets == ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
    assert LoggingConfig().sample_skip_layer == 22
    assert LoggingConfig().comfy_export is True
    assert LoggingConfig().comfy_export_checkpoints is False
    assert DataConfig().drop_last is True
    assert ModelConfig().dtype == "bf16"


def test_load_full_yaml(tmp_path):
    payload = {
        "model": {"path": "models/MOSS-Image-Base", "revision": "abc", "dtype": "bf16",
                  "local_files_only": True},
        "data": {"cache_dir": "cache", "num_workers": 2, "batch_size": 3, "drop_last": True,
                 "pin_memory": True},
        "train": {"device": "cuda", "output_dir": "runs/x", "max_steps": 10, "betas": [0.9, 0.95],
                  "timestep_sampling": "uniform_grid"},
        "lora": {"rank": 8, "alpha": None, "targets": ["q", "v"], "text_projection": True},
        "logging": {"sample_val_ids": ["item_00000"], "sample_skip_layer": -1,
                    "comfy_export": False, "comfy_export_checkpoints": True},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(path)
    assert config.model.path == "models/MOSS-Image-Base"
    assert config.data.batch_size == 3
    assert config.train.timestep_sampling == "uniform_grid"
    assert config.lora.alpha is None
    assert config.lora.text_projection is True
    assert config.logging.sample_skip_layer == -1
    assert config.logging.comfy_export is False
    assert config.logging.comfy_export_checkpoints is True


def test_unknown_keys_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {"unknown_key": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_key"):
        load_config(path)
    path.write_text(yaml.safe_dump({"nope": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="section"):
        load_config(path)


def test_cli_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"train": {"max_steps": 100, "output_dir": "runs/a"},
                                    "data": {"batch_size": 1},
                                    "logging": {"sample_val_ids": ["x"]}}), encoding="utf-8")
    config = apply_cli_overrides(load_config(path), max_steps=7, output_dir="runs/b", device="cpu",
                                 batch_size=5, val_ids=["a", "b"], comfy_export=False)
    assert config.train.max_steps == 7
    assert config.train.output_dir == "runs/b"
    assert config.train.device == "cpu"
    assert config.data.batch_size == 5
    assert config.logging.sample_val_ids == ["a", "b"]
    assert config.logging.comfy_export is False
    unchanged = apply_cli_overrides(load_config(path))
    assert unchanged.train.max_steps == 100
    assert unchanged.logging.sample_val_ids == ["x"]
    assert unchanged.logging.comfy_export is True


@pytest.mark.parametrize("name,expected", [
    ("bf16", __import__("torch").bfloat16),
    ("fp32", __import__("torch").float32),
    ("fp16", __import__("torch").float16),
])
def test_dtype_parse(name, expected):
    assert parse_dtype(name) == expected
    with pytest.raises(ValueError):
        parse_dtype("int4")


def test_shipped_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("train_5090_1m.yaml", "train_5090_512.yaml", "tiny_cpu.yaml"):
        config = load_config(root / "configs" / name)
        assert config.train.timestep_sampling == "logit_normal"
        assert config.train.betas == (0.9, 0.95)
        assert config.lora.rank > 0
        assert config.logging.comfy_export is True
        assert config.logging.comfy_export_checkpoints is False
    one_m = load_config(root / "configs" / "train_5090_1m.yaml")
    fast = load_config(root / "configs" / "train_5090_512.yaml")
    assert one_m.data.batch_size == 1 and one_m.train.max_steps == 2000
    assert one_m.train.gradient_checkpointing is True
    assert fast.data.batch_size == 4 and fast.train.max_steps == 1500
    assert fast.train.gradient_checkpointing is True
    assert one_m.model.revision == "1ceede80badddd6a7812069619edaa5633587192"
    tiny = load_config(root / "configs" / "tiny_cpu.yaml")
    assert tiny.train.device == "cpu" and tiny.model.dtype == "fp32"
