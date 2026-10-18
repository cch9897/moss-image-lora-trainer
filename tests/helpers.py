"""Shared tiny-model fixtures for CPU tests and scripts/make_tiny_fixtures.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from moss_lora.model import MOSSImageTransformer

TINY_KWARGS: dict[str, Any] = dict(
    dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=32, freq_dim=16, latent_channels=8,
)
REVISION = "tinyrev"


def tiny_transformer(**overrides: Any) -> MOSSImageTransformer:
    torch.manual_seed(1234)
    return MOSSImageTransformer(**{**TINY_KWARGS, **overrides})


def make_tiny_model_dir(root: Path) -> Path:
    """Full-checkout-shaped directory: transformer/, tokenizer/ (no vae, no text encoder weights)."""
    root.mkdir(parents=True, exist_ok=True)
    tiny_transformer().save_pretrained(root / "transformer")
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (root / "text_encoder").mkdir(exist_ok=True)
    (root / "text_encoder" / "config.json").write_text("{}\n", encoding="utf-8")
    return root


class FakeTextEncoder:
    """Deterministic stand-in for the frozen Qwen3.5 encoder: [L, text_dim] bf16 per prompt."""

    def __init__(self, text_dim: int = 32, min_len: int = 3, max_len: int = 9):
        self.text_dim = text_dim
        self.min_len = min_len
        self.max_len = max_len
        self.calls: list[tuple[str, int | None]] = []
        self.unloaded = False

    def encode(self, prompt: str, max_length: int | None = None) -> torch.Tensor:
        self.calls.append((prompt, max_length))
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        length = self.min_len + digest[0] % (self.max_len - self.min_len + 1)
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:4], "big"))
        return torch.randn(length, self.text_dim, generator=generator).to(torch.bfloat16)

    def unload(self) -> None:
        self.unloaded = True


def fake_latent_encoder_factory(latent_channels: int = 8):
    """factory(device) -> deterministic area-downsample 'VAE' encoder for tests."""
    import torch.nn.functional as F

    def factory(device: str):
        def encode(image: torch.Tensor) -> torch.Tensor:
            height, width = image.shape[-2] // 16, image.shape[-1] // 16
            base = F.interpolate(image.float(), size=(height, width), mode="area")
            return base.repeat(1, (latent_channels + 2) // 3, 1, 1)[:, :latent_channels]

        return encode

    return factory


def gradient_image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    base = np.clip(x[None, :] + y + rng.normal(0, 0.1, (height, width)).astype(np.float32), -1, 1)
    array = np.stack([base, -base, rng.uniform(-1, 1, (height, width)).astype(np.float32)], -1)
    return Image.fromarray(((array + 1.0) * 127.5).astype(np.uint8), mode="RGB")


CAPTIONS = [
    "a red gradient square",
    "a blue gradient rectangle",
    "a green diagonal gradient",
    "a noisy color field",
]


def make_tiny_cache(cache_dir: Path, *, text_dim: int = 32, latent_channels: int = 8,
                    n_items: int = 4, target_area: int = 256 * 256) -> Path:
    """Synthetic dataset + cache: folder sidecars, JSONL dict-caption variants and a layout row."""
    from moss_lora.cache import build_cache

    root = cache_dir.parent / "tiny_dataset"
    root.mkdir(parents=True, exist_ok=True)
    sizes = [(256, 256), (256, 256), (320, 192), (192, 320)]
    jsonl_rows = []
    for index in range(n_items):
        width, height = sizes[index % len(sizes)]
        image_path = root / f"img_{index:02d}.png"
        gradient_image(width, height, seed=index).save(image_path)
        if index % 4 == 0:
            jsonl_rows.append({"image": f"img_{index:02d}.png", "text": CAPTIONS[index % len(CAPTIONS)]})
        elif index % 4 == 1:
            jsonl_rows.append({"image": f"img_{index:02d}.png", "caption": {
                "long": f"a long detailed caption for image {index}",
                "medium": f"a medium caption {index}",
                "short": f"short {index}",
            }})
        elif index % 4 == 2:
            # No caption in the row: collect_items falls back to the same-stem .json sidecar.
            sidecar = {"caption": {"long": f"sidecar long {index}", "short": f"sidecar short {index}"}}
            image_path.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False),
                                                       encoding="utf-8")
            jsonl_rows.append({"image": f"img_{index:02d}.png"})
        else:
            jsonl_rows.append({"file_name": f"img_{index:02d}.png", "layout": {
                "summary": {"caption": f"layout summary {index}"},
                "items": [{"type": "text", "text": "hello", "bbox": [0, 0, 10, 10]}],
            }})
    manifest = root / "meta.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in jsonl_rows),
                        encoding="utf-8")

    text_encoder = FakeTextEncoder(text_dim=text_dim)
    latent_factory = fake_latent_encoder_factory(latent_channels)
    build_cache(
        manifest, cache_dir,
        text_encoder_factory=lambda: text_encoder,
        latent_encoder_factory=latent_factory,
        target_area=target_area,
        max_text_length=1024,
        revision=REVISION,
        negative_prompt="blurry, low quality",
        device="cpu",
        progress=False,
    )
    return cache_dir


class FakeVae:
    """Minimal stand-in with the BatchNorm stats the pipeline decode path touches."""

    def __init__(self, latent_channels: int = 8, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        self.bn = torch.nn.BatchNorm2d(latent_channels)
        with torch.no_grad():
            self.bn.running_mean.copy_(torch.randn(latent_channels, generator=generator) * 0.5)
            self.bn.running_var.copy_(torch.rand(latent_channels, generator=generator) * 0.5 + 0.75)
            self.bn.weight.fill_(1.0)
            self.bn.bias.fill_(0.0)
        self.config = type("Config", (), {"batch_norm_eps": 1e-5})()
        self.dtype = torch.float32


    def encode(self, image: torch.Tensor):
        raw = torch.nn.functional.interpolate(image.float().mean(1, keepdim=True),
                                              scale_factor=1 / 8, mode="area")
        raw = raw.repeat(1, self.bn.num_features // 4, 1, 1)
        dist = type("Dist", (), {"mode": lambda _self: raw})()
        return type("Encoded", (), {"latent_dist": dist})()

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        image = torch.tanh(z.mean(1, keepdim=True) * 2.0).repeat(1, 3, 1, 1)
        image = torch.nn.functional.interpolate(image, scale_factor=8, mode="nearest")
        return (image,) if not return_dict else {"sample": image}


def write_config(path: Path, **overrides: Any) -> Path:
    import yaml

    from tests.helpers import base_config_dict

    payload = base_config_dict()
    for section, values in overrides.items():
        payload[section].update(values)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def base_config_dict() -> dict[str, Any]:
    return {
        "model": {"path": "", "revision": None, "dtype": "fp32", "local_files_only": True},
        "data": {"cache_dir": "", "num_workers": 0, "batch_size": 2, "drop_last": False,
                 "pin_memory": False},
        "train": {
            "device": "cpu", "output_dir": "", "max_steps": 2, "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.95], "warmup_steps": 1,
            "lr_scheduler": "cosine", "max_grad_norm": 1.0, "seed": 42,
            "gradient_checkpointing": True, "compile": False, "compile_mode": "default",
            "sdpa_backend": "auto", "ema_decay": 0.9, "timestep_sampling": "logit_normal",
            "logit_normal_mean": -1.09, "logit_normal_std": 0.95, "sigma_cutoff": 0.95,
            "uniform_mix": 0.01,
        },
        "lora": {"rank": 4, "alpha": 4, "dropout": 0.0,
                 "targets": ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"],
                 "text_projection": False},
        "logging": {"log_every": 1, "save_every": 2, "sample_every": 2, "sample_steps": 2,
                    "sample_cfg": 3.0, "sample_skip_layer": -1, "sample_negative_prompt": "",
                    "sample_val_ids": ["item_00000"]},
    }
