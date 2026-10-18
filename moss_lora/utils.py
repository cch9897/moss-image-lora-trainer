"""Shared helpers: seeding, hashing, atomic IO, latent patchify/decode math, metrics logging."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (including CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: str | os.PathLike, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def atomic_save_safetensors(tensors: dict[str, torch.Tensor], path: str | os.PathLike,
                            metadata: dict[str, str] | None = None) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    payload = {k: v.contiguous() for k, v in tensors.items()}
    save_file(payload, str(tmp), metadata=metadata or {"format": "pt"})
    os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module behind torch.compile (model._orig_mod) or the module itself."""
    return getattr(model, "_orig_mod", model)


def patchify_latent(raw: torch.Tensor) -> torch.Tensor:
    """[B, C, 2h, 2w] raw VAE latent -> [B, 4C, h, w] transformer latent.

    Exact inverse of the reshape inside MOSSImagePipeline.__call__ (and of
    unpatchify_latent). Transformer channel index = 4*c + 2*i + j, where (i, j)
    is the position inside the 2x2 patch.
    """
    b, c, h, w = raw.shape
    if h % 2 or w % 2:
        raise ValueError(f"Latent spatial dims must be even, got {h}x{w}.")
    return raw.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def resolve_pretrained_target(root: str | os.PathLike, component: str) -> tuple[str, str | None]:
    """(load_path, subfolder) for from_pretrained-style loading of a model component.

    - Hub id (not a local directory): (repo_id, component);
    - local full checkout: (root/component, None);
    - local directory that is the component itself (config.json at the root): (root, None).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return str(root_path), component
    if (root_path / "config.json").is_file():
        return str(root_path), None
    return str(root_path / component), None


def resolve_tokenizer_path(root: str | os.PathLike) -> str:
    """Tokenizer location of a model checkout: the root itself or the `tokenizer/` subfolder.

    Non-directories (Hub ids) pass through unchanged.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return str(root_path)
    primary = "tokenizer_config.json"
    fallbacks = ("tokenizer.json", "vocab.json", "vocab.txt", "spiece.model")
    for candidate in (root_path, root_path / "tokenizer"):
        if (candidate / primary).is_file():
            return str(candidate)
    for candidate in (root_path, root_path / "tokenizer"):
        if any((candidate / name).is_file() for name in fallbacks):
            return str(candidate)
    raise FileNotFoundError(
        f"No tokenizer files under {root_path} (looked in {root_path} and {root_path / 'tokenizer'}).")


def unpatchify_latent(latent: torch.Tensor) -> torch.Tensor:
    """[B, 4C, h, w] transformer latent -> [B, C, 2h, 2w] raw VAE latent.

    Verbatim mirror of the reshape used by MOSSImagePipeline.__call__ before
    vae.decode: z.reshape(b, c//4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).
    """
    b, c, h, w = latent.shape
    if c % 4:
        raise ValueError(f"Transformer latent channel count must be a multiple of 4, got {c}.")
    return latent.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def vae_bn_stats(vae) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, std) of the FLUX.2 VAE output BatchNorm, shaped [1, C, 1, 1], fp32 on the VAE device."""
    eps = vae.config.batch_norm_eps
    mean = vae.bn.running_mean.detach().float().view(1, -1, 1, 1)
    std = (vae.bn.running_var.detach().float() + eps).sqrt().view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_image_latent(image: torch.Tensor, vae) -> torch.Tensor:
    """Image tensor [-1, 1] [B, 3, H, W] -> normalized patchified latent [B, 4C, H/16, W/16].

    Uses the VAE posterior mode, patchify, and BatchNorm normalization so that the
    cached latent is exactly what MOSSImagePipeline's decode path expects.
    """
    mean, std = vae_bn_stats(vae)
    latent = vae.encode(image.to(vae.dtype)).latent_dist.mode()
    latent = patchify_latent(latent.float())
    return (latent - mean.to(latent)) / std.to(latent)


@torch.no_grad()
def decode_image(latent: torch.Tensor, vae) -> torch.Tensor:
    """Normalized transformer latent [B, 4C, h, w] -> image tensor [-1, 1] [B, 3, 2h*8, 2w*8].

    Mirrors the exact decode math of MOSSImagePipeline.__call__ (fp32 VAE).
    """
    mean, std = vae_bn_stats(vae)
    z = latent.to(vae.dtype) * std.to(latent) + mean.to(latent)
    z = unpatchify_latent(z)
    return vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)


def tensor_to_pil_image(image: torch.Tensor):
    """[-1, 1] CHW float tensor -> PIL RGB image."""
    from PIL import Image

    array = ((image.detach().float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).contiguous().cpu().numpy(), mode="RGB")


def sdpa_context(name: str | None):
    """Context manager restricting scaled_dot_product_attention backends, or a null context."""
    if not name or name == "auto":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "math": [SDPBackend.MATH],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
    }
    if name not in backends:
        raise ValueError(f"Unknown sdpa_backend {name!r}; choose from auto, {', '.join(backends)}.")
    return sdpa_kernel(backends[name])


def peak_vram_gb() -> float:
    """Peak CUDA memory allocated (GiB) since the last reset; 0.0 without CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**30


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def save_rng_state(path: str | os.PathLike) -> None:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    atomic_torch_save(state, path)


def load_rng_state(path: str | os.PathLike) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class MetricsLogger:
    """Appends step metrics to CSV and mirrors them to the console."""

    def __init__(self, path: str | os.PathLike, console: bool = True):
        self.path = Path(path)
        self.console = console
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._header_written
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_file:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
        if self.console:
            print(
                f"[step {row.get('step', '?')}] "
                + " ".join(f"{key}={row[key]}" for key in row if key != "step"),
                flush=True,
            )

    def iter_rows(self) -> Iterator[dict[str, str]]:
        if not self.path.exists():
            return
        with open(self.path, newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def image_tensor_from_pil(image) -> torch.Tensor:
    """PIL RGB image -> [-1, 1] float32 CHW tensor."""
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def buffer_bytes(buffer: io.BytesIO) -> bytes:
    return buffer.getvalue()
