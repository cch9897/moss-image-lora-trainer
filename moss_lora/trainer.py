"""LoRA trainer: bf16 base DiT + fp32 LoRA, flow-matching loss on the report's noise schedule.

Sigma sampling follows the MOSS-Image report's final pretraining schedule
(truncated logit-normal mu=-1.09, std=0.95, cutoff=0.95, 1% uniform mixing on the
discrete 1000-step grid); x_t = (1 - s) * x0 + s * noise with target velocity
noise - x0 and fp32 MSE. The text encoder and VAE never participate in the
training loop; validation samples through the vendored MOSSImagePipeline with
cached embeddings and a lazily loaded VAE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

from moss_lora import __version__
from moss_lora.config import TrainerConfig, apply_cli_overrides, load_config
from moss_lora.dataset import BucketBatchSampler, CachedDataset, build_dataloader
from moss_lora.lora import (
    ADAPTER_EMA_NAME,
    inject_lora,
    load_adapter,
    save_adapter,
)
from moss_lora.model import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from moss_lora.utils import (
    MetricsLogger,
    atomic_save_safetensors,
    atomic_torch_save,
    atomic_write_json,
    decode_image,
    load_rng_state,
    peak_vram_gb,
    reset_peak_vram,
    resolve_pretrained_target,
    save_rng_state,
    sdpa_context,
    seed_everything,
    tensor_to_pil_image,
    unwrap_model,
)

CHECKPOINTS_DIR = "checkpoints"
SAMPLES_DIR = "samples"
METRICS_NAME = "metrics.csv"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
LR_SCHEDULER_NAME = "lr_scheduler.pt"
RNG_NAME = "rng.pt"
NEGATIVE_NAME = "negative.safetensors"
NUM_TIMESTEPS = 1000


# --------------------------------------------------------------------------- noise schedule


def sample_timestep(rng: random.Random, train_cfg) -> int:
    """Discrete timestep j in [1, 1000] from the report's Stage-3 schedule.

    logit_normal: with probability uniform_mix draw a uniform-grid j, otherwise repeat
    z ~ N(logit_normal_mean, logit_normal_std), s = 1 - sigmoid(z) until s <= sigma_cutoff,
    then j = clamp(ceil(s * 1000 / sigma_cutoff), 1, 1000). sigma = j / 1000 and the model
    receives timestep == j == sigma * 1000.
    uniform_grid: j ~ randint(1, 1000) (ablation baseline).
    """
    if train_cfg.timestep_sampling == "uniform_grid":
        return rng.randint(1, NUM_TIMESTEPS)
    if train_cfg.timestep_sampling != "logit_normal":
        raise ValueError(f"Unknown timestep_sampling {train_cfg.timestep_sampling!r}.")
    if rng.random() < train_cfg.uniform_mix:
        return rng.randint(1, NUM_TIMESTEPS)
    for _ in range(10_000):  # rejection guard; unreachable for sane parameters
        z = rng.gauss(train_cfg.logit_normal_mean, train_cfg.logit_normal_std)
        sigma = 1.0 - 1.0 / (1.0 + math.exp(-z))
        if sigma <= train_cfg.sigma_cutoff:
            return min(max(math.ceil(sigma * NUM_TIMESTEPS / train_cfg.sigma_cutoff), 1), NUM_TIMESTEPS)
    raise RuntimeError(
        "Timestep rejection sampling did not converge; check logit_normal_mean/std and sigma_cutoff.")


# --------------------------------------------------------------------------- setup helpers




def load_transformer(config: TrainerConfig, device: torch.device) -> MOSSImageTransformer:
    load_path, subfolder = resolve_pretrained_target(config.model.path, "transformer")
    kwargs: dict[str, Any] = {"torch_dtype": config.model.torch_dtype(),
                              "local_files_only": config.model.local_files_only}
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    transformer = MOSSImageTransformer.from_pretrained(load_path, **kwargs)
    return transformer.to(device)


def load_validation_vae(config: TrainerConfig, device: torch.device):
    """fp32 VAE for validation decoding; returns None when the model ships no VAE."""
    from diffusers import AutoencoderKLFlux2

    load_path, subfolder = resolve_pretrained_target(config.model.path, "vae")
    # Local checkouts must ship vae/config.json; a Hub id is resolved by from_pretrained.
    if Path(config.model.path).is_dir() and not (Path(load_path) / "config.json").is_file():
        return None
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32,
                              "local_files_only": config.model.local_files_only}
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    return AutoencoderKLFlux2.from_pretrained(load_path, **kwargs).eval().requires_grad_(False).to(device)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, train_cfg):
    warmup = max(0, train_cfg.warmup_steps)
    total = max(1, train_cfg.max_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if train_cfg.lr_scheduler == "constant":
            return 1.0
        if train_cfg.lr_scheduler != "cosine":
            raise ValueError(f"Unknown lr_scheduler {train_cfg.lr_scheduler!r}; use cosine or constant.")
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ParamEMA:
    """fp32 exponential moving average of the trainable (LoRA) parameters."""

    def __init__(self, named_params: list[tuple[str, torch.nn.Parameter]], decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}.")
        self.decay = decay
        self.shadow = {name: param.detach().float().clone() for name, param in named_params}

    @torch.no_grad()
    def update(self, named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        for name, param in named_params:
            self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
        backup = {}
        for name, param in named_params:
            backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].to(param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, backup: dict[str, torch.Tensor],
                named_params: list[tuple[str, torch.nn.Parameter]]) -> None:
        for name, param in named_params:
            param.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing = sorted(set(self.shadow) - set(state))
        unexpected = sorted(set(state) - set(self.shadow))
        if missing or unexpected:
            raise ValueError(f"EMA state mismatch (missing={missing}, unexpected={unexpected}).")
        for name, tensor in state.items():
            self.shadow[name].copy_(tensor.float())


# --------------------------------------------------------------------------- checkpointing


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / CHECKPOINTS_DIR / f"step_{step:06d}"


def save_checkpoint(config: TrainerConfig, model: MOSSImageTransformer, optimizer, lr_scheduler,
                    ema: ParamEMA | None, step: int, epoch: int, batch_index: int) -> Path:
    directory = checkpoint_dir(config.train.output_dir, step)
    save_adapter(unwrap_model(model), directory,
                 base_model=str(config.model.path), revision=config.model.revision)
    if ema is not None:
        atomic_save_safetensors(ema.state_dict(), directory / ADAPTER_EMA_NAME,
                                metadata={"format": "pt", "moss_lora_version": __version__,
                                          "ema_decay": str(ema.decay)})
    atomic_torch_save(optimizer.state_dict(), directory / OPTIMIZER_NAME)
    atomic_torch_save(lr_scheduler.state_dict(), directory / LR_SCHEDULER_NAME)
    save_rng_state(directory / RNG_NAME)
    atomic_write_json(directory / TRAINER_STATE_NAME, {
        "step": step, "epoch": epoch, "batch_index": batch_index,
        "moss_lora_version": __version__, "config": config.to_dict(),
    })
    return directory


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if root.name.startswith("step_") and (root / TRAINER_STATE_NAME).is_file():
        return root
    candidates = sorted((root / CHECKPOINTS_DIR).glob("step_*")) if (root / CHECKPOINTS_DIR).is_dir() else []
    for candidate in reversed(candidates):
        if (candidate / TRAINER_STATE_NAME).is_file():
            return candidate
    return None


def resume_checkpoint(config: TrainerConfig, model: MOSSImageTransformer, optimizer, lr_scheduler,
                      ema: ParamEMA | None, resume_path: str) -> tuple[int, int, int]:
    path = Path(resume_path)
    if not (path / TRAINER_STATE_NAME).is_file():
        latest = find_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint with {TRAINER_STATE_NAME} under {path}")
        path = latest
    payload = json.loads((path / TRAINER_STATE_NAME).read_text(encoding="utf-8"))
    load_adapter(unwrap_model(model), path)
    optimizer.load_state_dict(torch.load(path / OPTIMIZER_NAME, map_location="cpu", weights_only=False))
    lr_scheduler.load_state_dict(torch.load(path / LR_SCHEDULER_NAME, map_location="cpu", weights_only=False))
    load_rng_state(path / RNG_NAME)
    if ema is not None:
        ema_path = path / ADAPTER_EMA_NAME
        if not ema_path.is_file():
            raise FileNotFoundError(f"Checkpoint has no {ADAPTER_EMA_NAME} but EMA is enabled")
        from safetensors.torch import load_file

        ema.load_state_dict(load_file(str(ema_path)))
    return int(payload["step"]), int(payload["epoch"]), int(payload["batch_index"])


# --------------------------------------------------------------------------- validation


@torch.no_grad()
def run_validation(config: TrainerConfig, model: MOSSImageTransformer, dataset: CachedDataset,
                   vae, step: int, device: torch.device) -> None:
    """Sample every validation id at its own bucket size; PNG with a VAE, latent .pt without."""
    log_cfg = config.logging
    val_ids = log_cfg.sample_val_ids
    if not val_ids:
        return
    negative_path = Path(config.data.cache_dir) / NEGATIVE_NAME
    if not negative_path.is_file():
        raise FileNotFoundError(
            f"{negative_path} not found; run moss-lora-cache with --negative-prompt to enable validation.")
    with safe_open(str(negative_path), framework="pt", device="cpu") as handle:
        negative = handle.get_tensor("embedding")
    transformer = unwrap_model(model)
    transformer.eval()
    pipe = MOSSImagePipeline(transformer=transformer, scheduler=MOSSImageScheduler(),
                             text_encoder=None, tokenizer=None, vae=vae)
    pipe.set_progress_bar_config(disable=True)
    skip_layers = [] if log_cfg.sample_skip_layer < 0 else [log_cfg.sample_skip_layer]
    sample_dir = Path(config.train.output_dir) / SAMPLES_DIR / f"step_{step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, item_id in enumerate(val_ids):
        bucket = next((item for item in dataset.items if item["id"] == item_id), None)
        if bucket is None:
            raise KeyError(f"Validation id {item_id!r} is not in the cache index.")
        embedding = dataset.text_embedding(item_id, variant=0)[None].to(device)
        latents = pipe(
            prompt_embeds=embedding,
            negative_prompt_embeds=negative[None].to(device),
            height=bucket["bucket_h"], width=bucket["bucket_w"],
            num_inference_steps=log_cfg.sample_steps,
            guidance_scale=log_cfg.sample_cfg,
            uncond_skip_layers=skip_layers,
            generator=torch.Generator("cpu").manual_seed(config.train.seed + step + index),
            output_type="latent",
        ).images
        if vae is not None:
            image = decode_image(latents.float(), vae)[0]
            tensor_to_pil_image(image).save(sample_dir / f"{item_id}.png")
        else:
            torch.save(latents[0].cpu(), sample_dir / f"{item_id}.latent.pt")
    del pipe
    transformer.train()


# --------------------------------------------------------------------------- training


def train(config: TrainerConfig, resume_path: str | None = None) -> dict[str, Any]:
    device = torch.device(config.train.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is cuda but CUDA is not available.")
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.train.seed)
    reset_peak_vram()

    transformer = load_transformer(config, device)
    # Freeze the whole base model BEFORE injection; fresh LoRA parameters stay trainable.
    transformer.requires_grad_(False)
    adapters = inject_lora(
        transformer,
        rank=config.lora.rank,
        alpha=config.lora.alpha if config.lora.alpha is not None else config.lora.rank,
        dropout=config.lora.dropout,
        targets=config.lora.targets,
        text_projection=config.lora.text_projection,
        final_linear=config.lora.final_linear,
    )
    trainable_named = [(name, param) for name, param in unwrap_model(transformer).named_parameters()
                       if param.requires_grad]
    unexpected = [name for name, _ in trainable_named if not name.endswith((".lora_A", ".lora_B"))]
    if unexpected or len(trainable_named) != 2 * len(adapters):
        raise RuntimeError(
            f"Trainable parameter check failed: {len(trainable_named)} params, unexpected={unexpected[:5]}")
    if config.train.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    trainable = [param for _, param in trainable_named]

    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.learning_rate, betas=tuple(config.train.betas),
        eps=config.train.adamw_eps, weight_decay=config.train.weight_decay,
        fused=device.type == "cuda")
    lr_scheduler = build_lr_scheduler(optimizer, config.train)
    ema = ParamEMA(trainable_named, config.train.ema_decay) if config.train.ema_decay > 0 else None
    model = (torch.compile(transformer, dynamic=True, mode=config.train.compile_mode)
             if config.train.compile else transformer)

    dataset = CachedDataset(config.data.cache_dir)
    buckets = [(item["bucket_h"], item["bucket_w"]) for item in dataset.items]
    sampler = BucketBatchSampler(buckets, config.data.batch_size, drop_last=config.data.drop_last,
                                 seed=config.train.seed)

    start_step, start_epoch, start_batch = 0, 0, 0
    if resume_path:
        start_step, start_epoch, start_batch = resume_checkpoint(
            config, transformer, optimizer, lr_scheduler, ema, resume_path)

    vae = None
    if config.logging.sample_every > 0 and config.logging.sample_val_ids:
        vae = load_validation_vae(config, device)
    metrics = MetricsLogger(output_dir / METRICS_NAME)
    accum = max(1, config.train.gradient_accumulation_steps)
    micro_in_epoch = start_batch
    completed_step = start_step
    epoch = start_epoch
    loss_accum = 0.0
    t_last = time.perf_counter()

    if start_step == 0 and config.logging.sample_every > 0 and config.logging.sample_val_ids:
        run_validation(config, transformer, dataset, vae, 0, device)

    transformer.train()
    while completed_step < config.train.max_steps:
        sampler.set_epoch(epoch)
        batches = sampler.batches_for_epoch(epoch)
        if micro_in_epoch:
            batches = batches[micro_in_epoch:]
        if not batches:
            raise RuntimeError(
                f"No batches in epoch {epoch}; dataset too small for batch_size="
                f"{config.data.batch_size} with drop_last={config.data.drop_last}.")
        loader = build_dataloader(dataset, batches, num_workers=config.data.num_workers,
                                  pin_memory=config.data.pin_memory)
        for batch in loader:
            x0 = batch["latents"].to(device, non_blocking=True)
            context = batch["text"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            noise = torch.randn_like(x0)
            timesteps = [sample_timestep(random, config.train) for _ in range(x0.shape[0])]
            sigma = torch.tensor([t / NUM_TIMESTEPS for t in timesteps], device=device,
                                 dtype=torch.float32).view(-1, 1, 1, 1)
            noisy = (1.0 - sigma) * x0 + sigma * noise
            with sdpa_context(config.train.sdpa_backend):
                prediction = model(noisy, torch.tensor(timesteps, device=device, dtype=torch.float32),
                                   context, mask, return_dict=False)[0]
            loss = torch.mean((prediction.float() - (noise - x0)) ** 2) / accum
            loss.backward()
            loss_accum += float(loss.detach())
            micro_in_epoch += 1
            if micro_in_epoch % accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.train.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(trainable_named)
            completed_step += 1
            loss_value = loss_accum * accum
            loss_accum = 0.0
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Loss became {loss_value} at step {completed_step}; try a lower learning rate.")
            if (completed_step % max(1, config.logging.log_every) == 0
                    or completed_step == config.train.max_steps):
                now = time.perf_counter()
                metrics.log({
                    "step": completed_step,
                    "loss": f"{loss_value:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.3e}",
                    "sigma": f"{float(sigma.mean()):.4f}",
                    "grad_norm": f"{float(grad_norm):.4f}",
                    "sec_per_step": f"{(now - t_last) / max(1, config.logging.log_every):.3f}",
                    "peak_vram_gb": f"{peak_vram_gb():.2f}",
                })
                t_last = now
            if (config.logging.sample_every > 0 and config.logging.sample_val_ids
                    and completed_step % config.logging.sample_every == 0):
                if vae is None:
                    print(f"[step {completed_step}] no VAE found; saving latent samples instead of PNG")
                ema_backup = ema.apply(trainable_named) if ema is not None else None
                run_validation(config, transformer, dataset, vae, completed_step, device)
                if ema_backup is not None:
                    ema.restore(ema_backup, trainable_named)
            if completed_step % max(1, config.logging.save_every) == 0:
                save_checkpoint(config, transformer, optimizer, lr_scheduler, ema,
                                completed_step, epoch, micro_in_epoch)
            if completed_step >= config.train.max_steps:
                break
        if micro_in_epoch % accum:  # drop a partial accumulation at the epoch boundary
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
        epoch += 1
        micro_in_epoch = 0

    final_dir = save_checkpoint(config, transformer, optimizer, lr_scheduler, ema,
                                completed_step, epoch, micro_in_epoch)
    save_adapter(unwrap_model(transformer), output_dir,
                 base_model=str(config.model.path), revision=config.model.revision)
    if ema is not None:
        atomic_save_safetensors(ema.state_dict(), output_dir / ADAPTER_EMA_NAME,
                                metadata={"format": "pt", "moss_lora_version": __version__,
                                          "ema_decay": str(ema.decay)})
    return {"steps": completed_step, "checkpoint": str(final_dir), "output_dir": str(output_dir)}


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(prog="moss-lora-train",
                                     description="LoRA fine-tuning for MOSS-Image-Base.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--resume", default=None, help="Run dir or step_XXXXXX checkpoint dir")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-ids", default=None, help="Comma-separated cache item ids for validation")
    args = parser.parse_args(argv)

    config = apply_cli_overrides(
        load_config(args.config), max_steps=args.max_steps, output_dir=args.output_dir,
        device=args.device, batch_size=args.batch_size,
        val_ids=[value for value in (args.val_ids or "").split(",") if value])
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    sys.exit(main())
