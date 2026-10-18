"""Benchmark CLI: real-model synthetic training steps (throughput + VRAM) on the user's GPU.

Synthetic inputs exercise the full training step (LoRA forward, fp32 flow-matching
loss, backward, AdamW) against the real checkpoint; no dataset or text encoder is
required. Run on the RTX 5090 to size batch/resolution/checkpointing choices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from moss_lora.lora import inject_lora
from moss_lora.model import MOSSImageTransformer
from moss_lora.utils import (
    peak_vram_gb,
    reset_peak_vram,
    resolve_pretrained_target,
    sdpa_context,
    unwrap_model,
)

NUM_TIMESTEPS = 1000


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(prog="moss-lora-bench",
                                     description="Synthetic LoRA training-step benchmark for MOSS-Image.")
    parser.add_argument("--model", required=True, help="MOSS-Image model dir (transformer/) or Hub id")
    parser.add_argument("--height", type=int, default=1024, help="Training height (multiple of 16)")
    parser.add_argument("--width", type=int, default=1024, help="Training width (multiple of 16)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20, help="Measured steps after warmup")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--text-length", type=int, default=512, help="Synthetic prompt token count")
    parser.add_argument("--checkpointing", dest="checkpointing", action="store_true", default=True)
    parser.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--sdpa-backend", default="auto",
                        choices=["auto", "flash", "efficient", "math", "cudnn"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the result dict as JSON only")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; run the benchmark on the training GPU.")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]

    load_path, subfolder = resolve_pretrained_target(args.model, "transformer")
    kwargs: dict = {"torch_dtype": dtype, "local_files_only": args.local_files_only}
    if subfolder:
        kwargs["subfolder"] = subfolder
    transformer = MOSSImageTransformer.from_pretrained(load_path, **kwargs).to(device)
    transformer.requires_grad_(False)
    adapters = inject_lora(transformer, rank=args.rank, alpha=args.alpha)
    if args.checkpointing:
        transformer.enable_gradient_checkpointing()
    trainable = [param for name, param in unwrap_model(transformer).named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01,
                                  fused=device.type == "cuda")
    model = torch.compile(transformer, dynamic=True, mode=args.compile_mode) if args.compile else transformer

    latent_h, latent_w = args.height // 16, args.width // 16
    x0 = torch.randn(args.batch_size, transformer.config.latent_channels, latent_h, latent_w,
                     device=device, dtype=torch.float32)
    context = torch.randn(args.batch_size, args.text_length, transformer.config.text_dim,
                          device=device, dtype=dtype)
    mask = torch.ones(args.batch_size, args.text_length, dtype=torch.bool, device=device)
    timesteps = torch.randint(1, NUM_TIMESTEPS + 1, (args.batch_size,), device=device, dtype=torch.float32)

    transformer.train()
    reset_peak_vram()

    def step() -> None:
        noise = torch.randn_like(x0)
        noisy = (1.0 - timesteps / NUM_TIMESTEPS) * x0 + (timesteps / NUM_TIMESTEPS) * noise
        with sdpa_context(args.sdpa_backend):
            prediction = model(noisy, timesteps, context, mask, return_dict=False)[0]
        loss = torch.mean((prediction.float() - (noise - x0)) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(max(0, args.warmup)):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(1, args.steps)):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    result = {
        "model": str(args.model),
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "alpha": args.alpha,
        "checkpointing": args.checkpointing,
        "compile": args.compile,
        "sdpa_backend": args.sdpa_backend,
        "dtype": args.dtype,
        "trainable_params": sum(param.numel() for param in trainable),
        "lora_targets": len(adapters),
        "steps": max(1, args.steps),
        "sec_per_step": seconds / max(1, args.steps),
        "samples_per_sec": args.batch_size * max(1, args.steps) / seconds,
        "peak_allocated_gb": peak_vram_gb() if device.type == "cuda" else None,
        "peak_reserved_gb": (torch.cuda.max_memory_reserved() / 2**30) if device.type == "cuda" else None,
        "device": str(device),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"config: {args.height}x{args.width} bs={args.batch_size} rank={args.rank} "
              f"ckpt={args.checkpointing} compile={args.compile} backend={args.sdpa_backend}")
        print(f"s/step: {result['sec_per_step']:.3f}   samples/s: {result['samples_per_sec']:.2f}")
        print(f"peak allocated: {result['peak_allocated_gb']} GiB   "
              f"peak reserved: {result['peak_reserved_gb']} GiB")
        print(f"trainable params: {result['trainable_params'] / 1e6:.1f}M")


if __name__ == "__main__":
    sys.exit(main())
