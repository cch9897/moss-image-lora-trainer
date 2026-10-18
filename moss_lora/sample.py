"""Sampling CLI: encode prompts with the text encoder, offload it, then run the vendored pipeline.

The text encoder (~18GB) is loaded first, encodes all prompts and the negative
prompt on GPU, and is freed before the transformer (+VAE) is loaded, so the two
never coexist. The transformer is moved to the target device before the pipeline
call so the pipeline's execution device is correct; prompts are passed as
prompt_embeds (never re-encoded by the pipeline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from moss_lora.lora import ADAPTER_EMA_NAME, ADAPTER_WEIGHTS_NAME, load_lora_into_model
from moss_lora.model import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from moss_lora.text import TextEncoder, load_text_encoder
from moss_lora.utils import (
    resolve_pretrained_target,
    resolve_tokenizer_path,
)


def resolve_adapter_weights(adapter: str | Path, use_ema: bool) -> Path:
    path = Path(adapter)
    directory = path if path.is_dir() else path.parent
    if use_ema:
        ema = directory / ADAPTER_EMA_NAME
        if not ema.is_file():
            raise FileNotFoundError(f"--ema requested but {ema} does not exist.")
        return ema
    if path.is_file():
        return path
    weights = directory / ADAPTER_WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"No {ADAPTER_WEIGHTS_NAME} in {directory}")
    return weights


def load_sampling_transformer(model: str, device: str, dtype: torch.dtype, local_files_only: bool,
                              revision: str | None, adapter: str | None, use_ema: bool,
                              base_for_adapter: str | None = None) -> MOSSImageTransformer:
    load_path, subfolder = resolve_pretrained_target(model, "transformer")
    kwargs: dict = {"torch_dtype": dtype, "local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    transformer = MOSSImageTransformer.from_pretrained(load_path, **kwargs).eval().requires_grad_(False)
    if adapter:
        load_lora_into_model(transformer, resolve_adapter_weights(adapter, use_ema),
                             base_model=base_for_adapter)
    return transformer.to(device)


def load_sampling_vae(model: str, device: str, local_files_only: bool, revision: str | None):
    """fp32 VAE or None when the model directory ships no vae/ component."""
    from diffusers import AutoencoderKLFlux2

    load_path, subfolder = resolve_pretrained_target(model, "vae")
    # A local checkout without vae/ (or a nonexistent component path) has no config.json;
    # a Hub id is not a local directory and is resolved by from_pretrained itself.
    if Path(model).is_dir() and not (Path(load_path) / "config.json").is_file():
        return None
    kwargs: dict = {"torch_dtype": torch.float32, "local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    return AutoencoderKLFlux2.from_pretrained(load_path, **kwargs).eval().requires_grad_(False).to(device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="moss-lora-sample",
                                     description="Sample images from MOSS-Image (+LoRA).")
    parser.add_argument("--model", required=True, help="MOSS-Image model dir or Hub id")
    parser.add_argument("--prompt", nargs="+", required=True, help="One or more prompts")
    parser.add_argument("--lora", default=None,
                        help="Adapter dir (adapter_model.safetensors) or weights file")
    parser.add_argument("--ema", action="store_true", help="Load adapter_ema.safetensors instead")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0, help="guidance_scale (1.0 disables CFG)")
    parser.add_argument("--transformer", default=None,
                        help="Transformer override (e.g. a merged save_pretrained dir); "
                             "--model still supplies the text encoder, tokenizer and VAE")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--skip-layer", type=int, default=22,
                        help="Negative-branch skipped block; -1 disables skipping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024, help="Multiple of 16")
    parser.add_argument("--width", type=int, default=1024, help="Multiple of 16")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="outputs", help="Output PNG path or directory")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--output-type", default="pil", choices=["pil", "latent", "pt", "np"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-text-length", type=int, default=1024,
                        help="Tokenizer truncation limit (cache default is 1024; official max 4096)")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]

    # Phase 1: text encoding only; the encoder is freed before the DiT is loaded.
    text_load_path, text_subfolder = resolve_pretrained_target(args.model, "text_encoder")
    tokenizer_path = resolve_tokenizer_path(args.model)
    encoder: TextEncoder = load_text_encoder(
        text_load_path, device=device, dtype=dtype, local_files_only=args.local_files_only,
        revision=args.revision, max_length=args.max_text_length, subfolder=text_subfolder,
        tokenizer_path=tokenizer_path)
    prompt_embeds = [encoder.encode(prompt).detach() for prompt in args.prompt]
    negative_embeds = encoder.encode(args.negative_prompt).detach()
    encoder.unload()

    # Phase 2: transformer (+ optional LoRA) on the execution device, then the pipeline.
    transformer = load_sampling_transformer(
        args.transformer or args.model, str(device), dtype, args.local_files_only, args.revision,
        args.lora, args.ema, base_for_adapter=str(args.model))
    vae = None if args.output_type == "latent" else load_sampling_vae(
        args.model, str(device), args.local_files_only, args.revision)
    pipe = MOSSImagePipeline(transformer=transformer, scheduler=MOSSImageScheduler(shift=args.shift),
                             text_encoder=None, tokenizer=None, vae=vae)
    pipe.set_progress_bar_config(disable=True)
    skip_layers = [] if args.skip_layer < 0 else [args.skip_layer]

    output = Path(args.output)
    if len(args.prompt) > 1 and output.suffix:
        raise ValueError("--output must be a directory when sampling multiple prompts.")
    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
    for index, (prompt, embeds) in enumerate(zip(args.prompt, prompt_embeds)):
        images = pipe(
            prompt_embeds=embeds[None].to(device),
            negative_prompt_embeds=negative_embeds[None].to(device),
            height=args.height, width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            uncond_skip_layers=skip_layers,
            generator=torch.Generator("cpu").manual_seed(args.seed + index),
            output_type=args.output_type,
        ).images
        stem = output.with_suffix("").name if output.suffix else "image"
        if args.output_type == "pil":
            path = output if output.suffix else output / f"{stem}_{index:02d}.png"
            # The pipeline's image processor already returns PIL for output_type="pil".
            images[0].save(path)
            print(f"saved {path}")
        else:
            path = output if output.suffix else output / f"{stem}_{index:02d}"
            image = images[0]
            if args.output_type == "np":
                path = path.with_suffix(".npy")
                np.save(path, image)
                print(f"saved {path} (np array)")
            else:
                path = path.with_suffix(".pt")
                if hasattr(image, "cpu"):
                    image = image.cpu()
                torch.save(image, path)
                print(f"saved {path} ({args.output_type} tensor)")


if __name__ == "__main__":
    sys.exit(main())
