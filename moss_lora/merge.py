"""Merge CLI: fold a trained adapter into the base transformer and save it."""

from __future__ import annotations

import argparse
import sys

import torch

from moss_lora.lora import load_lora_into_model, merge_lora
from moss_lora.model import MOSSImageTransformer
from moss_lora.utils import resolve_pretrained_target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="moss-lora-merge",
                                     description="Merge a LoRA adapter into MOSS-Image-Base weights.")
    parser.add_argument("--model", required=True, help="MOSS-Image model dir (transformer/) or Hub id")
    parser.add_argument("--lora", required=True,
                        help="Adapter dir (adapter_model.safetensors) or weights file")
    parser.add_argument("--output", required=True,
                        help="save_pretrained directory for the merged transformer")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--device", default="cpu", help="Merge on CPU unless the model already fits on GPU")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    load_path, subfolder = resolve_pretrained_target(args.model, "transformer")
    kwargs: dict = {"torch_dtype": dtype, "local_files_only": args.local_files_only}
    if args.revision:
        kwargs["revision"] = args.revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    transformer = MOSSImageTransformer.from_pretrained(load_path, **kwargs).eval().requires_grad_(False)
    transformer.to(device)
    load_lora_into_model(transformer, args.lora, base_model=str(args.model))
    merge_lora(transformer)
    transformer.save_pretrained(args.output, safe_serialization=True)
    print(
        f"Merged transformer saved to {args.output}.\n"
        "Sample it with the vendored pipeline, passing the merged dir as the transformer\n"
        "override and the base model for the text encoder/VAE:\n"
        f"  moss-lora-sample --model {args.model} --transformer {args.output} --prompt \"...\"\n"
        "The vendored MOSSImagePipeline (moss_lora.model.moss_image) is required; stock diffusers\n"
        "does not ship a MOSS-Image pipeline, and the merged directory contains only the transformer."
    )


if __name__ == "__main__":
    sys.exit(main())
