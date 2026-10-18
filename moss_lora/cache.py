"""Latent/text caching: manifest collection, bucketing, two-phase encoding, atomic resume.

Two phases keep peak memory low (the ~18GB text encoder and the VAE never
coexist, and neither loads at all when the cache is already complete):
  phase 1 - encode every caption variant with the frozen text encoder, store
            per-item `<id>.text.safetensors` plus `negative.safetensors`,
            then free the text encoder;
  phase 2 - load the VAE, encode one latent per image, and write the final
            `<id>.safetensors` combining latent + cached text tensors.

Every file is written atomically; existing files with matching image/caption
hashes and model revision are reused on resume.

Latents are the exact inverse of the pipeline decode: posterior.mode() ->
patchify ([B, 32, 2h, 2w] -> [B, 128, h, w]) -> BatchNorm normalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image, ImageOps
from safetensors.torch import safe_open
from tqdm import tqdm

from moss_lora.utils import (
    atomic_save_safetensors,
    atomic_write_json,
    encode_image_latent,
    image_tensor_from_pil,
    resolve_pretrained_target,
    resolve_tokenizer_path,
    sha256_file,
    sha256_text,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ASPECT_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1), (4, 3), (3, 4), (16, 9), (9, 16), (3, 2), (2, 3), (5, 4), (4, 5), (2, 1), (1, 2),
)
BUCKET_STEP = 64
MIN_SIDE = 512
MAX_SIDE = 2048
TEXT_SUFFIX = ".text.safetensors"
ITEM_SUFFIX = ".safetensors"
NEGATIVE_NAME = "negative.safetensors"
INDEX_NAME = "index.json"
DEFAULT_MAX_TEXT_LENGTH = 1024
CAPTION_KEYS = ("text", "caption", "prompt", "image_description", "prompts", "layout")


# --------------------------------------------------------------------------- buckets


def build_buckets(target_area: int, step: int = BUCKET_STEP,
                  ratios: Iterable[tuple[int, int]] = ASPECT_RATIOS,
                  min_side: int = MIN_SIDE, max_side: int = MAX_SIDE) -> list[tuple[int, int]]:
    """(width, height) buckets near target_area; sides are multiples of `step` within bounds.

    The lower bound relaxes to square/2 for small target areas so requested aspect
    ratios (e.g. 2:1 at 512^2) survive instead of collapsing toward the square.
    """

    def clamp_side(value: float, floor: int) -> int:
        side = int(round(value / step)) * step
        return min(max(side, floor), max_side)

    square = clamp_side(math.sqrt(target_area), step)
    floor = min(min_side, square // 2)
    buckets: set[tuple[int, int]] = set()
    for ratio_w, ratio_h in ratios:
        aspect = ratio_w / ratio_h
        ideal_w, ideal_h = math.sqrt(target_area * aspect), math.sqrt(target_area / aspect)

        def candidates(ideal: float) -> list[int]:
            return sorted({clamp_side(math.floor(ideal / step) * step, floor),
                           clamp_side(math.ceil(ideal / step) * step, floor)})

        # Joint floor/ceil search keeps the requested aspect alive despite the 64-grid snap.
        width, height = min(
            ((w, h) for w in candidates(ideal_w) for h in candidates(ideal_h)),
            key=lambda pair: (abs(math.log(pair[0] / pair[1]) - math.log(aspect)),
                              abs(pair[0] * pair[1] - target_area)),
        )
        buckets.add((width, height))
    buckets.add((square, square))
    return sorted(buckets)


def assign_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    aspect = width / height
    return min(buckets, key=lambda bucket: abs(math.log(aspect) - math.log(bucket[0] / bucket[1])))


def fit_image(image: Image.Image, bucket: tuple[int, int]) -> Image.Image:
    """Cover-resize (LANCZOS) to the bucket then center-crop; EXIF transpose applied."""
    bucket_w, bucket_h = bucket
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = max(bucket_w / image.width, bucket_h / image.height)
    new_w = max(bucket_w, round(image.width * scale))
    new_h = max(bucket_h, round(image.height * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - bucket_w) // 2
    top = (new_h - bucket_h) // 2
    return image.crop((left, top, left + bucket_w, top + bucket_h))


# --------------------------------------------------------------------------- captions


def caption_variants(raw: Any) -> list[str]:
    """All prompt strings one item can train with.

    - {"caption": {"long", "medium", "short"}}: every non-empty variant (1:1:1 sampled later);
    - layout dicts ({"summary": ..., "items": ...}): the compact JSON string the pipeline expects;
    - dicts: first string among text/caption/prompt/image_description keys
      (image_description covers the annotated/sampled sidecar schema);
    - strings: parsed as JSON when they look like objects, else a single variant.
    """
    data: Any = None
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            data = None

    def long_medium_short(candidate: Any) -> list[str]:
        if not isinstance(candidate, dict):
            return []
        variants = [candidate.get(key) for key in ("long", "medium", "short")]
        return [text for text in variants if isinstance(text, str) and text.strip()]

    if isinstance(data, dict):
        # A caption field may be the recaption schema itself, or nested under "caption".
        variants = long_medium_short(data) or long_medium_short(data.get("caption"))
        if variants:
            unique: list[str] = []
            for text in variants:
                if text not in unique:
                    unique.append(text)
            return unique
    if isinstance(data, dict) and ("items" in data or "summary" in data):
        return [json.dumps(data, ensure_ascii=False, separators=(",", ":"))]
    if isinstance(data, dict):
        raw = next((data[key] for key in ("text", "caption", "prompt", "image_description")
                    if isinstance(data.get(key), str)), None)
        if not (isinstance(raw, str) and raw.strip()):
            keys = ", ".join(sorted(data)) or "(none)"
            raise ValueError(
                f"Caption is empty or unsupported (dict keys: {keys}; expected one of "
                f"text/caption/prompt/image_description to hold non-empty text)")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    raise ValueError(f"Caption is empty or unsupported: {raw!r}")


# --------------------------------------------------------------------------- manifests


def _caption_from_row(row: dict[str, Any]) -> Any:
    for key in ("caption", "text", "prompt", "image_description", "layout"):
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def _read_metadata(root: Path) -> dict[str, Any]:
    """imagefolder-style metadata.jsonl / metadata.csv: resolved image path -> raw caption."""
    caps: dict[str, Any] = {}
    for meta in root.rglob("metadata.jsonl"):
        for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("file_name")
            caption = _caption_from_row(row)
            if isinstance(name, str) and name and caption is not None:
                caps[str((meta.parent / name).resolve())] = caption
    for meta in root.rglob("metadata.csv"):
        with open(meta, newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                name = row.get("file_name")
                caption = _caption_from_row(row)
                if name and caption is not None:
                    caps[str((meta.parent / name).resolve())] = caption
    return caps


def _read_sidecar(image_path: Path) -> Any:
    """First non-blank same-stem .json/.txt caption; blank files do not shadow a usable one."""
    for ext in (".json", ".txt"):
        side = image_path.with_suffix(ext)
        if not side.is_file():
            continue
        text = side.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return text
    return None


def collect_items(source: str | Path) -> list[tuple[Path, Any]]:
    """[(image path, raw caption)] from an image folder or a JSONL manifest.

    Folder mode reads metadata.jsonl/metadata.csv plus same-stem .json/.txt sidecars.
    JSONL rows accept `image` (or `file_name`) and
    `caption`/`text`/`prompt`/`image_description`/`layout`;
    a row without a caption falls back to the image's same-stem .json/.txt sidecar.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if source.is_file():
        if source.suffix != ".jsonl":
            raise ValueError(f"JSONL expected, got {source}")
        return _collect_jsonl(source)
    return _collect_folder(source)


def _collect_folder(root: Path) -> list[tuple[Path, Any]]:
    metadata = _read_metadata(root)
    items: list[tuple[Path, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in IMAGE_EXTS or path.name.startswith("."):
            continue
        caption = metadata.get(str(path.resolve()))
        if caption is None:
            caption = _read_sidecar(path)
        items.append((path, caption))
    if not items:
        raise ValueError(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found under {root}")
    return items


def _collect_jsonl(path: Path) -> list[tuple[Path, Any]]:
    items: list[tuple[Path, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({error})") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        image = row.get("image") or row.get("file_name")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{path}:{line_no}: missing 'image' (or 'file_name') path")
        image_path = (path.parent / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{path}:{line_no}: image not found: {image_path}")
        caption = _caption_from_row(row)
        if caption is None:
            # Same-stem sidecars keep manifests small and match folder-mode behavior.
            caption = _read_sidecar(image_path)
        # A caption-less row is returned as None so build_cache can apply
        # missing_caption (skip/error) uniformly for JSONL and folder sources.
        items.append((image_path, caption))
    if not items:
        raise ValueError(f"{path} contains no rows")
    return items


# --------------------------------------------------------------------------- validation


def _file_matches(path: Path, image_sha: str, caption_shas: list[str], revision: str | None) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
    except Exception:
        return False
    try:
        stored_shas = json.loads(meta.get("caption_sha256", "[]"))
    except json.JSONDecodeError:
        return False
    return (meta.get("image_sha256") == image_sha and stored_shas == caption_shas
            and meta.get("revision") == (revision or ""))


def _latent_is_valid(path: Path, bucket: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            shape = handle.get_slice("latent").get_shape()
            keys = set(handle.keys())
    except Exception:
        return False
    expected_hw = (bucket[1] // 16, bucket[0] // 16)  # (h, w)
    return len(shape) == 3 and tuple(shape[1:]) == expected_hw and "text_lengths" in keys


# --------------------------------------------------------------------------- build


def build_cache(
    source: str | Path,
    output_dir: str | Path,
    *,
    text_encoder_factory: Callable[[], Any] | None = None,
    latent_encoder_factory: (Callable[[str], Callable[[torch.Tensor], torch.Tensor]] | None) = None,
    target_area: int = 1024 * 1024,
    latent_dtype: torch.dtype = torch.float32,
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    revision: str | None = None,
    negative_prompt: str = "",
    device: str = "cpu",
    progress: bool = True,
    missing_caption: str = "skip",
) -> Path:
    """Build (or resume) the cache; returns the output directory.

    text_encoder_factory: called lazily only when text embeddings are missing; the
    returned object must provide .encode(prompt, max_length=...) -> [L, D] and is
    unloaded before the latent phase. latent_encoder_factory(device) -> callable
    ([-1,1] image tensor -> [C, h, w] normalized latent), called lazily only when
    latents are missing. A complete cache loads neither model.

    missing_caption: "skip" (default) warns, omits images without a usable caption
    from the index and continues; "error" aborts on the first such image with its
    path. Skipped images are not encoded and get no cache files; item ids are
    reassigned over the kept images in sorted order, so resume is deterministic.
    """
    if missing_caption not in ("skip", "error"):
        raise ValueError(
            f"missing_caption must be 'skip' or 'error', got {missing_caption!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(source)
    buckets = build_buckets(target_area)

    records: list[dict[str, Any]] = []
    skipped: list[Path] = []
    for image_path, raw_caption in items:
        try:
            variants = caption_variants(raw_caption)
        except ValueError as error:
            if missing_caption == "error":
                raise ValueError(
                    f"{image_path}: {error}. Add a non-empty same-stem .json/.txt "
                    f"caption (dict with caption/text/prompt/image_description or "
                    f"plain text), or pass --missing-caption skip to omit this image."
                ) from error
            print(f"warning: skipping {image_path}: {error}", file=sys.stderr)
            skipped.append(image_path)
            continue
        record_id = f"item_{len(records):05d}"
        image_sha = sha256_file(image_path)
        caption_shas = [sha256_text(variant) for variant in variants]
        width, height = Image.open(image_path).size
        bucket = assign_bucket(width, height, buckets)
        records.append({
            "id": record_id, "image": str(image_path), "image_sha256": image_sha,
            "caption_sha256": caption_shas, "variants": variants,
            "bucket_w": bucket[0], "bucket_h": bucket[1],
        })
    print(f"{len(items)} images, {len(skipped)} skipped (no/empty caption)", file=sys.stderr)
    if not records:
        raise ValueError(
            f"No usable captions under {source}: all {len(items)} image(s) were skipped. "
            f"First failing image: {skipped[0] if skipped else 'n/a'}. Fix the sidecars "
            f"(same-stem .json/.txt) or run with --missing-caption error for details.")

    text_path = lambda record: output_dir / f"{record['id']}{TEXT_SUFFIX}"  # noqa: E731
    item_path = lambda record: output_dir / f"{record['id']}{ITEM_SUFFIX}"  # noqa: E731

    def report(records_: Iterable[dict[str, Any]], phase: str):
        return tqdm(records_, desc=phase, unit="item", disable=not progress)

    # ---------------- phase 1: text -------------------------------------------------
    missing_text = [r for r in records if not (
        _file_matches(item_path(r), r["image_sha256"], r["caption_sha256"], revision)
        or _file_matches(text_path(r), r["image_sha256"], r["caption_sha256"], revision))]
    text_dim = None
    if missing_text:
        if text_encoder_factory is None:
            raise RuntimeError("Text embeddings missing but no text encoder factory was provided.")
        encoder = text_encoder_factory()
        for record in report(missing_text, "text"):
            embeddings = [encoder.encode(variant, max_length=max_text_length or 4096).detach().cpu()
                          for variant in record["variants"]]
            embeddings = [embedding.to(torch.bfloat16) for embedding in embeddings]
            text_dim = int(embeddings[0].shape[1])
            tensors = {f"text_{position}": e for position, e in enumerate(embeddings)}
            tensors["text_lengths"] = torch.tensor([e.shape[0] for e in embeddings], dtype=torch.int64)
            atomic_save_safetensors(
                tensors, text_path(record),
                metadata={"format": "pt", "image_sha256": record["image_sha256"],
                          "caption_sha256": json.dumps(record["caption_sha256"]), "revision": revision or ""},
            )
        negative_emb = encoder.encode(negative_prompt, max_length=max_text_length or 4096)
        atomic_save_safetensors(
            {"embedding": negative_emb.detach().cpu().to(torch.bfloat16),
             "length": torch.tensor(negative_emb.shape[0], dtype=torch.int64)},
            output_dir / NEGATIVE_NAME,
            metadata={"format": "pt", "caption_sha256": json.dumps([sha256_text(negative_prompt)]),
                      "revision": revision or ""},
        )
        if hasattr(encoder, "unload"):
            encoder.unload()
    elif not (output_dir / NEGATIVE_NAME).is_file():
        raise RuntimeError("Text cache is complete but negative.safetensors is missing; rebuild the cache.")

    # ---------------- phase 2: latents ----------------------------------------------
    # Shape alone is not enough: re-enumerated ids can point one id at a different
    # image/caption, so the stored hash metadata must match before reuse.
    missing_latent = [r for r in records
                      if not (_file_matches(item_path(r), r["image_sha256"],
                                            r["caption_sha256"], revision)
                              and _latent_is_valid(item_path(r), (r["bucket_w"], r["bucket_h"])))]
    latent_channels = None
    if missing_latent:
        if latent_encoder_factory is None:
            raise RuntimeError("Latents missing but no latent encoder factory was provided.")
        encode = latent_encoder_factory(device)
        for record in report(missing_latent, "latent"):
            # Text may live in the intermediate text-only file or, after a crash during the
            # latent phase, inside a previous (hash-valid) item file.
            source = next((path for path in (text_path(record), item_path(record))
                           if _file_matches(path, record["image_sha256"],
                                            record["caption_sha256"], revision)),
                          None)
            if source is None:
                raise RuntimeError(f"Text cache missing for {record['id']}; rerun the cache build.")
            with safe_open(str(source), framework="pt", device="cpu") as handle:
                text_tensors = {key: handle.get_tensor(key) for key in handle.keys()}
                text_lengths = text_tensors.pop("text_lengths")
            text_tensors.pop("latent", None)
            image = Image.open(record["image"])
            fitted = fit_image(image, (record["bucket_w"], record["bucket_h"]))
            pixel = image_tensor_from_pil(fitted)[None].to(device)
            latent = encode(pixel)[0].to(latent_dtype).cpu()
            latent_channels = int(latent.shape[0])
            item_tensors: dict[str, torch.Tensor] = {"latent": latent, "text_lengths": text_lengths}
            item_tensors.update(text_tensors)
            atomic_save_safetensors(
                item_tensors, item_path(record),
                metadata={"format": "pt", "image_sha256": record["image_sha256"],
                          "caption_sha256": json.dumps(record["caption_sha256"]), "revision": revision or ""},
            )
            text_path(record).unlink(missing_ok=True)  # absent when sourced from an item file
    else:
        with safe_open(str(item_path(records[0])), framework="pt", device="cpu") as handle:
            latent_channels = int(handle.get_slice("latent").get_shape()[0])

    index = {
        "revision": revision or "",
        "target_area": target_area,
        "latent_channels": latent_channels,
        "text_dim": text_dim,
        "buckets": [list(bucket) for bucket in buckets],
        "negative_prompt_sha256": sha256_text(negative_prompt),
        "max_text_length": max_text_length,
        "items": [
            {key: record[key] for key in
             ("id", "image", "image_sha256", "caption_sha256", "bucket_w", "bucket_h")}
            | {"text_lengths": _stored_text_lengths(output_dir, record["id"])}
            for record in records
        ],
    }
    atomic_write_json(output_dir / INDEX_NAME, index)
    if skipped:
        # Re-enumerated ids can leave files from an earlier run outside the index.
        expected = {item_path(record).name for record in records}
        for stale in output_dir.iterdir():
            if (stale.is_file() and stale.name not in expected
                    and re.fullmatch(r"item_\d+(?:\.text)?\.safetensors", stale.name)):
                stale.unlink()
    return output_dir


def _stored_text_lengths(output_dir: Path, item_id: str) -> list[int]:
    path = output_dir / f"{item_id}{ITEM_SUFFIX}"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return [int(value) for value in handle.get_tensor("text_lengths").tolist()]


# --------------------------------------------------------------------------- real encoders


def make_vae_latent_encoder(model_path: str, device: str = "cuda", local_files_only: bool = False,
                            revision: str | None = None):
    """Load the FLUX.2 VAE (fp32) and return image_tensor -> normalized latent."""
    from diffusers import AutoencoderKLFlux2

    load_path, subfolder = resolve_pretrained_target(model_path, "vae")
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    if subfolder:
        kwargs["subfolder"] = subfolder
    vae = AutoencoderKLFlux2.from_pretrained(load_path, **kwargs).eval().requires_grad_(False).to(device)

    def encode(image: torch.Tensor) -> torch.Tensor:
        return encode_image_latent(image, vae)

    return encode


def make_text_encoder(model_path: str, device: str, dtype: torch.dtype = torch.bfloat16,
                      local_files_only: bool = False, revision: str | None = None,
                      max_text_length: int = DEFAULT_MAX_TEXT_LENGTH):
    """Frozen text encoder; tokenizer from the checkout's tokenizer/ (or root) directory."""
    from moss_lora.text import load_text_encoder

    load_path, subfolder = resolve_pretrained_target(model_path, "text_encoder")
    tokenizer_path = resolve_tokenizer_path(model_path)
    return load_text_encoder(load_path, device=device, dtype=dtype, local_files_only=local_files_only,
                             revision=revision, max_length=max_text_length or 4096, subfolder=subfolder,
                             tokenizer_path=tokenizer_path)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moss-lora-cache",
        description="Cache text embeddings and VAE latents for MOSS-Image LoRA training.")
    parser.add_argument("--input", required=True, help="Image folder (sidecars/metadata) or JSONL manifest")
    parser.add_argument("--output", required=True, help="Cache output directory")
    parser.add_argument("--model", required=True,
                        help="MOSS-Image model dir (text_encoder/ + vae/) or Hub id")
    parser.add_argument("--revision", default=None, help="Model revision recorded in the cache index")
    parser.add_argument("--device", default="cuda", help="Device for the text encoder and VAE")
    parser.add_argument("--text-dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--latent-dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--target-area", type=int, default=1024 * 1024,
                        help="Bucket target pixel area (1024^2 default; 512^2 for the fast config)")
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH,
                        help="Tokenizer truncation limit in tokens (storage keeps the true length)")
    parser.add_argument("--negative-prompt", default="",
                        help="Negative prompt cached once for CFG validation sampling")
    parser.add_argument("--missing-caption", default="skip", choices=["skip", "error"],
                        help="skip (default): warn and omit images with no/empty caption; "
                             "error: abort on the first such image with its path")
    parser.add_argument("--local-files-only", action="store_true", help="Never touch the Hub")
    args = parser.parse_args(argv)

    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.text_dtype]
    latent_dtype = torch.float32 if args.latent_dtype == "fp32" else torch.bfloat16
    build_cache(
        args.input,
        args.output,
        text_encoder_factory=lambda: make_text_encoder(
            args.model, args.device, torch_dtype, args.local_files_only, args.revision, args.max_text_length),
        latent_encoder_factory=lambda device_: make_vae_latent_encoder(
            args.model, device_, args.local_files_only, args.revision),
        target_area=args.target_area,
        latent_dtype=latent_dtype,
        max_text_length=args.max_text_length,
        revision=args.revision,
        negative_prompt=args.negative_prompt,
        device=args.device,
        missing_caption=args.missing_caption,
    )


if __name__ == "__main__":
    sys.exit(main())
