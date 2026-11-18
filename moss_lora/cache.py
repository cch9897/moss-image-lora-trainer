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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
MAX_TEXT_LENGTH = 4096
CAPTION_KEYS = ("text", "caption", "prompt", "image_description", "prompts", "layout")
VARIANT_ORDER = ("original", "long", "medium", "short", "layout")
DEFAULT_VARIANT_WEIGHTS: dict[str, float] = {
    "original": 0.16,
    "long": 0.64 / 3,
    "medium": 0.64 / 3,
    "short": 0.64 / 3,
    "layout": 0.20,
}


@dataclass(frozen=True)
class CaptionVariant:
    """One cache text variant: deterministic kind plus the exact text to encode."""

    kind: str
    text: str


def normalize_variant_weights(
    kinds: Iterable[str],
    override: Mapping[str, float] | None = None,
) -> list[float]:
    """Per-item weights for the variants that actually exist.

    Missing kinds are dropped by renormalizing the surviving default weights to
    1.0 (spec 5.1). An explicit override maps kind -> weight; absent kinds get 0.
    If every weight is zero the variants are sampled uniformly.
    """
    kind_list = list(kinds)
    if not kind_list:
        raise ValueError("At least one caption variant is required.")
    weights: list[float] = []
    for kind in kind_list:
        value = float(override.get(kind, 0.0)) if override is not None else float(
            DEFAULT_VARIANT_WEIGHTS.get(kind, 1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Variant weight for {kind!r} must be finite and >= 0, got {value!r}")
        weights.append(value)
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(kind_list)] * len(kind_list)
    return [weight / total for weight in weights]


def parse_variant_weights(spec: str | None) -> dict[str, float] | None:
    """Parse ``--variant-weights long=0.5,medium=0.3,short=0.2``."""
    if spec is None or not spec.strip():
        return None
    weights: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--variant-weights entry must be kind=weight, got {item!r}")
        kind, value = (part.strip() for part in item.split("=", 1))
        if not kind:
            raise ValueError("--variant-weights kind must be non-empty")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"--variant-weights {kind} must be finite and >= 0, got {value!r}")
        weights[kind] = parsed
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("--variant-weights must contain at least one positive weight")
    return weights


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


def _is_layout_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and ("items" in value or "summary" in value)


def _layout_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("layout text is empty")
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            if _is_layout_mapping(parsed):
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return stripped
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"Caption layout must be an object or JSON string, got {type(value).__name__}")


def _caption_records(data: Any) -> list[CaptionVariant]:
    """Extract ordered, deduped variants from one caption payload."""
    if not isinstance(data, Mapping):
        if isinstance(data, str) and data.strip():
            return [CaptionVariant("original", data.strip())]
        raise ValueError(f"Caption is empty or unsupported: {data!r}")

    records: list[CaptionVariant] = []
    if _is_layout_mapping(data) and not any(
            key in data for key in ("caption", "caption_original", "text", "prompt",
                                    "image_description", "long", "medium", "short")):
        return [CaptionVariant("layout", _layout_text(data))]

    original = data.get("caption_original")
    if isinstance(original, str) and original.strip():
        records.append(CaptionVariant("original", original.strip()))

    caption_value: Any = None
    for key in ("caption", "text", "prompt", "image_description"):
        if data.get(key) not in (None, ""):
            caption_value = data[key]
            break
    if caption_value is None and any(key in data for key in ("long", "medium", "short")):
        caption_value = data

    if _is_layout_mapping(caption_value):
        records.append(CaptionVariant("layout", _layout_text(caption_value)))
        caption_value = None
    elif isinstance(caption_value, Mapping):
        inner = caption_value
        if "caption" in inner and not any(key in inner for key in ("long", "medium", "short")):
            inner = inner["caption"]
        if isinstance(inner, Mapping):
            for kind in ("long", "medium", "short"):
                text = inner.get(kind)
                if not isinstance(text, str) or not text.strip():
                    continue
                stripped = text.strip()
                if kind == "long" and stripped.startswith("{"):
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError:
                        parsed = None
                    if _is_layout_mapping(parsed):
                        # Legacy `both` mode replaced caption.long with layout JSON.
                        # Emit it once as the separate layout variant, not as prose.
                        records.append(CaptionVariant("layout", _layout_text(parsed)))
                        continue
                records.append(CaptionVariant(kind, stripped))
        elif isinstance(inner, str) and inner.strip():
            if original in (None, ""):
                records.insert(0, CaptionVariant("original", inner.strip()))
            else:
                records.append(CaptionVariant("long", inner.strip()))
        elif _is_layout_mapping(inner):
            records.append(CaptionVariant("layout", _layout_text(inner)))
    elif isinstance(caption_value, str) and caption_value.strip():
        stripped = caption_value.strip()
        parsed_layout: Any = None
        if stripped.startswith("{"):
            try:
                parsed_layout = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_layout = None
        if _is_layout_mapping(parsed_layout):
            records.append(CaptionVariant("layout", _layout_text(parsed_layout)))
        elif original in (None, ""):
            records.insert(0, CaptionVariant("original", stripped))
        else:
            records.append(CaptionVariant("long", stripped))

    layout = data.get("layout")
    if layout not in (None, ""):
        records.append(CaptionVariant("layout", _layout_text(layout)))

    if not records:
        keys = ", ".join(sorted(data)) or "(none)"
        raise ValueError(
            f"Caption is empty or unsupported (dict keys: {keys}; expected "
            f"caption_original/caption/text/prompt/image_description/layout)")
    unique: list[CaptionVariant] = []
    seen: set[Any] = set()
    for record in records:
        if record.kind == "layout":
            try:
                parsed = json.loads(record.text)
                key: Any = ("layout", json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")))
            except (json.JSONDecodeError, TypeError):
                key = ("layout", record.text)
        else:
            key = record.text
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def caption_variant_records(raw: Any) -> list[CaptionVariant]:
    """All cached text variants for one metadata row.

    A v2 row with ``caption_original`` + ``caption.{long,medium,short}`` +
    ``layout`` yields five variants in ``VARIANT_ORDER``. The layout is always a
    separate ``layout`` variant; it never replaces ``caption.long``. Identical
    texts are deduped (first kind wins), so an all-fallback row keeps one
    ``original`` variant.
    """
    data: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("Caption is empty or unsupported: ''")
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = text
    records = _caption_records(data)
    v2_payload = isinstance(data, Mapping) and bool(
        str(data.get("caption_original") or "").strip())
    if v2_payload:
        order = {kind: position for position, kind in enumerate(VARIANT_ORDER)}
        records.sort(key=lambda record: order.get(record.kind, len(order)))
    return records


def caption_variants(raw: Any) -> list[str]:
    """Backward-compatible list of prompt strings for one item."""
    return [record.text for record in caption_variant_records(raw)]


# --------------------------------------------------------------------------- manifests


def _caption_from_row(row: dict[str, Any]) -> Any:
    """Raw caption payload for one row.

    Legacy rows keep their original single value. A row that also carries
    ``caption_original`` or ``layout`` is passed as a combined payload so the
    cache can emit every v2 variant without the layout replacing ``caption.long``.
    """
    caption = next((row[key] for key in ("caption", "text", "prompt", "image_description")
                    if row.get(key) not in (None, "")), None)
    original = row.get("caption_original")
    layout = row.get("layout")
    parts = [value for value in (caption, original, layout) if value not in (None, "")]
    if len(parts) > 1:
        combined: dict[str, Any] = {}
        if caption is not None:
            combined["caption"] = caption
        if original not in (None, ""):
            combined["caption_original"] = original
        if layout not in (None, ""):
            combined["layout"] = layout
        return combined
    return parts[0] if parts else None


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


def _file_matches(path: Path, image_sha: str, caption_shas: list[str], revision: str | None,
                  max_text_length: int | None = None) -> bool:
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
    stored_max = meta.get("max_text_length")
    expected_max = (DEFAULT_MAX_TEXT_LENGTH if max_text_length is None else int(max_text_length))
    # Legacy files predate this metadata and were tokenized with the 1024 default.
    stored_max_value = int(stored_max) if stored_max not in (None, "") else DEFAULT_MAX_TEXT_LENGTH
    return (meta.get("image_sha256") == image_sha and stored_shas == caption_shas
            and meta.get("revision") == (revision or "")
            and stored_max_value == expected_max)


def _negative_matches(path: Path, prompt_sha: str, revision: str | None,
                      max_text_length: int) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
            stored_length = int(handle.get_tensor("length"))
    except Exception:
        return False
    try:
        stored_shas = json.loads(meta.get("caption_sha256", "[]"))
    except json.JSONDecodeError:
        return False
    stored_max = meta.get("max_text_length")
    stored_max_value = int(stored_max) if stored_max not in (None, "") else DEFAULT_MAX_TEXT_LENGTH
    return (stored_shas == [prompt_sha] and meta.get("revision") == (revision or "")
            and stored_max_value == int(max_text_length) and stored_length > 0)


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
    variant_weights: Mapping[str, float] | None = None,
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

    variant_weights overrides the per-kind weights for every item; kinds absent
    from an item are dropped and surviving weights renormalized to 1.0.
    """
    if missing_caption not in ("skip", "error"):
        raise ValueError(
            f"missing_caption must be 'skip' or 'error', got {missing_caption!r}")
    if not 1 <= int(max_text_length) <= MAX_TEXT_LENGTH:
        raise ValueError(
            f"max_text_length must be in [1, {MAX_TEXT_LENGTH}], got {max_text_length!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_items(source)
    buckets = build_buckets(target_area)

    records: list[dict[str, Any]] = []
    skipped: list[Path] = []
    for image_path, raw_caption in items:
        try:
            caption_records = caption_variant_records(raw_caption)
            variants = [record.text for record in caption_records]
            kinds = [record.kind for record in caption_records]
            weights = normalize_variant_weights(kinds, variant_weights)
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
            "variant_kinds": kinds, "variant_weights": weights,
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
        _file_matches(item_path(r), r["image_sha256"], r["caption_sha256"], revision,
                      max_text_length)
        or _file_matches(text_path(r), r["image_sha256"], r["caption_sha256"], revision,
                         max_text_length))]
    negative_path = output_dir / NEGATIVE_NAME
    negative_ok = _negative_matches(negative_path, sha256_text(negative_prompt), revision,
                                    max_text_length)
    text_dim = None
    if missing_text or not negative_ok:
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
                          "caption_sha256": json.dumps(record["caption_sha256"]),
                          "revision": revision or "",
                          "max_text_length": str(int(max_text_length))},
            )
        negative_emb = encoder.encode(negative_prompt, max_length=max_text_length or 4096)
        atomic_save_safetensors(
            {"embedding": negative_emb.detach().cpu().to(torch.bfloat16),
             "length": torch.tensor(negative_emb.shape[0], dtype=torch.int64)},
            negative_path,
            metadata={"format": "pt", "caption_sha256": json.dumps([sha256_text(negative_prompt)]),
                      "revision": revision or "",
                      "max_text_length": str(int(max_text_length))},
        )
        if hasattr(encoder, "unload"):
            encoder.unload()
    elif not negative_path.is_file():
        raise RuntimeError("Text cache is complete but negative.safetensors is missing; rebuild the cache.")

    # ---------------- phase 2: latents ----------------------------------------------
    # Shape alone is not enough: re-enumerated ids can point one id at a different
    # image/caption, so the stored hash metadata must match before reuse.
    missing_latent = [r for r in records
                      if not (_file_matches(item_path(r), r["image_sha256"],
                                            r["caption_sha256"], revision, max_text_length)
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
                                            record["caption_sha256"], revision,
                                            max_text_length)),
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
                          "caption_sha256": json.dumps(record["caption_sha256"]),
                          "revision": revision or "",
                          "max_text_length": str(int(max_text_length))},
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
        "variant_weight_overrides": dict(variant_weights or {}),
        "items": [
            {key: record[key] for key in
             ("id", "image", "image_sha256", "caption_sha256", "bucket_w", "bucket_h",
              "variant_kinds", "variant_weights")}
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
                        help=f"Tokenizer truncation limit in tokens, max {MAX_TEXT_LENGTH} "
                             f"(storage keeps the true length)")
    parser.add_argument("--variant-weights", default=None,
                        help="Per-kind override, e.g. long=0.5,medium=0.3,short=0.2 "
                             "(missing kinds are renormalized)")
    parser.add_argument("--negative-prompt", default="",
                        help="Negative prompt cached once for CFG validation sampling")
    parser.add_argument("--missing-caption", default="skip", choices=["skip", "error"],
                        help="skip (default): warn and omit images with no/empty caption; "
                             "error: abort on the first such image with its path")
    parser.add_argument("--local-files-only", action="store_true", help="Never touch the Hub")
    args = parser.parse_args(argv)

    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.text_dtype]
    latent_dtype = torch.float32 if args.latent_dtype == "fp32" else torch.bfloat16
    if not 1 <= args.max_text_length <= MAX_TEXT_LENGTH:
        parser.error(f"--max-text-length must be in [1, {MAX_TEXT_LENGTH}]")
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
        variant_weights=parse_variant_weights(args.variant_weights),
    )


if __name__ == "__main__":
    sys.exit(main())
