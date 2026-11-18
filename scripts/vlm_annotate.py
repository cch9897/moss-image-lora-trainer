#!/usr/bin/env python
"""Annotate images with a local llama.cpp/OpenAI-compatible VLM (resumable).

This script never runs the full pool unless pointed at it, never downloads a
model and always uses the deterministic metadata fields for artists/characters.
Output is one JSONL record per image keyed by image SHA-256; reruns skip valid
records unless ``--force`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.annotations import (  # noqa: E402
    bucket_bbox_to_normalized,
)
from moss_lora.cache import assign_bucket, build_buckets  # noqa: E402
from moss_lora.ocr import bucket_geometry  # noqa: E402
from moss_lora.utils import atomic_write_bytes, atomic_write_json  # noqa: E402
from moss_lora.vlm import (  # noqa: E402
    PROMPT_VERSION,
    VLMClient,
    VLMConfig,
    reference_tags,
    sanitize_reference,
)

ANNOTATIONS_NAME = "annotations_vlm.jsonl"
REPORT_NAME = "report.json"
VALID_STATUSES = frozenset({"ok", "repaired"})
DEFAULT_MAX_ATTEMPTS = 3
REFERENCE_PRIORITY_TAGS = (
    "nude", "naked", "nipple", "pussy", "vagina", "vulva", "clitoris", "penis",
    "testicle", "anus", "anal", "sex", "intercourse", "oral", "fellatio", "paizuri",
    "masturbat", "cum", "semen", "ejaculat", "bondage", "rope", "uncensored",
    "mosaic", "topless", "bottomless",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    if isinstance(path, (list, tuple)):
        return [dict(row) for row in path if isinstance(row, Mapping)]
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    atomic_write_bytes(path, payload.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_image_path(row: Mapping[str, Any], images_root: str | Path) -> Path:
    root = Path(images_root)
    source = row.get("source_path")
    if isinstance(source, str) and source:
        path = Path(source)
        return path if path.is_absolute() else root / path
    name = row.get("file_name")
    if isinstance(name, str) and name:
        return root / name
    raise FileNotFoundError("metadata row has neither source_path nor file_name")


def row_image_sha(row: Mapping[str, Any], images_root: str | Path) -> str:
    for key in ("image_sha256", "sha256"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    return sha256_file(row_image_path(row, images_root))


def row_size(row: Mapping[str, Any], images_root: str | Path) -> tuple[int, int]:
    width, height = row.get("width"), row.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    from PIL import Image, ImageOps

    with Image.open(row_image_path(row, images_root)) as opened:
        image = ImageOps.exif_transpose(opened)
        return image.size


def _row_ocr_layout(row: Mapping[str, Any], ocr_index: Mapping[str, Mapping[str, Any]]
                    ) -> Mapping[str, Any] | None:
    for key in ("image_sha256", "sha256"):
        value = row.get(key)
        if isinstance(value, str) and value in ocr_index:
            return ocr_index[value]
    for key in ("source_path", "file_name"):
        value = row.get(key)
        if isinstance(value, str) and value in ocr_index:
            return ocr_index[value]
    layout = row.get("layout_ocr")
    if isinstance(layout, Mapping):
        return layout
    layout = row.get("layout")
    return layout if isinstance(layout, Mapping) else None


def ocr_context_for_row(row: Mapping[str, Any], images_root: str | Path, target_area: int,
                        ocr_index: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    layout = _row_ocr_layout(row, ocr_index or {})
    if not isinstance(layout, Mapping):
        return []
    items = layout.get("items")
    if not isinstance(items, list):
        return []
    space = layout.get("bbox_coord_space", "abs_centralized")
    context: list[dict[str, Any]] = []
    if space == "normalized_1000":
        for item in items:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                context.append({"text": item["text"], "bbox": item.get("bbox")})
    else:
        try:
            width, height = row_size(row, images_root)
            bucket_w, bucket_h = row.get("bucket_w"), row.get("bucket_h")
            if not (isinstance(bucket_w, int) and isinstance(bucket_h, int)):
                bucket_w, bucket_h = assign_bucket(width, height, build_buckets(target_area))
            geometry = bucket_geometry((width, height), (bucket_w, bucket_h))
        except (OSError, ValueError):
            return []
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                continue
            bbox = item.get("bbox")
            try:
                normalized = bucket_bbox_to_normalized(bbox, geometry)
            except Exception:
                continue
            context.append({"text": item["text"], "bbox": list(normalized)})
    return context[:20]


def _reference_base(row: Mapping[str, Any]) -> str:
    for key in ("reference_prose", "source_prose", "prose"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return sanitize_reference(value).strip()
    original = row.get("caption_original")
    if isinstance(original, str) and original.strip():
        cleaned = sanitize_reference(original).strip()
        if cleaned:
            return cleaned
    caption = row.get("caption")
    if isinstance(caption, Mapping):
        long_value = caption.get("long")
        if isinstance(long_value, str) and long_value.strip() and not long_value.lstrip().startswith("{"):
            return sanitize_reference(long_value).strip()
    return ""


def reference_for_row(row: Mapping[str, Any]) -> str:
    """Original prose plus a compact, NSFW-first ground-truth tag list."""
    base = _reference_base(row)
    raw_tags = row.get("tags")
    tags = [str(tag).replace("_", " ").strip() for tag in raw_tags] if isinstance(
        raw_tags, list) else []
    tags = [tag for tag in tags if tag]
    if tags:
        priority = [tag for tag in tags if any(
            key in tag.lower() for key in REFERENCE_PRIORITY_TAGS)]
        rest = [tag for tag in tags if tag not in priority]
        ordered = list(dict.fromkeys(priority + rest))[:40]
        suffix = "Ground-truth tags: " + ", ".join(ordered)
        base = f"{base}\n{suffix}".strip() if base else suffix
    if base:
        return base
    return reference_tags(tags)


def _ocr_sha(context: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(context), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _existing_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (record.get("image_sha256"), record.get("model"),
            record.get("prompt_version"), record.get("config_signature"))


def _is_resumable(record: Mapping[str, Any]) -> bool:
    return record.get("status") in VALID_STATUSES and record.get("annotation_validated")


def annotate_row(row: Mapping[str, Any], client: VLMClient, *, images_root: str | Path,
                 model_sha: str | None, target_area: int,
                 ocr_index: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    path = row_image_path(row, images_root)
    image_sha = row_image_sha(row, images_root)
    width, height = row_size(row, images_root)
    context = ocr_context_for_row(row, images_root, target_area, ocr_index)
    reference = reference_for_row(row)
    result = client.annotate_image(path, reference, ocr_context=context,
                                   source_size=(width, height))
    payload = result.to_dict()
    payload.update({
        "file_name": row.get("file_name"),
        "source_path": str(path),
        "image_sha256": image_sha,
        "source_width": width,
        "source_height": height,
        "model_sha": model_sha or row.get("model_sha"),
        "config_signature": client.config.signature(),
        "ocr_sha256": _ocr_sha(context),
        "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def _report(records: Sequence[Mapping[str, Any]], skipped: int, total: int) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    latencies: list[float] = []
    tokens = 0
    for record in records:
        status = str(record.get("status", "invalid"))
        statuses[status] = statuses.get(status, 0) + 1
        latencies.append(float(record.get("latency_seconds", 0.0) or 0.0))
        tokens += int((record.get("usage") or {}).get("total_tokens", 0) or 0)
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    refusals = statuses.get("refused", 0)
    return {
        "prompt_version": PROMPT_VERSION,
        "total_rows": total,
        "records": len(records),
        "skipped_resume": skipped,
        "statuses": statuses,
        "refusal_rate": round(refusals / max(1, len(records)), 4),
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95": round(p95, 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "total_tokens": tokens,
        "fallbacks": sum(1 for record in records if record.get("fallback")),
        "routes": {route: sum(1 for record in records if record.get("route") == route)
                   for route in ("caption", "combined", "split", "primary", "fallback")},
    }


def annotate_metadata(
    rows: Sequence[Mapping[str, Any]],
    client: VLMClient,
    *,
    images_root: str | Path,
    output: str | Path,
    model_sha: str | None = None,
    target_area: int = 1024 * 1024,
    limit: int | None = None,
    force: bool = False,
    concurrency: int = 1,
    flush_every: int = 10,
    ocr_rows: Sequence[Mapping[str, Any]] = (),
    progress: bool = True,
) -> dict[str, Any]:
    """Annotate rows and atomically maintain ``annotations_vlm.jsonl``.

    Existing valid records are skipped by ``(image_sha256, model, prompt_version)``.
    """
    output_path = Path(output)
    if output_path.suffix.lower() != ".jsonl":
        output_path = output_path / ANNOTATIONS_NAME
    existing = read_jsonl(output_path) if output_path.is_file() else []
    existing_by_key = {_existing_key(record): record for record in existing
                       if _is_resumable(record)}
    ocr_index: dict[str, Mapping[str, Any]] = {}
    for row in ocr_rows:
        layout = row.get("layout_ocr") or row.get("layout")
        if not isinstance(layout, Mapping):
            continue
        for key in ("image_sha256", "sha256", "source_path", "file_name"):
            value = row.get(key)
            if isinstance(value, str) and value:
                ocr_index.setdefault(value, layout)

    selected = list(rows[:limit] if limit is not None else rows)
    records: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    pending: list[tuple[int, Mapping[str, Any], str]] = []
    for index, row in enumerate(selected):
        try:
            image_sha = row_image_sha(row, images_root)
        except OSError as error:
            records.append((index, {
                "file_name": row.get("file_name"), "source_path": row.get("source_path"),
                "image_sha256": row.get("image_sha256"), "status": "invalid",
                "errors": [f"missing image: {error}"], "model": client.config.model,
                "prompt_version": PROMPT_VERSION,
            }))
            continue
        key = (image_sha, client.config.model, PROMPT_VERSION,
               client.config.signature())
        if not force and key in existing_by_key:
            skipped += 1
            continue
        pending.append((index, row, image_sha))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_one(index: int, row: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            record = annotate_row(row, client, images_root=images_root, model_sha=model_sha,
                                  target_area=target_area, ocr_index=ocr_index)
        except Exception as error:  # keep the batch alive; record the failure
            record = {
                "file_name": row.get("file_name"),
                "source_path": row.get("source_path"),
                "image_sha256": row.get("image_sha256"),
                "status": "invalid", "model": client.config.model,
                "prompt_version": PROMPT_VERSION,
                "errors": [f"{type(error).__name__}: {error}"],
            }
        return index, record

    def merged_records() -> list[dict[str, Any]]:
        merged = list(existing)
        positions = {_existing_key(record): position for position, record in enumerate(merged)}
        for _index, record in sorted(records, key=lambda pair: pair[0]):
            key = _existing_key(record)
            if key in positions:
                merged[positions[key]] = record
            else:
                positions[key] = len(merged)
                merged.append(record)
        return merged

    def flush() -> None:
        write_jsonl_atomic(output_path, merged_records())

    if concurrency <= 1:
        for index, row, _sha in pending:
            records.append(run_one(index, row))
            if flush_every > 0 and len(records) % flush_every == 0:
                flush()
    else:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {
                executor.submit(run_one, index, row): index
                for index, row, _ in pending
            }
            for future in as_completed(futures):
                records.append(future.result())
                if flush_every > 0 and len(records) % flush_every == 0:
                    flush()

    flush()
    report = _report(merged_records(), skipped, len(selected))
    report["config"] = {
        "endpoint": client.config.endpoint,
        "model": client.config.model,
        "model_family": client.config.model_family,
        "layout_mode": client.config.layout_mode,
        "temperature": client.config.temperature,
        "top_p": client.config.top_p,
        "max_tokens": client.config.max_tokens,
        "layout_max_tokens": client.config.layout_max_tokens,
        "thinking": client.config.thinking,
        "response_format": client.config.response_format,
        "use_system_prompt": client.config.use_system_prompt,
        "max_image_side": client.config.max_image_side,
        "prompt_path": client.config.prompt_path,
        "prompt_sha256": client.config.prompt_sha256,
        "system_prompt_path": client.config.system_prompt_path,
        "system_prompt_sha256": client.config.system_prompt_sha256,
        "signature": client.config.signature(),
    }
    atomic_write_json(output_path.parent / REPORT_NAME, report)
    if progress:
        print(json.dumps({key: report[key] for key in ("records", "statuses", "skipped_resume")}),
              file=sys.stderr)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="moss-lora-vlm",
        description="Annotate staged metadata with a local OpenAI-compatible VLM.")
    parser.add_argument("--input", required=True, help="Metadata JSONL (staging or a small pilot)")
    parser.add_argument("--images", default=None, help="Image root (default: input's parent)")
    parser.add_argument("--output", default="vlm", help="Output directory or annotations JSONL path")
    parser.add_argument("--endpoint", required=True, help="llama.cpp/OpenAI-compatible chat completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-family", default="auto", choices=["auto", "gemma", "qwen"])
    parser.add_argument("--model-sha", default=None)
    parser.add_argument("--fallback-endpoint", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--fallback-family", default="qwen", choices=["auto", "gemma", "qwen"])
    parser.add_argument("--fallback-api-key", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max-http-retries", type=int, default=1)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--max-image-side", type=int, default=None)
    parser.add_argument("--layout-mode", default="combined",
                        choices=["combined", "caption", "split"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--layout-max-tokens", type=int, default=4096)
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-file", default=None,
                        help="Caption prompt template (default: docs/caption_prompt.txt)")
    parser.add_argument("--system-prompt-file", default=None,
                        help="System prompt file (default: built-in)")
    parser.add_argument("--target-area", type=int, default=1024 * 1024)
    parser.add_argument("--ocr-jsonl", default=None, help="Optional OCR metadata keyed by SHA/file_name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Re-annotate rows already present")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    images_root = Path(args.images) if args.images else input_path.parent
    output = Path(args.output)
    if output.suffix == ".jsonl":
        output_path = output
    else:
        output_path = output / ANNOTATIONS_NAME
    config = VLMConfig(
        endpoint=args.endpoint, model=args.model, api_key=args.api_key,
        model_family=args.model_family, layout_mode=args.layout_mode,
        timeout=args.timeout, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, layout_max_tokens=args.layout_max_tokens,
        response_format=args.json_mode, use_system_prompt=args.system_prompt,
        prompt_path=args.prompt_file, system_prompt_path=args.system_prompt_file,
        max_attempts=args.max_attempts, max_http_retries=args.max_http_retries,
        backoff_base=args.backoff_base, max_image_side=args.max_image_side,
        thinking=args.thinking,
    )
    fallback = None
    if args.fallback_endpoint and args.fallback_model:
        fallback = VLMConfig(
            endpoint=args.fallback_endpoint, model=args.fallback_model,
            api_key=args.fallback_api_key, model_family=args.fallback_family,
            layout_mode=args.layout_mode,
            timeout=args.timeout, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, layout_max_tokens=args.layout_max_tokens,
            response_format=args.json_mode, use_system_prompt=args.system_prompt,
            prompt_path=args.prompt_file, system_prompt_path=args.system_prompt_file,
            max_attempts=args.max_attempts,
            max_http_retries=args.max_http_retries, backoff_base=args.backoff_base,
            max_image_side=args.max_image_side, thinking=args.thinking,
        )
    client = VLMClient(config, fallback)
    ocr_rows = read_jsonl(args.ocr_jsonl) if args.ocr_jsonl else []
    annotate_metadata(
        rows, client, images_root=images_root, output=output_path,
        model_sha=args.model_sha, target_area=args.target_area, limit=args.limit,
        force=args.force, concurrency=args.concurrency, flush_every=args.flush_every,
        ocr_rows=ocr_rows,
    )


if __name__ == "__main__":
    main()
