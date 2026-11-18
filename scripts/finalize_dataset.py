#!/usr/bin/env python
"""Merge staged metadata, PaddleOCR layout and VLM prose into final v2 metadata.

Design correction from the 100-image bakeoff: VLM ``items[].text`` and text
boxes are *not* authoritative. Final ``layout`` text items always come from the
PaddleOCR layout (``layout_ocr``); VLM text items are dropped (they remain in
``raw_annotations`` for audit). VLM object/character boxes are omitted by
default and only included with ``--vlm-objects audit-passed`` after the
200-image audit. VLM prose still supplies long/medium/short, watermark/borders.

Output is one metadata row per image, one separate layout variant (not embedded
in ``caption.long``), raw normalized-1000 annotations for relayout, symlinked
images and a validation report. Nothing here calls a VLM or processes the full
pool unless pointed at it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.annotations import (  # noqa: E402
    BUCKET_SPACE,
    LAYOUT_LABELS,
    MAX_LAYOUT_ITEMS,
    MAX_LAYOUT_TEXT_ITEMS,
    NORMALIZED_SPACE,
    UNKNOWN_ARTIST,
    AnnotationError,
    Layout,
    LayoutItem,
    Watermark,
    box_iou,
    build_caption_variants,
    build_prefix,
    compose_caption,
    dedupe_tags,
    enforce_variant_token_budgets,
    filter_layout_watermarks,
    find_watermark_text,
    layout_json,
    layout_to_bucket,
    layout_to_normalized,
    missing_prefix_labels,
    parse_normalized_box,
    parse_vlm_annotation,
    token_length,
    watermark_area_ratio,
)
from moss_lora.cache import assign_bucket, build_buckets  # noqa: E402
from moss_lora.detectors import item_covered_by_detector  # noqa: E402
from moss_lora.utils import atomic_write_bytes, atomic_write_json  # noqa: E402

FINALIZE_VERSION = "v2"
METADATA_NAME = "metadata.jsonl"
RAW_DIR = "raw_annotations"
REPORT_NAME = "report.json"
REPORT_MD = "report.md"
IMAGES_DIR = "images"
WATERMARK_AREA_THRESHOLD = 0.005


class FinalizeError(RuntimeError):
    """Invalid final metadata configuration or validation failure."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- I/O


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSONL not found: {source}")
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def image_path_for_row(row: Mapping[str, Any], root: Path) -> Path:
    source = row.get("source_path")
    if isinstance(source, str) and source:
        path = Path(source)
        return path if path.is_absolute() else root / path
    name = row.get("file_name")
    if isinstance(name, str) and name:
        return root / name
    raise FinalizeError("base row has neither source_path nor file_name")


def image_sha_for_row(row: Mapping[str, Any], path: Path) -> str:
    for key in ("image_sha256", "sha256"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    return sha256_file(path)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).size


# --------------------------------------------------------------------------- layout helpers


def _layout_from_mapping(mapping: Mapping[str, Any], *, space: str) -> Layout | None:
    items_raw = mapping.get("items")
    if not isinstance(items_raw, list):
        return None
    items: list[LayoutItem] = []
    for raw in items_raw:
        if not isinstance(raw, Mapping):
            continue
        label = raw.get("label", "text")
        if label not in LAYOUT_LABELS:
            continue
        bbox = raw.get("bbox")
        try:
            if space == NORMALIZED_SPACE:
                y1, x1, y2, x2 = parse_normalized_box(
                    bbox, min_area_ratio=0.0, max_area_ratio=1.0)
            else:
                y1, x1, y2, x2 = (int(value) for value in bbox)
                if y2 <= y1 or x2 <= x1:
                    continue
        except (AnnotationError, TypeError, ValueError):
            continue
        description = raw.get("caption")
        if not isinstance(description, str) or not description.strip():
            description = "visible text" if label == "text" else label
        text = raw.get("text") if label == "text" else None
        if label == "text" and (not isinstance(text, str) or not text.strip()):
            continue
        try:
            items.append(LayoutItem(label=label, caption=description,
                                    bbox=(y1, x1, y2, x2), text=text))
        except AnnotationError:
            continue
    if not items:
        return None
    summary = mapping.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    meta = summary.get("meta")
    meta_tuple = tuple(str(value) for value in meta) if isinstance(meta, list) and meta else ("anime",)
    canvas_value = summary.get("canvas")
    canvas = canvas_value if canvas_value in ("single", "multi_panel", "collage") else "single"
    caption = summary.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        caption = "anime illustration"
    return Layout(summary_caption=caption, items=tuple(items), summary_meta=meta_tuple,
                  canvas=canvas, bbox_coord_space=space, bbox_format="yxyx")


def ocr_layout_for_row(row: Mapping[str, Any], width: int, height: int,
                       bucket: tuple[int, int]) -> tuple[Layout | None, Layout | None]:
    """Return ``(normalized_ocr_layout, bucket_ocr_layout)`` text-only.

    Final layout text is OCR-only by contract. Any object/character items in a
    base layout field are ignored here.
    """
    raw = row.get("layout_ocr") or row.get("layout")
    if not isinstance(raw, Mapping):
        return None, None
    space = raw.get("bbox_coord_space", BUCKET_SPACE)
    layout = _layout_from_mapping(raw, space=space)
    if layout is None:
        return None, None
    layout = Layout(
        summary_caption=layout.summary_caption,
        items=tuple(item for item in layout.items if item.label == "text"),
        summary_meta=layout.summary_meta,
        canvas=layout.canvas,
        bbox_coord_space=layout.bbox_coord_space,
        bbox_format=layout.bbox_format,
    )
    if not layout.items:
        return None, None
    if layout.bbox_coord_space == BUCKET_SPACE:
        normalized, _ = layout_to_normalized(layout, (width, height), bucket)
        return normalized, layout
    bucket_layout, _ = layout_to_bucket(layout, (width, height), bucket)
    return layout, bucket_layout


def _ensure_prefix(text: str, prefix: Sequence[str], tags: Sequence[str] = ()) -> str:
    if not text.strip():
        return compose_caption(prefix, tags)
    artists = [part[7:] for part in prefix if part.startswith("artist:")]
    characters = [part[10:] for part in prefix if part.startswith("character:")]
    if not missing_prefix_labels(
            text, prefix[0], artists, characters,
            unknown_if_missing=bool(artists)):
        return text
    return compose_caption(prefix, tags, text)


_UNKNOWN_ARTIST_RE = re.compile(r"(?i)\bartist\s*:\s*unknown\b\s*,?\s*")


def _strip_unknown_artist(text: str) -> str:
    """Remove a placeholder ``artist:unknown`` from an existing caption."""
    return _UNKNOWN_ARTIST_RE.sub("", text).strip(" ,;\n\t")


def _caption_original(row: Mapping[str, Any], prefix: Sequence[str], tags: Sequence[str],
                      source_prose: str) -> str:
    existing = row.get("caption_original")
    if isinstance(existing, str) and existing.strip():
        return _ensure_prefix(existing.strip(), prefix)
    return compose_caption(prefix, tags, source_prose, max_tags=40)


def _source_prose(row: Mapping[str, Any]) -> str:
    for key in ("source_prose", "prose"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    caption = row.get("caption")
    if isinstance(caption, Mapping):
        value = caption.get("long")
        if isinstance(value, str) and value.strip() and not value.lstrip().startswith("{"):
            return value.strip()
    return ""


def _vlm_annotation(record: Mapping[str, Any] | None):
    if not isinstance(record, Mapping) or record.get("status") not in ("ok", "repaired"):
        return None
    validated = record.get("annotation_validated")

    def parse(payload: Mapping[str, Any]):
        layout_raw = payload.get("layout")
        require_layout = bool(
            isinstance(layout_raw, Mapping) and layout_raw.get("items"))
        return parse_vlm_annotation(payload, require_layout=require_layout)

    try:
        if isinstance(validated, Mapping):
            return parse(validated)
        raw = record.get("annotation")
        if isinstance(raw, Mapping) and raw:
            return parse(raw)
    except AnnotationError:
        return None
    return None


def _watermark_overlaps_subject(watermarks: Sequence[Watermark], items: Sequence[LayoutItem],
                                threshold: float = 0.3) -> bool:
    return any(box_iou(mark.bbox, item.bbox) > threshold
               for mark in watermarks for item in items)


def _layout_item_bounds_ok(layout: Layout, bucket: tuple[int, int]) -> list[str]:
    errors: list[str] = []
    half_w, half_h = bucket[0] // 2, bucket[1] // 2
    for index, item in enumerate(layout.items):
        y1, x1, y2, x2 = item.bbox
        if not (-half_h <= y1 < y2 <= half_h and -half_w <= x1 < x2 <= half_w):
            errors.append(f"layout.items[{index}].bbox outside bucket: {item.bbox}")
    return errors


def _layout_for_row(
    row: Mapping[str, Any],
    annotation,
    *,
    width: int,
    height: int,
    bucket: tuple[int, int],
    prefix: Sequence[str],
    tags: Sequence[str],
    include_vlm_objects: bool,
    allow_ocr_layout: bool,
    detector_items: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Layout | None, Layout | None, dict[str, Any]]:
    """Compose the final normalized layout and the OCR-only metadata layout."""
    ocr_norm, ocr_bucket = ocr_layout_for_row(row, width, height, bucket)
    watermarks = annotation.watermark if annotation is not None else ()
    labels = [mark.label for mark in watermarks]
    meta: dict[str, Any] = {
        "vlm_text_items_dropped": 0,
        "vlm_objects_included": 0,
        "vlm_objects_covered_dropped": 0,
        "vlm_objects_dropped_cap": 0,
        "detector_items": 0,
        "detector_items_dropped_cap": 0,
        "text_items_dropped_cap": 0,
        "watermark_action": "none",
        "border_action": "kept" if annotation is not None and not annotation.borders.is_none else "none",
    }

    if ocr_norm is not None:
        kept = []
        for item in ocr_norm.items:
            if item.label != "text":
                continue
            if labels and find_watermark_text(item.text or "", labels):
                meta["vlm_text_items_dropped"] += 1
                continue
            kept.append(item)
        ocr_norm = ocr_norm.with_items(kept)
        ocr_norm, _ = filter_layout_watermarks(ocr_norm, watermarks)

    vlm_layout = annotation.layout if annotation is not None else None
    if vlm_layout is not None:
        meta["vlm_text_items_dropped"] += sum(1 for item in vlm_layout.items if item.label == "text")

    detector_layout: list[LayoutItem] = []
    detector_coverage: list[dict[str, Any]] = []
    for raw in detector_items or ():
        if not isinstance(raw, Mapping):
            continue
        label = raw.get("label")
        if label not in ("object", "character"):
            continue
        try:
            y1, x1, y2, x2 = parse_normalized_box(
                raw.get("bbox"), min_area_ratio=0.0, max_area_ratio=1.0)
        except (AnnotationError, TypeError, ValueError):
            continue
        description = raw.get("caption") or raw.get("tag") or label
        if not isinstance(description, str) or not description.strip():
            description = label
        try:
            detector_layout.append(LayoutItem(
                label=label, caption=description, bbox=(y1, x1, y2, x2)))
            detector_coverage.append({
                "label": label,
                "bbox": [y1, x1, y2, x2],
                "tag": raw.get("tag") or description,
            })
        except AnnotationError:
            continue
    meta["detector_items"] = len(detector_layout)

    object_items: list[LayoutItem] = []
    if include_vlm_objects and vlm_layout is not None:
        candidates = [item for item in vlm_layout.items if item.label in ("character", "object")]
        if detector_layout:
            kept_vlm = []
            for item in candidates:
                if item_covered_by_detector(item.to_dict(), detector_coverage):
                    meta["vlm_objects_covered_dropped"] += 1
                else:
                    kept_vlm.append(item)
            object_items = kept_vlm
        else:
            object_items = candidates
        meta["vlm_objects_included"] = len(object_items)

    text_items = list(ocr_norm.items) if ocr_norm is not None else []
    if len(text_items) > MAX_LAYOUT_TEXT_ITEMS:
        meta["text_items_dropped_cap"] = len(text_items) - MAX_LAYOUT_TEXT_ITEMS
        text_items = text_items[:MAX_LAYOUT_TEXT_ITEMS]
    slots = max(0, MAX_LAYOUT_ITEMS - len(text_items))
    nontext = detector_layout + object_items
    kept_nontext = nontext[:slots]
    detector_kept = min(len(detector_layout), len(kept_nontext))
    vlm_kept = max(0, len(kept_nontext) - detector_kept)
    meta["detector_items_dropped_cap"] = len(detector_layout) - detector_kept
    meta["vlm_objects_dropped_cap"] = len(object_items) - vlm_kept
    combined_items: list[LayoutItem] = text_items + kept_nontext

    excluded_layout = False
    if watermarks:
        ratio = watermark_area_ratio(watermarks)
        overlaps_subject = _watermark_overlaps_subject(watermarks, combined_items)
        if ratio > WATERMARK_AREA_THRESHOLD and overlaps_subject:
            meta["watermark_action"] = "exclude_layout"
            combined_items = []
            excluded_layout = True
        elif ratio > WATERMARK_AREA_THRESHOLD:
            meta["watermark_action"] = "filtered_text"
        else:
            meta["watermark_action"] = "recorded"

    if not combined_items:
        if allow_ocr_layout and not excluded_layout and ocr_norm is not None and ocr_norm.items:
            combined_items = list(ocr_norm.items)
        else:
            return None, None, meta

    summary = (vlm_layout.summary_caption if vlm_layout is not None
               else (ocr_norm.summary_caption if ocr_norm is not None else "anime illustration"))
    summary = _ensure_prefix(summary, prefix, tags)
    meta_tuple = (vlm_layout.summary_meta if vlm_layout is not None
                  else (ocr_norm.summary_meta if ocr_norm is not None else ("anime",)))
    canvas = (vlm_layout.canvas if vlm_layout is not None
              else (ocr_norm.canvas if ocr_norm is not None else "single"))
    combined = Layout(
        summary_caption=summary,
        items=tuple(combined_items),
        summary_meta=tuple(meta_tuple),
        canvas=canvas,
        bbox_coord_space=NORMALIZED_SPACE,
        bbox_format="yxyx",
    )
    bucket_layout, stats = layout_to_bucket(combined, (width, height), bucket)
    if not bucket_layout.items:
        return None, None, meta
    meta["transform"] = stats.as_dict()
    return combined, bucket_layout, meta


# --------------------------------------------------------------------------- finalization


def _row_artists(row: Mapping[str, Any]) -> list[str]:
    values = row.get("artists")
    if isinstance(values, list):
        artists = [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
    else:
        artist = row.get("artist")
        artists = [artist.strip()] if isinstance(artist, str) and artist.strip() else []
    return [value for value in artists if value.lower() != UNKNOWN_ARTIST]


def _row_characters(row: Mapping[str, Any]) -> list[str]:
    values = row.get("character")
    if isinstance(values, list):
        return [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
    if isinstance(values, str) and values.strip():
        return [values.strip()]
    return []


def _row_tags(row: Mapping[str, Any]) -> list[str]:
    values = row.get("tags")
    return dedupe_tags(str(value) for value in values) if isinstance(values, list) else []


def _annotation_payload(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {"model": None, "prompt_version": None, "status": "fallback", "attempts": 0,
                "warnings": []}
    raw = record.get("annotation")
    raw = raw if isinstance(raw, Mapping) else {}
    return {
        "model": record.get("model"),
        "model_sha": record.get("model_sha"),
        "endpoint": record.get("endpoint"),
        "prompt_version": record.get("prompt_version"),
        "prompt_sha256": record.get("prompt_sha256"),
        "config_signature": record.get("config_signature"),
        "status": "fallback" if record.get("status") not in ("ok", "repaired") else record.get("status"),
        "attempts": record.get("attempts", 0),
        "refusal": record.get("refusal"),
        "warnings": list(record.get("warnings") or []),
        "rating": raw.get("rating"),
        "nsfw": raw.get("nsfw"),
        "raw_layout_space": f"{NORMALIZED_SPACE}_yxyx",
        "source_width": record.get("source_width"),
        "source_height": record.get("source_height"),
        "ocr_sha256": record.get("ocr_sha256"),
    }


def _symlink_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    os.symlink(os.path.relpath(source.resolve(), target.parent), target)


def finalize_dataset(
    base: str | Path,
    vlm: str | Path | None,
    output: str | Path,
    *,
    images_root: str | Path | None = None,
    ocr: str | Path | None = None,
    detectors: str | Path | None = None,
    trigger: str = "@CH3OCH3",
    target_area: int = 1024 * 1024,
    layout_weight: float = 0.20,
    original_weight: float = 0.20,
    max_tags: int = 40,
    tokenizer: Any | None = None,
    include_vlm_objects: bool = False,
    allow_ocr_layout: bool = False,
    limit: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge base + OCR + VLM annotations and write final metadata/report."""
    if not 0.0 <= layout_weight <= 0.3:
        raise FinalizeError("layout_weight must be in [0, 0.3]")
    if not 0.0 <= original_weight <= 1.0:
        raise FinalizeError("original_weight must be in [0, 1]")
    base_path = Path(base)
    base_rows = read_jsonl(base_path)
    if limit is not None:
        base_rows = base_rows[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / RAW_DIR).mkdir(parents=True, exist_ok=True)
    (output / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    images_root = Path(images_root) if images_root else base_path.parent

    vlm_path = Path(vlm) if vlm is not None else None
    if vlm_path is not None and vlm_path.is_dir():
        vlm_path = vlm_path / "annotations_vlm.jsonl"
    vlm_records = read_jsonl(vlm_path) if vlm_path is not None else []
    vlm_candidates: dict[str, list[Mapping[str, Any]]] = {}
    for record in vlm_records:
        for key in ("image_sha256", "sha256", "source_path", "file_name"):
            value = record.get(key)
            if isinstance(value, str) and value:
                vlm_candidates.setdefault(value, []).append(record)
    vlm_index: dict[str, Mapping[str, Any]] = {}
    for key, candidates in vlm_candidates.items():
        # Prefer the latest valid annotation when model/prompt versions coexist.
        vlm_index[key] = next(
            (record for record in reversed(candidates)
             if record.get("status") in ("ok", "repaired")),
            candidates[-1],
        )

    ocr_rows = read_jsonl(ocr) if ocr is not None else []
    ocr_index: dict[str, Mapping[str, Any]] = {}
    for row in ocr_rows:
        layout = row.get("layout_ocr") or row.get("layout")
        if not isinstance(layout, Mapping):
            continue
        for key in ("image_sha256", "sha256", "source_path", "file_name"):
            value = row.get(key)
            if isinstance(value, str) and value:
                ocr_index.setdefault(value, layout)

    detector_rows = read_jsonl(detectors) if detectors is not None else []
    detector_index: dict[str, list[Mapping[str, Any]]] = {}
    for row in detector_rows:
        items = row.get("items")
        if not isinstance(items, list):
            continue
        for key in ("image_sha256", "sha256", "source_path", "file_name"):
            value = row.get(key)
            if isinstance(value, str) and value:
                detector_index.setdefault(value, items)

    buckets = build_buckets(target_area)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    actions: dict[str, int] = {}
    borders_seen: dict[str, int] = {}
    token_samples: dict[str, list[int]] = {name: [] for name in
                                           ("original", "long", "medium", "short", "layout")}
    variant_counts = {"original": 0, "long": 0, "medium": 0, "short": 0, "layout": 0}
    layout_items = 0
    detector_items_included = 0
    detector_items_capped = 0
    text_items_capped = 0
    vlm_objects_covered = 0

    for index, base_row in enumerate(base_rows):
        try:
            source_path = image_path_for_row(base_row, images_root)
            if not source_path.is_file():
                raise FinalizeError(f"image not found: {source_path}")
            image_sha = image_sha_for_row(base_row, source_path)
            width, height = (base_row.get("width"), base_row.get("height"))
            if not (isinstance(width, int) and isinstance(height, int)):
                width, height = image_size(source_path)
            bucket = assign_bucket(width, height, buckets)
            external_ocr = None
            for key in (image_sha, base_row.get("source_path"), base_row.get("file_name")):
                if isinstance(key, str) and key in ocr_index:
                    external_ocr = ocr_index[key]
                    break
            row_for_ocr = dict(base_row)
            if external_ocr is not None:
                row_for_ocr["layout_ocr"] = external_ocr
            vlm_record = None
            for key in (image_sha, base_row.get("source_path"), base_row.get("file_name")):
                if isinstance(key, str) and key in vlm_index:
                    vlm_record = vlm_index[key]
                    break
            annotation = _vlm_annotation(vlm_record)
            detector_items = None
            for key in (image_sha, base_row.get("source_path"), base_row.get("file_name")):
                if isinstance(key, str) and key in detector_index:
                    detector_items = detector_index[key]
                    break

            artists = _row_artists(base_row)
            characters = _row_characters(base_row)
            tags = _row_tags(base_row)
            if annotation is not None and annotation.nsfw:
                nsfw_tags = [str(tag).strip() for tag in annotation.nsfw.get("tags", [])
                             if str(tag).strip()]
                tags = dedupe_tags(nsfw_tags + list(tags))
            prefix = build_prefix(trigger, artists, characters,
                                  unknown_if_missing=bool(artists))
            source_prose = _source_prose(base_row)
            caption_original = _caption_original(base_row, prefix, tags, source_prose)
            if annotation is not None:
                variants = build_caption_variants(
                    annotation.caption, original_prose=source_prose, trigger=trigger,
                    artists=artists, characters=characters, tags=tags, max_tags=max_tags,
                    unknown_if_missing=bool(artists))
                long_text, medium_text, short_text = variants.long, variants.medium, variants.short
            else:
                long_text = compose_caption(prefix, tags, source_prose, max_tags=40)
                medium_text = compose_caption(prefix, tags, source_prose, max_tags=25)
                short_text = compose_caption(prefix, tags, source_prose, max_tags=12)
            if not artists:
                caption_original = _strip_unknown_artist(caption_original)
                long_text = _strip_unknown_artist(long_text)
                medium_text = _strip_unknown_artist(medium_text)
                short_text = _strip_unknown_artist(short_text)
            caption_original = _ensure_prefix(caption_original, prefix)
            caption_no_artist = compose_caption(
                build_prefix(trigger, artists, characters, include_artist=False), tags,
                annotation.caption.long if annotation is not None else source_prose,
                max_tags=40)
            if not artists:
                caption_no_artist = _strip_unknown_artist(caption_no_artist)

            raw_layout, bucket_layout, action_meta = _layout_for_row(
                row_for_ocr, annotation, width=width, height=height, bucket=bucket,
                prefix=prefix, tags=tags,
                include_vlm_objects=include_vlm_objects, allow_ocr_layout=allow_ocr_layout,
                detector_items=detector_items)
            actions[action_meta.get("watermark_action", "none")] = (
                actions.get(action_meta.get("watermark_action", "none"), 0) + 1)
            detector_items_included += int(action_meta.get("detector_items", 0))
            detector_items_capped += int(action_meta.get("detector_items_dropped_cap", 0))
            text_items_capped += int(action_meta.get("text_items_dropped_cap", 0))
            vlm_objects_covered += int(action_meta.get("vlm_objects_covered_dropped", 0))
            border_value = ",".join(annotation.borders.edges) if annotation is not None else "none"
            borders_seen[border_value] = borders_seen.get(border_value, 0) + 1

            out_name = f"{IMAGES_DIR}/{index + 1:06d}{source_path.suffix.lower()}"
            _symlink_image(source_path, output / out_name)
            annotation_payload = _annotation_payload(vlm_record)
            annotation_payload.update({
                "vlm_text_items_dropped": action_meta.get("vlm_text_items_dropped", 0),
                "vlm_objects_included": action_meta.get("vlm_objects_included", 0),
                "vlm_objects_covered_dropped": action_meta.get("vlm_objects_covered_dropped", 0),
                "detector_items": action_meta.get("detector_items", 0),
                "detector_items_dropped_cap": action_meta.get("detector_items_dropped_cap", 0),
                "text_items_dropped_cap": action_meta.get("text_items_dropped_cap", 0),
                "watermark_action": action_meta.get("watermark_action", "none"),
                "border_action": action_meta.get("border_action", "none"),
            })
            layout_ocr_meta = None
            _, ocr_bucket = ocr_layout_for_row(row_for_ocr, width, height, bucket)
            if ocr_bucket is not None:
                layout_ocr_meta = ocr_bucket.to_dict()

            row: dict[str, Any] = {
                "file_name": out_name,
                "source_path": str(source_path),
                "image_sha256": image_sha,
                "width": width,
                "height": height,
                "bucket_w": bucket[0],
                "bucket_h": bucket[1],
                "caption": {"long": long_text, "medium": medium_text, "short": short_text},
                "caption_original": caption_original,
                "caption_no_artist": caption_no_artist,
                "artist": artists[0] if artists else UNKNOWN_ARTIST,
                "artists": artists,
                "character": characters,
                "tags": tags,
                "tag_confidence": base_row.get("tag_confidence") or {},
                "rating": (annotation.rating if annotation is not None and annotation.rating
                           else base_row.get("rating")),
                "nsfw": annotation.nsfw if annotation is not None else None,
                "watermark": [mark.to_dict() for mark in (annotation.watermark if annotation else ())],
                "borders": annotation.borders.to_list() if annotation is not None else ["none"],
                "annotation": annotation_payload,
                "finalize_version": FINALIZE_VERSION,
            }
            if bucket_layout is not None:
                row["layout"] = bucket_layout.to_dict()
            if layout_ocr_meta is not None:
                row["layout_ocr"] = layout_ocr_meta

            row_errors: list[str] = []
            for field in ("caption_original", "long", "medium", "short"):
                value = caption_original if field == "caption_original" else row["caption"][field]
                if not isinstance(value, str) or not value.strip():
                    row_errors.append(f"{field} is empty")
            for kind, text in (("original", caption_original),
                               ("long", long_text), ("medium", medium_text), ("short", short_text)):
                if missing_prefix_labels(text, trigger, artists, characters,
                                         unknown_if_missing=bool(artists)):
                    row_errors.append(f"{kind} is missing the deterministic prefix")
            if bucket_layout is not None:
                row_errors.extend(_layout_item_bounds_ok(bucket_layout, bucket))
                layout_items += len(bucket_layout.items)
            lengths = {
                "original": token_length(caption_original, tokenizer),
                "long": token_length(long_text, tokenizer),
                "medium": token_length(medium_text, tokenizer),
                "short": token_length(short_text, tokenizer),
            }
            if bucket_layout is not None:
                lengths["layout"] = token_length(layout_json(bucket_layout), tokenizer)
            for kind, length in lengths.items():
                token_samples.setdefault(kind, []).append(length)
            try:
                enforce_variant_token_budgets({
                    "original": caption_original, "long": long_text,
                    "medium": medium_text, "short": short_text,
                    **({"layout": layout_json(bucket_layout)} if bucket_layout is not None else {}),
                }, tokenizer=tokenizer)
            except Exception as error:
                row_errors.append(str(error))
            if any(length > 4096 for length in lengths.values()):
                row_errors.append("a variant exceeds the 4096-token model ceiling")
            for kind in variant_counts:
                if kind == "layout":
                    variant_counts[kind] += int(bucket_layout is not None)
                else:
                    variant_counts[kind] += 1
            if vlm_record is None:
                statuses["fallback_missing"] = statuses.get("fallback_missing", 0) + 1
            else:
                status = str(vlm_record.get("status", "invalid"))
                statuses[status] = statuses.get(status, 0) + 1
            if row_errors:
                errors.append({"file_name": out_name, "errors": row_errors})
            row["_validation_errors"] = row_errors
            rows.append(row)

            raw_payload = {
                "image_sha256": image_sha,
                "image_path": str(source_path),
                "model": vlm_record.get("model") if vlm_record else None,
                "prompt_version": vlm_record.get("prompt_version") if vlm_record else None,
                "source_width": width,
                "source_height": height,
                "caption_raw": annotation.caption.as_dict() if annotation else None,
                "watermark_raw": [mark.to_dict() for mark in annotation.watermark] if annotation else [],
                "borders_raw": annotation.borders.to_list() if annotation else ["none"],
                "layout_raw": raw_layout.to_dict() if raw_layout is not None else None,
                "vlm_layout_raw": annotation.layout.to_dict() if annotation is not None else None,
                "ocr": layout_ocr_meta,
                "validation": {
                    "status": "ok" if not row_errors else "invalid",
                    "errors": row_errors,
                    "attempts": vlm_record.get("attempts") if vlm_record else 0,
                    "warnings": list(vlm_record.get("warnings") or []) if vlm_record else [],
                },
                "vlm_text_items_dropped": action_meta.get("vlm_text_items_dropped", 0),
                "vlm_objects_included": action_meta.get("vlm_objects_included", 0),
            }
            atomic_write_json(output / RAW_DIR / f"{image_sha}.json", raw_payload)
        except Exception as error:
            errors.append({"row": index + 1, "errors": [f"{type(error).__name__}: {error}"]})

    clean_rows = []
    for row in rows:
        clean_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
    write_jsonl(output / METADATA_NAME, clean_rows)

    def percentile(values: Sequence[int], fraction: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return int(ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))])

    report = {
        "finalize_version": FINALIZE_VERSION,
        "base": str(base_path),
        "vlm": str(vlm_path) if vlm_path is not None else None,
        "detectors": str(detectors) if detectors is not None else None,
        "output": str(output),
        "rows": len(clean_rows),
        "layout_items": layout_items,
        "variant_counts": variant_counts,
        "variant_weights": {
            "original": round(original_weight * (1.0 - layout_weight), 6),
            "long": round((1.0 - layout_weight) * (1.0 - original_weight) / 3.0, 6),
            "medium": round((1.0 - layout_weight) * (1.0 - original_weight) / 3.0, 6),
            "short": round((1.0 - layout_weight) * (1.0 - original_weight) / 3.0, 6),
            "layout": round(layout_weight, 6),
        },
        "layout_mode": "separate",
        "vlm_statuses": statuses,
        "watermark_actions": actions,
        "borders": borders_seen,
        "vlm_text_items_dropped": sum(
            int((row.get("annotation") or {}).get("vlm_text_items_dropped", 0)) for row in clean_rows),
        "vlm_objects_included": sum(
            int((row.get("annotation") or {}).get("vlm_objects_included", 0)) for row in clean_rows),
        "detector_items": detector_items_included,
        "detector_items_dropped_cap": detector_items_capped,
        "text_items_dropped_cap": text_items_capped,
        "vlm_objects_covered_dropped": vlm_objects_covered,
        "token_lengths": {
            kind: {"count": len(values), "p50": percentile(values, 0.5),
                   "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
                   "max": max(values) if values else 0}
            for kind, values in token_samples.items()
        },
        "errors": errors,
        "deviations": [
            "Final layout text items come from PaddleOCR; VLM text items are dropped for audit only.",
            "VLM object/character items are omitted unless --vlm-objects audit-passed.",
            "Border cropping is recorded, not performed; boxes remain on the original canvas.",
            "Watermark overlap/exclusion is recorded; no generative inpainting is used.",
        ],
    }
    atomic_write_json(output / REPORT_NAME, report)
    _write_markdown_report(output / REPORT_MD, report)
    if errors and strict:
        raise FinalizeError(f"{len(errors)} row(s) failed finalize validation", report)
    return report


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# v2 finalize report",
        "",
        f"- Rows: {report['rows']}",
        f"- Layout items: {report['layout_items']}",
        f"- VLM statuses: `{json.dumps(report['vlm_statuses'], sort_keys=True)}`",
        f"- Watermark actions: `{json.dumps(report['watermark_actions'], sort_keys=True)}`",
        f"- Borders: `{json.dumps(report['borders'], sort_keys=True)}`",
        f"- VLM text items dropped: {report['vlm_text_items_dropped']}",
        f"- VLM object items included: {report['vlm_objects_included']}",
        f"- Detector items included: {report['detector_items']}",
        f"- Detector items dropped by cap: {report['detector_items_dropped_cap']}",
        f"- Text items dropped by cap: {report['text_items_dropped_cap']}",
        f"- VLM objects covered by detector and dropped: {report['vlm_objects_covered_dropped']}",
        "",
        "## Variant coverage",
        "",
        "| variant | count |",
        "| --- | ---: |",
        *[f"| {kind} | {count} |" for kind, count in report["variant_counts"].items()],
        "",
        "## Token lengths",
        "",
        "| variant | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {kind} | {values['p50']} | {values['p95']} | {values['p99']} | {values['max']} |"
          for kind, values in report["token_lengths"].items()],
        "",
    ]
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{json.dumps(error, ensure_ascii=False)}`" for error in report["errors"])
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moss-lora-finalize",
        description="Merge base metadata + PaddleOCR + VLM annotations into v2 metadata.")
    parser.add_argument("--base", required=True, help="Staging/prepared metadata JSONL")
    parser.add_argument("--vlm", default=None, help="annotations_vlm.jsonl or its directory")
    parser.add_argument("--ocr", default=None, help="Optional OCR metadata JSONL")
    parser.add_argument("--detectors", default=None,
                        help="Optional deepghs detector JSONL keyed by image SHA/source/file_name")
    parser.add_argument("--images", default=None, help="Base image root (default: base parent)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trigger", default="@CH3OCH3")
    parser.add_argument("--target-area", type=int, default=1024 * 1024)
    parser.add_argument("--layout-weight", type=float, default=0.20)
    parser.add_argument("--original-weight", type=float, default=0.20)
    parser.add_argument("--max-tags", type=int, default=40)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--vlm-objects", default="omit", choices=["omit", "audit-passed"],
                        help="Include VLM character/object boxes only after the 200-image audit")
    parser.add_argument("--allow-ocr-layout", action="store_true",
                        help="Promote OCR-only layout when the VLM provided no layout")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        except Exception as error:
            print(f"warning: tokenizer unavailable ({error}); reporting estimates", file=sys.stderr)
    try:
        report = finalize_dataset(
            args.base, args.vlm, args.output, images_root=args.images, ocr=args.ocr,
            detectors=args.detectors, trigger=args.trigger, target_area=args.target_area,
            layout_weight=args.layout_weight, original_weight=args.original_weight,
            max_tags=args.max_tags, tokenizer=tokenizer,
            include_vlm_objects=args.vlm_objects == "audit-passed",
            allow_ocr_layout=args.allow_ocr_layout, limit=args.limit,
            strict=not args.allow_invalid)
    except FinalizeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({key: report[key] for key in ("rows", "layout_items", "variant_counts")}),
          file=sys.stderr)


if __name__ == "__main__":
    main()
