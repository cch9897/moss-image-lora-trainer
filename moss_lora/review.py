"""Local bbox/text review server for MOSS layout datasets.

OCR is only a prefill.  This module serves a small, local-only FastAPI page where a
human can inspect the exact bucket-cropped training image, add/move/resize/delete
text boxes, correct the transcription, and mark images that contain no text.  It
never loads a model, GPU, or OCR backend.

Geometry is deliberately shared with training: images are cropped by
:func:`moss_lora.cache.fit_image` using :func:`build_buckets` /
:func:`assign_bucket`, and boxes are stored as integer ``yxyx`` /
``abs_centralized`` bucket coordinates.  The layout mirrors
``scripts/prepare_captions.py`` in its default ``both`` policy: the layout is
written to ``metadata.layout`` and also to ``caption.long`` so the cache trains
it, while ``caption.medium``/``caption.short`` stay prose.  Saving "no text"
removes ``layout`` and restores the prose long caption.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image

from moss_lora.cache import IMAGE_EXTS, assign_bucket, build_buckets, fit_image
from moss_lora.ocr import BucketGeometry, bucket_geometry, layout_json
from moss_lora.utils import atomic_write_bytes, atomic_write_json

METADATA_REVIEWED_NAME = "metadata_reviewed.jsonl"
STATE_NAME = "review_state.json"
REPORT_NAME = "review_report.md"
REVIEW_HTML = Path(__file__).with_name("review.html")
STATE_VERSION = 1
DEFAULT_TARGET_AREA = 1_048_576
MAX_BOXES = 1000
MAX_TEXT_LENGTH = 4000
COMPLETED_STATUSES = frozenset({"edited", "reviewed", "no_text"})
TEXT_REVIEW_EXTRA_HINT = (
    "Layout review needs optional dependencies. Install them with: "
    "pip install -e '.[review]'"
)

# Prose/tag signals that an image is worth reviewing even when OCR found nothing.
TEXT_HINT_RE = re.compile(
    r"\b(?:text|speech\s+bubbles?|signs?|writing|written|lettering|logo|logos|"
    r"watermarks?|words?|quotes?|quoted|dialogue|subtitles?|captions?|kanji|kana|"
    r"english\s+text|chinese\s+text|japanese\s+text)\b",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r'"[^"\n]{1,400}"|“[^”\n]{1,400}”|「[^」\n]{1,400}」|『[^』\n]{1,400}』')
NO_STORE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache", "Expires": "0"}


class ReviewError(ValueError):
    """Invalid metadata, path, bbox, or save request."""


# --------------------------------------------------------------------------- geometry


def bbox_to_bucket_rect(bbox: Sequence[int], bucket: tuple[int, int]
                        ) -> tuple[float, float, float, float]:
    """Official ``yxyx`` abs_centralized bbox -> ``x, y, w, h`` bucket pixels."""
    y1, x1, y2, x2 = (int(value) for value in bbox)
    bucket_w, bucket_h = bucket
    return (x1 + bucket_w / 2.0, y1 + bucket_h / 2.0, x2 - x1, y2 - y1)


def bucket_rect_to_bbox(x: float, y: float, w: float, h: float,
                        bucket: tuple[int, int]) -> list[int]:
    """``x, y, w, h`` bucket pixels -> integer ``yxyx`` abs_centralized bbox.

    Edges are rounded before subtracting the half-canvas offset, matching the
    rounding used by ``moss_lora.ocr.transform_box``.
    """
    bucket_w, bucket_h = bucket
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + max(0.0, w)))
    y2 = int(round(y + max(0.0, h)))
    half_w = bucket_w // 2
    half_h = bucket_h // 2
    return [y1 - half_h, x1 - half_w, y2 - half_h, x2 - half_w]


def coerce_bbox(bbox: Any, bucket: tuple[int, int], *, strict: bool = True) -> list[int]:
    """Validate and normalize one bbox; raise :class:`ReviewError` when invalid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ReviewError("bbox must be a list of four integers [y1, x1, y2, x2].")
    values: list[int] = []
    for value in bbox:
        if isinstance(value, bool):
            raise ReviewError("bbox coordinates must be integers.")
        if isinstance(value, int):
            values.append(value)
            continue
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            values.append(int(value))
            continue
        if not strict and isinstance(value, str):
            try:
                values.append(int(value.strip()))
                continue
            except ValueError:
                pass
        raise ReviewError("bbox coordinates must be integers.")
    y1, x1, y2, x2 = values
    bucket_w, bucket_h = bucket
    half_w = bucket_w // 2
    half_h = bucket_h // 2
    max_x = bucket_w - half_w
    max_y = bucket_h - half_h
    if not (-half_h <= y1 <= max_y and -half_h <= y2 <= max_y):
        raise ReviewError("bbox y coordinates are outside the bucket.")
    if not (-half_w <= x1 <= max_x and -half_w <= x2 <= max_x):
        raise ReviewError("bbox x coordinates are outside the bucket.")
    if y2 <= y1 or x2 <= x1:
        raise ReviewError("bbox must have positive width and height (y1 < y2, x1 < x2).")
    return [y1, x1, y2, x2]


# --------------------------------------------------------------------------- metadata


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewError(f"cannot read metadata {path}: {exc}") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ReviewError(f"{path}:{number}: each line must be a JSON object.")
        rows.append(row)
    if not rows:
        raise ReviewError(f"metadata is empty: {path}")
    return rows


def _parse_layout_text(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value if ("items" in value or "summary" in value) else None
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and ("items" in parsed or "summary" in parsed):
            return parsed
    return None


def layout_of(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the layout currently stored on a row (field or long-caption variant)."""
    layout = _parse_layout_text(row.get("layout"))
    if layout is not None:
        return layout
    caption = row.get("caption")
    if isinstance(caption, dict):
        return _parse_layout_text(caption.get("long"))
    if isinstance(caption, str):
        return _parse_layout_text(caption)
    return None


def layout_items(layout: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(layout, dict):
        return []
    items = layout.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _summary_caption(layout: dict[str, Any] | None) -> str:
    if not isinstance(layout, dict):
        return ""
    summary = layout.get("summary")
    if not isinstance(summary, dict):
        return ""
    caption = summary.get("caption")
    return caption.strip() if isinstance(caption, str) else ""


def _caption_variants(row: dict[str, Any]) -> dict[str, Any]:
    caption = row.get("caption")
    variants: dict[str, Any] = dict(caption) if isinstance(caption, dict) else {}
    if not variants and isinstance(caption, str):
        variants["long"] = caption
    return variants


def prose_long(row: dict[str, Any]) -> str:
    """Long prose caption, unwrapping a layout JSON stored in ``caption.long``."""
    variants = _caption_variants(row)
    long_value = variants.get("long")
    parsed = _parse_layout_text(long_value)
    if parsed is not None:
        text = _summary_caption(parsed)
        if text:
            return text
    if isinstance(long_value, str) and long_value.strip():
        return long_value.strip()
    for key in ("medium", "short"):
        value = variants.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = _parse_layout_text(row.get("layout"))
    text = _summary_caption(nested)
    if text:
        return text
    for key in ("text", "prompt", "image_description", "caption_no_artist"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tags(row: dict[str, Any]) -> list[str]:
    tags = row.get("tags")
    if isinstance(tags, list):
        return [str(tag) for tag in tags if isinstance(tag, (str, int, float))]
    return []


def _characters(row: dict[str, Any]) -> list[str]:
    value = row.get("character")
    if isinstance(value, list):
        return [str(character) for character in value if isinstance(character, (str, int, float))]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _artist(row: dict[str, Any]) -> str:
    value = row.get("artist")
    return value.strip() if isinstance(value, str) else ""


def _artists(row: dict[str, Any]) -> list[str]:
    values = row.get("artists")
    if isinstance(values, list):
        return [str(value).strip() for value in values
                if isinstance(value, (str, int, float)) and str(value).strip()]
    artist = _artist(row)
    return [artist] if artist else []


def _tag_text(row: dict[str, Any]) -> str:
    return ", ".join(_tags(row))


def _hint_strings(row: dict[str, Any]) -> list[str]:
    variants = _caption_variants(row)
    values: list[str] = [prose_long(row)]
    for key in ("medium", "short"):
        value = variants.get(key)
        if isinstance(value, str):
            values.append(value)
    no_artist = row.get("caption_no_artist")
    if isinstance(no_artist, str):
        values.append(no_artist)
    values.append(_tag_text(row))
    return [value for value in values if value.strip()]


def text_hint(row: dict[str, Any]) -> bool:
    """True when prose or tags mention visible text, lettering, or quotes."""
    for value in _hint_strings(row):
        if TEXT_HINT_RE.search(value) or QUOTE_RE.search(value):
            return True
    return False


def _quoted_suggestions(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for value in _hint_strings(row):
        for match in QUOTE_RE.finditer(value):
            text = match.group(0).strip()
            if len(text) >= 2 and text[0] in '"“「『':
                text = text[1:-1].strip()
            if text and text not in texts:
                texts.append(text)
    return texts


def _layout_meta(layout: dict[str, Any] | None) -> tuple[list[str], str]:
    summary = layout.get("summary") if isinstance(layout, dict) else None
    if not isinstance(summary, dict):
        return ["anime"], "single"
    meta = summary.get("meta")
    meta_list = [str(value) for value in meta] if isinstance(meta, list) and meta else ["anime"]
    canvas = summary.get("canvas")
    return meta_list, canvas.strip() if isinstance(canvas, str) and canvas.strip() else "single"


def _ensure_summary(summary: str, prose: str, row: dict[str, Any]) -> str:
    """Keep trigger/artist/character conditioning in ``summary.caption``."""
    text = summary.strip() or prose.strip()
    additions: list[str] = []
    for artist in _artists(row):
        if artist and f"artist:{artist}" not in text:
            additions.append(f"artist:{artist}")
    for character in _characters(row):
        if character and f"character:{character}" not in text:
            additions.append(f"character:{character}")
    trigger = prose.split(",", 1)[0].strip()
    if (trigger and len(trigger) <= 48 and " " not in trigger
            and ":" not in trigger and trigger not in text):
        additions.append(trigger)
    if not text:
        tags = _tags(row)
        text = ", ".join(additions + tags) if (additions or tags) else "anime"
    elif additions:
        text = ", ".join(additions + [text])
    return text


def _row_box_records(layout: dict[str, Any] | None, bucket: tuple[int, int]
                     ) -> list[dict[str, Any]]:
    """Layout items with label/caption/text preserved (review and cache contract)."""
    records: list[dict[str, Any]] = []
    for item in layout_items(layout):
        label = item.get("label") if item.get("label") in ("character", "object", "text") else "text"
        text = item.get("text") if isinstance(item.get("text"), str) else ""
        caption = item.get("caption") if isinstance(item.get("caption"), str) else ""
        if label == "text":
            caption = caption.strip() or "visible text"
            text = text.strip()
        else:
            caption = caption.strip() or text.strip() or label
            text = ""
        if label == "text" and not text:
            continue
        try:
            bbox = coerce_bbox(item.get("bbox"), bucket)
        except ReviewError:
            continue
        records.append({"bbox": bbox, "label": label, "text": text, "caption": caption})
    return records


def _row_boxes(layout: dict[str, Any] | None, bucket: tuple[int, int]
               ) -> list[tuple[list[int], str]]:
    """Backward-compatible (bbox, text) view for callers/tests."""
    return [(record["bbox"], record["text"] or record["caption"])
            for record in _row_box_records(layout, bucket)]


def build_layout(row: dict[str, Any], boxes: Sequence[Any], prose: str) -> dict[str, Any]:
    current = layout_of(row)
    meta, canvas = _layout_meta(current)
    summary = _summary_caption(current) or prose
    items: list[dict[str, Any]] = []
    for entry in boxes:
        if isinstance(entry, Mapping):
            bbox = entry.get("bbox")
            label = entry.get("label", "text")
            if label not in ("character", "object", "text"):
                label = "text"
            caption = entry.get("caption") if isinstance(entry.get("caption"), str) else ""
            text = entry.get("text") if isinstance(entry.get("text"), str) else ""
        else:
            bbox, value = entry
            label = "text"
            caption = "visible text"
            text = str(value)
        if label == "text":
            item = {"label": "text", "caption": caption.strip() or "visible text",
                    "text": text.strip(), "bbox": list(bbox)}
        else:
            item = {"label": label, "caption": caption.strip() or str(label),
                    "bbox": list(bbox)}
        items.append(item)
    return {
        "summary": {
            "meta": meta,
            "canvas": canvas,
            "caption": _ensure_summary(summary, prose, row),
        },
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


# --------------------------------------------------------------------------- filesystem


def _safe_relative(file_name: str) -> PurePosixPath:
    if not isinstance(file_name, str) or not file_name.strip():
        raise ReviewError("file_name is empty.")
    if "\x00" in file_name or "\\" in file_name:
        raise ReviewError("file_name contains forbidden characters.")
    relative = PurePosixPath(file_name.strip())
    if relative.is_absolute():
        raise ReviewError("file_name must be relative.")
    if any(part in ("", ".", "..") for part in relative.parts):
        raise ReviewError("file_name must not contain '.', '..', or empty path parts.")
    return relative


def resolve_image_path(images_root: str | os.PathLike[str], metadata_dir: str | os.PathLike[str],
                       file_name: str) -> Path:
    """Resolve a metadata image name under ``images_root`` or raise.

    ``metadata_dir / file_name`` is preferred so a dataset root that contains both
    ``metadata.jsonl`` and ``images/`` works.  A basename fallback supports
    ``--images`` pointing directly at an image folder.  The candidate must be
    lexically inside ``images_root`` and have a supported image suffix.  Symlinks
    are followed (the sample dataset uses them) but the browser never supplies a
    path: image routes are index-based.  No path outside the root is constructed
    from user input.
    """
    relative = _safe_relative(file_name)
    root = Path(images_root).resolve()
    metadata = Path(metadata_dir).resolve()
    if relative.suffix.lower() not in IMAGE_EXTS:
        raise ReviewError(f"unsupported image suffix: {relative.suffix or '(none)'}")

    candidates = [metadata / relative, root / relative]
    if str(relative) != relative.name:
        candidates.append(root / relative.name)
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    raise ReviewError(f"image not found under {root}: {file_name}")


def _source_size(path: Path) -> tuple[int, int]:
    """EXIF-transposed source size without decoding the full image."""
    with Image.open(path) as image:
        width, height = image.size
        try:
            orientation = image.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return width, height


def _image_version(path: Path, bucket: tuple[int, int]) -> str:
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return f"{mtime}-{bucket[0]}x{bucket[1]}"


# --------------------------------------------------------------------------- store


@dataclass
class ItemRecord:
    index: int
    row: dict[str, Any]
    path: Path
    file_name: str
    prose: str
    original_layout: dict[str, Any] | None
    source_w: int | None = None
    source_h: int | None = None
    bucket: tuple[int, int] | None = None
    geometry: BucketGeometry | None = None

    def require_geometry(self) -> BucketGeometry:
        if self.geometry is None or self.bucket is None:
            raise ReviewError(f"geometry was not loaded for {self.file_name}")
        return self.geometry

    @property
    def crop_pct(self) -> float:
        geometry = self.require_geometry()
        area = geometry.resize_w * geometry.resize_h
        if area <= 0:
            return 0.0
        assert self.bucket is not None
        return round(100.0 * (1.0 - (self.bucket[0] * self.bucket[1]) / area), 2)

    @property
    def version(self) -> str:
        assert self.bucket is not None
        return _image_version(self.path, self.bucket)


class ReviewStore:
    """Metadata rows, bucket geometry, edits, and atomic export for one dataset."""

    def __init__(self, metadata: str | os.PathLike[str], images: str | os.PathLike[str],
                 output: str | os.PathLike[str], *, target_area: int = DEFAULT_TARGET_AREA,
                 fresh: bool = False, layout_policy: str = "legacy") -> None:
        self.metadata_path = Path(metadata).resolve()
        self.images_root = Path(images).resolve()
        self.output_dir = Path(output).resolve()
        if layout_policy not in ("legacy", "separate"):
            raise ReviewError("layout_policy must be 'legacy' or 'separate'.")
        self.layout_policy = layout_policy
        if not self.metadata_path.is_file():
            raise ReviewError(f"metadata not found: {self.metadata_path}")
        if not self.images_root.is_dir():
            raise ReviewError(f"image root is not a directory: {self.images_root}")
        if not isinstance(target_area, int) or isinstance(target_area, bool) or target_area < 65536:
            raise ReviewError("target-area must be an integer >= 65536.")
        self.target_area = target_area
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buckets = build_buckets(target_area)
        self.reviewed_path = self.output_dir / METADATA_REVIEWED_NAME
        self.state_path = self.output_dir / STATE_NAME
        self.report_path = self.output_dir / REPORT_NAME
        self.lock = threading.RLock()
        self.skipped: list[str] = []

        resumed = not fresh and self.reviewed_path.is_file()
        load_path = self.reviewed_path if resumed else self.metadata_path
        self.loaded_from_reviewed = resumed
        self.rows = _read_jsonl(load_path)
        self.records: dict[int, ItemRecord] = {}
        for index, row in enumerate(self.rows):
            name = row.get("file_name")
            if not isinstance(name, str):
                self.skipped.append(f"row {index + 1}: missing file_name")
                continue
            try:
                path = resolve_image_path(self.images_root, self.metadata_path.parent, name)
            except ReviewError as exc:
                self.skipped.append(f"{name}: {exc}")
                continue
            self.records[index] = ItemRecord(
                index=index,
                row=row,
                path=path,
                file_name=name,
                prose=prose_long(row),
                original_layout=copy.deepcopy(layout_of(row)),
            )
        if not self.records:
            detail = "; ".join(self.skipped[:5]) or "no supported images"
            raise ReviewError(f"no usable images found ({detail})")
        self.state = self._load_state()

    # -- state -------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"version": STATE_VERSION, "items": {}}
        if not self.state_path.is_file():
            return state
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(loaded, dict) or not isinstance(loaded.get("items"), dict):
            return state
        items: dict[str, Any] = {}
        for key, entry in loaded["items"].items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status")
            if status not in COMPLETED_STATUSES:
                continue
            index = int(key) if str(key).isdigit() else None
            if index is None or index not in self.records:
                continue
            record = self.records[index]
            if entry.get("file_name") not in (None, record.file_name):
                continue
            items[str(index)] = {
                "file_name": record.file_name,
                "status": status,
                "boxes": max(0, int(entry.get("boxes", 0) or 0)),
                "no_text": status == "no_text",
                "updated_at": entry.get("updated_at"),
            }
        state["items"] = items
        return state

    def _status(self, index: int) -> dict[str, Any]:
        return self.state["items"].get(str(index), {})

    def record(self, index: int) -> ItemRecord:
        try:
            return self.records[int(index)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"item {index} not found") from exc

    def ensure_geometry(self, record: ItemRecord) -> ItemRecord:
        """Load source dimensions and bucket geometry on first use."""
        if record.geometry is not None:
            return record
        try:
            source_w, source_h = _source_size(record.path)
        except OSError as exc:
            raise ReviewError(f"cannot read image {record.file_name}: {exc}") from exc
        bucket = assign_bucket(source_w, source_h, self.buckets)
        record.source_w = source_w
        record.source_h = source_h
        record.bucket = bucket
        record.geometry = bucket_geometry((source_w, source_h), bucket)
        return record

    # -- queue / summaries -------------------------------------------------

    def queue_info(self, record: ItemRecord) -> dict[str, Any]:
        layout = layout_of(record.row)
        items = layout_items(layout)
        hint = text_hint(record.row)
        if layout is not None:
            reason = "layout" if items else "empty_layout"
        elif hint:
            reason = "text_no_layout"
        else:
            reason = None
        return {
            "in_queue": reason is not None,
            "reason": reason,
            "text_hint": hint,
            "has_layout": layout is not None,
            "layout_items": len(items),
        }

    def _summary_item(self, record: ItemRecord) -> dict[str, Any]:
        queue = self.queue_info(record)
        status = self._status(record.index)
        tags = _tags(record.row)
        characters = _characters(record.row)
        artist = _artist(record.row)
        variants = _caption_variants(record.row)
        no_artist = record.row.get("caption_no_artist")
        keywords = " ".join(
            [
                record.file_name,
                artist,
                *characters,
                *tags,
                record.prose[:600],
                str(variants.get("medium") or "")[:300],
                str(variants.get("short") or "")[:300],
                no_artist[:300] if isinstance(no_artist, str) else "",
            ]
        ).lower()
        return {
            "index": record.index,
            "file_name": record.file_name,
            "artist": artist,
            "artists": _artists(record.row),
            "characters": characters,
            "caption_excerpt": record.prose[:280],
            "keywords": keywords,
            "status": status.get("status", "pending"),
            "saved_at": status.get("updated_at"),
            **queue,
        }

    def counts(self) -> dict[str, Any]:
        queue = 0
        remaining = 0
        completed = 0
        edited = 0
        reviewed = 0
        no_text = 0
        layouts = 0
        empty_layouts = 0
        text_no_layout = 0
        boxes = 0
        for record in self.records.values():
            info = self.queue_info(record)
            status = self._status(record.index).get("status", "pending")
            if info["in_queue"]:
                queue += 1
                if status not in COMPLETED_STATUSES:
                    remaining += 1
                else:
                    completed += 1
            if status == "edited":
                edited += 1
            elif status == "reviewed":
                reviewed += 1
            elif status == "no_text":
                no_text += 1
            if info["reason"] == "layout":
                layouts += 1
            elif info["reason"] == "empty_layout":
                empty_layouts += 1
            elif info["reason"] == "text_no_layout":
                text_no_layout += 1
            boxes += info["layout_items"]
        return {
            "total": len(self.records),
            "queue": queue,
            "remaining_queue": remaining,
            "completed": edited + reviewed + no_text,
            "completed_queue": completed,
            "edited": edited,
            "reviewed": reviewed,
            "no_text": no_text,
            "layouts": layouts,
            "empty_layouts": empty_layouts,
            "text_no_layout": text_no_layout,
            "boxes": boxes,
            "with_layout": layouts + empty_layouts,
            "target_area": self.target_area,
            "bucket_count": len(self.buckets),
            "skipped": len(self.skipped),
        }

    def list_items(self, filter_name: str = "text", query: str | None = None) -> dict[str, Any]:
        if filter_name not in ("text", "all"):
            raise ReviewError("filter must be 'text' or 'all'.")
        needle = (query or "").strip().lower()
        items: list[dict[str, Any]] = []
        for index in sorted(self.records):
            item = self._summary_item(self.records[index])
            if filter_name == "text" and not item["in_queue"]:
                continue
            if needle and needle not in item["keywords"]:
                continue
            items.append(item)
        counts = self.counts()
        counts["filtered"] = len(items)
        return {
            "filter": filter_name,
            "query": needle,
            "source_metadata": self.metadata_path.name,
            "source_metadata_path": str(self.metadata_path),
            "resumed_from_reviewed": self.loaded_from_reviewed,
            "output_dir": str(self.output_dir),
            "counts": counts,
            "items": items,
        }

    def detail(self, index: int) -> dict[str, Any]:
        record = self.ensure_geometry(self.record(index))
        layout = layout_of(record.row)
        boxes: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for box_record in _row_box_records(layout, record.bucket):
            label = box_record["label"]
            text = box_record["text"] or box_record["caption"]
            box = {"bbox": box_record["bbox"], "text": text}
            if label != "text":
                box.update({"label": label, "caption": box_record["caption"], "text": ""})
            boxes.append(box)
            if text and text not in seen_text:
                seen_text.add(text)
                suggestions.append({"text": text, "source": "ocr" if label == "text" else label})
        for item in layout_items(layout):
            text = item.get("text")
            if isinstance(text, str) and text.strip() and text.strip() not in seen_text:
                seen_text.add(text.strip())
                suggestions.append({"text": text.strip(), "source": "ocr"})
        for text in _quoted_suggestions(record.row):
            if text not in seen_text:
                seen_text.add(text)
                suggestions.append({"text": text, "source": "caption"})
        used = {box["text"] for box in boxes if box["text"]}
        for suggestion in suggestions:
            suggestion["used"] = suggestion["text"] in used
        variants = _caption_variants(record.row)
        status = self._status(index)
        return {
            "index": record.index,
            "file_name": record.file_name,
            "artist": _artist(record.row),
            "artists": _artists(record.row),
            "characters": _characters(record.row),
            "caption_original": record.row.get("caption_original")
            if isinstance(record.row.get("caption_original"), str) else "",
            "layout_ocr": record.row.get("layout_ocr")
            if isinstance(record.row.get("layout_ocr"), dict) else None,
            "source_handle": record.row.get("source_handle")
            if isinstance(record.row.get("source_handle"), str) else "",
            "tags": _tags(record.row),
            "caption": {
                "prose": record.prose,
                "long": variants.get("long") if isinstance(variants.get("long"), str) else "",
                "medium": variants.get("medium") if isinstance(variants.get("medium"), str) else "",
                "short": variants.get("short") if isinstance(variants.get("short"), str) else "",
                "no_artist": record.row.get("caption_no_artist")
                if isinstance(record.row.get("caption_no_artist"), str) else "",
            },
            "queue": self.queue_info(record),
            "layout": layout,
            "boxes": boxes,
            "suggestions": suggestions,
            "invalid_boxes": max(0, len(layout_items(layout)) - len(boxes)),
            "bucket": {
                "width": record.bucket[0],
                "height": record.bucket[1],
                "source_width": record.source_w,
                "source_height": record.source_h,
                "resize_width": record.geometry.resize_w,
                "resize_height": record.geometry.resize_h,
                "crop_left": record.geometry.left,
                "crop_top": record.geometry.top,
                "crop_pct": record.crop_pct,
            },
            "status": status.get("status", "pending"),
            "no_text": status.get("status") == "no_text",
            "saved_at": status.get("updated_at"),
            "image_url": f"/api/image/{record.index}?v={record.version}",
            "layout_policy": self.layout_policy,
        }

    # -- images ------------------------------------------------------------

    def image_bytes(self, index: int) -> tuple[bytes, str]:
        record = self.ensure_geometry(self.record(index))
        try:
            with Image.open(record.path) as image:
                fitted = fit_image(image, record.bucket)
        except OSError as exc:
            raise ReviewError(f"cannot decode {record.file_name}: {exc}") from exc
        buffer = BytesIO()
        fitted.save(buffer, format="JPEG", quality=92, subsampling=0, optimize=True)
        return buffer.getvalue(), "image/jpeg"

    # -- persistence -------------------------------------------------------

    def _write_metadata(self) -> None:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in self.rows
        )
        atomic_write_bytes(self.reviewed_path, payload.encode("utf-8"))

    def _write_state(self) -> None:
        counts = self.counts()
        payload = {
            "version": STATE_VERSION,
            "metadata": self.metadata_path.name,
            "updated_at": _utc_now(),
            "counts": counts,
            "items": self.state["items"],
        }
        atomic_write_json(self.state_path, payload)

    def _write_report(self) -> None:
        counts = self.counts()
        lines = [
            "# Layout review report",
            "",
            f"- Source metadata: `{self.metadata_path}`",
            f"- Reviewed metadata: `{self.reviewed_path}`",
            f"- Image root: `{self.images_root}`",
            f"- Target area: `{self.target_area}` px²; buckets: {counts['bucket_count']}",
            f"- Updated: {_utc_now()}",
            "",
            "## Progress",
            "",
            "| metric | count |",
            "| --- | ---: |",
            f"| usable items | {counts['total']} |",
            f"| review queue | {counts['queue']} |",
            f"| remaining queue | {counts['remaining_queue']} |",
            f"| completed queue items | {counts['completed_queue']} |",
            f"| edited items | {counts['edited']} |",
            f"| reviewed unchanged | {counts['reviewed']} |",
            f"| marked no text | {counts['no_text']} |",
            f"| text hints without layout | {counts['text_no_layout']} |",
            f"| items with existing layout | {counts['with_layout']} |",
            f"| empty layouts | {counts['empty_layouts']} |",
            f"| text boxes | {counts['boxes']} |",
            "",
            "OCR is a prefill; boxes and transcriptions are expected to be corrected.",
            "",
        ]
        atomic_write_bytes(self.report_path, "\n".join(lines).encode("utf-8"))

    def export(self) -> dict[str, Any]:
        with self.lock:
            self._write_metadata()
            self._write_state()
            self._write_report()
            return {
                "metadata": str(self.reviewed_path),
                "state": str(self.state_path),
                "report": str(self.report_path),
                "counts": self.counts(),
            }

    # -- save --------------------------------------------------------------

    def _normalize_save_boxes(self, raw_boxes: Any, bucket: tuple[int, int]
                              ) -> list[dict[str, Any]]:
        if not isinstance(raw_boxes, list):
            raise ReviewError("boxes must be a list.")
        if len(raw_boxes) > MAX_BOXES:
            raise ReviewError(f"too many boxes (limit {MAX_BOXES}).")
        boxes: list[dict[str, Any]] = []
        for number, entry in enumerate(raw_boxes, start=1):
            if not isinstance(entry, dict):
                raise ReviewError(f"box {number} must be an object.")
            label = entry.get("label", "text")
            if label not in ("character", "object", "text"):
                raise ReviewError(f"box {number} label must be character, object, or text.")
            value = entry.get("text") if label == "text" else entry.get("caption", entry.get("text"))
            if not isinstance(value, str):
                raise ReviewError(f"box {number} has no {'text' if label == 'text' else 'caption'} string.")
            value = value.strip()
            if not value:
                raise ReviewError(f"box {number} has empty {'text' if label == 'text' else 'caption'}.")
            if len(value) > MAX_TEXT_LENGTH:
                raise ReviewError(f"box {number} text is too long (limit {MAX_TEXT_LENGTH}).")
            bbox = coerce_bbox(entry.get("bbox"), bucket)
            if label == "text":
                boxes.append({"bbox": bbox, "label": "text", "text": value,
                              "caption": str(entry.get("caption") or "visible text").strip()})
            else:
                boxes.append({"bbox": bbox, "label": label, "text": "", "caption": value})
        return boxes

    def _original_boxes(self, record: ItemRecord) -> list[dict[str, Any]]:
        return _row_box_records(record.original_layout, record.bucket)

    def save(self, index: int, raw_boxes: Any, no_text: bool = False) -> dict[str, Any]:
        record = self.ensure_geometry(self.record(index))
        if not isinstance(no_text, bool):
            raise ReviewError("no_text must be a boolean.")
        with self.lock:
            boxes = [] if no_text else self._normalize_save_boxes(raw_boxes, record.bucket)
            effective_no_text = bool(no_text) or not boxes
            row = record.row
            old_row = copy.deepcopy(row)
            old_entry = copy.deepcopy(self.state["items"].get(str(index)))
            variants = _caption_variants(row)
            prose = record.prose or prose_long(row)

            if effective_no_text:
                row.pop("layout", None)
                variants["long"] = prose
                status = "no_text"
                saved_boxes: list[dict[str, Any]] = []
            else:
                layout = build_layout(row, boxes, prose)
                row["layout"] = layout
                if self.layout_policy == "legacy":
                    variants["long"] = layout_json(layout)
                else:
                    variants["long"] = prose
                status = "reviewed" if boxes == self._original_boxes(record) else "edited"
                saved_boxes = boxes

            variants["long"] = variants.get("long") or prose
            if not isinstance(variants.get("medium"), str) or not variants["medium"].strip():
                variants["medium"] = prose
            if not isinstance(variants.get("short"), str) or not variants["short"].strip():
                variants["short"] = prose
            row["caption"] = variants
            timestamp = _utc_now()
            self.state["items"][str(index)] = {
                "file_name": record.file_name,
                "status": status,
                "boxes": len(saved_boxes),
                "no_text": effective_no_text,
                "updated_at": timestamp,
            }
            try:
                self._write_metadata()
                self._write_state()
                self._write_report()
            except OSError:
                row.clear()
                row.update(old_row)
                if old_entry is None:
                    self.state["items"].pop(str(index), None)
                else:
                    self.state["items"][str(index)] = old_entry
                raise
            detail = self.detail(index)
            detail["saved_at"] = timestamp
            detail["counts"] = self.counts()
            return detail


# --------------------------------------------------------------------------- FastAPI app


def create_app(store: ReviewStore, initial_filter: str = "text") -> Any:
    """Build the local FastAPI application around an already-loaded store."""
    if initial_filter not in ("text", "all"):
        raise ReviewError("initial filter must be 'text' or 'all'.")
    try:
        from fastapi import Body, FastAPI, HTTPException, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import HTMLResponse, JSONResponse, Response
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"{TEXT_REVIEW_EXTRA_HINT} ({exc})") from exc

    app = FastAPI(
        title="MOSS-Image layout review",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store

    @app.middleware("http")
    async def _no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(NO_STORE)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "Invalid request parameters."}, status_code=400)

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        try:
            html = REVIEW_HTML.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(500, f"review.html could not be read: {exc}") from exc
        bootstrap = json.dumps(
            {"filter": initial_filter, "metadata": store.metadata_path.name},
            ensure_ascii=True,
        ).replace("</", "<\\/")
        html, replaced = re.subn(
            r"/\*__BOOTSTRAP_START__\*/.*?/\*__BOOTSTRAP_END__\*/",
            bootstrap,
            html,
            count=1,
            flags=re.DOTALL,
        )
        if not replaced:
            raise HTTPException(500, "review.html bootstrap marker is missing")
        return HTMLResponse(html)

    @app.get("/api/items")
    def api_items(filter: str | None = None, q: str | None = None) -> dict[str, Any]:
        try:
            return store.list_items(filter or initial_filter, q)
        except ReviewError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/items/{index}")
    def api_item(index: int) -> dict[str, Any]:
        try:
            return store.detail(index)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/image/{index}")
    def api_image(index: int) -> Any:
        try:
            content, media_type = store.image_bytes(index)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=content, media_type=media_type)

    @app.post("/api/items/{index}/save")
    def api_save(index: int, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(400, "Request body must be a JSON object.")
        try:
            return store.save(
                index,
                payload.get("boxes", []),
                no_text=payload.get("no_text", False),
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ReviewError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f"save failed: {exc}") from exc

    @app.post("/api/export")
    def api_export() -> dict[str, Any]:
        try:
            return store.export()
        except OSError as exc:
            raise HTTPException(500, f"export failed: {exc}") from exc

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moss-lora-review",
        description="Local browser review of bucket-space OCR/layout boxes.",
    )
    parser.add_argument("--metadata", required=True,
                        help="metadata.jsonl to review (or a previous metadata_reviewed.jsonl)")
    parser.add_argument("--images", required=True,
                        help="Source dataset root or its images/ directory")
    parser.add_argument("--output", required=True,
                        help="Directory for metadata_reviewed.jsonl, review_state.json, and the report")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=7862, help="Port (default: 7862)")
    parser.add_argument("--filter", choices=("text", "all"), default="text",
                        help="Initial queue filter (default: text)")
    parser.add_argument("--target-area", type=int, default=DEFAULT_TARGET_AREA,
                        help="Training bucket target area in px² (default: 1048576)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore an existing metadata_reviewed.jsonl and start from --metadata")
    parser.add_argument("--layout-policy", choices=("legacy", "separate"), default="legacy",
                        help="legacy puts layout JSON in caption.long; separate keeps prose long "
                             "and writes the layout field only (v2)")
    parser.add_argument("--open", action="store_true", help="Open the review page in a browser")
    args = parser.parse_args(argv)

    try:
        store = ReviewStore(
            args.metadata, args.images, args.output,
            target_area=args.target_area, fresh=args.fresh,
            layout_policy=args.layout_policy,
        )
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        print(f"error: {TEXT_REVIEW_EXTRA_HINT} ({exc})", file=sys.stderr)
        return 1

    app = create_app(store, initial_filter=args.filter)
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}/"
    counts = store.counts()
    print(f"moss-lora-review: {store.metadata_path}")
    print(f"  images: {store.images_root}")
    print(f"  output: {store.output_dir}")
    print(f"  buckets: {counts['bucket_count']} (target area {args.target_area} px²)")
    print(f"  queue: {counts['remaining_queue']} remaining / {counts['queue']} total; "
          f"{counts['text_no_layout']} text hints without layout")
    if store.loaded_from_reviewed:
        print(f"  resumed from {store.reviewed_path}")
    for warning in store.skipped[:10]:
        print(f"  warning: skipped {warning}", file=sys.stderr)
    if len(store.skipped) > 10:
        print(f"  warning: {len(store.skipped) - 10} more skipped rows", file=sys.stderr)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("  warning: binding beyond loopback exposes review data; use a trusted network",
              file=sys.stderr)
    print(f"  {url}")
    print("  Ctrl+C to stop")

    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
