"""Annotation schema, coordinate transforms, deterministic prefixes and token budgets.

This module is the v2 contract shared by staging, finalize, cache and review:

- strict dataclasses for raw VLM captions, layout items, watermarks, borders and OCR text;
- normalized 0-1000 ``yxyx`` validation and conversion to the official bucket
  ``abs_centralized`` / ``yxyx`` space used by the cache;
- deterministic ``@CH3OCH3, artist:<id>, character:<id>`` captions that the VLM
  never writes;
- VLM ``artist:``/``character:`` leakage stripping and watermark text filtering;
- token-budget helpers for the five cached variants.

Raw VLM boxes are always ``[y1, x1, y2, x2]`` integers on a 0-1000 grid relative
to the full source image. Only this module converts them; nothing downstream
should re-implement the geometry.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from moss_lora.ocr import BucketGeometry, bucket_geometry

TRIGGER_DEFAULT = "@CH3OCH3"
UNKNOWN_ARTIST = "unknown"
MODEL_TOKEN_CEILING = 4096
LAYOUT_TOKEN_BUDGET = 1024
VARIANT_TOKEN_LIMITS: dict[str, int] = {
    "original": 1200,
    "long": 1500,
    "medium": 512,
    "short": 128,
    "layout": 1024,
}
MEDIUM_TAG_LIMIT = 25
SHORT_TAG_LIMIT = 12
LONG_TAG_LIMIT = 40
MAX_LAYOUT_ITEMS = 12
MAX_LAYOUT_TEXT_ITEMS = 8
MAX_SUMMARY_META = 3

LAYOUT_LABELS = ("character", "object", "text")
CANVAS_VALUES = ("single", "multi_panel", "collage")
RATING_VALUES = ("general", "sensitive", "questionable", "explicit")
BORDER_EDGES = ("left", "right", "up", "down")
BORDER_NONE = "none"
BOX_FORMAT = "yxyx"
NORMALIZED_SPACE = "normalized_1000"
BUCKET_SPACE = "abs_centralized"
MIN_BOX_AREA_RATIO = 0.0005
MAX_BOX_AREA_RATIO = 0.95
WATERMARK_IOU_THRESHOLD = 0.3

# Boilerplate negatives must not be reinforced as positive training tags.
NEGATIVE_TAGS = (
    "worst quality",
    "bad anatomy",
    "watermark",
    "jpeg artifacts",
    "bad hands",
    "extra fingers",
    "missing fingers",
)
NEGATIVE_TAG_KEYS = frozenset(tag.replace("_", " ") for tag in NEGATIVE_TAGS)

_METADATA_LABEL_RE = re.compile(r"(?i)\b(artist|character)\s*:\s*([^,;\n]*)")
_LABEL_PREFIX_RE = re.compile(r"(?i)^\s*(?:artist|character)\s*:\s*")
_ARTIST_TAG_RE = re.compile(r"(?i)^\s*artist\s*:\s*(.+?)\s*$")
_CHARACTER_TAG_RE = re.compile(r"(?i)^\s*character\s*:\s*(.+?)\s*$")


class AnnotationError(ValueError):
    """Raw annotation does not satisfy the v2 schema or coordinate contract."""


class TokenBudgetError(ValueError):
    """A caption variant exceeds its configured token budget."""


# --------------------------------------------------------------------------- text helpers


def normalize_name(value: str) -> str:
    """NFKC, full-width @ -> @, collapse whitespace and trim separators."""
    text = unicodedata.normalize("NFKC", str(value)).replace("\uff20", "@")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,;，；")


def normalize_tag_key(value: str) -> str:
    """Case/underscore/space-insensitive key used for dedupe and matching."""
    text = normalize_name(value).lower().replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def dedupe_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_name(value)
        key = normalize_tag_key(clean)
        if clean and key and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def dedupe_tags(values: Iterable[str]) -> list[str]:
    """Normalize/dedupe positive tags and drop boilerplate negatives."""
    seen: set[str] = set()
    tags: list[str] = []
    for value in values:
        clean = normalize_name(value)
        key = normalize_tag_key(clean)
        if not key or key in NEGATIVE_TAG_KEYS or key in seen:
            continue
        seen.add(key)
        tags.append(clean)
    return tags


def count_cjk(text: str) -> int:
    return sum(
        1
        for char in text
        if ("\u3400" <= char <= "\u9fff")
        or ("\uf900" <= char <= "\ufaff")
        or ("\U00020000" <= char <= "\U0002ffff")
    )


def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate: CJK characters + ASCII words, else ~4 chars/token."""
    if not text or not text.strip():
        return 0
    cjk = count_cjk(text)
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    if cjk:
        return max(1, cjk + words)
    return max(1, words, math.ceil(len(text) / 4))


def token_length(text: str, tokenizer: Any | None = None) -> int:
    """Token count using an optional Hugging Face tokenizer, else ``estimate_tokens``."""
    if tokenizer is None or not text:
        return 0 if not text else estimate_tokens(text)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return len(encode(text))
            except Exception:
                pass
        except Exception:
            pass
    return estimate_tokens(text)


def enforce_variant_token_budgets(
    variants: Mapping[str, str],
    *,
    tokenizer: Any | None = None,
    limits: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Return per-variant token counts, raising ``TokenBudgetError`` on overflow.

    ``TokenBudgetError`` carries ``.lengths``, ``.limits`` and ``.over`` so a
    finalize report can list every offending variant at once.
    """
    limits = dict(VARIANT_TOKEN_LIMITS if limits is None else limits)
    lengths: dict[str, int] = {}
    over: dict[str, tuple[int, int]] = {}
    for kind, text in variants.items():
        length = token_length(text, tokenizer)
        lengths[kind] = length
        limit = limits.get(kind)
        if limit is not None and length > limit:
            over[kind] = (length, limit)
    if over:
        detail = ", ".join(f"{kind}={length}>{limit}" for kind, (length, limit) in over.items())
        error = TokenBudgetError(f"Variant token budget exceeded: {detail}")
        error.lengths = lengths  # type: ignore[attr-defined]
        error.limits = limits  # type: ignore[attr-defined]
        error.over = over  # type: ignore[attr-defined]
        raise error
    return lengths


def truncate_to_token_budget(text: str, limit: int, tokenizer: Any | None = None) -> tuple[str, int]:
    """Explicitly truncate text to ``limit`` tokens; returns ``(text, length)``.

    Layout JSON must not be truncated silently; callers should only use this for
    prose variants after deciding that a shortened text is acceptable.
    """
    if limit <= 0:
        return "", 0
    if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= limit:
                return text, len(ids)
            truncated = tokenizer.decode(ids[:limit], skip_special_tokens=True)
            return truncated, token_length(truncated, tokenizer)
        except Exception:
            pass
    if token_length(text, None) <= limit:
        return text, estimate_tokens(text)
    words = text.split()
    if limit <= len(words):
        truncated = " ".join(words[:limit])
    else:
        approximate_chars = max(1, int(len(text) * limit / max(1, estimate_tokens(text))))
        truncated = text[:approximate_chars].rstrip()
    # The heuristic can over-estimate words + chars; trim explicitly until the
    # measured estimate respects the budget.
    while truncated and token_length(truncated, None) > limit:
        next_words = truncated.split()
        if len(next_words) > 1:
            truncated = " ".join(next_words[:-1])
        else:
            truncated = truncated[:-1].rstrip()
    return truncated, token_length(truncated, None)


# --------------------------------------------------------------------------- label leakage


@dataclass(frozen=True)
class LeakScan:
    text: str
    leaks: tuple[str, ...]

    @property
    def leaked(self) -> bool:
        return bool(self.leaks)


def strip_metadata_labels(text: str) -> LeakScan:
    """Remove VLM-written ``artist:``/``character:`` clauses from prose.

    Returns the cleaned text and the removed ``artist:<value>`` tokens. The
    upstream finalize step logs ``vlm_label_leak`` when ``leaks`` is non-empty.
    """
    leaks = tuple(match.group(0).strip() for match in _METADATA_LABEL_RE.finditer(text))
    if not leaks:
        return LeakScan(text=text, leaks=())
    cleaned = _METADATA_LABEL_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"(?m)^[\s,;]+|[\s,;]+$", "", cleaned).strip()
    return LeakScan(text=cleaned, leaks=leaks)


def contains_metadata_labels(text: str) -> bool:
    return bool(_METADATA_LABEL_RE.search(text))


def find_watermark_text(text: str, labels: Iterable[str]) -> list[str]:
    """Case-insensitive watermark label occurrences in a caption/prose field."""
    found: list[str] = []
    lower = text.casefold()
    for label in labels:
        clean = normalize_name(label)
        if len(clean) < 2:
            continue
        if clean.casefold() in lower:
            found.append(clean)
    return found


def strip_watermark_text(text: str, labels: Iterable[str]) -> str:
    cleaned = text
    for label in labels:
        clean = normalize_name(label)
        if len(clean) < 2:
            continue
        cleaned = re.sub(re.escape(clean), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    return cleaned.strip(" ,;\n\t")


# --------------------------------------------------------------------------- boxes


@dataclass(frozen=True)
class TransformResult:
    """One source box mapped to bucket coordinates; ``bbox`` is None when dropped."""

    bbox: tuple[int, int, int, int] | None
    clipped: bool = False
    outside: bool = False


def _as_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise AnnotationError(f"{name} must be finite, got {value!r}")
    return value


def parse_normalized_box(
    value: Any,
    *,
    name: str = "bbox",
    min_area_ratio: float = MIN_BOX_AREA_RATIO,
    max_area_ratio: float = MAX_BOX_AREA_RATIO,
    require_integers: bool = True,
    sort_corners: bool = True,
) -> tuple[int, int, int, int]:
    """Validate one ``[y1, x1, y2, x2]`` box on the normalized 0-1000 grid.

    Swapped corners (``y2 < y1`` or ``x2 < x1``) are canonicalized by default:
    the VLM occasionally emits right/bottom before left/top and a sorted box is
    still correct. The v2 bakeoff correction explicitly requires this.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AnnotationError(f"{name} must be a list of 4 numbers, got {value!r}")
    numbers = [_as_number(item, name) for item in value]
    if require_integers and any(not float(item).is_integer() for item in numbers):
        raise AnnotationError(f"{name} must contain integers 0-1000, got {value!r}")
    y1, x1, y2, x2 = (int(round(item)) for item in numbers)
    if sort_corners:
        y1, y2 = sorted((y1, y2))
        x1, x2 = sorted((x1, x2))
    if not (0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000):
        raise AnnotationError(
            f"{name} must satisfy 0<=y1<y2<=1000 and 0<=x1<x2<=1000, got {value!r}")
    area_ratio = ((y2 - y1) * (x2 - x1)) / 1_000_000
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        raise AnnotationError(
            f"{name} area ratio {area_ratio:.5f} outside [{min_area_ratio}, {max_area_ratio}]: {value!r}")
    return y1, x1, y2, x2


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU for two boxes in the same space, either ``yxyx`` or ``xyxy``."""
    if len(a) != 4 or len(b) != 4:
        return 0.0
    y1, x1, y2, x2 = (float(value) for value in a)
    by1, bx1, by2, bx2 = (float(value) for value in b)
    if not (y1 < y2 and x1 < x2 and by1 < by2 and bx1 < bx2):
        return 0.0
    inter_h = max(0.0, min(y2, by2) - max(y1, by1))
    inter_w = max(0.0, min(x2, bx2) - max(x1, bx1))
    intersection = inter_h * inter_w
    if intersection <= 0:
        return 0.0
    union = (y2 - y1) * (x2 - x1) + (by2 - by1) * (bx2 - bx1) - intersection
    return intersection / union if union > 0 else 0.0


def normalized_to_source_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Normalized ``yxyx`` -> source pixels ``x1, y1, x2, y2``."""
    if width <= 0 or height <= 0:
        raise AnnotationError(f"Image size must be positive, got {width}x{height}")
    y1, x1, y2, x2 = (float(value) for value in box)
    return (
        x1 / 1000.0 * width,
        y1 / 1000.0 * height,
        x2 / 1000.0 * width,
        y2 / 1000.0 * height,
    )


def source_box_to_normalized(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Source pixels ``x1, y1, x2, y2`` -> rounded normalized ``yxyx``."""
    if width <= 0 or height <= 0:
        raise AnnotationError(f"Image size must be positive, got {width}x{height}")
    x1, y1, x2, y2 = (float(value) for value in box)
    values = [
        int(round(y1 / height * 1000)),
        int(round(x1 / width * 1000)),
        int(round(y2 / height * 1000)),
        int(round(x2 / width * 1000)),
    ]
    return (
        max(0, min(1000, values[0])),
        max(0, min(1000, values[1])),
        max(0, min(1000, values[2])),
        max(0, min(1000, values[3])),
    )


def source_box_to_bucket_bbox(
    box: Sequence[float],
    geometry: BucketGeometry,
    *,
    clip: bool = True,
) -> TransformResult:
    """Source pixels ``x1, y1, x2, y2`` -> bucket ``abs_centralized`` ``yxyx``.

    ``x1``/``y1`` are floored and ``x2``/``y2`` are ceiled before the center
    subtraction, per the v2 contract. Fully outside boxes are dropped; partial
    boxes are clipped when ``clip`` is true and otherwise dropped.
    """
    if len(box) != 4:
        raise AnnotationError(f"box must have 4 values, got {box!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bx1 = math.floor(x1 * geometry.scale_x - geometry.left)
    by1 = math.floor(y1 * geometry.scale_y - geometry.top)
    bx2 = math.ceil(x2 * geometry.scale_x - geometry.left)
    by2 = math.ceil(y2 * geometry.scale_y - geometry.top)

    clipped = (bx1 < 0 or by1 < 0 or bx2 > geometry.bucket_w or by2 > geometry.bucket_h)
    if clipped:
        if not clip:
            return TransformResult(bbox=None, clipped=True, outside=True)
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(geometry.bucket_w, bx2)
        by2 = min(geometry.bucket_h, by2)

    if bx2 <= bx1 or by2 <= by1:
        return TransformResult(bbox=None, clipped=clipped, outside=True)

    center_x = geometry.bucket_w / 2.0
    center_y = geometry.bucket_h / 2.0
    bbox = (
        int(round(by1 - center_y)),
        int(round(bx1 - center_x)),
        int(round(by2 - center_y)),
        int(round(bx2 - center_x)),
    )
    half_w, half_h = geometry.bucket_w // 2, geometry.bucket_h // 2
    bbox = (
        max(-half_h, min(half_h, bbox[0])),
        max(-half_w, min(half_w, bbox[1])),
        max(-half_h, min(half_h, bbox[2])),
        max(-half_w, min(half_w, bbox[3])),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return TransformResult(bbox=None, clipped=clipped, outside=True)
    return TransformResult(bbox=bbox, clipped=clipped, outside=False)


def normalized_box_to_bucket_bbox(
    box: Sequence[float],
    source_size: tuple[int, int],
    bucket: tuple[int, int],
    *,
    clip: bool = True,
    geometry: BucketGeometry | None = None,
) -> TransformResult:
    """Normalized 0-1000 ``yxyx`` -> bucket ``abs_centralized`` ``yxyx``."""
    width, height = source_size
    geometry = geometry or bucket_geometry(source_size, bucket)
    source_box = normalized_to_source_box(box, width, height)
    return source_box_to_bucket_bbox(source_box, geometry, clip=clip)


def bucket_bbox_to_source_box(
    bbox: Sequence[int],
    geometry: BucketGeometry,
) -> tuple[float, float, float, float]:
    """Bucket ``abs_centralized`` ``yxyx`` -> source pixels ``x1, y1, x2, y2``."""
    y1, x1, y2, x2 = (float(value) for value in bbox)
    center_x = geometry.bucket_w / 2.0
    center_y = geometry.bucket_h / 2.0
    return (
        (x1 + center_x + geometry.left) / geometry.scale_x,
        (y1 + center_y + geometry.top) / geometry.scale_y,
        (x2 + center_x + geometry.left) / geometry.scale_x,
        (y2 + center_y + geometry.top) / geometry.scale_y,
    )


def bucket_bbox_to_normalized(
    bbox: Sequence[int],
    geometry: BucketGeometry,
) -> tuple[int, int, int, int]:
    source = bucket_bbox_to_source_box(bbox, geometry)
    return source_box_to_normalized(source, geometry.source_w, geometry.source_h)


def quarter_turn_size(size: tuple[int, int], exif_orientation: int) -> tuple[int, int]:
    """Swap width/height for EXIF orientations 5-8."""
    width, height = size
    return (height, width) if exif_orientation in (5, 6, 7, 8) else (width, height)


# --------------------------------------------------------------------------- schema dataclasses


@dataclass(frozen=True)
class CaptionSet:
    """The four caption variants; ``original`` may be empty while still raw VLM data."""

    long: str
    medium: str
    short: str
    original: str = ""

    def __post_init__(self) -> None:
        for name in ("long", "medium", "short"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AnnotationError(f"caption.{name} must be a non-empty string")
        if self.original is not None and not isinstance(self.original, str):
            raise AnnotationError("caption_original must be a string")

    def as_dict(self) -> dict[str, str]:
        result = {"long": self.long, "medium": self.medium, "short": self.short}
        if self.original:
            result["original"] = self.original
        return result


@dataclass(frozen=True)
class LayoutItem:
    label: str
    caption: str
    bbox: tuple[int, int, int, int]
    text: str | None = None

    def __post_init__(self) -> None:
        if self.label not in LAYOUT_LABELS:
            raise AnnotationError(f"layout item label must be one of {LAYOUT_LABELS}, got {self.label!r}")
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise AnnotationError("layout item caption must be non-empty")
        if len(self.bbox) != 4:
            raise AnnotationError(f"layout item bbox must have 4 values, got {self.bbox!r}")
        y1, x1, y2, x2 = (int(round(float(value))) for value in self.bbox)
        if y2 <= y1 or x2 <= x1:
            raise AnnotationError(f"layout item bbox must have positive area, got {self.bbox!r}")
        if self.label == "text" and (self.text is None or not str(self.text).strip()):
            raise AnnotationError("layout text items require non-empty `text`")
        if self.label != "text" and self.text:
            raise AnnotationError("only layout text items may set `text`")

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "label": self.label,
            "caption": self.caption,
            "bbox": list(self.bbox),
        }
        if self.label == "text":
            item["text"] = self.text
        return item


@dataclass(frozen=True)
class Layout:
    summary_caption: str
    items: tuple[LayoutItem, ...]
    summary_meta: tuple[str, ...] = ("anime",)
    canvas: str = "single"
    bbox_coord_space: str = NORMALIZED_SPACE
    bbox_format: str = BOX_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.summary_caption, str) or not self.summary_caption.strip():
            raise AnnotationError("layout summary caption must be non-empty")
        if self.canvas not in CANVAS_VALUES:
            raise AnnotationError(f"layout canvas must be one of {CANVAS_VALUES}, got {self.canvas!r}")
        if self.bbox_format != BOX_FORMAT:
            raise AnnotationError(f"layout bbox_format must be {BOX_FORMAT!r}, got {self.bbox_format!r}")
        if self.bbox_coord_space not in (NORMALIZED_SPACE, BUCKET_SPACE):
            raise AnnotationError(
                f"layout bbox_coord_space must be {NORMALIZED_SPACE!r} or {BUCKET_SPACE!r}, "
                f"got {self.bbox_coord_space!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "meta": list(self.summary_meta),
                "canvas": self.canvas,
                "caption": self.summary_caption,
            },
            "items": [item.to_dict() for item in self.items],
            "bbox_coord_space": self.bbox_coord_space,
            "bbox_format": self.bbox_format,
        }

    def with_items(self, items: Sequence[LayoutItem]) -> Layout:
        return Layout(
            summary_caption=self.summary_caption,
            items=tuple(items),
            summary_meta=self.summary_meta,
            canvas=self.canvas,
            bbox_coord_space=self.bbox_coord_space,
            bbox_format=self.bbox_format,
        )


@dataclass(frozen=True)
class Watermark:
    bbox: tuple[int, int, int, int]
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise AnnotationError("watermark label must be non-empty")
        y1, x1, y2, x2 = (int(round(float(value))) for value in self.bbox)
        if not (0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000):
            raise AnnotationError(f"watermark bbox outside the 0-1000 grid: {self.bbox!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"box_2d": list(self.bbox), "label": self.label}


@dataclass(frozen=True)
class Borders:
    edges: tuple[str, ...] = (BORDER_NONE,)

    def __post_init__(self) -> None:
        if not self.edges:
            raise AnnotationError("borders must not be empty; use ['none']")
        unknown = [edge for edge in self.edges if edge not in (BORDER_NONE, *BORDER_EDGES)]
        if unknown:
            raise AnnotationError(f"unknown border values {unknown}; allowed {BORDER_EDGES + (BORDER_NONE,)}")
        if BORDER_NONE in self.edges and len(self.edges) > 1:
            raise AnnotationError("'none' must not be combined with a border direction")
        if len(set(self.edges)) != len(self.edges):
            raise AnnotationError(f"duplicate border values: {self.edges!r}")

    @property
    def is_none(self) -> bool:
        return self.edges == (BORDER_NONE,)

    def to_list(self) -> list[str]:
        return list(self.edges)


@dataclass(frozen=True)
class OCRTextItem:
    """One OCR line in EXIF-transposed source pixels ``x1, y1, x2, y2``."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise AnnotationError("OCR item text must be non-empty")
        if not math.isfinite(float(self.confidence)):
            raise AnnotationError("OCR confidence must be finite")
        if len(self.bbox) != 4:
            raise AnnotationError("OCR bbox must have 4 values")


@dataclass(frozen=True)
class VLMAnnotation:
    caption: CaptionSet
    layout: Layout
    watermark: tuple[Watermark, ...] = ()
    borders: Borders = Borders()
    warnings: tuple[str, ...] = ()
    rating: str | None = None
    nsfw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "caption": self.caption.as_dict(),
            "watermark": [item.to_dict() for item in self.watermark],
            "borders": self.borders.to_list(),
            "layout": self.layout.to_dict(),
        }
        if self.rating is not None:
            payload["rating"] = self.rating
        if self.nsfw is not None:
            payload["nsfw"] = self.nsfw
        return payload


def parse_raw_caption(value: Any) -> CaptionSet:
    """Strictly parse the VLM/legacy ``{'long','medium','short'}`` caption object."""
    if not isinstance(value, Mapping):
        raise AnnotationError(f"caption must be an object, got {type(value).__name__}")
    missing = [key for key in ("long", "medium", "short")
               if not isinstance(value.get(key), str) or not value[key].strip()]
    if missing:
        raise AnnotationError(f"caption fields missing/empty: {missing}")
    return CaptionSet(long=value["long"], medium=value["medium"], short=value["short"])


def parse_borders(value: Any) -> Borders:
    if value is None:
        return Borders()
    if not isinstance(value, (list, tuple)):
        raise AnnotationError(f"borders must be a list, got {value!r}")
    return Borders(tuple(str(edge).strip() for edge in value))


def parse_watermarks(value: Any) -> tuple[Watermark, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise AnnotationError(f"watermark must be a list, got {value!r}")
    watermarks: list[Watermark] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AnnotationError(f"watermark[{index}] must be an object")
        bbox_value = item.get("box_2d", item.get("bbox"))
        if not isinstance(bbox_value, (list, tuple)):
            raise AnnotationError(f"watermark[{index}].box_2d must be a list")
        y1, x1, y2, x2 = parse_normalized_box(
            bbox_value, name=f"watermark[{index}].box_2d",
            min_area_ratio=0.0, max_area_ratio=1.0)
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise AnnotationError(f"watermark[{index}].label must be non-empty")
        watermarks.append(Watermark(bbox=(y1, x1, y2, x2), label=label.strip()))
    return tuple(watermarks)


def parse_rating(value: Any) -> str | None:
    """Optional Danbooru-style rating; unknown values are a hard schema error."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise AnnotationError(f"rating must be a string, got {type(value).__name__}")
    rating = value.strip().lower()
    if rating not in RATING_VALUES:
        raise AnnotationError(f"rating must be one of {RATING_VALUES}, got {value!r}")
    return rating


def parse_nsfw(value: Any) -> dict[str, Any] | None:
    """Optional ``{"tags": [...], "description": "..."}`` NSFW summary."""
    if value is None or value == "":
        return None
    if not isinstance(value, Mapping):
        raise AnnotationError(f"nsfw must be an object, got {type(value).__name__}")
    raw_tags = value.get("tags", [])
    if not isinstance(raw_tags, (list, tuple)) or not all(
            isinstance(tag, str) for tag in raw_tags):
        raise AnnotationError("nsfw.tags must be a list of strings")
    description = value.get("description", "")
    if not isinstance(description, str):
        raise AnnotationError("nsfw.description must be a string")
    tags = dedupe_tags(tag.strip() for tag in raw_tags if tag.strip())[:12]
    return {"tags": tags, "description": description.strip()}


def _reading_order(item: LayoutItem) -> tuple[int, int, int, int]:
    y1, x1, y2, x2 = item.bbox
    return (y1, x1, y2, x2)


def parse_layout(
    value: Any,
    *,
    expected_space: str = NORMALIZED_SPACE,
    require_items: bool = True,
    warnings: list[str] | None = None,
) -> Layout:
    """Strictly parse and validate a layout object in normalized ``yxyx`` space."""
    if not isinstance(value, Mapping):
        raise AnnotationError(f"layout must be an object, got {type(value).__name__}")
    space = value.get("bbox_coord_space", NORMALIZED_SPACE)
    fmt = value.get("bbox_format", BOX_FORMAT)
    if space != expected_space:
        raise AnnotationError(f"layout bbox_coord_space must be {expected_space!r}, got {space!r}")
    if fmt != BOX_FORMAT:
        raise AnnotationError(f"layout bbox_format must be {BOX_FORMAT!r}, got {fmt!r}")

    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise AnnotationError("layout.summary must be an object")
    caption = summary.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise AnnotationError("layout.summary.caption must be non-empty")

    meta_raw = summary.get("meta", ["anime"])
    if not isinstance(meta_raw, (list, tuple)) or not all(isinstance(item, str) for item in meta_raw):
        raise AnnotationError("layout.summary.meta must be a list of strings")
    meta = tuple(item.strip() for item in meta_raw if item.strip()) or ("anime",)
    if len(meta) > MAX_SUMMARY_META:
        if warnings is not None:
            warnings.append(f"layout.summary.meta truncated from {len(meta)} to {MAX_SUMMARY_META}")
        meta = meta[:MAX_SUMMARY_META]
    canvas = summary.get("canvas", "single")
    if canvas is None:
        canvas = "single"
    if canvas not in CANVAS_VALUES:
        raise AnnotationError(f"layout.summary.canvas must be one of {CANVAS_VALUES}, got {canvas!r}")

    raw_items = value.get("items")
    if not isinstance(raw_items, (list, tuple)):
        raise AnnotationError("layout.items must be a list")
    if require_items and not raw_items:
        raise AnnotationError("layout.items must contain at least one item")
    if len(raw_items) > MAX_LAYOUT_ITEMS:
        raise AnnotationError(f"layout.items has {len(raw_items)} items; maximum is {MAX_LAYOUT_ITEMS}")

    items: list[LayoutItem] = []
    text_items = 0
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise AnnotationError(f"layout.items[{index}] must be an object")
        label = raw.get("label")
        if label not in LAYOUT_LABELS:
            raise AnnotationError(
                f"layout.items[{index}].label must be one of {LAYOUT_LABELS}, got {label!r}")
        item_caption = raw.get("caption")
        if not isinstance(item_caption, str) or not item_caption.strip():
            raise AnnotationError(f"layout.items[{index}].caption must be non-empty")
        bbox = parse_normalized_box(raw.get("bbox"), name=f"layout.items[{index}].bbox")
        text = raw.get("text")
        if label == "text":
            text_items += 1
            if not isinstance(text, str) or not text.strip():
                raise AnnotationError(f"layout.items[{index}] is text but has no non-empty `text`")
        elif text:
            raise AnnotationError(f"layout.items[{index}] is {label!r} and must not set `text`")
        item = LayoutItem(label=label, caption=item_caption, bbox=bbox,
                          text=text if label == "text" else None)
        items.append(item)
    if text_items > MAX_LAYOUT_TEXT_ITEMS:
        raise AnnotationError(
            f"layout has {text_items} text items; maximum is {MAX_LAYOUT_TEXT_ITEMS}")

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            first, second = items[left], items[right]
            iou = box_iou(first.bbox, second.bbox)
            if first.label == second.label and iou > 0.9:
                raise AnnotationError(
                    f"duplicate layout boxes at items[{left}] and items[{right}] (IoU={iou:.2f})")
            if (first.label == second.label == "text" and first.text == second.text and iou > 0.9):
                raise AnnotationError(
                    f"duplicate layout text at items[{left}] and items[{right}] (IoU={iou:.2f})")

    items.sort(key=_reading_order)
    return Layout(
        summary_caption=caption.strip(),
        items=tuple(items),
        summary_meta=meta,
        canvas=canvas,
        bbox_coord_space=space,
        bbox_format=fmt,
    )


def parse_vlm_annotation(
    raw: Mapping[str, Any],
    *,
    strip_leaks: bool = True,
    filter_watermarks: bool = True,
    require_layout: bool = True,
) -> VLMAnnotation:
    """Validate one combined VLM response and apply the spec's soft fixes.

    Hard schema violations raise ``AnnotationError`` so the caller can retry.
    Soft fixes (label leakage, watermark label/overlap) are recorded in
    ``VLMAnnotation.warnings``.
    """
    if not isinstance(raw, Mapping):
        raise AnnotationError(f"annotation must be an object, got {type(raw).__name__}")
    warnings: list[str] = []
    caption = parse_raw_caption(raw.get("caption"))
    watermark = parse_watermarks(raw.get("watermark"))
    borders = parse_borders(raw.get("borders"))
    rating = parse_rating(raw.get("rating"))
    nsfw = parse_nsfw(raw.get("nsfw"))
    layout = parse_layout(raw.get("layout"), warnings=warnings,
                          require_items=require_layout)

    labels = [item.label for item in watermark]
    captions: dict[str, str] = {}
    for kind in ("long", "medium", "short"):
        text = getattr(caption, kind)
        if strip_leaks:
            scan = strip_metadata_labels(text)
            if scan.leaked:
                warnings.append(f"vlm_label_leak:{kind}")
                text = scan.text
        if filter_watermarks:
            leaked = find_watermark_text(text, labels)
            if leaked:
                warnings.append(f"watermark_text_leak:{kind}")
                text = strip_watermark_text(text, labels)
        if not text.strip():
            raise AnnotationError(f"caption.{kind} is empty after leak/watermark filtering")
        captions[kind] = text
    caption = CaptionSet(long=captions["long"], medium=captions["medium"],
                         short=captions["short"])

    summary_caption = layout.summary_caption
    if strip_leaks:
        scan = strip_metadata_labels(summary_caption)
        if scan.leaked:
            warnings.append("vlm_label_leak:layout.summary.caption")
            summary_caption = scan.text
    if filter_watermarks:
        leaked = find_watermark_text(summary_caption, labels)
        if leaked:
            warnings.append("watermark_text_leak:layout.summary.caption")
            summary_caption = strip_watermark_text(summary_caption, labels)
    if not summary_caption.strip():
        raise AnnotationError("layout.summary.caption is empty after filtering")

    items: list[LayoutItem] = []
    dropped = 0
    for item in layout.items:
        item_caption = item.caption
        if strip_leaks:
            scan = strip_metadata_labels(item_caption)
            if scan.leaked:
                warnings.append("vlm_label_leak:layout.items[].caption")
                item_caption = scan.text
        if filter_watermarks:
            overlaps = any(box_iou(item.bbox, mark.bbox) > WATERMARK_IOU_THRESHOLD
                           for mark in watermark)
            text_leak = (item.label == "text" and item.text is not None
                         and bool(find_watermark_text(item.text, labels)))
            if item.label == "text" and (overlaps or text_leak):
                dropped += 1
                continue
            leaked = find_watermark_text(item_caption, labels)
            if leaked:
                warnings.append("watermark_text_leak:layout.items[].caption")
                item_caption = strip_watermark_text(item_caption, labels)
        if not item_caption.strip():
            raise AnnotationError("layout item caption is empty after filtering")
        items.append(LayoutItem(label=item.label, caption=item_caption,
                                bbox=item.bbox, text=item.text))
    if dropped:
        warnings.append(f"watermark_overlap_dropped_items={dropped}")
    if not items and require_layout:
        raise AnnotationError("layout has no items after watermark filtering")

    return VLMAnnotation(
        caption=caption,
        layout=Layout(
            summary_caption=summary_caption,
            items=tuple(items),
            summary_meta=layout.summary_meta,
            canvas=layout.canvas,
            bbox_coord_space=layout.bbox_coord_space,
            bbox_format=layout.bbox_format,
        ),
        watermark=watermark,
        borders=borders,
        warnings=tuple(warnings),
        rating=rating,
        nsfw=nsfw,
    )


def filter_layout_watermarks(
    layout: Layout,
    watermarks: Sequence[Watermark],
    *,
    iou_threshold: float = WATERMARK_IOU_THRESHOLD,
    labels: Sequence[str] = ("text",),
) -> tuple[Layout, tuple[int, ...]]:
    """Drop text items that overlap a watermark box (same coordinate space)."""
    kept: list[LayoutItem] = []
    dropped: list[int] = []
    for index, item in enumerate(layout.items):
        overlaps = item.label in labels and any(
            box_iou(item.bbox, mark.bbox) > iou_threshold for mark in watermarks)
        if overlaps:
            dropped.append(index)
        else:
            kept.append(item)
    return layout.with_items(kept), tuple(dropped)


def filter_ocr_watermarks(
    items: Sequence[OCRTextItem],
    watermarks: Sequence[Watermark],
    *,
    width: int,
    height: int,
    iou_threshold: float = WATERMARK_IOU_THRESHOLD,
) -> tuple[list[OCRTextItem], list[OCRTextItem]]:
    """Drop OCR lines whose source box overlaps a normalized watermark box."""
    mark_boxes = []
    for mark in watermarks:
        y1, x1, y2, x2 = mark.bbox
        mark_boxes.append((
            x1 / 1000.0 * width,
            y1 / 1000.0 * height,
            x2 / 1000.0 * width,
            y2 / 1000.0 * height,
        ))
    kept: list[OCRTextItem] = []
    dropped: list[OCRTextItem] = []
    for item in items:
        # OCR boxes are xyxy; watermark boxes are yxyx, so compare with converted tuples.
        overlap = any(
            box_iou(
                (item.bbox[1], item.bbox[0], item.bbox[3], item.bbox[2]),
                (mark[1], mark[0], mark[3], mark[2]),
            ) > iou_threshold
            for mark in mark_boxes
        )
        (dropped if overlap else kept).append(item)
    return kept, dropped


def watermark_area_ratio(watermarks: Sequence[Watermark]) -> float:
    """Total watermark box area as a fraction of the canvas (union-free upper bound)."""
    return sum(((box[2] - box[0]) * (box[3] - box[1])) / 1_000_000 for box in
               (mark.bbox for mark in watermarks))


# --------------------------------------------------------------------------- prefix / captions


def _clean_prefix_value(value: Any, kind: str) -> str:
    text = normalize_name(str(value))
    text = re.sub(r"(?i)^(?:artist|character)\s*:\s*", "", text)
    text = re.sub(r"\s*,\s*", " ", text).strip()
    if not text:
        raise AnnotationError(f"{kind} label must be non-empty")
    return text


def build_prefix(
    trigger: str = TRIGGER_DEFAULT,
    artists: Iterable[str] = (),
    characters: Iterable[str] = (),
    *,
    include_artist: bool = True,
    unknown_if_missing: bool = True,
) -> list[str]:
    """Deterministic ``@trigger, artist:<id>..., character:<id>...`` prefix parts."""
    trigger = normalize_name(trigger)
    if not trigger:
        raise AnnotationError("trigger must be non-empty")
    if not trigger.startswith("@"):
        trigger = f"@{trigger}"
    artist_values = [_clean_prefix_value(value, "artist") for value in artists]
    character_values = [_clean_prefix_value(value, "character") for value in characters]
    artist_values = dedupe_names(artist_values)
    character_values = dedupe_names(character_values)
    parts = [trigger]
    if include_artist:
        if not artist_values and unknown_if_missing:
            artist_values = [UNKNOWN_ARTIST]
        parts.extend(f"artist:{value}" for value in artist_values)
    parts.extend(f"character:{value}" for value in character_values)
    return parts


def compose_caption(
    prefix: Sequence[str],
    tags: Iterable[str] = (),
    prose: str = "",
    *,
    max_tags: int | None = None,
) -> str:
    """Join prefix, deduped tags and prose with the official ``, `` separator."""
    clean_tags = dedupe_tags(tags)
    if max_tags is not None:
        clean_tags = clean_tags[: max(0, max_tags)]
    parts = [normalize_name(part) for part in prefix if normalize_name(part)]
    parts.extend(clean_tags)
    if isinstance(prose, str) and prose.strip():
        parts.append(prose.strip())
    return ", ".join(parts)


@dataclass(frozen=True)
class CaptionVariants:
    """Final deterministic caption strings keyed by cache variant kind."""

    original: str
    long: str
    medium: str
    short: str

    def as_dict(self) -> dict[str, str]:
        return {"original": self.original, "long": self.long,
                "medium": self.medium, "short": self.short}


def build_caption_variants(
    caption: CaptionSet,
    *,
    original_prose: str,
    trigger: str = TRIGGER_DEFAULT,
    artists: Iterable[str] = (),
    characters: Iterable[str] = (),
    tags: Iterable[str] = (),
    max_tags: int = LONG_TAG_LIMIT,
    unknown_if_missing: bool = True,
) -> CaptionVariants:
    """Compose the four deterministic caption variants with tag caps 40/40/25/12.

    ``caption_original`` never uses VLM prose: ``original_prose`` is the source
    prose (or empty for expansion rows), and tags carry the description.
    """
    if not isinstance(original_prose, str):
        raise AnnotationError("original_prose must be a string")
    prefix = build_prefix(trigger, artists, characters,
                          unknown_if_missing=unknown_if_missing)
    max_tags = max(0, max_tags)
    return CaptionVariants(
        original=compose_caption(prefix, tags, original_prose,
                                 max_tags=min(LONG_TAG_LIMIT, max_tags)),
        long=compose_caption(prefix, tags, caption.long,
                             max_tags=min(LONG_TAG_LIMIT, max_tags)),
        medium=compose_caption(prefix, tags, caption.medium,
                               max_tags=min(MEDIUM_TAG_LIMIT, max_tags)),
        short=compose_caption(prefix, tags, caption.short,
                              max_tags=min(SHORT_TAG_LIMIT, max_tags)),
    )


def missing_prefix_labels(
    text: str,
    trigger: str = TRIGGER_DEFAULT,
    artists: Iterable[str] = (),
    characters: Iterable[str] = (),
    *,
    unknown_if_missing: bool = True,
) -> list[str]:
    """Return prefix tokens absent from ``text`` (I2/I3 validator helper)."""
    expected = build_prefix(trigger, artists, characters,
                            unknown_if_missing=unknown_if_missing)
    return [token for token in expected if token not in text]


def validate_caption_prefix(
    text: str,
    trigger: str = TRIGGER_DEFAULT,
    artists: Iterable[str] = (),
    characters: Iterable[str] = (),
    *,
    unknown_if_missing: bool = True,
) -> None:
    """Raise ``AnnotationError`` unless trigger/artist/character tokens are present."""
    missing = missing_prefix_labels(text, trigger, artists, characters,
                                    unknown_if_missing=unknown_if_missing)
    if missing:
        raise AnnotationError(f"caption is missing deterministic prefix tokens: {missing}")


# --------------------------------------------------------------------------- layout transforms


@dataclass
class LayoutTransformStats:
    total: int = 0
    outside: int = 0
    clipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "outside_crop": self.outside, "clipped": self.clipped}


def layout_to_bucket(
    layout: Layout,
    source_size: tuple[int, int],
    bucket: tuple[int, int],
    *,
    clip: bool = True,
    geometry: BucketGeometry | None = None,
) -> tuple[Layout, LayoutTransformStats]:
    """Transform a normalized layout into bucket ``abs_centralized`` space."""
    if layout.bbox_coord_space != NORMALIZED_SPACE:
        raise AnnotationError(
            f"layout is already {layout.bbox_coord_space!r}; expected {NORMALIZED_SPACE!r}")
    geometry = geometry or bucket_geometry(source_size, bucket)
    items: list[LayoutItem] = []
    stats = LayoutTransformStats(total=len(layout.items))
    for item in layout.items:
        result = normalized_box_to_bucket_bbox(
            item.bbox, source_size, bucket, clip=clip, geometry=geometry)
        if result.bbox is None:
            stats.outside += 1
            continue
        stats.clipped += int(result.clipped)
        items.append(LayoutItem(label=item.label, caption=item.caption,
                                bbox=result.bbox, text=item.text))
    items.sort(key=_reading_order)
    return (
        Layout(
            summary_caption=layout.summary_caption,
            items=tuple(items),
            summary_meta=layout.summary_meta,
            canvas=layout.canvas,
            bbox_coord_space=BUCKET_SPACE,
            bbox_format=BOX_FORMAT,
        ),
        stats,
    )


def layout_to_normalized(
    layout: Layout,
    source_size: tuple[int, int],
    bucket: tuple[int, int],
    *,
    geometry: BucketGeometry | None = None,
) -> tuple[Layout, LayoutTransformStats]:
    """Transform a bucket ``abs_centralized`` layout back to normalized ``yxyx``."""
    if layout.bbox_coord_space != BUCKET_SPACE:
        raise AnnotationError(
            f"layout is already {layout.bbox_coord_space!r}; expected {BUCKET_SPACE!r}")
    geometry = geometry or bucket_geometry(source_size, bucket)
    items: list[LayoutItem] = []
    stats = LayoutTransformStats(total=len(layout.items))
    for item in layout.items:
        y1, x1, y2, x2 = bucket_bbox_to_normalized(item.bbox, geometry)
        if y2 <= y1 or x2 <= x1:
            stats.outside += 1
            continue
        items.append(LayoutItem(label=item.label, caption=item.caption,
                                bbox=(y1, x1, y2, x2), text=item.text))
    items.sort(key=_reading_order)
    return (
        Layout(
            summary_caption=layout.summary_caption,
            items=tuple(items),
            summary_meta=layout.summary_meta,
            canvas=layout.canvas,
            bbox_coord_space=NORMALIZED_SPACE,
            bbox_format=BOX_FORMAT,
        ),
        stats,
    )


def layout_json(layout: Layout | Mapping[str, Any]) -> str:
    """Compact JSON encoding used as the cache ``layout`` text variant."""
    payload = layout.to_dict() if isinstance(layout, Layout) else dict(layout)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
