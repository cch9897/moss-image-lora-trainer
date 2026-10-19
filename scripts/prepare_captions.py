#!/usr/bin/env python3
"""Prepare mixed-artist MOSS-Image captions from raw prompt sidecars or JSONL.

The expected raw sidecar is a tag line followed by prose paragraphs::

    @source_handle, artist:alice, character:hero, landscape, sunset, city lights

    <Chinese prose paragraph(s)>

    <English prose paragraph(s)>
    #vlm

The script deliberately keeps the artist tag in every training variant so a
LoRA can learn artist-conditioned mixing.  It also emits ``caption_no_artist``
for future artist-dropout experiments, strips source watermarks/handles and
``#vlm`` markers, removes boilerplate quality negatives, and reports caption
length, text/hand-tag coverage and center-crop risk.

Output layout (consumable by ``moss-lora-cache --input <output>``)::

    <output>/metadata.jsonl
    <output>/report.json
    <output>/report.md
    <output>/images/000001.jpg -> <source image>   (or a copy)

Only this script writes to ``<output>``; source images/sidecars are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moss_lora.cache import IMAGE_EXTS, assign_bucket, build_buckets  # noqa: E402

TRIGGER_DEFAULT = "mystyle"
UNKNOWN_ARTIST = "unknown"
MEDIUM_TAG_LIMIT = 25
SHORT_TAG_LIMIT = 12
LONG_TAIL_BUCKETS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, None))

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

TEXT_HAND_TERMS = ("text", "speech bubble", "sign", "writing", "hand", "hands", "fingers", "holding")
TEXT_GROUP = frozenset(("text", "speech bubble", "sign", "writing"))
HAND_GROUP = frozenset(("hand", "hands", "fingers", "holding"))
ZH_COVERAGE_TERMS: dict[str, tuple[str, ...]] = {
    "text": ("文字", "文本", "字样"),
    "speech bubble": ("对话气泡", "气泡", "对话框"),
    "sign": ("招牌", "标牌", "指示牌", "牌子"),
    "writing": ("书写", "字迹", "笔迹"),
    "hand": ("手", "手掌"),
    "hands": ("双手", "手"),
    "fingers": ("手指", "指头"),
    "holding": ("手持", "拿着", "握住", "握着", "举起"),
}

UNSUPPORTED_IMAGE_EXTS = frozenset((".gif", ".tif", ".tiff", ".avif", ".heic", ".heif"))
SIDECAR_EXTS = (".json", ".txt")


class PrepareError(RuntimeError):
    """Raised for invalid input/output configuration."""


class SidecarError(ValueError):
    """A sidecar exists but cannot be parsed into a usable prompt."""


# --------------------------------------------------------------------------- normalization


def normalize_name(value: str) -> str:
    """NFKC, full-width @ -> @, collapse whitespace and trim separators."""
    text = unicodedata.normalize("NFKC", str(value)).replace("\uff20", "@")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,;，；")


def normalize_tag_key(value: str) -> str:
    """Case/underscore/space-insensitive key used for dedupe and matching."""
    text = normalize_name(value).lower().replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_handle(value: str) -> str | None:
    handle = normalize_name(value).lstrip("@").strip()
    return f"@{handle}" if handle else None


def count_cjk(text: str) -> int:
    return sum(
        1
        for char in text
        if ("\u3400" <= char <= "\u9fff")
        or ("\uf900" <= char <= "\ufaff")
        or ("\U00020000" <= char <= "\U0002ffff")
    )


def cjk_ratio(text: str) -> float:
    cjk = count_cjk(text)
    letters = sum(1 for char in text if char.isascii() and char.isalpha())
    return cjk / max(1, cjk + letters)


def is_zh_block(text: str) -> bool:
    """A prose block is Chinese when it has enough CJK characters.

    A stray CJK character inside English prose (artist/character names are
    common) must not flip the whole language choice.
    """
    return count_cjk(text) >= 4 and cjk_ratio(text) >= 0.15


def strip_vlm(text: str) -> str:
    return re.sub(r"(?i)#vlm\b", "", text)


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


def dedupe_tags(values: Iterable[str]) -> tuple[list[str], Counter[str]]:
    """Normalize/dedupe positive tags; returns (tags, removed-negative counts)."""
    seen: set[str] = set()
    tags: list[str] = []
    removed: Counter[str] = Counter()
    for value in values:
        clean = normalize_name(value)
        key = normalize_tag_key(clean)
        if not key:
            continue
        if key in NEGATIVE_TAG_KEYS:
            removed[key] += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        tags.append(clean)
    return tags, removed


# --------------------------------------------------------------------------- parsing


@dataclass
class ParsedPrompt:
    artist_values: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_handle: str | None = None
    extra_handles: list[str] = field(default_factory=list)
    prose_en: list[str] = field(default_factory=list)
    prose_zh: list[str] = field(default_factory=list)
    tag_line_found: bool = False
    preference_level: Any = None
    layout: Any = None


@dataclass
class SourceItem:
    image: Path
    source_rel: str
    payload: Any
    sidecar: Path | None = None


@dataclass
class PreparedItem:
    image: Path
    source_rel: str
    parsed: ParsedPrompt
    tags: list[str]
    removed_negatives: Counter[str]
    artist: str
    artist_missing: bool
    characters: list[str]
    source_handle: str | None
    prose_en: list[str]
    prose_zh: list[str]
    preference_level: Any = None
    layout: Any = None
    artist_values: list[str] = field(default_factory=list)


def looks_like_tag_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if text[0] in ("@", "\uff20"):
        return True
    if re.search(r"(?i)(?:^|[,，]\s*)(?:artist|character)\s*:", text):
        return True
    # A long comma-separated line without sentence punctuation is a tag line,
    # not prose.  This keeps generic "tag line first" sidecars working.
    return (
        len(re.findall(r"[,，]", text)) >= 2
        and not re.search(r"[.!?。！？]\s*$", text)
        and cjk_ratio(text) < 0.2
    )


def looks_like_raw_prompt(text: str) -> bool:
    first = next((line.strip() for line in strip_vlm(text).splitlines() if line.strip()), "")
    if not first:
        return False
    return first[0] in ("@", "\uff20") or bool(
        re.search(r"(?i)\b(?:artist|character)\s*:", first)
    )


def split_tag_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                result.extend(part.strip() for part in re.split(r"[,，]", entry) if part.strip())
        return result
    return [str(value)]


def _name_values(value: Any) -> list[str]:
    """Artist/character values keep commas inside one name (unlike tags)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(entry).strip() for entry in value if str(entry).strip()]
    return [str(value)]


def parse_tag_line(line: str) -> ParsedPrompt:
    parsed = ParsedPrompt(tag_line_found=True)
    for raw in re.split(r"[,，]", strip_vlm(line)):
        token = normalize_name(raw)
        if not token:
            continue
        if token[0] in ("@", "\uff20"):
            handle = normalize_handle(token)
            if handle:
                if parsed.source_handle is None:
                    parsed.source_handle = handle
                else:
                    parsed.extra_handles.append(handle)
            continue
        match = re.match(r"(?i)^(artist|character)\s*:\s*(.*)$", token)
        if match:
            kind, value = match.group(1).lower(), normalize_name(match.group(2))
            if not value:
                continue
            if kind == "artist":
                parsed.artist_values.append(value)
            else:
                parsed.characters.append(value)
            continue
        parsed.tags.append(token)
    parsed.artist_values = dedupe_names(parsed.artist_values)
    parsed.characters = dedupe_names(parsed.characters)
    return parsed


def _split_paragraphs(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) <= 1:
        joined = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return [joined] if joined else []
    return blocks


def _classify_paragraphs(blocks: Iterable[str]) -> tuple[list[str], list[str]]:
    english: list[str] = []
    chinese: list[str] = []
    for block in blocks:
        clean = block.strip()
        if not clean:
            continue
        (chinese if is_zh_block(clean) else english).append(clean)
    return english, chinese


def parse_raw_prompt(text: str) -> ParsedPrompt:
    text = strip_vlm(text)
    lines = text.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        raise SidecarError("empty prompt text")
    tag_line = ""
    remainder_start = first
    if looks_like_tag_line(lines[first]):
        tag_line = lines[first].strip()
        remainder_start = first + 1
    parsed = parse_tag_line(tag_line) if tag_line else ParsedPrompt()
    remainder = "\n".join(lines[remainder_start:])
    parsed.prose_en, parsed.prose_zh = _classify_paragraphs(_split_paragraphs(remainder))
    if not parsed.tags and not parsed.artist_values and not parsed.characters and not (
        parsed.prose_en or parsed.prose_zh
    ):
        raise SidecarError("no tags, artist, character or prose")
    return parsed


def _first_caption_text(caption: Any) -> str | None:
    if isinstance(caption, dict):
        for key in ("long", "medium", "short"):
            value = caption.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(caption, str) and caption.strip():
        return caption
    return None


def _layout_prose(layout: Any) -> str | None:
    if not isinstance(layout, dict):
        return None
    summary = layout.get("summary")
    if isinstance(summary, dict):
        caption = summary.get("caption")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
    pieces: list[str] = []
    items = layout.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("caption", "text", "label"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
                    break
    return " ".join(pieces) if pieces else None


def parse_json_sidecar(data: dict[str, Any], *, allow_nested: bool = True) -> ParsedPrompt:
    parsed = ParsedPrompt(
        preference_level=data.get("preference_level"),
        layout=data.get("layout"),
        source_handle=normalize_handle(str(data["source_handle"]))
        if isinstance(data.get("source_handle"), str) else None,
    )
    explicit_artists = dedupe_names(_name_values(data.get("artist") or data.get("artists")))
    explicit_characters = dedupe_names(
        _name_values(data.get("character") or data.get("characters"))
    )
    explicit_tags = split_tag_values(data.get("extra_tags") or data.get("tags"))

    caption = data.get("caption")
    if caption is None:
        caption = data.get("text", data.get("prompt"))
    nested: dict[str, Any] | None = None
    caption_text: str | None = None
    if isinstance(caption, str) and caption.strip().startswith("{"):
        if allow_nested:
            try:
                candidate = json.loads(caption)
            except json.JSONDecodeError as error:
                raise SidecarError(f"invalid nested caption JSON: {error}") from error
            if isinstance(candidate, dict):
                nested = candidate
    elif isinstance(caption, dict):
        caption_text = _first_caption_text(caption)
        if caption_text is None:
            if parsed.layout is None:
                parsed.layout = caption
            caption_text = _layout_prose(caption)
    elif isinstance(caption, str) and caption.strip():
        caption_text = caption.strip()

    if nested is not None:
        inner = parse_json_sidecar(nested, allow_nested=False)
        parsed.artist_values = explicit_artists or inner.artist_values
        parsed.characters = dedupe_names(explicit_characters + inner.characters)
        parsed.tags = explicit_tags + inner.tags
        parsed.source_handle = parsed.source_handle or inner.source_handle
        parsed.extra_handles = inner.extra_handles
        parsed.prose_en = inner.prose_en
        parsed.prose_zh = inner.prose_zh
        parsed.tag_line_found = inner.tag_line_found
        if parsed.layout is None:
            parsed.layout = inner.layout
        if parsed.preference_level is None:
            parsed.preference_level = inner.preference_level
        return parsed

    prose_source = caption_text or data.get("image_description")
    if isinstance(prose_source, str) and prose_source.strip():
        prose_source = strip_vlm(prose_source).strip()
        if looks_like_raw_prompt(prose_source):
            raw = parse_raw_prompt(prose_source)
            parsed.artist_values = explicit_artists or raw.artist_values
            parsed.characters = dedupe_names(explicit_characters + raw.characters)
            parsed.tags = explicit_tags + raw.tags
            parsed.source_handle = parsed.source_handle or raw.source_handle
            parsed.extra_handles = raw.extra_handles
            parsed.prose_en = raw.prose_en
            parsed.prose_zh = raw.prose_zh
            parsed.tag_line_found = raw.tag_line_found
        else:
            parsed.artist_values = explicit_artists
            parsed.characters = explicit_characters
            parsed.tags = explicit_tags
            parsed.prose_en, parsed.prose_zh = _classify_paragraphs(_split_paragraphs(prose_source))
    else:
        parsed.artist_values = explicit_artists
        parsed.characters = explicit_characters
        parsed.tags = explicit_tags
        layout_prose = _layout_prose(parsed.layout)
        if layout_prose:
            parsed.prose_en, parsed.prose_zh = _classify_paragraphs(_split_paragraphs(layout_prose))

    if not parsed.tags and not parsed.artist_values and not parsed.characters and not (
        parsed.prose_en or parsed.prose_zh
    ):
        keys = ", ".join(sorted(data)) or "(none)"
        raise SidecarError(f"no usable caption fields (keys: {keys})")
    return parsed


def parse_payload(payload: Any) -> ParsedPrompt:
    if isinstance(payload, dict):
        return parse_json_sidecar(payload)
    if isinstance(payload, str):
        return parse_raw_prompt(payload)
    raise SidecarError(f"unsupported sidecar payload type: {type(payload).__name__}")


# --------------------------------------------------------------------------- discovery


@dataclass
class Discovery:
    items: list[SourceItem] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    unsupported_images: list[str] = field(default_factory=list)
    orphan_sidecars: list[str] = field(default_factory=list)
    both_sidecars: list[str] = field(default_factory=list)
    bad_metadata_lines: list[str] = field(default_factory=list)
    duplicate_manifest_rows: int = 0
    supported: int = 0


def _source_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _find_sidecar(image: Path) -> Path | None:
    candidates = [image.with_suffix(suffix) for suffix in SIDECAR_EXTS]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _read_sidecar_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise SidecarError(f"invalid UTF-8: {error}") from error
    if not text.strip():
        raise SidecarError("empty sidecar")
    stripped = text.lstrip()
    if path.suffix.lower() == ".json" or stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise SidecarError(f"invalid JSON: {error}") from error
        if not isinstance(data, dict):
            raise SidecarError(f"JSON sidecar must be an object, got {type(data).__name__}")
        return data
    return text


def _has_caption_fields(row: dict[str, Any]) -> bool:
    for key in ("caption", "text", "prompt", "image_description", "layout", "artist", "artists",
                "character", "characters", "extra_tags"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, dict, tuple)) and value:
            return True
        if value is not None and not isinstance(value, (str, list, dict, tuple)):
            return True
    return False


def _merge_row_metadata(payload: Any, row: dict[str, Any]) -> Any:
    """Keep row-level metadata (e.g. preference_level) when the caption comes from a sidecar."""
    extra = {key: row[key] for key in ("preference_level",) if row.get(key) is not None}
    if not extra:
        return payload
    if isinstance(payload, dict):
        return {**extra, **payload}
    return {"image_description": payload, **extra}


def _read_directory_metadata(root: Path) -> tuple[dict[Path, dict[str, Any]], list[str]]:
    """Rows from root/metadata.jsonl keyed by resolved image path."""
    rows: dict[Path, dict[str, Any]] = {}
    bad: list[str] = []
    manifest = root / "metadata.jsonl"
    if not manifest.is_file():
        return rows, bad
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return rows, ["metadata.jsonl (invalid UTF-8)"]
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad.append(f"metadata.jsonl:{line_no}")
            continue
        if not isinstance(row, dict):
            bad.append(f"metadata.jsonl:{line_no}")
            continue
        name = row.get("file_name") or row.get("image")
        if isinstance(name, str) and name.strip():
            rows[(manifest.parent / name).resolve()] = row
    return rows, bad


def discover_directory(root: Path) -> Discovery:
    discovery = Discovery()
    metadata, bad_lines = _read_directory_metadata(root)
    discovery.bad_metadata_lines = bad_lines
    all_files = sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))
    image_stems: set[str] = set()
    sidecar_files: list[Path] = []
    for path in all_files:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS or suffix in UNSUPPORTED_IMAGE_EXTS:
            image_stems.add(path.stem)
        elif suffix in SIDECAR_EXTS:
            sidecar_files.append(path)
    for path in all_files:
        suffix = path.suffix.lower()
        if suffix in UNSUPPORTED_IMAGE_EXTS:
            source_rel = _source_rel(path, root)
            discovery.unsupported_images.append(source_rel)
            discovery.skipped.append({"source": source_rel, "reason": "unsupported_extension"})
            continue
        if suffix not in IMAGE_EXTS:
            continue
        row = metadata.get(path.resolve())
        sidecar = _find_sidecar(path)
        if sidecar is not None:
            other = path.with_suffix(".txt" if sidecar.suffix.lower() == ".json" else ".json")
            if other.is_file():
                discovery.both_sidecars.append(_source_rel(path, root))
        discovery.supported += 1
        if row is not None and _has_caption_fields(row):
            payload: Any = row
        elif sidecar is not None:
            try:
                payload = _read_sidecar_payload(sidecar)
            except SidecarError as error:
                discovery.skipped.append(
                    {"source": _source_rel(path, root), "reason": "unparsable_sidecar", "detail": str(error)}
                )
                continue
            if row is not None:
                payload = _merge_row_metadata(payload, row)
        else:
            discovery.skipped.append({"source": _source_rel(path, root), "reason": "missing_sidecar"})
            continue
        discovery.items.append(SourceItem(
            image=path.resolve(),
            source_rel=_source_rel(path, root),
            payload=payload,
            sidecar=sidecar.resolve() if sidecar else None,
        ))
    for path in sidecar_files:
        if path.stem not in image_stems:
            discovery.orphan_sidecars.append(_source_rel(path, root))
    return discovery


def discover_jsonl(path: Path) -> Discovery:
    discovery = Discovery()
    seen: set[Path] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PrepareError(f"{path}: invalid UTF-8 ({error})") from error
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            discovery.skipped.append(
                {"source": f"{path.name}:{line_no}", "reason": "bad_manifest_row", "detail": str(error)}
            )
            continue
        if not isinstance(row, dict):
            discovery.skipped.append(
                {"source": f"{path.name}:{line_no}", "reason": "bad_manifest_row",
                 "detail": "row is not an object"}
            )
            continue
        name = row.get("image") or row.get("file_name")
        if not isinstance(name, str) or not name.strip():
            discovery.skipped.append(
                {"source": f"{path.name}:{line_no}", "reason": "bad_manifest_row",
                 "detail": "missing image/file_name"}
            )
            continue
        image = (path.parent / name).resolve()
        if image in seen:
            discovery.duplicate_manifest_rows += 1
            continue
        seen.add(image)
        if not image.is_file():
            discovery.skipped.append(
                {"source": _source_rel(image, path.parent), "reason": "missing_image"}
            )
            continue
        if image.suffix.lower() not in IMAGE_EXTS:
            source_rel = _source_rel(image, path.parent)
            discovery.unsupported_images.append(source_rel)
            discovery.skipped.append({"source": source_rel, "reason": "unsupported_extension"})
            continue
        discovery.supported += 1
        payload: Any = row
        if not _has_caption_fields(row):
            sidecar = _find_sidecar(image)
            if sidecar is not None:
                try:
                    payload = _read_sidecar_payload(sidecar)
                except SidecarError as error:
                    discovery.skipped.append(
                        {"source": _source_rel(image, path.parent), "reason": "unparsable_sidecar",
                         "detail": str(error)}
                    )
                    continue
                payload = _merge_row_metadata(payload, row)
            else:
                discovery.skipped.append(
                    {"source": _source_rel(image, path.parent), "reason": "missing_caption"}
                )
                continue
        discovery.items.append(SourceItem(
            image=image,
            source_rel=_source_rel(image, path.parent),
            payload=payload,
            sidecar=_find_sidecar(image),
        ))
    discovery.items.sort(key=lambda item: item.source_rel)
    return discovery


def discover(input_path: Path) -> Discovery:
    if not input_path.exists():
        raise PrepareError(f"input not found: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".jsonl":
            raise PrepareError(f"input must be a directory or .jsonl file, got {input_path}")
        return discover_jsonl(input_path)
    if not input_path.is_dir():
        raise PrepareError(f"input is not a directory: {input_path}")
    return discover_directory(input_path)


# --------------------------------------------------------------------------- artist maps / filters


def load_artist_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"cannot read --artist-map {path}: {error}") from error
    if not isinstance(data, dict):
        raise PrepareError("--artist-map must be a JSON object mapping source names to canonical names")
    mapping: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PrepareError("--artist-map keys and values must be strings")
        clean_key = normalize_tag_key(key)
        if clean_key:
            mapping[clean_key] = normalize_name(value) if value.strip() else UNKNOWN_ARTIST
    return mapping


def apply_artist_map(values: Sequence[str], mapping: dict[str, str]) -> list[str]:
    if not mapping:
        return list(values)
    return [mapping.get(normalize_tag_key(value), value) for value in values]


def prepare_item(item: SourceItem, artist_map: dict[str, str]) -> PreparedItem:
    parsed = parse_payload(item.payload)
    artist_values = dedupe_names(apply_artist_map(parsed.artist_values, artist_map))
    artist_missing = not artist_values
    artist = artist_values[0] if artist_values else UNKNOWN_ARTIST
    tags, removed = dedupe_tags(parsed.tags)
    return PreparedItem(
        image=item.image,
        source_rel=item.source_rel,
        parsed=parsed,
        tags=tags,
        removed_negatives=removed,
        artist=artist,
        artist_missing=artist_missing,
        characters=dedupe_names(parsed.characters),
        source_handle=parsed.source_handle,
        prose_en=parsed.prose_en,
        prose_zh=parsed.prose_zh,
        preference_level=parsed.preference_level,
        layout=parsed.layout,
        artist_values=artist_values,
    )


def apply_filters(
    items: list[PreparedItem],
    *,
    min_artist_count: int,
    max_per_artist: int | None,
    require_tags: Sequence[str],
) -> tuple[list[PreparedItem], dict[str, Any]]:
    artist_counts = Counter(item.artist for item in items)
    required_keys = [normalize_tag_key(tag) for tag in require_tags if normalize_tag_key(tag)]
    kept: list[PreparedItem] = []
    dropped: Counter[str] = Counter()
    per_artist: Counter[str] = Counter()
    for item in items:
        if artist_counts[item.artist] < max(1, min_artist_count):
            dropped["min_artist_count"] += 1
            continue
        if max_per_artist is not None and per_artist[item.artist] >= max_per_artist:
            dropped["max_per_artist"] += 1
            continue
        tag_keys = {normalize_tag_key(tag) for tag in item.tags}
        if required_keys and not all(key in tag_keys for key in required_keys):
            dropped["require_tag"] += 1
            continue
        per_artist[item.artist] += 1
        kept.append(item)
    return kept, {
        "artist_counts": artist_counts,
        "dropped": dropped,
        "required_tags": required_keys,
    }


# --------------------------------------------------------------------------- captions


def select_prose(item: PreparedItem, language: str) -> tuple[str, str, str]:
    """Returns (full prose, first paragraph, language_used). Exactly one language."""
    if language == "en":
        blocks = item.prose_en
        used = "en" if blocks else "none"
    elif language == "zh":
        blocks = item.prose_zh
        used = "zh" if blocks else "none"
    elif language == "auto":
        if item.prose_zh:
            blocks, used = item.prose_zh, "zh"
        elif item.prose_en:
            blocks, used = item.prose_en, "en"
        else:
            blocks, used = [], "none"
    else:
        raise ValueError(f"unknown language: {language}")
    if not blocks:
        return "", "", used
    return "\n\n".join(blocks), blocks[0], used


def _prefix(trigger: str, artist: str, characters: Sequence[str], *, include_artist: bool) -> list[str]:
    parts = [trigger]
    if include_artist:
        parts.append(f"artist:{artist}")
    parts.extend(f"character:{name}" for name in characters)
    return parts


def _compose(prefix: Sequence[str], tags: Sequence[str], prose: str) -> str:
    parts = [part for part in prefix if part]
    parts.extend(tags)
    if prose:
        parts.append(prose)
    return ", ".join(parts)


def build_captions(item: PreparedItem, *, trigger: str, language: str,
                   max_tags: int) -> dict[str, Any]:
    full_prose, first_prose, language_used = select_prose(item, language)
    tags = item.tags[: max(0, max_tags)]
    medium_tags = tags[: min(MEDIUM_TAG_LIMIT, max(0, max_tags))]
    short_tags = tags[: min(SHORT_TAG_LIMIT, max(0, max_tags))]
    with_artist = _prefix(trigger, item.artist, item.characters, include_artist=True)
    without_artist = _prefix(trigger, item.artist, item.characters, include_artist=False)
    return {
        "long": _compose(with_artist, tags, full_prose),
        "medium": _compose(with_artist, medium_tags, first_prose),
        "short": _compose(with_artist, short_tags, ""),
        "no_artist": _compose(without_artist, tags, full_prose),
        "language_used": language_used,
        "language_fallback": language in ("en", "zh") and language_used == "none",
        "full_prose": full_prose,
        "first_prose": first_prose,
    }


# --------------------------------------------------------------------------- image/report helpers


def oriented_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def crop_fraction(width: int, height: int, bucket: tuple[int, int]) -> float:
    bucket_w, bucket_h = bucket
    scale = max(bucket_w / width, bucket_h / height)
    new_w = max(bucket_w, round(width * scale))
    new_h = max(bucket_h, round(height * scale))
    return max(0.0, 1.0 - (bucket_w * bucket_h) / (new_w * new_h))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_info(path: Path, buckets: Sequence[tuple[int, int]]) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(274, 1)
    width, height = oriented_size(width, height, orientation)
    bucket = assign_bucket(width, height, list(buckets))
    crop = crop_fraction(width, height, bucket)
    return {
        "width": width,
        "height": height,
        "orientation": orientation,
        "bucket": list(bucket),
        "crop": crop,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


def distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0,
                "mean": 0.0}
    return {
        "count": len(values),
        "min": int(min(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": int(max(values)),
        "mean": round(sum(values) / len(values), 2),
    }


def over_threshold(values: Sequence[int], threshold: int) -> dict[str, Any]:
    count = sum(1 for value in values if value > threshold)
    return {"count": count, "pct": round(100.0 * count / max(1, len(values)), 2)}


def estimate_tokens(text: str) -> int:
    """Fallback when the model tokenizer is unavailable: CJK chars + word estimate."""
    cjk = count_cjk(text)
    words = len(re.findall(r"[A-Za-z0-9_]+", text))
    if cjk:
        return max(1, cjk + words)
    return max(1, words, math.ceil(len(text) / 4))


def load_tokenizer(tokenizer: str | Path | None) -> tuple[Any | None, str | None, str]:
    """Returns (tokenizer, resolved path, error). Path None auto-detects known locals."""
    if isinstance(tokenizer, str) and tokenizer.lower() in ("none", "off", "estimate", ""):
        return None, None, ""
    candidates: list[Path] = []
    if tokenizer is not None:
        candidates.append(Path(tokenizer))
    else:
        env = os.environ.get("MOSS_IMAGE_TOKENIZER")
        if env:
            candidates.append(Path(env))
        candidates.append(_REPO_ROOT / "models/MOSS-Image-Base/tokenizer")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None, None, "tokenizer path not found" if tokenizer is not None else ""
    try:
        from transformers import AutoTokenizer

        loaded = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        return loaded, str(path), ""
    except Exception as error:  # optional dependency / incompatible tokenizer
        return None, str(path), f"{type(error).__name__}: {error}"


def token_length(tokenizer: Any | None, text: str) -> int:
    if tokenizer is None:
        return estimate_tokens(text)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def _tag_hits(caption: dict[str, Any]) -> set[str]:
    keys = {normalize_tag_key(tag) for tag in caption.get("tags", [])}
    return {term for term in TEXT_HAND_TERMS if term in keys}


def _prose_hits(caption: dict[str, Any]) -> set[str]:
    prose = caption.get("prose", "")
    hits: set[str] = set()
    for term in TEXT_HAND_TERMS:
        if re.search(rf"(?i)\b{re.escape(term)}\b", prose):
            hits.add(term)
        elif any(zh in prose for zh in ZH_COVERAGE_TERMS[term]):
            hits.add(term)
    return hits


def _coverage_from_hits(hits: Sequence[set[str]]) -> dict[str, Any]:
    total = len(hits)
    counts = Counter(term for item_hits in hits for term in item_hits)
    text_group = sum(1 for item_hits in hits if item_hits & TEXT_GROUP)
    hand_group = sum(1 for item_hits in hits if item_hits & HAND_GROUP)
    return {
        "total": total,
        "terms": {
            term: {"count": counts[term], "pct": round(100.0 * counts[term] / max(1, total), 2)}
            for term in TEXT_HAND_TERMS
        },
        "text_group": {
            "count": text_group,
            "pct": round(100.0 * text_group / max(1, total), 2),
        },
        "hand_group": {
            "count": hand_group,
            "pct": round(100.0 * hand_group / max(1, total), 2),
        },
    }


def coverage_report(captions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tag_hits = [_tag_hits(caption) for caption in captions]
    prose_hits = [_prose_hits(caption) for caption in captions]
    any_hits = [tags | prose for tags, prose in zip(tag_hits, prose_hits)]
    return {
        "tags": _coverage_from_hits(tag_hits),
        "prose": _coverage_from_hits(prose_hits),
        "any": _coverage_from_hits(any_hits),
    }


def duplicate_stats(hashes: Sequence[str]) -> dict[str, Any]:
    counts = Counter(hashes)
    groups = [count for count in counts.values() if count > 1]
    return {
        "duplicate_groups": len(groups),
        "items_in_duplicate_groups": sum(groups),
        "max_group": max(groups) if groups else 0,
        "top_groups": [
            {"sha256": digest[:16], "count": count}
            for digest, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
            if count > 1
        ],
    }


def long_tail(counts: Counter[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for low, high in LONG_TAIL_BUCKETS:
        if high is None:
            result[f"count_{low}_plus"] = sum(1 for value in counts.values() if value >= low)
        elif low == high:
            result[f"count_{low}"] = sum(1 for value in counts.values() if value == low)
        else:
            result[f"count_{low}_to_{high}"] = sum(
                1 for value in counts.values() if low <= value <= high
            )
    return result


# --------------------------------------------------------------------------- output helpers


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    fd, temp_name = tempfile.mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.",
                                     suffix=".tmp")
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copy2(source, temp)
        os.replace(temp, destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def materialize_image(source: Path, destination: Path, *, copy_images: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        _atomic_copy(source, destination)
        return "copy"
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    target = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    try:
        os.symlink(target, temporary)
    except (OSError, NotImplementedError):
        _atomic_copy(source, destination)
        return "copy"
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "symlink"


def output_size(path: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    symlinks = 0
    linked_bytes = 0
    if not path.exists():
        return {"files": 0, "bytes": 0, "symlinks": 0, "linked_image_bytes": 0}
    for entry in path.rglob("*"):
        if not (entry.is_file() or entry.is_symlink()):
            continue
        stat = entry.lstat()
        files += 1
        bytes_total += stat.st_size
        if entry.is_symlink():
            symlinks += 1
            try:
                linked_bytes += entry.resolve().stat().st_size
            except OSError:
                pass
    return {"files": files, "bytes": bytes_total, "symlinks": symlinks,
            "linked_image_bytes": linked_bytes}


# --------------------------------------------------------------------------- report


def render_report_markdown(report: dict[str, Any]) -> str:
    images = report["images"]
    tokens = report["tokens"]
    crop = report["crop_risk"]
    artists = report["artists"]
    lines = [
        "# Caption preprocessing report",
        "",
        f"- Input: `{report['input']}`",
        f"- Output: `{report['output']}`",
        f"- Trigger: `{report['trigger']}`",
        f"- Language requested: `{report['language']['requested']}`",
        f"- Items written: **{images['written']}**",
        "",
        "## Images and captions",
        "",
        "| metric | count |",
        "| --- | ---: |",
        f"| supported images discovered | {images['supported']} |",
        f"| items written | {images['written']} |",
        f"| missing sidecar | {images['missing_sidecar']} |",
        f"| unparsable sidecar | {images['unparsable_sidecar']} |",
        f"| unsupported image extension | {images['unsupported_extension']} |",
        f"| unreadable image | {images['unreadable_image']} |",
        f"| filtered by min artist count | {images['filtered']['min_artist_count']} |",
        f"| filtered by max per artist | {images['filtered']['max_per_artist']} |",
        f"| filtered by required tag | {images['filtered']['require_tag']} |",
        "",
        "## Artists",
        "",
        f"- Unique artists: **{artists['unique']}**",
        f"- Items without an `artist:` tag: **{artists['missing_artist_items']}** "
        f"(captioned as `artist:{UNKNOWN_ARTIST}`)",
        f"- Long tail: `{json.dumps(artists['long_tail'], sort_keys=True)}`",
        f"- Rare artists (fewer than {artists['rare_threshold']} images): "
        f"**{artists['rare_artist_count']}**",
        "",
        "Top artists:",
        "",
        "| artist | images |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in artists["top"])
    lines.extend([
        "",
        "## Tags and token lengths",
        "",
        "| variant | p50 | p95 | max | >1024 | >2048 | >4096 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in ("long", "medium", "short"):
        stats = tokens[name]
        lines.append(
            f"| {name} | {stats['p50']} | {stats['p95']} | {stats['max']} | "
            f"{stats['over_1024']['pct']}% | {stats['over_2048']['pct']}% | "
            f"{stats['over_4096']['pct']}% |"
        )
    lines.extend([
        "",
        f"- Tokenizer: `{tokens['mode']}`" + (f" (`{tokens['path']}`)" if tokens.get("path") else ""),
        f"- Tags per item (before cap): `{json.dumps(report['tags']['per_item'], sort_keys=True)}`",
        "",
        "## Text / hand coverage",
        "",
        "| term | tag % | prose % | any % |",
        "| --- | ---: | ---: | ---: |",
    ])
    for term in TEXT_HAND_TERMS:
        lines.append(
            f"| `{term}` | {report['text_hand_coverage']['tags']['terms'][term]['pct']} | "
            f"{report['text_hand_coverage']['prose']['terms'][term]['pct']} | "
            f"{report['text_hand_coverage']['any']['terms'][term]['pct']} |"
        )
    lines.extend([
        "",
        f"- Any text group: **{report['text_hand_coverage']['any']['text_group']['pct']}%**",
        f"- Any hand group: **{report['text_hand_coverage']['any']['hand_group']['pct']}%**",
        "",
        "## Crop risk",
        "",
        f"- Target area: {crop['target_area']} px²",
        f"- Items needing >5% center-crop area: **{crop['over_5pct']} / {crop['items']}**",
        f"- Crop % p50 / p95 / max: {crop['p50_crop_pct']} / {crop['p95_crop_pct']} / "
        f"{crop['max_crop_pct']}",
        "",
        "| source | size | bucket | crop % |",
        "| --- | --- | --- | ---: |",
    ])
    lines.extend(
        f"| `{row['source']}` | {row['width']}x{row['height']} | "
        f"{row['bucket'][0]}x{row['bucket'][1]} | {row['crop_pct']} |"
        for row in crop["worst"]
    )
    lines.extend([
        "",
        "## Duplicates and skips",
        "",
        f"- Duplicate image hashes: {report['duplicates']['images']['duplicate_groups']} groups, "
        f"{report['duplicates']['images']['items_in_duplicate_groups']} items",
        f"- Duplicate long-caption hashes: {report['duplicates']['captions']['duplicate_groups']} groups, "
        f"{report['duplicates']['captions']['items_in_duplicate_groups']} items",
        f"- Skipped entries: {len(report['skipped'])}",
        "",
    ])
    if report["skipped"]:
        lines.extend(
            f"- `{entry['source']}`: {entry['reason']}"
            + (f" ({entry['detail']})" if entry.get("detail") else "")
            for entry in report["skipped"][:100]
        )
        if len(report["skipped"]) > 100:
            lines.append(f"- ... {len(report['skipped']) - 100} more")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main pipeline


@dataclass
class PrepareConfig:
    trigger: str = TRIGGER_DEFAULT
    language: str = "en"
    max_tags: int = 40
    min_artist_count: int = 1
    max_per_artist: int | None = None
    require_tags: tuple[str, ...] = ()
    copy_images: bool = False
    artist_map: dict[str, str] = field(default_factory=dict)
    tokenizer: str | Path | None = None
    target_area: int = 1024 * 1024
    force: bool = False


def prepare_dataset(input_path: str | Path, output_dir: str | Path,
                    config: PrepareConfig | None = None) -> dict[str, Any]:
    config = config or PrepareConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if config.language not in ("en", "zh", "auto"):
        raise PrepareError(f"language must be en, zh or auto, got {config.language!r}")
    if config.max_tags < 0:
        raise PrepareError("--max-tags must be >= 0")
    if config.min_artist_count < 1:
        raise PrepareError("--min-artist-count must be >= 1")
    if config.max_per_artist is not None and config.max_per_artist < 1:
        raise PrepareError("--max-per-artist must be >= 1")

    input_resolved = input_path.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise PrepareError("input and output must differ")
    if input_path.is_dir() and output_resolved.is_relative_to(input_resolved):
        raise PrepareError("output must not be inside the input directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise PrepareError(f"output path exists and is not a directory: {output_dir}")
    if config.force and output_dir.exists():
        shutil.rmtree(output_dir / "images", ignore_errors=True)
        for name in ("metadata.jsonl", "report.json", "report.md"):
            (output_dir / name).unlink(missing_ok=True)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise PrepareError(f"output directory is not empty: {output_dir} (pass --force to replace)")

    discovery = discover(input_path)
    skipped = list(discovery.skipped)
    prepared: list[PreparedItem] = []
    for item in discovery.items:
        try:
            prepared.append(prepare_item(item, config.artist_map))
        except SidecarError as error:
            skipped.append({"source": item.source_rel, "reason": "unparsable_sidecar",
                            "detail": str(error)})

    kept, filter_report = apply_filters(
        prepared,
        min_artist_count=config.min_artist_count,
        max_per_artist=config.max_per_artist,
        require_tags=config.require_tags,
    )
    artist_counts: Counter[str] = filter_report["artist_counts"]
    dropped: Counter[str] = filter_report["dropped"]

    buckets = build_buckets(config.target_area)
    tokenizer, tokenizer_path, tokenizer_error = load_tokenizer(config.tokenizer)
    trig = config.trigger.strip() or TRIGGER_DEFAULT

    records: list[dict[str, Any]] = []
    caption_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    image_hashes: list[str] = []
    caption_hashes: list[str] = []
    token_values: dict[str, list[int]] = {"long": [], "medium": [], "short": []}
    language_counts: Counter[str] = Counter()
    removed_negatives: Counter[str] = Counter()
    tag_counts: list[int] = []
    output_modes: Counter[str] = Counter()
    linked_bytes = 0
    unreadable = 0
    copy_failed = 0
    index = 0

    for item in kept:
        try:
            info = image_info(item.image, buckets)
        except Exception as error:
            unreadable += 1
            skipped.append({"source": item.source_rel, "reason": "unreadable_image",
                            "detail": f"{type(error).__name__}: {error}"})
            continue
        captions = build_captions(item, trigger=trig, language=config.language,
                                  max_tags=config.max_tags)
        index += 1
        suffix = item.image.suffix.lower()
        file_name = f"images/{index:06d}{suffix}"
        destination = output_dir / file_name
        try:
            mode = materialize_image(item.image, destination, copy_images=config.copy_images)
        except Exception as error:
            copy_failed += 1
            skipped.append({"source": item.source_rel, "reason": "image_write_failed",
                            "detail": f"{type(error).__name__}: {error}"})
            index -= 1
            continue
        output_modes[mode] += 1
        linked_bytes += info["bytes"]
        image_hashes.append(info["sha256"])
        caption_hashes.append(sha256_text(captions["long"]))
        removed_negatives.update(item.removed_negatives)
        tag_counts.append(len(item.tags))
        language_counts[captions["language_used"]] += 1
        for name in token_values:
            token_values[name].append(token_length(tokenizer, captions[name]))

        record: dict[str, Any] = {
            "file_name": file_name,
            "caption": {name: captions[name] for name in ("long", "medium", "short")},
            "artist": item.artist,
            "character": item.characters,
            "tags": item.tags,
            "source_handle": item.source_handle,
            "preference_level": item.preference_level,
            "caption_no_artist": captions["no_artist"],
        }
        if item.layout is not None:
            record["layout"] = item.layout
        records.append(record)
        caption_rows.append({
            "source": item.source_rel,
            "tags": item.tags,
            "prose": captions["full_prose"],
            "language": captions["language_used"],
        })
        crop_rows.append({
            "source": item.source_rel,
            "file_name": file_name,
            "width": info["width"],
            "height": info["height"],
            "bucket": info["bucket"],
            "crop": info["crop"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_text(output_dir / "metadata.jsonl", metadata_text)

    crop_rows.sort(key=lambda row: (-row["crop"], row["source"]))
    crop_values = [row["crop"] for row in crop_rows]
    token_report: dict[str, Any] = {"mode": "estimate" if tokenizer is None else "model",
                                    "path": tokenizer_path}
    if tokenizer_error:
        token_report["tokenizer_error"] = tokenizer_error
    for name, values in token_values.items():
        token_report[name] = {
            **distribution(values),
            "over_512": over_threshold(values, 512),
            "over_1024": over_threshold(values, 1024),
            "over_2048": over_threshold(values, 2048),
            "over_4096": over_threshold(values, 4096),
        }

    coverage = coverage_report(caption_rows)

    real_artist_counts = Counter({
        name: count for name, count in artist_counts.items() if name != UNKNOWN_ARTIST
    })
    top_artists = [
        [name, count]
        for name, count in sorted(
            real_artist_counts.items(), key=lambda pair: (-pair[1], pair[0])
        )[:20]
    ]
    rare_threshold = max(2, config.min_artist_count)
    rare_artists = [
        [name, count]
        for name, count in sorted(
            ((name, count) for name, count in real_artist_counts.items()
             if count < rare_threshold),
            key=lambda pair: (pair[1], pair[0]),
        )
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "trigger": trig,
        "language": {
            "requested": config.language,
            "used": dict(sorted(language_counts.items())),
            "fallback_to_no_prose": language_counts.get("none", 0),
        },
        "images": {
            "supported": discovery.supported,
            "written": len(records),
            "missing_sidecar": sum(1 for entry in skipped if entry["reason"] == "missing_sidecar"),
            "missing_caption": sum(1 for entry in skipped if entry["reason"] == "missing_caption"),
            "unparsable_sidecar": sum(
                1 for entry in skipped if entry["reason"] == "unparsable_sidecar"
            ),
            "unsupported_extension": len(discovery.unsupported_images),
            "unreadable_image": unreadable,
            "image_write_failed": copy_failed,
            "filtered": {
                "min_artist_count": dropped["min_artist_count"],
                "max_per_artist": dropped["max_per_artist"],
                "require_tag": dropped["require_tag"],
            },
            "output_modes": dict(sorted(output_modes.items())),
        },
        "artists": {
            "unique": len(real_artist_counts),
            "missing_artist_items": sum(1 for item in kept if item.artist_missing),
            "top": top_artists,
            "long_tail": long_tail(real_artist_counts),
            "rare_threshold": rare_threshold,
            "rare_artist_count": len(rare_artists),
            "rare": rare_artists[:50],
            "min_artist_count": config.min_artist_count,
            "rare_artist_items": sum(
                1 for item in prepared
                if item.artist != UNKNOWN_ARTIST
                and artist_counts[item.artist] < config.min_artist_count
            ),
            "max_per_artist": config.max_per_artist,
            "artist_map_entries": len(config.artist_map),
        },
        "characters": {
            "unique": len({normalize_tag_key(name) for item in kept for name in item.characters}),
            "items_without_character": sum(1 for item in kept if not item.characters),
            "top": [
                [name, count]
                for name, count in sorted(
                    Counter(name for item in kept for name in item.characters).items(),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:20]
            ],
        },
        "captions": {
            "items": len(records),
            "variants": {
                "long": len(records),
                "medium": len(records),
                "short": len(records),
            },
            "with_prose": sum(1 for row in caption_rows if row.get("prose")),
            "without_prose": sum(1 for row in caption_rows if not row.get("prose")),
        },
        "tags": {
            "items": len(records),
            "total": sum(tag_counts),
            "per_item": distribution(tag_counts),
            "removed_negatives": dict(sorted(removed_negatives.items())),
            "items_over_max_tags": sum(1 for count in tag_counts if count > config.max_tags),
            "max_tags": config.max_tags,
        },
        "tokens": token_report,
        "text_hand_coverage": coverage,
        "duplicates": {
            "images": duplicate_stats(image_hashes),
            "captions": duplicate_stats(caption_hashes),
        },
        "crop_risk": {
            "target_area": config.target_area,
            "buckets": [list(bucket) for bucket in buckets],
            "items": len(crop_rows),
            "over_5pct": sum(1 for value in crop_values if value > 0.05),
            "p50_crop_pct": round(
                100.0 * (percentile([int(v * 10000) for v in crop_values], 0.50) / 10000), 2),
            "p95_crop_pct": round(
                100.0 * (percentile([int(v * 10000) for v in crop_values], 0.95) / 10000), 2),
            "max_crop_pct": round(100.0 * (max(crop_values) if crop_values else 0.0), 2),
            "worst": [
                {
                    "source": row["source"],
                    "file_name": row["file_name"],
                    "width": row["width"],
                    "height": row["height"],
                    "bucket": row["bucket"],
                    "crop_pct": round(100.0 * row["crop"], 2),
                }
                for row in crop_rows[:10]
            ],
        },
        "skipped": skipped,
        "sidecars": {
            "both_json_and_txt": len(discovery.both_sidecars),
            "orphans": len(discovery.orphan_sidecars),
            "bad_metadata_lines": len(discovery.bad_metadata_lines),
            "duplicate_manifest_rows": discovery.duplicate_manifest_rows,
        },
        "output_stats": {
            **output_size(output_dir),
            "linked_source_image_bytes": linked_bytes,
        },
    }
    _atomic_write_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_dir / "report.md", render_report_markdown(report))
    report["output_stats"] = {**output_size(output_dir), "linked_source_image_bytes": linked_bytes}
    _atomic_write_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if not records:
        raise PrepareError(
            f"no usable items written to {output_dir}; see report.json (skipped: {len(skipped)})"
        )
    return report


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_captions",
        description="Prepare mixed-artist long/medium/short captions for MOSS-Image LoRA training.",
    )
    parser.add_argument("--input", required=True,
                        help="Image directory with .txt/.json sidecars (or metadata.jsonl), "
                             "or a .jsonl manifest")
    parser.add_argument("--output", required=True,
                        help="Output directory for images/, metadata.jsonl and reports")
    parser.add_argument("--trigger", default=TRIGGER_DEFAULT,
                        help=f"Trigger token prepended to every caption (default: {TRIGGER_DEFAULT})")
    parser.add_argument("--language", default="en", choices=["en", "zh", "auto"],
                        help="Prose language: en (default), zh, or auto (zh when Chinese prose exists)")
    parser.add_argument("--max-tags", type=int, default=40,
                        help="Maximum tags in the long caption (default: 40)")
    parser.add_argument("--min-artist-count", type=int, default=1,
                        help="Drop images whose artist has fewer than N images (default: 1 = keep all)")
    parser.add_argument("--max-per-artist", type=int, default=None,
                        help="Keep at most N images per artist (deterministic source order)")
    parser.add_argument("--require-tag", action="append", default=[],
                        help="Only keep items containing this tag (repeatable; all required)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of relative symlinks")
    parser.add_argument("--artist-map", default=None,
                        help="JSON object mapping raw artist names to canonical names")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer directory for report token counts "
                             "(default: auto-detect; 'none' disables)")
    parser.add_argument("--target-area", type=int, default=1024 * 1024,
                        help="Cache bucket target area for crop-risk reporting (default: 1024^2)")
    parser.add_argument("--force", action="store_true",
                        help="Replace generated files in a non-empty output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PrepareConfig(
        trigger=args.trigger,
        language=args.language,
        max_tags=args.max_tags,
        min_artist_count=args.min_artist_count,
        max_per_artist=args.max_per_artist,
        require_tags=tuple(args.require_tag),
        copy_images=args.copy_images,
        artist_map=load_artist_map(Path(args.artist_map) if args.artist_map else None),
        tokenizer=args.tokenizer,
        target_area=args.target_area,
        force=args.force,
    )
    try:
        report = prepare_dataset(Path(args.input), Path(args.output), config)
    except (PrepareError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {report['images']['written']} captions to {report['output']} "
        f"({report['images']['filtered']['min_artist_count']} min-artist drops, "
        f"{report['images']['filtered']['max_per_artist']} max-per-artist drops, "
        f"{report['crop_risk']['over_5pct']} images >5% crop)"
    )
    if report["language"]["fallback_to_no_prose"]:
        print(
            f"warning: {report['language']['fallback_to_no_prose']} items had no prose in "
            f"requested language {args.language!r}",
            file=sys.stderr,
        )
    if report["tokens"]["mode"] == "estimate":
        print("warning: model tokenizer unavailable; token lengths are estimates", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
