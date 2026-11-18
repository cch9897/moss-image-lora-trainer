"""Expansion-pool staging: filtering, artist resolution, dedupe and SQLite index.

The staging stage deliberately does not caption anything. It validates images,
rejects GIF/video/unsupported or broken rows, resolves artists from the
artist-directory layout, and writes one metadata row per accepted image plus
reject/dedupe reports and a resumable SQLite index.

No full-pool scan happens in tests or on import; the CLI is the only entry point
that walks a real pool.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from moss_lora.annotations import UNKNOWN_ARTIST, normalize_name, normalize_tag_key
from moss_lora.utils import atomic_write_json

IMAGE_EXTS = frozenset((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
GIF_EXTS = frozenset((".gif",))
VIDEO_EXTS = frozenset((".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv"))
UNSUPPORTED_EXTS = frozenset((".tif", ".tiff", ".avif", ".heic", ".heif"))
SIDECAR_EXTS = frozenset((".txt", ".json"))
UNKNOWN_ARTIST_DIRS = frozenset(("", "unknown", "misc", "others", "_unknown"))

STAGING_VERSION = "v2"
METADATA_NAME = "metadata_staging.jsonl"
REJECTS_NAME = "rejects.jsonl"
DEDUPE_REPORT_NAME = "dedupe_report.json"
INDEX_DIR = "staging"
INDEX_NAME = "index.sqlite"
ARTIST_MAP_NAME = "artist_map.json"
QUARANTINE_DIR = "quarantine"
IMAGES_DIR = "images"
EMBEDDING_FULL_SCAN_LIMIT = 5000


class StagingError(RuntimeError):
    """Invalid staging configuration or an unwritable output."""


class RejectedImage(ValueError):
    """Image failed a hard staging gate; carries the report reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------- config


@dataclass
class StagingConfig:
    input_root: Path
    output_dir: Path
    min_side: int = 512
    max_side: int = 12000
    max_megapixels: float = 80.0
    min_aspect: float = 0.25
    max_aspect: float = 4.0
    blank_entropy: float = 3.5
    blank_std: float = 8.0
    phash_duplicate: int = 2
    phash_near: int = 6
    phash_review: int = 10
    embedding_duplicate: float = 0.985
    embedding_near: float = 0.96
    embedding_review: float = 0.92
    near_crop_aspect_delta: float = 0.10
    link_mode: str = "symlink"  # symlink | copy | none
    resume: bool = False

    def __post_init__(self) -> None:
        self.input_root = Path(self.input_root)
        self.output_dir = Path(self.output_dir)
        if not self.input_root.is_dir():
            raise StagingError(f"Input pool is not a directory: {self.input_root}")
        if self.min_side < 1 or self.max_side < self.min_side:
            raise StagingError("min_side must be >= 1 and <= max_side")
        if self.max_megapixels <= 0:
            raise StagingError("max_megapixels must be positive")
        if not (0 < self.min_aspect <= 1 <= self.max_aspect):
            raise StagingError("aspect bounds must satisfy 0 < min_aspect <= 1 <= max_aspect")
        if self.link_mode not in ("symlink", "copy", "none"):
            raise StagingError(f"link_mode must be symlink/copy/none, got {self.link_mode!r}")
        limits = (self.phash_duplicate, self.phash_near, self.phash_review)
        if not (0 <= limits[0] <= limits[1] <= limits[2] <= 64):
            raise StagingError("pHash thresholds must satisfy 0 <= duplicate <= near <= review <= 64")


# --------------------------------------------------------------------------- scan / classification


def classify_suffix(path: Path) -> str:
    """Return ``image``, ``gif``, ``video``, ``unsupported``, ``sidecar`` or ``other``."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in GIF_EXTS:
        return "gif"
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in UNSUPPORTED_EXTS:
        return "unsupported"
    if suffix in SIDECAR_EXTS:
        return "sidecar"
    return "other"


def _is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def iter_pool(root: str | Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, relative_posix)`` in deterministic shortest-path-first order.

    Hidden files/directories are skipped. Sorting by ``(len(relative), relative)``
    makes the first exact-duplicate encountered the deterministic winner.
    """
    root = Path(root)
    entries: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_hidden(relative):
            continue
        entries.append((path, relative.as_posix()))
    entries.sort(key=lambda pair: (len(pair[1]), pair[1]))
    yield from entries


# --------------------------------------------------------------------------- artists


@dataclass(frozen=True)
class ArtistResolution:
    raw: str
    artist: str
    artists: tuple[str, ...]
    missing: bool
    ambiguous: bool


def load_artist_map(path: str | Path | None) -> dict[str, str]:
    """JSON object mapping raw directory names/aliases to canonical artist IDs."""
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StagingError("artist map must be a JSON object")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise StagingError("artist map keys and values must be strings")
        clean_key = normalize_tag_key(key)
        if not clean_key:
            continue
        mapping[clean_key] = normalize_name(value) or UNKNOWN_ARTIST
    return mapping


def load_known_artists(metadata_path: str | Path | None) -> set[str]:
    """Collect canonical artist keys from a metadata JSONL (``artist``/``artists``)."""
    if metadata_path is None:
        return set()
    path = Path(metadata_path)
    if not path.is_file():
        return set()
    known: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        values: list[Any] = []
        artist = row.get("artist")
        if isinstance(artist, str) and artist.strip():
            values.append(artist)
        artists = row.get("artists")
        if isinstance(artists, list):
            values.extend(item for item in artists if isinstance(item, str))
        for value in values:
            clean = normalize_name(value)
            if clean and clean != UNKNOWN_ARTIST:
                known.add(normalize_tag_key(clean))
    return known


def resolve_artist(
    relative: str | Path,
    *,
    artist_map: dict[str, str] | None = None,
    known_artists: Iterable[str] = (),
) -> ArtistResolution:
    """Resolve the first path component to canonical artist IDs, directory-first."""
    parts = Path(relative).parts
    raw = normalize_name(parts[0]) if parts else ""
    if not raw or normalize_tag_key(raw) in UNKNOWN_ARTIST_DIRS:
        return ArtistResolution(raw=raw, artist=UNKNOWN_ARTIST, artists=(), missing=True, ambiguous=False)

    known = {normalize_tag_key(value) for value in known_artists}
    mapping = artist_map or {}
    candidates = [normalize_name(part) for part in _split_multi_artist(raw)]
    mapped = [mapping.get(normalize_tag_key(candidate), candidate) for candidate in candidates]
    ambiguous = False
    if len(candidates) > 1:
        if not all(normalize_tag_key(candidate) in known for candidate in mapped):
            ambiguous = True
            mapped = [raw]
    resolved = dedupe_artist_names(mapped)
    if not resolved:
        return ArtistResolution(raw=raw, artist=UNKNOWN_ARTIST, artists=(), missing=True,
                                ambiguous=ambiguous)
    return ArtistResolution(raw=raw, artist=resolved[0], artists=tuple(resolved),
                            missing=False, ambiguous=ambiguous)


def _split_multi_artist(raw: str) -> list[str]:
    """Split only on explicit multi-artist separators; caller validates against known IDs."""
    return [part.strip(" _") for part in re.split(r"\s*(?:\+|&|,|_x_|\band\b)\s*", raw)
            if part.strip(" _")]


def dedupe_artist_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_name(value)
        key = normalize_tag_key(clean)
        if clean and key and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


# --------------------------------------------------------------------------- image validation


@dataclass
class ImageInfo:
    width: int
    height: int
    sha256: str
    phash: str
    alpha_composited: bool = False
    image: Image.Image | None = field(default=None, repr=False, compare=False)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def phash64(image: Image.Image) -> str:
    """64-bit dHash (PIL + numpy); hex string, comparable by XOR popcount.

    Spec allows a local dHash implementation instead of the optional imagehash
    dependency. Decode only from a thumbnail to keep validation streaming.
    """
    gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash_hamming(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 64


def inspect_image(
    path: str | Path,
    config: StagingConfig,
    *,
    keep_image: bool = False,
) -> ImageInfo:
    """Validate one image with a single decode and return its hash/metadata.

    Raises ``RejectedImage`` with the report reason for every hard gate.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RejectedImage("unreadable", str(error)) from error
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            if width <= 0 or height <= 0:
                raise RejectedImage("unreadable", "zero-size image")
            if width * height > config.max_megapixels * 1_000_000:
                raise RejectedImage("too_large", f"{width}x{height} exceeds {config.max_megapixels} MP")
            alpha = False
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                alpha = True
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                rgb = background
            else:
                rgb = image.convert("RGB")
            rgb.load()
            phash = phash64(rgb)
            if min(width, height) < config.min_side:
                raise RejectedImage("too_small", f"shortest side {min(width, height)} < {config.min_side}")
            if max(width, height) > config.max_side:
                raise RejectedImage("too_large", f"longest side {max(width, height)} > {config.max_side}")
            aspect = width / height
            if aspect < config.min_aspect or aspect > config.max_aspect:
                raise RejectedImage("bad_aspect", f"aspect {aspect:.3f} outside "
                                                   f"[{config.min_aspect}, {config.max_aspect}]")
            _check_not_blank(rgb, config)
            result_image = rgb if keep_image else None
            if not keep_image:
                rgb.close()
            return ImageInfo(width=width, height=height, sha256=sha256, phash=phash,
                             alpha_composited=alpha, image=result_image)
    except RejectedImage:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as error:
        raise RejectedImage("unreadable", f"{type(error).__name__}: {error}") from error


def _check_not_blank(image: Image.Image, config: StagingConfig) -> None:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / max(1.0, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    std = float(gray.std())
    if entropy < config.blank_entropy or std < config.blank_std:
        raise RejectedImage(
            "blank", f"entropy={entropy:.2f} (min {config.blank_entropy}), "
                     f"luminance_std={std:.2f} (min {config.blank_std})")


# --------------------------------------------------------------------------- current corpus hashes


@dataclass(frozen=True)
class HashRecord:
    sha256: str
    phash: str
    aspect: float
    path: str


def load_current_records(source: str | Path | None, *, progress: bool = False) -> list[HashRecord]:
    """Load SHA-256/pHash/aspect for the current corpus.

    ``source`` may be a metadata JSONL (``file_name``/``image`` resolved next to
    the file) or an image directory. Missing files are skipped; previously
    computed ``sha256``/``phash``/width/height columns are reused when present.
    """
    if source is None:
        return []
    source = Path(source)
    records: list[HashRecord] = []
    if source.is_dir():
        paths = [path for path in sorted(source.rglob("*")) if path.is_file()
                 and path.suffix.lower() in IMAGE_EXTS]
        rows = [{"file_name": str(path)} for path in paths]
        base = Path("/")
    else:
        rows = []
        base = source.parent
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
    iterator: Iterable[dict[str, Any]] = rows
    if progress:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="current hashes", unit="image")
    for row in iterator:
        name = row.get("file_name") or row.get("image")
        if not isinstance(name, str) or not name:
            continue
        path = (base / name).resolve() if not Path(name).is_absolute() else Path(name)
        if not path.is_file():
            continue
        sha = row.get("sha256")
        phash = row.get("phash")
        width, height = row.get("width"), row.get("height")
        if not (isinstance(sha, str) and len(sha) == 64 and isinstance(phash, str) and width and height):
            try:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    width, height = image.size
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                    phash = phash64(image)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
                continue
        records.append(HashRecord(sha256=str(sha), phash=str(phash),
                                  aspect=float(width) / float(height), path=str(path)))
    return records


# --------------------------------------------------------------------------- dedupe


@dataclass(frozen=True)
class DedupeEntry:
    sha256: str
    phash: str
    aspect: float
    path: str
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DedupeResult:
    status: str  # new | exact_dup | near_dup
    reason: str | None = None
    nearest: str | None = None
    phash_distance: int | None = None
    embedding_cosine: float | None = None
    review: bool = False
    nearest_aspect: float | None = None


class _BKNode:
    __slots__ = ("entry", "children")

    def __init__(self, entry: DedupeEntry):
        self.entry = entry
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    """Hamming-distance BK-tree over 64-bit pHashes.

    Exact-duplicate detection is a dict lookup; near-duplicate search stays
    sublinear instead of comparing every accepted row on a 35k-image pool.
    """

    def __init__(self) -> None:
        self.root: _BKNode | None = None
        self.size = 0

    @staticmethod
    def _value(entry: DedupeEntry) -> int:
        try:
            return int(entry.phash, 16)
        except ValueError:
            return 0

    def add(self, entry: DedupeEntry) -> None:
        value = self._value(entry)
        self.size += 1
        if self.root is None:
            self.root = _BKNode(entry)
            return
        node = self.root
        while True:
            distance = (value ^ self._value(node.entry)).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(entry)
                return
            node = child

    def search(self, value: int, max_distance: int) -> list[tuple[int, DedupeEntry]]:
        if self.root is None:
            return []
        found: list[tuple[int, DedupeEntry]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ self._value(node.entry)).bit_count()
            if distance <= max_distance:
                found.append((distance, node.entry))
            low, high = distance - max_distance, distance + max_distance
            stack.extend(child for child_distance, child in node.children.items()
                         if low <= child_distance <= high)
        return found


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
    norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if norm_left <= 0 or norm_right <= 0:
        return None
    return dot / (norm_left * norm_right)


class DedupeIndex:
    """Exact + pHash/SigLIP near-duplicate bookkeeping across current and expansion corpora."""

    def __init__(
        self,
        current: Iterable[HashRecord | DedupeEntry] = (),
        *,
        phash_duplicate: int = 2,
        phash_near: int = 6,
        phash_review: int = 10,
        embedding_duplicate: float = 0.985,
        embedding_near: float = 0.96,
        embedding_review: float = 0.92,
        near_crop_aspect_delta: float = 0.10,
    ):
        self.phash_duplicate = phash_duplicate
        self.phash_near = phash_near
        self.phash_review = phash_review
        self.embedding_duplicate = embedding_duplicate
        self.embedding_near = embedding_near
        self.embedding_review = embedding_review
        self.near_crop_aspect_delta = near_crop_aspect_delta
        self.by_sha: dict[str, DedupeEntry] = {}
        self.entries: list[DedupeEntry] = []
        self.tree = _BKTree()
        for record in current:
            entry = record if isinstance(record, DedupeEntry) else DedupeEntry(
                sha256=record.sha256, phash=record.phash, aspect=record.aspect, path=record.path)
            self.add_entry(entry)

    @property
    def embedding_enabled(self) -> bool:
        return any(entry.embedding is not None for entry in self.entries)

    def add_entry(self, entry: DedupeEntry) -> None:
        existing = self.by_sha.get(entry.sha256)
        if existing is None:
            self.by_sha[entry.sha256] = entry
        self.entries.append(entry)
        self.tree.add(entry)

    def add(self, info: ImageInfo, path: str, embedding: Sequence[float] | None = None) -> None:
        self.add_entry(DedupeEntry(
            sha256=info.sha256, phash=info.phash, aspect=info.aspect, path=path,
            embedding=tuple(float(value) for value in embedding) if embedding is not None else None,
        ))

    def check(
        self,
        info: ImageInfo,
        path: str,
        embedding: Sequence[float] | None = None,
    ) -> DedupeResult:
        exact = self.by_sha.get(info.sha256)
        if exact is not None:
            return DedupeResult(status="exact_dup", reason="exact_sha256", nearest=exact.path)

        query = int(info.phash, 16)
        candidates = self.tree.search(query, self.phash_review)
        nearest: tuple[int, DedupeEntry] | None = (
            min(candidates, key=lambda pair: (pair[0], len(pair[1].path), pair[1].path))
            if candidates else None
        )
        best_cosine: tuple[float, DedupeEntry] | None = None
        embedding = tuple(float(value) for value in embedding) if embedding is not None else None
        if embedding is not None:
            if len(self.entries) <= EMBEDDING_FULL_SCAN_LIMIT:
                scan = self.entries
            else:
                scan = [entry for _, entry in candidates]
            for entry in scan:
                if entry.embedding is None:
                    continue
                cosine = _cosine(embedding, entry.embedding)
                if cosine is None:
                    continue
                if best_cosine is None or cosine > best_cosine[0]:
                    best_cosine = (cosine, entry)

        distance = nearest[0] if nearest else None
        if distance is not None and distance <= self.phash_duplicate:
            cosine = None
            nearest_entry = nearest[1]
            if embedding is not None and nearest_entry.embedding is not None:
                cosine = _cosine(embedding, nearest_entry.embedding)
            near_embedding = cosine is not None and cosine >= self.embedding_near
            reason = "phash_and_embedding" if near_embedding else "phash"
            return DedupeResult(status="near_dup", reason=reason, nearest=nearest_entry.path,
                                phash_distance=distance, embedding_cosine=cosine,
                                nearest_aspect=nearest_entry.aspect)

        if best_cosine is not None and best_cosine[0] >= self.embedding_near:
            cosine, entry = best_cosine
            return DedupeResult(
                status="near_dup",
                reason="embedding" if distance is None or distance > self.phash_near else "embedding",
                nearest=entry.path, phash_distance=distance, embedding_cosine=cosine,
                nearest_aspect=entry.aspect,
            )

        if distance is not None and distance <= self.phash_near:
            nearest_entry = nearest[1]
            aspect_delta = abs(info.aspect - nearest_entry.aspect) / max(
                info.aspect, nearest_entry.aspect, 1e-9)
            reason = "near_crop" if aspect_delta > self.near_crop_aspect_delta else "phash_near"
            cosine = None
            if embedding is not None and nearest_entry.embedding is not None:
                cosine = _cosine(embedding, nearest_entry.embedding)
            return DedupeResult(status="near_dup", reason=reason, nearest=nearest_entry.path,
                                phash_distance=distance, embedding_cosine=cosine,
                                nearest_aspect=nearest_entry.aspect)

        review = distance is not None and distance <= self.phash_review
        cosine = None
        if best_cosine is not None:
            cosine = best_cosine[0]
            review = review or cosine >= self.embedding_review
        result_nearest = nearest[1].path if nearest else (best_cosine[1].path if best_cosine else None)
        result_aspect = nearest[1].aspect if nearest else (
            best_cosine[1].aspect if best_cosine else None)
        return DedupeResult(
            status="new",
            reason="phash_review" if review else None,
            nearest=result_nearest,
            phash_distance=distance,
            embedding_cosine=cosine,
            review=review,
            nearest_aspect=result_aspect,
        )


# --------------------------------------------------------------------------- SQLite index


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  source_path TEXT UNIQUE NOT NULL,
  file_name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  phash TEXT,
  width INTEGER, height INTEGER,
  artist TEXT, artists_json TEXT,
  characters_json TEXT, tags_json TEXT, tag_confidence_json TEXT,
  rating TEXT,
  status TEXT NOT NULL,
  reason TEXT,
  nearest_path TEXT,
  phash_distance INTEGER,
  embedding_cosine REAL
);
CREATE TABLE IF NOT EXISTS tags (
  image_id INTEGER, tag TEXT, score REAL, category TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_sha ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_tags_image ON tags(image_id);
"""


class StagingIndex:
    """Resumable stdlib-SQLite staging index (WAL)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> StagingIndex:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def next_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM images").fetchone()
        return int(row[0])

    def upsert_image(self, row: dict[str, Any], image_id: int | None = None) -> None:
        payload = {
            "id": image_id if image_id is not None else self.next_id(),
            "source_path": row["source_path"],
            "file_name": row.get("file_name", ""),
            "sha256": row.get("sha256", ""),
            "phash": row.get("phash"),
            "width": row.get("width"),
            "height": row.get("height"),
            "artist": row.get("artist"),
            "artists_json": json.dumps(row.get("artists", []), ensure_ascii=False),
            "characters_json": json.dumps(row.get("characters", []), ensure_ascii=False),
            "tags_json": json.dumps(row.get("tags", []), ensure_ascii=False),
            "tag_confidence_json": json.dumps(row.get("tag_confidence", {}), ensure_ascii=False),
            "rating": row.get("rating"),
            "status": row.get("status", "new"),
            "reason": row.get("reason"),
            "nearest_path": row.get("nearest_path"),
            "phash_distance": row.get("phash_distance"),
            "embedding_cosine": row.get("embedding_cosine"),
        }
        self.connection.execute(
            """
            INSERT INTO images (id, source_path, file_name, sha256, phash, width, height, artist,
                                artists_json, characters_json, tags_json, tag_confidence_json,
                                rating, status, reason, nearest_path, phash_distance, embedding_cosine)
            VALUES (:id, :source_path, :file_name, :sha256, :phash, :width, :height, :artist,
                    :artists_json, :characters_json, :tags_json, :tag_confidence_json,
                    :rating, :status, :reason, :nearest_path, :phash_distance, :embedding_cosine)
            ON CONFLICT(source_path) DO UPDATE SET
                file_name=excluded.file_name, sha256=excluded.sha256, phash=excluded.phash,
                width=excluded.width, height=excluded.height, artist=excluded.artist,
                artists_json=excluded.artists_json, characters_json=excluded.characters_json,
                tags_json=excluded.tags_json, tag_confidence_json=excluded.tag_confidence_json,
                rating=excluded.rating, status=excluded.status, reason=excluded.reason,
                nearest_path=excluded.nearest_path, phash_distance=excluded.phash_distance,
                embedding_cosine=excluded.embedding_cosine
            """,
            payload,
        )
        self.connection.commit()

    def add_tag(self, image_id: int, tag: str, score: float, category: str = "general") -> None:
        self.connection.execute(
            "INSERT INTO tags (image_id, tag, score, category) VALUES (?, ?, ?, ?)",
            (image_id, tag, float(score), category))
        self.connection.commit()

    def get(self, source_path: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM images WHERE source_path = ?", (source_path,)).fetchone()
        return dict(row) if row is not None else None


# --------------------------------------------------------------------------- staging run


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def _link(source: Path, target: Path, mode: str) -> None:
    if mode == "none":
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        os.symlink(os.path.relpath(source, target.parent), target)


def _reject_row(source: Path, reason: str, detail: str = "") -> dict[str, Any]:
    return {"source_path": str(source), "reason": reason, "detail": detail}


def _dedupe_payload(result: DedupeResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "nearest": result.nearest,
        "phash_distance": result.phash_distance,
        "embedding_cosine": result.embedding_cosine,
    }


def stage_pool(
    config: StagingConfig,
    *,
    artist_map: dict[str, str] | None = None,
    known_artists: Iterable[str] = (),
    current_records: Iterable[HashRecord] = (),
    embedding_fn: Callable[[Image.Image], Sequence[float]] | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the staging pass and write metadata/rejects/index/report under ``output_dir``."""
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / INDEX_DIR).mkdir(parents=True, exist_ok=True)
    index = StagingIndex(output / INDEX_DIR / INDEX_NAME)
    accepted_rows = _read_jsonl(output / METADATA_NAME) if config.resume else []
    reject_rows = _read_jsonl(output / REJECTS_NAME) if config.resume else []
    accepted_paths = {row.get("source_path") for row in accepted_rows}
    reject_paths = {row.get("source_path") for row in reject_rows}

    dedupe = DedupeIndex(current_records)
    if config.resume:
        for row in index.connection.execute(
                "SELECT * FROM images WHERE status IN ('accepted', 'near_dup')").fetchall():
            if not row["phash"] or not row["width"] or not row["height"]:
                continue
            dedupe.add_entry(DedupeEntry(
                sha256=row["sha256"], phash=row["phash"],
                aspect=float(row["width"]) / float(row["height"]), path=row["source_path"]))

    counts: Counter[str] = Counter()
    accepted_sequence = len(accepted_rows)
    phash_distance_histogram: Counter[int] = Counter()
    exact_groups: dict[str, list[str]] = {}
    near_rows: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    embedding_available = embedding_fn is not None
    artists_seen: Counter[str] = Counter()
    ambiguous_artists: list[str] = []
    processed = 0

    try:
        entries = iter_pool(config.input_root)
        iterator: Iterable[tuple[Path, str]] = entries
        if progress:
            from tqdm import tqdm

            iterator = tqdm(entries, desc="staging", unit="file")
        for source, relative in iterator:
            if limit is not None and processed >= limit:
                break
            processed += 1
            source_abs = str(source.resolve())
            kind = classify_suffix(source)
            if kind in ("gif", "video", "unsupported", "other"):
                reason = kind if kind != "other" else "unsupported"
                if source_abs not in reject_paths:
                    reject_rows.append(_reject_row(source, reason, source.suffix.lower()))
                    reject_paths.add(source_abs)
                counts[reason] += 1
                index.upsert_image({"source_path": source_abs, "status": "reject", "reason": reason})
                continue
            if kind == "sidecar":
                counts["sidecar"] += 1
                continue
            if config.resume:
                existing = index.get(source_abs)
                if existing is not None and existing["status"] in ("accepted", "near_dup"):
                    counts["resumed"] += 1
                    continue

            image: Image.Image | None = None
            try:
                info = inspect_image(source, config, keep_image=embedding_fn is not None)
                image = info.image
                resolution = resolve_artist(relative, artist_map=artist_map,
                                            known_artists=known_artists)
                embedding = embedding_fn(image) if embedding_fn is not None and image is not None else None
                result = dedupe.check(info, source_abs, embedding)
                dedupe.add(info, source_abs, embedding)
                if resolution.ambiguous:
                    ambiguous_artists.append(resolution.raw)
                artists_seen[resolution.artist] += 1
            except RejectedImage as rejection:
                if source_abs not in reject_paths:
                    reject_rows.append(_reject_row(source, rejection.reason, rejection.detail))
                    reject_paths.add(source_abs)
                counts[rejection.reason] += 1
                index.upsert_image({"source_path": source_abs, "status": "reject",
                                    "reason": rejection.reason})
                continue
            finally:
                if image is not None:
                    image.close()

            if result.phash_distance is not None:
                phash_distance_histogram[result.phash_distance] += 1
            if result.status == "exact_dup":
                group = exact_groups.setdefault(info.sha256, [result.nearest or str(source)])
                group.append(str(source))
                counts["exact_dup"] += 1
                detail = f"duplicate of {result.nearest}"
                if source_abs not in reject_paths:
                    reject_rows.append(_reject_row(source, "exact_dup", detail))
                    reject_paths.add(source_abs)
                index.upsert_image({
                    "source_path": source_abs, "sha256": info.sha256, "phash": info.phash,
                    "width": info.width, "height": info.height, "artist": resolution.artist,
                    "artists": list(resolution.artists), "status": "exact_dup",
                    "reason": "exact_sha256", "nearest_path": result.nearest,
                    "phash_distance": result.phash_distance,
                    "embedding_cosine": result.embedding_cosine,
                })
                continue

            image_id = index.next_id()
            suffix = source.suffix.lower()
            if result.status == "near_dup":
                file_name = f"{QUARANTINE_DIR}/{image_id:07d}{suffix}"
                _link(source, output / file_name, config.link_mode)
                counts["near_dup"] += 1
                near_rows.append({
                    "source_path": source_abs, "sha256": info.sha256,
                    "nearest": result.nearest, "reason": result.reason,
                    "phash_distance": result.phash_distance,
                    "embedding_cosine": result.embedding_cosine,
                })
                if source_abs not in reject_paths:
                    reject_rows.append(_reject_row(
                        source, "near_dup", f"{result.reason}; nearest={result.nearest}"))
                    reject_paths.add(source_abs)
                index.upsert_image({
                    "source_path": source_abs, "file_name": file_name, "sha256": info.sha256,
                    "phash": info.phash, "width": info.width, "height": info.height,
                    "artist": resolution.artist, "artists": list(resolution.artists),
                    "status": "near_dup", "reason": result.reason,
                    "nearest_path": result.nearest, "phash_distance": result.phash_distance,
                    "embedding_cosine": result.embedding_cosine,
                }, image_id=image_id)
                continue

            file_name = f"{IMAGES_DIR}/{accepted_sequence + 1:07d}{suffix}"
            accepted_sequence += 1
            _link(source, output / file_name, config.link_mode)
            counts["accepted"] += 1
            if result.review:
                reviews.append({"source_path": source_abs, "nearest": result.nearest,
                                "phash_distance": result.phash_distance,
                                "embedding_cosine": result.embedding_cosine})
            row = {
                "file_name": file_name,
                "source_path": source_abs,
                "source_kind": "image",
                "width": info.width,
                "height": info.height,
                "sha256": info.sha256,
                "phash": info.phash,
                "artist": resolution.artist,
                "artists": list(resolution.artists),
                "artist_missing": resolution.missing,
                "character": [],
                "character_missing": True,
                "tags": [],
                "tag_confidence": {},
                "rating": None,
                "alpha_composited": info.alpha_composited,
                "dedupe": _dedupe_payload(result),
                "staging_version": STAGING_VERSION,
            }
            if config.resume and source_abs in accepted_paths:
                accepted_rows = [item for item in accepted_rows if item.get("source_path") != source_abs]
            accepted_rows.append(row)
            accepted_paths.add(source_abs)
            index.upsert_image({
                "source_path": source_abs, "file_name": file_name, "sha256": info.sha256,
                "phash": info.phash, "width": info.width, "height": info.height,
                "artist": resolution.artist, "artists": list(resolution.artists),
                "status": "accepted", "nearest_path": result.nearest,
                "phash_distance": result.phash_distance,
                "embedding_cosine": result.embedding_cosine,
            }, image_id=image_id)
    finally:
        index.close()

    _write_jsonl(output / METADATA_NAME, accepted_rows)
    _write_jsonl(output / REJECTS_NAME, reject_rows)
    if artist_map is not None:
        atomic_write_json(output / ARTIST_MAP_NAME, artist_map)
    counts["accepted"] = len(accepted_rows)
    report = {
        "staging_version": STAGING_VERSION,
        "input_root": str(config.input_root),
        "output_dir": str(output),
        "processed": processed,
        "accepted": len(accepted_rows),
        "counts": dict(sorted(counts.items())),
        "exact_duplicate_groups": [
            {"sha256": sha, "paths": paths} for sha, paths in sorted(exact_groups.items())
        ],
        "near_duplicates": near_rows,
        "review_candidates": reviews,
        "phash_distance_histogram": {str(key): value
                                     for key, value in sorted(phash_distance_histogram.items())},
        "embedding": "available" if embedding_available else "unavailable",
        "embedding_scope": ("full" if len(dedupe.entries) <= EMBEDDING_FULL_SCAN_LIMIT
                            else "phash_candidates"),
        "artists": {"unique": len(artists_seen), "top": artists_seen.most_common(20),
                    "ambiguous": sorted(set(ambiguous_artists))},
        "config": {
            "min_side": config.min_side,
            "max_side": config.max_side,
            "max_megapixels": config.max_megapixels,
            "min_aspect": config.min_aspect,
            "max_aspect": config.max_aspect,
            "blank_entropy": config.blank_entropy,
            "blank_std": config.blank_std,
            "phash_duplicate": config.phash_duplicate,
            "phash_near": config.phash_near,
            "phash_review": config.phash_review,
            "embedding_duplicate": config.embedding_duplicate,
            "embedding_near": config.embedding_near,
            "embedding_review": config.embedding_review,
            "link_mode": config.link_mode,
        },
    }
    atomic_write_json(output / DEDUPE_REPORT_NAME, report)
    return report


def make_siglip_embedder(
    model_path: str | Path,
    *,
    device: str = "cuda",
    local_files_only: bool = True,
) -> Callable[[Image.Image], list[float]]:
    """Lazy SigLIP2 image embedder (no download); CPU fallback when CUDA is absent."""
    import torch
    from transformers import AutoImageProcessor, Siglip2Model, SiglipModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=local_files_only)
    model = None
    load_errors: list[str] = []
    for model_class in (Siglip2Model, SiglipModel):
        try:
            model = model_class.from_pretrained(str(model_path), local_files_only=local_files_only)
            break
        except Exception as error:  # try the other architecture before failing
            load_errors.append(f"{model_class.__name__}: {type(error).__name__}")
    if model is None:
        raise RuntimeError(f"Could not load a SigLIP image encoder from {model_path}: {load_errors}")
    model.eval().to(device)

    def embed(image: Image.Image) -> list[float]:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        vector = features[0].float().cpu()
        norm = vector.norm()
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    return embed


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="moss-lora-stage",
        description="Stage/filter/dedupe an artist-directory expansion pool (no captions).")
    parser.add_argument("--input", required=True, help="Artist-directory pool root")
    parser.add_argument("--output", required=True, help="Staging output directory")
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=12000)
    parser.add_argument("--max-megapixels", type=float, default=80.0)
    parser.add_argument("--min-aspect", type=float, default=0.25)
    parser.add_argument("--max-aspect", type=float, default=4.0)
    parser.add_argument("--blank-entropy", type=float, default=3.5)
    parser.add_argument("--blank-std", type=float, default=8.0)
    parser.add_argument("--dedupe-against", default=None,
                        help="Current-corpus metadata.jsonl or image directory")
    parser.add_argument("--artist-map", default=None, help="JSON raw-name -> canonical-ID map")
    parser.add_argument("--link-mode", default="symlink", choices=["symlink", "copy", "none"])
    parser.add_argument("--siglip-model", default=None,
                        help="Local SigLIP2 checkout for embedding near-dedupe (optional)")
    parser.add_argument("--siglip-device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Skip already accepted/near rows")
    parser.add_argument("--limit", type=int, default=None, help="Pilot limit over sorted pool entries")
    args = parser.parse_args(argv)

    config = StagingConfig(
        input_root=Path(args.input), output_dir=Path(args.output),
        min_side=args.min_side, max_side=args.max_side,
        max_megapixels=args.max_megapixels, min_aspect=args.min_aspect, max_aspect=args.max_aspect,
        blank_entropy=args.blank_entropy, blank_std=args.blank_std,
        link_mode=args.link_mode, resume=args.resume,
    )
    artist_map = load_artist_map(args.artist_map)
    known_source = Path(args.dedupe_against) if args.dedupe_against else None
    if known_source is not None and known_source.is_dir():
        known_source = known_source / "metadata.jsonl"
    known_artists = load_known_artists(known_source)
    current = load_current_records(args.dedupe_against, progress=True)
    embedding_fn = (make_siglip_embedder(args.siglip_model, device=args.siglip_device)
                    if args.siglip_model else None)
    report = stage_pool(config, artist_map=artist_map, known_artists=known_artists,
                        current_records=current, embedding_fn=embedding_fn,
                        limit=args.limit)
    print(json.dumps({key: report[key] for key in ("processed", "accepted", "counts")},
                     ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
