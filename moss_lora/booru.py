"""Read-only Danbooru MD5 lookup, tag mapping and resumable SQLite cache.

The expansion pool stores original bytes under SHA-256 filenames, so a booru
lookup must hash those exact bytes (no re-encode, no download) and query the
public post API by ``md5:``. This module owns:

- :func:`file_md5`, a raw-byte MD5 (only the hash differs from the filename);
- :class:`DanbooruClient`, a credential-free client with a >=1.2 s token
  bucket, capped 429/5xx exponential backoff, ``Retry-After`` support and a
  hard stop on 401/403;
- tag normalization plus mapping to the calibrated character/general vocab;
- :class:`BooruCache`, an atomic stdlib-SQLite cache that makes a pass
  resumable by ``(sha256, site)``.

No function here prints tag values. Callers write aggregate counts only; the
full post JSON (which contains tags) is stored locally in SQLite for the
training export and is never echoed to logs.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from moss_lora.annotations import NEGATIVE_TAG_KEYS, normalize_name, normalize_tag_key

SITE_DANBOORU = "danbooru"
SITE_YANDERE = "yandere"
SUPPORTED_SITES = (SITE_DANBOORU,)

DANBOORU_BASE_URL = "https://danbooru.donmai.us"
USER_AGENT = (
    "MOSS-LoRA-booru-recovery/1.0 "
    "(read-only metadata recovery for a local LoRA dataset; no downloads)"
)

DEFAULT_MIN_INTERVAL = 1.2
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_BACKOFF_CAP = 60.0
DEFAULT_MAX_RETRY_AFTER = 300.0

STATUS_HIT = "hit"
STATUS_MISS = "miss"
STATUS_AUTH_BLOCKED = "auth_blocked"
STATUS_ERROR = "error"
TERMINAL_STATUSES = frozenset((STATUS_HIT, STATUS_MISS, STATUS_AUTH_BLOCKED))

# Danbooru short ratings -> the long labels used by moss_lora.tagger.
RATING_LABELS: dict[str, str] = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit",
}

_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_INT_FIELDS = ("file_size", "image_width", "image_height")


class BooruError(RuntimeError):
    """Malformed API payload or invalid lookup input."""


class BooruCacheError(RuntimeError):
    """The cache file cannot be opened or is structurally invalid."""


# --------------------------------------------------------------------------- hashing


def file_md5(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """MD5 of the exact file bytes (no decode, no re-encode)."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_md5(value: str) -> str:
    digest = str(value).strip().lower()
    if not _MD5_RE.fullmatch(digest):
        raise BooruError(f"invalid MD5 digest: {value!r}")
    return digest


# --------------------------------------------------------------------------- post parsing


@dataclass(frozen=True)
class BooruPost:
    """Typed view of one booru post; ``raw`` is the untouched API object."""

    site: str
    post_id: int | None
    md5: str | None
    rating_raw: str | None
    rating: str | None
    source: str | None
    file_size: int | None
    width: int | None
    height: int | None
    character_tags: tuple[str, ...]
    artist_tags: tuple[str, ...]
    general_tags: tuple[str, ...]
    copyright_tags: tuple[str, ...] = ()
    meta_tags: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def has_character(self) -> bool:
        return bool(self.character_tags)

    @property
    def has_artist(self) -> bool:
        return bool(self.artist_tags)

    @property
    def has_general(self) -> bool:
        return bool(self.general_tags)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_rating(value: Any) -> str | None:
    """Danbooru ``g/s/q/e`` (or already-long labels) -> tagger rating label."""
    text = _as_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in RATING_LABELS:
        return RATING_LABELS[lowered]
    return lowered if lowered in RATING_LABELS.values() else None


def split_tag_string(value: Any) -> tuple[str, ...]:
    """Normalize a space-separated booru tag string, preserving tag order."""
    values: Iterable[Any]
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = str(value).split()
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        clean = normalize_name(str(item))
        key = normalize_tag_key(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return tuple(result)


def parse_post(row: Mapping[str, Any], *, site: str = SITE_DANBOORU) -> BooruPost:
    if not isinstance(row, Mapping):
        raise BooruError("post payload is not a JSON object")
    raw_rating = _as_text(row.get("rating"))
    return BooruPost(
        site=site,
        post_id=_as_int(row.get("id")),
        md5=_as_text(row.get("md5")),
        rating_raw=raw_rating,
        rating=normalize_rating(raw_rating),
        source=_as_text(row.get("source")),
        file_size=_as_int(row.get("file_size")),
        width=_as_int(row.get("image_width", row.get("width"))),
        height=_as_int(row.get("image_height", row.get("height"))),
        character_tags=split_tag_string(row.get("tag_string_character")),
        artist_tags=split_tag_string(row.get("tag_string_artist")),
        general_tags=split_tag_string(row.get("tag_string_general")),
        copyright_tags=split_tag_string(row.get("tag_string_copyright")),
        meta_tags=split_tag_string(row.get("tag_string_meta")),
        raw=dict(row),
    )


def source_host(source: str | None) -> str | None:
    if not source:
        return None
    try:
        host = urllib.parse.urlsplit(source).hostname
    except ValueError:
        return None
    return host.lower() if host else None


# --------------------------------------------------------------------------- vocabulary


def load_vocab(path: str | Path) -> dict[str, str]:
    """Load a vocab JSON (``{key: display}`` or ``[display, ...]``).

    Keys are normalized with :func:`normalize_tag_key`; the display value is
    preserved (character vocabularies use underscores, general tags spaces).
    First spelling wins, deterministically.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        items = payload.items()
    elif isinstance(payload, list):
        items = ((value, value) for value in payload)
    else:
        raise BooruError(f"vocabulary must be a JSON object or list: {path}")
    vocab: dict[str, str] = {}
    for key, value in items:
        clean_key = normalize_tag_key(str(key))
        if not clean_key or clean_key in vocab:
            continue
        display = normalize_name(str(value))
        if display:
            vocab[clean_key] = display
    return vocab


@dataclass(frozen=True)
class MappedTags:
    characters: tuple[str, ...]
    tags: tuple[str, ...]
    new_character: tuple[str, ...]
    new_tags: tuple[str, ...]


def _map_tags(values: Iterable[str], vocab: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapped: list[str] = []
    new: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_tag_key(value)
        if not key or key in NEGATIVE_TAG_KEYS or key in seen:
            continue
        seen.add(key)
        display = vocab.get(key)
        if display is None:
            display = normalize_name(value)
            new.append(display)
        mapped.append(display)
    return tuple(mapped), tuple(new)


def map_post_tags(
    post: BooruPost,
    character_vocab: Mapping[str, str] | None = None,
    general_vocab: Mapping[str, str] | None = None,
) -> MappedTags:
    """Map character/general tags to the calibrated vocab; keep source-only tags.

    ``new_character``/``new_tags`` are tags present only in the booru source,
    i.e. not in the current vocab. They are reported as counts and can seed a
    later vocabulary review; they are not silently dropped from the output.
    """
    characters, new_character = _map_tags(post.character_tags, character_vocab or {})
    tags, new_tags = _map_tags(post.general_tags, general_vocab or {})
    return MappedTags(characters, tags, new_character, new_tags)


def artist_key(value: str) -> str:
    """Normalize an artist handle for comparison only (never for output labels)."""
    key = normalize_tag_key(value)
    return re.sub(r"[\s_\-]+$", "", key)


def artist_matches(artist: str, post: BooruPost) -> bool:
    directory_key = artist_key(artist)
    return bool(directory_key) and any(artist_key(tag) == directory_key for tag in post.artist_tags)


# --------------------------------------------------------------------------- rate limiting


class TokenBucket:
    """Sequential minimum-interval limiter (one request at a time)."""

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0

    def acquire(self) -> None:
        now = self._clock()
        wait = self._next - now
        if wait > 0:
            self._sleep(wait)
            now = max(self._clock(), self._next)
        self._next = now + self.min_interval


# --------------------------------------------------------------------------- HTTP client


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""


Fetch = Callable[[str, Mapping[str, str], float], HttpResponse]


def _urlopen_fetch(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=int(getattr(response, "status", response.getcode())),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
        except Exception:  # noqa: BLE001 - error body is optional
            body = b""
        return HttpResponse(
            status=int(error.code),
            headers=dict(error.headers.items()) if error.headers else {},
            body=body,
        )


@dataclass(frozen=True)
class LookupResult:
    site: str
    md5: str
    status: str
    http_code: int | None = None
    post: BooruPost | None = None
    error: str | None = None
    attempts: int = 0

    @property
    def hit(self) -> bool:
        return self.status == STATUS_HIT


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return None


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from ``Retry-After`` (delta-seconds or HTTP-date)."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (target - current).total_seconds())
    return max(0.0, seconds)


class DanbooruClient:
    """Anonymous Danbooru ``md5:`` lookup client (no API key, no retries on 404).

    A 401/403 latches the client as blocked; every later lookup returns
    ``auth_blocked`` without touching the network. 429/5xx are retried with
    ``Retry-After`` or capped exponential backoff; a 429 whose ``Retry-After``
    exceeds ``max_retry_after`` is returned as a non-terminal error so the
    caller can stop instead of hammering the site.
    """

    site = SITE_DANBOORU

    def __init__(
        self,
        *,
        base_url: str = DANBOORU_BASE_URL,
        user_agent: str = USER_AGENT,
        fetch: Fetch | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.max_retry_after = float(max_retry_after)
        self._fetch = fetch or _urlopen_fetch
        self._bucket = TokenBucket(min_interval, clock=clock, sleep=sleep)
        self._sleep = sleep
        self._blocked_code: int | None = None

    @property
    def blocked_code(self) -> int | None:
        return self._blocked_code

    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "application/json"}

    def post_url(self, md5: str) -> str:
        query = urllib.parse.urlencode({"tags": f"md5:{normalize_md5(md5)}", "limit": 1})
        return f"{self.base_url}/posts.json?{query}"

    def _retry_delay(self, headers: Mapping[str, str], attempt: int) -> float:
        retry_after = parse_retry_after(_header(headers, "Retry-After"))
        if retry_after is not None:
            return min(retry_after, self.max_retry_after)
        return min(self.backoff_base * (2.0 ** (attempt - 1)), self.backoff_cap)

    def lookup(self, md5: str) -> LookupResult:
        digest = normalize_md5(md5)
        if self._blocked_code is not None:
            return LookupResult(
                self.site, digest, STATUS_AUTH_BLOCKED, http_code=self._blocked_code,
                error=f"site previously blocked with HTTP {self._blocked_code}",
            )

        url = self.post_url(digest)
        last_code: int | None = None
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._bucket.acquire()
            try:
                response = self._fetch(url, self.headers(), self.timeout)
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                last_code = None
                last_error = f"network {type(error).__name__}"
                if attempt < self.max_attempts:
                    self._sleep(min(self.backoff_base * (2.0 ** (attempt - 1)), self.backoff_cap))
                    continue
                break

            code = int(response.status)
            last_code = code

            if code in (401, 403):
                self._blocked_code = code
                return LookupResult(
                    self.site, digest, STATUS_AUTH_BLOCKED, http_code=code,
                    error=f"HTTP {code} (authentication required/forbidden)",
                    attempts=attempt,
                )
            if code == 404:
                return LookupResult(self.site, digest, STATUS_MISS, http_code=code, attempts=attempt)
            if code == 200:
                return self._parse_ok(digest, response.body, attempt)
            if code == 429 or code >= 500:
                retry_after = parse_retry_after(_header(response.headers, "Retry-After"))
                if code == 429 and retry_after is not None and retry_after > self.max_retry_after:
                    return LookupResult(
                        self.site, digest, STATUS_ERROR, http_code=code,
                        error=f"HTTP 429 Retry-After {retry_after:.0f}s exceeds cap",
                        attempts=attempt,
                    )
                last_error = f"HTTP {code} after {attempt} attempt(s)"
                if attempt < self.max_attempts:
                    self._sleep(self._retry_delay(response.headers, attempt))
                    continue
                break
            return LookupResult(
                self.site, digest, STATUS_ERROR, http_code=code,
                error=f"unexpected HTTP {code}", attempts=attempt,
            )

        return LookupResult(
            self.site, digest, STATUS_ERROR, http_code=last_code,
            error=last_error or "lookup failed", attempts=self.max_attempts,
        )

    def _parse_ok(self, digest: str, body: bytes, attempt: int) -> LookupResult:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return LookupResult(
                self.site, digest, STATUS_ERROR, http_code=200,
                error="response is not valid JSON", attempts=attempt,
            )
        if not isinstance(payload, list):
            return LookupResult(
                self.site, digest, STATUS_ERROR, http_code=200,
                error="response is not a JSON list", attempts=attempt,
            )
        if not payload:
            return LookupResult(self.site, digest, STATUS_MISS, http_code=200, attempts=attempt)
        try:
            post = parse_post(payload[0], site=self.site)
        except BooruError as error:
            return LookupResult(
                self.site, digest, STATUS_ERROR, http_code=200,
                error=str(error), attempts=attempt,
            )
        return LookupResult(
            self.site, digest, STATUS_HIT, http_code=200, post=post, attempts=attempt,
        )


# --------------------------------------------------------------------------- SQLite cache


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    sha256 TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    artist TEXT NOT NULL,
    bytes INTEGER,
    width INTEGER,
    height INTEGER,
    md5 TEXT,
    md5_done_at TEXT
);

CREATE TABLE IF NOT EXISTS site_probe (
    sha256 TEXT NOT NULL,
    site TEXT NOT NULL,
    status TEXT NOT NULL,
    http_code INTEGER,
    hit INTEGER NOT NULL DEFAULT 0,
    post_id TEXT,
    rating TEXT,
    has_character INTEGER NOT NULL DEFAULT 0,
    has_artist INTEGER NOT NULL DEFAULT 0,
    has_general INTEGER NOT NULL DEFAULT 0,
    character_count INTEGER NOT NULL DEFAULT 0,
    general_count INTEGER NOT NULL DEFAULT 0,
    new_character_count INTEGER NOT NULL DEFAULT 0,
    new_general_count INTEGER NOT NULL DEFAULT 0,
    api_md5 TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    source_host TEXT,
    fetched_at TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (sha256, site)
);

CREATE INDEX IF NOT EXISTS site_probe_status ON site_probe(site, status);

CREATE TABLE IF NOT EXISTS posts (
    site TEXT NOT NULL,
    post_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (site, post_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BooruCache:
    """Atomic, resumable SQLite cache keyed by SHA-256 and ``(sha256, site)``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.path))
        except (OSError, sqlite3.Error) as error:
            raise BooruCacheError(f"cannot open cache {self.path}: {error}") from error
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.executescript(SCHEMA_SQL)
            self.connection.commit()
        except sqlite3.Error as error:
            self.connection.close()
            raise BooruCacheError(f"cannot initialize cache {self.path}: {error}") from error

    def __enter__(self) -> "BooruCache":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None  # type: ignore[assignment]

    # -- files

    def record_file(
        self,
        sha256: str,
        *,
        path: str,
        artist: str,
        size: int | None = None,
        md5: str | None = None,
    ) -> None:
        stamp = utc_now() if md5 else None
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO files (sha256, path, artist, bytes, md5, md5_done_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    path = excluded.path,
                    artist = excluded.artist,
                    bytes = COALESCE(excluded.bytes, files.bytes),
                    md5 = COALESCE(excluded.md5, files.md5),
                    md5_done_at = CASE
                        WHEN excluded.md5 IS NOT NULL THEN excluded.md5_done_at
                        ELSE files.md5_done_at END
                """,
                (sha256, path, artist, size, md5, stamp),
            )

    def get_md5(self, sha256: str) -> str | None:
        row = self.connection.execute(
            "SELECT md5 FROM files WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return str(row["md5"]) if row is not None and row["md5"] else None

    def get_file(self, sha256: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM files WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return dict(row) if row is not None else None

    # -- probes

    def record_probe(
        self,
        sha256: str,
        *,
        site: str,
        status: str,
        http_code: int | None = None,
        hit: bool = False,
        post_id: str | int | None = None,
        rating: str | None = None,
        has_character: bool = False,
        has_artist: bool = False,
        has_general: bool = False,
        character_count: int = 0,
        general_count: int = 0,
        new_character_count: int = 0,
        new_general_count: int = 0,
        api_md5: str | None = None,
        file_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        source_host: str | None = None,
        error: str | None = None,
        fetched_at: str | None = None,
    ) -> None:
        row = (
            sha256, site, status, http_code, int(bool(hit)),
            None if post_id is None else str(post_id), rating,
            int(bool(has_character)), int(bool(has_artist)), int(bool(has_general)),
            int(character_count), int(general_count),
            int(new_character_count), int(new_general_count),
            api_md5, file_size, width, height, source_host,
            fetched_at or utc_now(), error,
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO site_probe (
                    sha256, site, status, http_code, hit, post_id, rating,
                    has_character, has_artist, has_general,
                    character_count, general_count,
                    new_character_count, new_general_count,
                    api_md5, file_size, width, height, source_host, fetched_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256, site) DO UPDATE SET
                    status = excluded.status,
                    http_code = excluded.http_code,
                    hit = excluded.hit,
                    post_id = excluded.post_id,
                    rating = excluded.rating,
                    has_character = excluded.has_character,
                    has_artist = excluded.has_artist,
                    has_general = excluded.has_general,
                    character_count = excluded.character_count,
                    general_count = excluded.general_count,
                    new_character_count = excluded.new_character_count,
                    new_general_count = excluded.new_general_count,
                    api_md5 = excluded.api_md5,
                    file_size = excluded.file_size,
                    width = excluded.width,
                    height = excluded.height,
                    source_host = excluded.source_host,
                    fetched_at = excluded.fetched_at,
                    error = excluded.error
                """,
                row,
            )

    def probe(self, sha256: str, site: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM site_probe WHERE sha256 = ? AND site = ?", (sha256, site)
        ).fetchone()
        return dict(row) if row is not None else None

    def terminal_sha256(self, site: str) -> set[str]:
        statuses = tuple(sorted(TERMINAL_STATUSES))
        placeholders = ", ".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"SELECT sha256 FROM site_probe WHERE site = ? AND status IN ({placeholders})",
            (site, *statuses),
        ).fetchall()
        return {str(row["sha256"]) for row in rows}

    # -- posts

    def store_post(
        self,
        site: str,
        post_id: str | int,
        raw_json: str,
        *,
        fetched_at: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO posts (site, post_id, raw_json, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(site, post_id) DO UPDATE SET
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at
                """,
                (site, str(post_id), raw_json, fetched_at or utc_now()),
            )

    def load_post(self, site: str, post_id: str | int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT raw_json FROM posts WHERE site = ? AND post_id = ?", (site, str(post_id))
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["raw_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    # -- site state

    def set_meta(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def mark_blocked(self, site: str, http_code: int | None) -> None:
        self.set_meta(f"auth_blocked:{site}", f"{http_code or ''}")

    def blocked_code(self, site: str) -> int | None:
        value = self.get_meta(f"auth_blocked:{site}")
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return 0

    def clear_blocked(self, site: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM meta WHERE key = ?", (f"auth_blocked:{site}",))

    # -- reporting

    def stats(self, site: str) -> dict[str, Any]:
        def scalar(sql: str, params: tuple[Any, ...] = ()) -> Any:
            row = self.connection.execute(sql, params).fetchone()
            return row[0] if row is not None else None

        statuses = {
            str(row["status"]): int(row["n"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS n FROM site_probe WHERE site = ? GROUP BY status",
                (site,),
            )
        }
        ratings = {
            str(row["rating"]): int(row["n"])
            for row in self.connection.execute(
                "SELECT rating, COUNT(*) AS n FROM site_probe "
                "WHERE site = ? AND status = 'hit' AND rating IS NOT NULL GROUP BY rating",
                (site,),
            )
        }
        hosts = {
            str(row["source_host"]): int(row["n"])
            for row in self.connection.execute(
                "SELECT source_host, COUNT(*) AS n FROM site_probe "
                "WHERE site = ? AND status = 'hit' AND source_host IS NOT NULL "
                "GROUP BY source_host ORDER BY n DESC LIMIT 10",
                (site,),
            )
        }
        return {
            "files": int(scalar("SELECT COUNT(*) FROM files") or 0),
            "files_with_md5": int(scalar("SELECT COUNT(*) FROM files WHERE md5 IS NOT NULL") or 0),
            "posts": int(scalar("SELECT COUNT(*) FROM posts WHERE site = ?", (site,)) or 0),
            "statuses": statuses,
            "hits": statuses.get(STATUS_HIT, 0),
            "misses": statuses.get(STATUS_MISS, 0),
            "auth_blocked": statuses.get(STATUS_AUTH_BLOCKED, 0),
            "errors": statuses.get(STATUS_ERROR, 0),
            "hit_with_character": int(scalar(
                "SELECT COUNT(*) FROM site_probe "
                "WHERE site = ? AND status = 'hit' AND has_character = 1", (site,)
            ) or 0),
            "hit_with_artist": int(scalar(
                "SELECT COUNT(*) FROM site_probe "
                "WHERE site = ? AND status = 'hit' AND has_artist = 1", (site,)
            ) or 0),
            "hit_with_general": int(scalar(
                "SELECT COUNT(*) FROM site_probe "
                "WHERE site = ? AND status = 'hit' AND has_general = 1", (site,)
            ) or 0),
            "character_tags": int(scalar(
                "SELECT COALESCE(SUM(character_count), 0) FROM site_probe "
                "WHERE site = ? AND status = 'hit'", (site,)
            ) or 0),
            "general_tags": int(scalar(
                "SELECT COALESCE(SUM(general_count), 0) FROM site_probe "
                "WHERE site = ? AND status = 'hit'", (site,)
            ) or 0),
            "new_character_tags": int(scalar(
                "SELECT COALESCE(SUM(new_character_count), 0) FROM site_probe "
                "WHERE site = ? AND status = 'hit'", (site,)
            ) or 0),
            "new_general_tags": int(scalar(
                "SELECT COALESCE(SUM(new_general_count), 0) FROM site_probe "
                "WHERE site = ? AND status = 'hit'", (site,)
            ) or 0),
            "ratings": ratings,
            "source_hosts": hosts,
            "first_fetched_at": scalar(
                "SELECT MIN(fetched_at) FROM site_probe WHERE site = ?", (site,)
            ),
            "last_fetched_at": scalar(
                "SELECT MAX(fetched_at) FROM site_probe WHERE site = ?", (site,)
            ),
        }
