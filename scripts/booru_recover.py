#!/usr/bin/env python
"""Resumable Danbooru MD5 tag recovery for the artist-directory expansion pool.

Scans the pool's original still images only (``<artist>/<2hex>/<sha256>.<ext>``;
``.thumbs``, GIFs and videos are excluded), computes MD5 from the raw bytes,
queries Danbooru by ``md5:`` and writes one JSONL record per image keyed by
SHA-256. Artist is always the parent-directory handle; booru artist tags are
recorded separately and never override it.

The pass is resumable through ``(sha256, site)`` rows in a SQLite cache and an
append-only JSONL output. Logs and the generated report contain counts and
field names only: no tag values or captions are printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.booru import (  # noqa: E402
    STATUS_AUTH_BLOCKED,
    STATUS_ERROR,
    STATUS_HIT,
    STATUS_MISS,
    SUPPORTED_SITES,
    BooruCache,
    BooruError,
    DanbooruClient,
    LookupResult,
    artist_matches,
    file_md5,
    load_vocab,
    map_post_tags,
    parse_post,
    source_host,
    utc_now,
)

DEFAULT_POOL = Path("/mnt/data/sieve-artist-pool/images")
DEFAULT_CACHE = Path("/mnt/data/train_test/booru_md5_cache.sqlite")
DEFAULT_OUT = Path("/mnt/data/train_test/booru_recovered.jsonl")
DEFAULT_REPORT = Path("/mnt/data/train_test/booru_recovery_report.md")
DEFAULT_VOCAB_DIR = Path("/mnt/data/train_test/tagger_calibration")

STATIC_IMAGE_EXTS = frozenset((".jpg", ".jpeg", ".png", ".webp"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"^[0-9a-f]{2}$")

MAX_CONSECUTIVE_ERRORS = 5
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_STOPPED = 3
EXIT_ALREADY_BLOCKED = 4


class RecoveryError(RuntimeError):
    """Invalid recovery configuration or unreadable pool."""


@dataclass(frozen=True)
class PoolImage:
    sha256: str
    path: Path
    artist: str
    size: int


@dataclass
class RecoveryConfig:
    pool_root: Path = DEFAULT_POOL
    cache_path: Path = DEFAULT_CACHE
    out_path: Path = DEFAULT_OUT
    report_path: Path = DEFAULT_REPORT
    vocab_dir: Path = DEFAULT_VOCAB_DIR
    site: str = "danbooru"
    limit: int | None = None
    spread: bool = False
    resume: bool = True
    dry_run: bool = False
    progress_every: int = 100
    min_interval: float = 1.2
    timeout: float = 20.0
    max_attempts: int = 4
    quiet: bool = False


# --------------------------------------------------------------------------- pool scan


def iter_pool_images(root: str | Path) -> Iterator[PoolImage]:
    """Yield deterministic original still images; never enters hidden dirs."""
    root = Path(root)
    if not root.is_dir():
        raise RecoveryError(f"pool root is not a directory: {root}")
    for artist_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not artist_dir.is_dir() or artist_dir.name.startswith("."):
            continue
        for shard in sorted(artist_dir.iterdir(), key=lambda path: path.name):
            if not shard.is_dir() or not _SHARD_RE.fullmatch(shard.name):
                continue
            for path in sorted(shard.iterdir(), key=lambda item: item.name):
                if path.suffix.lower() not in STATIC_IMAGE_EXTS or not path.is_file():
                    continue
                if not _SHA256_RE.fullmatch(path.stem):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                yield PoolImage(sha256=path.stem, path=path, artist=artist_dir.name, size=size)


def spread_select(items: Sequence[PoolImage], limit: int) -> list[PoolImage]:
    """Pick ``limit`` items evenly spaced across the deterministic order."""
    if limit <= 0:
        return []
    if limit >= len(items):
        return list(items)
    step = len(items) / float(limit)
    return [items[min(len(items) - 1, int(index * step))] for index in range(limit)]


# --------------------------------------------------------------------------- JSONL state


def repair_jsonl_tail(path: Path) -> None:
    """Drop a torn final line left by a crash so the file stays valid NDJSON."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        position = size
        while position > 0:
            chunk = min(1 << 16, position)
            handle.seek(position - chunk)
            data = handle.read(chunk)
            newline = data.rfind(b"\n")
            if newline >= 0:
                handle.truncate(position - chunk + newline + 1)
                return
            position -= chunk
        handle.truncate(0)


def read_jsonl_keys(path: Path, site: str) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            sha256 = row.get("sha256")
            row_site = row.get("site") or "danbooru"
            if isinstance(sha256, str) and row_site == site:
                keys.add(sha256)
    return keys


def reconcile_missing_rows(
    cache: BooruCache,
    config: RecoveryConfig,
    character_vocab: Mapping[str, str],
    general_vocab: Mapping[str, str],
    jsonl_keys: set[str],
) -> int:
    """Rebuild JSONL rows for terminal cache rows written by an older crash window.

    The current loop appends JSONL before committing the cache, so a crash can
    only lose cache state, never output. This repair handles output written by
    earlier versions that committed the cache first: hits are rebuilt from
    ``posts.raw_json`` and misses/auth blocks from ``site_probe``.
    """
    missing = cache.terminal_sha256(config.site) - jsonl_keys
    if not missing:
        return 0
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = 0
    with open(config.out_path, "a", encoding="utf-8") as output:
        for sha256 in sorted(missing):
            probe = cache.probe(sha256, config.site)
            file_row = cache.get_file(sha256)
            if probe is None or file_row is None or not file_row.get("md5"):
                continue
            status = str(probe.get("status"))
            result: LookupResult | None = None
            if status == STATUS_HIT:
                raw = cache.load_post(config.site, probe.get("post_id") or "")
                if raw is None:
                    continue
                try:
                    post = parse_post(raw, site=config.site)
                except BooruError:
                    continue
                result = LookupResult(
                    config.site, str(file_row["md5"]), STATUS_HIT,
                    http_code=probe.get("http_code"), post=post,
                )
            elif status in (STATUS_MISS, STATUS_AUTH_BLOCKED):
                result = LookupResult(
                    config.site, str(file_row["md5"]), status,
                    http_code=probe.get("http_code"), error=probe.get("error"),
                )
            if result is None:
                continue
            image = PoolImage(
                sha256=sha256,
                path=Path(str(file_row.get("path") or "")),
                artist=str(file_row.get("artist") or ""),
                size=int(file_row.get("bytes") or 0),
            )
            row = build_row(
                image, md5=str(file_row["md5"]), site=config.site, result=result,
                character_vocab=character_vocab, general_vocab=general_vocab,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            repaired += 1
        output.flush()
        if repaired:
            os.fsync(output.fileno())
    return repaired


# --------------------------------------------------------------------------- row building


def build_row(
    image: PoolImage,
    *,
    md5: str,
    site: str,
    result: LookupResult,
    character_vocab: Mapping[str, str],
    general_vocab: Mapping[str, str],
) -> dict[str, Any]:
    post = result.post if result.status == STATUS_HIT else None
    mapped = (
        map_post_tags(post, character_vocab, general_vocab) if post is not None else None
    )
    return {
        "sha256": image.sha256,
        "site": site,
        "artist": image.artist,
        "characters": list(mapped.characters) if mapped else [],
        "tags": list(mapped.tags) if mapped else [],
        "rating": post.rating_raw if post else None,
        "rating_label": post.rating if post else None,
        "source": post.source if post else None,
        "source_host": source_host(post.source) if post else None,
        "post_id": post.post_id if post else None,
        "hit": result.status == STATUS_HIT,
        "status": result.status,
        "md5": md5,
        "api_md5": post.md5 if post else None,
        "file_size": post.file_size if post else None,
        "width": post.width if post else None,
        "height": post.height if post else None,
        "booru_artist": list(post.artist_tags) if post else [],
        "artist_match": bool(post and artist_matches(image.artist, post)),
        "new_character": list(mapped.new_character) if mapped else [],
        "new_tags": list(mapped.new_tags) if mapped else [],
        "character_count": len(mapped.characters) if mapped else 0,
        "general_count": len(mapped.tags) if mapped else 0,
        "http_code": result.http_code,
        "error": result.error,
    }


def load_vocabs(directory: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    missing: list[str] = []
    vocabularies: list[dict[str, str]] = []
    for name in ("character_vocab.json", "general_vocab.json"):
        path = directory / name
        if path.is_file():
            vocabularies.append(load_vocab(path))
        else:
            missing.append(str(path))
            vocabularies.append({})
    return vocabularies[0], vocabularies[1], missing


# --------------------------------------------------------------------------- reporting


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "n/a"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_report(
    config: RecoveryConfig,
    stats: Mapping[str, Any],
    run: Mapping[str, Any],
) -> str:
    """Aggregate markdown; counts and field names only, never tag values."""
    statuses = stats.get("statuses", {})
    hits = int(stats.get("hits", 0))
    misses = int(stats.get("misses", 0))
    decided = hits + misses
    hit_rate = (hits / decided * 100.0) if decided else 0.0
    character_rate = (int(stats.get("hit_with_character", 0)) / hits * 100.0) if hits else 0.0
    artist_rate = (int(stats.get("hit_with_artist", 0)) / hits * 100.0) if hits else 0.0
    general_rate = (int(stats.get("hit_with_general", 0)) / hits * 100.0) if hits else 0.0

    lines = [
        "# Booru MD5 recovery",
        "",
        f"Generated: {utc_now()}",
        "",
        "Counts and field names only. No tag values or captions are reproduced.",
        "",
        "## Scope",
        "",
        f"- Pool root: `{config.pool_root}`",
        f"- Eligible still images: **{int(run.get('eligible', 0)):,}** "
        "(JPG/PNG/WebP originals; `.thumbs`, GIF and video excluded)",
        f"- Selected this run: {int(run.get('selected', 0)):,}",
        f"- Processed this run: {int(run.get('processed', 0)):,}",
        f"- Skipped as already terminal: {int(run.get('skipped', 0)):,}",
        f"- Site: `{config.site}` (Danbooru only; Gelbooru/Konachan not queried)",
        f"- Cache: `{config.cache_path}`",
        f"- Output JSONL: `{config.out_path}`",
        f"- Exit reason: {run.get('exit_reason', 'ok')}",
        "",
        "## Coverage (cumulative cache)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Posts probed | {sum(int(value) for value in statuses.values()):,} |",
        f"| Hits | {hits:,} |",
        f"| Misses | {misses:,} |",
        f"| Auth blocked | {int(stats.get('auth_blocked', 0)):,} |",
        f"| Transient errors (retried later) | {int(stats.get('errors', 0)):,} |",
        f"| Hit rate (hit / (hit+miss)) | **{hit_rate:.1f}%** |",
        f"| Hits with non-empty character field | {int(stats.get('hit_with_character', 0)):,} "
        f"({character_rate:.1f}% of hits) |",
        f"| Hits with non-empty artist field | {int(stats.get('hit_with_artist', 0)):,} "
        f"({artist_rate:.1f}% of hits) |",
        f"| Hits with non-empty general field | {int(stats.get('hit_with_general', 0)):,} "
        f"({general_rate:.1f}% of hits) |",
        f"| Character tag occurrences (mapped + source-only) | "
        f"{int(stats.get('character_tags', 0)):,} |",
        f"| General tag occurrences (mapped + source-only) | "
        f"{int(stats.get('general_tags', 0)):,} |",
        f"| Source-only character tag occurrences | {int(stats.get('new_character_tags', 0)):,} |",
        f"| Source-only general tag occurrences | {int(stats.get('new_general_tags', 0)):,} |",
        f"| Posts stored | {int(stats.get('posts', 0)):,} |",
        f"| Files with MD5 | {int(stats.get('files_with_md5', 0)):,} / "
        f"{int(stats.get('files', 0)):,} |",
        "",
        "## Rating distribution (hits)",
        "",
    ]
    ratings = stats.get("ratings", {})
    if ratings:
        labels = {"g": "general", "s": "sensitive", "q": "questionable", "e": "explicit"}
        lines += ["| Rating | Count |", "| --- | ---: |"]
        for key, value in sorted(ratings.items(), key=lambda item: (-int(item[1]), item[0])):
            lines.append(f"| {labels.get(key, key)} | {int(value):,} |")
    else:
        lines.append("No ratings recorded yet.")

    lines += ["", "## Source hosts (hits)", ""]
    hosts = stats.get("source_hosts", {})
    if hosts:
        lines += ["| Host | Count |", "| --- | ---: |"]
        for key, value in hosts.items():
            lines.append(f"| `{key}` | {int(value):,} |")
    else:
        lines.append("No source hosts recorded yet.")

    elapsed = float(run.get("elapsed_seconds", 0.0))
    rate = float(run.get("rate", 0.0))
    remaining = int(run.get("remaining", 0))
    eta = (remaining / rate) if rate > 0 else None
    lines += [
        "",
        "## Throughput",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Elapsed this run | {_format_duration(elapsed)} |",
        f"| Rows this run | {int(run.get('processed', 0)):,} |",
        f"| Throughput | {rate:.3f} rows/s ({rate * 3600:.0f} rows/h) |",
        f"| Remaining selected | {remaining:,} |",
        f"| ETA | {_format_duration(eta)} |",
        f"| First request | {stats.get('first_fetched_at') or 'n/a'} |",
        f"| Last request | {stats.get('last_fetched_at') or 'n/a'} |",
        "",
        "## Notes",
        "",
        "- Artist labels are the deterministic parent-directory handles; booru "
        "artist tags are stored separately and never override them.",
        "- Misses are unresolved, not negative labels: they may be unindexed, "
        "deleted or re-encoded files.",
        "- Gelbooru (HTTP 401) and Konachan (HTTP 403) are not queried. Yande.re "
        "is optional and was non-additive in the 50-image probe.",
        "- This pass downloads no images and modifies no pool/dataset/cache files.",
        "",
    ]
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- run


def _log(config: RecoveryConfig, message: str) -> None:
    if config.quiet:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def _checkpoint(
    config: RecoveryConfig,
    cache: BooruCache,
    run: dict[str, Any],
    *,
    total_pending: int,
    started: float,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    processed = int(run["processed"])
    run["elapsed_seconds"] = elapsed
    run["rate"] = processed / elapsed if processed else 0.0
    run["remaining"] = max(total_pending - processed, 0)
    _atomic_write_text(config.report_path, render_report(config, cache.stats(config.site), run))


def run_recovery(
    config: RecoveryConfig,
    *,
    client: DanbooruClient | None = None,
    cache: BooruCache | None = None,
) -> dict[str, Any]:
    if config.site not in SUPPORTED_SITES:
        raise RecoveryError(f"unsupported site {config.site!r}; supported: {SUPPORTED_SITES}")

    started = time.monotonic()
    images = list(iter_pool_images(config.pool_root))
    eligible = len(images)
    selected = spread_select(images, config.limit) if config.spread and config.limit else images
    if config.limit is not None and not config.spread:
        selected = selected[: config.limit]

    run: dict[str, Any] = {
        "eligible": eligible,
        "selected": len(selected),
        "processed": 0,
        "skipped": 0,
        "hits": 0,
        "misses": 0,
        "errors": 0,
        "started_at": utc_now(),
        "exit_reason": "ok",
        "exit_code": EXIT_OK,
        "elapsed_seconds": 0.0,
        "rate": 0.0,
        "remaining": 0,
    }

    if config.dry_run:
        by_ext: dict[str, int] = {}
        for image in selected:
            suffix = image.path.suffix.lower()
            by_ext[suffix] = by_ext.get(suffix, 0) + 1
        run["by_extension"] = dict(sorted(by_ext.items()))
        run["bytes"] = sum(image.size for image in selected)
        run["exit_reason"] = "dry-run"
        return run

    character_vocab, general_vocab, missing_vocabs = load_vocabs(config.vocab_dir)
    if missing_vocabs:
        _log(config, f"WARNING missing vocabularies: {', '.join(missing_vocabs)}; "
                     "all source tags will be recorded as source-only")

    owns_cache = cache is None
    cache = cache or BooruCache(config.cache_path)
    try:
        if config.resume and cache.blocked_code(config.site) is not None:
            code = cache.blocked_code(config.site)
            _log(config, f"site {config.site} is latched auth_blocked (HTTP {code}); "
                         "rerun with --clear-block after credentials/access change")
            run["exit_reason"] = f"auth_blocked (HTTP {code})"
            run["exit_code"] = EXIT_ALREADY_BLOCKED
            _checkpoint(config, cache, run, total_pending=0, started=started)
            return run

        done: set[str] = set()
        if config.resume:
            repair_jsonl_tail(config.out_path)
            jsonl_keys = read_jsonl_keys(config.out_path, config.site)
            terminal = cache.terminal_sha256(config.site)
            run["reconciled"] = reconcile_missing_rows(
                cache, config, character_vocab, general_vocab, jsonl_keys)
            if run["reconciled"]:
                _log(config, f"reconciled {run['reconciled']} terminal cache row(s) "
                             "missing from the JSONL")
            done = jsonl_keys | terminal
        remaining_images = [image for image in selected if image.sha256 not in done]
        run["skipped"] = len(selected) - len(remaining_images)
        pending = remaining_images
        if config.limit is not None and not config.spread:
            pending = pending[: config.limit]
        total_pending = len(pending)
        if not pending:
            run["exit_reason"] = "nothing to do (all selected rows terminal)"

        client = client or DanbooruClient(
            min_interval=config.min_interval,
            timeout=config.timeout,
            max_attempts=config.max_attempts,
        )
        if client.blocked_code is not None:
            cache.mark_blocked(config.site, client.blocked_code)

        config.out_path.parent.mkdir(parents=True, exist_ok=True)
        consecutive_errors = 0
        last_checkpoint = 0
        with open(config.out_path, "a", encoding="utf-8") as output:
            for index, image in enumerate(pending, start=1):
                md5 = cache.get_md5(image.sha256)
                if md5 is None:
                    md5 = file_md5(image.path)
                    cache.record_file(
                        image.sha256, path=str(image.path), artist=image.artist,
                        size=image.size, md5=md5,
                    )

                result = client.lookup(md5)
                row = build_row(
                    image, md5=md5, site=config.site, result=result,
                    character_vocab=character_vocab, general_vocab=general_vocab,
                )

                if result.status in (STATUS_HIT, STATUS_MISS, STATUS_AUTH_BLOCKED):
                    # Output first: a crash can lose cache state (resume re-queries
                    # or reconciles) but must never lose a completed lookup.
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()

                if result.status == STATUS_HIT and result.post is not None:
                    post = result.post
                    if post.post_id is not None:
                        cache.store_post(
                            config.site, post.post_id,
                            json.dumps(dict(post.raw), ensure_ascii=False),
                        )
                    cache.record_probe(
                        image.sha256,
                        site=config.site,
                        status=result.status,
                        http_code=result.http_code,
                        hit=True,
                        post_id=post.post_id,
                        rating=post.rating_raw,
                        has_character=post.has_character,
                        has_artist=post.has_artist,
                        has_general=post.has_general,
                        character_count=row["character_count"],
                        general_count=row["general_count"],
                        new_character_count=len(row["new_character"]),
                        new_general_count=len(row["new_tags"]),
                        api_md5=post.md5,
                        file_size=post.file_size,
                        width=post.width,
                        height=post.height,
                        source_host=row["source_host"],
                    )
                    run["hits"] += 1
                    run["processed"] += 1
                    consecutive_errors = 0
                elif result.status == STATUS_MISS:
                    cache.record_probe(
                        image.sha256, site=config.site, status=result.status,
                        http_code=result.http_code, error=result.error,
                    )
                    run["misses"] += 1
                    run["processed"] += 1
                    consecutive_errors = 0
                elif result.status == STATUS_AUTH_BLOCKED:
                    cache.record_probe(
                        image.sha256, site=config.site, status=result.status,
                        http_code=result.http_code, error=result.error,
                    )
                    cache.mark_blocked(config.site, result.http_code)
                    run["processed"] += 1
                    run["errors"] += 1
                    os.fsync(output.fileno())
                    run["exit_reason"] = f"auth_blocked (HTTP {result.http_code})"
                    run["exit_code"] = EXIT_BLOCKED
                    _log(config, f"STOP: HTTP {result.http_code} for site {config.site}; "
                                 "site latched auth_blocked")
                    break
                else:
                    cache.record_probe(
                        image.sha256, site=config.site, status=result.status,
                        http_code=result.http_code, error=result.error,
                    )
                    run["errors"] += 1
                    consecutive_errors += 1
                    _log(config, f"error sha256={image.sha256[:12]} http={result.http_code} "
                                 f"kind={result.error} consecutive={consecutive_errors}")

                if result.status == STATUS_ERROR:
                    if result.http_code == 429:
                        run["exit_reason"] = "stopped: repeated HTTP 429 rate limit"
                        run["exit_code"] = EXIT_STOPPED
                        _log(config, "STOP: HTTP 429 persisted after retries; resume later")
                        break
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        run["exit_reason"] = "stopped: consecutive transient errors"
                        run["exit_code"] = EXIT_STOPPED
                        _log(config, f"STOP: {consecutive_errors} consecutive errors")
                        break

                checkpoint = run["processed"] // config.progress_every
                if run["processed"] and checkpoint > last_checkpoint:
                    last_checkpoint = checkpoint
                    os.fsync(output.fileno())
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = run["processed"] / elapsed
                    remaining = len(pending) - index
                    eta = remaining / rate if rate > 0 else None
                    _log(
                        config,
                        f"processed={run['processed']}/{len(pending)} "
                        f"hits={run['hits']} misses={run['misses']} errors={run['errors']} "
                        f"rate={rate:.2f}/s elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(eta)}",
                    )
                    _checkpoint(config, cache, run, total_pending=len(pending), started=started)

            os.fsync(output.fileno())

        _checkpoint(config, cache, run, total_pending=total_pending, started=started)
        if not config.quiet:
            _log(
                config,
                f"finished reason={run['exit_reason']} processed={run['processed']} "
                f"hits={run['hits']} misses={run['misses']} errors={run['errors']} "
                f"rate={run['rate']:.2f}/s",
            )
        return run
    finally:
        if owns_cache:
            cache.close()


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL,
                        help=f"artist pool images root (default: {DEFAULT_POOL})")
    parser.add_argument("--site", choices=SUPPORTED_SITES, default="danbooru")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum images to process in this run")
    parser.add_argument("--spread", action="store_true",
                        help="with --limit, pick images evenly spaced across the pool")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip rows already terminal in JSONL/cache (default: on)")
    parser.add_argument("--dry-run", action="store_true", help="scan and report only; no HTTP/DB writes")
    parser.add_argument("--clear-block", action="store_true",
                        help="clear a persisted auth_blocked latch before running")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--min-interval", type=float, default=1.2,
                        help="minimum seconds between HTTP requests (default: 1.2)")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RecoveryConfig(
        pool_root=args.pool,
        cache_path=args.cache,
        out_path=args.out,
        report_path=args.report,
        vocab_dir=args.vocab_dir,
        site=args.site,
        limit=args.limit,
        spread=args.spread,
        resume=args.resume,
        dry_run=args.dry_run,
        progress_every=max(1, args.progress_every),
        min_interval=args.min_interval,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        quiet=args.quiet,
    )
    if args.clear_block and not config.dry_run:
        with BooruCache(config.cache_path) as cache:
            cache.clear_blocked(config.site)
            _log(config, f"cleared auth_blocked latch for {config.site}")
    try:
        run = run_recovery(config)
    except (RecoveryError, BooruError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if config.dry_run:
        print(json.dumps(run, indent=2, sort_keys=True))
    return int(run.get("exit_code", EXIT_OK))


if __name__ == "__main__":
    raise SystemExit(main())
