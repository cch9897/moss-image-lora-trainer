#!/usr/bin/env python
"""Merge booru MD5 recovery labels into staging/tagged metadata for finalize.

The join is strictly by image SHA-256 (the pool filename). A booru record with
``hit=false`` (miss / error / auth block) is unresolved, never a negative label:
the row keeps tagger labels, marks ``character_missing`` when no character
survives, and records ``booru_unresolved`` for review.

Precedence:
- artist: always the directory-derived ``artist``/``artists``; ``booru_artist``
  is audit-only and never overrides;
- character: booru characters on a hit, else tagger characters, else empty +
  ``character_missing=true`` (no VLM auto-label);
- tags: booru general + tagger general union, deduped, characters/artists/rating
  removed, capped (default 40), with per-row provenance and conflict counts;
- rating: booru rating when present, else the tagger rating, with
  ``rating_source``.

``layout_ocr``/``caption_original``/``source_path``/image metadata are copied
through so the output is a drop-in ``--base`` for ``scripts/finalize_dataset.py``.
Reports contain counts and field names only, never tag values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.annotations import (  # noqa: E402
    NEGATIVE_TAG_KEYS,
    UNKNOWN_ARTIST,
    normalize_name,
    normalize_tag_key,
)
from moss_lora.booru import RATING_LABELS, artist_key, normalize_rating  # noqa: E402

DEFAULT_MAX_TAGS = 40
RATING_VALUES = frozenset(RATING_LABELS.values())
DEFAULT_BOORU = Path("/mnt/data/train_test/booru_recovered.jsonl")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MergeError(RuntimeError):
    """Invalid merge input or unwritable output."""


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


def write_jsonl_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


# --------------------------------------------------------------------------- indexing


def row_sha256(row: Mapping[str, Any]) -> str | None:
    """Strict SHA-256 join key; never falls back to paths or file names."""
    for key in ("sha256", "image_sha256"):
        value = row.get(key)
        if isinstance(value, str):
            digest = value.strip().lower()
            if _SHA256_RE.fullmatch(digest):
                return digest
    return None


def _index_keys(row: Mapping[str, Any]) -> Iterable[str]:
    for key in ("sha256", "image_sha256", "source_path", "file_name"):
        value = row.get(key)
        if isinstance(value, str) and value:
            yield value


def index_booru(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], int]:
    """Index by SHA-256; a hit wins over a later miss when duplicates exist."""
    index: dict[str, Mapping[str, Any]] = {}
    duplicates = 0
    for row in rows:
        digest = row_sha256(row)
        if digest is None:
            continue
        previous = index.get(digest)
        if previous is not None:
            duplicates += 1
            if previous.get("hit") is True and row.get("hit") is not True:
                continue
        index[digest] = row
    return index, duplicates


def index_ocr(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        layout = row.get("layout_ocr") or row.get("layout")
        if not isinstance(layout, Mapping):
            continue
        for key in _index_keys(row):
            index.setdefault(key, layout)
    return index


def index_raw_annotations(directory: str | Path | None) -> dict[str, Mapping[str, Any]]:
    if directory is None:
        return {}
    root = Path(directory)
    if not root.is_dir():
        raise MergeError(f"raw annotations directory not found: {root}")
    index: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        layout = payload.get("ocr")
        if not isinstance(layout, Mapping):
            continue
        candidates = [str(payload.get("image_sha256") or ""), path.stem]
        for key in candidates:
            if key:
                index.setdefault(key, layout)
    return index


# --------------------------------------------------------------------------- row merge


def _clean_values(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = normalize_name(str(value))
        key = normalize_tag_key(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _row_artists(row: Mapping[str, Any]) -> list[str]:
    values = row.get("artists")
    artists = _clean_values(values) if isinstance(values, list) else []
    if not artists:
        artist = row.get("artist")
        if isinstance(artist, str) and artist.strip():
            artists = _clean_values([artist])
    return [value for value in artists if value.lower() != UNKNOWN_ARTIST]


def _row_characters(row: Mapping[str, Any]) -> list[str]:
    values = row.get("character")
    if not isinstance(values, list):
        values = row.get("characters")
    return _clean_values(values)


def _row_tags(row: Mapping[str, Any]) -> list[str]:
    return _clean_values(row.get("tags"))


def _booru_hit(record: Mapping[str, Any] | None) -> bool:
    return bool(record is not None and record.get("hit") is True and record.get("status") == "hit")


def _tag_union(
    booru_tags: Sequence[str],
    tagger_tags: Sequence[str],
    forbidden: set[str],
    *,
    max_tags: int,
) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    seen: set[str] = set()
    origins: dict[str, str] = {}
    counts: Counter[str] = Counter()

    def add(values: Sequence[str], origin: str) -> None:
        for value in values:
            clean = normalize_name(value)
            key = normalize_tag_key(clean)
            if not key:
                counts["empty"] += 1
                continue
            if key in seen:
                counts["duplicate"] += 1
                continue
            if key in NEGATIVE_TAG_KEYS:
                counts["negative"] += 1
                continue
            if key in forbidden:
                counts["forbidden_identity_or_rating"] += 1
                continue
            if len(selected) >= max_tags:
                counts["cap"] += 1
                continue
            seen.add(key)
            origins[key] = origin
            selected.append(clean)

    add(booru_tags, "booru")
    add(tagger_tags, "tagger")

    booru_keys = {normalize_tag_key(value) for value in booru_tags}
    tagger_keys = {normalize_tag_key(value) for value in tagger_tags}
    conflicts = {
        "shared": len(booru_keys & tagger_keys),
        "booru_only": len(booru_keys - tagger_keys),
        "tagger_only": len(tagger_keys - booru_keys),
        "selected_booru": sum(1 for value in origins.values() if value == "booru"),
        "selected_tagger": sum(1 for value in origins.values() if value == "tagger"),
        **dict(counts),
    }
    return selected, dict(conflicts)


def merge_row(
    base: Mapping[str, Any],
    booru: Mapping[str, Any] | None,
    *,
    layout_ocr: Mapping[str, Any] | None = None,
    max_tags: int = DEFAULT_MAX_TAGS,
) -> dict[str, Any]:
    out = dict(base)
    digest = row_sha256(base)
    hit = _booru_hit(booru)

    artists = _row_artists(base)
    tagger_characters = _row_characters(base)
    tagger_tags = _row_tags(base)
    booru_characters = _clean_values(booru.get("characters")) if hit else []
    booru_tags = _clean_values(booru.get("tags")) if hit else []
    booru_rating = normalize_rating(booru.get("rating") or booru.get("rating_label")) if hit else None
    tagger_rating = normalize_rating(base.get("rating"))

    characters = booru_characters or tagger_characters
    if booru_characters:
        character_source = "booru"
    elif tagger_characters:
        character_source = "tagger"
    else:
        character_source = "none"

    forbidden = {normalize_tag_key(value) for value in characters}
    forbidden |= {normalize_tag_key(value) for value in tagger_characters}
    artist_names = list(artists)
    if hit:
        artist_names.extend(_clean_values(booru.get("booru_artist")))
    for artist_name in artist_names:
        forbidden.add(normalize_tag_key(artist_name))
        forbidden.add(artist_key(artist_name))
    forbidden |= RATING_VALUES

    tags, conflicts = _tag_union(booru_tags, tagger_tags, forbidden, max_tags=max_tags)
    selected_keys = {normalize_tag_key(value) for value in tags}
    if conflicts.get("selected_booru") and conflicts.get("selected_tagger"):
        tag_source = "booru+tagger"
    elif conflicts.get("selected_booru"):
        tag_source = "booru"
    elif conflicts.get("selected_tagger"):
        tag_source = "tagger"
    else:
        tag_source = "none"

    rating = booru_rating or tagger_rating
    rating_source = "booru" if booru_rating else ("tagger" if tagger_rating else "none")

    status = str(booru.get("status") or "unknown") if booru is not None else "no_record"
    booru_unresolved = not hit

    out["sha256"] = digest or base.get("sha256")
    out["character"] = list(characters)
    if "characters" in base:
        out["characters"] = list(characters)
    out["character_missing"] = not characters
    out["tags"] = list(tags)
    out["rating"] = rating
    out["character_source"] = character_source
    out["tag_source"] = tag_source
    out["rating_source"] = rating_source
    out["tag_conflicts"] = conflicts
    out["booru_hit"] = hit
    out["booru_status"] = status
    out["booru_unresolved"] = booru_unresolved
    out["booru_post_id"] = booru.get("post_id") if hit else None
    out["booru_source"] = booru.get("source") if hit else None
    out["booru_source_host"] = booru.get("source_host") if hit else None
    out["booru_artist"] = list(booru.get("booru_artist") or []) if hit else []
    out["booru_artist_match"] = bool(booru.get("artist_match")) if hit else False
    out["booru_md5"] = booru.get("md5") if booru is not None else None
    out["booru_api_md5"] = booru.get("api_md5") if hit else None
    out["booru_new_character"] = list(booru.get("new_character") or []) if hit else []
    out["booru_new_tags"] = list(booru.get("new_tags") or []) if hit else []

    confidence = base.get("tag_confidence")
    if isinstance(confidence, Mapping):
        out["tag_confidence"] = {
            str(key): value for key, value in confidence.items()
            if normalize_tag_key(str(key)) in selected_keys
        }

    if layout_ocr is not None and not isinstance(out.get("layout_ocr"), Mapping):
        out["layout_ocr"] = dict(layout_ocr)
    return out


def merge_metadata(
    base_rows: Sequence[Mapping[str, Any]],
    booru_rows: Sequence[Mapping[str, Any]],
    *,
    ocr_rows: Sequence[Mapping[str, Any]] = (),
    raw_annotations: str | Path | None = None,
    max_tags: int = DEFAULT_MAX_TAGS,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    booru_index, duplicate_booru = index_booru(booru_rows)
    ocr_index = index_ocr(ocr_rows)
    ocr_index.update(index_raw_annotations(raw_annotations))

    selected = list(base_rows)
    if limit is not None:
        selected = selected[:limit]
    base_shas = {digest for row in base_rows if (digest := row_sha256(row)) is not None}

    rows: list[dict[str, Any]] = []
    by_artist: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    tag_conflict_totals: Counter[str] = Counter()

    for base in selected:
        digest = row_sha256(base)
        booru = booru_index.get(digest) if digest is not None else None
        layout_ocr = None
        for key in (digest, base.get("source_path"), base.get("file_name")):
            if isinstance(key, str) and key in ocr_index:
                layout_ocr = ocr_index[key]
                break
        row = merge_row(base, booru, layout_ocr=layout_ocr, max_tags=max_tags)
        rows.append(row)

        artist = str(row.get("artist") or UNKNOWN_ARTIST)
        bucket = by_artist.setdefault(artist, Counter())
        for name, counter in (("totals", totals), (f"artist:{artist}", bucket)):
            target = totals if name == "totals" else counter
            target["rows"] += 1
            target["booru_hit"] += int(row["booru_hit"])
            target["booru_unresolved"] += int(row["booru_unresolved"])
            target["character_from_booru"] += int(row["character_source"] == "booru")
            target["character_from_tagger"] += int(row["character_source"] == "tagger")
            target["character_missing"] += int(row["character_missing"])
            target["tagger_fallback"] += int(row["character_source"] == "tagger")
            target["tags_from_booru_only"] += int(row["tag_source"] == "booru")
            target["tags_from_tagger_only"] += int(row["tag_source"] == "tagger")
            target["tags_from_both"] += int(row["tag_source"] == "booru+tagger")
            target["tags_from_none"] += int(row["tag_source"] == "none")
            target["rating_from_booru"] += int(row["rating_source"] == "booru")
            target["rating_from_tagger"] += int(row["rating_source"] == "tagger")
            target["rating_from_none"] += int(row["rating_source"] == "none")
            target["layout_ocr"] += int(isinstance(row.get("layout_ocr"), Mapping))

        for key, value in row["tag_conflicts"].items():
            if isinstance(value, int):
                tag_conflict_totals[key] += value

    booru_only = len(set(booru_index) - base_shas)
    statuses = Counter(
        str(record.get("status") or "unknown") for record in booru_index.values()
    )
    report: dict[str, Any] = {
        "rows": len(rows),
        "base_rows": len(base_rows),
        "booru_records": len(booru_index),
        "booru_duplicate_records": duplicate_booru,
        "booru_only_records": booru_only,
        "booru_hit": int(totals["booru_hit"]),
        "booru_unresolved": int(totals["booru_unresolved"]),
        "booru_statuses": dict(sorted(statuses.items())),
        "character_from_booru": int(totals["character_from_booru"]),
        "character_from_tagger": int(totals["character_from_tagger"]),
        "character_missing": int(totals["character_missing"]),
        "tagger_fallback": int(totals["tagger_fallback"]),
        "tags_from_booru_only": int(totals["tags_from_booru_only"]),
        "tags_from_tagger_only": int(totals["tags_from_tagger_only"]),
        "tags_from_both": int(totals["tags_from_both"]),
        "tags_from_none": int(totals["tags_from_none"]),
        "rating_from_booru": int(totals["rating_from_booru"]),
        "rating_from_tagger": int(totals["rating_from_tagger"]),
        "rating_from_none": int(totals["rating_from_none"]),
        "layout_ocr_rows": int(totals["layout_ocr"]),
        "max_tags": max_tags,
        "tag_conflicts": dict(sorted(tag_conflict_totals.items())),
        "by_artist": {artist: dict(sorted(counter.items()))
                      for artist, counter in sorted(by_artist.items())},
        "hits_by_artist": {
            artist: counter["booru_hit"] for artist, counter in sorted(by_artist.items())
        },
    }
    return rows, report


# --------------------------------------------------------------------------- reports


def render_markdown(report: Mapping[str, Any]) -> str:
    """Counts/field names only; no tag values are ever included."""
    rows = int(report["rows"])
    hits = int(report["booru_hit"])
    lines = [
        "# Booru merge report",
        "",
        "Counts and field names only. No tag values or captions are reproduced.",
        "",
        "## Coverage",
        "",
        "| Metric | Count | Share |",
        "| --- | ---: | ---: |",
        f"| Rows | {rows:,} | 100% |",
        f"| Booru hit | {hits:,} | {_share(hits, rows)} |",
        f"| Booru unresolved (miss/error/no record) | {int(report['booru_unresolved']):,} | "
        f"{_share(int(report['booru_unresolved']), rows)} |",
        f"| Character from booru | {int(report['character_from_booru']):,} | "
        f"{_share(int(report['character_from_booru']), rows)} |",
        f"| Character from tagger | {int(report['character_from_tagger']):,} | "
        f"{_share(int(report['character_from_tagger']), rows)} |",
        f"| Character still missing | {int(report['character_missing']):,} | "
        f"{_share(int(report['character_missing']), rows)} |",
        f"| Tagger fallback (no booru hit) | {int(report['tagger_fallback']):,} | "
        f"{_share(int(report['tagger_fallback']), rows)} |",
        f"| Tags from booru only | {int(report['tags_from_booru_only']):,} | "
        f"{_share(int(report['tags_from_booru_only']), rows)} |",
        f"| Tags from booru + tagger | {int(report['tags_from_both']):,} | "
        f"{_share(int(report['tags_from_both']), rows)} |",
        f"| Tags from tagger only | {int(report['tags_from_tagger_only']):,} | "
        f"{_share(int(report['tags_from_tagger_only']), rows)} |",
        f"| Tags from neither | {int(report['tags_from_none']):,} | "
        f"{_share(int(report['tags_from_none']), rows)} |",
        f"| Rating from booru | {int(report['rating_from_booru']):,} | "
        f"{_share(int(report['rating_from_booru']), rows)} |",
        f"| Rating from tagger | {int(report['rating_from_tagger']):,} | "
        f"{_share(int(report['rating_from_tagger']), rows)} |",
        f"| Rows carrying layout_ocr | {int(report['layout_ocr_rows']):,} | "
        f"{_share(int(report['layout_ocr_rows']), rows)} |",
        "",
        "## Booru status records",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        *[f"| {status} | {count:,} |" for status, count in report["booru_statuses"].items()],
        "",
        "## Tag conflicts",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
        *[f"| {name} | {count:,} |" for name, count in report["tag_conflicts"].items()],
        "",
        "## By artist",
        "",
        "| Artist | Rows | Booru hit | From booru | From tagger | Missing | Tagger fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for artist, values in report["by_artist"].items():
        lines.append(
            f"| `{artist}` | {int(values['rows']):,} | {int(values['booru_hit']):,} | "
            f"{int(values['character_from_booru']):,} | "
            f"{int(values['character_from_tagger']):,} | "
            f"{int(values['character_missing']):,} | "
            f"{int(values['tagger_fallback']):,} |"
        )
    lines.append("")
    return "\n".join(lines)


def _share(value: int, total: int) -> str:
    return f"{(value / total * 100.0):.1f}%" if total else "n/a"


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="metadata_staging.jsonl or metadata_tagged.jsonl")
    parser.add_argument("--booru", type=Path, default=DEFAULT_BOORU,
                        help=f"booru_recovered.jsonl (default: {DEFAULT_BOORU})")
    parser.add_argument("--out", type=Path, required=True,
                        help="merged metadata JSONL consumed by finalize --base")
    parser.add_argument("--report", type=Path, default=None,
                        help="JSON report (default: <out stem>_report.json)")
    parser.add_argument("--md-report", type=Path, default=None,
                        help="Markdown report (default: <out stem>_report.md)")
    parser.add_argument("--ocr", type=Path, default=None,
                        help="Optional OCR metadata JSONL (layout_ocr rows)")
    parser.add_argument("--raw-annotations", type=Path, default=None,
                        help="Optional finalize raw_annotations/ directory")
    parser.add_argument("--max-tags", type=int, default=DEFAULT_MAX_TAGS)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tags < 1:
        raise SystemExit("--max-tags must be >= 1")
    report_path = args.report or args.out.with_name(args.out.stem + "_report.json")
    md_path = args.md_report or args.out.with_name(args.out.stem + "_report.md")
    try:
        base_rows = read_jsonl(args.input)
        booru_rows = read_jsonl(args.booru)
        ocr_rows = read_jsonl(args.ocr) if args.ocr is not None else []
        rows, report = merge_metadata(
            base_rows, booru_rows, ocr_rows=ocr_rows,
            raw_annotations=args.raw_annotations,
            max_tags=args.max_tags, limit=args.limit,
        )
        write_jsonl_atomic(args.out, rows)
        atomic_write_json(report_path, report)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = md_path.with_name(md_path.name + f".tmp{os.getpid()}")
        tmp.write_text(render_markdown(report), encoding="utf-8")
        os.replace(tmp, md_path)
    except (OSError, MergeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "rows": report["rows"],
        "booru_hit": report["booru_hit"],
        "character_from_booru": report["character_from_booru"],
        "character_from_tagger": report["character_from_tagger"],
        "character_missing": report["character_missing"],
        "report": str(report_path),
        "out": str(args.out),
    }), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
