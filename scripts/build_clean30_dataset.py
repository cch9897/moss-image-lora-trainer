#!/usr/bin/env python
"""Build the clean30 top-256-artist dataset (current + expansion).

Policy (see /mnt/data/train_test/style_artist_strategy.md and the run brief):

- artist pool: top 256 canonical artists by combined image count (>=28 images);
  ``unknown`` is a count bucket and never emitted as an ``artist:`` tag;
- character pool: base identities seen with >=2 distinct artists, capped at 2
  per image, ordered by corpus frequency (costume qualifiers stripped);
- current rows keep their existing tag/prose text and OCR layout; only the
  deterministic artist/character prefix is rewritten (case/space canonical and
  unknown artist removed);
- expansion rows use tags + deterministic prefix, plus the 727 valid/repaired
  VLM captions from the 1k pilot; invalid VLM attempts never contribute text;
- OCR text boxes (max 8) and Deepghs detector boxes (max 4, high confidence)
  form the bucket-space layout variant;
- five cache variants: original, long, medium, short, layout (layout only where
  OCR/detector boxes or an existing layout exist);
- exact-duplicate images (sha256) are dropped, preferring rows with a known
  artist.

The script only reads metadata/annotation JSONL and images for hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.annotations import (  # noqa: E402
    BUCKET_SPACE,
    TRIGGER_DEFAULT,
    UNKNOWN_ARTIST,
    Layout,
    LayoutItem,
    build_prefix,
    compose_caption,
    dedupe_tags,
    layout_to_bucket,
    layout_to_normalized,
    parse_normalized_box,
    parse_vlm_annotation,
)
from moss_lora.ocr import BucketGeometry  # noqa: E402

COSTUME_RE = re.compile(
    r"\((?:[^()]*costume[^()]*|bunny|swimsuit|school_uniform|maid|nurse|kimono|"
    r"yukata|cheerleader|wedding|wedding_dress|dress|casual|formal|armor|plugsuit|"
    r"track_suit|gym_uniform|pajamas|santa|halloween|valentine|china_dress|qipao|"
    r"idol|police|military|suit|sportswear|lingerie|underwear|bikini|leotard|"
    r"bodysuit|ninja|samurai|pirate|vampire|devil|angel|mermaid|demon|new_year|"
    r"christmas|birthday|anniversary|race_queen|waitress|bunny_girl|"
    r"1st_costume|2nd_costume|3rd_costume|4th_costume|5th_costume)\)"
)
PREFIX_RE = re.compile(r"^\s*(?:artist|character)\s*:\s*", re.IGNORECASE)
TRIGGER_TOKEN_RE = re.compile(r"^\s*@[^,]*$")
MAX_LAYOUT_TEXT = 8
MAX_LAYOUT_DETECTOR = 4
DETECTOR_MIN_CONF = 0.35
OCR_MIN_CONF = 0.50
LAYOUT_ITEM_AREA_MIN = 0.0005
LAYOUT_ITEM_AREA_MAX = 0.95


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
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


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def norm_artist(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def base_char(tag: str) -> str:
    prev = None
    while prev != tag:
        prev = tag
        tag = COSTUME_RE.sub("", tag)
    return re.sub(r"_{2,}", "_", tag).strip("_")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_characters(row: Mapping[str, Any], keep: set[str], freq: Counter[str]) -> list[str]:
    values = row.get("character")
    if not isinstance(values, list):
        return []
    chars = {base_char(str(value)) for value in values if str(value).strip()}
    ranked = sorted((char for char in chars if char in keep), key=lambda char: (-freq[char], char))
    return ranked[:2]


def row_artists(row: Mapping[str, Any]) -> list[str]:
    values = row.get("artists")
    if isinstance(values, list):
        raw = [str(value) for value in values if str(value).strip()]
    else:
        raw = [str(row.get("artist") or "")]
    out: list[str] = []
    for value in raw:
        canonical = norm_artist(value)
        if canonical and canonical != UNKNOWN_ARTIST and canonical not in out:
            out.append(canonical)
    return out


def trigger_parts(row: Mapping[str, Any], keep: set[str], freq: Counter[str]) -> list[str]:
    artist = norm_artist(row.get("artist"))
    artists = [artist] if artist and artist != UNKNOWN_ARTIST else []
    return build_prefix(TRIGGER_DEFAULT, artists, row_characters(row, keep, freq),
                        unknown_if_missing=False)


def text_tokens(text: str) -> list[str]:
    """Split a deterministic comma tag line, dropping trigger/prefix tokens."""
    parts: list[str] = []
    for raw in str(text or "").replace("\n", " ").split(","):
        token = raw.strip()
        if not token or TRIGGER_TOKEN_RE.match(token) or PREFIX_RE.match(token):
            continue
        parts.append(token)
    return parts


def clean_tag_text(text: str, prefix: Sequence[str]) -> str:
    """Preserve body tags/prose and replace the deterministic prefix.

    Current captions are comma-separated tag lists and were already capped by
    ``prepare_captions``; capping again would delete the trailing prose that is
    embedded in layout summaries, so this only rewrites the prefix tokens.
    """
    tokens = text_tokens(text)
    return compose_caption(prefix, tokens)


def layout_summary_clean(summary: Mapping[str, Any], prefix: Sequence[str]) -> dict[str, Any]:
    result = dict(summary)
    caption = result.get("caption")
    if isinstance(caption, str) and caption.strip():
        result["caption"] = clean_tag_text(caption, prefix)
    else:
        result["caption"] = compose_caption(prefix)
    return result


def parse_current_layout(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def current_layout(value: Any, prefix: Sequence[str]) -> dict[str, Any] | None:
    parsed = parse_current_layout(value)
    if parsed is None:
        return None
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        return None
    result = dict(parsed)
    summary = result.get("summary")
    result["summary"] = layout_summary_clean(summary if isinstance(summary, Mapping) else {},
                                             prefix)
    result.setdefault("bbox_coord_space", BUCKET_SPACE)
    result.setdefault("bbox_format", "yxyx")
    return result


def detector_items(row: Mapping[str, Any] | None) -> list[LayoutItem]:
    if not isinstance(row, Mapping):
        return []
    raw_items = row.get("items")
    if not isinstance(raw_items, list):
        return []
    candidates: list[tuple[float, LayoutItem]] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        label = raw.get("label")
        if label not in ("object", "character"):
            continue
        try:
            confidence = float(raw.get("conf", raw.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
        if confidence < DETECTOR_MIN_CONF:
            continue
        try:
            y1, x1, y2, x2 = parse_normalized_box(
                raw.get("bbox"), min_area_ratio=LAYOUT_ITEM_AREA_MIN,
                max_area_ratio=LAYOUT_ITEM_AREA_MAX, require_integers=False)
            caption = str(raw.get("tag") or raw.get("caption") or label)
            item = LayoutItem(label=str(label), caption=caption, bbox=(y1, x1, y2, x2))
        except Exception:
            continue
        candidates.append((confidence, item))
    candidates.sort(key=lambda pair: -pair[0])
    return [item for _, item in candidates[:MAX_LAYOUT_DETECTOR]]


def ocr_text_items(ocr_row: Mapping[str, Any], bucket: tuple[int, int]) -> list[LayoutItem]:
    raw_items = ocr_row.get("items")
    if not isinstance(raw_items, list):
        return []
    half_w, half_h = bucket[0] // 2, bucket[1] // 2
    items: list[LayoutItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("low_confidence"):
            continue
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        text = str(raw.get("text") or "").strip()
        if not text or confidence < OCR_MIN_CONF:
            continue
        box = raw.get("box_bucket")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            y1, x1, y2, x2 = (int(round(float(value))) for value in box)
        except (TypeError, ValueError):
            continue
        y1, y2 = max(-half_h, min(y1, half_h - 1)), max(-half_h + 1, min(y2, half_h))
        x1, x2 = max(-half_w, min(x1, half_w - 1)), max(-half_w + 1, min(x2, half_w))
        if y2 <= y1 or x2 <= x1:
            continue
        try:
            items.append(LayoutItem(label="text", caption="visible text",
                                    bbox=(y1, x1, y2, x2), text=text))
        except Exception:
            continue
        if len(items) >= MAX_LAYOUT_TEXT:
            break
    return items


def build_expansion_layout(
    row: Mapping[str, Any],
    ocr_row: Mapping[str, Any] | None,
    detector_row: Mapping[str, Any] | None,
    prefix: Sequence[str],
    tags: Sequence[str],
    bucket: tuple[int, int],
) -> dict[str, Any] | None:
    if not isinstance(ocr_row, Mapping):
        return None
    text_items = ocr_text_items(ocr_row, bucket)
    det_items = detector_items(detector_row)
    if not text_items and not det_items:
        return None
    width = int(row.get("width") or ocr_row.get("width") or 0)
    height = int(row.get("height") or ocr_row.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    geometry_raw = ocr_row.get("bucket_geometry")
    if not isinstance(geometry_raw, Mapping):
        return None
    geometry = BucketGeometry(
        source_w=width, source_h=height, bucket_w=bucket[0], bucket_h=bucket[1],
        resize_w=int(geometry_raw["resize_w"]), resize_h=int(geometry_raw["resize_h"]),
        left=int(geometry_raw["left"]), top=int(geometry_raw["top"]),
        scale_x=float(geometry_raw["scale_x"]), scale_y=float(geometry_raw["scale_y"]),
    )
    normalized = Layout(
        summary_caption=compose_caption(prefix, tags, max_tags=40),
        items=tuple(text_items),
        summary_meta=("anime",),
        canvas="single",
        bbox_coord_space=BUCKET_SPACE,
        bbox_format="yxyx",
    )
    try:
        normalized, _ = layout_to_normalized(normalized, (width, height), bucket,
                                             geometry=geometry)
    except Exception:
        return None
    combined = Layout(
        summary_caption=normalized.summary_caption,
        items=tuple(list(normalized.items) + det_items),
        summary_meta=("anime",),
        canvas="single",
        bbox_coord_space=normalized.bbox_coord_space,
        bbox_format="yxyx",
    )
    try:
        bucket_layout, _ = layout_to_bucket(combined, (width, height), bucket,
                                            geometry=geometry)
    except Exception:
        return None
    if not bucket_layout.items:
        return None
    return bucket_layout.to_dict()


def variant_text(prefix: Sequence[str], tags: Sequence[str], cap: int,
                 prose: str = "") -> str:
    return compose_caption(prefix, tags, prose, max_tags=cap)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expansion", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-images", action="store_true",
                        help="Create hardlinked sequential image names in <output>/images")
    parser.add_argument("--max-artists", type=int, default=256)
    args = parser.parse_args(argv)

    current = read_jsonl(args.current)
    expansion = read_jsonl(args.expansion)
    all_rows = current + expansion
    print(f"current={len(current)} expansion={len(expansion)} combined={len(all_rows)}")

    artist_counts = Counter(
        norm_artist(row.get("artist") or (row.get("artists") or ["unknown"])[0])
        for row in all_rows
    )
    top_artists = [name for name, _ in artist_counts.most_common(args.max_artists)]
    artist_set = set(top_artists)
    print(f"top{args.max_artists} artists cover "
          f"{sum(artist_counts[name] for name in top_artists)} rows; "
          f"cutoff={artist_counts[top_artists[-1]]}; unknown={artist_counts['unknown']}")

    char_artists: dict[str, set[str]] = defaultdict(set)
    char_freq: Counter[str] = Counter()
    for row in all_rows:
        artist = norm_artist(row.get("artist") or "unknown")
        chars = {base_char(str(value)) for value in row.get("character") or [] if str(value).strip()}
        for char in chars:
            char_artists[char].add(artist)
            char_freq[char] += 1
    keep_characters = {char for char, artists in char_artists.items() if len(artists) >= 2}
    print(f"characters kept {len(keep_characters)}/{len(char_artists)}")

    vlm_index: dict[str, Mapping[str, Any]] = {}
    for record in read_jsonl(args.vlm):
        if record.get("status") not in ("ok", "repaired"):
            continue
        if not isinstance(record.get("annotation_validated"), Mapping):
            continue
        key = record.get("image_sha256")
        if isinstance(key, str):
            vlm_index[key] = record
    ocr_index: dict[str, Mapping[str, Any]] = {}
    for record in read_jsonl(args.ocr):
        key = record.get("sha256")
        if isinstance(key, str):
            ocr_index[key] = record
    detector_index: dict[str, Mapping[str, Any]] = {}
    for record in read_jsonl(args.detector):
        key = record.get("sha256")
        if isinstance(key, str):
            detector_index[key] = record
    print(f"vlm valid={len(vlm_index)} ocr={len(ocr_index)} detector={len(detector_index)}")

    current_root = Path(args.current).resolve().parent
    selected_current = [
        row for row in current if norm_artist(row.get("artist")) in artist_set
    ]
    selected_expansion = [
        row for row in expansion
        if norm_artist(row.get("artist") or (row.get("artists") or ["unknown"])[0]) in artist_set
    ]
    print(f"selected current={len(selected_current)} expansion={len(selected_expansion)}")

    # Resolve source paths and sha256. Current metadata has no sha256.
    current_entries: list[dict[str, Any]] = []
    seen_sha: dict[str, dict[str, Any]] = {}
    dropped_duplicates: list[dict[str, Any]] = []
    for row in selected_current:
        source = (current_root / str(row["file_name"]))
        if not source.is_file():
            raise FileNotFoundError(f"current image missing: {source}")
        sha = sha256_file(source)
        entry = {"row": row, "source": source, "sha256": sha, "source_kind": "current"}
        current_entries.append(entry)
    expansion_entries: list[dict[str, Any]] = []
    for row in selected_expansion:
        source = Path(str(row["source_path"]))
        if not source.is_file():
            raise FileNotFoundError(f"expansion image missing: {source}")
        sha = str(row["sha256"])
        entry = {"row": row, "source": source, "sha256": sha, "source_kind": "expansion"}
        expansion_entries.append(entry)

    # Exact-duplicate removal. Prefer a known artist, then more character tags,
    # then more general tags, then the earlier row.
    def duplicate_key(entry: Mapping[str, Any]) -> tuple:
        row = entry["row"]
        return (
            0 if row.get("artist") and norm_artist(row.get("artist")) != UNKNOWN_ARTIST else 1,
            -len(row.get("character") or []),
            -len(row.get("tags") or []),
        )

    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in current_entries + expansion_entries:
        by_sha[str(entry["sha256"])].append(entry)
    chosen: list[dict[str, Any]] = []
    dropped = 0
    for sha, entries in by_sha.items():
        entries.sort(key=duplicate_key)
        chosen.append(entries[0])
        for extra in entries[1:]:
            dropped += 1
            dropped_duplicates.append({
                "sha256": sha,
                "kept": str(entries[0]["row"].get("file_name")),
                "kept_kind": entries[0]["source_kind"],
                "dropped": str(extra["row"].get("file_name")),
                "dropped_kind": extra["source_kind"],
            })

    order = {id(entry): index for index, entry in enumerate(current_entries + expansion_entries)}
    chosen.sort(key=lambda entry: order[id(entry)])
    print(f"deduped rows={len(chosen)} dropped_duplicates={dropped}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    if args.stage_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, Any]] = []
    copy_rows: list[dict[str, Any]] = []
    variant_presence: Counter[str] = Counter()
    stats = Counter()
    for index, entry in enumerate(chosen):
        row = entry["row"]
        kind = entry["source_kind"]
        suffix = entry["source"].suffix.lower()
        out_name = f"images/{index + 1:06d}{suffix}"
        if args.stage_images:
            target = output / out_name
            if target.exists() or target.is_symlink():
                target.unlink()
            os.link(entry["source"], target)
        prefix = trigger_parts(row, keep_characters, char_freq)
        raw_tags = dedupe_tags(row.get("tags") or [])
        general_tags = [tag for tag in raw_tags if not PREFIX_RE.match(tag)]
        caption: dict[str, str] = {}
        layout: dict[str, Any] | None = None
        original: str
        no_artist: str

        if kind == "current":
            original = clean_tag_text(str(row.get("caption_no_artist") or ""), prefix)
            long_value = row.get("caption", {}).get("long") if isinstance(row.get("caption"), Mapping) else ""
            medium_value = row.get("caption", {}).get("medium") if isinstance(row.get("caption"), Mapping) else ""
            short_value = row.get("caption", {}).get("short") if isinstance(row.get("caption"), Mapping) else ""
            parsed_layout = current_layout(long_value, prefix)
            if parsed_layout is not None:
                layout = parsed_layout
                long_text = clean_tag_text(str(medium_value or ""), prefix)
            else:
                long_text = clean_tag_text(str(long_value or ""), prefix)
            medium_text = clean_tag_text(str(medium_value or ""), prefix)
            short_text = clean_tag_text(str(short_value or ""), prefix)
            existing_layout = current_layout(row.get("layout"), prefix)
            if layout is None and existing_layout is not None:
                layout = existing_layout
            stats["current"] += 1
        else:
            vlm = vlm_index.get(str(entry["sha256"]))
            prose = None
            if vlm is not None:
                try:
                    prose = parse_vlm_annotation(vlm["annotation_validated"])
                except Exception:
                    prose = None
            original = variant_text(prefix, general_tags, 40)
            if prose is not None:
                long_text = variant_text(prefix, general_tags, 40, prose.caption.long)
                medium_text = variant_text(prefix, general_tags, 25, prose.caption.medium)
                short_text = variant_text(prefix, general_tags, 12, prose.caption.short)
                stats["expansion_vlm"] += 1
            else:
                long_text = variant_text(prefix, general_tags, 40)
                medium_text = variant_text(prefix, general_tags, 25)
                short_text = variant_text(prefix, general_tags, 12)
                stats["expansion_tag_fallback"] += 1
            ocr_row = ocr_index.get(str(entry["sha256"]))
            det_row = detector_index.get(str(entry["sha256"]))
            bucket: tuple[int, int] = (0, 0)
            if isinstance(ocr_row, Mapping):
                raw_bucket = ocr_row.get("bucket")
                if isinstance(raw_bucket, (list, tuple)) and len(raw_bucket) == 2:
                    bucket = (int(raw_bucket[0]), int(raw_bucket[1]))
                elif isinstance(ocr_row.get("bucket_geometry"), Mapping):
                    geom = ocr_row["bucket_geometry"]
                    bucket = (int(geom.get("bucket_w") or 0), int(geom.get("bucket_h") or 0))
            layout = build_expansion_layout(
                row, ocr_row, det_row, prefix, general_tags, bucket,
            )
            if layout is not None:
                for item in layout["items"]:
                    stats["layout_items"] += 1
                    stats[f"layout_{item['label']}_items"] += 1
            stats["expansion"] += 1

        caption = {"long": long_text, "medium": medium_text, "short": short_text}
        no_artist_prefix = build_prefix(TRIGGER_DEFAULT, [], row_characters(row, keep_characters, char_freq),
                                        unknown_if_missing=False)
        no_artist = compose_caption(no_artist_prefix, general_tags, max_tags=40)
        if kind == "current":
            no_artist = clean_tag_text(str(row.get("caption_no_artist") or ""), no_artist_prefix)

        record: dict[str, Any] = {
            "file_name": out_name,
            "source_path": str(entry["source"]),
            "image_sha256": entry["sha256"],
            "source_kind": kind,
            "width": int(row.get("width") or 0) or None,
            "height": int(row.get("height") or 0) or None,
            "artist": norm_artist(row.get("artist") or "") or UNKNOWN_ARTIST,
            "artists": row_artists(row),
            "character": row_characters(row, keep_characters, char_freq),
            "tags": general_tags,
            "caption_original": original,
            "caption": caption,
            "caption_no_artist": no_artist,
            "finalize_version": "clean30-v1",
        }
        if layout is not None:
            record["layout"] = layout
        metadata.append(record)
        copy_rows.append({
            "file_name": out_name,
            "source_path": str(entry["source"]),
            "sha256": entry["sha256"],
            "source_kind": kind,
        })
        variant_presence["rows"] += 1
        variant_presence["with_layout"] += int(layout is not None)
        variant_presence["with_original"] += int(bool(original.strip()))
        variant_presence["with_long"] += int(bool(long_text.strip()))
        variant_presence["with_medium"] += int(bool(medium_text.strip()))
        variant_presence["with_short"] += int(bool(short_text.strip()))

    # Validation ------------------------------------------------------------
    problems: list[str] = []
    if len({row["file_name"] for row in metadata}) != len(metadata):
        problems.append("duplicate file_name")
    if len({row["image_sha256"] for row in metadata}) != len(metadata):
        problems.append("duplicate image_sha256")
    for row in metadata:
        for field in ("caption_original", "long", "medium", "short"):
            value = row["caption_original"] if field == "caption_original" else row["caption"][field]
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{row['file_name']}: empty {field}")
                break
        if "artist:unknown" in row["caption_original"]:
            problems.append(f"{row['file_name']}: artist:unknown")
            break
        if len(row["character"]) > 2:
            problems.append(f"{row['file_name']}: character cap")
            break
        if row.get("layout"):
            items = row["layout"].get("items") or []
            if len(items) > 24:
                problems.append(f"{row['file_name']}: layout cap")
                break

    write_jsonl(output / "metadata.jsonl", metadata)
    write_jsonl(output / "copy_manifest.jsonl", copy_rows)
    report = {
        "rows": len(metadata),
        "target_rows": 20959,
        "dropped_duplicates": dropped,
        "duplicate_details": dropped_duplicates,
        "current_rows": sum(1 for row in metadata if row["source_kind"] == "current"),
        "expansion_rows": sum(1 for row in metadata if row["source_kind"] == "expansion"),
        "artists_with_conditioning": len({
            row["artist"] for row in metadata if row["artist"] != UNKNOWN_ARTIST
        }),
        "rows_without_artist": sum(1 for row in metadata if row["artist"] == UNKNOWN_ARTIST),
        "top_artists": top_artists,
        "artist_counts": {name: artist_counts[name] for name in top_artists},
        "characters_kept": len(keep_characters),
        "characters_total": len(char_artists),
        "variant_presence": dict(variant_presence),
        "caption_stats": dict(stats),
        "problems": problems[:50],
        "problem_count": len(problems),
    }
    (output / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "rows", "current_rows", "expansion_rows", "rows_without_artist",
        "artists_with_conditioning", "characters_kept", "variant_presence",
        "caption_stats", "problem_count")}, indent=1))
    if problems:
        raise SystemExit(f"validation failed: {problems[:10]}")


if __name__ == "__main__":
    sys.exit(main())
