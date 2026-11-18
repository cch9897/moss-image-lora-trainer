#!/usr/bin/env python
"""Re-derive bucket layouts from stored raw normalized annotations (no VLM calls)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.annotations import (  # noqa: E402
    NORMALIZED_SPACE,
    AnnotationError,
    LayoutItem,
    layout_to_bucket,
    parse_normalized_box,
)
from moss_lora.cache import assign_bucket, build_buckets  # noqa: E402
from moss_lora.utils import atomic_write_bytes, atomic_write_json  # noqa: E402

METADATA_NAME = "metadata.jsonl"
REPORT_NAME = "relayout_report.json"


class RelayoutError(RuntimeError):
    """Invalid raw annotation or metadata row."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
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
    payload = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).size


def _raw_layout(raw: Mapping[str, Any]) -> Mapping[str, Any] | None:
    layout = raw.get("layout_raw")
    return layout if isinstance(layout, Mapping) else None


def _layout_from_raw(raw_layout: Mapping[str, Any]) -> Any:
    from moss_lora.annotations import Layout

    items_raw = raw_layout.get("items")
    if not isinstance(items_raw, list):
        return None
    items: list[LayoutItem] = []
    for entry in items_raw:
        if not isinstance(entry, Mapping):
            continue
        label = entry.get("label", "text")
        if label not in ("character", "object", "text"):
            continue
        try:
            bbox = parse_normalized_box(entry.get("bbox"), min_area_ratio=0.0, max_area_ratio=1.0)
        except (AnnotationError, TypeError, ValueError):
            continue
        text = entry.get("text") if label == "text" else None
        if label == "text" and (not isinstance(text, str) or not text.strip()):
            continue
        caption = entry.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            caption = "visible text" if label == "text" else label
        items.append(LayoutItem(label=label, caption=caption, bbox=bbox, text=text))
    if not items:
        return None
    summary = raw_layout.get("summary") if isinstance(raw_layout.get("summary"), Mapping) else {}
    meta = tuple(str(value) for value in summary.get("meta", ["anime"])) or ("anime",)
    canvas_value = summary.get("canvas")
    canvas = canvas_value if canvas_value in ("single", "multi_panel", "collage") else "single"
    caption = summary.get("caption") if isinstance(summary.get("caption"), str) else "anime illustration"
    return Layout(summary_caption=caption, items=tuple(items), summary_meta=meta,
                  canvas=canvas, bbox_coord_space=NORMALIZED_SPACE, bbox_format="yxyx")


def relayout_metadata(
    metadata: str | Path,
    raw_dir: str | Path,
    output: str | Path,
    *,
    images_root: str | Path | None = None,
    target_area: int = 1024 * 1024,
    limit: int | None = None,
) -> dict[str, Any]:
    metadata_path = Path(metadata)
    rows = read_jsonl(metadata_path)
    if limit is not None:
        rows = rows[:limit]
    root = Path(images_root) if images_root else metadata_path.parent
    raw_root = Path(raw_dir)
    buckets = build_buckets(target_area)
    output_rows: list[dict[str, Any]] = []
    updated = 0
    missing = 0
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            name = row.get("file_name")
            if not isinstance(name, str) or not name:
                raise RelayoutError("missing file_name")
            image = root / name
            sha = row.get("image_sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                if not image.is_file():
                    raise RelayoutError(f"image not found: {image}")
                sha = sha256_file(image)
            raw_path = raw_root / f"{sha}.json"
            if not raw_path.is_file():
                missing += 1
                output_rows.append(dict(row))
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_layout = _raw_layout(raw)
            if raw_layout is None:
                missing += 1
                output_rows.append(dict(row))
                continue
            layout = _layout_from_raw(raw_layout)
            if layout is None:
                raise RelayoutError("raw layout has no valid items")
            width, height = row.get("width"), row.get("height")
            if not (isinstance(width, int) and isinstance(height, int)):
                if not image.is_file():
                    raise RelayoutError(f"image not found for size: {image}")
                width, height = image_size(image)
            bucket = assign_bucket(width, height, buckets)
            bucket_layout, stats = layout_to_bucket(layout, (width, height), bucket)
            if not bucket_layout.items:
                raise RelayoutError("all layout items fall outside the new bucket")
            new_row = dict(row)
            new_row["layout"] = bucket_layout.to_dict()
            new_row["bucket_w"], new_row["bucket_h"] = bucket
            new_row["relayout"] = {"target_area": target_area, **stats.as_dict()}
            output_rows.append(new_row)
            updated += 1
        except Exception as error:
            errors.append({"row": index + 1, "error": f"{type(error).__name__}: {error}"})
            output_rows.append(dict(row))
    output_path = Path(output)
    if output_path.suffix.lower() != ".jsonl":
        output_path = output_path / "metadata.relayout.jsonl"
    write_jsonl(output_path, output_rows)
    report = {
        "metadata": str(metadata_path),
        "raw_dir": str(raw_root),
        "output": str(output_path),
        "target_area": target_area,
        "rows": len(output_rows),
        "updated": updated,
        "missing_raw": missing,
        "errors": errors,
    }
    atomic_write_json(output_path.parent / REPORT_NAME, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moss-lora-relayout",
        description="Re-derive bucket layouts from raw normalized annotations without a VLM.")
    parser.add_argument("--raw", required=True, help="raw_annotations directory")
    parser.add_argument("--metadata", required=True, help="Final metadata JSONL")
    parser.add_argument("--images", default=None, help="Image root (default: metadata parent)")
    parser.add_argument("--target-area", type=int, default=1024 * 1024)
    parser.add_argument("--output", required=True, help="Output directory or metadata.relayout.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    report = relayout_metadata(args.metadata, args.raw, args.output,
                               images_root=args.images, target_area=args.target_area,
                               limit=args.limit)
    print(json.dumps({key: report[key] for key in ("rows", "updated", "missing_raw")}),
          file=sys.stderr)


if __name__ == "__main__":
    main()
