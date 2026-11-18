#!/usr/bin/env python3
"""Run the deepghs detector bundle over a metadata JSONL, keyed by image SHA-256.

No cache/training side effects. Output is one JSONL row per image:
  {sha256, file_name, source_path, width, height, items, subparts, dropped, providers}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_lora.detectors import DeepghsDetector  # noqa: E402

DEFAULT_MODELS = os.environ.get("DEEPCHS_MODELS_DIR", "/mnt/data/models/grounding/deepghs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(row: dict, root: Path) -> Path:
    for key in ("source_path", "file_name"):
        value = row.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else root / path
    raise ValueError("row has neither source_path nor file_name")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run deepghs ONNX detectors over metadata")
    ap.add_argument("--base", required=True, help="metadata JSONL with file_name/source_path")
    ap.add_argument("--images", default=None, help="image root (default: base parent)")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--providers", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--emit-subparts", action="store_true",
                    help="Also emit face/head as layout items (default: QA-only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    base = Path(args.base)
    root = Path(args.images) if args.images else base.parent
    rows = [json.loads(line) for line in open(base, encoding="utf-8")]
    if args.limit is not None:
        rows = rows[:args.limit]

    providers = {
        "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "cpu": ("CPUExecutionProvider",),
        "auto": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    }[args.providers]
    detector = DeepghsDetector(args.models_dir, providers=providers,
                               conf_threshold=args.conf, iou_threshold=args.iou,
                               emit_subparts=args.emit_subparts)
    print(f"providers: {detector.providers}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["sha256"])
            except Exception:
                continue
        print(f"resume: {len(done)} done", file=sys.stderr)

    from PIL import Image
    written = 0
    for index, row in enumerate(rows, 1):
        path = image_path(row, root)
        if not path.is_file():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        sha = row.get("image_sha256") or row.get("sha256") or sha256_file(path)
        if sha in done:
            continue
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            result = detector.detect(image)
        payload = {
            "sha256": sha,
            "file_name": row.get("file_name"),
            "source_path": str(path),
            "width": width,
            "height": height,
            "model": "deepghs_bundle_v1",
            "providers": result["providers"],
            "items": result["items"],
            "subparts": result["subparts"],
            "dropped": result["dropped"],
        }
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        done.add(sha)
        written += 1
        print(f"[{index}/{len(rows)}] {path.name} items={len(result['items'])} "
              f"subparts={len(result['subparts'])}", flush=True)
    print(f"wrote {written} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
