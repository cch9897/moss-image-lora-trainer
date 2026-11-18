"""Optional OCR backends and bucket-space bbox geometry for MOSS layouts.

The geometry here deliberately mirrors :func:`moss_lora.cache.fit_image`:
cover-resize to the bucket's rounded size, center-crop, then express text boxes
in the official ``abs_centralized`` / ``yxyx`` coordinate space.

OCR backends are optional.  ``tesseract`` needs the system binary, while the
Python backends are imported lazily only when selected.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageOps

CJK_RANGES = ((0x3400, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FFFF))
TESSERACT_LANGS = {
    "en": "eng",
    "eng": "eng",
    "ch_sim": "chi_sim",
    "zh": "chi_sim",
    "chi_sim": "chi_sim",
    "ch_tra": "chi_tra",
    "chi_tra": "chi_tra",
    "ja": "jpn",
    "jp": "jpn",
    "jpn": "jpn",
    "japan": "jpn",
}
EASYOCR_LANGS = {"zh": "ch_sim", "chinese": "ch_sim", "ch_sim": "ch_sim",
                 "en": "en", "english": "en", "ja": "ja", "jp": "ja", "jpn": "ja"}
PADDLE_LANGS = {"zh": "ch", "ch_sim": "ch", "chinese": "ch", "en": "en",
                "english": "en", "ja": "japan", "jp": "japan", "jpn": "japan"}

OCR_BACKENDS = ("none", "easyocr", "paddle", "rapidocr", "tesseract")
OCR_DEVICES = ("auto", "cpu", "gpu")
RAPIDOCR_MODULE = "rapidocr_onnxruntime"
# PaddleOCR 3.x engines: "onnxruntime" runs the official PP-OCRv6 medium ONNX
# models, "paddle" (default) needs the PaddlePaddle runtime.
PADDLE_ONNX_ENGINE = "onnxruntime"
PADDLE_ONNX_VERSION = "PP-OCRv6"
# CUDA provider options for the ONNX engine. The stock EXHAUSTIVE cudnn kernel
# search is the expensive one: measured on an 8GB card at --ocr-max-side 2048 it
# held 6.8GB (det + rec sessions) and failed allocations once other processes
# used the GPU. HEURISTIC search drops that to ~5-6GB with 0 failures over 250
# diverse images; forcing arena_extend_strategy=kSameAsRequested instead made the
# arena plateau 1.1GB higher (7.2GB), so only the search mode is overridden.
PADDLE_CUDA_PROVIDER_OPTIONS = ({"cudnn_conv_algo_search": "HEURISTIC"}, {})


def cuda_provider_options(gpu_mem_limit_mib: int = 0) -> list[dict[str, Any]]:
    """CUDA provider options for the ONNX engine, optionally VRAM-capped.

    ``gpu_mem_limit_mib`` bounds ONNX Runtime's arena so a long run cannot grow
    until an arena extension for the next image no longer fits (which surfaces
    as "Failed to allocate memory" on every following image).
    """
    cuda_options = dict(PADDLE_CUDA_PROVIDER_OPTIONS[0])
    if gpu_mem_limit_mib > 0:
        cuda_options["gpu_mem_limit"] = int(gpu_mem_limit_mib) * 1024 * 1024
    return [cuda_options, dict(PADDLE_CUDA_PROVIDER_OPTIONS[1])]


def strict_cuda_provider_options(gpu_mem_limit_mib: int = 0
                                 ) -> tuple[list[str], list[dict[str, Any]]]:
    """Providers/options that leave no CPU fallback path for inference.

    ONNX Runtime always appends ``CPUExecutionProvider`` to a session (host and
    shape ops need it), but requesting CUDA alone means a node the CUDA EP
    cannot run fails the session instead of silently costing CPU speed. Callers
    must still assert ``CUDAExecutionProvider`` is first in ``session.providers``
    because a missing CUDA shared library is only logged as a warning and the
    session quietly lands on CPU.
    """
    return ["CUDAExecutionProvider"], [cuda_provider_options(gpu_mem_limit_mib)[0]]


class OCRError(RuntimeError):
    """OCR backend missing, misconfigured, or failed on an image."""


@dataclass(frozen=True)
class OCRItem:
    """One recognized text region in EXIF-transposed source-image pixels."""

    text: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class OCRResult:
    items: list[OCRItem]
    raw_count: int
    deduped: int
    truncated: int
    device: str
    backend: str
    engine: str = ""
    # Boxes below ``min_confidence``, kept in source pixels for manual review.
    # They never reach the trained labels.
    low_confidence: list[OCRItem] = field(default_factory=list)
    low_confidence_dropped: int = 0


@dataclass(frozen=True)
class BucketGeometry:
    """Exact cover-resize/center-crop geometry for one image and bucket."""

    source_w: int
    source_h: int
    bucket_w: int
    bucket_h: int
    resize_w: int
    resize_h: int
    left: int
    top: int
    scale_x: float
    scale_y: float

    def source_box_to_bucket(self, box: tuple[float, float, float, float]
                             ) -> tuple[float, float, float, float]:
        """Map x1,y1,x2,y2 source pixels to bucket pixel coordinates (pre-center)."""
        x1, y1, x2, y2 = box
        return (
            x1 * self.scale_x - self.left,
            y1 * self.scale_y - self.top,
            x2 * self.scale_x - self.left,
            y2 * self.scale_y - self.top,
        )


def parse_languages(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return ["ch_sim", "en"]
    if isinstance(value, str):
        parts = re.split(r"[+,;]", value)
    else:
        parts = [part for value in value for part in re.split(r"[+,;]", str(value))]
    return [part.strip() for part in parts if part.strip()] or ["ch_sim", "en"]


def backend_available(backend: str) -> bool:
    if backend == "none":
        return True
    if backend == "tesseract":
        return shutil.which("tesseract") is not None
    module = {
        "easyocr": "easyocr",
        "paddle": "paddleocr",
        "rapidocr": RAPIDOCR_MODULE,
    }.get(backend)
    return bool(module and importlib.util.find_spec(module) is not None)


def available_backends() -> dict[str, bool]:
    return {name: backend_available(name) for name in OCR_BACKENDS}


def runtime_providers() -> dict[str, Any]:
    """Describe the installed ONNX Runtime: availability, version, providers."""
    try:
        import onnxruntime
    except Exception:  # pragma: no cover - onnxruntime is optional
        return {"available": False, "version": None, "providers": []}
    try:
        providers = [str(name) for name in onnxruntime.get_available_providers()]
    except Exception:  # pragma: no cover - broken installs only
        providers = []
    return {
        "available": True,
        "version": str(getattr(onnxruntime, "__version__", "")),
        "providers": providers,
    }


def _cuda_provider_available() -> bool:
    return "CUDAExecutionProvider" in runtime_providers()["providers"]


def resolve_ocr_device(backend: str, device: str = "auto") -> str:
    """Resolve the effective device name for one backend and request.

    ``auto`` selects the GPU when the installed runtime exposes CUDA, otherwise
    CPU. onnxruntime backends report ``cuda``; the PaddlePaddle runtime reports
    ``gpu``. An unavailable request falls back to ``cpu`` instead of failing.
    """
    if device not in OCR_DEVICES:
        raise OCRError(f"unknown OCR device {device!r}; expected one of {', '.join(OCR_DEVICES)}")
    if backend in ("none", "tesseract") or device == "cpu":
        return "cpu"
    if backend in ("rapidocr", "paddle"):
        if _cuda_provider_available():
            return "cuda" if backend == "rapidocr" else "gpu"
        if backend == "paddle":
            try:
                import paddle

                return "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
            except Exception:
                return "cpu"
        return "cpu"
    if backend == "easyocr":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    # OCR often splits CJK into separate tokens; remove the resulting spaces.
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    return text.strip()


def _join_words(words: Iterable[str]) -> str:
    return _normalize_text(" ".join(word.strip() for word in words if word.strip()))


def _polygon_box(points: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _clip_box(box: tuple[float, float, float, float], width: int, height: int
              ) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return x1, y1, x2, y2


def box_iou(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def dedupe_ocr_items(items: Sequence[OCRItem], *, iou_threshold: float = 0.5
                     ) -> tuple[list[OCRItem], int]:
    """Drop repeated recognitions of the same text at (nearly) the same box."""
    ordered = sorted(items, key=lambda item: (-item.confidence, item.box[1], item.box[0]))
    kept: list[OCRItem] = []
    removed = 0
    for item in ordered:
        key = _normalize_text(item.text).casefold()
        if not key:
            removed += 1
            continue
        if any(
            _normalize_text(other.text).casefold() == key
            and box_iou(item.box, other.box) >= iou_threshold
            for other in kept
        ):
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def select_ocr_items(items: Sequence[OCRItem], max_items: int | None) -> tuple[list[OCRItem], int]:
    """Keep the most confident ``max_items`` then restore reading order."""
    if max_items is None or len(items) <= max_items:
        selected = list(items)
        truncated = 0
    else:
        selected = sorted(
            items, key=lambda item: (-item.confidence, item.box[1], item.box[0])
        )[:max_items]
        truncated = len(items) - len(selected)
    selected.sort(key=lambda item: (item.box[1], item.box[0], item.box[3], item.box[2]))
    return selected, truncated


def parse_tesseract_tsv(tsv: str) -> list[OCRItem]:
    """Group tesseract word-level TSV rows into text-line OCR items."""
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("level")) != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf", "-1")) / 100.0
            left = float(row.get("left", "0"))
            top = float(row.get("top", "0"))
            width = float(row.get("width", "0"))
            height = float(row.get("height", "0"))
        except ValueError:
            continue
        if confidence < 0:
            continue
        key = (row.get("page_num", ""), row.get("block_num", ""),
               row.get("par_num", ""), row.get("line_num", ""))
        group = groups.setdefault(key, {"words": [], "confs": [], "boxes": []})
        group["words"].append(text)
        group["confs"].append(confidence)
        group["boxes"].append((left, top, left + width, top + height))
    items: list[OCRItem] = []
    for group in groups.values():
        text = _join_words(group["words"])
        if not text:
            continue
        boxes = group["boxes"]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes))
        items.append(OCRItem(text=text, confidence=sum(group["confs"]) / len(group["confs"]), box=box))
    return items


def items_from_easyocr(result: Iterable[Any]) -> list[OCRItem]:
    items: list[OCRItem] = []
    for entry in result or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        box = _polygon_box(entry[0])
        text = _normalize_text(entry[1])
        try:
            confidence = float(entry[2])
        except (TypeError, ValueError):
            continue
        if box and text:
            items.append(OCRItem(text=text, confidence=confidence, box=box))
    return items


def items_from_paddle(result: Any) -> list[OCRItem]:
    items: list[OCRItem] = []
    if not result:
        return items
    # PaddleOCR returns [[ [box, (text, score)], ... ]] on 2.x and a different
    # nested list on some 3.x builds; flatten defensively.
    layers = result if isinstance(result, list) else [result]
    for layer in layers:
        if not isinstance(layer, list):
            continue
        for entry in layer:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            box = _polygon_box(entry[0])
            record = entry[1]
            if isinstance(record, (list, tuple)) and len(record) >= 2:
                text, confidence = record[0], record[1]
            else:
                continue
            text = _normalize_text(text)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if box and text:
                items.append(OCRItem(text=text, confidence=confidence, box=box))
    return items


def items_from_paddle_v3(result: Any) -> list[OCRItem]:
    """Parse PaddleOCR 3.x ``predict`` results (one dict-like object per page).

    Each page exposes ``rec_texts``/``rec_scores`` with either ``rec_polys``
    (quadrilaterals) or ``rec_boxes`` (x1, y1, x2, y2) in input-image pixels.
    """
    items: list[OCRItem] = []
    pages = result if isinstance(result, (list, tuple)) else [result]
    for page in pages:
        if not isinstance(page, dict):
            continue
        texts = list(page.get("rec_texts") or [])
        scores = list(page.get("rec_scores") or [])
        polys = page.get("rec_polys")
        if polys is None:
            polys = page.get("dt_polys")
        boxes = page.get("rec_boxes")
        for index, raw_text in enumerate(texts):
            text = _normalize_text(raw_text)
            if not text:
                continue
            box: tuple[float, float, float, float] | None = None
            if polys is not None and index < len(polys):
                box = _polygon_box(polys[index])
            if box is None and boxes is not None and index < len(boxes):
                values = [float(value) for value in list(boxes[index])[:4]]
                if len(values) == 4:
                    box = (values[0], values[1], values[2], values[3])
            try:
                confidence = float(scores[index])
            except (IndexError, TypeError, ValueError):
                continue
            if box is not None and box[2] > box[0] and box[3] > box[1]:
                items.append(OCRItem(text=text, confidence=confidence, box=box))
    return items


def items_from_rapidocr(result: Iterable[Any]) -> list[OCRItem]:
    items: list[OCRItem] = []
    for entry in result or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        box = _polygon_box(entry[0])
        text = _normalize_text(entry[1])
        try:
            confidence = float(entry[2])
        except (TypeError, ValueError):
            continue
        if box and text:
            items.append(OCRItem(text=text, confidence=confidence, box=box))
    return items


def bucket_geometry(source_size: tuple[int, int], bucket: tuple[int, int]) -> BucketGeometry:
    """Mirror ``moss_lora.cache.fit_image`` cover-resize and center-crop."""
    source_w, source_h = source_size
    bucket_w, bucket_h = bucket
    scale = max(bucket_w / source_w, bucket_h / source_h)
    resize_w = max(bucket_w, round(source_w * scale))
    resize_h = max(bucket_h, round(source_h * scale))
    left = (resize_w - bucket_w) // 2
    top = (resize_h - bucket_h) // 2
    return BucketGeometry(
        source_w=source_w,
        source_h=source_h,
        bucket_w=bucket_w,
        bucket_h=bucket_h,
        resize_w=resize_w,
        resize_h=resize_h,
        left=left,
        top=top,
        scale_x=resize_w / source_w,
        scale_y=resize_h / source_h,
    )


def transform_box(box: tuple[float, float, float, float], geometry: BucketGeometry
                  ) -> tuple[list[int], bool] | None:
    """Source x1,y1,x2,y2 -> official yxyx abs_centralized bucket bbox.

    Returns ``(bbox, clipped)`` or ``None`` when the box lies entirely outside
    the center crop.
    """
    x1, y1, x2, y2 = geometry.source_box_to_bucket(box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    visible_x1, visible_x2 = max(0.0, x1), min(float(geometry.bucket_w), x2)
    visible_y1, visible_y2 = max(0.0, y1), min(float(geometry.bucket_h), y2)
    if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
        return None
    clipped = (visible_x1, visible_y1, visible_x2, visible_y2) != (x1, y1, x2, y2)
    center_x = geometry.bucket_w / 2.0
    center_y = geometry.bucket_h / 2.0
    bbox = [
        int(round(visible_y1 - center_y)),
        int(round(visible_x1 - center_x)),
        int(round(visible_y2 - center_y)),
        int(round(visible_x2 - center_x)),
    ]
    bbox[0] = max(-geometry.bucket_h // 2, min(geometry.bucket_h // 2, bbox[0]))
    bbox[2] = max(-geometry.bucket_h // 2, min(geometry.bucket_h // 2, bbox[2]))
    bbox[1] = max(-geometry.bucket_w // 2, min(geometry.bucket_w // 2, bbox[1]))
    bbox[3] = max(-geometry.bucket_w // 2, min(geometry.bucket_w // 2, bbox[3]))
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox, clipped


def transform_ocr_items(items: Sequence[OCRItem], geometry: BucketGeometry
                        ) -> tuple[list[tuple[list[int], OCRItem]], dict[str, int]]:
    visible: list[tuple[list[int], OCRItem]] = []
    outside = 0
    clipped = 0
    for item in items:
        transformed = transform_box(item.box, geometry)
        if transformed is None:
            outside += 1
            continue
        bbox, was_clipped = transformed
        clipped += int(was_clipped)
        visible.append((bbox, item))
    visible.sort(key=lambda pair: (pair[0][0], pair[0][1], pair[0][2], pair[0][3]))
    return visible, {"outside_crop": outside, "clipped": clipped}


def make_layout(summary_caption: str, items: Sequence[tuple[list[int], OCRItem]],
                *, meta: Sequence[str] = ("anime",), canvas: str = "single") -> dict[str, Any]:
    """Official MOSS layout object; items are already in bucket/central coords."""
    return {
        "summary": {
            "meta": list(meta),
            "canvas": canvas,
            "caption": summary_caption,
        },
        "items": [
            {
                "label": "text",
                "caption": "visible text",
                "text": item.text,
                "bbox": list(bbox),
            }
            for bbox, item in items
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def layout_json(layout: dict[str, Any]) -> str:
    import json

    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- backends


def _load_image(image: Image.Image, max_side: int | None
                ) -> tuple[Image.Image, float, float, tuple[int, int]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    original_w, original_h = original_size
    if max_side and max_side > 0 and max(original_w, original_h) > max_side:
        scale = max_side / max(original_w, original_h)
        new_size = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    return image, original_w / image.width, original_h / image.height, original_size


def _scale_items(items: Sequence[OCRItem], scale_x: float, scale_y: float,
                 width: int, height: int) -> list[OCRItem]:
    scaled: list[OCRItem] = []
    for item in items:
        box = (item.box[0] * scale_x, item.box[1] * scale_y,
               item.box[2] * scale_x, item.box[3] * scale_y)
        clipped = _clip_box(box, width, height)
        if clipped is not None:
            scaled.append(OCRItem(text=item.text, confidence=item.confidence, box=clipped))
    return scaled


def _run_tesseract(image: Image.Image, languages: Sequence[str], timeout: int = 120) -> list[OCRItem]:
    binary = shutil.which("tesseract")
    if binary is None:
        raise OCRError("tesseract is not installed (apt install tesseract-ocr)")
    lang = "+".join(TESSERACT_LANGS.get(item.lower(), item) for item in languages)
    with tempfile.TemporaryDirectory(prefix="moss-ocr-") as temp_dir:
        source = Path(temp_dir) / "image.png"
        image.save(source)
        command = [binary, str(source), "stdout", "-l", lang, "--psm", "11", "tsv"]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise OCRError(f"tesseract timed out after {timeout}s") from error
    if process.returncode != 0:
        message = (process.stderr or process.stdout or "").strip()
        raise OCRError(f"tesseract failed ({process.returncode}): {message[:500]}")
    return parse_tesseract_tsv(process.stdout)


@lru_cache(maxsize=4)
def _easyocr_reader(languages: tuple[str, ...], gpu: bool):
    import easyocr

    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


@lru_cache(maxsize=4)
def _paddle_engine(language: str, device: str, gpu_mem_limit_mib: int = 0,
                   strict_cuda: bool = False) -> tuple[Any, str]:
    """Build one PaddleOCR engine and report which runtime it uses.

    PaddleOCR 3.x is built with ``engine="onnxruntime"`` so the official
    PP-OCRv6 medium ONNX models run through ONNX Runtime (CUDA or CPU).
    PaddleOCR 2.x installs fall back to the PaddlePaddle runtime.
    ``strict_cuda`` requests the CUDA EP without the CPU fallback provider.
    """
    from paddleocr import PaddleOCR

    if importlib.util.find_spec("onnxruntime") is not None:
        engine_config: dict[str, Any] = {"device_type": device}
        if device == "gpu":
            if strict_cuda:
                providers, provider_options = strict_cuda_provider_options(gpu_mem_limit_mib)
                engine_config["providers"] = providers
                engine_config["provider_options"] = provider_options
            else:
                engine_config["provider_options"] = cuda_provider_options(gpu_mem_limit_mib)
        try:
            engine = PaddleOCR(
                lang=language,
                device=device,
                engine=PADDLE_ONNX_ENGINE,
                engine_config=engine_config,
                ocr_version=PADDLE_ONNX_VERSION,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:  # PaddleOCR 2.x has no engine=/device= keyword arguments
            pass
        else:
            return engine, PADDLE_ONNX_ENGINE
    try:
        return PaddleOCR(lang=language, use_gpu=device == "gpu", show_log=False), "paddle"
    except TypeError:
        return PaddleOCR(lang=language), "paddle"


def paddle_ocr_engine(*, device: str = "gpu",
                      languages: str | Iterable[str] | None = None,
                      gpu_mem_limit_mib: int = 0, strict_cuda: bool = False
                      ) -> tuple[Any, str]:
    """Build (and cache) the Paddle OCR engine that ``run_ocr_batch`` uses."""
    language_list = parse_languages(languages)
    language = PADDLE_LANGS.get(language_list[0].lower(), language_list[0])
    resolved = resolve_ocr_device("paddle", device)
    if strict_cuda and resolved != "gpu":
        raise OCRError(
            "strict CUDA requested but the Paddle ONNX engine resolved to "
            f"{resolved!r}: CUDAExecutionProvider is unavailable"
        )
    return _paddle_engine(language, resolved, gpu_mem_limit_mib, strict_cuda)


def onnx_session_providers(engine: Any, *, max_depth: int = 8) -> list[list[str]]:
    """Execution providers of every ONNX Runtime session inside ``engine``.

    A session's first provider runs the graph; ONNX Runtime always appends
    ``CPUExecutionProvider`` for host/shape ops, so a caller that requires GPU
    inference must check the first entry of each list. A CUDA shared-library
    load failure only logs a warning and silently leaves the session on CPU,
    which is why this inspects the built sessions instead of trusting
    ``onnxruntime.get_available_providers()``.
    """
    import onnxruntime

    seen: set[int] = set()
    sessions: list[Any] = []

    def walk(value: Any, depth: int) -> None:
        if depth > max_depth or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, onnxruntime.InferenceSession):
            sessions.append(value)
            return
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for child in attributes.values():
                walk(child, depth + 1)

    walk(engine, 0)
    return [[str(name) for name in session.get_providers()] for session in sessions]


def _run_paddle_engine(engine: Any, array: Any, det_side: int | None = None) -> list[OCRItem]:
    """Run one PaddleOCR engine through its 3.x or 2.x entry point.

    ``det_side`` caps the text-detection input side; recognition still crops
    from the full-resolution array, so a cap buys VRAM headroom without
    blurring the text that gets read.
    """
    predict = getattr(engine, "predict", None)
    if callable(predict):
        if det_side:
            return items_from_paddle_v3(predict(
                array, text_det_limit_side_len=int(det_side), text_det_limit_type="max",
            ))
        return items_from_paddle_v3(predict(array))
    return items_from_paddle(engine.ocr(array, cls=True))


@lru_cache(maxsize=2)
def _rapidocr_engine(use_cuda: bool):
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(use_cuda=use_cuda)


def _run_python_backend(image: Image.Image, backend: str, languages: Sequence[str],
                        device: str, det_side: int | None = None,
                        gpu_mem_limit_mib: int = 0) -> tuple[list[OCRItem], str]:
    import numpy as np

    array = np.array(image)
    if backend == "easyocr":
        mapped = [EASYOCR_LANGS.get(item.lower(), item) for item in languages]
        reader = _easyocr_reader(tuple(mapped), device == "cuda")
        return items_from_easyocr(reader.readtext(array, detail=1, paragraph=False)), "torch"
    if backend == "paddle":
        language = PADDLE_LANGS.get(languages[0].lower(), languages[0])
        engine, engine_kind = _paddle_engine(language, device, gpu_mem_limit_mib)
        # PaddleX reads ndarray input as BGR (ReadImage(format="BGR") does not
        # convert arrays), so hand it BGR instead of the RGB PIL data.
        bgr = np.ascontiguousarray(array[:, :, ::-1])
        return _run_paddle_engine(engine, bgr, det_side), engine_kind
    if backend == "rapidocr":
        engine = _rapidocr_engine(device == "cuda")
        result, _ = engine(array)
        return items_from_rapidocr(result or []), "onnxruntime"
    raise OCRError(f"unsupported OCR backend: {backend}")


def _finalize_ocr(raw: Sequence[OCRItem], *, scale_x: float, scale_y: float,
                  source_size: tuple[int, int], min_confidence: float,
                  max_items: int | None, device: str, backend: str,
                  engine_kind: str) -> OCRResult:
    """Confidence filter, scale back to source pixels, dedupe, cap (shared by
    single-image and batched runs)."""
    raw = [item for item in raw if item.text.strip()]
    kept_raw = [item for item in raw if item.confidence >= min_confidence]
    below_raw = [item for item in raw if item.confidence < min_confidence]
    raw_count = len(kept_raw)
    scaled = _scale_items(kept_raw, scale_x, scale_y, *source_size)
    deduped_items, deduped = dedupe_ocr_items(scaled)
    selected, truncated = select_ocr_items(deduped_items, max_items)
    below_source = _scale_items(below_raw, scale_x, scale_y, *source_size)
    review_items, _ = select_ocr_items(below_source, max_items)
    return OCRResult(
        items=selected,
        raw_count=raw_count,
        deduped=deduped,
        truncated=truncated,
        device=device,
        backend=backend,
        engine=engine_kind,
        low_confidence=review_items,
        low_confidence_dropped=len(below_source),
    )


def run_ocr(image: Image.Image, *, backend: str = "rapidocr",
            languages: str | Iterable[str] | None = None, device: str = "auto",
            min_confidence: float = 0.5, max_items: int | None = 20,
            max_side: int | None = 1600, det_side: int | None = None,
            gpu_mem_limit_mib: int = 0) -> OCRResult:
    """Run one backend on an EXIF-transposed RGB image.

    Boxes are returned in original (EXIF-transposed) source-image pixels even
    when OCR itself ran on a downscaled copy. ``device`` is ``auto`` (GPU when
    available), ``cpu`` or ``gpu``.
    """
    if backend == "none":
        return OCRResult([], 0, 0, 0, "cpu", "none")
    if not backend_available(backend):
        raise OCRError(
            f"OCR backend {backend!r} is not available; install it or use "
            f"'--ocr none' / another backend (available: "
            f"{', '.join(name for name, ok in available_backends().items() if ok)})"
        )
    language_list = parse_languages(languages)
    resolved_device = resolve_ocr_device(backend, device)
    working, scale_x, scale_y, source_size = _load_image(image, max_side)
    engine_kind = "tesseract" if backend == "tesseract" else ""
    if backend == "tesseract":
        raw = _run_tesseract(working, language_list)
    else:
        raw, engine_kind = _run_python_backend(working, backend, language_list, resolved_device,
                                              det_side, gpu_mem_limit_mib)
    return _finalize_ocr(
        raw, scale_x=scale_x, scale_y=scale_y, source_size=source_size,
        min_confidence=min_confidence, max_items=max_items,
        device=resolved_device, backend=backend, engine_kind=engine_kind,
    )


def run_ocr_batch(images: Sequence[Image.Image], *, backend: str = "paddle",
                  languages: str | Iterable[str] | None = None, device: str = "gpu",
                  min_confidence: float = 0.5, max_items: int | None = None,
                  max_side: int | None = 2048, det_side: int | None = 1024,
                  gpu_mem_limit_mib: int = 0, strict_cuda: bool = False
                  ) -> list[OCRResult]:
    """Run OCR over a batch of images with one engine, returning one result each.

    Only the Paddle backend batches (PaddleOCR 3.x accepts a list of arrays and
    recognizes the text-line crops of the whole batch together); other backends
    fall back to one ``run_ocr`` call per image. Boxes follow the same contract
    as ``run_ocr``. ``strict_cuda`` builds the engine on the CUDA EP alone and
    raises when that is not what actually runs.
    """
    images = list(images)
    if not images:
        return []
    if backend != "paddle":
        return [run_ocr(image, backend=backend, languages=languages, device=device,
                        min_confidence=min_confidence, max_items=max_items,
                        max_side=max_side, det_side=det_side,
                        gpu_mem_limit_mib=gpu_mem_limit_mib)
                for image in images]
    if not backend_available(backend):
        raise OCRError(f"OCR backend {backend!r} is not available")
    language_list = parse_languages(languages)
    resolved_device = resolve_ocr_device(backend, device)
    if strict_cuda and resolved_device != "gpu":
        raise OCRError(
            "strict CUDA requested but the Paddle ONNX engine resolved to "
            f"{resolved_device!r}: CUDAExecutionProvider is unavailable"
        )
    engine, engine_kind = paddle_ocr_engine(
        device=device, languages=language_list, gpu_mem_limit_mib=gpu_mem_limit_mib,
        strict_cuda=strict_cuda,
    )
    if strict_cuda and engine_kind != PADDLE_ONNX_ENGINE:
        raise OCRError(
            f"strict CUDA requested but the engine is {engine_kind!r}, not "
            f"{PADDLE_ONNX_ENGINE!r}"
        )
    loaded = [_load_image(image, max_side) for image in images]
    predict = getattr(engine, "predict", None)
    if not callable(predict):  # PaddleOCR 2.x: no batched predict, no strict CUDA
        return [run_ocr(image, backend=backend, languages=languages, device=device,
                        min_confidence=min_confidence, max_items=max_items,
                        max_side=max_side, det_side=det_side,
                        gpu_mem_limit_mib=gpu_mem_limit_mib)
                for image in images]
    import numpy as np

    # PaddleX reads ndarray input as BGR, so hand it BGR like the per-image path.
    arrays = [np.ascontiguousarray(np.array(working)[:, :, ::-1])
              for working, _, _, _ in loaded]
    kwargs: dict[str, Any] = {}
    if det_side:
        kwargs = {"text_det_limit_side_len": int(det_side), "text_det_limit_type": "max"}
    pages = list(predict(arrays, **kwargs))
    if len(pages) != len(arrays):
        raise OCRError(
            f"PaddleOCR returned {len(pages)} results for {len(arrays)} images"
        )
    return [
        _finalize_ocr(
            items_from_paddle_v3(page),
            scale_x=scale_x, scale_y=scale_y, source_size=source_size,
            min_confidence=min_confidence, max_items=max_items,
            device=resolved_device, backend=backend, engine_kind=engine_kind,
        )
        for page, (_, scale_x, scale_y, source_size) in zip(pages, loaded)
    ]


def confidence_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return round(ordered[index], 4)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }
