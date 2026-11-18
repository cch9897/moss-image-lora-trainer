"""deepghs anime detector integration for MOSS layouts.

The detector bundle is four ONNX YOLO models:
  * censor: three fixed anatomy/censor classes -> object items (C1-C3);
  * person: person -> character item (P1);
  * face/head: subparts, QA-only by default (not emitted as layout items).

Outputs use the same normalized 0-1000 ``yxyx`` contract as the rest of the
layout pipeline. The module is import-safe without onnxruntime; sessions are
only created by :class:`DeepghsDetector`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

COORD_MAX = 1000
FULL_CANVAS_AREA = 0.98
SAME_TAG_NMS_IOU = 0.50
CROSS_TAG_IOU = 0.90
CROSS_TAG_AREA_RATIO = 1.25
DEFAULT_IMAGE_SIZE = 640


@dataclass(frozen=True)
class DetectorClass:
    index: int
    tag: str
    code: str
    label: str
    min_area: float
    max_area: float
    emit: bool = True


@dataclass(frozen=True)
class DetectorModel:
    name: str
    filename: str
    classes: tuple[DetectorClass, ...]


CENSOR = DetectorModel(
    "censor", "censor_detect_v1.0_s/model.onnx",
    (
        DetectorClass(0, "nipples", "C1", "object", 0.00005, 0.15),
        DetectorClass(1, "penis", "C2", "object", 0.0002, 0.25),
        DetectorClass(2, "pussy", "C3", "object", 0.0001, 0.20),
    ),
)
PERSON = DetectorModel(
    "person", "person_detect_v1.2_s/model.onnx",
    (DetectorClass(0, "person", "P1", "character", 0.02, 0.95),),
)
FACE = DetectorModel(
    "face", "face_detect_v1.4_n/model.onnx",
    (DetectorClass(0, "face", "F1", "object", 0.0005, 0.20, emit=False),),
)
HEAD = DetectorModel(
    "head", "head_detect_v2.0_n/model.onnx",
    (DetectorClass(0, "head", "H1", "object", 0.0005, 0.20, emit=False),),
)
MODELS = (CENSOR, PERSON, FACE, HEAD)

# Caption terms used only to decide whether a VLM item is already covered by a
# detector tag. They are not emitted as labels.
CAPTION_TERMS = {
    "nipples": ("nipple", "areola", "breast", "bust", "chest"),
    "penis": ("penis", "phallus", "shaft", "glans", "member", "organ"),
    "pussy": ("pussy", "vagina", "vulva", "labia", "crotch", "groin"),
    "person": ("person", "character", "figure", "girl", "boy", "man", "woman"),
}


def normalize_box(bbox: Sequence[float]) -> tuple[int, int, int, int]:
    y1, x1, y2, x2 = (int(round(float(v))) for v in bbox)
    return (y1, x1, y2, x2)


def box_area(bbox: Sequence[float]) -> float:
    y1, x1, y2, x2 = bbox
    return max(0.0, y2 - y1) * max(0.0, x2 - x1) / (COORD_MAX * COORD_MAX)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ya1, xa1, ya2, xa2 = a
    yb1, xb1, yb2, xb2 = b
    iy = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    ix = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter = iy * ix
    union = (ya2 - ya1) * (xa2 - xa1) + (yb2 - yb1) * (xb2 - xb1) - inter
    return inter / union if union > 0 else 0.0


def letterbox(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE):
    """Return (NCHW float32 tensor, scale, pad_x, pad_y)."""
    im = image.convert("RGB")
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...], scale, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolo(output: np.ndarray, model: DetectorModel, scale: float,
                pad_x: float, pad_y: float, width: int, height: int,
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.50) -> list[dict[str, Any]]:
    """Decode one YOLOv8 output into normalized 0-1000 item dicts."""
    pred = output
    if pred.ndim == 3:
        pred = pred[0]
    ncls = len(model.classes)
    if pred.shape[0] == 4 + ncls and pred.shape[1] != 4 + ncls:
        pred = pred.T
    elif pred.shape[1] != 4 + ncls:
        raise ValueError(f"unexpected output shape {output.shape} for {ncls} classes")
    boxes = pred[:, :4].astype(np.float32)
    scores = pred[:, 4:4 + ncls]
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    keep = conf >= conf_threshold
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    if len(boxes) == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    out: list[dict[str, Any]] = []
    for c, spec in enumerate(model.classes):
        idx = np.where(cls == c)[0]
        if len(idx) == 0:
            continue
        for k in _nms(xyxy[idx], conf[idx], iou_threshold):
            j = idx[k]
            x1, y1, x2, y2 = (float(v) for v in xyxy[j])
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = normalize_box((y1 / height * COORD_MAX, x1 / width * COORD_MAX,
                                  y2 / height * COORD_MAX, x2 / width * COORD_MAX))
            if box_area(bbox) <= 0:
                continue
            out.append({
                "label": spec.label,
                "caption": spec.tag,
                "tag": spec.tag,
                "tag_code": spec.code,
                "class_index": spec.index,
                "confidence": round(float(conf[j]), 4),
                "bbox": list(bbox),
                "source": f"deepghs_{model.name}",
                "_emit": spec.emit,
            })
    return out


def filter_items(items: Iterable[Mapping[str, Any]], *, drop_full: bool = True
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Area/full-canvas/NMS/cross-tag filters. Returns (kept, dropped)."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        bbox = tuple(int(v) for v in item["bbox"])
        area = box_area(bbox)
        tag = str(item.get("tag") or item.get("caption") or "")
        spec = next((c for m in MODELS for c in m.classes if c.tag == tag), None)
        if drop_full and area >= FULL_CANVAS_AREA:
            dropped.append({"tag": tag, "reason": "full_canvas", "area": area})
            continue
        if spec is not None and area < spec.min_area:
            dropped.append({"tag": tag, "reason": "below_min_area", "area": area})
            continue
        if spec is not None and area > spec.max_area:
            dropped.append({"tag": tag, "reason": "above_max_area", "area": area})
            continue
        item["bbox"] = list(bbox)
        kept.append(item)

    by_tag: dict[str, list[dict[str, Any]]] = {}
    for item in kept:
        by_tag.setdefault(item["tag"], []).append(item)
    kept = []
    for group in by_tag.values():
        ordered = sorted(group, key=lambda x: (-x["confidence"], box_area(x["bbox"])))
        for item in ordered:
            if any(box_iou(item["bbox"], other["bbox"]) > SAME_TAG_NMS_IOU for other in kept
                   if other["tag"] == item["tag"]):
                dropped.append({"tag": item["tag"], "reason": "same_tag_nms"})
            else:
                kept.append(item)

    kept.sort(key=lambda x: (-x["confidence"], box_area(x["bbox"])))
    final: list[dict[str, Any]] = []
    for item in kept:
        dup = False
        for other in final:
            if item["tag"] == other["tag"]:
                continue
            iou = box_iou(item["bbox"], other["bbox"])
            a, b = box_area(item["bbox"]), box_area(other["bbox"])
            ratio = max(a, b) / max(1e-12, min(a, b))
            if iou >= CROSS_TAG_IOU and ratio <= CROSS_TAG_AREA_RATIO:
                dropped.append({"tag": item["tag"], "reason": "cross_tag_duplicate",
                                "iou": round(iou, 4)})
                dup = True
                break
        if not dup:
            final.append(item)
    return final, dropped


def detector_tag_matches(caption: str, tag: str) -> bool:
    """True when a VLM caption already names a detector tag's concept."""
    text = (caption or "").lower()
    terms = CAPTION_TERMS.get(tag, ())
    return any(term in text for term in terms)


def item_covered_by_detector(item: Mapping[str, Any],
                             detector_items: Sequence[Mapping[str, Any]],
                             *, iou_threshold: float = 0.50) -> bool:
    """Whether a VLM item is already covered by an authoritative detector item."""
    for det in detector_items:
        if det.get("label") != item.get("label"):
            continue
        if box_iou(det["bbox"], item["bbox"]) >= iou_threshold:
            return True
        if detector_tag_matches(item.get("caption", ""), det.get("tag", "")):
            return True
    return False


class DeepghsDetector:
    """Load and run the four ONNX models. Lazy onnxruntime import."""

    def __init__(self, models_dir: str | Path, *,
                 providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
                 conf_threshold: float = 0.25, iou_threshold: float = 0.50,
                 emit_subparts: bool = False):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("onnxruntime is required for DeepghsDetector") from exc
        self.models_dir = Path(models_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.emit_subparts = emit_subparts
        self.ort = ort
        self.sessions: dict[str, Any] = {}
        self.providers: dict[str, list[str]] = {}
        for model in MODELS:
            path = self.models_dir / model.filename
            if not path.is_file():
                raise FileNotFoundError(f"detector model not found: {path}")
            sess = ort.InferenceSession(str(path), providers=list(providers))
            self.sessions[model.name] = sess
            self.providers[model.name] = list(sess.get_providers())

    def detect(self, image: Image.Image) -> dict[str, Any]:
        tensor, scale, pad_x, pad_y = letterbox(image)
        width, height = image.size
        emitted: list[dict[str, Any]] = []
        subparts: list[dict[str, Any]] = []
        for model in MODELS:
            sess = self.sessions[model.name]
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: tensor})[0]
            for item in decode_yolo(output, model, scale, pad_x, pad_y, width, height,
                                    self.conf_threshold, self.iou_threshold):
                if item.pop("_emit", True) and (model in (CENSOR, PERSON) or self.emit_subparts):
                    emitted.append(item)
                else:
                    subparts.append(item)
        kept, dropped = filter_items(emitted)
        sub_kept, sub_dropped = filter_items(subparts)
        return {
            "items": kept,
            "subparts": sub_kept,
            "dropped": dropped + sub_dropped,
            "providers": self.providers,
        }
