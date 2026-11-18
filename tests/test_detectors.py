"""Synthetic tests for the deepghs detector adapter (no ONNX runtime/model)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from moss_lora import detectors as d


def test_letterbox_shape():
    tensor, scale, pad_x, pad_y = d.letterbox(Image.new("RGB", (100, 50)), size=640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 6.4
    assert pad_x == 0
    assert pad_y > 0


def test_decode_censor_class_mapping_and_nms():
    # Two overlapping C1 boxes; the lower-confidence one is NMS-suppressed.
    pred = np.zeros((1, 7, 2), dtype=np.float32)
    pred[0, :4, 0] = [320, 320, 32, 32]
    pred[0, 4, 0] = 0.90
    pred[0, :4, 1] = [325, 325, 32, 32]
    pred[0, 4, 1] = 0.60
    out = d.decode_yolo(pred, d.CENSOR, scale=1.0, pad_x=0, pad_y=0,
                        width=640, height=640)
    assert len(out) == 1
    item = out[0]
    assert item["tag"] == "nipples"
    assert item["tag_code"] == "C1"
    assert item["label"] == "object"
    assert item["bbox"] == [475, 475, 525, 525]
    assert item["confidence"] == 0.9


def test_decode_person_is_character():
    pred = np.zeros((1, 5, 1), dtype=np.float32)
    pred[0, :4, 0] = [320, 320, 100, 100]
    pred[0, 4, 0] = 0.8
    out = d.decode_yolo(pred, d.PERSON, scale=1.0, pad_x=0, pad_y=0,
                        width=640, height=640)
    assert len(out) == 1
    assert out[0]["label"] == "character"
    assert out[0]["tag"] == "person"
    assert out[0]["tag_code"] == "P1"


def test_face_head_are_subparts_by_default():
    pred = np.zeros((1, 5, 1), dtype=np.float32)
    pred[0, :4, 0] = [100, 100, 20, 20]
    pred[0, 4, 0] = 0.8
    out = d.decode_yolo(pred, d.FACE, scale=1.0, pad_x=0, pad_y=0,
                        width=640, height=640)
    assert out and out[0]["_emit"] is False


def test_filter_items_full_canvas_and_area_bounds():
    items = [
        {"tag": "nipples", "label": "object", "bbox": [0, 0, 1000, 1000],
         "confidence": 0.9},
        {"tag": "nipples", "label": "object", "bbox": [0, 0, 1, 1],
         "confidence": 0.9},
        {"tag": "nipples", "label": "object", "bbox": [0, 0, 600, 600],
         "confidence": 0.9},
    ]
    kept, dropped = d.filter_items(items)
    reasons = [x["reason"] for x in dropped]
    assert kept == []
    assert "full_canvas" in reasons
    assert "below_min_area" in reasons
    assert "above_max_area" in reasons


def test_filter_items_same_tag_nms_and_cross_tag_duplicate():
    same = [
        {"tag": "nipples", "label": "object", "bbox": [100, 100, 300, 300],
         "confidence": 0.9},
        {"tag": "nipples", "label": "object", "bbox": [105, 105, 305, 305],
         "confidence": 0.5},
    ]
    kept, dropped = d.filter_items(same)
    assert len(kept) == 1 and kept[0]["confidence"] == 0.9
    assert any(x["reason"] == "same_tag_nms" for x in dropped)

    cross = [
        {"tag": "nipples", "label": "object", "bbox": [100, 100, 300, 300],
         "confidence": 0.9},
        {"tag": "person", "label": "character", "bbox": [100, 100, 300, 300],
         "confidence": 0.5},
    ]
    kept, dropped = d.filter_items(cross)
    assert len(kept) == 1
    assert any(x["reason"] == "cross_tag_duplicate" for x in dropped)


def test_caption_coverage_helper():
    assert d.detector_tag_matches("a woman's chest", "nipples")
    assert not d.detector_tag_matches("a chair", "nipples")
    item = {"label": "object", "caption": "chest", "bbox": [100, 100, 300, 300]}
    det = {"label": "object", "tag": "nipples", "bbox": [100, 100, 300, 300]}
    assert d.item_covered_by_detector(item, [det])
    far = {"label": "object", "tag": "nipples", "bbox": [700, 700, 900, 900]}
    assert not d.item_covered_by_detector({"label": "object", "caption": "chair",
                                           "bbox": [100, 100, 300, 300]}, [far])
