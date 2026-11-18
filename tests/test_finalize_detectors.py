"""Finalize integration: detector items are authoritative; VLM fills only uncovered regions."""

from __future__ import annotations

import json

from moss_lora.utils import sha256_file
from scripts.finalize_dataset import finalize_dataset
from tests.helpers import gradient_image


def _bucket_layout(items):
    return {
        "summary": {"meta": ["anime"], "canvas": "single", "caption": "OCR summary"},
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def _case(tmp_path, ocr_items=None, detector_items=None, vlm_payload=None):
    images = tmp_path / "base"
    images.mkdir()
    image = images / "a.png"
    gradient_image(256, 256, 1).save(image)
    sha = sha256_file(image)
    row = {
        "file_name": "a.png", "width": 256, "height": 256,
        "artist": "alice", "artists": ["alice", "bob"],
        "character": ["hero"], "tags": ["1girl", "solo"],
        "layout_ocr": _bucket_layout(ocr_items or [
            {"label": "text", "caption": "visible text", "text": "OCR",
             "bbox": [-100, -100, -50, 0]},
        ]),
    }
    metadata = images / "metadata_staging.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")

    vlm_dir = None
    if vlm_payload is not None:
        vlm_dir = tmp_path / "vlm"
        vlm_dir.mkdir()
        (vlm_dir / "annotations_vlm.jsonl").write_text(json.dumps({
            "image_sha256": sha, "status": "ok", "model": "gemma-test",
            "prompt_version": "moss-caption-v2", "attempts": 1,
            "warnings": [], "annotation": vlm_payload,
            "source_width": 256, "source_height": 256,
        }) + "\n", encoding="utf-8")

    det_path = None
    if detector_items is not None:
        det_path = tmp_path / "detectors.jsonl"
        det_path.write_text(json.dumps({
            "sha256": sha, "items": detector_items,
        }) + "\n", encoding="utf-8")
    return metadata, vlm_dir, det_path, sha


def _vlm_payload():
    return {
        "caption": {"long": "A detailed scene.", "medium": "A scene.", "short": "A scene."},
        "watermark": [], "borders": ["none"],
        "layout": {
            "summary": {"meta": ["anime"], "canvas": "single", "caption": "scene"},
            "items": [
                {"label": "object", "caption": "chest", "bbox": [100, 100, 300, 300]},
                {"label": "character", "caption": "the character", "bbox": [50, 50, 900, 900]},
                {"label": "object", "caption": "a chair", "bbox": [600, 600, 800, 800]},
                {"label": "text", "caption": "VLM text", "text": "VLM WINS?",
                 "bbox": [500, 500, 600, 650]},
            ],
            "bbox_coord_space": "normalized_1000", "bbox_format": "yxyx",
        },
    }


def test_detector_items_are_authoritative_and_vlm_is_filtered(tmp_path):
    detector_items = [
        {"label": "object", "caption": "nipples", "tag": "nipples",
         "tag_code": "C1", "bbox": [100, 100, 300, 300], "confidence": 0.9},
        {"label": "character", "caption": "person", "tag": "person",
         "tag_code": "P1", "bbox": [40, 40, 950, 950], "confidence": 0.9},
    ]
    metadata, vlm_dir, det_path, _ = _case(tmp_path, detector_items=detector_items,
                                           vlm_payload=_vlm_payload())
    output = tmp_path / "final"
    report = finalize_dataset(metadata, vlm_dir, output, target_area=256 * 256,
                              detectors=det_path, include_vlm_objects=True)
    row = json.loads((output / "metadata.jsonl").read_text().splitlines()[0])

    items = row["layout"]["items"]
    texts = [item["text"] for item in items if item["label"] == "text"]
    captions = [item["caption"] for item in items if item["label"] != "text"]
    assert texts == ["OCR"]                       # VLM text dropped
    assert "nipples" in captions                    # detector object kept
    assert "person" in captions                    # detector person kept
    assert "a chair" in captions                   # uncovered VLM object kept
    assert "chest" not in captions                  # covered VLM object dropped
    assert "the character" not in captions          # covered VLM character dropped
    assert row["artists"] == ["alice", "bob"]
    assert row["character"] == ["hero"]
    assert row["annotation"]["detector_items"] == 2
    assert row["annotation"]["vlm_objects_included"] == 1
    assert row["annotation"]["vlm_objects_covered_dropped"] == 2
    assert report["detector_items"] == 2
    assert report["vlm_objects_covered_dropped"] == 2


def test_finalize_caps_text_and_detector_items(tmp_path):
    ocr_items = [
        {"label": "text", "caption": "visible text", "text": f"T{i}",
         "bbox": [-100 + i * 8, -100, -95 + i * 8, -60]}
        for i in range(9)
    ]
    detector_items = [
        {"label": "object", "caption": "nipples", "tag": "nipples",
         "tag_code": "C1", "bbox": [0, 0, 40, 40], "confidence": 0.9},
        {"label": "object", "caption": "penis", "tag": "penis",
         "tag_code": "C2", "bbox": [100, 100, 140, 140], "confidence": 0.9},
        {"label": "object", "caption": "pussy", "tag": "pussy",
         "tag_code": "C3", "bbox": [200, 200, 240, 240], "confidence": 0.9},
        {"label": "character", "caption": "person", "tag": "person",
         "tag_code": "P1", "bbox": [300, 300, 600, 600], "confidence": 0.9},
        {"label": "object", "caption": "face", "tag": "face",
         "tag_code": "F1", "bbox": [400, 400, 450, 450], "confidence": 0.9},
        {"label": "object", "caption": "head", "tag": "head",
         "tag_code": "H1", "bbox": [500, 500, 550, 550], "confidence": 0.9},
    ]
    metadata, _, det_path, _ = _case(tmp_path, ocr_items=ocr_items,
                                     detector_items=detector_items)
    output = tmp_path / "final"
    report = finalize_dataset(metadata, None, output, target_area=256 * 256,
                              detectors=det_path)
    row = json.loads((output / "metadata.jsonl").read_text().splitlines()[0])
    items = row["layout"]["items"]
    assert len(items) == 12
    assert sum(1 for item in items if item["label"] == "text") == 8
    assert report["detector_items"] == 6
    assert report["detector_items_dropped_cap"] == 2
    assert report["text_items_dropped_cap"] == 1
