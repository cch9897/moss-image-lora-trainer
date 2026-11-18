"""Finalize/relayout: OCR-authoritative text, VLM prose, separate layout variant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moss_lora.utils import sha256_file
from scripts.finalize_dataset import FinalizeError, finalize_dataset
from scripts.relayout import relayout_metadata
from tests.helpers import gradient_image


def bucket_layout(text: str = "EXIT", bbox=(-100, -100, -50, 0)) -> dict:
    return {
        "summary": {"meta": ["anime"], "canvas": "single", "caption": "OCR summary"},
        "items": [{"label": "text", "caption": "visible text", "text": text,
                   "bbox": list(bbox)}],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def vlm_payload(*, object_bbox=(100, 100, 800, 800), watermark=None) -> dict:
    return {
        "caption": {
            "long": "A detailed painting of a girl in a city.",
            "medium": "A girl in a city.",
            "short": "A girl outdoors.",
        },
        "watermark": watermark or [],
        "borders": ["none"],
        "layout": {
            "summary": {"meta": ["anime"], "canvas": "single", "caption": "city scene"},
            "items": [
                {"label": "character", "caption": "the character", "bbox": list(object_bbox)},
                {"label": "text", "caption": "VLM text", "text": "VLM WINS?",
                 "bbox": [500, 500, 600, 650]},
            ],
            "bbox_coord_space": "normalized_1000",
            "bbox_format": "yxyx",
        },
    }


def make_case(tmp_path: Path) -> tuple[Path, Path, str]:
    images = tmp_path / "base"
    images.mkdir()
    image = images / "a.png"
    gradient_image(256, 256, 1).save(image)
    sha = sha256_file(image)
    row = {
        "file_name": "a.png",
        "width": 256,
        "height": 256,
        "artist": "alice",
        "artists": ["alice", "bob"],
        "character": ["hero"],
        "tags": ["1girl", "solo", "city"],
        "layout_ocr": bucket_layout(),
    }
    metadata = images / "metadata_staging.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    vlm_dir = tmp_path / "vlm"
    vlm_dir.mkdir()
    (vlm_dir / "annotations_vlm.jsonl").write_text(json.dumps({
        "image_sha256": sha, "status": "ok", "model": "gemma-test",
        "prompt_version": "moss-caption-v2", "attempts": 1,
        "warnings": [], "annotation": vlm_payload(),
        "source_width": 256, "source_height": 256,
    }) + "\n", encoding="utf-8")
    return metadata, vlm_dir, sha


def test_finalize_uses_ocr_text_and_keeps_long_prose(tmp_path):
    metadata, vlm_dir, sha = make_case(tmp_path)
    output = tmp_path / "final"
    report = finalize_dataset(metadata, vlm_dir, output, target_area=256 * 256)
    assert report["rows"] == 1
    row = json.loads((output / "metadata.jsonl").read_text().splitlines()[0])
    assert row["caption_original"].startswith("@CH3OCH3, artist:alice, artist:bob, character:hero")
    assert row["caption"]["long"].startswith("@CH3OCH3, artist:alice, artist:bob, character:hero")
    assert "A detailed painting" in row["caption"]["long"]
    assert not row["caption"]["long"].lstrip().startswith("{")
    assert row["layout"]["bbox_coord_space"] == "abs_centralized"
    texts = [item["text"] for item in row["layout"]["items"] if item["label"] == "text"]
    assert texts == ["EXIT"]  # OCR is authoritative; VLM text dropped
    assert all(item["label"] != "character" for item in row["layout"]["items"])
    assert row["layout_ocr"]["items"][0]["text"] == "EXIT"
    assert row["annotation"]["vlm_text_items_dropped"] == 1
    assert row["annotation"]["vlm_objects_included"] == 0
    assert row["artists"] == ["alice", "bob"]
    assert row["character"] == ["hero"]
    raw = json.loads((output / "raw_annotations" / f"{sha}.json").read_text())
    assert raw["layout_raw"]["bbox_coord_space"] == "normalized_1000"
    assert report["variant_weights"]["original"] == pytest.approx(0.16)
    assert report["variant_weights"]["long"] == pytest.approx(0.213333)


def test_finalize_audit_passed_includes_object_boxes(tmp_path):
    metadata, vlm_dir, _sha = make_case(tmp_path)
    report = finalize_dataset(metadata, vlm_dir, tmp_path / "final", target_area=256 * 256,
                              include_vlm_objects=True)
    row = json.loads((tmp_path / "final" / "metadata.jsonl").read_text().splitlines()[0])
    labels = [item["label"] for item in row["layout"]["items"]]
    assert labels == ["character", "text"]
    assert report["vlm_objects_included"] == 1


def test_finalize_omits_layout_when_watermark_covers_subject(tmp_path):
    metadata, vlm_dir, _sha = make_case(tmp_path)
    vlm_path = vlm_dir / "annotations_vlm.jsonl"
    record = json.loads(vlm_path.read_text().splitlines()[0])
    record["annotation"] = vlm_payload(watermark=[{"box_2d": [100, 100, 800, 800],
                                                    "label": "@artist"}])
    vlm_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    finalize_dataset(metadata, vlm_dir, tmp_path / "final", target_area=256 * 256,
                     include_vlm_objects=True)
    row = json.loads((tmp_path / "final" / "metadata.jsonl").read_text().splitlines()[0])
    assert "layout" not in row
    assert row["layout_ocr"] is not None
    assert row["annotation"]["watermark_action"] == "exclude_layout"


def test_finalize_fallback_without_vlm_and_allow_ocr_layout(tmp_path):
    metadata, _vlm_dir, _sha = make_case(tmp_path)
    output = tmp_path / "final"
    report = finalize_dataset(metadata, None, output, target_area=256 * 256,
                              allow_ocr_layout=True)
    row = json.loads((output / "metadata.jsonl").read_text().splitlines()[0])
    assert row["caption"]["long"] == row["caption_original"]
    assert row["layout"]["items"][0]["text"] == "EXIT"
    assert row["annotation"]["status"] == "fallback"
    assert report["vlm_statuses"]["fallback_missing"] == 1


def test_finalize_rejects_invalid_prefix(monkeypatch, tmp_path):
    metadata, vlm_dir, _sha = make_case(tmp_path)
    import scripts.finalize_dataset as module

    monkeypatch.setattr(module, "_caption_original", lambda *args, **kwargs: "no prefix")
    monkeypatch.setattr(module, "_ensure_prefix", lambda text, *args, **kwargs: text)
    with pytest.raises(FinalizeError):
        finalize_dataset(metadata, vlm_dir, tmp_path / "final", target_area=256 * 256)


def test_relayout_reuses_raw_annotations_without_vlm(tmp_path):
    metadata, vlm_dir, _sha = make_case(tmp_path)
    final = tmp_path / "final"
    finalize_dataset(metadata, vlm_dir, final, target_area=256 * 256)
    result = relayout_metadata(final / "metadata.jsonl", final / "raw_annotations",
                               tmp_path / "relayout", target_area=512 * 512)
    assert result["updated"] == 1
    row = json.loads((tmp_path / "relayout" / "metadata.relayout.jsonl").read_text().splitlines()[0])
    assert row["bucket_w"] == row["bucket_h"] == 512
    assert row["layout"]["bbox_coord_space"] == "abs_centralized"
    for item in row["layout"]["items"]:
        y1, x1, y2, x2 = item["bbox"]
        assert -256 <= y1 < y2 <= 256 and -256 <= x1 < x2 <= 256


def test_finalize_sorts_swapped_vlm_corners(tmp_path):
    metadata, vlm_dir, _sha = make_case(tmp_path)
    vlm_path = vlm_dir / "annotations_vlm.jsonl"
    record = json.loads(vlm_path.read_text().splitlines()[0])
    record["annotation"] = vlm_payload(object_bbox=(800, 800, 100, 100))
    vlm_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    finalize_dataset(metadata, vlm_dir, tmp_path / "final", target_area=256 * 256,
                     include_vlm_objects=True)
    row = json.loads((tmp_path / "final" / "metadata.jsonl").read_text().splitlines()[0])
    character = next(item for item in row["layout"]["items"] if item["label"] == "character")
    y1, x1, y2, x2 = character["bbox"]
    assert y1 < y2 and x1 < x2


def test_finalize_does_not_gate_on_vlm_text_geometry(tmp_path):
    """VLM text boxes are dropped; only PaddleOCR text survives regardless of IoU."""
    metadata, vlm_dir, _sha = make_case(tmp_path)
    finalize_dataset(metadata, vlm_dir, tmp_path / "final", target_area=256 * 256,
                     include_vlm_objects=True)
    row = json.loads((tmp_path / "final" / "metadata.jsonl").read_text().splitlines()[0])
    text_items = [item for item in row["layout"]["items"] if item["label"] == "text"]
    assert [item["text"] for item in text_items] == ["EXIT"]
