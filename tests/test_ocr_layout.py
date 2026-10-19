"""OCR/bbox/layout tests: geometry, backend parsing, field/variant modes."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from moss_lora.cache import caption_variants, collect_items, fit_image
from moss_lora.ocr import (
    OCRError,
    OCRItem,
    OCRResult,
    _load_image,
    _scale_items,
    backend_available,
    bucket_geometry,
    confidence_distribution,
    dedupe_ocr_items,
    items_from_rapidocr,
    layout_json,
    make_layout,
    parse_tesseract_tsv,
    run_ocr,
    select_ocr_items,
    transform_box,
    transform_ocr_items,
)
from scripts import prepare_captions
from scripts.prepare_captions import PrepareConfig, prepare_dataset

TEXT = "A quiet street at dusk with a clear sign and a raised hand."


def write_case(root: Path, name: str = "a", *, width: int = 256, height: int = 256) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = root / f"{name}.png"
    Image.new("RGB", (width, height), (120, 130, 140)).save(image)
    image.with_suffix(".txt").write_text(
        f"@h, artist:alice, character:hero, 1girl, solo, sign\n\n{TEXT}\n",
        encoding="utf-8",
    )
    return image


def fake_ocr(items: list[OCRItem] | None = None):
    items = items if items is not None else [OCRItem("EXIT", 0.9, (10, 10, 100, 50))]

    def runner(image, **kwargs):  # noqa: ARG001
        return OCRResult(items=list(items), raw_count=len(items), deduped=0,
                         truncated=0, device="cpu", backend="rapidocr")

    return runner


def config(**overrides) -> PrepareConfig:
    values = dict(tokenizer="none", target_area=256 * 256, ocr="rapidocr",
                  layout_mode="both", layout_max_tokens=1024)
    values.update(overrides)
    return PrepareConfig(**values)


def load_row(output: Path) -> dict:
    return json.loads((output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])


# --------------------------------------------------------------------------- geometry


def test_bucket_geometry_matches_fit_image_pixels():
    source = Image.new("RGB", (1000, 2000), (128, 128, 128))
    ImageDraw.Draw(source).rectangle((100, 700, 299, 899), fill=(255, 0, 0))
    bucket = (1024, 1024)
    fitted = np.asarray(fit_image(source, bucket))
    mask = (fitted[:, :, 0] > 200) & (fitted[:, :, 1] < 80) & (fitted[:, :, 2] < 80)
    ys, xs = np.where(mask)
    assert len(xs) > 0
    actual = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    geometry = bucket_geometry(source.size, bucket)
    assert (geometry.resize_w, geometry.resize_h, geometry.left, geometry.top) == (1024, 2048, 0, 512)
    transformed = transform_box((100, 700, 300, 900), geometry)
    assert transformed is not None
    bbox, clipped = transformed
    assert clipped is False
    # bbox is yxyx in canvas-center coordinates; convert back for comparison.
    expected = (
        bbox[1] + bucket[0] // 2,
        bbox[0] + bucket[1] // 2,
        bbox[3] + bucket[0] // 2,
        bbox[2] + bucket[1] // 2,
    )
    assert actual == pytest.approx(expected, abs=2)

    # Entirely above the center crop -> dropped.
    assert transform_box((100, 100, 300, 300), geometry) is None
    # Partially outside -> clipped to the top edge, still visible.
    partial = transform_box((100, 100, 300, 700), geometry)
    assert partial is not None
    partial_bbox, partial_clipped = partial
    assert partial_clipped is True
    assert partial_bbox[0] == -bucket[1] // 2


def test_exif_orientation_source_size_and_box_clipping():
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise -> transposed is 20x40
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    with Image.open(buffer) as loaded:
        working, scale_x, scale_y, source_size = _load_image(loaded, max_side=0)
    assert working.size == (20, 40)
    assert source_size == (20, 40)
    assert (scale_x, scale_y) == (1.0, 1.0)
    # The old bug clipped with the untransposed 40x20 size; a full-height box
    # must survive the transposed 20x40 source.
    clipped = _scale_items([OCRItem("text", 0.9, (0, 0, 19, 39))], 1.0, 1.0, *source_size)
    assert clipped[0].box == (0.0, 0.0, 19.0, 39.0)


def test_transform_items_sorts_and_reports_drops():
    geometry = bucket_geometry((1000, 2000), (1024, 1024))
    items = [
        OCRItem("BOTTOM", 0.9, (100, 1200, 300, 1300)),
        OCRItem("TOP", 0.8, (100, 650, 300, 750)),
        OCRItem("OUTSIDE", 0.9, (100, 10, 300, 100)),
    ]
    visible, stats = transform_ocr_items(items, geometry)
    assert [item.text for _, item in visible] == ["TOP", "BOTTOM"]
    assert stats == {"outside_crop": 1, "clipped": 0}
    assert visible[0][0][0] < visible[1][0][0]


# --------------------------------------------------------------------------- parsing/backends


def test_tesseract_tsv_line_grouping():
    tsv = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t90.0\tHELLO",
        "5\t1\t1\t1\t1\t2\t45\t20\t40\t10\t80.0\tWORLD",
        "5\t1\t1\t1\t2\t1\t10\t50\t20\t10\t70.0\t你好",
        "4\t1\t1\t1\t1\t0\t10\t20\t75\t10\t-1\t",
    ])
    items = parse_tesseract_tsv(tsv)
    assert [item.text for item in items] == ["HELLO WORLD", "你好"]
    assert items[0].confidence == pytest.approx(0.85)
    assert items[0].box == (10, 20, 85, 30)


def test_rapidocr_result_parser_and_dedupe():
    raw = [
        [[[10, 20], [50, 20], [50, 40], [10, 40]], "HELLO", 0.9],
        [[[12, 21], [49, 21], [49, 39], [12, 39]], "hello", 0.8],
        [[[100, 100], [150, 100], [150, 130], [100, 130]], "HELLO", 0.7],
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "  ", 0.9],
    ]
    items = items_from_rapidocr(raw)
    assert len(items) == 3
    assert items[0].box == (10.0, 20.0, 50.0, 40.0)
    deduped, removed = dedupe_ocr_items(items)
    assert removed == 1
    assert len(deduped) == 2
    assert {item.text for item in deduped} == {"HELLO"}  # case-insensitive at same location
    assert len(deduped) == 2  # same text far away is kept


def test_select_items_and_confidence_distribution():
    items = [
        OCRItem("low", 0.4, (0, 0, 10, 10)),
        OCRItem("high", 0.9, (0, 20, 10, 30)),
        OCRItem("mid", 0.7, (0, 10, 10, 20)),
    ]
    selected, truncated = select_ocr_items(items, 2)
    assert truncated == 1
    assert [item.text for item in selected] == ["mid", "high"]  # reading order after confidence selection
    stats = confidence_distribution([0.4, 0.9, 0.7])
    assert stats["p50"] == 0.7
    assert stats["p95"] == 0.9


def test_none_backend_and_unknown_backend():
    result = run_ocr(Image.new("RGB", (32, 32)), backend="none")
    assert result.items == [] and result.backend == "none"
    assert backend_available("none") is True
    with pytest.raises(OCRError):
        run_ocr(Image.new("RGB", (32, 32)), backend="does-not-exist")


def test_layout_object_schema_and_bbox():
    geometry = bucket_geometry((256, 256), (256, 256))
    visible, _ = transform_ocr_items([OCRItem("EXIT", 0.9, (10, 10, 100, 50))], geometry)
    layout = make_layout("trigger, artist:alice, character:hero, a sign", visible)
    assert layout["summary"]["canvas"] == "single"
    assert layout["summary"]["meta"] == ["anime"]
    assert "artist:alice" in layout["summary"]["caption"]
    assert layout["bbox_format"] == "yxyx"
    assert layout["bbox_coord_space"] == "abs_centralized"
    assert layout["items"] == [{
        "label": "text",
        "caption": "visible text",
        "text": "EXIT",
        "bbox": [-118, -118, -78, -28],
    }]
    assert json.loads(layout_json(layout)) == layout


# --------------------------------------------------------------------------- layout modes


@pytest.mark.parametrize("mode", ["field", "variant", "both"])
def test_layout_modes_metadata_and_caption(tmp_path, monkeypatch, mode):
    root = tmp_path / "data"
    write_case(root)
    monkeypatch.setattr(prepare_captions, "backend_available", lambda backend: True)
    monkeypatch.setattr(prepare_captions, "run_ocr", fake_ocr())
    output = tmp_path / mode
    report = prepare_dataset(root, output, config(layout_mode=mode))

    row = load_row(output)
    layout_present = "layout" in row
    long_is_layout = row["caption"]["long"].startswith("{")
    assert layout_present is (mode in ("field", "both"))
    assert long_is_layout is (mode in ("variant", "both"))
    assert "artist:alice" not in row["caption_no_artist"]
    assert report["ocr"]["images_with_text"] == 1
    assert report["ocr"]["kept_boxes"] == 1
    assert report["layout"]["images"] == 1
    if layout_present:
        assert row["layout"]["items"][0]["text"] == "EXIT"
        assert row["layout"]["items"][0]["bbox"] == [-118, -118, -78, -28]
    if long_is_layout:
        parsed = json.loads(row["caption"]["long"])
        assert "artist:alice" in parsed["summary"]["caption"]
        assert TEXT in parsed["summary"]["caption"]
        assert parsed["items"][0]["text"] == "EXIT"
    else:
        assert row["caption"]["long"].startswith("mystyle, artist:alice, character:hero")


def test_ocr_output_is_consumable_by_cache_and_variants(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_case(root)
    monkeypatch.setattr(prepare_captions, "backend_available", lambda backend: True)
    monkeypatch.setattr(prepare_captions, "run_ocr", fake_ocr())
    output = tmp_path / "out"
    prepare_dataset(root, output, config(layout_mode="both"))

    items = collect_items(output)
    assert len(items) == 1
    variants = caption_variants(items[0][1])
    assert len(variants) == 3
    layout = json.loads(variants[0])
    assert layout["bbox_format"] == "yxyx"
    assert "artist:alice" in layout["summary"]["caption"]
    assert "artist:alice" in variants[1]
    assert "artist:alice" in variants[2]


def test_layout_token_budget_shortens_and_reports(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_case(root)
    monkeypatch.setattr(prepare_captions, "backend_available", lambda backend: True)
    monkeypatch.setattr(prepare_captions, "run_ocr", fake_ocr())
    output = tmp_path / "out"
    report = prepare_dataset(
        root, output, config(layout_mode="both", layout_max_tokens=64)
    )
    row = load_row(output)
    layout = json.loads(row["caption"]["long"])
    assert "artist:alice" in layout["summary"]["caption"]
    assert report["layout"]["truncated"] >= 1
    assert report["layout"]["tokens"]["max"] <= 64
    assert report["layout"]["truncation_reasons"]


def test_ocr_errors_do_not_abort_captions(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_case(root)

    def failing(image, **kwargs):  # noqa: ARG001
        raise OCRError("synthetic failure")

    monkeypatch.setattr(prepare_captions, "backend_available", lambda backend: True)
    monkeypatch.setattr(prepare_captions, "run_ocr", failing)
    output = tmp_path / "out"
    report = prepare_dataset(root, output, config(layout_mode="both"))
    assert report["images"]["written"] == 1
    assert report["ocr"]["images_errors"] == 1
    assert report["layout"]["images"] == 0
    row = load_row(output)
    assert not row["caption"]["long"].startswith("{")
    assert "layout" not in row


def test_no_ocr_keeps_existing_behavior(tmp_path):
    root = tmp_path / "data"
    write_case(root)
    output = tmp_path / "out"
    report = prepare_dataset(root, output, PrepareConfig(tokenizer="none", target_area=256 * 256))
    row = load_row(output)
    assert "layout" not in row
    assert not row["caption"]["long"].startswith("{")
    assert report["ocr"]["backend"] == "none"
    assert report["layout"]["images"] == 0
