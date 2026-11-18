"""Annotation contract tests: bbox transforms, schema, prefixes, leaks, budgets."""

from __future__ import annotations

import json
import math

import pytest

from moss_lora.annotations import (
    BUCKET_SPACE,
    NORMALIZED_SPACE,
    WATERMARK_IOU_THRESHOLD,
    AnnotationError,
    Borders,
    CaptionSet,
    Layout,
    LayoutItem,
    OCRTextItem,
    TokenBudgetError,
    VLMAnnotation,
    Watermark,
    box_iou,
    bucket_bbox_to_normalized,
    build_caption_variants,
    build_prefix,
    compose_caption,
    enforce_variant_token_budgets,
    estimate_tokens,
    filter_ocr_watermarks,
    find_watermark_text,
    layout_json,
    layout_to_bucket,
    missing_prefix_labels,
    normalized_box_to_bucket_bbox,
    normalized_to_source_box,
    parse_layout,
    parse_normalized_box,
    parse_vlm_annotation,
    source_box_to_bucket_bbox,
    source_box_to_normalized,
    strip_metadata_labels,
    strip_watermark_text,
    validate_caption_prefix,
    watermark_area_ratio,
)
from moss_lora.ocr import bucket_geometry

# --------------------------------------------------------------------------- transforms


def test_source_box_to_bucket_known_1024_example():
    geometry = bucket_geometry((1024, 1024), (1024, 1024))
    result = source_box_to_bucket_bbox((128, 96, 896, 224), geometry)
    assert result.bbox == (-416, -384, -288, 384)  # docs/DATASET.md 3.2 example
    assert not result.clipped and not result.outside


def test_normalized_to_source_and_bucket_exact_1024():
    box = (93.75, 125.0, 218.75, 875.0)  # y1, x1, y2, x2
    assert normalized_to_source_box(box, 1024, 1024) == (128.0, 96.0, 896.0, 224.0)
    result = normalized_box_to_bucket_bbox(box, (1024, 1024), (1024, 1024))
    assert result.bbox == (-416, -384, -288, 384)
    assert bucket_bbox_to_normalized(result.bbox, bucket_geometry((1024, 1024), (1024, 1024))) == (
        94, 125, 219, 875)


def test_yxyx_ordering_and_floor_ceil():
    geometry = bucket_geometry((1000, 1000), (1000, 1000))
    result = source_box_to_bucket_bbox((200, 100, 400, 300), geometry)
    # y1=100-500, x1=200-500, y2=300-500, x2=400-500
    assert result.bbox == (-400, -300, -200, -100)

    geometry = bucket_geometry((1000, 1000), (1000, 1000))
    result = source_box_to_bucket_bbox((1.2, 2.2, 3.8, 4.8), geometry)
    assert result.bbox == (-498, -499, -495, -496)


def test_cover_crop_drop_clip_and_roundtrip():
    geometry = bucket_geometry((2000, 1000), (1024, 1024))
    assert geometry.left == 512 and geometry.top == 0 and geometry.resize_w == 2048
    outside = source_box_to_bucket_bbox((0, 100, 100, 300), geometry)
    assert outside.bbox is None and outside.outside
    clipped = source_box_to_bucket_bbox((400, 100, 600, 300), geometry)
    assert clipped.bbox is not None and clipped.clipped
    assert clipped.bbox[1] == -512 and clipped.bbox[3] == -409

    right = source_box_to_bucket_bbox((1400, 100, 1600, 300), geometry)
    assert right.bbox is not None and right.clipped and right.bbox[3] == 512
    with pytest.raises(AnnotationError):
        source_box_to_bucket_bbox((0, 0), geometry)


def test_normalized_box_validation():
    assert parse_normalized_box([100, 200, 300, 400]) == (100, 200, 300, 400)
    assert parse_normalized_box([400, 200, 100, 300]) == (100, 200, 400, 300)
    with pytest.raises(AnnotationError):
        parse_normalized_box([0, 0, 1000, 2000])  # out of range
    with pytest.raises(AnnotationError):
        parse_normalized_box([0, 0, 10, 10])  # area < 0.0005
    with pytest.raises(AnnotationError):
        parse_normalized_box([0, 0, 1000, 1000])  # area > 0.95
    with pytest.raises(AnnotationError):
        parse_normalized_box([0, 0, 10.5, 10.5])  # non-integer
    with pytest.raises(AnnotationError):
        parse_normalized_box([0, 0, 10])


def test_box_iou_and_source_normalized_roundtrip():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert source_box_to_normalized((128, 96, 896, 224), 1024, 1024) == (94, 125, 219, 875)


# --------------------------------------------------------------------------- schema


def layout_payload(**overrides):
    payload = {
        "summary": {"meta": ["anime"], "canvas": "single", "caption": "A poster."},
        "items": [
            {"label": "text", "caption": "visible text", "text": "EXIT",
             "bbox": [100, 100, 200, 300]},
            {"label": "character", "caption": "the character in the center",
             "bbox": [300, 200, 800, 700]},
        ],
        "bbox_coord_space": "normalized_1000",
        "bbox_format": "yxyx",
    }
    payload.update(overrides)
    return payload


def test_parse_layout_sorts_and_serializes_compact():
    layout = parse_layout(layout_payload())
    assert layout.bbox_coord_space == NORMALIZED_SPACE
    assert [item.label for item in layout.items] == ["text", "character"]  # top-to-bottom
    text = layout_json(layout)
    assert text == json.dumps(layout.to_dict(), ensure_ascii=False, separators=(",", ":"))
    assert '": ' not in text
    assert json.loads(text)["items"][0]["text"] == "EXIT"


def test_parse_layout_hard_rejections():
    cases = [
        {**layout_payload(), "bbox_format": "xyxy"},
        {**layout_payload(), "bbox_coord_space": "source_px"},
        {**layout_payload(), "items": [{"label": "text", "caption": "x", "bbox": [0, 0, 10, 10]}]},
        {**layout_payload(), "items": [{"label": "bogus", "caption": "x",
                                        "bbox": [100, 100, 200, 200]}]},
        {**layout_payload(), "items": [
            {"label": "object", "caption": "a", "bbox": [100, 100, 300, 300]},
            {"label": "object", "caption": "b", "bbox": [101, 101, 300, 300]},
        ]},
        {**layout_payload(), "items": [
            {"label": "text", "caption": "x", "text": "A", "bbox": [0, 0, 100, 100]}
        ] * 13},
    ]
    for payload in cases:
        with pytest.raises(AnnotationError):
            parse_layout(payload)
    warnings: list[str] = []
    layout = parse_layout({**layout_payload(), "summary": {
        "meta": ["a", "b", "c", "d"], "canvas": "single", "caption": "x"}}, warnings=warnings)
    assert layout.summary_meta == ("a", "b", "c")
    assert warnings


def test_borders_and_watermarks_validation():
    assert Borders().to_list() == ["none"]
    assert Borders(("left", "up")).to_list() == ["left", "up"]
    with pytest.raises(AnnotationError):
        Borders(())
    with pytest.raises(AnnotationError):
        Borders(("none", "left"))
    with pytest.raises(AnnotationError):
        Borders(("top",))


def test_parse_watermark_area_and_ocr_filter():
    mark = Watermark(bbox=(0, 0, 200, 1000), label="@artist")
    assert watermark_area_ratio([mark]) == pytest.approx(0.2)
    kept, dropped = filter_ocr_watermarks(
        [OCRTextItem("EXIT", 0.9, (10, 10, 900, 190)),
         OCRTextItem("SAFE", 0.9, (10, 500, 100, 600))],
        [mark], width=1000, height=1000)
    assert [item.text for item in dropped] == ["EXIT"]
    assert [item.text for item in kept] == ["SAFE"]


def test_parse_vlm_annotation_strips_leaks_and_watermarks():
    raw = {
        "caption": {
            "long": "A girl stands. artist:alice, character:hero, detailed city.",
            "medium": "The girl stands in a city.",
            "short": "A girl in a city.",
        },
        "watermark": [{"box_2d": [0, 0, 200, 1000], "label": "@artist"}],
        "borders": ["none"],
        "layout": layout_payload(items=[
            {"label": "text", "caption": "white sign lettering", "text": "EXIT",
             "bbox": [300, 300, 400, 450]},
            {"label": "text", "caption": "watermark text", "text": "@artist",
             "bbox": [50, 100, 150, 900]},
            {"label": "text", "caption": "watermark label without overlap", "text": "@artist",
             "bbox": [400, 600, 450, 700]},
        ]),
    }
    annotation = parse_vlm_annotation(raw)
    assert isinstance(annotation, VLMAnnotation)
    assert "artist:" not in annotation.caption.long
    assert "character:" not in annotation.caption.long
    assert "@artist" not in annotation.caption.long
    assert len(annotation.layout.items) == 1  # watermark-overlapping text dropped
    assert any("vlm_label_leak:long" in warning for warning in annotation.warnings)
    assert any("watermark_overlap_dropped_items=2" in warning for warning in annotation.warnings)
    assert annotation.borders.is_none


def test_strip_metadata_labels_and_watermark_text():
    scan = strip_metadata_labels("artist:alice, a girl, character:hero, smiling")
    assert scan.leaked
    assert "artist:" not in scan.text and "character:" not in scan.text
    assert "a girl" in scan.text and "smiling" in scan.text
    assert find_watermark_text("A sign by @artist Studio", ["@artist"]) == ["@artist"]
    assert strip_watermark_text("A sign by @artist Studio", ["@artist"]).strip() == "A sign by Studio"


# --------------------------------------------------------------------------- prefixes / captions


def test_prefix_builder_deterministic():
    assert build_prefix() == ["@CH3OCH3", "artist:unknown"]
    assert build_prefix("@CH3OCH3", ["alice", "bob"], ["hero", "mage"]) == [
        "@CH3OCH3", "artist:alice", "artist:bob", "character:hero", "character:mage"]
    assert build_prefix("CH3OCH3", ["artist: alice"], []) == ["@CH3OCH3", "artist:alice"]
    assert build_prefix("@CH3OCH3", ["alice"], [], include_artist=False) == ["@CH3OCH3"]


def test_caption_variants_prefix_tags_and_caps():
    caption = CaptionSet(long="Long prose.", medium="Medium prose.", short="Short prose.")
    tags = [f"tag{index}" for index in range(45)]
    variants = build_caption_variants(
        caption, original_prose="source prose", artists=["alice"], characters=["hero"],
        tags=tags)
    assert variants.original.startswith("@CH3OCH3, artist:alice, character:hero, tag0")
    assert "source prose" in variants.original
    assert "Long prose." in variants.long and "Medium prose." in variants.medium
    assert "Short prose." in variants.short
    for text in variants.as_dict().values():
        assert missing_prefix_labels(text, artists=["alice"], characters=["hero"]) == []
        validate_caption_prefix(text, artists=["alice"], characters=["hero"])
    # tag caps: 40 / 40 / 25 / 12
    assert variants.long.count("tag") - variants.original.count("tag") == 0
    assert sum(1 for index in range(45) if f"tag{index}," in variants.medium) == 25
    assert sum(1 for index in range(45) if f"tag{index}," in variants.short) == 12
    with pytest.raises(AnnotationError):
        validate_caption_prefix("no prefix here", artists=["alice"])


def test_caption_set_requires_all_three_prose_variants():
    with pytest.raises(AnnotationError):
        CaptionSet(long="x", medium="", short="z")
    assert compose_caption(["@CH3OCH3"], ["a", "a", "watermark", "b"], "prose", max_tags=2) == (
        "@CH3OCH3, a, b, prose")


# --------------------------------------------------------------------------- token budgets


def test_token_budget_helpers():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3
    lengths = enforce_variant_token_budgets({"short": "a short sentence", "long": "word " * 10})
    assert set(lengths) == {"short", "long"}
    with pytest.raises(TokenBudgetError) as info:
        enforce_variant_token_budgets({"short": "word " * 200})
    assert info.value.over["short"][1] == 128


def test_layout_to_bucket_drops_and_clips():
    layout = Layout(
        summary_caption="Poster.",
        items=(
            LayoutItem("text", "visible", bbox=(100, 300, 200, 450), text="EXIT"),
            LayoutItem("object", "left edge", bbox=(100, 100, 300, 450)),
            LayoutItem("object", "outside", bbox=(400, 100, 500, 200)),
        ),
    )
    transformed, stats = layout_to_bucket(layout, (4000, 1000), (1024, 1024))
    assert transformed.bbox_coord_space == BUCKET_SPACE
    assert stats.total == 3
    assert stats.outside == 1
    assert stats.clipped == 2
    assert len(transformed.items) == 2
    assert all(item.bbox[1] >= -512 and item.bbox[3] <= 512 for item in transformed.items)
    with pytest.raises(AnnotationError):
        layout_to_bucket(transformed, (4000, 1000), (1024, 1024))


def test_watermark_overlap_threshold_constant():
    assert WATERMARK_IOU_THRESHOLD == 0.3
    assert math.isclose(box_iou((0, 0, 10, 10), (0, 0, 10, 5)), 0.5)


def test_truncate_to_token_budget_respects_measured_estimate():
    from moss_lora.annotations import truncate_to_token_budget

    text = "word " * 200
    truncated, length = truncate_to_token_budget(text, 20)
    assert length <= 20
    assert estimate_tokens(truncated) <= 20
